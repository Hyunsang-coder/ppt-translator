"""LangChain translation pipeline and progress-aware helpers."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.chains.llm_factory import Provider, create_llm

LOGGER = logging.getLogger(__name__)


class TranslationCancelled(Exception):
    """Raised when a cooperative cancel flag is set mid-translation (C-1)."""


def _is_cancelled(cancel_event) -> bool:
    """True when a cancellation flag is present and set."""
    return cancel_event is not None and cancel_event.is_set()


class IndexedTranslation(BaseModel):
    """One translated text tagged with its 1-based input index."""

    index: int = Field(description="The 1-based number of the source text this translates")
    text: str = Field(description="The translated text for that numbered source")


class TranslationOutput(BaseModel):
    """Structured output model for translation results.

    ``items`` carries each translation tagged with the 1-based index of the
    source line it corresponds to (L-1). Reconstructing by index means a
    dropped middle item leaves a gap in the right slot instead of shifting
    every later translation onto the wrong paragraph. ``translations`` is kept
    as a compatibility accessor for callers/tests that build outputs directly.
    """

    items: List[IndexedTranslation] = Field(
        default_factory=list,
        description="One entry per source text, each tagging its 1-based index.",
    )

    def __init__(self, **data: Any) -> None:
        # Back-compat: allow TranslationOutput(translations=[...]) construction.
        # Positional-index items are synthesised from the plain list so existing
        # tests and any direct callers keep working.
        if "items" not in data and "translations" in data:
            plain = data.pop("translations")
            data["items"] = [
                {"index": i, "text": t} for i, t in enumerate(plain, start=1)
            ]
        super().__init__(**data)

    @property
    def translations(self) -> List[str]:
        """Return translated texts ordered by their reported index.

        Gaps (missing indices) collapse — callers compare the resulting length
        against the expected count and fall back to safe padding when they
        differ, so a dropped item never silently shifts the alignment.
        """
        return [it.text for it in sorted(self.items, key=lambda it: it.index)]

    def aligned(self, expected_count: int) -> List[str] | None:
        """Return a list of length ``expected_count`` aligned by 1-based index.

        Each item is placed at its reported slot; missing slots are ``None``.
        Returns ``None`` when any index is out of range or duplicated, signalling
        the caller to treat the batch as a mismatch and retry.
        """
        slots: List[str | None] = [None] * expected_count
        for it in self.items:
            pos = it.index - 1
            if pos < 0 or pos >= expected_count or slots[pos] is not None:
                return None
            slots[pos] = it.text
        return slots

PROMPT_TEMPLATE = """
You are a professional translator specializing in PowerPoint presentations.
Slide body text favors noun phrases and concise phrasing over full sentences. Avoid target-side expansion (especially when translating Korean to English) and prefer wording that reads clearly within a limited text box. This brevity guidance applies to slide bodies, not to speaker notes.

**Team Translation Rules (hard constraints — these override general fluency):**
{team_rules}

**Context (Full Presentation):**
{ppt_context}

**Background Information:**
{context}

**Glossary:**
{glossary_terms}

**Translation Style/Tone Guidelines:**
{instructions}
{length_constraint}
**Task:**
Translate the following texts from {source_lang} to {target_lang}.
Maintain consistency with the context, background information, and glossary.
Follow the translation style/tone guidelines if provided.
If a sentence or phrase appears more than once in the source, translate it identically every time unless the glossary overrides it.
Return exactly {expected_count} items in the items array — one per numbered source
text. For each item, set "index" to the source text's number (the digit before the
"." on its line) and "text" to its translation. Do not skip, merge, or renumber
items; every source number from 1 to {expected_count} must appear exactly once.

**Texts to translate:**
{texts}
"""


def _build_length_constraint(length_limit: int | None) -> str:
    """Build the length constraint instruction for the prompt.

    Args:
        length_limit: Maximum translation length as percentage of original
            (e.g., 110, 130, 150). None means no constraint.

    Returns:
        Length constraint instruction string, or empty string if no limit.
    """
    if length_limit is None:
        return ""
    return (
        f"\n**Length Constraint:**\n"
        f"IMPORTANT: Keep each translation concise. "
        f"The translated text for each item MUST NOT exceed {length_limit}% "
        f"of the original text length (character count). "
        f"If the direct translation is too long, rephrase it more concisely "
        f"while preserving the core meaning.\n"
    )


def create_translation_chain(
    model_name: str,
    source_lang: str,
    target_lang: str,
    context: str | None = None,
    instructions: str | None = None,
    provider: Provider = "anthropic",
    *,
    length_limit: int | None = None,
    team_rules: str | None = None,
    max_tokens: int | None = None,
    user_prompt: str | None = None,  # Deprecated: for backward compatibility
):
    """Create a LangChain sequence for translation.

    Args:
        model_name: Model identifier (e.g., gpt-5.5-2026-04-23, claude-sonnet-5).
        source_lang: Display name of the source language.
        target_lang: Display name of the target language.
        context: Optional background information about the presentation.
        instructions: Optional translation style/tone guidelines.
        provider: LLM provider ("openai" or "anthropic").
        length_limit: Optional max translation length as percentage of original.
        team_rules: Optional pre-formatted team translation-rules block. Injected
            job-wide into the prompt's stable prefix. None -> "None" (no rules).
        max_tokens: Optional response token ceiling. None -> provider default.
            Sized to the batch's output budget (P-3) to avoid truncating large
            batches.
        user_prompt: Deprecated. Use 'instructions' instead. If provided and
            'instructions' is not set, this value will be used as instructions.

    Returns:
        Configured LangChain runnable sequence with structured output.
    """

    llm_kwargs: dict = {"provider": provider, "model_name": model_name}
    if max_tokens is not None:
        llm_kwargs["max_tokens"] = max_tokens
    llm = create_llm(**llm_kwargs)
    # Wrap LLM with structured output for type-safe parsing
    structured_llm = llm.with_structured_output(TranslationOutput)

    # Backward compatibility: use user_prompt as instructions if instructions not provided
    if instructions is None and user_prompt is not None:
        instructions = user_prompt

    # Use provided values or sensible defaults
    context_text = context or "No additional background information provided."
    instructions_text = instructions or "Translate naturally and professionally."
    length_constraint_text = _build_length_constraint(length_limit)
    team_rules_text = team_rules or "None"

    chain = (
        RunnablePassthrough.assign(
            ppt_context=lambda x: x.get("ppt_context", ""),
            glossary_terms=lambda x: x.get("glossary_terms", "None"),
            team_rules=lambda _: team_rules_text,
            source_lang=lambda _: source_lang,
            target_lang=lambda _: target_lang,
            context=lambda _: context_text,
            instructions=lambda _: instructions_text,
            length_constraint=lambda _: length_constraint_text,
            expected_count=lambda x: int(x.get("expected_count", 0)),
        )
        | PromptTemplate.from_template(PROMPT_TEMPLATE)
        | structured_llm
    )

    return chain


def translate_with_progress(
    chain,
    batches: List[Dict[str, object]],
    progress_tracker: Any = None,
    max_concurrency: int = 1,
    cancel_event: Any = None,
) -> List[str]:
    """Translate batches using LangChain batch API with progress updates.

    Args:
        chain: Configured LangChain runnable sequence with structured output.
        batches: Prepared batch payloads from :func:`chunk_paragraphs`.
        progress_tracker: Optional tracker used to update the UI.
        max_concurrency: Number of batches allowed to run in parallel (>=1).
        cancel_event: Optional ``threading.Event`` checked between batch waves;
            when set, translation stops and raises :class:`TranslationCancelled`
            (C-1) so a cancelled job stops calling the LLM.

    Returns:
        Flattened list of translated texts.

    Raises:
        TranslationCancelled: If ``cancel_event`` is set before/during the run.
    """

    total_batches = len(batches)
    total_sentences = sum(len(batch.get("paragraphs", [])) for batch in batches)

    if _is_cancelled(cancel_event):
        raise TranslationCancelled("Translation cancelled before start")

    if progress_tracker is not None:
        progress_tracker.reset(
            total_batches=total_batches,
            total_sentences=total_sentences,
        )

    LOGGER.info(
        "Beginning translation of %d batches (total %d sentences) with max_concurrency=%d.",
        total_batches,
        total_sentences,
        max_concurrency,
    )

    # Use LangChain batch_as_completed for real-time progress updates.
    # The accumulator is shared across tenacity retries so successful
    # batches are never re-translated.
    config = RunnableConfig(max_concurrency=max(1, int(max_concurrency)))
    accumulator: List[TranslationOutput | None] = [None] * len(batches)
    ordered_results: List[TranslationOutput | None] = _batch_translate_with_retry(
        chain, batches, config, progress_tracker, total_batches, accumulator,
        cancel_event,
    )

    # Assemble translations in input order
    translations: List[str] = []
    for index, (batch, result) in enumerate(zip(batches, ordered_results), start=1):
        if result is None:
            # Should not happen after _batch_translate_with_retry validates,
            # but guard against it to avoid silent data loss.
            raise RuntimeError(
                f"Batch {index} of {len(batches)} returned no result "
                "after all retry attempts"
            )

        translations.extend(
            _resolve_batch_parts(chain, batch, result, config, index)
        )

    return translations


def _resolve_batch_parts(
    chain,
    batch: Dict[str, object],
    result: "TranslationOutput",
    config: RunnableConfig,
    batch_index: int,
) -> List[str]:
    """Return one batch's translations aligned to its paragraphs (length-exact).

    Alignment is by the model's reported 1-based index (L-1): a dropped middle
    item leaves a gap in the correct slot instead of shifting every later
    translation onto the wrong paragraph. Clean results pass through; otherwise
    the batch is retried once, then any remaining gaps are padded with the
    source text so a paragraph is never erased or misaligned.
    """
    expected_count = len(batch.get("paragraphs", []))
    slots = result.aligned(expected_count)
    if slots is not None and all(s is not None for s in slots):
        return [s for s in slots if s is not None]

    if slots is not None:
        LOGGER.warning(
            "Batch %d: %d of %d indexed translations missing; retrying.",
            batch_index,
            sum(1 for s in slots if s is None),
            expected_count,
        )
    else:
        LOGGER.warning(
            "Batch %d: translation indices out of range/duplicated "
            "(expected %d); retrying.",
            batch_index,
            expected_count,
        )

    retry_result = _retry_count_mismatch_batch(
        chain=chain,
        batch=batch,
        config=config,
        batch_index=batch_index,
        expected_count=expected_count,
    )
    retry_slots = (
        retry_result.aligned(expected_count) if retry_result is not None else None
    )
    if retry_slots is not None and all(s is not None for s in retry_slots):
        return [s for s in retry_slots if s is not None]

    # Still incomplete after the retry. Keep the first attempt's indexed slots
    # (its translations are as trustworthy as the retry's partial set) and pad
    # the remaining gaps with the source text — never empty, never shifted.
    originals = [p.original_text for p in batch.get("paragraphs", [])]
    if slots is not None:
        LOGGER.error(
            "Batch %d: still incomplete after retry; padding %d gap(s) with original text.",
            batch_index,
            sum(1 for s in slots if s is None),
        )
        return [
            s if s is not None else (originals[i] if i < len(originals) else "")
            for i, s in enumerate(slots)
        ]

    # No usable indexed slots from either attempt (all indices malformed).
    # Fall back to positional padding/trimming against the plain list.
    LOGGER.error(
        "Batch %d: no usable indexed result; falling back to positional padding.",
        batch_index,
    )
    return _force_match_expected(result.translations, expected_count, originals)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    # C-1: a cancellation must abort immediately, not be retried 3x by tenacity.
    retry=retry_if_not_exception_type(TranslationCancelled),
)
def _batch_translate_with_retry(
    chain,
    batches: List[Dict[str, object]],
    config: RunnableConfig,
    progress_tracker: Any = None,
    total_batches: int = 0,
    ordered_results: List[TranslationOutput | None] | None = None,
    cancel_event: Any = None,
) -> List[TranslationOutput | None]:
    """Execute batch translation with retry logic and real-time progress.

    Uses ``batch_as_completed`` so that each batch reports progress as soon
    as it finishes, instead of waiting for all batches to complete.

    On a tenacity retry, only batches that still have no result are
    re-submitted — successful batches from a prior attempt are preserved in
    ``ordered_results`` so we don't pay to re-translate them.

    Args:
        chain: Configured LangChain runnable sequence with structured output.
        batches: Batch payloads to translate.
        config: RunnableConfig with max_concurrency setting.
        progress_tracker: Optional tracker for real-time progress updates.
        total_batches: Total number of batches (for logging).
        ordered_results: Shared accumulator preserved across retries. Caller
            must pass the same list on every attempt.

    Returns:
        List of TranslationOutput results in input order.
    """
    if ordered_results is None:
        ordered_results = [None] * len(batches)

    # Only retranslate batches that have not yet succeeded.
    pending = [i for i, r in enumerate(ordered_results) if r is None]
    pending_batches = [batches[i] for i in pending]

    # Reset progress for this attempt, then re-credit already-completed
    # batches so the percentage reflects real total progress (not just this
    # attempt's pending subset).
    if progress_tracker is not None:
        total_sentences = sum(len(b.get("paragraphs", [])) for b in batches)
        progress_tracker.reset(
            total_batches=total_batches,
            total_sentences=total_sentences,
        )
        for i, result in enumerate(ordered_results):
            if result is not None:
                batch = batches[i]
                progress_tracker.batch_completed(
                    int(batch.get("start_idx", i + 1)),
                    int(batch.get("end_idx", i + 1)),
                )

    LOGGER.debug(
        "Invoking batch translation for %d batch(es) (%d already done).",
        len(pending_batches),
        len(batches) - len(pending_batches),
    )

    # C-1: bail before submitting a fresh wave when cancellation was requested.
    if _is_cancelled(cancel_event):
        raise TranslationCancelled("Translation cancelled before batch submission")

    for local_idx, result in chain.batch_as_completed(pending_batches, config=config):
        completed_idx = pending[local_idx]
        ordered_results[completed_idx] = result

        # C-1: stop consuming further completed batches once cancelled. In-flight
        # requests already submitted to the pool cannot be recalled, but we stop
        # processing/paying for the rest and abort the run.
        if _is_cancelled(cancel_event):
            raise TranslationCancelled("Translation cancelled mid-batch")

        if progress_tracker is not None:
            batch = batches[completed_idx]
            start_idx = int(batch.get("start_idx", completed_idx + 1))
            end_idx = int(batch.get("end_idx", completed_idx + 1))
            progress_tracker.batch_completed(start_idx, end_idx)

        LOGGER.info(
            "Completed batch %d/%d.",
            completed_idx + 1,
            total_batches,
        )

    # Fail-fast: if any batch yielded no result, raise so tenacity can retry
    missing = [i for i, r in enumerate(ordered_results) if r is None]
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(batches)} batch(es) returned no result "
            f"(batch indices: {missing})"
        )

    return ordered_results


def _retry_count_mismatch_batch(
    chain,
    batch: Dict[str, object],
    config: RunnableConfig,
    batch_index: int,
    expected_count: int,
) -> TranslationOutput | None:
    """Retry a single batch when structured output has the wrong item count."""

    try:
        result: TranslationOutput = chain.invoke(batch, config=config)
    except Exception:
        LOGGER.exception("Batch %d: single-batch retry failed.", batch_index)
        return None

    actual_count = len(result.translations)
    if actual_count != expected_count:
        LOGGER.warning(
            "Batch %d: single-batch retry count %d still differs from expected %d.",
            batch_index,
            actual_count,
            expected_count,
        )

    return result


def _force_match_expected(
    parts: List[str], expected_count: int, originals: List[str] | None = None
) -> List[str]:
    """Pad or trim decoded translations to the expected count.

    Missing translations are padded with the corresponding original text (not
    empty strings) so a count mismatch never erases the source paragraph.
    """

    if len(parts) < expected_count:
        missing = expected_count - len(parts)
        LOGGER.warning(
            "Padding %d missing translation(s) with original text due to format mismatch.",
            missing,
        )
        for idx in range(len(parts), expected_count):
            fallback = originals[idx] if originals and idx < len(originals) else ""
            parts = parts + [fallback]
    elif len(parts) > expected_count:
        LOGGER.debug(
            "Received %d translations but expected %d; trimming extras after successful parse.",
            len(parts),
            expected_count,
        )
        parts = parts[:expected_count]

    return parts
