"""LangChain translation pipeline and progress-aware helpers."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from src.chains.llm_factory import Provider, create_llm

LOGGER = logging.getLogger(__name__)


class TranslationOutput(BaseModel):
    """Structured output model for translation results."""

    translations: List[str] = Field(description="List of translated texts")

PROMPT_TEMPLATE = """
You are a professional translator specializing in PowerPoint presentations.

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
Return exactly {expected_count} translated texts in the translations array.

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
    user_prompt: str | None = None,  # Deprecated: for backward compatibility
):
    """Create a LangChain sequence for translation.

    Args:
        model_name: Model identifier (e.g., gpt-5.5-2026-04-23, claude-sonnet-4-6).
        source_lang: Display name of the source language.
        target_lang: Display name of the target language.
        context: Optional background information about the presentation.
        instructions: Optional translation style/tone guidelines.
        provider: LLM provider ("openai" or "anthropic").
        length_limit: Optional max translation length as percentage of original.
        user_prompt: Deprecated. Use 'instructions' instead. If provided and
            'instructions' is not set, this value will be used as instructions.

    Returns:
        Configured LangChain runnable sequence with structured output.
    """

    llm = create_llm(provider=provider, model_name=model_name)
    # Wrap LLM with structured output for type-safe parsing
    structured_llm = llm.with_structured_output(TranslationOutput)

    # Backward compatibility: use user_prompt as instructions if instructions not provided
    if instructions is None and user_prompt is not None:
        instructions = user_prompt

    # Use provided values or sensible defaults
    context_text = context or "No additional background information provided."
    instructions_text = instructions or "Translate naturally and professionally."
    length_constraint_text = _build_length_constraint(length_limit)

    chain = (
        RunnablePassthrough.assign(
            ppt_context=lambda x: x.get("ppt_context", ""),
            glossary_terms=lambda x: x.get("glossary_terms", "None"),
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
) -> List[str]:
    """Translate batches using LangChain batch API with progress updates.

    Args:
        chain: Configured LangChain runnable sequence with structured output.
        batches: Prepared batch payloads from :func:`chunk_paragraphs`.
        progress_tracker: Optional tracker used to update the UI.
        max_concurrency: Number of batches allowed to run in parallel (>=1).

    Returns:
        Flattened list of translated texts.
    """

    total_batches = len(batches)
    total_sentences = sum(len(batch.get("paragraphs", [])) for batch in batches)

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
        chain, batches, config, progress_tracker, total_batches, accumulator
    )

    # Assemble translations in input order
    translations: List[str] = []
    for index, (batch, result) in enumerate(zip(batches, ordered_results), start=1):
        expected_count = len(batch.get("paragraphs", []))

        if result is None:
            # Should not happen after _batch_translate_with_retry validates,
            # but guard against it to avoid silent data loss.
            raise RuntimeError(
                f"Batch {index} of {len(batches)} returned no result "
                "after all retry attempts"
            )

        parts = result.translations

        if len(parts) != expected_count:
            LOGGER.warning(
                "Batch %d: translation count %d differs from expected %d.",
                index,
                len(parts),
                expected_count,
            )
            retry_result = _retry_count_mismatch_batch(
                chain=chain,
                batch=batch,
                config=config,
                batch_index=index,
                expected_count=expected_count,
            )
            if retry_result is not None and len(retry_result.translations) == expected_count:
                parts = retry_result.translations
            else:
                if retry_result is not None:
                    LOGGER.error(
                        "Batch %d: retry still returned %d translations; "
                        "falling back to safe padding/trimming.",
                        index,
                        len(retry_result.translations),
                    )
                else:
                    LOGGER.error(
                        "Batch %d: retry failed; falling back to safe padding/trimming.",
                        index,
                    )
                originals = [p.original_text for p in batch.get("paragraphs", [])]
                parts = _force_match_expected(parts, expected_count, originals)

        translations.extend(parts)

    return translations


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def _batch_translate_with_retry(
    chain,
    batches: List[Dict[str, object]],
    config: RunnableConfig,
    progress_tracker: Any = None,
    total_batches: int = 0,
    ordered_results: List[TranslationOutput | None] | None = None,
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

    for local_idx, result in chain.batch_as_completed(pending_batches, config=config):
        completed_idx = pending[local_idx]
        ordered_results[completed_idx] = result

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
