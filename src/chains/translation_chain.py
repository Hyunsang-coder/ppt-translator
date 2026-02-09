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

**Task:**
Translate the following texts from {source_lang} to {target_lang}.
Maintain consistency with the context, background information, and glossary.
Follow the translation style/tone guidelines if provided.
If a sentence or phrase appears more than once in the source, translate it identically every time unless the glossary overrides it.
Return exactly {expected_count} translated texts in the translations array.

**Texts to translate:**
{texts}
"""


def create_translation_chain(
    model_name: str,
    source_lang: str,
    target_lang: str,
    context: str | None = None,
    instructions: str | None = None,
    provider: Provider = "anthropic",
    *,
    user_prompt: str | None = None,  # Deprecated: for backward compatibility
):
    """Create a LangChain sequence for translation.

    Args:
        model_name: Model identifier (e.g., gpt-5.2, claude-sonnet-4-5-20250929).
        source_lang: Display name of the source language.
        target_lang: Display name of the target language.
        context: Optional background information about the presentation.
        instructions: Optional translation style/tone guidelines.
        provider: LLM provider ("openai" or "anthropic").
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

    chain = (
        RunnablePassthrough.assign(
            ppt_context=lambda x: x.get("ppt_context", ""),
            glossary_terms=lambda x: x.get("glossary_terms", "None"),
            source_lang=lambda _: source_lang,
            target_lang=lambda _: target_lang,
            context=lambda _: context_text,
            instructions=lambda _: instructions_text,
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

    # Use LangChain batch_as_completed for real-time progress updates
    config = RunnableConfig(max_concurrency=max(1, int(max_concurrency)))
    ordered_results: List[TranslationOutput | None] = _batch_translate_with_retry(
        chain, batches, config, progress_tracker, total_batches
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
            LOGGER.debug(
                "Batch %d: translation count %d differs from expected %d.",
                index,
                len(parts),
                expected_count,
            )
            parts = _force_match_expected(parts, expected_count)

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
) -> List[TranslationOutput | None]:
    """Execute batch translation with retry logic and real-time progress.

    Uses ``batch_as_completed`` so that each batch reports progress as soon
    as it finishes, instead of waiting for all batches to complete.

    Args:
        chain: Configured LangChain runnable sequence with structured output.
        batches: Batch payloads to translate.
        config: RunnableConfig with max_concurrency setting.
        progress_tracker: Optional tracker for real-time progress updates.
        total_batches: Total number of batches (for logging).

    Returns:
        List of TranslationOutput results in input order.
    """
    # Reset progress for this attempt — on the first call this is
    # redundant (translate_with_progress already called reset), but on
    # tenacity retries it prevents batch_completed() from double-counting.
    if progress_tracker is not None:
        total_sentences = sum(len(b.get("paragraphs", [])) for b in batches)
        progress_tracker.reset(
            total_batches=total_batches,
            total_sentences=total_sentences,
        )

    LOGGER.debug("Invoking batch translation for %d batches.", len(batches))
    ordered_results: List[TranslationOutput | None] = [None] * len(batches)

    for completed_idx, result in chain.batch_as_completed(batches, config=config):
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


def _force_match_expected(parts: List[str], expected_count: int) -> List[str]:
    """Pad or trim decoded translations without emitting warnings."""

    if len(parts) < expected_count:
        missing = expected_count - len(parts)
        LOGGER.debug("Padding %d missing translations with empty strings due to format mismatch.", missing)
        parts = parts + ["" for _ in range(missing)]
    elif len(parts) > expected_count:
        LOGGER.debug(
            "Received %d translations but expected %d; trimming extras after successful parse.",
            len(parts),
            expected_count,
        )
        parts = parts[:expected_count]

    return parts
