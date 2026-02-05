"""LangChain translation pipeline and progress-aware helpers."""

from __future__ import annotations

import logging
from typing import Dict, List

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from src.chains.llm_factory import Provider, create_llm
from src.ui.progress_tracker import ProgressTracker

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
    provider: Provider = "openai",
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
    progress_tracker: ProgressTracker | None = None,
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

    if progress_tracker is None:
        progress_tracker = ProgressTracker(
            total_batches=total_batches,
            total_sentences=total_sentences,
        )
    else:
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

    # Use LangChain batch API for parallel execution
    config = RunnableConfig(max_concurrency=max(1, int(max_concurrency)))
    results: List[TranslationOutput] = _batch_translate_with_retry(chain, batches, config)

    # Process results and update progress
    translations: List[str] = []
    for index, (batch, result) in enumerate(zip(batches, results), start=1):
        expected_count = len(batch.get("paragraphs", []))
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

        start_idx = int(batch.get("start_idx", index))
        end_idx = int(batch.get("end_idx", index))
        progress_tracker.batch_completed(start_idx, end_idx)

        LOGGER.info(
            "Completed batch %d/%d (received %d parts).",
            index,
            total_batches,
            len(parts),
        )

    return translations


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def _batch_translate_with_retry(
    chain,
    batches: List[Dict[str, object]],
    config: RunnableConfig,
) -> List[TranslationOutput]:
    """Execute batch translation with retry logic.

    Args:
        chain: Configured LangChain runnable sequence with structured output.
        batches: Batch payloads to translate.
        config: RunnableConfig with max_concurrency setting.

    Returns:
        List of TranslationOutput results in input order.
    """
    LOGGER.debug("Invoking batch translation for %d batches.", len(batches))
    return chain.batch(batches, config=config)


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
