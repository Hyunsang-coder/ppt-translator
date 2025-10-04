"""LangChain translation pipeline and progress-aware helpers."""

from __future__ import annotations

import logging
from typing import Dict, List

from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.ui.progress_tracker import ProgressTracker

LOGGER = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
You are a professional translator specializing in PowerPoint presentations.

**Context (Full Presentation):**
{ppt_context}

**Glossary:**
{glossary_terms}

**User Instructions:**
{user_prompt}

**Task:**
Translate the following texts from {source_lang} to {target_lang}.
Maintain consistency with the context and glossary.
Return ONLY the translated texts in the same order, separated by "|||".

**Texts to translate:**
{texts}

**Translations:**
"""


def create_translation_chain(
    model_name: str,
    source_lang: str,
    target_lang: str,
    user_prompt: str | None,
):
    """Create a LangChain sequence for translation relying on ChatGPT models.

    Args:
        model_name: OpenAI model identifier (gpt-5 or gpt-5-mini).
        source_lang: Display name of the source language.
        target_lang: Display name of the target language.
        user_prompt: Optional custom instruction string.

    Returns:
        Configured LangChain runnable sequence.
    """

    llm = ChatOpenAI(model=model_name)
    default_prompt = user_prompt or "Translate naturally and professionally."

    chain = (
        RunnablePassthrough.assign(
            ppt_context=lambda x: x.get("ppt_context", ""),
            glossary_terms=lambda x: x.get("glossary_terms", "None"),
            source_lang=lambda _: source_lang,
            target_lang=lambda _: target_lang,
            user_prompt=lambda _: default_prompt,
        )
        | PromptTemplate.from_template(PROMPT_TEMPLATE)
        | llm
        | StrOutputParser()
    )

    return chain


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def translate_batch_with_retry(chain, batch: Dict[str, object]) -> str:
    """Invoke the chain with retry logic to mitigate transient API issues.

    Args:
        chain: Configured LangChain runnable sequence.
        batch: Batch payload containing texts and context information.

    Returns:
        Raw translation string with entries separated by ``|||``.
    """

    LOGGER.debug("Invoking translation chain for batch %s-%s", batch.get("start_idx"), batch.get("end_idx"))
    return chain.invoke(batch)


def translate_with_progress(
    chain,
    batches: List[Dict[str, object]],
    progress_tracker: ProgressTracker | None = None,
) -> List[str]:
    """Translate batches sequentially while updating Streamlit progress UI.

    Args:
        chain: Configured LangChain runnable sequence.
        batches: Prepared batch payloads from :func:`chunk_paragraphs`.
        progress_tracker: Optional tracker used to update the UI.

    Returns:
        Flattened list of translated texts.
    """

    translations: List[str] = []
    total_batches = len(batches)

    if progress_tracker is None:
        progress_tracker = ProgressTracker(
            total_batches=total_batches,
            total_sentences=sum(len(batch.get("paragraphs", [])) for batch in batches),
        )
    else:
        progress_tracker.total_batches = total_batches
        progress_tracker.total_sentences = sum(len(batch.get("paragraphs", [])) for batch in batches)

    for index, batch in enumerate(batches, start=1):
        start_idx = int(batch.get("start_idx", index))
        end_idx = int(batch.get("end_idx", index))
        progress_tracker.update(index, start_idx, end_idx)

        raw_result = translate_batch_with_retry(chain, batch)
        parts = [item.strip() for item in raw_result.split("|||")]
        expected_count = len(batch.get("paragraphs", []))

        if len(parts) < expected_count:
            LOGGER.warning(
                "Received %d translations but expected %d; padding with empty strings.",
                len(parts),
                expected_count,
            )
            parts.extend(["" for _ in range(expected_count - len(parts))])
        elif len(parts) > expected_count:
            LOGGER.warning(
                "Received %d translations but expected %d; truncating extra results.",
                len(parts),
                expected_count,
            )
            parts = parts[:expected_count]

        translations.extend(parts)
        progress_tracker.complete(index)

    progress_tracker.finish()
    return translations
