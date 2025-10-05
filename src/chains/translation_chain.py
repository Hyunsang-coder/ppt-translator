"""LangChain translation pipeline and progress-aware helpers."""

from __future__ import annotations

import json
import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
If a sentence or phrase appears more than once in the source, translate it identically every time unless the glossary overrides it.
Return EXACTLY {expected_count} translated texts as a JSON array of strings.
Do not add explanations, code fences, or additional fields.
Expected output format: ["translation 1", "translation 2", ...]

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
            expected_count=lambda x: int(x.get("expected_count", 0)),
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
        Raw translation string expected to be JSON formatted.
    """

    LOGGER.debug("Invoking translation chain for batch %s-%s", batch.get("start_idx"), batch.get("end_idx"))
    return chain.invoke(batch)


def translate_with_progress(
    chain,
    batches: List[Dict[str, object]],
    progress_tracker: ProgressTracker | None = None,
    max_concurrency: int = 1,
) -> List[str]:
    """Translate batches while updating Streamlit progress UI.

    Args:
        chain: Configured LangChain runnable sequence.
        batches: Prepared batch payloads from :func:`chunk_paragraphs`.
        progress_tracker: Optional tracker used to update the UI.
        max_concurrency: Number of batches allowed to run in parallel (>=1).

    Returns:
        Flattened list of translated texts.
    """

    translations: List[str] = []
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

    LOGGER.info("Beginning translation of %d batches (total %d sentences).", total_batches, progress_tracker.total_sentences)

    if max_concurrency <= 1 or total_batches <= 1:
        for index, batch in enumerate(batches, start=1):
            start_idx = int(batch.get("start_idx", index))
            end_idx = int(batch.get("end_idx", index))

            LOGGER.info("Translating batch %d/%d (%d-%d).", index, total_batches, start_idx, end_idx)

            parts = _translate_single_batch(chain, batch)
            translations.extend(parts)
            progress_tracker.batch_completed(start_idx, end_idx)

            LOGGER.info("Completed batch %d/%d (received %d parts).", index, total_batches, len(parts))

        return translations

    max_concurrency = max(1, min(int(max_concurrency), total_batches))

    batch_metadata: List[Dict[str, object]] = []
    for index, batch in enumerate(batches, start=1):
        batch_metadata.append(
            {
                "index": index,
                "batch": batch,
                "start_idx": int(batch.get("start_idx", index)),
                "end_idx": int(batch.get("end_idx", index)),
            }
        )

    translations_per_batch: List[List[str] | None] = [None] * total_batches

    def submit_batch(executor: ThreadPoolExecutor, metadata: Dict[str, object], active_futures: Dict) -> None:
        batch_index = metadata["index"]
        LOGGER.info(
            "Submitting batch %d/%d (%d-%d) to executor.",
            batch_index,
            total_batches,
            metadata["start_idx"],
            metadata["end_idx"],
        )
        future = executor.submit(_translate_single_batch, chain, metadata["batch"])
        active_futures[future] = metadata

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        active: Dict = {}
        meta_iterator = iter(batch_metadata)

        try:
            for _ in range(max_concurrency):
                metadata = next(meta_iterator)
                submit_batch(executor, metadata, active)
        except StopIteration:
            pass

        try:
            while active:
                futures = list(active.keys())
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    metadata = active.pop(future)
                    batch_index = metadata["index"]
                    try:
                        parts = future.result()
                    except Exception:
                        for pending in active:
                            pending.cancel()
                        raise

                    translations_per_batch[batch_index - 1] = parts
                    progress_tracker.batch_completed(metadata["start_idx"], metadata["end_idx"])

                    LOGGER.info(
                        "Completed batch %d/%d (received %d parts).",
                        batch_index,
                        total_batches,
                        len(parts),
                    )

                    try:
                        next_metadata = next(meta_iterator)
                    except StopIteration:
                        continue
                    submit_batch(executor, next_metadata, active)
        finally:
            for pending in active:
                pending.cancel()

    for batch_parts in translations_per_batch:
        if batch_parts is not None:
            translations.extend(batch_parts)

    return translations


def _translate_single_batch(chain, batch: Dict[str, object]) -> List[str]:
    """Invoke translation for a single batch and normalise the response length."""

    expected_count = len(batch.get("paragraphs", []))
    raw_result = translate_batch_with_retry(chain, batch)
    parts = _parse_translation_output(raw_result, expected_count)

    if len(parts) != expected_count:
        parts = _force_match_expected(parts, expected_count)

    return parts


def _parse_translation_output(raw_result: str, expected_count: int) -> List[str]:
    """Parse the LLM response favouring strict JSON output."""

    cleaned = raw_result.strip()

    if cleaned.startswith("```"):
        without_prefix = cleaned[3:].lstrip()
        if without_prefix.lower().startswith("json"):
            without_prefix = without_prefix[4:].lstrip()
        cleaned = without_prefix
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            if len(parsed) != expected_count:
                LOGGER.info(
                    "Parsed JSON array length %d differs from expected %d.",
                    len(parsed),
                    expected_count,
                )
            return [str(item).strip() for item in parsed]
    except json.JSONDecodeError:
        LOGGER.debug("Failed to parse translation output as JSON. Falling back to delimiter split.")

    parts = [item.strip() for item in cleaned.split("|||")]
    if len(parts) == 1 and expected_count > 1:
        LOGGER.debug("Delimiter split produced a single item; retrying with newline split.")
        parts = [item.strip() for item in cleaned.splitlines() if item.strip()]

    return parts


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
