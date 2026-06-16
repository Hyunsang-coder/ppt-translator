"""General helper functions for the PPT translator app."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List, Sequence

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from src.core.ppt_parser import ParagraphInfo


def clean_text(text: str) -> str:
    """Normalise paragraph text before sending it to the LLM.

    Args:
        text: Raw paragraph text including potential newlines.

    Returns:
        Sanitised single-line text.
    """

    return (text or "").replace("\r", " ").replace("\n", " ").strip()


def chunk_paragraphs(
    paragraphs: Iterable["ParagraphInfo"],
    batch_size: int,
    ppt_context: str,
    glossary_terms: str,
    prepared_texts: Sequence[str] | None = None,
) -> List[Dict[str, object]]:
    """Split paragraphs into batches suitable for sequential translation.

    Args:
        paragraphs: Iterable of paragraph metadata objects.
        batch_size: Desired maximum number of paragraphs per batch.
        ppt_context: Global presentation context string.
        glossary_terms: Glossary string injected into prompts.
        prepared_texts: Optional list of preprocessed texts aligned with the
            paragraphs, used when glossary substitutions were already applied.

    Returns:
        A list of batch dictionaries ready for the translation chain.
    """

    paragraph_list = list(paragraphs)
    batches: List[Dict[str, object]] = []
    total = len(paragraph_list)

    for start_idx in range(0, total, batch_size):
        chunk = paragraph_list[start_idx : start_idx + batch_size]
        lines = []
        for offset, paragraph in enumerate(chunk, start=1):
            if prepared_texts is not None:
                raw_text = prepared_texts[start_idx + offset - 1]
            else:
                raw_text = paragraph.original_text
            cleaned = clean_text(raw_text)
            text = cleaned or "[EMPTY]"
            lines.append(f"{offset}. {text}")

        batches.append(
            {
                "paragraphs": chunk,
                "texts": "\n".join(lines),
                "ppt_context": ppt_context,
                "glossary_terms": glossary_terms,
                "start_idx": start_idx + 1,
                "end_idx": start_idx + len(chunk),
                "expected_count": len(chunk),
            }
        )

    return batches


def split_text_into_segments(
    text: str,
    segments: int,
    weights: Sequence[int] | None = None,
) -> List[str]:
    """Split text into `segments` parts using approximate weight ratios.

    Args:
        text: The translated text to distribute across runs.
        segments: Number of runs available in the original paragraph.
        weights: Optional weights derived from original run lengths.

    Returns:
        List of text segments whose concatenation equals the original text.
    """

    if segments <= 1:
        return [text]

    text = text or ""
    if not text:
        return ["" for _ in range(segments)]

    if weights is None or sum(weights) <= 0:
        weights = [1] * segments
    else:
        weights = list(weights)

    total_weight = float(sum(weights))
    allocations = [max(1, round(len(text) * (weight / total_weight))) for weight in weights]

    diff = len(text) - sum(allocations)
    index = 0
    max_iterations = max(1, len(allocations) * 4)
    iterations = 0
    while diff != 0 and allocations and iterations < max_iterations:
        adj_index = index % len(allocations)
        proposed = allocations[adj_index] + (1 if diff > 0 else -1)
        if proposed < 1:
            index += 1
            iterations += 1
            continue
        allocations[adj_index] = proposed
        diff = len(text) - sum(allocations)
        index += 1
        iterations += 1

    if diff != 0 and allocations:
        allocations[-1] = max(1, allocations[-1] + diff)

    parts: List[str] = []
    cursor = 0
    for segment_index, allocation in enumerate(allocations):
        if segment_index == len(allocations) - 1:
            parts.append(text[cursor:])
            break
        next_cursor = cursor + allocation
        parts.append(text[cursor:next_cursor])
        cursor = next_cursor

    while len(parts) < segments:
        parts.append("")

    return parts
