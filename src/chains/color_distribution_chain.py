"""LLM chain for distributing translated text across format groups."""

from __future__ import annotations

import logging
from typing import List

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from src.chains.llm_factory import Provider, create_llm

LOGGER = logging.getLogger(__name__)

# Maximum number of paragraphs per LLM call to avoid token overflow and
# reduce blast radius when a single call fails.
_BATCH_SIZE = 8


class ColoredSegment(BaseModel):
    """A segment of translated text mapped to a format group."""

    text: str = Field(description="Text content for this segment")
    group_index: int = Field(description="Index of the original format group (0-based)")


class ColorDistributionOutput(BaseModel):
    """Structured output: text segments for each paragraph's format groups."""

    distributions: List[List[ColoredSegment]] = Field(
        description="For each paragraph, a list of ColoredSegment objects. "
        "Each segment has text and the group_index it belongs to. "
        "Segments are in translation word order (may differ from original)."
    )


COLOR_DISTRIBUTION_PROMPT = """원본 텍스트의 서식 구간별 텍스트와 번역 결과가 주어집니다.
번역된 텍스트를 원본 서식 구간의 의미에 맞게 분배하세요.

예시:
입력: 원본 구간(2개): [[0]"Revenue " | [1]"increased by 20%"], 번역: "매출이 20% 증가했습니다"
출력: [{{"text": "매출이 ", "group_index": 0}}, {{"text": "20% 증가했습니다", "group_index": 1}}]

입력: 원본 구간(3개): [[0]"Click " | [1]"here" | [2]" to continue"], 번역: "계속하려면 여기를 클릭하세요"
출력: [{{"text": "계속하려면 ", "group_index": 2}}, {{"text": "여기", "group_index": 1}}, {{"text": "를 클릭하세요", "group_index": 0}}]

입력: 원본 구간(2개): [[0]"Total: " | [1]"$1,500"], 번역: "합계: $1,500"
출력: [{{"text": "합계: ", "group_index": 0}}, {{"text": "$1,500", "group_index": 1}}]

지금 분배할 항목:
{items}

규칙:
- 각 세그먼트에 해당하는 원본 구간 번호(group_index)를 지정하세요.
- 세그먼트 순서는 번역문의 어순을 따르세요 (원본과 달라도 됩니다).
- 모든 세그먼트의 text를 이어 붙이면 번역 텍스트와 정확히 동일해야 합니다. 공백 하나도 빠지면 안 됩니다.
- 원본 각 구간의 의미에 대응하는 번역 부분을 해당 group_index에 배치하세요.
- 대응하는 의미가 없는 구간은 빈 문자열("")로 채우세요.
- 숫자, 기호, 고유명사 등 번역 후에도 동일한 텍스트는 반드시 해당 원본 구간에 배치하세요."""


def _format_items(
    original_groups: list[list[str]],
    translated_texts: list[str],
) -> str:
    """Format paragraph items for the prompt.

    Args:
        original_groups: For each paragraph, list of group texts (one string per group).
        translated_texts: The translated text for each paragraph.

    Returns:
        Formatted string for prompt injection.
    """
    lines = []
    for i, (groups, translation) in enumerate(zip(original_groups, translated_texts), start=1):
        group_display = " | ".join(f'[{j}]"{g}"' for j, g in enumerate(groups))
        lines.append(
            f'{i}. 원본 구간({len(groups)}개): [{group_display}], 번역: "{translation}"'
        )
    return "\n".join(lines)


def _invoke_batch(
    original_groups: list[list[str]],
    translated_texts: list[str],
    provider: Provider,
    model_name: str | None,
) -> list[list[ColoredSegment]] | None:
    """Invoke the LLM for a single batch of paragraphs.

    Returns:
        List of distributions for the batch, or None on failure.
    """
    items_str = _format_items(original_groups, translated_texts)

    try:
        llm = create_llm(
            provider=provider,
            model_name=model_name,
            max_tokens=4096,
            temperature=0,
        )
        structured_llm = llm.with_structured_output(ColorDistributionOutput)

        prompt = PromptTemplate(
            input_variables=["items"],
            template=COLOR_DISTRIBUTION_PROMPT,
        )
        chain = prompt | structured_llm

        result: ColorDistributionOutput = chain.invoke({"items": items_str})
        return result.distributions
    except Exception:
        LOGGER.exception("Color distribution chain failed for batch of %d", len(original_groups))
        return None


def distribute_colors(
    original_groups: list[list[str]],
    translated_texts: list[str],
    provider: Provider = "openai",
    model_name: str | None = None,
) -> list[list[ColoredSegment]] | None:
    """Call LLM to distribute translated text across format groups.

    Splits items into smaller batches to improve accuracy and reduce the
    blast radius of individual call failures.

    Args:
        original_groups: For each paragraph, list of group texts.
        translated_texts: Translated text for each paragraph.
        provider: LLM provider to use.
        model_name: Model to use. If None, defaults to provider's default.

    Returns:
        List of distributions (one per paragraph, each a list of ColoredSegment),
        or None on complete failure.
    """
    if not original_groups:
        return None

    total = len(original_groups)
    LOGGER.info(
        "Distributing colors for %d paragraphs with provider=%s, model=%s (batch_size=%d)",
        total,
        provider,
        model_name,
        _BATCH_SIZE,
    )

    all_distributions: list[list[ColoredSegment] | None] = [None] * total
    any_success = False

    for start in range(0, total, _BATCH_SIZE):
        end = min(start + _BATCH_SIZE, total)
        batch_groups = original_groups[start:end]
        batch_texts = translated_texts[start:end]

        batch_result = _invoke_batch(batch_groups, batch_texts, provider, model_name)

        if batch_result is not None and len(batch_result) == len(batch_groups):
            for i, dist in enumerate(batch_result):
                all_distributions[start + i] = dist
            any_success = True
        else:
            LOGGER.warning(
                "Color distribution batch [%d:%d] failed or returned wrong count "
                "(expected %d, got %s). These paragraphs will use fallback.",
                start, end, len(batch_groups),
                len(batch_result) if batch_result else "None",
            )

    if not any_success:
        return None

    return all_distributions
