"""LLM chain for distributing translated text across format groups."""

from __future__ import annotations

import logging
from typing import List

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from src.chains.llm_factory import Provider, create_llm

LOGGER = logging.getLogger(__name__)

LIGHTWEIGHT_MODELS = {
    "openai": "gpt-5-mini",
    "anthropic": "claude-haiku-4-5",
}


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

{items}

규칙:
- 각 세그먼트에 해당하는 원본 구간 번호(group_index)를 지정하세요.
- 세그먼트 순서는 번역문의 어순을 따르세요 (원본과 달라도 됩니다).
- 모든 세그먼트의 text를 이어 붙이면 번역 텍스트와 정확히 동일해야 합니다.
- 원본 각 구간의 의미에 대응하는 번역 부분을 해당 group_index에 배치하세요.
- 대응하는 의미가 없는 구간은 빈 문자열("")로 채우세요."""


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


def distribute_colors(
    original_groups: list[list[str]],
    translated_texts: list[str],
    provider: Provider = "openai",
) -> list[list[ColoredSegment]] | None:
    """Call lightweight LLM to distribute translated text across format groups.

    Args:
        original_groups: For each paragraph, list of group texts.
        translated_texts: Translated text for each paragraph.
        provider: LLM provider to use.

    Returns:
        List of distributions (one per paragraph, each a list of ColoredSegment),
        or None on failure.
    """
    if not original_groups:
        return None

    model = LIGHTWEIGHT_MODELS.get(provider, "gpt-5-mini")
    items_str = _format_items(original_groups, translated_texts)

    LOGGER.info(
        "Distributing colors for %d paragraphs with provider=%s, model=%s",
        len(original_groups),
        provider,
        model,
    )

    try:
        llm = create_llm(provider=provider, model_name=model, max_tokens=4096)
        structured_llm = llm.with_structured_output(ColorDistributionOutput)

        prompt = PromptTemplate(
            input_variables=["items"],
            template=COLOR_DISTRIBUTION_PROMPT,
        )
        chain = prompt | structured_llm

        result: ColorDistributionOutput = chain.invoke({"items": items_str})
        return result.distributions
    except Exception:
        LOGGER.exception("Color distribution chain failed")
        return None
