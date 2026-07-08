"""LLM chain for distributing translated text across format groups."""

from __future__ import annotations

import logging
import json
from typing import List

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, TypeVar

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, field_validator

from src.chains.llm_factory import Provider, create_llm
from src.utils.config import get_settings

LOGGER = logging.getLogger(__name__)

# Maximum number of paragraphs per LLM call to avoid token overflow and
# reduce blast radius when a single call fails.
_BATCH_SIZE = 8

_T = TypeVar("_T")


def _color_concurrency() -> int:
    """Concurrency for the colored-paragraph batch loops (P-2).

    Reuses the translation concurrency budget so the color passes fan out the
    same way the main translation does instead of running strictly serially.
    """
    try:
        return max(1, int(get_settings().max_concurrency))
    except Exception:  # pragma: no cover - defensive; settings always load
        return 1


def _run_batches_concurrently(
    starts: list[int],
    call: "Callable[[int], Optional[_T]]",
) -> dict[int, "Optional[_T]"]:
    """Invoke ``call(start)`` for each batch start concurrently (P-2).

    Returns a mapping ``start -> result`` (result is whatever ``call`` returns,
    including None on failure). A single batch raising never aborts the others;
    its slot is recorded as None so the caller applies its per-batch fallback.
    Serial when there is 0/1 batch to avoid pool overhead.
    """
    if len(starts) <= 1:
        return {start: call(start) for start in starts}

    workers = min(_color_concurrency(), len(starts))
    results: dict[int, "Optional[_T]"] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_start = {executor.submit(call, start): start for start in starts}
        for future in future_to_start:
            start = future_to_start[future]
            try:
                results[start] = future.result()
            except Exception:  # pylint: disable=broad-except
                LOGGER.exception("Colored batch starting at %d raised.", start)
                results[start] = None
    return results


class ColoredSegment(BaseModel):
    """A segment of translated text mapped to a format group."""

    text: str = Field(description="Text content for this segment")
    group_index: int = Field(description="Index of the original format group (0-based)")


class ColoredTranslation(BaseModel):
    """Natural translation plus color/style mapping for one paragraph."""

    translation: str = Field(description="Natural full translation for the paragraph")
    segments: List[ColoredSegment] = Field(
        description="Segments in translation word order. Concatenating text fields "
        "must exactly equal translation."
    )

    @field_validator("segments", mode="before")
    @classmethod
    def parse_stringified_segments(cls, value):
        """Accept providers that return nested segment arrays as JSON strings."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value


class ColoredTranslationOutput(BaseModel):
    """Structured output for translating colored paragraphs."""

    items: List[ColoredTranslation] = Field(
        description="One translated item per input paragraph, in the same order."
    )

    @field_validator("items", mode="before")
    @classmethod
    def parse_stringified_items(cls, value):
        """Accept providers that return the items array as a JSON-encoded string."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value


class ColorDistributionOutput(BaseModel):
    """Structured output: text segments for each paragraph's format groups."""

    distributions: List[List[ColoredSegment]] = Field(
        description="For each paragraph, a list of ColoredSegment objects. "
        "Each segment has text and the group_index it belongs to. "
        "Segments are in translation word order (may differ from original)."
    )

    @field_validator("distributions", mode="before")
    @classmethod
    def parse_stringified_distributions(cls, value):
        """Accept providers that return the array as a JSON-encoded string."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value


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


COLORED_TRANSLATION_PROMPT = """You are a professional translator specializing in PowerPoint presentations.
Slide body text favors noun phrases and concise phrasing over full sentences. Avoid target-side expansion (especially when translating Korean to English) and prefer wording that reads clearly within a limited text box.

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
Translate each item naturally from {source_lang} to {target_lang}.
Each item is split into original formatting groups. The groups are ONLY hints for
which translated words should inherit each original style/color.

Return exactly {expected_count} items in the items array.
For each item:
- translation: a natural full-sentence translation. Do NOT translate group-by-group.
- segments: pieces of the translation in final reading order.
- Concatenating every segment.text must exactly equal translation.
- Each segment.group_index must point to the original formatting group whose meaning it carries.
- Prefer natural phrasing over preserving source word order.
- If one highlighted source group maps to several possible translated words, tag the smallest clear phrase.
- If the meaning is ambiguous or diffused, assign uncertain connective/filler text to the surrounding/base group rather than forcing a highlight.
- Numeric, symbolic, product, or proper-noun text that survives translation should keep the matching source group.

Items:
{items}
"""


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


def _format_source_items(original_groups: list[list[str]]) -> str:
    """Format source-only colored paragraph items for translation."""
    lines = []
    for i, groups in enumerate(original_groups, start=1):
        group_display = " | ".join(f'[{j}]"{g}"' for j, g in enumerate(groups))
        lines.append(f'{i}. 원본 구간({len(groups)}개): [{group_display}]')
    return "\n".join(lines)


def _build_length_constraint(length_limit: int | None) -> str:
    if length_limit is None:
        return ""
    return (
        f"\n**Length Constraint:**\n"
        f"Keep each translation concise and do not exceed {length_limit}% "
        f"of the source character length unless required for natural phrasing.\n"
    )


def _invoke_colored_translation_batch(
    chain,
    original_groups: list[list[str]],
) -> list[ColoredTranslation] | None:
    """Translate and map one batch of colored paragraphs."""
    items_str = _format_source_items(original_groups)

    try:
        result: ColoredTranslationOutput = chain.invoke(
            {
                "items": items_str,
                "expected_count": len(original_groups),
            }
        )
        return result.items
    except Exception:
        LOGGER.exception(
            "Colored translation chain failed for batch of %d",
            len(original_groups),
        )
        return None


def translate_with_color_segments(
    original_groups: list[list[str]],
    *,
    source_lang: str,
    target_lang: str,
    provider: Provider = "openai",
    model_name: str | None = None,
    ppt_context: str = "",
    context: str | None = None,
    instructions: str | None = None,
    glossary_terms: str = "None",
    length_limit: int | None = None,
    team_rules: str = "None",
) -> list[ColoredTranslation | None] | None:
    """Translate colored paragraphs and return semantic style segments."""
    if not original_groups:
        return None

    total = len(original_groups)
    LOGGER.info(
        "Translating %d colored paragraphs with provider=%s, model=%s (batch_size=%d)",
        total,
        provider,
        model_name,
        _BATCH_SIZE,
    )

    llm = create_llm(
        provider=provider,
        model_name=model_name,
        max_tokens=4096,
        temperature=0,
    )
    chain = (
        PromptTemplate(
            input_variables=["items", "expected_count"],
            template=COLORED_TRANSLATION_PROMPT,
        ).partial(
            ppt_context=ppt_context,
            context=context or "No additional background information provided.",
            glossary_terms=glossary_terms,
            team_rules=team_rules,
            instructions=instructions or "Translate naturally and professionally.",
            length_constraint=_build_length_constraint(length_limit),
            source_lang=source_lang,
            target_lang=target_lang,
        )
        | llm.with_structured_output(ColoredTranslationOutput)
    )

    translated_items: list[ColoredTranslation | None] = [None] * total

    # P-2: run the independent per-batch LLM calls concurrently instead of
    # strictly serial. Each batch call is thread-safe (own request); results are
    # written back by their original start offset so ordering is preserved.
    starts = list(range(0, total, _BATCH_SIZE))
    results_by_start = _run_batches_concurrently(
        starts,
        lambda start: _invoke_colored_translation_batch(
            chain, original_groups[start : start + _BATCH_SIZE]
        ),
    )

    any_success = False
    for start in starts:
        end = min(start + _BATCH_SIZE, total)
        batch_groups = original_groups[start:end]
        batch_result = results_by_start.get(start)

        if batch_result is not None and len(batch_result) == len(batch_groups):
            for i, item in enumerate(batch_result):
                translated_items[start + i] = item
            any_success = True
        else:
            LOGGER.warning(
                "Colored translation batch [%d:%d] failed or returned wrong count "
                "(expected %d, got %s). These paragraphs will use fallback.",
                start, end, len(batch_groups),
                len(batch_result) if batch_result else "None",
            )

    if not any_success:
        return None

    return translated_items


def _invoke_batch(
    chain,
    original_groups: list[list[str]],
    translated_texts: list[str],
) -> list[list[ColoredSegment]] | None:
    """Invoke a prebuilt chain for a single batch of paragraphs.

    Returns:
        List of distributions for the batch, or None on failure.
    """
    items_str = _format_items(original_groups, translated_texts)

    try:
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
) -> list[list[ColoredSegment] | None] | None:
    """Call LLM to distribute translated text across format groups.

    Splits items into smaller batches to improve accuracy and reduce the
    blast radius of individual call failures.

    Args:
        original_groups: For each paragraph, list of group texts.
        translated_texts: Translated text for each paragraph.
        provider: LLM provider to use.
        model_name: Model to use. If None, defaults to provider's default.

    Returns:
        List of distributions (one per paragraph, each a list of ColoredSegment
        or None when that paragraph's batch failed), or None on complete failure.
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

    # Build the LLM and chain once and reuse across batches. Creating them
    # per batch would also spin up a fresh per-instance rate limiter each
    # time, defeating rate limiting across the batch loop.
    llm = create_llm(
        provider=provider,
        model_name=model_name,
        max_tokens=4096,
        temperature=0,
    )
    chain = (
        PromptTemplate(input_variables=["items"], template=COLOR_DISTRIBUTION_PROMPT)
        | llm.with_structured_output(ColorDistributionOutput)
    )

    all_distributions: list[list[ColoredSegment] | None] = [None] * total

    # P-2: fan the independent per-batch calls out concurrently.
    starts = list(range(0, total, _BATCH_SIZE))
    results_by_start = _run_batches_concurrently(
        starts,
        lambda start: _invoke_batch(
            chain,
            original_groups[start : start + _BATCH_SIZE],
            translated_texts[start : start + _BATCH_SIZE],
        ),
    )

    any_success = False
    for start in starts:
        end = min(start + _BATCH_SIZE, total)
        batch_groups = original_groups[start:end]
        batch_result = results_by_start.get(start)

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
