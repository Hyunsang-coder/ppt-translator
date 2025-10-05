"""Utilities for identifying repeated paragraphs and applying cached translations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, TYPE_CHECKING

from src.utils.helpers import clean_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.core.ppt_parser import ParagraphInfo


@dataclass(slots=True)
class RepetitionPlan:
    """Mapping from original paragraphs to their canonical occurrences."""

    unique_indices: List[int]
    canonical_map: Dict[int, int | None]
    normalized_texts: List[str]
    counts: Dict[str, int]

    def duplicate_counts(self) -> Dict[str, int]:
        """Return only entries that appear more than once."""

        return {text: count for text, count in self.counts.items() if count > 1}


def build_repetition_plan(paragraphs: Sequence["ParagraphInfo"]) -> RepetitionPlan:
    """Create a plan describing which paragraphs repeat verbatim."""

    normalized_texts: List[str] = [clean_text(info.original_text) for info in paragraphs]
    counts = Counter(text for text in normalized_texts if text)

    canonical_map: Dict[int, int | None] = {}
    seen: Dict[str, int] = {}
    unique_indices: List[int] = []

    for idx, text in enumerate(normalized_texts):
        if not text:
            canonical_map[idx] = None
            continue
        if text not in seen:
            seen[text] = idx
            unique_indices.append(idx)
            canonical_map[idx] = idx
        else:
            canonical_map[idx] = seen[text]

    return RepetitionPlan(
        unique_indices=unique_indices,
        canonical_map=canonical_map,
        normalized_texts=normalized_texts,
        counts=dict(counts),
    )


def expand_translations(plan: RepetitionPlan, unique_translations: Sequence[str], total_count: int) -> List[str]:
    """Expand translations for unique paragraphs back to the full paragraph set."""

    translation_map = {
        unique_idx: unique_translations[pos]
        for pos, unique_idx in enumerate(plan.unique_indices)
    }

    expanded: List[str] = []
    for idx in range(total_count):
        canonical = plan.canonical_map.get(idx)
        if canonical is None:
            expanded.append("")
            continue
        expanded.append(translation_map.get(canonical, ""))

    return expanded


__all__ = ["RepetitionPlan", "build_repetition_plan", "expand_translations"]
