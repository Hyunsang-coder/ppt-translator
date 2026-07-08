"""Post-translation review/edit session (WP-C5).

Holds the editable state of a completed translation job in memory so the review
screen can list fragments (source/target + detection badges), edit or
re-translate a single fragment, and propagate a fix to identical fragments — all
without re-parsing the deck. The live ``presentation`` and ``paragraphs`` (with
live python-pptx handles) stay on the session; edits mutate them in place and
re-save the output pptx.

In-memory only (rides the Job's 1h TTL; gone on restart) per the locked design
decision. See ``consulting/ppt-translator-quality-design.md`` §WP-C5.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.core.ppt_parser import ParagraphInfo
from src.core.ppt_writer import PPTWriter, restore_fit_geometry
from src.services.consistency_sweep import Finding
from src.utils.repetition import RepetitionPlan, build_repetition_plan

LOGGER = logging.getLogger(__name__)


@dataclass
class FragmentView:
    """Serializable view of one fragment for the review list (GET response)."""

    index: int
    slide: int  # 1-based
    shape: int
    paragraph: int
    slide_title: Optional[str]
    is_note: bool
    source: str
    target: str
    repeat_count: int  # how many fragments share this normalized source
    findings: List[dict] = field(default_factory=list)  # detection badges
    edited: bool = False


@dataclass
class ReviewSession:
    """Editable, in-memory state for a completed translation job."""

    presentation: object  # live python-pptx Presentation
    paragraphs: List[ParagraphInfo]
    translated_texts: List[str]
    findings: List[Finding]
    source_lang: str
    target_lang: str
    model: Optional[str]
    provider: str = "anthropic"
    text_fit_mode: str = "none"
    min_font_ratio: int = 80
    # Glossary + presentation context strings, in the exact format the
    # translation chain consumed them, so re-translation of a single fragment
    # stays consistent with the original run. Defaults match "no glossary".
    ppt_context: str = ""
    glossary_terms: str = "None"
    # Per-fragment color segments for multi-color paragraphs (fragment index ->
    # list[ColoredSegment]). Seeded from the original run so multi-color
    # fragments keep their colors on re-render; updated on re-translate.
    color_distributions: Dict[int, list] = field(default_factory=dict)
    # C-2: pristine font sizes + shape geometry captured before the first
    # text-fit pass. Restored before every render() so repeated re-renders don't
    # cumulatively shrink fonts / grow boxes. None -> no restore (text_fit off).
    fit_snapshot: Optional[dict] = field(default=None, repr=False)
    # Built lazily from paragraphs; canonical_map drives identical-fragment
    # propagation.
    _plan: Optional[RepetitionPlan] = field(default=None, repr=False)
    edited_indices: set = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        self._plan = build_repetition_plan(self.paragraphs)

    # -- read ---------------------------------------------------------------

    def _repeat_count(self, index: int) -> int:
        assert self._plan is not None
        norm = self._plan.normalized_texts[index]
        return self._plan.counts.get(norm, 1) if norm else 1

    def _findings_by_index(self) -> Dict[int, List[dict]]:
        by_index: Dict[int, List[dict]] = {}
        for f in self.findings:
            if f.fragment_index < 0:
                continue
            by_index.setdefault(f.fragment_index, []).append(
                {
                    "type": f.type,
                    "severity": f.severity,
                    "description": f.description,
                    "suggested_fix": f.suggested_fix,
                    "related_location": f.related_location,
                }
            )
        return by_index

    def fragments(self) -> List[FragmentView]:
        """Return the serializable fragment list for the review screen."""
        by_index = self._findings_by_index()
        views: List[FragmentView] = []
        for idx, info in enumerate(self.paragraphs):
            target = self.translated_texts[idx] if idx < len(self.translated_texts) else ""
            views.append(
                FragmentView(
                    index=idx,
                    slide=info.slide_index + 1,
                    shape=info.shape_index,
                    paragraph=info.paragraph_index,
                    slide_title=info.slide_title,
                    is_note=info.is_note,
                    source=info.original_text or "",
                    target=target or "",
                    repeat_count=self._repeat_count(idx),
                    findings=by_index.get(idx, []),
                    edited=idx in self.edited_indices,
                )
            )
        return views

    # -- propagation --------------------------------------------------------

    def identical_indices(self, index: int) -> List[int]:
        """Indices of fragments with the same normalized source (incl. self)."""
        assert self._plan is not None
        canonical = self._plan.canonical_map.get(index)
        if canonical is None:
            return [index]
        return [
            i for i, c in self._plan.canonical_map.items() if c == canonical
        ]

    def length_budget(self, index: int) -> Optional[int]:
        """Max character budget for a fragment's box (for re-translation).

        Uses the source text length scaled by the expansion policy as a proxy —
        the deterministic box-capacity math lives in the writer's text-fit path;
        here we expose a conservative character budget = source length (slides
        should not expand). Returns None for notes (document text, no box
        constraint).
        """
        info = self.paragraphs[index]
        if info.is_note:
            return None
        src_len = len((info.original_text or "").strip())
        # Slide bodies should read within the box: budget ~= source length.
        # A small allowance avoids over-tight budgets on very short strings.
        return max(src_len, 8)

    # -- write --------------------------------------------------------------

    def apply_edit(
        self,
        index: int,
        new_target: str,
        *,
        propagate_identical: bool = False,
    ) -> List[int]:
        """Apply an edited translation to a fragment (and optional identicals).

        Mutates the in-memory translated_texts. The output pptx is re-rendered
        separately via :meth:`render`. Returns the list of indices changed.
        """
        if not (0 <= index < len(self.translated_texts)):
            raise IndexError(f"fragment index {index} out of range")

        changed = [index]
        self.translated_texts[index] = new_target
        self.edited_indices.add(index)
        # New text no longer matches any stored color segmentation; drop it so
        # the fragment re-renders as safe single-color. Re-translate re-seeds it.
        self.color_distributions.pop(index, None)

        if propagate_identical:
            for other in self.identical_indices(index):
                if other != index:
                    self.translated_texts[other] = new_target
                    self.edited_indices.add(other)
                    self.color_distributions.pop(other, None)
                    changed.append(other)

        LOGGER.info(
            "Applied edit to fragment %d (propagated to %d identical fragments).",
            index,
            len(changed) - 1,
        )
        return changed

    def retranslate_fragment(
        self,
        index: int,
        instruction: Optional[str],
        *,
        model: str,
        provider: str,
    ) -> tuple[str, Optional[list]]:
        """Re-translate one fragment with an optional instruction + length budget.

        Runs a single-item translation chain reusing this session's glossary and
        presentation context so the re-translation stays consistent with the rest
        of the deck. The length budget (WP-C5) is passed as a character
        constraint so slide-box overflows can be tightened.

        Returns ``(new_target, color_segments)`` where ``color_segments`` is a
        list[ColoredSegment] when the fragment is multi-color and was
        successfully re-mapped, else None. Does not mutate session state — the
        caller applies the result via :meth:`apply_edit` /
        :meth:`set_color_distribution`. Raises on translation failure.
        """
        # Imported lazily: these pull in the heavy LangChain translation stack,
        # which the review path only needs when a re-translate is requested.
        from src.chains.translation_chain import (
            create_translation_chain,
            translate_with_progress,
        )
        from src.services.translation_service import TranslationService
        from src.utils.helpers import chunk_paragraphs

        info = self.paragraphs[index]
        budget = self.length_budget(index)

        extra = []
        if instruction:
            extra.append(instruction.strip())
        if budget is not None:
            extra.append(
                f"이 텍스트는 슬라이드 박스에 들어가야 합니다. 번역은 최대 {budget}자 이내로, "
                f"공간 안에서 명확히 읽히도록 간결하게 작성하세요."
            )
        combined_instructions = "\n".join(f"- {e}" for e in extra) if extra else None

        chain = create_translation_chain(
            model_name=model,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            instructions=combined_instructions,
            provider=provider,
        )
        batches = chunk_paragraphs(
            [info],
            batch_size=1,
            ppt_context=self.ppt_context,
            glossary_terms=self.glossary_terms,
        )

        # No progress tracker: a single-fragment re-translation needs no progress
        # UI, and translate_with_progress skips all tracker calls when None.
        results = translate_with_progress(chain, batches, None, max_concurrency=1)
        if not results:
            raise RuntimeError("re-translation returned no result")
        new_target = results[0]

        # Multi-color fragments: re-map the segment colors onto the new
        # translation so highlighted words keep their color. Single-color
        # fragments return None. The helper operates on aligned lists keyed by
        # list-index (0 here); it may replace texts[0] with a more natural
        # multi-color translation.
        color_segments: Optional[list] = None
        try:
            texts = [new_target]
            dist = TranslationService._translate_colored_paragraphs_with_segments(
                [info],
                texts,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                provider=provider,
                model_name=model,
                ppt_context=self.ppt_context,
                context=None,
                instructions=combined_instructions,
                glossary_terms=self.glossary_terms,
                length_limit=None,
            )
            new_target = texts[0]
            if dist:
                color_segments = dist.get(0)
        except Exception:  # pylint: disable=broad-except
            LOGGER.exception(
                "Color re-mapping failed on re-translate; single-color fallback."
            )

        return new_target, color_segments

    def set_color_distribution(self, index: int, segments: Optional[list]) -> None:
        """Store (or clear) multi-color segments for a fragment after re-translate.

        `segments` is a list[ColoredSegment] when the fragment is multi-color and
        was successfully mapped; None/empty clears any stale distribution so the
        fragment renders as single-color.
        """
        if segments:
            self.color_distributions[index] = segments
        else:
            self.color_distributions.pop(index, None)

    def partial_match_candidates(self, index: int, phrase: str) -> List[dict]:
        """Fragments whose target *contains* `phrase` (for partial propagation).

        No auto-substitution (P1: Korean morphology). Returns candidates for the
        user to select from.
        """
        results: List[dict] = []
        if not phrase:
            return results
        needle = phrase.strip()
        if not needle:
            return results
        for i, target in enumerate(self.translated_texts):
            if i == index or not target:
                continue
            if needle in target:
                info = self.paragraphs[i]
                results.append(
                    {
                        "index": i,
                        "slide": info.slide_index + 1,
                        "is_note": info.is_note,
                        "target": target,
                    }
                )
        return results

    def render(self) -> io.BytesIO:
        """Re-render the output pptx from the current translated_texts.

        Reuses the batch writer over the full aligned lists. Formatting is
        preserved by the writer's run-grouping. Text-fit uses the session's
        configured mode.

        C-2: font sizes and shape geometry are reset to the pre-fit snapshot
        first, so re-rendering after an edit doesn't compound the shrink/expand
        from previous renders (fonts would otherwise drift toward 0 and boxes
        keep growing across edits).
        """
        restore_fit_geometry(self.presentation, self.fit_snapshot)
        writer = PPTWriter()
        return writer.apply_translations(
            self.paragraphs,
            self.translated_texts,
            self.presentation,
            text_fit_mode=self.text_fit_mode,
            min_font_ratio=self.min_font_ratio,
            color_distributions=self.color_distributions or None,
        )
