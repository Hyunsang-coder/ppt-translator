"""Transactional post-translation review/edit sessions.

Review edits are kept as an in-memory draft.  The published PPTX is only
re-rendered when the user commits the draft, and every render starts from the
pristine source PPTX.  This is important for run-level formatting: deriving
format groups from an already translated paragraph makes reordered highlights
swap colours on every subsequent render.
"""

from __future__ import annotations

import io
import logging
import math
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from lxml import etree

from src.core.ppt_parser import PPTParser, ParagraphInfo
from src.core.ppt_writer import PPTWriter, _group_runs_by_format
from src.services.consistency_sweep import Finding, run_sweep
from src.utils.repetition import RepetitionPlan, build_repetition_plan

LOGGER = logging.getLogger(__name__)
_EMU_PER_POINT = 12700
_A_NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}


@dataclass
class FragmentView:
    """Serializable view of one fragment for the review list."""

    index: int
    slide: int
    shape: int
    paragraph: int
    slide_title: Optional[str]
    is_note: bool
    source: str
    target: str
    repeat_count: int
    length_budget: Optional[int]
    findings: List[dict] = field(default_factory=list)
    edited: bool = False
    style_segments: List[dict] = field(default_factory=list)
    style_status: str = "single_style"


@dataclass
class ProposedEdit:
    """Server-side candidate that can be compared before it is staged."""

    id: str
    index: int
    base_revision: int
    old_target: str
    target: str
    changed_indices: List[int]
    propagate_identical: bool
    color_distributions: Dict[int, list]
    style_segments: List[dict]
    style_status: str
    partial_candidates: List[dict]
    over_budget: bool


@dataclass
class _DraftSnapshot:
    texts: Dict[int, str]
    colors: Dict[int, Optional[list]]
    edited_indices: set


@dataclass
class ReviewSession:
    """Editable draft state for a completed translation job."""

    # ``presentation`` remains for backward compatibility with callers/tests,
    # but renders never mutate it.  ``source_pptx`` is the source of truth.
    presentation: object
    paragraphs: List[ParagraphInfo]
    translated_texts: List[str]
    findings: List[Finding]
    source_lang: str
    target_lang: str
    model: Optional[str]
    provider: str = "anthropic"
    text_fit_mode: str = "none"
    min_font_ratio: int = 80
    ppt_context: str = ""
    glossary_terms: str = "None"
    glossary: Optional[Dict[str, str]] = None
    context: Optional[str] = None
    instructions: Optional[str] = None
    team_rules: str = "None"
    locked_terms: Optional[Dict[str, str]] = None
    color_distributions: Dict[int, list] = field(default_factory=dict)
    # Deprecated compatibility field. Fresh-source rendering makes geometry
    # restoration unnecessary, but older tests/callers can still provide it.
    fit_snapshot: Optional[dict] = field(default=None, repr=False)
    source_pptx: Optional[bytes] = field(default=None, repr=False)
    revision: int = 0
    committed_revision: int = 0
    _plan: Optional[RepetitionPlan] = field(default=None, repr=False)
    edited_indices: set = field(default_factory=set, repr=False)
    dismissed_findings: set = field(default_factory=set, repr=False)
    _source_paragraphs: List[ParagraphInfo] = field(default_factory=list, repr=False)
    _theme_colors: Dict[str, str] = field(default_factory=dict, repr=False)
    _history: List[_DraftSnapshot] = field(default_factory=list, repr=False)
    _proposals: Dict[str, ProposedEdit] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.translated_texts = list(self.translated_texts)
        self.color_distributions = {
            idx: list(segments) for idx, segments in self.color_distributions.items()
        }
        self._plan = build_repetition_plan(self.paragraphs)

        if self.source_pptx is None:
            buffer = io.BytesIO()
            self.presentation.save(buffer)
            self.source_pptx = buffer.getvalue()

        self._source_paragraphs, source_presentation = PPTParser().extract_paragraphs(
            io.BytesIO(self.source_pptx),
            translate_notes=any(info.is_note for info in self.paragraphs),
        )
        self._theme_colors = self._extract_theme_colors(source_presentation)
        if len(self._source_paragraphs) != len(self.paragraphs):
            raise ValueError(
                "source paragraph alignment changed: "
                f"{len(self._source_paragraphs)} != {len(self.paragraphs)}"
            )

    @property
    def dirty(self) -> bool:
        return self.revision != self.committed_revision

    def _repeat_count(self, index: int) -> int:
        assert self._plan is not None
        norm = self._plan.normalized_texts[index]
        return self._plan.counts.get(norm, 1) if norm else 1

    @staticmethod
    def _finding_key(index: int, finding_type: str) -> tuple[int, str]:
        return (index, finding_type)

    def _findings_by_index(self) -> Dict[int, List[dict]]:
        by_index: Dict[int, List[dict]] = {}
        for finding in self.findings:
            if finding.fragment_index < 0:
                continue
            key = self._finding_key(finding.fragment_index, finding.type)
            if key in self.dismissed_findings:
                continue
            by_index.setdefault(finding.fragment_index, []).append(
                {
                    "type": finding.type,
                    "severity": finding.severity,
                    "description": finding.description,
                    "suggested_fix": finding.suggested_fix,
                    "related_location": finding.related_location,
                }
            )
        return by_index

    @staticmethod
    def _extract_theme_colors(presentation) -> Dict[str, str]:
        colors: Dict[str, str] = {}
        if not presentation.slide_masters:
            return colors
        master_part = presentation.slide_masters[0].part
        for rel in master_part.rels.values():
            if not rel.reltype.endswith("/theme"):
                continue
            root = etree.fromstring(rel.target_part.blob)
            scheme = root.find(".//a:clrScheme", _A_NS)
            if scheme is None:
                continue
            for entry in scheme:
                name = etree.QName(entry).localname
                child = next(iter(entry), None)
                if child is None:
                    continue
                value = child.get("val")
                if etree.QName(child).localname == "sysClr":
                    value = child.get("lastClr") or value
                if value and len(value) == 6:
                    colors[name] = f"#{value.upper()}"
            break
        colors.setdefault("tx1", colors.get("dk1", ""))
        colors.setdefault("tx2", colors.get("dk2", ""))
        colors.setdefault("bg1", colors.get("lt1", ""))
        colors.setdefault("bg2", colors.get("lt2", ""))
        return {key: value for key, value in colors.items() if value}

    def _style_descriptor(self, group_index: int, group) -> dict:
        run = group[0]
        rpr = run._r.rPr
        color: Optional[str] = None
        scheme: Optional[str] = None
        if rpr is not None:
            srgb = rpr.find(".//a:srgbClr", _A_NS)
            if srgb is not None and srgb.get("val"):
                color = f"#{srgb.get('val').upper()}"
            scheme_clr = rpr.find(".//a:schemeClr", _A_NS)
            if scheme_clr is not None:
                scheme = scheme_clr.get("val")
                if color is None and scheme:
                    color = self._theme_colors.get(scheme)
        return {
            "group_index": group_index,
            "color": color,
            "scheme": scheme,
            "bold": bool(getattr(run.font, "bold", False)),
            "italic": bool(getattr(run.font, "italic", False)),
        }

    def style_preview(
        self,
        index: int,
        *,
        target: Optional[str] = None,
        segments: Optional[list] = None,
    ) -> tuple[List[dict], str]:
        """Return target segments plus source style metadata for the UI."""
        info = self._source_paragraphs[index]
        groups = _group_runs_by_format(list(info.paragraph.runs))
        text = self.translated_texts[index] if target is None else target
        if len(groups) <= 1:
            style = self._style_descriptor(0, groups[0]) if groups else {
                "group_index": 0, "color": None, "scheme": None,
                "bold": False, "italic": False,
            }
            return [{"text": text, **style}], "single_style"

        distribution = (
            self.color_distributions.get(index) if segments is None else segments
        )
        if not distribution:
            style = self._style_descriptor(0, groups[0])
            return [{"text": text, **style}], "dropped"

        used: set[int] = set()
        rendered: List[dict] = []
        for segment in distribution:
            group_index = int(segment.group_index)
            if not (0 <= group_index < len(groups)):
                continue
            used.add(group_index)
            rendered.append(
                {"text": segment.text, **self._style_descriptor(group_index, groups[group_index])}
            )
        non_empty_groups = {
            idx for idx, group in enumerate(groups)
            if "".join(run.text or "" for run in group).strip()
        }
        status = "preserved" if non_empty_groups.issubset(used) else "partial"
        return rendered, status

    def fragments(self) -> List[FragmentView]:
        by_index = self._findings_by_index()
        views: List[FragmentView] = []
        for idx, info in enumerate(self.paragraphs):
            target = self.translated_texts[idx] if idx < len(self.translated_texts) else ""
            style_segments, style_status = self.style_preview(idx, target=target)
            findings = list(by_index.get(idx, []))
            if style_status in {"dropped", "partial"}:
                findings.append(
                    {
                        "type": "style.mapping_dropped",
                        "severity": "minor",
                        "description": (
                            "원문의 부분 강조를 모두 확실하게 매핑하지 못했습니다. "
                            "색상 미리보기를 확인하세요."
                        ),
                        "suggested_fix": None,
                        "related_location": None,
                    }
                )
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
                    length_budget=self.length_budget(idx),
                    findings=findings,
                    edited=idx in self.edited_indices,
                    style_segments=style_segments,
                    style_status=style_status,
                )
            )
        return views

    def identical_indices(self, index: int) -> List[int]:
        assert self._plan is not None
        canonical = self._plan.canonical_map.get(index)
        if canonical is None:
            return [index]
        return [i for i, candidate in self._plan.canonical_map.items() if candidate == canonical]

    def length_budget(self, index: int) -> Optional[int]:
        """Estimate target capacity from actual text-frame geometry and font size."""
        info = self._source_paragraphs[index]
        if info.is_note:
            return None
        paragraph = info.paragraph
        text_frame = paragraph._parent
        owner = getattr(text_frame, "_parent", None)
        width = getattr(owner, "width", None)
        height = getattr(owner, "height", None)
        source_len = len((info.original_text or "").strip())
        if not width or not height:
            return max(source_len, 8)

        width_pt = width / _EMU_PER_POINT
        height_pt = height / _EMU_PER_POINT
        horizontal_margin = (
            (getattr(text_frame, "margin_left", 0) or 0)
            + (getattr(text_frame, "margin_right", 0) or 0)
        ) / _EMU_PER_POINT
        vertical_margin = (
            (getattr(text_frame, "margin_top", 0) or 0)
            + (getattr(text_frame, "margin_bottom", 0) or 0)
        ) / _EMU_PER_POINT
        usable_width = max(1.0, width_pt - horizontal_margin)
        usable_height = max(1.0, height_pt - vertical_margin)

        sizes = [run.font.size.pt for run in paragraph.runs if run.font.size is not None]
        font_pt = sum(sizes) / len(sizes) if sizes else 18.0
        cjk_target = self.target_lang in {"한국어", "일본어", "중국어"}
        char_width = font_pt * (1.0 if cjk_target else 0.55)
        line_height = font_pt * 1.2
        chars_per_line = max(1, math.floor(usable_width / max(char_width, 1.0)))
        lines = max(1, math.floor(usable_height / max(line_height, 1.0)))
        capacity = chars_per_line * lines
        if self.text_fit_mode in {"expand_box", "shrink_then_expand"}:
            capacity = math.floor(capacity * 1.3)
        return max(8, capacity)

    def _snapshot(self, indices: List[int]) -> _DraftSnapshot:
        return _DraftSnapshot(
            texts={idx: self.translated_texts[idx] for idx in indices},
            colors={
                idx: deepcopy(self.color_distributions.get(idx))
                for idx in indices
            },
            edited_indices=set(self.edited_indices),
        )

    def apply_edit(
        self,
        index: int,
        new_target: str,
        *,
        propagate_identical: bool = False,
        color_segments_by_index: Optional[Dict[int, Optional[list]]] = None,
    ) -> List[int]:
        """Stage an edit without rendering the published PPTX."""
        if not (0 <= index < len(self.translated_texts)):
            raise IndexError(f"fragment index {index} out of range")
        indices = self.identical_indices(index) if propagate_identical else [index]
        changed = [idx for idx in indices if self.translated_texts[idx] != new_target]
        if not changed:
            return []

        self._history.append(self._snapshot(changed))
        for idx in changed:
            self.translated_texts[idx] = new_target
            self.edited_indices.add(idx)
            mapped = (color_segments_by_index or {}).get(idx)
            if mapped:
                self.color_distributions[idx] = list(mapped)
            else:
                self.color_distributions.pop(idx, None)
        self.revision += 1
        return changed

    def _map_color_distribution(
        self, index: int, target: str, *, model: str, provider: str
    ) -> Optional[list]:
        from src.services.translation_service import TranslationService

        info = self._source_paragraphs[index]
        distributions = TranslationService._fix_color_distributions(
            [info], [target], provider=provider, model_name=model
        )
        return distributions.get(0) if distributions else None

    def retranslate_fragment(
        self,
        index: int,
        instruction: Optional[str],
        *,
        model: str,
        provider: str,
    ) -> tuple[str, Optional[list]]:
        """Generate one candidate, retrying once when it exceeds box capacity."""
        from src.chains.translation_chain import create_translation_chain, translate_with_progress
        from src.utils.glossary_loader import GlossaryLoader
        from src.utils.helpers import chunk_paragraphs

        info = self._source_paragraphs[index]
        budget = self.length_budget(index)

        def translate_once(strict: bool) -> str:
            extra: List[str] = []
            if self.instructions:
                extra.append(self.instructions.strip())
            if instruction:
                extra.append(instruction.strip())
            if budget is not None:
                qualifier = "반드시" if strict else ""
                extra.append(
                    f"이 텍스트는 슬라이드 박스에 들어가야 합니다. 번역은 {qualifier} "
                    f"최대 {budget}자 이내로 간결하게 작성하세요."
                )
            chain = create_translation_chain(
                model_name=model,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                context=self.context,
                instructions="\n".join(f"- {item}" for item in extra) or None,
                provider=provider,
                team_rules=self.team_rules,
            )
            batches = chunk_paragraphs(
                [info], batch_size=1, ppt_context=self.ppt_context,
                glossary_terms=self.glossary_terms,
            )
            results = translate_with_progress(chain, batches, None, max_concurrency=1)
            if not results:
                raise RuntimeError("re-translation returned no result")
            result = results[0]
            if self.glossary:
                result = GlossaryLoader.apply_glossary_to_translation(result, self.glossary)
            return result

        new_target = translate_once(False)
        if budget is not None and len(new_target) > budget:
            new_target = translate_once(True)
        color_segments = self._map_color_distribution(
            index, new_target, model=model, provider=provider
        )
        return new_target, color_segments

    @staticmethod
    def _changed_phrase(old: str, new: str) -> tuple[str, str]:
        prefix = 0
        limit = min(len(old), len(new))
        while prefix < limit and old[prefix] == new[prefix]:
            prefix += 1
        suffix = 0
        while (
            suffix < len(old) - prefix
            and suffix < len(new) - prefix
            and old[len(old) - 1 - suffix] == new[len(new) - 1 - suffix]
        ):
            suffix += 1
        old_end = len(old) - suffix if suffix else len(old)
        new_end = len(new) - suffix if suffix else len(new)
        return old[prefix:old_end].strip(), new[prefix:new_end].strip()

    def partial_match_candidates(
        self, index: int, old_phrase: str, new_phrase: Optional[str] = None
    ) -> List[dict]:
        results: List[dict] = []
        needle = old_phrase.strip()
        if not needle:
            return results
        for other, target in enumerate(self.translated_texts):
            if other == index or needle not in target:
                continue
            info = self.paragraphs[other]
            proposed = target.replace(needle, new_phrase or needle, 1)
            results.append(
                {
                    "index": other,
                    "slide": info.slide_index + 1,
                    "is_note": info.is_note,
                    "target": target,
                    "proposed_target": proposed,
                    "old_phrase": needle,
                    "new_phrase": new_phrase or needle,
                }
            )
        return results

    def create_proposal(
        self,
        index: int,
        *,
        action: str,
        target: Optional[str],
        instruction: Optional[str],
        propagate_identical: bool,
        model: str,
        provider: str,
    ) -> ProposedEdit:
        old_target = self.translated_texts[index]
        primary_segments: Optional[list] = None
        if action == "retranslate":
            new_target, primary_segments = self.retranslate_fragment(
                index, instruction, model=model, provider=provider
            )
        else:
            if target is None:
                raise ValueError("edit proposal requires target")
            new_target = target

        indices = self.identical_indices(index) if propagate_identical else [index]
        colors: Dict[int, list] = {}
        for other in indices:
            if new_target == self.translated_texts[other]:
                existing = self.color_distributions.get(other)
                if existing:
                    colors[other] = list(existing)
                continue
            mapped = primary_segments if other == index else None
            if mapped is None:
                mapped = self._map_color_distribution(
                    other, new_target, model=model, provider=provider
                )
            if mapped:
                colors[other] = mapped

        old_phrase, new_phrase = self._changed_phrase(old_target, new_target)
        partial = self.partial_match_candidates(index, old_phrase, new_phrase)
        style_segments, style_status = self.style_preview(
            index, target=new_target, segments=colors.get(index, [])
        )
        budget = self.length_budget(index)
        proposal = ProposedEdit(
            id=uuid.uuid4().hex,
            index=index,
            base_revision=self.revision,
            old_target=old_target,
            target=new_target,
            changed_indices=[
                other for other in indices
                if self.translated_texts[other] != new_target
            ],
            propagate_identical=propagate_identical,
            color_distributions=colors,
            style_segments=style_segments,
            style_status=style_status,
            partial_candidates=partial,
            over_budget=budget is not None and len(new_target) > budget,
        )
        if len(self._proposals) >= 50:
            self._proposals.pop(next(iter(self._proposals)))
        self._proposals[proposal.id] = proposal
        return proposal

    def apply_proposal(self, proposal_id: str, expected_revision: int) -> ProposedEdit:
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise KeyError("proposal not found")
        if expected_revision != self.revision or proposal.base_revision != self.revision:
            raise RuntimeError("review revision conflict")
        changed = self.apply_edit(
            proposal.index,
            proposal.target,
            propagate_identical=proposal.propagate_identical,
            color_segments_by_index=proposal.color_distributions,
        )
        proposal.changed_indices = changed
        self._proposals.pop(proposal_id, None)
        return proposal

    def apply_partial_candidates(
        self,
        indices: List[int],
        *,
        old_phrase: str,
        new_phrase: str,
        expected_revision: int,
        model: str,
        provider: str,
    ) -> List[int]:
        if expected_revision != self.revision:
            raise RuntimeError("review revision conflict")
        valid = [
            idx for idx in indices
            if 0 <= idx < len(self.translated_texts)
            and old_phrase in self.translated_texts[idx]
        ]
        if not valid:
            return []
        self._history.append(self._snapshot(valid))
        for idx in valid:
            target = self.translated_texts[idx].replace(old_phrase, new_phrase, 1)
            self.translated_texts[idx] = target
            self.edited_indices.add(idx)
            mapped = self._map_color_distribution(idx, target, model=model, provider=provider)
            if mapped:
                self.color_distributions[idx] = mapped
            else:
                self.color_distributions.pop(idx, None)
        self.revision += 1
        return valid

    def undo(self, expected_revision: int) -> List[int]:
        if expected_revision != self.revision:
            raise RuntimeError("review revision conflict")
        if not self._history:
            return []
        snapshot = self._history.pop()
        for idx, text in snapshot.texts.items():
            self.translated_texts[idx] = text
            color = snapshot.colors[idx]
            if color:
                self.color_distributions[idx] = color
            else:
                self.color_distributions.pop(idx, None)
        self.edited_indices = snapshot.edited_indices
        self.revision += 1
        return sorted(snapshot.texts)

    def dismiss_finding(self, index: int, finding_type: str) -> None:
        self.dismissed_findings.add(self._finding_key(index, finding_type))

    def replace_findings(self, findings: List[Finding]) -> None:
        self.findings = findings

    def run_final_sweep(self) -> List[Finding]:
        """Recompute review findings against the current draft."""
        overflows: List[Finding] = []
        ordinal = 1
        for index, target in enumerate(self.translated_texts):
            budget = self.length_budget(index)
            if budget is None or len(target) <= budget:
                continue
            info = self.paragraphs[index]
            overflows.append(
                Finding(
                    type="fit.overflow",
                    severity="major",
                    description=(
                        f"번역 {len(target)}자가 예상 박스 용량 {budget}자를 초과합니다."
                    ),
                    location={
                        "slide": info.slide_index + 1,
                        "shape": info.shape_index,
                        "paragraph": info.paragraph_index,
                    },
                    segment={"source": info.original_text, "output": target},
                    ordinal=ordinal,
                    fragment_index=index,
                )
            )
            ordinal += 1
        findings = run_sweep(
            self.paragraphs,
            self.translated_texts,
            glossary=self.glossary,
            locked_terms=self.locked_terms,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            fit_overflows=overflows,
        )
        self.replace_findings(findings)
        return findings

    def render(self) -> io.BytesIO:
        """Render the current draft from pristine source bytes."""
        assert self.source_pptx is not None
        paragraphs, presentation = PPTParser().extract_paragraphs(
            io.BytesIO(self.source_pptx),
            translate_notes=any(info.is_note for info in self.paragraphs),
        )
        if len(paragraphs) != len(self.translated_texts):
            raise RuntimeError("source/draft paragraph count mismatch")
        return PPTWriter().apply_translations(
            paragraphs,
            self.translated_texts,
            presentation,
            text_fit_mode=self.text_fit_mode,
            min_font_ratio=self.min_font_ratio,
            color_distributions=self.color_distributions or None,
        )

    def mark_committed(self) -> None:
        self.committed_revision = self.revision
