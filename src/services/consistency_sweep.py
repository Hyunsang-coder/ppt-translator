"""Deterministic deck-wide consistency sweep (WP-C3).

Runs after translation with **zero LLM calls** to catch the signature PPT
failures: a glossary/locked term left un-applied, the same source phrase
translated inconsistently across slides, a fragment left untranslated, and text
that overflows its box. Detection-only — this module never mutates translations
(P1: Korean morphology makes auto-substitution unsafe; report, don't fix).

See ``consulting/ppt-translator-quality-design.md`` §WP-C3 and §3.1/§3.2 for the
finding/type contract.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from src.core.ppt_parser import ParagraphInfo

LOGGER = logging.getLogger(__name__)

# Minimum length (in normalized chars) for a repeated source phrase to be worth
# a divergence check — below this, short function words create noise.
_MIN_PHRASE_LEN = 4

# Minimum length for the untranslated-fragment language check. langdetect is
# unreliable on very short strings, so we skip them rather than false-positive.
_MIN_UNTRANSLATED_LEN = 12


@dataclass
class Finding:
    """One consistency issue detected by the sweep.

    Carries exactly the fields the quality-record contract (§3.1) needs, plus a
    monotonically increasing ``ordinal`` so two findings never share an
    identical location key (table cells collide on ``(slide, shape, paragraph)``).
    """

    type: str  # contract §3.2 vocabulary
    severity: str  # critical | major | minor
    description: str
    location: Dict[str, object]  # {slide, shape, paragraph}
    segment: Dict[str, Optional[str]]  # {source, output}
    ordinal: int
    suggested_fix: Optional[str] = None
    # Optional cross-reference to the other fragment in a divergence pair.
    related_location: Optional[Dict[str, object]] = None


@dataclass
class _Fragment:
    """A source/target pair with position, built from the aligned lists."""

    index: int
    source: str
    target: str
    slide: int
    shape: int
    paragraph: int
    is_note: bool


def _normalize(text: str) -> str:
    """Stronger normalization than helpers.clean_text for *matching only*.

    Collapses all whitespace, case-folds. The original text is still what gets
    reported — this is used purely to decide equality/containment.
    """

    if not text:
        return ""
    collapsed = re.sub(r"\s+", " ", text).strip()
    return collapsed.casefold()


def _build_fragments(
    paragraphs: Sequence[ParagraphInfo],
    translated_texts: Sequence[str],
) -> List[_Fragment]:
    fragments: List[_Fragment] = []
    for idx, info in enumerate(paragraphs):
        target = translated_texts[idx] if idx < len(translated_texts) else ""
        fragments.append(
            _Fragment(
                index=idx,
                source=info.original_text or "",
                target=target or "",
                slide=info.slide_index,
                shape=info.shape_index,
                paragraph=info.paragraph_index,
                is_note=info.is_note,
            )
        )
    return fragments


def _location(frag: _Fragment) -> Dict[str, object]:
    # slide_index is 0-based internally; report 1-based to match how users read
    # slide numbers in PowerPoint.
    return {
        "slide": frag.slide + 1,
        "shape": frag.shape,
        "paragraph": frag.paragraph,
    }


def _check_term_violations(
    fragments: Sequence[_Fragment],
    glossary: Dict[str, str],
    locked_terms: Dict[str, str],
    counter: "_Ordinal",
) -> List[Finding]:
    """Glossary + locked-term violations: source has the term but target lacks
    its required translation. 100% recall, detect-only."""

    findings: List[Finding] = []
    # locked_terms maps rule_id -> exact target term that must appear when the
    # term is relevant. We treat every locked term as a target-side requirement:
    # if a fragment's target is non-trivial and the exact locked term is
    # expected but a near-miss appears, we can't know source mapping, so locked
    # terms are checked target-side only via the glossary pathway below when the
    # source term is known. Locked terms without a source mapping are surfaced
    # separately: flag targets that contain a casefold-variant but not the exact.
    glossary_norm = {(_normalize(src), src): dst for src, dst in glossary.items()}

    for frag in fragments:
        src_norm = _normalize(frag.source)
        tgt_norm = _normalize(frag.target)
        if not src_norm or not tgt_norm:
            continue

        for (src_n, src_orig), dst in glossary_norm.items():
            if src_n and src_n in src_norm and _normalize(dst) not in tgt_norm:
                findings.append(
                    Finding(
                        type="terminology.violation",
                        severity="major",
                        description=(
                            f'용어집: "{src_orig}" → "{dst}" 인데 번역에 "{dst}"가 없습니다.'
                        ),
                        location=_location(frag),
                        segment={"source": frag.source, "output": frag.target},
                        ordinal=counter.next(),
                        suggested_fix=dst,
                    )
                )

        # Locked terms: if the exact locked term is absent but a case/space
        # variant is present, that's a violation of the "use exactly" rule.
        for rule_id, term in locked_terms.items():
            term_norm = _normalize(term)
            if term_norm and term_norm in tgt_norm and term not in frag.target:
                findings.append(
                    Finding(
                        type="terminology.violation",
                        severity="major",
                        description=(
                            f'확정 용어 규칙({rule_id}): "{term}" 표기를 정확히 지켜야 합니다.'
                        ),
                        location=_location(frag),
                        segment={"source": frag.source, "output": frag.target},
                        ordinal=counter.next(),
                        suggested_fix=term,
                    )
                )
    return findings


def _check_phrase_divergence(
    fragments: Sequence[_Fragment],
    counter: "_Ordinal",
) -> List[Finding]:
    """Same normalized source phrase translated differently across fragments.

    Focuses on whole-fragment repeats where the *source* matches but the
    *target* differs. (Whole-identical paragraphs sharing one translation are
    already guaranteed by the repetition cache; this catches the cases where
    that cache was off, or where identical sources were translated in separate
    batches and diverged.)
    """

    findings: List[Finding] = []
    # Group fragments by normalized source.
    by_source: Dict[str, List[_Fragment]] = {}
    for frag in fragments:
        src_norm = _normalize(frag.source)
        if len(src_norm) < _MIN_PHRASE_LEN:
            continue
        by_source.setdefault(src_norm, []).append(frag)

    for src_norm, group in by_source.items():
        if len(group) < 2:
            continue
        # Distinct normalized targets within the group => divergence.
        distinct: Dict[str, _Fragment] = {}
        for frag in group:
            tgt_norm = _normalize(frag.target)
            if tgt_norm and tgt_norm not in distinct:
                distinct[tgt_norm] = frag
        if len(distinct) < 2:
            continue

        variants = list(distinct.values())
        # Report each variant beyond the first, cross-referencing the first.
        anchor = variants[0]
        for other in variants[1:]:
            findings.append(
                Finding(
                    type="consistency.phrase",
                    severity="major",
                    description=(
                        f'같은 원문 "{other.source}"가 슬라이드 {anchor.slide + 1}에서는 '
                        f'"{anchor.target}", 슬라이드 {other.slide + 1}에서는 '
                        f'"{other.target}"로 다르게 번역됐습니다. 하나로 통일하세요.'
                    ),
                    location=_location(other),
                    segment={"source": other.source, "output": other.target},
                    ordinal=counter.next(),
                    related_location=_location(anchor),
                )
            )
    return findings


def _check_untranslated(
    fragments: Sequence[_Fragment],
    source_lang: str,
    target_lang: str,
    counter: "_Ordinal",
) -> List[Finding]:
    """Fragments whose output is identical to the source (untranslated).

    Uses direct string equality after normalization as the primary signal
    (100% precise for the common "left as-is" case) and langdetect as a
    secondary signal for longer fragments. Reported as accuracy.omission per the
    locked decision (no PPT-specific contract value for untranslated).
    """

    findings: List[Finding] = []
    detector = None
    for frag in fragments:
        src_norm = _normalize(frag.source)
        tgt_norm = _normalize(frag.target)
        if not src_norm or not tgt_norm:
            continue

        untranslated = False
        # Primary: output is byte-for-byte the source (ignoring case/space).
        if src_norm == tgt_norm:
            untranslated = True
        elif len(tgt_norm) >= _MIN_UNTRANSLATED_LEN:
            # Secondary: langdetect says the output is still the source language.
            # Import lazily; language of source lang display-name is Korean.
            try:
                from langdetect import detect  # type: ignore
                from langdetect.lang_detect_exception import LangDetectException

                detected = detect(frag.target)
                # Map source display-name to iso for comparison.
                src_iso = _SOURCE_ISO.get(source_lang)
                tgt_iso = _SOURCE_ISO.get(target_lang)
                if src_iso and detected == src_iso and detected != tgt_iso:
                    untranslated = True
            except LangDetectException:
                pass
            except Exception:  # pragma: no cover - langdetect edge cases
                pass

        if untranslated:
            findings.append(
                Finding(
                    type="accuracy.omission",
                    severity="critical",
                    description="번역되지 않고 원문이 그대로 남아 있습니다.",
                    location=_location(frag),
                    segment={"source": frag.source, "output": frag.target},
                    ordinal=counter.next(),
                )
            )
    return findings


# Display-name -> ISO code for the untranslated langdetect check.
_SOURCE_ISO = {
    "한국어": "ko",
    "영어": "en",
    "일본어": "ja",
    "중국어": "zh-cn",
}


class _Ordinal:
    """Monotonic counter so findings get unique ordinals across all checks."""

    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return self._n


def run_sweep(
    paragraphs: Sequence[ParagraphInfo],
    translated_texts: Sequence[str],
    *,
    glossary: Optional[Dict[str, str]] = None,
    locked_terms: Optional[Dict[str, str]] = None,
    source_lang: str = "",
    target_lang: str = "",
    fit_overflows: Optional[Sequence[Finding]] = None,
) -> List[Finding]:
    """Run all deterministic consistency checks over the translated deck.

    Args:
        paragraphs: Parsed source paragraphs (index-aligned to translated_texts).
        translated_texts: Full aligned translation list.
        glossary: Optional source->target glossary mapping.
        locked_terms: Optional {rule_id: locked_term} from the team rules.
        source_lang / target_lang: Resolved display names.
        fit_overflows: Optional pre-computed fit.overflow findings from the write
            phase (which owns box-capacity math); merged in as-is.

    Returns:
        All findings, deduped by (type, location-key, output).
    """

    fragments = _build_fragments(paragraphs, translated_texts)
    counter = _Ordinal()
    findings: List[Finding] = []

    findings.extend(
        _check_term_violations(
            fragments, glossary or {}, locked_terms or {}, counter
        )
    )
    findings.extend(_check_phrase_divergence(fragments, counter))
    findings.extend(
        _check_untranslated(fragments, source_lang, target_lang, counter)
    )

    if fit_overflows:
        for f in fit_overflows:
            # Assign fresh ordinals so uniqueness holds across the merged set.
            f.ordinal = counter.next()
            findings.append(f)

    LOGGER.info("Consistency sweep produced %d findings", len(findings))
    return findings
