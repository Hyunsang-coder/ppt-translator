"""Unit tests for the deterministic consistency sweep (WP-C3)."""

from __future__ import annotations

import unittest

from src.core.ppt_parser import ParagraphInfo
from src.services.consistency_sweep import run_sweep


def _p(slide: int, shape: int, para: int, src: str, is_note: bool = False) -> ParagraphInfo:
    return ParagraphInfo(
        slide_index=slide,
        shape_index=shape,
        paragraph_index=para,
        original_text=src,
        paragraph=None,
        slide_title=None,
        is_note=is_note,
    )


class PhraseDivergenceTestCase(unittest.TestCase):
    def test_same_source_two_translations_detected_with_slide_numbers(self) -> None:
        paras = [
            _p(6, 0, 0, "보급 상자 등장 주기 단축"),
            _p(11, 0, 0, "보급 상자 등장 주기 단축"),
        ]
        targets = [
            "Shorter Care Package spawn interval",
            "Shorter Supply Crate spawn interval",
        ]
        findings = run_sweep(paras, targets, source_lang="한국어", target_lang="영어")
        divergence = [f for f in findings if f.type == "consistency.phrase"]
        self.assertEqual(len(divergence), 1)
        f = divergence[0]
        # Both slide numbers (1-based) appear in the description.
        self.assertIn("7", f.description)
        self.assertIn("12", f.description)
        # Location is 1-based.
        self.assertEqual(f.location["slide"], 12)

    def test_consistent_translations_no_finding(self) -> None:
        paras = [_p(0, 0, 0, "보급 상자"), _p(1, 0, 0, "보급 상자")]
        targets = ["Care Package", "Care Package"]
        findings = run_sweep(paras, targets, source_lang="한국어", target_lang="영어")
        self.assertEqual([f for f in findings if f.type == "consistency.phrase"], [])


class GlossaryViolationTestCase(unittest.TestCase):
    def test_missing_glossary_term_detected(self) -> None:
        paras = [_p(2, 1, 0, "피격 시 저지력 효과를 20% 상향")]
        targets = ["Increased stopping power effect on hit by 20%"]
        findings = run_sweep(
            paras, targets, glossary={"저지력": "Aim Punch"},
            source_lang="한국어", target_lang="영어",
        )
        violations = [f for f in findings if f.type == "terminology.violation"]
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].suggested_fix, "Aim Punch")

    def test_applied_glossary_term_no_finding(self) -> None:
        paras = [_p(2, 1, 0, "저지력 상향")]
        targets = ["Increased Aim Punch"]
        findings = run_sweep(
            paras, targets, glossary={"저지력": "Aim Punch"},
            source_lang="한국어", target_lang="영어",
        )
        self.assertEqual([f for f in findings if f.type == "terminology.violation"], [])


class UntranslatedTestCase(unittest.TestCase):
    def test_identical_output_detected_as_omission(self) -> None:
        paras = [_p(11, 3, 0, "사내 공유용, 외부 배포 금지")]
        targets = ["사내 공유용, 외부 배포 금지"]
        findings = run_sweep(paras, targets, source_lang="한국어", target_lang="영어")
        omissions = [f for f in findings if f.type == "accuracy.omission"]
        self.assertEqual(len(omissions), 1)
        self.assertEqual(omissions[0].severity, "critical")


class CleanDeckTestCase(unittest.TestCase):
    def test_clean_deck_zero_findings(self) -> None:
        paras = [_p(0, 0, 0, "안녕하세요"), _p(1, 0, 0, "반갑습니다")]
        targets = ["Hello", "Nice to meet you"]
        findings = run_sweep(paras, targets, source_lang="한국어", target_lang="영어")
        self.assertEqual(findings, [])


class OrdinalUniquenessTestCase(unittest.TestCase):
    def test_ordinals_unique_across_findings(self) -> None:
        # Two table cells share (slide, shape, paragraph); ordinals must differ.
        paras = [
            _p(0, 5, 0, "저지력"),
            _p(0, 5, 0, "저지력"),
        ]
        targets = ["stopping power", "stopping power"]
        findings = run_sweep(
            paras, targets, glossary={"저지력": "Aim Punch"},
            source_lang="한국어", target_lang="영어",
        )
        ordinals = [f.ordinal for f in findings]
        self.assertEqual(len(ordinals), len(set(ordinals)))


if __name__ == "__main__":
    unittest.main()
