"""Tests for multi-color paragraph handling and color distribution."""

from __future__ import annotations

import types
import unittest
from copy import deepcopy
from unittest.mock import MagicMock, patch

from lxml import etree

from src.chains.color_distribution_chain import _format_items
from src.core.ppt_writer import _group_runs_by_format, _rpr_key


def _make_solid_fill(color_val: str) -> etree._Element:
    """Create a solidFill child element for rPr.

    Args:
        color_val: Hex color value, e.g. "FF0000" for red.
    """
    fill = etree.Element("solidFill")
    clr = etree.SubElement(fill, "srgbClr")
    clr.set("val", color_val)
    return fill


def _make_run(
    text: str,
    rpr_attrs: dict | None = None,
    rpr_children: list | None = None,
) -> MagicMock:
    """Create a mock run with optional rPr attributes and child elements.

    Args:
        text: The run text.
        rpr_attrs: Dict of XML attributes for the rPr element, or None for no rPr.
        rpr_children: List of lxml child elements to append to rPr.
    """
    run = MagicMock()
    run.text = text

    # Build a minimal lxml _r element with optional rPr
    r_elem = MagicMock()
    if rpr_attrs is not None or rpr_children is not None:
        rPr = etree.Element("rPr")
        if rpr_attrs:
            for k, v in rpr_attrs.items():
                rPr.set(k, v)
        if rpr_children:
            for child in rpr_children:
                rPr.append(child)
        r_elem.rPr = rPr
    else:
        r_elem.rPr = None

    run._r = r_elem
    return run


class RprKeyTestCase(unittest.TestCase):
    def test_none_rpr_returns_empty(self) -> None:
        run = _make_run("hello")
        self.assertEqual(_rpr_key(run), "")

    def test_same_format_same_key(self) -> None:
        run1 = _make_run("a", {"color": "red"})
        run2 = _make_run("b", {"color": "red"})
        self.assertEqual(_rpr_key(run1), _rpr_key(run2))

    def test_different_format_different_key(self) -> None:
        run1 = _make_run("a", {"color": "red"})
        run2 = _make_run("b", {"color": "blue"})
        self.assertNotEqual(_rpr_key(run1), _rpr_key(run2))


class RprKeyNonVisualExclusionTestCase(unittest.TestCase):
    """_rpr_key should ignore non-visual attributes (lang, dirty, etc.)."""

    def test_same_color_different_lang_same_key(self) -> None:
        """Runs with same color but different lang should produce the same key."""
        run1 = _make_run(
            "a",
            rpr_attrs={"lang": "en-US"},
            rpr_children=[_make_solid_fill("FF0000")],
        )
        run2 = _make_run(
            "b",
            rpr_attrs={"lang": "ko-KR"},
            rpr_children=[_make_solid_fill("FF0000")],
        )
        self.assertEqual(_rpr_key(run1), _rpr_key(run2))

    def test_same_color_different_dirty_same_key(self) -> None:
        """Runs with same color but different dirty flag should produce the same key."""
        run1 = _make_run(
            "a",
            rpr_attrs={"dirty": "0"},
            rpr_children=[_make_solid_fill("FF0000")],
        )
        run2 = _make_run(
            "b",
            rpr_attrs={"dirty": "1"},
            rpr_children=[_make_solid_fill("FF0000")],
        )
        self.assertEqual(_rpr_key(run1), _rpr_key(run2))

    def test_same_color_different_err_same_key(self) -> None:
        """Runs with same color but different err flag should produce the same key."""
        run1 = _make_run(
            "a",
            rpr_attrs={"err": "1"},
            rpr_children=[_make_solid_fill("FF0000")],
        )
        run2 = _make_run(
            "b",
            rpr_children=[_make_solid_fill("FF0000")],
        )
        self.assertEqual(_rpr_key(run1), _rpr_key(run2))

    def test_same_color_multiple_nonvisual_attrs_same_key(self) -> None:
        """Runs differing in multiple non-visual attrs should still match."""
        run1 = _make_run(
            "a",
            rpr_attrs={"lang": "en-US", "dirty": "0", "noProof": "1"},
            rpr_children=[_make_solid_fill("FF0000")],
        )
        run2 = _make_run(
            "b",
            rpr_attrs={"lang": "ko-KR", "dirty": "1", "smtClean": "0"},
            rpr_children=[_make_solid_fill("FF0000")],
        )
        self.assertEqual(_rpr_key(run1), _rpr_key(run2))

    def test_different_color_different_key(self) -> None:
        """Runs with different colors should produce different keys."""
        run1 = _make_run("a", rpr_children=[_make_solid_fill("FF0000")])
        run2 = _make_run("b", rpr_children=[_make_solid_fill("0000FF")])
        self.assertNotEqual(_rpr_key(run1), _rpr_key(run2))

    def test_same_color_different_bold_different_key(self) -> None:
        """Runs with same color but different bold should produce different keys."""
        run1 = _make_run(
            "a",
            rpr_attrs={"b": "1"},
            rpr_children=[_make_solid_fill("FF0000")],
        )
        run2 = _make_run(
            "b",
            rpr_attrs={"b": "0"},
            rpr_children=[_make_solid_fill("FF0000")],
        )
        self.assertNotEqual(_rpr_key(run1), _rpr_key(run2))

    def test_grouping_merges_lang_only_difference(self) -> None:
        """_group_runs_by_format should merge runs differing only in lang."""
        runs = [
            _make_run(
                "Hello ",
                rpr_attrs={"lang": "en-US"},
                rpr_children=[_make_solid_fill("FF0000")],
            ),
            _make_run(
                "세계",
                rpr_attrs={"lang": "ko-KR"},
                rpr_children=[_make_solid_fill("FF0000")],
            ),
        ]
        groups = _group_runs_by_format(runs)
        self.assertEqual(len(groups), 1)  # should be 1 group, not 2


class GroupRunsByFormatTestCase(unittest.TestCase):
    def test_empty_runs(self) -> None:
        self.assertEqual(_group_runs_by_format([]), [])

    def test_single_run(self) -> None:
        run = _make_run("hello", {"color": "red"})
        groups = _group_runs_by_format([run])
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 1)

    def test_uniform_format_single_group(self) -> None:
        runs = [_make_run("a", {"color": "red"}), _make_run("b", {"color": "red"})]
        groups = _group_runs_by_format(runs)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)

    def test_different_formats_multiple_groups(self) -> None:
        runs = [
            _make_run("Important ", {"color": "red"}),
            _make_run("notice", {"color": "blue"}),
        ]
        groups = _group_runs_by_format(runs)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0][0].text, "Important ")
        self.assertEqual(groups[1][0].text, "notice")

    def test_adjacent_same_format_merged(self) -> None:
        runs = [
            _make_run("a", {"color": "red"}),
            _make_run("b", {"color": "red"}),
            _make_run("c", {"color": "blue"}),
            _make_run("d", {"color": "blue"}),
        ]
        groups = _group_runs_by_format(runs)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]), 2)
        self.assertEqual(len(groups[1]), 2)

    def test_alternating_formats(self) -> None:
        runs = [
            _make_run("a", {"color": "red"}),
            _make_run("b", {"color": "blue"}),
            _make_run("c", {"color": "red"}),
        ]
        groups = _group_runs_by_format(runs)
        self.assertEqual(len(groups), 3)

    def test_none_rpr_grouped_together(self) -> None:
        runs = [_make_run("a"), _make_run("b")]
        groups = _group_runs_by_format(runs)
        self.assertEqual(len(groups), 1)


class FormatItemsTestCase(unittest.TestCase):
    def test_single_item(self) -> None:
        result = _format_items([["Important ", "notice"]], ["중요 공지"])
        self.assertIn("1.", result)
        self.assertIn('"Important "', result)
        self.assertIn('"notice"', result)
        self.assertIn("중요 공지", result)
        self.assertIn("2개", result)

    def test_multiple_items(self) -> None:
        result = _format_items(
            [["Click ", "here"], ["A", "B", "C"]],
            ["여기를 클릭", "가나다"],
        )
        self.assertIn("1.", result)
        self.assertIn("2.", result)
        self.assertIn("2개", result)
        self.assertIn("3개", result)


class ValidateDistributionTestCase(unittest.TestCase):
    """Tests for _validate_distribution helper."""

    def test_exact_match_passes(self) -> None:
        """Segments that concatenate exactly to translation should pass."""
        from src.services.translation_service import _validate_distribution

        result = _validate_distribution(
            segments=["매출이 ", "20% 증가했습니다"],
            translation="매출이 20% 증가했습니다",
            num_groups=2,
            para_idx=0,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result, ["매출이 ", "20% 증가했습니다"])

    def test_trailing_space_difference_passes(self) -> None:
        """Segments with trailing whitespace difference should pass after correction."""
        from src.services.translation_service import _validate_distribution

        result = _validate_distribution(
            segments=["매출이 ", "20% 증가했습니다 "],
            translation="매출이 20% 증가했습니다",
            num_groups=2,
            para_idx=0,
        )
        self.assertIsNotNone(result)
        self.assertEqual("".join(result), "매출이 20% 증가했습니다")

    def test_leading_space_difference_passes(self) -> None:
        """Segments with leading whitespace difference should pass after correction."""
        from src.services.translation_service import _validate_distribution

        result = _validate_distribution(
            segments=[" 매출이 ", "20% 증가했습니다"],
            translation="매출이 20% 증가했습니다",
            num_groups=2,
            para_idx=0,
        )
        self.assertIsNotNone(result)
        self.assertEqual("".join(result), "매출이 20% 증가했습니다")

    def test_nbsp_treated_as_space(self) -> None:
        """NBSP (\\u00a0) should be normalized to regular space."""
        from src.services.translation_service import _validate_distribution

        result = _validate_distribution(
            segments=["매출이\u00a0", "20% 증가했습니다"],
            translation="매출이 20% 증가했습니다",
            num_groups=2,
            para_idx=0,
        )
        self.assertIsNotNone(result)
        self.assertEqual("".join(result), "매출이 20% 증가했습니다")

    def test_completely_different_text_fails(self) -> None:
        """Segments that don't match the translation at all should fail."""
        from src.services.translation_service import _validate_distribution

        result = _validate_distribution(
            segments=["전혀 ", "다른 텍스트"],
            translation="매출이 20% 증가했습니다",
            num_groups=2,
            para_idx=0,
        )
        self.assertIsNone(result)

    def test_segment_count_mismatch_fails(self) -> None:
        """Wrong number of segments should fail."""
        from src.services.translation_service import _validate_distribution

        result = _validate_distribution(
            segments=["매출이 ", "20% ", "증가했습니다"],
            translation="매출이 20% 증가했습니다",
            num_groups=2,
            para_idx=0,
        )
        self.assertIsNone(result)


class ColoredSegmentSchemaTestCase(unittest.TestCase):
    """Tests for the ColoredSegment / ColorDistributionOutput schema."""

    def test_colored_segment_model_exists(self) -> None:
        """ColoredSegment model should be importable."""
        from src.chains.color_distribution_chain import ColoredSegment

        seg = ColoredSegment(text="매출이", group_index=0)
        self.assertEqual(seg.text, "매출이")
        self.assertEqual(seg.group_index, 0)

    def test_distribution_output_with_colored_segments(self) -> None:
        """ColorDistributionOutput should accept list of list of ColoredSegment."""
        from src.chains.color_distribution_chain import (
            ColorDistributionOutput,
            ColoredSegment,
        )

        output = ColorDistributionOutput(
            distributions=[
                [
                    ColoredSegment(text="매출이 ", group_index=0),
                    ColoredSegment(text="20% 증가했습니다", group_index=1),
                ],
            ]
        )
        self.assertEqual(len(output.distributions), 1)
        self.assertEqual(len(output.distributions[0]), 2)
        self.assertEqual(output.distributions[0][0].group_index, 0)

    def test_format_items_includes_group_indices(self) -> None:
        """_format_items should include group index info in output."""
        from src.chains.color_distribution_chain import _format_items

        result = _format_items(
            [["Revenue", " increased by 20%"]],
            ["매출이 20% 증가했습니다"],
        )
        # Should include group index markers [0] and [1]
        self.assertIn("[0]", result)
        self.assertIn("[1]", result)


class ValidateColoredSegmentsTestCase(unittest.TestCase):
    """Tests for _validate_colored_segments (group_index-based validation)."""

    def test_ordered_segments_pass(self) -> None:
        """Segments in group order should pass."""
        from src.chains.color_distribution_chain import ColoredSegment
        from src.services.translation_service import _validate_colored_segments

        segments = [
            ColoredSegment(text="매출이 ", group_index=0),
            ColoredSegment(text="20% 증가했습니다", group_index=1),
        ]
        result = _validate_colored_segments(
            segments=segments,
            translation="매출이 20% 증가했습니다",
            num_groups=2,
            para_idx=0,
        )
        self.assertIsNotNone(result)

    def test_reordered_segments_pass(self) -> None:
        """Segments with reordered group_index should pass (non-contiguous)."""
        from src.chains.color_distribution_chain import ColoredSegment
        from src.services.translation_service import _validate_colored_segments

        segments = [
            ColoredSegment(text="이 접근 방식을 ", group_index=1),
            ColoredSegment(text="강력히", group_index=0),
            ColoredSegment(text=" 추천합니다", group_index=1),
        ]
        result = _validate_colored_segments(
            segments=segments,
            translation="이 접근 방식을 강력히 추천합니다",
            num_groups=2,
            para_idx=0,
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)

    def test_group_index_out_of_range_fails(self) -> None:
        """Segments with group_index >= num_groups should fail."""
        from src.chains.color_distribution_chain import ColoredSegment
        from src.services.translation_service import _validate_colored_segments

        segments = [
            ColoredSegment(text="매출이 ", group_index=0),
            ColoredSegment(text="증가", group_index=5),  # out of range
        ]
        result = _validate_colored_segments(
            segments=segments,
            translation="매출이 증가",
            num_groups=2,
            para_idx=0,
        )
        self.assertIsNone(result)

    def test_concatenation_mismatch_fails(self) -> None:
        """Segments whose concatenation doesn't match translation should fail."""
        from src.chains.color_distribution_chain import ColoredSegment
        from src.services.translation_service import _validate_colored_segments

        segments = [
            ColoredSegment(text="완전히 ", group_index=0),
            ColoredSegment(text="다른 텍스트", group_index=1),
        ]
        result = _validate_colored_segments(
            segments=segments,
            translation="매출이 20% 증가했습니다",
            num_groups=2,
            para_idx=0,
        )
        self.assertIsNone(result)

    def test_empty_segments_allowed(self) -> None:
        """Empty text segments should be allowed."""
        from src.chains.color_distribution_chain import ColoredSegment
        from src.services.translation_service import _validate_colored_segments

        segments = [
            ColoredSegment(text="전체 텍스트", group_index=0),
            ColoredSegment(text="", group_index=1),
        ]
        result = _validate_colored_segments(
            segments=segments,
            translation="전체 텍스트",
            num_groups=2,
            para_idx=0,
        )
        self.assertIsNotNone(result)


class ApplyTranslationsGroupTestCase(unittest.TestCase):
    """Test PPTWriter.apply_translations with group-based distribution."""

    def _make_paragraph_info(self, runs):
        """Create a mock ParagraphInfo with given runs."""
        paragraph = MagicMock()
        paragraph.runs = runs

        # Simulate _parent for text_frame tracking
        paragraph._parent = MagicMock()

        info = types.SimpleNamespace(
            paragraph=paragraph,
            original_text="".join(r.text for r in runs),
            is_note=False,
            slide_index=0,
            shape_index=0,
            paragraph_index=0,
        )
        return info

    @patch("src.core.ppt_writer._group_runs_by_format")
    def test_uniform_format_all_in_first(self, mock_group) -> None:
        """Uniform formatting: all text in first run, rest cleared."""
        run1 = MagicMock()
        run1.text = "Hello "
        run2 = MagicMock()
        run2.text = "World"
        runs = [run1, run2]

        # Single group = uniform
        mock_group.return_value = [runs]

        para_info = self._make_paragraph_info(runs)

        from src.core.ppt_writer import PPTWriter

        writer = PPTWriter()
        mock_pres = MagicMock()
        mock_pres.slides = []

        writer.apply_translations(
            [para_info], ["안녕하세요 세계"], mock_pres, text_fit_mode="none"
        )

        self.assertEqual(run1.text, "안녕하세요 세계")
        self.assertEqual(run2.text, "")

    @patch("src.core.ppt_writer._group_runs_by_format")
    def test_multicolor_with_distribution(self, mock_group) -> None:
        """Multi-color with color_distributions: text assigned to group leaders."""
        run1 = MagicMock()
        run1.text = "Important "
        run2 = MagicMock()
        run2.text = "notice"
        runs = [run1, run2]

        # Two groups = multi-color
        mock_group.return_value = [[run1], [run2]]

        para_info = self._make_paragraph_info(runs)

        from src.core.ppt_writer import PPTWriter

        writer = PPTWriter()
        mock_pres = MagicMock()
        mock_pres.slides = []

        color_distributions = {0: ["중요 ", "공지"]}

        writer.apply_translations(
            [para_info],
            ["중요 공지"],
            mock_pres,
            text_fit_mode="none",
            color_distributions=color_distributions,
        )

        self.assertEqual(run1.text, "중요 ")
        self.assertEqual(run2.text, "공지")

    @patch("src.core.ppt_writer._group_runs_by_format")
    def test_multicolor_fallback_no_distribution(self, mock_group) -> None:
        """Multi-color without distribution: falls back to ratio-based."""
        run1 = MagicMock()
        run1.text = "Important "
        run2 = MagicMock()
        run2.text = "notice"
        runs = [run1, run2]

        mock_group.return_value = [[run1], [run2]]

        para_info = self._make_paragraph_info(runs)

        from src.core.ppt_writer import PPTWriter

        writer = PPTWriter()
        mock_pres = MagicMock()
        mock_pres.slides = []

        writer.apply_translations(
            [para_info],
            ["중요 공지"],
            mock_pres,
            text_fit_mode="none",
            color_distributions=None,
        )

        # Both runs should have some text (ratio-based split)
        combined = run1.text + run2.text
        self.assertEqual(combined, "중요 공지")


class ApplyTranslationsColoredSegmentsTestCase(unittest.TestCase):
    """Test PPTWriter.apply_translations with ColoredSegment-based distributions."""

    def _make_paragraph_info(self, runs):
        """Create a mock ParagraphInfo with given runs and lxml paragraph."""
        # Build a real-ish lxml paragraph element with run children
        p_elem = etree.Element("p")
        for run in runs:
            r_elem = etree.SubElement(p_elem, "r")
            # Attach rPr if the run has a real lxml rPr
            rPr = getattr(getattr(run, "_r", None), "rPr", None)
            if rPr is not None and isinstance(rPr, etree._Element):
                r_elem.append(deepcopy(rPr))
            t_elem = etree.SubElement(r_elem, "t")
            t_elem.text = run.text

        # Use a mutable list so add_run() appended runs are visible via .runs
        all_runs = list(runs)

        paragraph = MagicMock()
        # Make .runs return the mutable list
        type(paragraph).runs = property(lambda self: all_runs)
        paragraph._parent = MagicMock()
        # Store the lxml element for run recreation tests
        paragraph._p = p_elem

        def _mock_add_run():
            new_run = MagicMock()
            new_run.text = ""
            # Give it a minimal _r with rPr=None for deepcopy compat
            r_elem = MagicMock()
            r_elem.rPr = None
            new_run._r = r_elem
            all_runs.append(new_run)
            return new_run

        paragraph.add_run = _mock_add_run

        info = types.SimpleNamespace(
            paragraph=paragraph,
            original_text="".join(r.text for r in runs),
            is_note=False,
            slide_index=0,
            shape_index=0,
            paragraph_index=0,
        )
        return info

    @patch("src.core.ppt_writer._group_runs_by_format")
    def test_colored_segments_ordered(self, mock_group) -> None:
        """ColoredSegments in order should assign text to correct groups."""
        from src.chains.color_distribution_chain import ColoredSegment
        from src.core.ppt_writer import PPTWriter

        run1 = _make_run("Important ", {"color": "red"})
        run2 = _make_run("notice", {"color": "blue"})
        runs = [run1, run2]
        mock_group.return_value = [[run1], [run2]]

        para_info = self._make_paragraph_info(runs)

        writer = PPTWriter()
        mock_pres = MagicMock()
        mock_pres.slides = []

        color_distributions = {
            0: [
                ColoredSegment(text="중요 ", group_index=0),
                ColoredSegment(text="공지", group_index=1),
            ]
        }

        writer.apply_translations(
            [para_info],
            ["중요 공지"],
            mock_pres,
            text_fit_mode="none",
            color_distributions=color_distributions,
        )

        # Verify text was applied correctly
        all_text = "".join(r.text for r in runs if r.text)
        self.assertEqual(all_text, "중요 공지")

    @patch("src.core.ppt_writer._group_runs_by_format")
    def test_colored_segments_reordered(self, mock_group) -> None:
        """ColoredSegments with reordered group_index should create correct runs."""
        from src.chains.color_distribution_chain import ColoredSegment
        from src.core.ppt_writer import PPTWriter

        run1 = _make_run("strongly ", {"color": "red"})
        run2 = _make_run("recommend this approach", {"color": "blue"})
        runs = [run1, run2]
        mock_group.return_value = [[run1], [run2]]

        para_info = self._make_paragraph_info(runs)

        writer = PPTWriter()
        mock_pres = MagicMock()
        mock_pres.slides = []

        # Korean word order differs: group 1 text comes first, then group 0, then group 1 again
        color_distributions = {
            0: [
                ColoredSegment(text="이 접근 방식을 ", group_index=1),
                ColoredSegment(text="강력히", group_index=0),
                ColoredSegment(text=" 추천합니다", group_index=1),
            ]
        }

        writer.apply_translations(
            [para_info],
            ["이 접근 방식을 강력히 추천합니다"],
            mock_pres,
            text_fit_mode="none",
            color_distributions=color_distributions,
        )

        # The combined text should equal the full translation
        all_text = "".join(r.text for r in para_info.paragraph.runs if r.text)
        self.assertEqual(all_text, "이 접근 방식을 강력히 추천합니다")

    @patch("src.core.ppt_writer._group_runs_by_format")
    def test_single_group_unchanged(self, mock_group) -> None:
        """Single-group paragraphs should still work (regression test)."""
        from src.core.ppt_writer import PPTWriter

        run1 = MagicMock()
        run1.text = "Hello "
        run2 = MagicMock()
        run2.text = "World"
        runs = [run1, run2]
        mock_group.return_value = [runs]  # single group

        para_info = self._make_paragraph_info(runs)

        writer = PPTWriter()
        mock_pres = MagicMock()
        mock_pres.slides = []

        writer.apply_translations(
            [para_info], ["안녕하세요"], mock_pres, text_fit_mode="none"
        )

        self.assertEqual(run1.text, "안녕하세요")
        self.assertEqual(run2.text, "")


class RuleBasedDistributionTestCase(unittest.TestCase):
    """Tests for _try_rule_based_distribution in TranslationService."""

    def _try(self, group_texts, translation):
        from src.services.translation_service import TranslationService
        return TranslationService._try_rule_based_distribution(group_texts, translation)

    def test_anchor_at_end(self) -> None:
        """Anchor token at end of translation should be matched."""
        result = self._try(["Total: ", "$1,500"], "합계: $1,500")
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].text, "합계: ")
        self.assertEqual(result[0].group_index, 0)
        self.assertEqual(result[1].text, "$1,500")
        self.assertEqual(result[1].group_index, 1)

    def test_anchor_at_start(self) -> None:
        """Anchor token at start of translation should be matched."""
        result = self._try(["$500", " discount"], "$500 할인")
        self.assertIsNotNone(result)
        self.assertEqual(result[0].text, "$500")
        self.assertEqual(result[0].group_index, 0)
        self.assertEqual(result[1].text, " 할인")
        self.assertEqual(result[1].group_index, 1)

    def test_anchor_in_middle_returns_none(self) -> None:
        """Anchor in the middle of translation is ambiguous — should NOT match."""
        result = self._try(["Revenue increased by ", "20%"], "매출이 20% 증가했습니다")
        self.assertIsNone(result)

    def test_no_anchor_returns_none(self) -> None:
        """Pure text groups without numbers/symbols should not be rule-matched."""
        result = self._try(["Important ", "notice"], "중요 공지")
        self.assertIsNone(result)

    def test_three_groups_returns_none(self) -> None:
        """Only 2-group paragraphs are handled by rule-based."""
        result = self._try(["A", "B", "C"], "가나다")
        self.assertIsNone(result)

    def test_short_anchor_skipped(self) -> None:
        """Single-char anchors should be skipped (too ambiguous)."""
        result = self._try(["Price: ", "$"], "가격: $")
        self.assertIsNone(result)

    def test_anchor_not_found_returns_none(self) -> None:
        """If anchor text is not in translation, return None."""
        result = self._try(["Cost: ", "100USD"], "비용: 100달러")
        self.assertIsNone(result)


class DistributeColorsBatchingTestCase(unittest.TestCase):
    """Tests for distribute_colors batching behavior."""

    @patch("src.chains.color_distribution_chain._invoke_batch")
    def test_single_batch_success(self, mock_invoke) -> None:
        """Single small batch should work normally."""
        from src.chains.color_distribution_chain import ColoredSegment, distribute_colors

        mock_invoke.return_value = [
            [ColoredSegment(text="합계: ", group_index=0),
             ColoredSegment(text="$1,500", group_index=1)],
        ]

        result = distribute_colors(
            [["Total: ", "$1,500"]],
            ["합계: $1,500"],
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        mock_invoke.assert_called_once()

    @patch("src.chains.color_distribution_chain._invoke_batch")
    def test_partial_batch_failure(self, mock_invoke) -> None:
        """If one batch fails, others should still succeed."""
        from src.chains.color_distribution_chain import ColoredSegment, distribute_colors, _BATCH_SIZE

        seg = [ColoredSegment(text="text", group_index=0)]
        # Create enough items to span 2 batches
        n = _BATCH_SIZE + 1
        groups = [["a", "b"]] * n
        texts = ["번역"] * n

        # First batch succeeds, second fails
        mock_invoke.side_effect = [
            [seg] * _BATCH_SIZE,  # first batch succeeds
            None,  # second batch fails
        ]

        result = distribute_colors(groups, texts)
        self.assertIsNotNone(result)  # Should not be None since first batch worked
        # First batch items should have results
        self.assertIsNotNone(result[0])
        # Second batch item should be None
        self.assertIsNone(result[_BATCH_SIZE])

    @patch("src.chains.color_distribution_chain._invoke_batch")
    def test_all_batches_fail(self, mock_invoke) -> None:
        """If all batches fail, should return None."""
        from src.chains.color_distribution_chain import distribute_colors

        mock_invoke.return_value = None

        result = distribute_colors(
            [["a", "b"], ["c", "d"]],
            ["번역1", "번역2"],
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
