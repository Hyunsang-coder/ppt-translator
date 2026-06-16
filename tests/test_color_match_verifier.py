"""Tests for the color matching verification loop helpers."""

from __future__ import annotations

from src.utils.color_match_verifier import (
    DEFAULT_COLOR_MATCH_CASES,
    build_color_match_fixture,
    collect_paragraph_run_segments,
    verify_case_segments,
    verify_output_segments,
)


def test_verify_case_segments_passes_when_anchor_color_matches() -> None:
    case = DEFAULT_COLOR_MATCH_CASES[0]
    result = verify_case_segments(
        case,
        [
            ("북미 지역 매출이 ", "111111"),
            ("20%", "D72638"),
            (" 증가했습니다", "111111"),
        ],
    )

    assert result.passed
    assert result.errors == []


def test_verify_case_segments_fails_when_anchor_color_differs() -> None:
    case = DEFAULT_COLOR_MATCH_CASES[0]
    result = verify_case_segments(
        case,
        [
            ("북미 지역 매출이 ", "111111"),
            ("20%", "111111"),
            (" 증가했습니다", "111111"),
        ],
    )

    assert not result.passed
    assert "color mismatch" in result.errors[0]


def test_verify_output_segments_reports_missing_paragraph() -> None:
    results = verify_output_segments(DEFAULT_COLOR_MATCH_CASES[:1], [])

    assert len(results) == 1
    assert not results[0].passed
    assert results[0].errors == ["missing output paragraph"]


def test_build_color_match_fixture_round_trips_source_colors(tmp_path) -> None:
    fixture_path = tmp_path / "fixture.pptx"

    build_color_match_fixture(fixture_path, DEFAULT_COLOR_MATCH_CASES[:1])
    paragraphs = collect_paragraph_run_segments(fixture_path)

    assert len(paragraphs) == 1
    assert paragraphs[0] == [
        ("Revenue increased by ", "111111"),
        ("20%", "D72638"),
        (" in North America", "111111"),
    ]
