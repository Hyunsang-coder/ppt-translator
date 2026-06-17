"""Tests for the reusable PPTX color mapping audit script."""

from __future__ import annotations

from scripts.evaluate_color_mapping import ParagraphInfo, RunInfo, audit


def _run(
    text: str,
    color: str = "default",
    neutral: bool = True,
    start: int = 0,
) -> RunInfo:
    return RunInfo(
        text=text,
        color=color,
        neutral=neutral,
        start=start,
        end=start + len(text),
    )


def _paragraph(text: str, runs: list[RunInfo]) -> ParagraphInfo:
    return ParagraphInfo(
        slide=1,
        shape=1,
        paragraph=1,
        text=text,
        runs=runs,
    )


def test_audit_flags_dropped_source_highlight() -> None:
    source = [
        _paragraph(
            "짧은 쿨타임이 핵심",
            [
                _run("짧은 쿨타임", "#FF0000", neutral=False),
                _run("이 핵심", start=len("짧은 쿨타임")),
            ],
        )
    ]
    target = [_paragraph("Short cooldown is key", [_run("Short cooldown is key")])]

    findings = audit(source, target)

    assert len(findings) == 1
    assert findings[0].kind == "dropped_highlight"
    assert findings[0].severity == "low"


def test_audit_flags_position_like_highlight() -> None:
    source = [
        _paragraph(
            "자동으로 크로스헤어 고정",
            [
                _run("자동으로", "#FF0000", neutral=False),
                _run(" 크로스헤어 고정", start=len("자동으로")),
            ],
        )
    ]
    target = [
        _paragraph(
            "Automatically locks onto nearby enemies",
            [
                _run("Automatically", "#FF0000", neutral=False),
                _run(" locks onto nearby enemies", start=len("Automatically")),
            ],
        )
    ]

    findings = audit(source, target)

    assert len(findings) == 1
    assert findings[0].kind == "position_like_highlight"
    assert findings[0].severity == "high"


def test_audit_flags_target_added_highlight() -> None:
    source = [_paragraph("일반 문장", [_run("일반 문장")])]
    target = [
        _paragraph(
            "Plain sentence",
            [
                _run("Plain", "#00FF00", neutral=False),
                _run(" sentence", start=len("Plain")),
            ],
        )
    ]

    findings = audit(source, target)

    assert len(findings) == 1
    assert findings[0].kind == "target_added_highlight"
    assert findings[0].severity == "medium"
