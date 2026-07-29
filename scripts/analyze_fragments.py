#!/usr/bin/env python3
"""Measure how finely a deck gets split into review fragments.

The review screen lists one item per PPT paragraph, so a deck that wraps a
sentence with hard returns scatters it across several items. This script
measures that, and simulates the merge rule the review queue will use so the
rule's thresholds can be set from real decks instead of guessed.

Extraction goes through the production ``PPTParser``, so ``container_id`` and
``container_kind`` here are exactly what the review API serves.

Usage:
    python scripts/analyze_fragments.py deck.pptx [--notes] [--json out.json]
    python scripts/analyze_fragments.py deck.pptx --list-blocks
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.ppt_parser import ParagraphInfo, PPTParser  # noqa: E402

A_NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}

# --- merge rule (mirrors docs/REVIEW_QUEUE_PLAN.md §3.2) --------------------

# Several paragraphs in a bullet list are separate points, not one sentence.
# Speaker notes are prose where each paragraph stands alone.
NON_MERGING_KINDS = frozenset({"body", "notes"})
# A run longer than this is a list of some kind, whatever the container says.
MAX_MERGE_PARAGRAPHS = 4
# If the previous line already ended the sentence, the next one is a new thought.
SENTENCE_END = re.compile(r"[.!?。！？…:;]['\"”’)\]]*$")
# A continuation starts mid-sentence, i.e. lowercase. Measured over four real
# decks: without this, 370 of 377 merges were a heading with its description, a
# label/value pair, or a bullet list inside a plain text box. Scripts without
# letter case never merge — no genuine wrap was observed in them to calibrate a
# replacement signal against — except a trailing comma, which breaks a clause in
# any script (Korean decks close their list lines 개조식, on a noun or `~함`).
ENDS_MID_CLAUSE = re.compile(r"[,،、]$")

LENGTH_BUCKETS: Sequence[tuple[str, int]] = (
    ("1-5", 5),
    ("6-10", 10),
    ("11-20", 20),
    ("21-40", 40),
    ("41-80", 80),
    ("81-160", 160),
)


@dataclass
class Block:
    """Paragraphs the review screen would show as one item."""

    container_id: str
    kind: str
    slide: int
    items: List[ParagraphInfo] = field(default_factory=list)
    # Why this block stopped growing — for tuning the rule.
    blocked_by: str = "end"


def _line_breaks(info: ParagraphInfo) -> int:
    return len(info.paragraph._p.findall(f"{{{A_NS['a']}}}br"))


def _merge_block(previous: ParagraphInfo, current: ParagraphInfo, size: int) -> str:
    """Return "" when the two paragraphs merge, else why they do not."""
    if current.container_id != previous.container_id:
        return "container"
    if current.paragraph_index != previous.paragraph_index + 1:
        return "gap"  # an empty paragraph between them reads as a separator
    if current.container_kind in NON_MERGING_KINDS:
        return "kind"
    if size >= MAX_MERGE_PARAGRAPHS:
        return "cap"
    previous_text = (previous.original_text or "").strip()
    head = (current.original_text or "").lstrip()[:1]
    if not head.islower() and not ENDS_MID_CLAUSE.search(previous_text):
        return "not_continuation"
    if SENTENCE_END.search(previous_text):
        return "sentence_end"
    return ""


def build_blocks(paragraphs: Sequence[ParagraphInfo]) -> List[Block]:
    blocks: List[Block] = []
    for info in paragraphs:
        if blocks:
            current = blocks[-1]
            reason = _merge_block(current.items[-1], info, len(current.items))
            if not reason:
                current.items.append(info)
                continue
            current.blocked_by = reason
        blocks.append(
            Block(
                container_id=info.container_id,
                kind=info.container_kind,
                slide=info.slide_index + 1,
                items=[info],
            )
        )
    return blocks


def _length_histogram(lengths: Sequence[int]) -> dict[str, int]:
    histogram = {label: 0 for label, _ in LENGTH_BUCKETS}
    histogram["161+"] = 0
    for length in lengths:
        for label, upper in LENGTH_BUCKETS:
            if length <= upper:
                histogram[label] += 1
                break
        else:
            histogram["161+"] += 1
    return histogram


def summarize(
    paragraphs: Sequence[ParagraphInfo], blocks: Sequence[Block]
) -> dict:
    lengths = [len((p.original_text or "").strip()) for p in paragraphs]
    merged = [b for b in blocks if len(b.items) > 1]
    containers = {p.container_id for p in paragraphs}
    multi_paragraph_containers = {
        key for key, count in Counter(p.container_id for p in paragraphs).items()
        if count > 1
    }
    dropped = [p for p in paragraphs if _line_breaks(p)]

    return {
        "paragraphs": len(paragraphs),
        "containers": len(containers),
        "review_items": len(blocks),
        "reduction_pct": round(
            (1 - len(blocks) / len(paragraphs)) * 100, 1
        ) if paragraphs else 0.0,
        "paragraphs_by_kind": dict(Counter(p.container_kind for p in paragraphs)),
        "short_paragraphs": {
            "under_10_chars": sum(1 for n in lengths if n < 10),
            "under_20_chars": sum(1 for n in lengths if n < 20),
            "pct_under_20": round(
                sum(1 for n in lengths if n < 20) / len(lengths) * 100, 1
            ) if lengths else 0.0,
        },
        "length_histogram": _length_histogram(lengths),
        "split_containers": {
            "count": len(multi_paragraph_containers),
            "pct_of_containers": round(
                len(multi_paragraph_containers) / len(containers) * 100, 1
            ) if containers else 0.0,
        },
        "merged_blocks": {
            "count": len(merged),
            "paragraphs_absorbed": sum(len(b.items) for b in merged) - len(merged),
            "paragraphs_per_block": dict(
                sorted(Counter(len(b.items) for b in blocks).items())
            ),
            "max_paragraphs": max((len(b.items) for b in blocks), default=0),
        },
        # Why merges were refused — the knobs worth tuning.
        "blocked_by": dict(Counter(b.blocked_by for b in blocks)),
        "line_breaks_dropped": {
            "paragraphs": len(dropped),
            "total_breaks": sum(_line_breaks(p) for p in dropped),
            "note": "runs-join drops <a:br/>; those lines are glued with no separator",
        },
    }


def print_report(summary: dict, blocks: Sequence[Block], *, list_blocks: bool) -> None:
    print(f"\n문단 {summary['paragraphs']}개 · 컨테이너 {summary['containers']}개")
    print(f"검토 항목 {summary['review_items']}개 ({summary['reduction_pct']}% 감소)")
    print(f"  유형별 문단: {summary['paragraphs_by_kind']}")

    short = summary["short_paragraphs"]
    print(f"\n짧은 문단: 20자 미만 {short['under_20_chars']}개 ({short['pct_under_20']}%), "
          f"10자 미만 {short['under_10_chars']}개")
    print(f"  길이 분포: {summary['length_histogram']}")

    split = summary["split_containers"]
    print(f"\n2문단 이상 컨테이너: {split['count']}개 ({split['pct_of_containers']}%)")

    merged = summary["merged_blocks"]
    print(f"병합된 블록: {merged['count']}개, 흡수된 문단 {merged['paragraphs_absorbed']}개, "
          f"최대 {merged['max_paragraphs']}문단")
    print(f"  블록당 문단 수: {merged['paragraphs_per_block']}")
    print(f"  병합 거부 사유: {summary['blocked_by']}")

    breaks = summary["line_breaks_dropped"]
    print(f"\n줄바꿈(<a:br/>) 손실: 문단 {breaks['paragraphs']}개, "
          f"총 {breaks['total_breaks']}곳")

    if list_blocks:
        print("\n--- 병합된 블록 ---")
        for block in sorted(
            (b for b in blocks if len(b.items) > 1), key=lambda b: -len(b.items)
        )[:40]:
            print(f"\n[슬라이드 {block.slide}] {block.kind} · {len(block.items)}문단")
            for info in block.items:
                print(f"    {info.original_text!r}")

        print("\n--- 병합 안 된 2문단+ 컨테이너 (거부 사유 확인용) ---")
        by_container: dict[str, List[Block]] = {}
        for block in blocks:
            by_container.setdefault(block.container_id, []).append(block)
        shown = 0
        for container, group in by_container.items():
            if len(group) < 2 or shown >= 20:
                continue
            shown += 1
            print(f"\n[{container}] {group[0].kind} · {len(group)}개로 분리 "
                  f"(사유: {group[0].blocked_by})")
            for block in group[:5]:
                print(f"    {block.items[0].original_text!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pptx", type=Path, help="분석할 PPTX 파일")
    parser.add_argument("--notes", action="store_true", help="발표자 노트도 포함")
    parser.add_argument("--list-blocks", action="store_true", help="블록 원문 출력")
    parser.add_argument("--json", type=Path, help="요약을 JSON으로 저장")
    args = parser.parse_args()

    with open(args.pptx, "rb") as fh:
        import io

        paragraphs, _ = PPTParser().extract_paragraphs(
            io.BytesIO(fh.read()), translate_notes=args.notes
        )
    if not paragraphs:
        print("추출된 문단이 없습니다.")
        return

    blocks = build_blocks(paragraphs)
    summary = summarize(paragraphs, blocks)
    print_report(summary, blocks, list_blocks=args.list_blocks)

    if args.json:
        args.json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n저장: {args.json}")


if __name__ == "__main__":
    main()
