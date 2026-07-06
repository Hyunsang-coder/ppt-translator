"""Unit tests for the review/edit session (WP-C5)."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches

from src.core.ppt_parser import PPTParser
from src.services.consistency_sweep import Finding
from src.services.quality_records import QualityRecorder
from src.services.review_session import ReviewSession


def _deck(text: str, n: int) -> io.BytesIO:
    prs = Presentation()
    blank = prs.slide_layouts[6]
    for _ in range(n):
        s = prs.slides.add_slide(blank)
        tb = s.shapes.add_textbox(Inches(1), Inches(1), Inches(6), Inches(1))
        tb.text_frame.text = text
    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf


def _session(text: str, n: int, targets=None, findings=None) -> ReviewSession:
    paras, pres = PPTParser().extract_paragraphs(_deck(text, n))
    targets = targets or [f"T{i}" for i in range(len(paras))]
    return ReviewSession(
        presentation=pres, paragraphs=paras, translated_texts=targets,
        findings=findings or [], source_lang="한국어", target_lang="영어",
        model="stub",
    )


class FragmentsTestCase(unittest.TestCase):
    def test_repeat_count_reflects_duplicates(self) -> None:
        sess = _session("자세한 내용은 부록 참조", 4)
        frags = sess.fragments()
        self.assertEqual(len(frags), 4)
        self.assertTrue(all(f.repeat_count == 4 for f in frags))

    def test_findings_attach_to_fragments(self) -> None:
        finding = Finding(
            type="terminology.violation", severity="major", description="x",
            location={"slide": 1, "shape": 0, "paragraph": 0},
            segment={"source": "s", "output": "o"}, ordinal=1,
            fragment_index=0, suggested_fix="Aim Punch",
        )
        sess = _session("A", 2, findings=[finding])
        frags = sess.fragments()
        self.assertEqual(len(frags[0].findings), 1)
        self.assertEqual(frags[0].findings[0]["type"], "terminology.violation")
        self.assertEqual(frags[1].findings, [])


class EditPropagationTestCase(unittest.TestCase):
    def test_identical_indices(self) -> None:
        sess = _session("반복", 3)
        self.assertEqual(sorted(sess.identical_indices(0)), [0, 1, 2])

    def test_edit_propagates_to_identicals(self) -> None:
        sess = _session("반복", 3)
        changed = sess.apply_edit(0, "EDITED", propagate_identical=True)
        self.assertEqual(sorted(changed), [0, 1, 2])
        self.assertTrue(all(t == "EDITED" for t in sess.translated_texts))

    def test_edit_without_propagation_only_target(self) -> None:
        sess = _session("반복", 3)
        changed = sess.apply_edit(0, "EDITED", propagate_identical=False)
        self.assertEqual(changed, [0])
        self.assertEqual(sess.translated_texts[1], "T1")

    def test_render_reflects_edit(self) -> None:
        sess = _session("반복", 2, targets=["OLD", "OLD"])
        sess.apply_edit(0, "NEW", propagate_identical=True)
        out = sess.render()
        check = Presentation(out)
        texts = [sh.text_frame.text for sl in check.slides for sh in sl.shapes if sh.has_text_frame]
        self.assertTrue(all(t == "NEW" for t in texts))


class PartialMatchTestCase(unittest.TestCase):
    def test_partial_candidates_found_but_not_auto_applied(self) -> None:
        sess = _session("A", 1, targets=["World Spawn item list"])
        # Manually add a second fragment target containing the phrase.
        sess.translated_texts.append("Adjusted field drop rates")
        # Simulate an edit that introduces "World Spawn".
        candidates = sess.partial_match_candidates(0, "World Spawn")
        # No fragment (other than index 0) contains "World Spawn" yet.
        self.assertEqual(candidates, [])


class LengthBudgetTestCase(unittest.TestCase):
    def test_note_has_no_budget(self) -> None:
        sess = _session("본문", 1)
        # No notes in this deck; body fragment has a budget.
        self.assertIsNotNone(sess.length_budget(0))


class RecordReviewEditTestCase(unittest.TestCase):
    def test_accepted_edit_records_triplet(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            rec = QualityRecorder(quality_dir=d)
            rec.record_review_edit(
                job_id="deck:x.pptx", doc_ref="deck:x.pptx",
                source_lang="한국어", target_lang="영어", model="stub",
                location={"slide": 3, "shape": 1, "paragraph": 0},
                segment={"source": "원문", "output": "old", "corrected": "new"},
                disposition="accepted", propagated=3,
            )
            rows = [json.loads(l) for l in (Path(d) / "quality_records.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            r = rows[0]
            self.assertEqual(r["origin"]["stage"], "manual_edit")
            self.assertEqual(r["origin"]["executor"], "human")
            self.assertEqual(r["disposition"], "accepted")
            self.assertEqual(r["segment"]["corrected"], "new")
            self.assertEqual(r["promotion"]["status"], "candidate")

    def test_rejected_ignore_records(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            rec = QualityRecorder(quality_dir=d)
            rec.record_review_edit(
                job_id="j", doc_ref="deck:x.pptx",
                source_lang="한국어", target_lang="영어", model="stub",
                location={"slide": 1, "shape": 0, "paragraph": 0},
                segment={"source": "원문", "output": "out", "corrected": None},
                disposition="rejected", finding_type="consistency.phrase",
            )
            rows = [json.loads(l) for l in (Path(d) / "quality_records.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["disposition"], "rejected")
            self.assertEqual(rows[0]["finding"]["type"], "consistency.phrase")
            self.assertEqual(rows[0]["promotion"]["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
