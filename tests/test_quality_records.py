"""Unit tests for the quality-records JSONL ledger (WP-C2)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.services.consistency_sweep import Finding
from src.services.quality_records import QualityRecorder, direction_code


def _finding(ftype: str, severity: str, ordinal: int) -> Finding:
    return Finding(
        type=ftype,
        severity=severity,
        description="desc",
        location={"slide": 3, "shape": 1, "paragraph": 0},
        segment={"source": "원문", "output": "output"},
        ordinal=ordinal,
        suggested_fix="Aim Punch" if ftype == "terminology.violation" else None,
    )


class DirectionCodeTestCase(unittest.TestCase):
    def test_known_pair(self) -> None:
        self.assertEqual(direction_code("한국어", "영어"), "ko_to_en")
        self.assertEqual(direction_code("영어", "한국어"), "en_to_ko")

    def test_unknown_pair(self) -> None:
        self.assertEqual(direction_code("클링온어", "영어"), "xx_to_en")


class RecordRunTestCase(unittest.TestCase):
    def test_run_row_matches_contract(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            rec = QualityRecorder(quality_dir=d)
            findings = [
                _finding("terminology.violation", "major", 1),
                _finding("accuracy.omission", "critical", 2),
            ]
            rec.record_run(
                job_id="job_abc", model="claude-sonnet-5",
                source_lang="한국어", target_lang="영어",
                doc_words=3200, findings=findings,
            )
            rows = [json.loads(l) for l in (Path(d) / "quality_runs.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["client"], "ppt_translator")
            self.assertEqual(row["project_id"], "job_abc")
            self.assertEqual(row["stage"], "s1_translate")
            self.assertEqual(row["executor"], "app")
            self.assertEqual(row["direction"], "ko_to_en")
            self.assertEqual(row["doc_words"], 3200)
            self.assertEqual(row["findings_count"], {"critical": 1, "major": 1, "minor": 0})
            self.assertIsNone(row["route_id"])


class RecordFindingTestCase(unittest.TestCase):
    def test_record_row_matches_contract(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            rec = QualityRecorder(quality_dir=d)
            rec.record_findings(
                [_finding("terminology.violation", "major", 1)],
                job_id="job_abc", doc_ref="deck:Q3.pptx",
                source_lang="한국어", target_lang="영어", model="claude-sonnet-5",
            )
            rows = [json.loads(l) for l in (Path(d) / "quality_records.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["client"], "ppt_translator")
            self.assertEqual(row["doc_ref"], "deck:Q3.pptx")
            self.assertEqual(row["content_type"], "presentation")
            self.assertEqual(row["disposition"], "proposed")
            self.assertEqual(row["origin"]["stage"], "s4_consistency")
            self.assertEqual(row["origin"]["caught_by"], "s4_consistency")
            self.assertEqual(row["promotion"]["status"], "candidate")
            self.assertIsNone(row["segment"]["corrected"])
            self.assertEqual(row["finding"]["type"], "terminology.violation")
            self.assertEqual(row["finding"]["suggested_fix"], "Aim Punch")

    def test_append_accumulates(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            rec = QualityRecorder(quality_dir=d)
            for i in range(3):
                rec.record_findings(
                    [_finding("consistency.phrase", "major", i)],
                    job_id="j", doc_ref="deck:x.pptx",
                    source_lang="한국어", target_lang="영어", model="m",
                )
            rows = (Path(d) / "quality_records.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 3)


class BestEffortTestCase(unittest.TestCase):
    def test_write_failure_does_not_raise(self) -> None:
        # A path that cannot be created (a file component in the middle).
        rec = QualityRecorder(quality_dir="/dev/null/cannot/write/here")
        try:
            rec.record_run(
                job_id="j", model=None, source_lang="한국어",
                target_lang="영어", doc_words=0, findings=[],
            )
            rec.record_findings(
                [_finding("accuracy.omission", "critical", 1)],
                job_id="j", doc_ref="d", source_lang="한국어",
                target_lang="영어", model=None,
            )
        except Exception as exc:  # pragma: no cover
            self.fail(f"Ledger write must not raise, got: {exc}")


if __name__ == "__main__":
    unittest.main()
