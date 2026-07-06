"""Quality-record JSONL ledger (WP-C2).

Appends the translation team's shared quality contract records to two
append-only JSONL files so the team's rule-promotion loop and KPI tooling can
consume them. The JSONL files are the source of truth — this repo adds no DB.

All writes are **best-effort**: a write failure logs a warning and never
propagates into the translation path (contract §3.5). See
``consulting/ppt-translator-quality-design.md`` §3.1/§3.4 for the schemas.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from src.services.consistency_sweep import Finding

LOGGER = logging.getLogger(__name__)

_CLIENT = "ppt_translator"
_RECORDS_FILE = "quality_records.jsonl"
_RUNS_FILE = "quality_runs.jsonl"

# Finding type -> default severity bucket for run findings_count aggregation.
# (Individual findings carry their own severity; this is only the fallback map.)


def _default_quality_dir() -> Path:
    """Resolve the default ledger directory under the app data dir."""
    try:
        import platformdirs

        return Path(platformdirs.user_data_dir("ppt-translator")) / "quality"
    except Exception:  # pragma: no cover - platformdirs always present in prod
        return Path.home() / ".ppt-translator" / "quality"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def direction_code(source_lang: str, target_lang: str) -> str:
    """Map display-name language pair to the contract direction code."""
    m = {"한국어": "ko", "영어": "en", "일본어": "ja", "중국어": "zh"}
    src = m.get(source_lang, "xx")
    tgt = m.get(target_lang, "xx")
    return f"{src}_to_{tgt}"


class QualityRecorder:
    """Append quality runs and records to JSONL. Best-effort, never raises."""

    def __init__(self, quality_dir: Optional[str] = None) -> None:
        self._dir = Path(quality_dir) if quality_dir else _default_quality_dir()

    @property
    def records_path(self) -> Path:
        return self._dir / _RECORDS_FILE

    @property
    def runs_path(self) -> Path:
        return self._dir / _RUNS_FILE

    def _append(self, path: Path, row: dict) -> None:
        """Append one JSON row. Swallows all errors (best-effort)."""
        try:
            os.makedirs(path.parent, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Quality ledger write failed (%s): %s", path.name, exc)

    def record_run(
        self,
        *,
        job_id: str,
        model: Optional[str],
        source_lang: str,
        target_lang: str,
        doc_words: int,
        findings: List[Finding],
        notes: Optional[str] = None,
    ) -> None:
        """Append one quality_run row (contract §3.4)."""
        counts = {"critical": 0, "major": 0, "minor": 0}
        for f in findings:
            if f.severity in counts:
                counts[f.severity] += 1

        row = {
            "id": _new_id("run"),
            "client": _CLIENT,
            "project_id": job_id,
            "started_at": _now_ms(),
            "stage": "s1_translate",
            "executor": "app",
            "model": model,
            "direction": direction_code(source_lang, target_lang),
            "route_id": None,
            "doc_words": doc_words,
            "findings_count": counts,
            "notes": notes,
        }
        self._append(self.runs_path, row)

    def record_finding(
        self,
        finding: Finding,
        *,
        job_id: str,
        doc_ref: str,
        source_lang: str,
        target_lang: str,
        model: Optional[str],
    ) -> None:
        """Append one quality_record row (contract §3.1) for a sweep finding."""
        row = {
            "id": _new_id("qr"),
            "client": _CLIENT,
            "project_id": job_id,
            "created_at": _now_ms(),
            "doc_ref": doc_ref,
            "route_id": None,
            "direction": direction_code(source_lang, target_lang),
            "content_type": "presentation",
            "location": finding.location,
            "segment": {
                "source": finding.segment.get("source"),
                "output": finding.segment.get("output"),
                "corrected": None,
                "context": None,
            },
            "finding": {
                "type": finding.type,
                "severity": finding.severity,
                "description": finding.description,
                "suggested_fix": finding.suggested_fix,
            },
            "origin": {
                "stage": "s4_consistency",
                "caught_by": "s4_consistency",
                "executor": "app",
                "producer_model": model,
                "reviewer_model": None,
            },
            "disposition": "proposed",
            "promotion": {
                "status": "candidate",
                "matched_rule": None,
            },
        }
        self._append(self.records_path, row)

    def record_findings(
        self,
        findings: List[Finding],
        *,
        job_id: str,
        doc_ref: str,
        source_lang: str,
        target_lang: str,
        model: Optional[str],
    ) -> None:
        """Convenience: record every finding in a list."""
        for finding in findings:
            self.record_finding(
                finding,
                job_id=job_id,
                doc_ref=doc_ref,
                source_lang=source_lang,
                target_lang=target_lang,
                model=model,
            )
