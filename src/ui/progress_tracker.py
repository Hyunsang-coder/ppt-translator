"""Streamlit-driven progress tracking utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import streamlit as st


@dataclass
class ProgressTracker:
    """Manage Streamlit progress components for sequential translation."""

    total_batches: int
    total_sentences: int
    log_update_fn: Optional[Callable[[], None]] = field(default=None, repr=False)
    total_elapsed: float = field(init=False, default=0.0)
    _completed_batches: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.start_time = time.time()
        self.status_text.info(
            "📊 번역 대기 중... (0/%d 배치)" % max(self.total_batches, 1)
        )
        self._refresh_logs()

    def reset(self, total_batches: int, total_sentences: int) -> None:
        """Reset tracker state for a new translation run."""

        self.total_batches = total_batches
        self.total_sentences = total_sentences
        self._completed_batches = 0
        self.start_time = time.time()
        self.progress_bar.progress(0)
        self.status_text.info(
            "📊 번역 대기 중... (0/%d 배치)" % max(self.total_batches, 1)
        )
        self._refresh_logs()

    def batch_completed(self, batch_start_idx: int | None = None, batch_end_idx: int | None = None) -> None:
        """Record completion of a batch and refresh the status."""

        self._completed_batches = min(self.total_batches, self._completed_batches + 1)
        ratio = self._completed_batches / max(self.total_batches, 1)
        self.progress_bar.progress(ratio)

        if self._completed_batches < self.total_batches:
            if batch_start_idx is not None and batch_end_idx is not None:
                window = f" ({batch_start_idx}~{batch_end_idx})"
            else:
                window = ""
            message = "📊 번역 중... 완료 %d/%d%s" % (
                self._completed_batches,
                self.total_batches,
                window,
            )
        else:
            message = "📊 번역이 완료되었습니다. PPT 반영 중..."

        self.status_text.info(message)
        self._refresh_logs()

    def finish(self) -> float:
        """Finalize the progress display once translation completes."""

        self.total_elapsed = time.time() - self.start_time
        self.progress_bar.progress(1.0)
        minutes, seconds = divmod(self.total_elapsed, 60)
        self.status_text.success(
            "✅ 번역이 완료되었습니다! 총 소요: %d분 %.1f초"
            % (int(minutes), seconds)
        )
        self._refresh_logs()
        return self.total_elapsed

    def get_total_elapsed(self) -> float:
        """Return the measured total elapsed time for the translation run."""

        return self.total_elapsed

    def _refresh_logs(self) -> None:
        """Render buffered logs if new entries are available."""

        if self.log_update_fn is None:
            return
        self.log_update_fn()
