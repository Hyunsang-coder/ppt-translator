"""Streamlit-driven progress tracking utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import streamlit as st


@dataclass
class ProgressTracker:
    """Manage Streamlit progress components for sequential translation."""

    total_batches: int
    total_sentences: int
    total_elapsed: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.start_time = time.time()

    def update(self, current_batch: int, batch_start_idx: int, batch_end_idx: int) -> None:
        """Update the UI before processing a batch.

        Args:
            current_batch: 1-based index of the batch.
            batch_start_idx: 1-based starting paragraph index for this batch.
            batch_end_idx: 1-based ending paragraph index for this batch.
        """

        ratio = current_batch / max(self.total_batches, 1)
        elapsed = time.time() - self.start_time
        eta = (elapsed / ratio) * (1 - ratio) if ratio > 0 else 0

        self.status_text.info(
            "📊 번역 중... 배치 %d/%d (%d~%d/%d 문장) | 경과: %.1fs | 예상 잔여: %.1fs"
            % (current_batch, self.total_batches, batch_start_idx, batch_end_idx, self.total_sentences, elapsed, eta)
        )

    def complete(self, current_batch: int) -> None:
        """Mark a batch as completed in the progress bar.

        Args:
            current_batch: 1-based index of the recently finished batch.
        """

        ratio = current_batch / max(self.total_batches, 1)
        self.progress_bar.progress(ratio)

    def finish(self) -> float:
        """Finalize the progress display once translation completes."""

        self.total_elapsed = time.time() - self.start_time
        self.progress_bar.progress(1.0)
        minutes, seconds = divmod(self.total_elapsed, 60)
        self.status_text.success(
            "✅ 번역이 완료되었습니다! 총 소요: %d분 %.1f초"
            % (int(minutes), seconds)
        )
        return self.total_elapsed

    def get_total_elapsed(self) -> float:
        """Return the measured total elapsed time for the translation run."""

        return self.total_elapsed
