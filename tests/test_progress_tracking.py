"""Progress tracking regressions."""

from __future__ import annotations

from src.services.translation_service import ServiceProgressTracker


def test_sentence_progress_counts_completed_batches_not_last_finished_range() -> None:
    events = []
    tracker = ServiceProgressTracker(
        total_batches=3,
        total_sentences=90,
        callback=events.append,
    )

    tracker.reset(total_batches=3, total_sentences=90)
    tracker.batch_completed(31, 60)
    tracker.batch_completed(1, 30)
    tracker.batch_completed(61, 90)

    progress_events = [event for event in events if event.current_batch > 0]
    assert [event.percent for event in progress_events] == [33, 56, 80]
    assert [event.current_sentence for event in progress_events] == [30, 60, 90]


def test_retry_reset_after_progress_does_not_emit_regressive_start_event() -> None:
    events = []
    tracker = ServiceProgressTracker(
        total_batches=2,
        total_sentences=2,
        callback=events.append,
    )

    tracker.reset(total_batches=2, total_sentences=2)
    tracker.batch_completed(1, 1)

    events.clear()
    tracker.reset(total_batches=2, total_sentences=2)

    assert events == []
