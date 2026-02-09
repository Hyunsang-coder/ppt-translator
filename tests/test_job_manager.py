"""Tests for JobManager race conditions, cancellation, and event bounds."""

from __future__ import annotations

import asyncio
import io
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.job_manager import Job, JobEvent, JobManager, JobState, JobType


@pytest.fixture
def manager():
    """Fresh JobManager for each test."""
    return JobManager(max_jobs=100)


class TestJobStateLock:
    """Tests for race condition prevention in job state transitions."""

    @pytest.mark.asyncio
    async def test_complete_and_fail_concurrent(self, manager: JobManager):
        """Concurrent complete_job + fail_job should not corrupt state.

        One should win; the other should be ignored because the state
        is already terminal.
        """
        job = manager.create_job(JobType.TRANSLATION)
        manager.start_job(job.id, asyncio.ensure_future(asyncio.sleep(999)))

        # Fire both concurrently
        await asyncio.gather(
            manager.complete_job(
                job.id,
                output_file=io.BytesIO(b"data"),
                output_filename="out.pptx",
            ),
            manager.fail_job(job.id, "boom"),
        )

        final = manager.get_job(job.id)
        assert final is not None
        # State must be one of the terminal states, not stuck in RUNNING
        assert final.state in (JobState.COMPLETED, JobState.FAILED)
        # completed_at must be set exactly once (not None)
        assert final.completed_at is not None

    @pytest.mark.asyncio
    async def test_delete_then_complete_ignores_complete(self, manager: JobManager):
        """If job is cancelled, subsequent complete_job should be ignored."""
        job = manager.create_job(JobType.TRANSLATION)
        task = asyncio.ensure_future(asyncio.sleep(999))
        manager.start_job(job.id, task)

        await manager.delete_job(job.id)
        await manager.complete_job(job.id, output_file=io.BytesIO(b"x"))

        final = manager.get_job(job.id)
        assert final is not None
        assert final.state == JobState.CANCELLED

    @pytest.mark.asyncio
    async def test_fail_then_complete_ignores_complete(self, manager: JobManager):
        """If job failed, subsequent complete_job should be ignored."""
        job = manager.create_job(JobType.TRANSLATION)
        manager.start_job(job.id, asyncio.ensure_future(asyncio.sleep(999)))

        await manager.fail_job(job.id, "error")
        await manager.complete_job(job.id, output_file=io.BytesIO(b"x"))

        final = manager.get_job(job.id)
        assert final is not None
        assert final.state == JobState.FAILED


class TestDeleteJobAwaitsTask:
    """Tests that delete_job properly awaits task cancellation."""

    @pytest.mark.asyncio
    async def test_delete_awaits_running_task(self, manager: JobManager):
        """delete_job should await the cancelled task so it actually stops."""
        cancel_observed = False

        async def long_running():
            nonlocal cancel_observed
            try:
                await asyncio.sleep(999)
            except asyncio.CancelledError:
                cancel_observed = True
                raise

        task = asyncio.ensure_future(long_running())
        job = manager.create_job(JobType.TRANSLATION)
        manager.start_job(job.id, task)

        # Let the task start running before cancelling
        await asyncio.sleep(0)

        await manager.delete_job(job.id)

        # Task should be done after delete returns
        assert task.done()
        assert cancel_observed

    @pytest.mark.asyncio
    async def test_delete_job_without_task(self, manager: JobManager):
        """delete_job should work fine when no task is set (PENDING job)."""
        job = manager.create_job(JobType.TRANSLATION)
        result = await manager.delete_job(job.id)
        assert result is True
        assert manager.get_job(job.id).state == JobState.CANCELLED

    @pytest.mark.asyncio
    async def test_delete_nonexistent_job(self, manager: JobManager):
        """delete_job should return False for unknown job IDs."""
        result = await manager.delete_job("nonexistent")
        assert result is False


class TestEventsBounded:
    """Tests that job events don't grow unbounded."""

    def test_events_capped_at_maxlen(self, manager: JobManager):
        """Events list should not exceed the configured max length."""
        job = manager.create_job(JobType.TRANSLATION)

        # Add many more events than the cap
        for i in range(1000):
            job.add_event("progress", {"i": i})

        # Events should be capped (we'll use 500 as the cap)
        assert len(job.events) <= 500

    def test_events_preserves_latest(self, manager: JobManager):
        """When capped, latest events should be kept (oldest dropped)."""
        job = manager.create_job(JobType.TRANSLATION)

        for i in range(1000):
            job.add_event("progress", {"i": i})

        # The last event should have the highest i
        last_event = job.events[-1]
        assert last_event.data["i"] == 999


class TestCleanupReleasesResources:
    """Tests that cleanup properly releases resources."""

    def test_cleanup_clears_output_file(self, manager: JobManager):
        """Cleanup should release BytesIO references."""
        job = manager.create_job(JobType.TRANSLATION)
        job.state = JobState.COMPLETED
        job.completed_at = time.time() - 7200  # 2 hours ago
        job.output_file = io.BytesIO(b"x" * 10000)

        manager._cleanup_old_jobs()

        # Job should be removed entirely
        assert manager.get_job(job.id) is None
