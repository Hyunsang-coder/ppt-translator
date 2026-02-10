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


class TestConcurrencyLimits:
    """Tests for max_running concurrency control."""

    @pytest.mark.asyncio
    async def test_running_count(self):
        """get_running_count should track running jobs."""
        mgr = JobManager(max_jobs=100, max_running=5)
        assert mgr.get_running_count() == 0

        job = mgr.create_job(JobType.TRANSLATION)
        mgr.start_job(job.id, asyncio.ensure_future(asyncio.sleep(999)))
        assert mgr.get_running_count() == 1

    @pytest.mark.asyncio
    async def test_pending_count(self):
        """get_pending_count should track pending jobs."""
        mgr = JobManager(max_jobs=100, max_running=5)
        job = mgr.create_job(JobType.TRANSLATION)
        assert mgr.get_pending_count() == 1

        mgr.start_job(job.id, asyncio.ensure_future(asyncio.sleep(999)))
        assert mgr.get_pending_count() == 0

    @pytest.mark.asyncio
    async def test_active_count(self):
        """get_active_count returns running + pending."""
        mgr = JobManager(max_jobs=100, max_running=5)
        j1 = mgr.create_job(JobType.TRANSLATION)
        j2 = mgr.create_job(JobType.TRANSLATION)
        assert mgr.get_active_count() == 2  # both pending

        mgr.start_job(j1.id, asyncio.ensure_future(asyncio.sleep(999)))
        assert mgr.get_active_count() == 2  # 1 running + 1 pending

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent_execution(self):
        """Semaphore should limit concurrent access."""
        mgr = JobManager(max_jobs=100, max_running=2)
        sem = mgr.running_semaphore

        peak_concurrent = 0
        current = 0

        async def limited_work():
            nonlocal peak_concurrent, current
            async with sem:
                current += 1
                peak_concurrent = max(peak_concurrent, current)
                await asyncio.sleep(0.05)
                current -= 1

        # Launch 5 tasks competing for 2 slots
        await asyncio.gather(*[limited_work() for _ in range(5)])

        assert peak_concurrent == 2  # Never exceeded max_running

    @pytest.mark.asyncio
    async def test_max_running_property(self):
        """max_running property should reflect the configured value."""
        mgr = JobManager(max_jobs=100, max_running=3)
        assert mgr.max_running == 3

    def test_try_create_job_respects_active_capacity(self):
        """try_create_job should reject when active capacity is full."""
        mgr = JobManager(max_jobs=100, max_running=2)

        j1 = mgr.try_create_job(JobType.TRANSLATION, max_active=2)
        j2 = mgr.try_create_job(JobType.TRANSLATION, max_active=2)
        j3 = mgr.try_create_job(JobType.TRANSLATION, max_active=2)

        assert j1 is not None
        assert j2 is not None
        assert j3 is None
        assert mgr.get_active_count() == 2


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
