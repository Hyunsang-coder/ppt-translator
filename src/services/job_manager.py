"""Job manager for asynchronous translation and extraction tasks."""

from __future__ import annotations

import asyncio
import io
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from src.services.models import TranslationProgress, TranslationResult, TranslationStatus

LOGGER = logging.getLogger(__name__)

# Maximum number of events retained per job
_MAX_EVENTS = 500


class JobType(str, Enum):
    """Type of job."""

    TRANSLATION = "translation"
    EXTRACTION = "extraction"


class JobState(str, Enum):
    """Overall job state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL_STATES = frozenset({JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED})


@dataclass
class JobEvent:
    """Event emitted during job execution."""

    event_type: str  # progress, log, complete, error
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class Job:
    """Represents an async job."""

    id: str
    job_type: JobType
    state: JobState = JobState.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: Optional[TranslationProgress] = None
    result: Optional[TranslationResult] = None
    output_file: Optional[io.BytesIO] = None
    output_filename: Optional[str] = None
    error_message: Optional[str] = None
    extraction_result: Optional[str] = None  # For extraction jobs (markdown)
    events: deque = field(default_factory=lambda: deque(maxlen=_MAX_EVENTS))
    _task: Optional[asyncio.Task] = field(default=None, repr=False)
    _event_queue: Optional[asyncio.Queue] = field(default=None, repr=False)
    _loop: Optional[asyncio.AbstractEventLoop] = field(default=None, repr=False)
    _state_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def add_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Add an event to history deque and async queue.

        Thread-safe: both deque append and queue put are guaranteed to
        run on the event-loop thread to prevent ``RuntimeError: deque
        mutated during iteration`` when :meth:`stream_events` reads
        the deque concurrently.

        When no event loop is available (sync context), the deque is
        appended directly since no concurrent SSE streaming is possible.
        """
        event = JobEvent(event_type=event_type, data=data)

        if self._loop is None:
            # No async context — direct append, no concurrent access
            self.events.append(event)
            return

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is self._loop:
            # Event-loop thread — safe to modify directly
            self._dispatch_event(event)
        else:
            # Worker thread — schedule on event loop
            try:
                self._loop.call_soon_threadsafe(self._dispatch_event, event)
            except RuntimeError:
                # Event loop is closed; fall back to direct append
                self.events.append(event)

    def _dispatch_event(self, event: JobEvent) -> None:
        """Append event to history deque and push to async queue.

        Must run on the event-loop thread to avoid concurrent deque
        mutation with :meth:`stream_events`.
        """
        self.events.append(event)
        if self._event_queue is not None:
            try:
                self._event_queue.put_nowait(event)
            except asyncio.QueueFull:
                LOGGER.warning("Event queue full for job %s", self.id)


class JobManager:
    """Manages async jobs with event streaming."""

    def __init__(self, max_jobs: int = 100, max_running: int = 2) -> None:
        self._jobs: Dict[str, Job] = {}
        self._max_jobs = max_jobs
        self._max_running = max_running
        self._running_semaphore = asyncio.Semaphore(max_running)
        self._cleanup_task: Optional[asyncio.Task] = None

    def create_job(self, job_type: JobType) -> Job:
        """Create a new job."""
        # Clean up old completed jobs if at capacity
        if len(self._jobs) >= self._max_jobs:
            self._cleanup_old_jobs()

        job_id = str(uuid.uuid4())

        # Capture the running event loop so add_event() can bridge
        # from worker threads via call_soon_threadsafe.
        # When called outside an async context (e.g. in sync tests)
        # the queue is omitted — events still go into the deque.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        job = Job(
            id=job_id,
            job_type=job_type,
            _event_queue=asyncio.Queue(maxsize=100) if loop else None,
            _loop=loop,
        )
        self._jobs[job_id] = job
        LOGGER.info("Created job %s of type %s", job_id, job_type)
        return job

    def try_create_job(self, job_type: JobType, max_active: int) -> Optional[Job]:
        """Create a job only when active jobs are below capacity.

        This check-and-create path performs no ``await`` and is therefore
        effectively atomic on the event-loop thread.

        Args:
            job_type: Type of job to create.
            max_active: Maximum allowed count of running + pending jobs.

        Returns:
            Created Job when capacity is available, otherwise ``None``.
        """
        if self.get_active_count() >= max_active:
            return None
        return self.create_job(job_type)

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    async def delete_job(self, job_id: str) -> bool:
        """Cancel a job (keeps it in store for status queries; cleanup removes it later)."""
        job = self._jobs.get(job_id)
        if job is None:
            return False

        async with job._state_lock:
            if job._task is not None and not job._task.done():
                job._task.cancel()
                try:
                    await job._task
                except (asyncio.CancelledError, Exception):
                    pass

            job.state = JobState.CANCELLED
            job.completed_at = time.time()

        job.add_event("cancelled", {"message": "Job cancelled by user"})
        LOGGER.info("Cancelled job %s", job_id)
        return True

    def update_job_progress(self, job_id: str, progress: TranslationProgress) -> None:
        """Update job progress and emit event."""
        job = self._jobs.get(job_id)
        if job is None:
            return

        job.progress = progress
        job.add_event(
            "progress",
            {
                "status": progress.status.value,
                "current_batch": progress.current_batch,
                "total_batches": progress.total_batches,
                "current_sentence": progress.current_sentence,
                "total_sentences": progress.total_sentences,
                "percent": progress.percent,
                "message": progress.message,
            },
        )

    async def complete_job(
        self,
        job_id: str,
        result: Optional[TranslationResult] = None,
        output_file: Optional[io.BytesIO] = None,
        output_filename: Optional[str] = None,
        extraction_result: Optional[str] = None,
    ) -> None:
        """Mark job as completed."""
        job = self._jobs.get(job_id)
        if job is None:
            return

        async with job._state_lock:
            # Don't overwrite terminal states (e.g. already cancelled)
            if job.state in _TERMINAL_STATES:
                LOGGER.info("Job %s already %s, ignoring complete", job_id, job.state.value)
                return

            job.state = JobState.COMPLETED
            job.completed_at = time.time()
            job.result = result
            job.output_file = output_file
            job.output_filename = output_filename
            job.extraction_result = extraction_result

        event_data: Dict[str, Any] = {"status": "completed"}
        if result:
            event_data.update(
                {
                    "source_language": result.source_language_detected,
                    "target_language": result.target_language_used,
                    "total_paragraphs": result.total_paragraphs,
                    "elapsed_seconds": result.elapsed_seconds,
                }
            )
        if extraction_result:
            event_data["has_extraction_result"] = True

        job.add_event("complete", event_data)
        LOGGER.info("Job %s completed", job_id)

    async def fail_job(self, job_id: str, error_message: str) -> None:
        """Mark job as failed."""
        job = self._jobs.get(job_id)
        if job is None:
            return

        async with job._state_lock:
            # Don't overwrite terminal states (e.g. already cancelled)
            if job.state in _TERMINAL_STATES:
                LOGGER.info("Job %s already %s, ignoring failure", job_id, job.state.value)
                return

            job.state = JobState.FAILED
            job.completed_at = time.time()
            job.error_message = error_message

        job.add_event("error", {"message": error_message})
        LOGGER.error("Job %s failed: %s", job_id, error_message)

    def start_job(self, job_id: str, task: asyncio.Task) -> None:
        """Mark job as running with associated task."""
        job = self._jobs.get(job_id)
        if job is None:
            return

        job.state = JobState.RUNNING
        job.started_at = time.time()
        job._task = task
        job.add_event("started", {"message": "Job started"})

    async def stream_events(
        self, job_id: str, timeout: float = 300.0
    ) -> AsyncIterator[JobEvent]:
        """Stream events for a job via SSE."""
        job = self._jobs.get(job_id)
        if job is None:
            return

        # Send any existing events first
        for event in list(job.events):
            yield event

        # If job is already done, stop streaming
        if job.state in _TERMINAL_STATES:
            return

        # Stream new events
        queue = job._event_queue
        if queue is None:
            return

        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                break

            if job.state in _TERMINAL_STATES:
                # Drain remaining events
                while not queue.empty():
                    try:
                        event = queue.get_nowait()
                        yield event
                    except asyncio.QueueEmpty:
                        break
                break

            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                yield event
            except asyncio.TimeoutError:
                # Send keepalive
                yield JobEvent(event_type="keepalive", data={})

    def _cleanup_old_jobs(self) -> None:
        """Remove old completed/failed jobs."""
        now = time.time()
        max_age = 3600  # 1 hour

        to_remove = []
        for job_id, job in self._jobs.items():
            if job.state in _TERMINAL_STATES:
                if job.completed_at and (now - job.completed_at) > max_age:
                    to_remove.append(job_id)

        for job_id in to_remove:
            del self._jobs[job_id]
            LOGGER.info("Cleaned up old job %s", job_id)

    def get_all_jobs(self) -> List[Job]:
        """Get all jobs."""
        return list(self._jobs.values())

    def get_running_count(self) -> int:
        """Return the number of currently running jobs."""
        return sum(1 for j in self._jobs.values() if j.state == JobState.RUNNING)

    def get_pending_count(self) -> int:
        """Return the number of pending (queued) jobs."""
        return sum(1 for j in self._jobs.values() if j.state == JobState.PENDING)

    def get_active_count(self) -> int:
        """Return running + pending jobs count."""
        return sum(
            1
            for j in self._jobs.values()
            if j.state in (JobState.RUNNING, JobState.PENDING)
        )

    @property
    def max_running(self) -> int:
        """Maximum number of concurrently running jobs."""
        return self._max_running

    @property
    def running_semaphore(self) -> asyncio.Semaphore:
        """Semaphore controlling concurrent job execution."""
        return self._running_semaphore


# Global job manager instance
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get the global job manager instance."""
    global _job_manager
    if _job_manager is None:
        from src.utils.config import get_settings

        settings = get_settings()
        _job_manager = JobManager(max_running=settings.max_running_jobs)
    return _job_manager
