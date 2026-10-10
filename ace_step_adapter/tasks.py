"""Task queue with a single worker thread.

The GPU runs one generation at a time. Tasks wait in a FIFO queue. Each task
reports status 0 (in progress), 1 (done) or 2 (failed), as in ACE-Step.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .engine import EngineResult
from .parsing import GenerationRequest

log = logging.getLogger("ace_step_adapter.tasks")

STATUS_RUNNING = 0
STATUS_DONE = 1
STATUS_FAILED = 2


@dataclass
class Task:
    id: str
    request: GenerationRequest
    submitted_at: float = field(default_factory=time.monotonic)
    status: int = STATUS_RUNNING
    files: list[Path] = field(default_factory=list)
    seed: int | None = None
    error: str | None = None
    timed_out: bool = False


class TaskManager:
    def __init__(self, engine, output_dir: Path, timeout_seconds: int):
        self._engine = engine
        self._output_dir = output_dir
        self._timeout = timeout_seconds
        self._queue: queue.Queue[Task] = queue.Queue()
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run_forever, name="ace-worker", daemon=True)

    def start(self) -> None:
        if not self._worker.is_alive():
            self._worker.start()

    def submit(self, request: GenerationRequest) -> Task:
        task = Task(id=uuid.uuid4().hex, request=request)
        with self._lock:
            self._tasks[task.id] = task
        self._queue.put(task)
        log.info("Task %s queued (duration=%ss, format=%s)", task.id, request.duration_seconds, request.audio_format)
        return task

    def queue_position(self, task_id: str) -> int:
        with self._lock:
            pending = [t.id for t in list(self._queue.queue)]
        return pending.index(task_id) + 1 if task_id in pending else 0

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            task = self._tasks.get(task_id)
        if task is not None:
            self._expire_if_overdue(task)
        return task

    def _expire_if_overdue(self, task: Task) -> None:
        with self._lock:
            if task.status != STATUS_RUNNING:
                return
            if time.monotonic() - task.submitted_at <= self._timeout:
                return
            task.status = STATUS_FAILED
            task.timed_out = True
            task.error = f"timeout of {self._timeout}s exceeded"
        log.error("Task %s exceeded the %ss timeout and is reported as failed", task.id, self._timeout)

    def _run_forever(self) -> None:
        while True:
            task = self._queue.get()
            try:
                self._run(task)
            finally:
                self._queue.task_done()

    def _run(self, task: Task) -> None:
        with self._lock:
            if task.timed_out:
                log.warning("Task %s expired before it started, skipping", task.id)
                return
        started = time.monotonic()
        log.info("Task %s started", task.id)
        try:
            outcome: EngineResult = self._engine.generate(task.request, self._output_dir / task.id)
        except Exception as exc:  # the worker must survive any failure
            self._finish(task, STATUS_FAILED, error=str(exc) or type(exc).__name__)
            log.error("Task %s failed after %.1fs: %s", task.id, time.monotonic() - started, exc)
            return
        self._finish(task, STATUS_DONE, files=outcome.files, seed=outcome.seed)
        log.info("Task %s finished in %.1fs", task.id, time.monotonic() - started)

    def _finish(self, task: Task, status: int, **changes) -> None:
        with self._lock:
            if task.timed_out:
                # The client already received a failure; keep it consistent.
                return
            task.status = status
            for name, value in changes.items():
                setattr(task, name, value)
            if status == STATUS_FAILED and "error" not in changes:
                task.error = "generation failed"
