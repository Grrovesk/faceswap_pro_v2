"""Render queue + background worker.

A single FIFO queue of Job entries. One worker thread drains it
serially (we have 1 GPU, can't parallelise lipsync). Each Job
wraps a LipsyncJob so the queue can be used directly by the
existing orchestrator.render.

UI reads list_all() and refreshes the Queue tab on demand.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .config import LipsyncJob

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """One queued render. Status flows: queued -> running -> "
    "completed | failed | cancelled."""
    id: str
    label: str                        # user-facing short name
    lipsync_job: LipsyncJob
    status: str = "queued"            # queued/running/completed/failed/cancelled
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result_path: Optional[str] = None
    error: Optional[str] = None

    @property
    def elapsed_s(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at else time.time()
        return end - self.started_at


class JobQueue:
    """Singleton thread-safe queue + worker thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: List[Job] = []
        self._worker_running = False
        self._worker: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

    # ---- submission ----
    def submit(self, lipsync_job: LipsyncJob, label: str = "") -> Job:
        with self._lock:
            jid = uuid.uuid4().hex[:8]
            if not label:
                # auto-name from primary face clip
                try:
                    label = Path(lipsync_job.face_paths[0]).stem[:40]
                except Exception:
                    label = jid
            j = Job(id=jid, label=label, lipsync_job=lipsync_job)
            self._jobs.append(j)
        self._ensure_worker()
        logger.info("queue: submitted %s (%s)", jid, label)
        return j

    # ---- inspection ----
    def list_all(self) -> List[Job]:
        with self._lock:
            return list(self._jobs)

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            for j in self._jobs:
                if j.id == job_id:
                    return j
            return None

    # ---- mutations ----
    def cancel(self, job_id: str) -> bool:
        """Cancel a job. queued -> cancelled instantly; running is
        flagged and will report cancelled when current frame finishes
        (best-effort -- lipsync subprocess can't be interrupted)."""
        with self._lock:
            for j in self._jobs:
                if j.id == job_id and j.status == "queued":
                    j.status = "cancelled"
                    j.finished_at = time.time()
                    logger.info("queue: cancelled queued job %s", job_id)
                    return True
        return False

    def clear_completed(self) -> int:
        """Remove completed/failed/cancelled jobs. Returns count removed."""
        with self._lock:
            before = len(self._jobs)
            self._jobs = [j for j in self._jobs
                          if j.status in ("queued", "running")]
            return before - len(self._jobs)

    def stop(self):
        self._stop_evt.set()
        w = self._worker
        if w is not None:
            w.join(timeout=5.0)

    # ---- worker ----
    def _ensure_worker(self):
        with self._lock:
            if self._worker_running:
                return
            self._worker_running = True
            self._stop_evt.clear()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()
        logger.info("queue: worker started")

    def _next_queued(self) -> Optional[Job]:
        with self._lock:
            for j in self._jobs:
                if j.status == "queued":
                    return j
            return None

    def _loop(self):
        from . import orchestrator
        while not self._stop_evt.is_set():
            job = self._next_queued()
            if job is None:
                # Idle: park worker for a bit before re-checking
                if self._stop_evt.wait(timeout=1.5):
                    break
                continue
            with self._lock:
                job.status = "running"
                job.started_at = time.time()
            logger.info("queue: running %s (%s)", job.id, job.label)
            try:
                def _log(msg, _job=job):
                    print(f"[queue {_job.id}] {msg}", flush=True)
                out = orchestrator.render(job.lipsync_job, log=_log)
                with self._lock:
                    job.status = "completed"
                    job.result_path = str(out)
                    job.finished_at = time.time()
                logger.info("queue: completed %s -> %s", job.id, out)
            except Exception as exc:
                with self._lock:
                    job.status = "failed"
                    job.error = (str(exc)
                                  + "\n" + traceback.format_exc()[-1000:])
                    job.finished_at = time.time()
                logger.exception("queue: job %s failed", job.id)
        with self._lock:
            self._worker_running = False
        logger.info("queue: worker stopped")


# Module-level singleton
_QUEUE: Optional[JobQueue] = None
_QUEUE_LOCK = threading.Lock()


def get_queue() -> JobQueue:
    global _QUEUE
    with _QUEUE_LOCK:
        if _QUEUE is None:
            _QUEUE = JobQueue()
    return _QUEUE


# ---- helpers for UI ----
def jobs_as_rows() -> list:
    """Return [[id, label, status, elapsed, result/error], ...]
    sorted oldest-first."""
    q = get_queue()
    rows = []
    for j in q.list_all():
        elapsed = ""
        if j.elapsed_s is not None:
            elapsed = f"{j.elapsed_s:.0f}s"
        result = ""
        if j.status == "completed" and j.result_path:
            result = Path(j.result_path).name
        elif j.status == "failed" and j.error:
            result = (j.error.splitlines()[0]
                       if j.error else "")[:80]
        rows.append([j.id, j.label, j.status, elapsed, result])
    return rows
