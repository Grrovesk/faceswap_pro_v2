"""SAM2 daemon client.

Spawns ``core/_sam2_worker.py --daemon`` as a long-running subprocess
that loads SAM2 weights ONCE at startup and then accepts JSON-lines
requests over stdin.  Returns sub-second responses for click events
once warm.

Used by the Rotoscoping tab so the user gets interactive segmentation
without paying the 10-30 second model-load tax per click that the
legacy one-shot CLI mode incurs.

Thread-safety
-------------
Requests are matched to responses by ``request_id`` (uuid).  Multiple
threads can call methods on the same :class:`SAM2Daemon` instance
concurrently; each call gets its own response queue.  The underlying
SAM2 inference is GPU-bound, so concurrent calls serialize at the
GPU anyway, but the client doesn't block the event loop.

Lifecycle
---------

    daemon = SAM2Daemon.singleton()
    daemon.start()                              # blocks until model loaded
    info = daemon.load_video("path/to.mp4")     # binds state to video
    mask_info = daemon.click(x, y, frame_idx)   # < 1 s once warm
    daemon.propagate(masks_dir, on_progress=lambda d: ...)
    daemon.shutdown()                           # idempotent
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


# How long to wait at each lifecycle step.  Click responses come back
# almost instantly once the model is warm; propagate can take minutes
# on long videos so its timeout is per-batch (we wait for the next
# stdout line, not for the whole job).
DEFAULT_LOAD_TIMEOUT_S = 120.0   # SAM2 model load can be slow on cold disk
DEFAULT_VIDEO_LOAD_TIMEOUT_S = 60.0
DEFAULT_CLICK_TIMEOUT_S = 30.0
DEFAULT_PROPAGATE_TIMEOUT_S = 1800.0  # 30 min hard cap


class SAM2DaemonError(RuntimeError):
    """Anything that goes wrong in the client-side protocol."""


class SAM2Daemon:
    """Client wrapper around a long-running ``_sam2_worker.py --daemon``."""

    _INSTANCE: Optional["SAM2Daemon"] = None
    _INSTANCE_LOCK = threading.Lock()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        py_path: Optional[str] = None,
        worker_path: Optional[str] = None,
        sam2_ckpt: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self._py_path = py_path or sys.executable
        if worker_path:
            self._worker_path = str(worker_path)
        else:
            self._worker_path = str(
                Path(__file__).resolve().parent / "_sam2_worker.py"
            )
        self._ckpt_path = str(sam2_ckpt) if sam2_ckpt else None
        self._env = dict(os.environ)
        if env:
            self._env.update(env)
        self._env.setdefault("PYTHONUNBUFFERED", "1")
        self._env.setdefault("PYTHONWARNINGS", "ignore")

        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None

        self._pending: Dict[str, Queue] = {}
        self._progress_callbacks: Dict[str, Callable[[dict], None]] = {}
        self._pending_lock = threading.Lock()

        self._ready_event = threading.Event()
        self._start_error: Optional[str] = None
        self._shutdown = False

    @classmethod
    def singleton(cls, **kwargs) -> "SAM2Daemon":
        """Return the process-wide singleton, constructing on first call.

        Subsequent calls ignore ``kwargs``; if you need a fresh daemon
        with different settings, call :meth:`shutdown` first.
        """
        with cls._INSTANCE_LOCK:
            if cls._INSTANCE is None:
                cls._INSTANCE = cls(**kwargs)
            return cls._INSTANCE

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self, timeout_s: float = DEFAULT_LOAD_TIMEOUT_S) -> None:
        """Spawn the worker subprocess and wait for the ``ready`` event.

        Idempotent: calling start() on an already-running daemon is a
        no-op.  On failure, raises :class:`SAM2DaemonError`.
        """
        if self._proc is not None and self._proc.poll() is None:
            return
        if not self._ckpt_path:
            self._ckpt_path = self._resolve_ckpt_path()
        if not os.path.isfile(self._ckpt_path):
            raise SAM2DaemonError(
                f"SAM2 weights not found: {self._ckpt_path}.  "
                f"Run sam2_install.ensure_sam2_weights() first."
            )

        argv = [self._py_path, self._worker_path,
                "--daemon", "--sam2_ckpt", self._ckpt_path]
        logger.info("SAM2 daemon spawn: %s", argv)
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(Path(self._worker_path).parent),
            env=self._env,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(
            target=self._read_stdout_loop, daemon=True,
            name="sam2-daemon-stdout",
        )
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(
            target=self._read_stderr_loop, daemon=True,
            name="sam2-daemon-stderr",
        )
        self._stderr_thread.start()

        if not self._ready_event.wait(timeout=timeout_s):
            self.shutdown(graceful=False)
            raise SAM2DaemonError(
                f"SAM2 daemon did not become ready in {timeout_s:.0f}s.  "
                f"Last error: {self._start_error or '(none)'}"
            )
        if self._start_error:
            err = self._start_error
            self.shutdown(graceful=False)
            raise SAM2DaemonError(f"SAM2 daemon startup failed: {err}")

    def shutdown(self, graceful: bool = True, timeout_s: float = 5.0) -> None:
        """Terminate the worker subprocess.  Idempotent."""
        if self._proc is None:
            return
        if self._shutdown:
            return
        self._shutdown = True
        try:
            if graceful and self._proc.poll() is None:
                try:
                    self._proc.stdin.write(
                        json.dumps({"op": "shutdown"}) + "\n")
                    self._proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    self._proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=timeout_s)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
            else:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        finally:
            with self._pending_lock:
                # Wake any waiters with an error response.
                for q in self._pending.values():
                    q.put({"status": "error",
                            "message": "daemon shut down"})
                self._pending.clear()
                self._progress_callbacks.clear()
            with type(self)._INSTANCE_LOCK:
                if type(self)._INSTANCE is self:
                    type(self)._INSTANCE = None

    def is_running(self) -> bool:
        return (self._proc is not None
                and self._proc.poll() is None
                and self._ready_event.is_set())

    # ------------------------------------------------------------------
    # Public RPC surface
    # ------------------------------------------------------------------
    def load_video(self, video_path: str,
                   timeout_s: float = DEFAULT_VIDEO_LOAD_TIMEOUT_S) -> dict:
        return self._call("load_video",
                          {"video_path": str(video_path)},
                          timeout_s=timeout_s)

    def click(self, x: int, y: int, frame_idx: int,
              label: int = 1, obj_id: int = 1,
              mask_out_path: Optional[str] = None,
              return_b64: bool = False,
              timeout_s: float = DEFAULT_CLICK_TIMEOUT_S) -> dict:
        payload: Dict[str, Any] = {
            "x": int(x), "y": int(y),
            "frame_idx": int(frame_idx),
            "label": int(label), "obj_id": int(obj_id),
            "return_b64": bool(return_b64),
        }
        if mask_out_path:
            payload["mask_out_path"] = str(mask_out_path)
        return self._call("click", payload, timeout_s=timeout_s)

    def apply_click(self, frame_idx: int, obj_id: int,
                      points: list, labels: list,
                      masks_out_root: Optional[str] = None,
                      return_b64: bool = False,
                      timeout_s: float = DEFAULT_CLICK_TIMEOUT_S) -> dict:
        """Demo-style incremental multi-object update.  Sends the
        full point list for one obj_id on one frame; daemon does a
        single add_new_points_or_box call (no reset_state) and
        returns masks for every currently-tracked obj_id at this
        frame.  points is [[x, y], ...], labels is
        [1|0, ...] matching points.  An empty points list
        removes the object from tracking.
        """
        payload: Dict[str, Any] = {
            "frame_idx": int(frame_idx),
            "obj_id": int(obj_id),
            "points": [[int(p[0]), int(p[1])] for p in (points or [])],
            "labels": [int(l) for l in (labels or [])],
            "return_b64": bool(return_b64),
        }
        if masks_out_root:
            payload["masks_out_root"] = str(masks_out_root)
        return self._call("apply_click", payload, timeout_s=timeout_s)

    def set_prompts(self, frame_idx: int, prompts: list,
                    obj_id: int = 1,
                    mask_out_path: Optional[str] = None,
                    return_b64: bool = False,
                    neg_carve_radius: int = 0,
                    timeout_s: float = DEFAULT_CLICK_TIMEOUT_S) -> dict:
        """Send the FULL list of clicks for ``obj_id`` and get the mask
        SAM2 produces from exactly that set.  The daemon does a
        ``reset_state`` first then submits all points at once, so the
        result is a pure function of the prompts (deterministic, no
        cross-call drift, no first-click weirdness, no accumulation
        hacks).  Use this in place of incremental ``click()`` calls
        whenever the caller knows the current full intent.

        ``prompts`` is a list of dicts: ``{"x": int, "y": int,
        "label": 1|0, "frame_idx": int}`` -- ``frame_idx`` is optional
        and defaults to the top-level ``frame_idx`` (display frame).
        """
        payload: Dict[str, Any] = {
            "obj_id": int(obj_id),
            "frame_idx": int(frame_idx),
            "prompts": list(prompts),
            "return_b64": bool(return_b64),
            "neg_carve_radius": int(neg_carve_radius),
        }
        if mask_out_path:
            payload["mask_out_path"] = str(mask_out_path)
        return self._call("set_prompts", payload, timeout_s=timeout_s)

    def set_all_prompts(self, frame_idx: int, objects: list,
                        masks_root=None, return_b64: bool = False,
                        timeout_s: float = DEFAULT_CLICK_TIMEOUT_S) -> dict:
        """Multi-object peer of set_prompts.  Submit the COMPLETE
        prompt list for every tracked object in one call.  Daemon does
        a single reset_state then populates every object's points
        atomically, so different objects no longer wipe each other on
        consecutive single-object set_prompts calls.

        objects is a list of dicts:
        {"obj_id": int, "prompts": [{"x", "y", "label",
        "frame_idx"}, ...]}.  Returns a dict whose "objects" key
        is a list of {"obj_id", "nonzero_pixels", "mask_b64"?}
        entries for the display frame, in input order.
        """
        payload: Dict[str, Any] = {
            "frame_idx": int(frame_idx),
            "objects": list(objects),
            "return_b64": bool(return_b64),
        }
        if masks_root:
            payload["masks_root"] = str(masks_root)
        return self._call("set_all_prompts", payload,
                          timeout_s=timeout_s)

    def propagate(self, masks_dir: str,
                  on_progress: Optional[Callable[[dict], None]] = None,
                  timeout_s: float = DEFAULT_PROPAGATE_TIMEOUT_S) -> dict:
        return self._call("propagate",
                          {"masks_dir": str(masks_dir)},
                          timeout_s=timeout_s,
                          progress_cb=on_progress)

    def clear(self, obj_id: Optional[int] = None,
              timeout_s: float = 15.0) -> dict:
        return self._call("clear", {"obj_id": obj_id},
                          timeout_s=timeout_s)

    def ping(self, timeout_s: float = 5.0) -> dict:
        return self._call("ping", {}, timeout_s=timeout_s)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_ckpt_path() -> str:
        try:
            from core import sam2_install as _si
        except ImportError:
            raise SAM2DaemonError(
                "core.sam2_install unavailable; pass sam2_ckpt explicitly "
                "to SAM2Daemon().")
        ckpt = _si.ensure_sam2_weights(log=lambda *_a, **_k: None)
        return str(ckpt)

    def _call(self, op: str, payload: dict, timeout_s: float,
              progress_cb: Optional[Callable[[dict], None]] = None) -> dict:
        if not self.is_running():
            raise SAM2DaemonError("SAM2 daemon is not running; call start()")
        if self._proc.poll() is not None:
            raise SAM2DaemonError(
                f"SAM2 daemon subprocess died (exit={self._proc.returncode})"
            )
        req_id = str(uuid.uuid4())
        q: Queue = Queue()
        with self._pending_lock:
            self._pending[req_id] = q
            if progress_cb:
                self._progress_callbacks[req_id] = progress_cb
        try:
            req = {"op": op, "request_id": req_id, **payload}
            try:
                self._proc.stdin.write(json.dumps(req) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise SAM2DaemonError(
                    f"could not write to daemon stdin: {exc}") from exc

            try:
                resp = q.get(timeout=timeout_s)
            except Empty:
                raise SAM2DaemonError(
                    f"SAM2 daemon op={op} timed out after {timeout_s:.0f}s")
            if resp.get("status") == "error":
                raise SAM2DaemonError(
                    f"daemon op={op} failed: {resp.get('message')}")
            return resp
        finally:
            with self._pending_lock:
                self._pending.pop(req_id, None)
                self._progress_callbacks.pop(req_id, None)

    def _read_stdout_loop(self) -> None:
        if not self._proc or not self._proc.stdout:
            return
        for raw in self._proc.stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("non-json from daemon stdout: %s", line[:200])
                continue

            ev = msg.get("event")
            if ev == "ready":
                logger.info("SAM2 daemon ready")
                self._ready_event.set()
                continue
            if ev == "error":
                self._start_error = msg.get("message", "")
                logger.error("SAM2 daemon startup error: %s",
                              self._start_error)
                self._ready_event.set()
                continue
            if ev == "starting":
                logger.info("SAM2 daemon starting")
                continue
            if ev == "model_loaded":
                logger.info("SAM2 daemon model loaded: %s",
                              msg.get("ckpt"))
                continue
            if ev == "warn":
                logger.warning("SAM2 daemon: %s", msg.get("message"))
                continue

            req_id = msg.get("request_id")
            if not req_id:
                logger.debug("unrouted daemon message: %s", msg)
                continue

            status = msg.get("status")
            if status == "progress":
                with self._pending_lock:
                    cb = self._progress_callbacks.get(req_id)
                if cb is not None:
                    try:
                        cb(msg)
                    except Exception as exc:
                        logger.warning(
                            "progress callback raised: %s", exc)
                continue

            with self._pending_lock:
                q = self._pending.get(req_id)
            if q is None:
                logger.debug(
                    "response for unknown request: %s", req_id)
                continue
            q.put(msg)

    def _read_stderr_loop(self) -> None:
        if not self._proc or not self._proc.stderr:
            return
        for line in self._proc.stderr:
            line = line.rstrip()
            if line:
                logger.info("sam2[%d] %s", self._proc.pid, line)


# Convenience for callers that just want a quick warm daemon.
def get_or_start_daemon(timeout_s: float = DEFAULT_LOAD_TIMEOUT_S
                         ) -> SAM2Daemon:
    """Return the singleton, starting it if necessary."""
    d = SAM2Daemon.singleton()
    if not d.is_running():
        d.start(timeout_s=timeout_s)
    return d


__all__ = ["SAM2Daemon", "SAM2DaemonError", "get_or_start_daemon"]
