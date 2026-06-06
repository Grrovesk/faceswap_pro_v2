"""FastAPI routes for the v2 webcam MJPEG stream. v2 owns this.

Endpoints (mirror legacy /webcam_stream/* paths so the browser JS
doesn't need to change):
    GET  /webcam_stream/video  -> MJPEG stream
    POST /webcam_stream/start  -> {running, error}
    POST /webcam_stream/stop   -> {running}
    GET  /webcam_stream/stats  -> {running, fps, latency_ms, frames, error}
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import cv2
import numpy as np

from .worker import SwapStreamWorker, get_worker

logger = logging.getLogger(__name__)


def mjpeg_generator(worker: SwapStreamWorker,
                    target_fps: float = 30.0):
    boundary = b"--frame"
    period = 1.0 / max(1.0, float(target_fps))
    placeholder = cv2.imencode(
        ".jpg", np.zeros((4, 4, 3), np.uint8))[1].tobytes()
    while True:
        if not worker.running:
            yield (boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                   + placeholder + b"\r\n")
            break
        jpeg = worker.get_latest_jpeg() or placeholder
        yield (boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n"
               + jpeg + b"\r\n")
        time.sleep(period)


def register_stream_routes(fast_app,
                            swap_fn: Optional[Callable] = None
                            ) -> SwapStreamWorker:
    """Attach /webcam_stream/* routes to a FastAPI app. Returns the
    singleton worker so the caller can manage shutdown."""
    from fastapi.responses import JSONResponse, StreamingResponse

    worker = get_worker(swap_fn)

    @fast_app.get("/webcam_stream/video")
    def _video():
        return StreamingResponse(
            mjpeg_generator(worker),
            media_type="multipart/x-mixed-replace; boundary=frame")

    @fast_app.post("/webcam_stream/start")
    def _start(device: int = 0, width: int = 640, height: int = 480,
                quality: int = 80):
        worker.start(device, width, height, quality)
        return JSONResponse({"running": worker.running,
                              "error": worker.error})

    @fast_app.post("/webcam_stream/stop")
    def _stop():
        worker.stop()
        return JSONResponse({"running": worker.running})

    @fast_app.post("/webcam_stream/record/start")
    def _rec_start():
        if not worker.running:
            return JSONResponse(
                {"recording": False,
                 "error": "camera not running -- start camera first"})
        path = worker.start_recording()
        return JSONResponse({
            "recording": worker.is_recording,
            "path": str(path) if path else "",
            "error": "" if path else "VideoWriter failed",
        })

    @fast_app.post("/webcam_stream/record/stop")
    def _rec_stop():
        path = worker.stop_recording()
        return JSONResponse({
            "recording": False,
            "path": str(path) if path else "",
            "saved": bool(path),
        })

    @fast_app.get("/webcam_stream/stats")
    def _stats():
        return JSONResponse({
            "running": worker.running,
            "fps": round(worker.fps, 1),
            "latency_ms": round(worker.latency_ms, 1),
            "frames": worker.frames,
            "error": worker.error,
            "recording": worker.is_recording,
        })

    @fast_app.post("/webcam_stream/open_recordings_dir")
    def _open_recordings_dir():
        """Open v2/recordings/webcam/ in the OS file explorer.

        Resolves junctions / symlinks first so the path passed to
        os.startfile is the real underlying directory (matters when
        the repo is run via the junction-set-up test harness)."""
        import os, subprocess, sys
        from ..paths import WEBCAM_RECORDINGS_DIR
        d = WEBCAM_RECORDINGS_DIR.resolve()
        d.mkdir(parents=True, exist_ok=True)
        path_str = str(d)
        logger.info("v2 webcam open_recordings_dir requested: %s",
                    path_str)
        try:
            if sys.platform == "win32":
                os.startfile(path_str)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path_str])
            else:
                subprocess.Popen(["xdg-open", path_str])
            return JSONResponse({"ok": True, "path": path_str})
        except Exception as exc:
            logger.warning("v2 webcam open_recordings_dir failed "
                           "for %s: %s", path_str, exc)
            return JSONResponse({"ok": False,
                                 "path": path_str,
                                 "error": str(exc)})

    logger.info("v2 webcam_stream routes registered")
    return worker
