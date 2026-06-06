"""Virtual camera output via pyvirtualcam.

Pipes swapped frames into a fake webcam device that OBS / Zoom /
Discord pick up. On Windows this uses the OBS Virtual Camera driver
(installed automatically by OBS Studio). On Linux uses v4l2loopback.

Graceful: if pyvirtualcam isn't installed OR no driver is available,
status() returns the reason and send_frame() is a no-op so the main
swap pipeline keeps running.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_CAM = None
_STATUS = "closed"
_W = 0
_H = 0


def status() -> str:
    """Human-readable status string for the UI."""
    return _STATUS


def is_open() -> bool:
    return _CAM is not None


def open_cam(width: int, height: int, fps: float = 30.0) -> str:
    """Open the virtual camera. Returns status string."""
    global _CAM, _STATUS, _W, _H
    with _LOCK:
        if _CAM is not None and (_W, _H) == (width, height):
            return _STATUS
        if _CAM is not None:
            try: _CAM.close()
            except Exception: pass
            _CAM = None
        try:
            import pyvirtualcam
        except ImportError:
            _STATUS = ("pyvirtualcam not installed -- "
                       "pip install pyvirtualcam")
            logger.warning(_STATUS)
            return _STATUS
        try:
            _CAM = pyvirtualcam.Camera(width=int(width),
                                        height=int(height),
                                        fps=float(fps))
            _W, _H = int(width), int(height)
            _STATUS = (f"open: {_CAM.device}  "
                       f"{_W}x{_H} @ {fps:.0f}fps")
            logger.info("virtual cam opened: %s", _STATUS)
        except Exception as exc:
            _STATUS = (f"could not open virtual cam: {exc}. "
                       "On Windows make sure OBS Studio is installed "
                       "(it provides the virtual camera driver). On "
                       "Linux load v4l2loopback.")
            _CAM = None
            logger.warning(_STATUS)
        return _STATUS


def close_cam() -> None:
    global _CAM, _STATUS
    with _LOCK:
        if _CAM is not None:
            try: _CAM.close()
            except Exception: pass
        _CAM = None
        _STATUS = "closed"
        logger.info("virtual cam closed")


def send_frame(frame_bgr) -> None:
    """Send one BGR frame to the virtual cam. No-op if not open."""
    cam = _CAM
    if cam is None or frame_bgr is None:
        return
    try:
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        cam.send(rgb)
        cam.sleep_until_next_frame()
    except Exception:
        logger.exception("virtual cam send_frame failed")
