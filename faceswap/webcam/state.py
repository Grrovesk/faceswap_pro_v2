"""Thread-safe state for the webcam pipeline.

Replaces the legacy module-level _STREAM_SOURCE / _STREAM_OPTS dicts
with a typed, locked container. Read by swap_fn on every frame; the
Gradio tab handlers write to it on user interaction.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class WebcamState:
    """Per-session swap configuration the worker reads each frame."""
    source_path: Optional[str] = None
    gpu_id: int = 0
    det_threshold: float = 0.5
    blend_method: str = "poisson"
    # Output filters (applied after swap, before encode + virtual cam)
    brightness: int = 0          # -100..+100
    contrast: float = 1.0        # 0.5..2.0
    saturation: float = 1.0      # 0.0..2.0
    # Virtual camera output (pyvirtualcam -> OBS / Zoom / Discord)
    virtual_cam_on: bool = False


_STATE = WebcamState()
_LOCK = threading.Lock()


def get_snapshot() -> WebcamState:
    """Return a frozen copy of the current state."""
    with _LOCK:
        return WebcamState(
            source_path=_STATE.source_path,
            gpu_id=_STATE.gpu_id,
            det_threshold=_STATE.det_threshold,
            blend_method=_STATE.blend_method,
            brightness=_STATE.brightness,
            contrast=_STATE.contrast,
            saturation=_STATE.saturation,
            virtual_cam_on=_STATE.virtual_cam_on,
        )


def set_source(path: Optional[str]) -> None:
    with _LOCK:
        _STATE.source_path = (str(path) if path else None)


def set_options(*, gpu_id: Optional[int] = None,
                  det_threshold: Optional[float] = None,
                  blend_method: Optional[str] = None,
                  brightness: Optional[int] = None,
                  contrast: Optional[float] = None,
                  saturation: Optional[float] = None,
                  virtual_cam_on: Optional[bool] = None) -> None:
    with _LOCK:
        if gpu_id is not None:        _STATE.gpu_id = int(gpu_id)
        if det_threshold is not None: _STATE.det_threshold = float(det_threshold)
        if blend_method is not None:  _STATE.blend_method = str(blend_method)
        if brightness is not None:    _STATE.brightness = int(brightness)
        if contrast is not None:      _STATE.contrast = float(contrast)
        if saturation is not None:    _STATE.saturation = float(saturation)
        if virtual_cam_on is not None:_STATE.virtual_cam_on = bool(virtual_cam_on)
