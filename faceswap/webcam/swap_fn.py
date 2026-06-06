"""Per-frame face-swap callable, fully owned by v2.

Uses ONLY core.pipeline (which v2 already imports for Face Swap tab).
No imports from ui.app, ui.stream_server, or any other legacy ui.* code.

Architecture: lazy-load a FaceSwapPipeline, then drive its private
stages on each incoming webcam frame. The pipeline is constructed
once when the first swap happens; the source embedding is cached
and only re-extracted when state.source_path changes.
"""
from __future__ import annotations

import logging
import sys
import threading
from typing import Callable, Optional

import numpy as np

from ..paths import PROJECT_ROOT
from . import state as webcam_state

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def _make_pipeline_cfg(gpu_id: int, det_threshold: float,
                        blend_method: str) -> dict:
    """Same cfg shape v2 Face Swap tab builds, minimal knobs."""
    return {
        "blending":     {"method": str(blend_method)},
        "optimization": {"cuda_device": int(gpu_id)},
        "identity":     {"drift_threshold": float(det_threshold)},
        "detection":    {"threshold": float(det_threshold)},
        "output":       {"quality": "visually_lossless"},
        "enhancement":  {"method": "none"},
    }


class _SwapContext:
    """Holds the pipeline + cached source state so we don't rebuild
    on every frame. Recomputes source embedding when state.source_path
    changes."""
    def __init__(self):
        self.pipeline = None
        self.cached_src_path: Optional[str] = None
        self.cached_cfg_key: Optional[tuple] = None
        self._lock = threading.Lock()

    def ensure(self, snap: webcam_state.WebcamState):
        """Build/rebuild pipeline if cfg changed. Refresh source
        embedding if source_path changed. Returns True if ready to
        swap (pipeline + source both present)."""
        if not snap.source_path:
            return False
        cfg_key = (snap.gpu_id, snap.det_threshold, snap.blend_method)
        with self._lock:
            if self.pipeline is None or self.cached_cfg_key != cfg_key:
                from core.pipeline import FaceSwapPipeline
                self.pipeline = FaceSwapPipeline(_make_pipeline_cfg(
                    snap.gpu_id, snap.det_threshold, snap.blend_method))
                self.pipeline._init_modules()
                self.cached_cfg_key = cfg_key
                self.cached_src_path = None  # force re-embed
                logger.info("v2 webcam: FaceSwapPipeline initialised "
                            "(gpu=%s thresh=%.2f blend=%s)",
                            snap.gpu_id, snap.det_threshold,
                            snap.blend_method)
            if self.cached_src_path != snap.source_path:
                emb = self.pipeline._extract_source_embedding(
                    snap.source_path)
                self.pipeline.identity_tracker.set_source(emb)
                self.pipeline.swap_engine.set_source_embedding(emb)
                self.cached_src_path = snap.source_path
                logger.info("v2 webcam: source embedding cached for %s",
                            snap.source_path)
        return True


_CTX = _SwapContext()


def make_swap_fn() -> Callable[[np.ndarray], np.ndarray]:
    """Return the per-frame callable the SwapStreamWorker drives.

    Reads webcam_state on every frame, so source/options changes take
    effect on the next frame without restarting the worker.
    """
    from core.pipeline import FramePacket

    def swap_fn(frame_bgr: np.ndarray) -> np.ndarray:
        snap = webcam_state.get_snapshot()
        if not _CTX.ensure(snap):
            return frame_bgr        # no source -> passthrough
        try:
            pkt = FramePacket(frame_idx=0, frame_bgr=frame_bgr)
            _CTX.pipeline._stage_detect(pkt)
            if not pkt.face_found:
                return frame_bgr
            _CTX.pipeline._stage_identity(pkt, verbose=False)
            _CTX.pipeline._stage_swap(pkt)
            _CTX.pipeline._stage_blend(pkt)
            return (pkt.output_frame if pkt.output_frame is not None
                    else frame_bgr)
        except Exception:
            logger.exception("v2 webcam: per-frame swap failed")
            return frame_bgr

    return swap_fn
