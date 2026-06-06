"""GFPGAN face restoration post-step.

PERF: tries the in-process persistent worker first
(v2/core/gfpgan_worker.py) which keeps the GFPGANer + face_helper
in memory across renders. On exception, falls back to the legacy
bridged helper in core/lipsync.py.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

from .paths import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def enhance(video_path: Path,
             log: Callable[[str], None] = print) -> Path:
    """Run GFPGAN on every frame of video_path. Returns the enhanced
    mp4 path (or video_path on best-effort failure)."""
    # Try in-process worker first (model stays loaded across renders).
    try:
        from core import gfpgan_worker
        t0 = time.perf_counter()
        out = gfpgan_worker.enhance(Path(video_path), log=log)
        dt = time.perf_counter() - t0
        log(f"[gfpgan] worker enhance OK in {dt:.1f}s")
        return Path(out) if out else video_path
    except Exception as exc:
        log(f"[gfpgan] worker FAILED ({exc}); falling back to legacy helper")

    # Fallback: legacy bridged helper from core/lipsync.py
    try:
        from core import lipsync as _ls
        helper = getattr(_ls, "_gfpgan_restore_video", None)
        if helper is None:
            log("[gfpgan] legacy helper not available; skipping enhance")
            return video_path
        out = helper(str(video_path), log=log)
        return Path(out) if out else video_path
    except Exception as exc:
        log(f"[gfpgan] legacy fallback also failed ({exc}); "
            f"returning un-enhanced video")
        return video_path
