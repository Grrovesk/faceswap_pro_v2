"""Face-restoration backend registry (T2-2, 2026-06-11).

Lets the Face Swap and Lip-Sync paths pick which face-restoration
model runs at the enhance stage.  All backends expose the same
contract used by the legacy `faceswap/gfpgan.py` entry point:

    enhance(video_path: Path, log) -> Path

so swapping the backend is a one-line dispatch through `enhance(name, ...)`.

Currently shipped backends:
    - "none"          : pass-through, returns the input video unchanged
    - "gfpgan"        : the existing in-process worker (default)
    - "codeformer"    : CodeFormer (basicsr arch, auto-downloads .pth)
    - "restoreformer" : RestoreFormer++ (NOT YET INSTALLED -- raises
                        clear error pointing to install instructions)

The RestoreFormer backend is intentionally stubbed in this iteration
to keep the install footprint sane.  When the user wants it, they
clone https://github.com/wzhouxiff/RestoreFormerPlusPlus into
v2/external_repos/RestoreFormerPlusPlus/ and re-run; the backend
will auto-detect and use it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Tuple

logger = logging.getLogger(__name__)


BACKEND_NAMES: Tuple[str, ...] = (
    "none", "gfpgan", "codeformer", "restoreformer",
)
DEFAULT_BACKEND = "gfpgan"


def list_backends() -> List[Tuple[str, str]]:
    """Return [(value, display_label), ...] for UI dropdowns."""
    return [
        ("none",         "None (no restoration)"),
        ("gfpgan",       "GFPGAN v1.4 (default, fast)"),
        ("codeformer",   "CodeFormer (often sharper eyes)"),
        ("restoreformer","RestoreFormer++ (real-world fidelity)"),
    ]


def enhance(name: str,
             video_path: Path,
             log: Callable[[str], None] = print,
             **kwargs) -> Path:
    """Dispatch by backend name.  Raises ValueError on unknown name.
    On a backend-specific runtime failure, returns the input path
    unchanged so the user still gets the un-enhanced render.
    """
    nm = (name or DEFAULT_BACKEND).lower().strip()
    if nm == "none":
        log(f"[face-restoration] backend=none -- pass-through")
        return Path(video_path)
    if nm == "gfpgan":
        from core import gfpgan_worker
        return gfpgan_worker.enhance(Path(video_path), log=log)
    if nm == "codeformer":
        from . import codeformer_backend
        return codeformer_backend.enhance(Path(video_path), log=log,
                                            **kwargs)
    if nm == "restoreformer":
        from . import restoreformer_backend
        return restoreformer_backend.enhance(Path(video_path), log=log,
                                                **kwargs)
    raise ValueError(f"unknown face-restoration backend: {nm!r}; "
                       f"valid: {BACKEND_NAMES}")
