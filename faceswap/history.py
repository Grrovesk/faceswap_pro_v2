"""Render history -- list past mp4s in recordings/lipsync with their
sidecar metadata.

NEW FEATURE: the legacy app had no way to see past renders without
opening Windows Explorer. v2 surfaces every prior render in the UI,
sortable by mtime, with one-click reload-into-player and the sidecar
JSON shown inline so you can compare settings across runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict

from . import previews
from .paths import RECORDINGS_DIR


def list_renders(max_items: int = 50) -> List[Dict]:
    """Return [{path, name, mtime, size_mb, sidecar (dict or None)}, ...]
    sorted by mtime DESC (newest first)."""
    if not RECORDINGS_DIR.is_dir():
        return []
    out = []
    for mp4 in sorted(RECORDINGS_DIR.glob("*.mp4"),
                       key=lambda p: -p.stat().st_mtime):
        try:
            st = mp4.stat()
            out.append({
                "path": str(mp4),
                "name": mp4.name,
                "mtime": st.st_mtime,
                "size_mb": st.st_size / (1024 * 1024),
                "sidecar": previews.load_sidecar(mp4),
            })
        except OSError:
            continue
        if len(out) >= max_items:
            break
    return out


def list_renders_for_dropdown() -> List[tuple]:
    """[(label, path), ...] for a Gradio Dropdown."""
    rows = list_renders()
    out = []
    for r in rows:
        sc = r.get("sidecar") or {}
        steps = sc.get("latentsync", {}).get("inference_steps", "?")
        seed = sc.get("latentsync", {}).get("seed", "?")
        elapsed = sc.get("elapsed_s")
        et_label = (f"  {elapsed:.0f}s" if isinstance(elapsed, (int, float))
                    else "")
        label = (f"{r['name']}  {r['size_mb']:.1f}MB  "
                 f"steps={steps} seed={seed}{et_label}")
        out.append((label, r["path"]))
    return out
