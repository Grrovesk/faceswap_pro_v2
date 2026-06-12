"""Project-level session bundling (T1-3).

A "project" is a named folder under ``v2/projects/<name>/`` that
captures all knob values across the Lip-Sync and Face Swap tabs in a
single ``project.json``.  Unlike presets (which are flat key/value),
projects are structured by tab so the Open handler can dispatch
``gr.update()`` calls to the right widgets across both tabs in one
click.

This is the MVP -- knob values only.  File paths are recorded in the
JSON for the user's reference but the files are NOT auto-copied
(deferred to a future iteration; the user re-uploads if files moved).

Public API:
    list_projects() -> List[str]
    save_project(name: str, blob: dict) -> Path
    load_project(name: str) -> Optional[dict]
    delete_project(name: str) -> bool
    project_dir(name: str) -> Path
    PROJECTS_DIR (constant)

Schema of project.json (v1):
    {
      "version": "v1",
      "name": "...",
      "created_at": <unix-timestamp>,
      "updated_at": <unix-timestamp>,
      "lipsync": {
          "face_paths": ["..."],
          "audio_path": "...",
          "isolate_vocals": true, "quick_test": false,
          "enhance_faces": true, "extend_single": false,
          "ls_steps": 20, "ls_guidance": 1.5,
          "ls_deepcache": true, "ls_seed": -1,
      },
      "face_swap": {
          "source_image": "...",
          "target_video": "...",
          "blend_method": "poisson",
          "enhance": false,
          ... (every vs_* knob)
      },
    }
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import List, Optional

from .paths import PROJECT_ROOT


PROJECTS_DIR = PROJECT_ROOT / "projects"
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_name(name: str) -> str:
    """Project names get used as directory names.  Strip everything
    that isn't filesystem-safe; collapse repeated dashes."""
    s = re.sub(r"[^\w\-. ]", "", (name or "")).strip().replace(" ", "_")
    s = re.sub(r"-+", "-", s)
    return s[:80] or "untitled"


def project_dir(name: str) -> Path:
    return PROJECTS_DIR / _safe_name(name)


def list_projects() -> List[str]:
    """Return the names of every project under PROJECTS_DIR sorted
    by most-recently-modified.  Names are the on-disk directory
    basenames (already safe).
    """
    if not PROJECTS_DIR.is_dir():
        return []
    rows = []
    for d in PROJECTS_DIR.iterdir():
        if not d.is_dir():
            continue
        mtime = 0.0
        manifest = d / "project.json"
        if manifest.is_file():
            try:
                mtime = manifest.stat().st_mtime
            except OSError:
                pass
        rows.append((d.name, mtime))
    rows.sort(key=lambda r: -r[1])
    return [r[0] for r in rows]


def save_project(name: str, blob: dict) -> Path:
    """Write ``project.json`` for the named project.  Creates the
    directory if it doesn't exist.  Returns the manifest path.
    """
    d = project_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    manifest = d / "project.json"
    now = time.time()
    out = dict(blob or {})
    out.setdefault("version", "v1")
    out["name"] = _safe_name(name)
    if "created_at" not in out:
        # Preserve created_at on overwrite if the file already exists.
        prior = load_project(name)
        out["created_at"] = (
            float(prior.get("created_at", now))
            if isinstance(prior, dict) else now
        )
    out["updated_at"] = now
    manifest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return manifest


def load_project(name: str) -> Optional[dict]:
    """Read ``project.json`` for the named project, or None."""
    manifest = project_dir(name) / "project.json"
    if not manifest.is_file():
        return None
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return None


def delete_project(name: str) -> bool:
    """Delete the project's directory tree.  Returns True on success.
    Use with caution -- this is irreversible.
    """
    import shutil
    d = project_dir(name)
    if not d.is_dir():
        return False
    try:
        shutil.rmtree(d)
        return True
    except OSError:
        return False
