"""Save / load named LipsyncJob configs as JSON.

NEW FEATURE: the legacy app only had 3 hardcoded presets
(Quick/Balanced/Best) — none persisted, none user-editable.
v2 lets you save the current settings under any name and recall them
later. JSON files land in v2/presets/.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from .config import LatentSyncKnobs, VoiceSwap


PRESETS_DIR = Path(__file__).resolve().parent.parent / "presets"


def _safe_name(name: str) -> str:
    """Strip/replace anything that would make a bad filename."""
    out = "".join(c if (c.isalnum() or c in "_- ") else "_"
                  for c in str(name).strip())
    return out.strip()[:60] or "untitled"


def list_presets() -> List[str]:
    if not PRESETS_DIR.is_dir():
        return []
    return sorted(p.stem for p in PRESETS_DIR.glob("*.json"))


def save_preset(name: str, isolate: bool, quick: bool, enhance: bool,
                 extend_single: bool, ls_steps: int, ls_guidance: float,
                 ls_deepcache: bool, ls_seed: int,
                 voice_model: str, voice_transpose: int) -> str:
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    safe = _safe_name(name)
    blob = {
        "name": safe,
        "isolate_vocals": bool(isolate),
        "quick_test": bool(quick),
        "enhance_faces": bool(enhance),
        "extend_single": bool(extend_single),
        "latentsync": asdict(LatentSyncKnobs(
            inference_steps=int(ls_steps),
            guidance_scale=float(ls_guidance),
            enable_deepcache=bool(ls_deepcache),
            seed=int(ls_seed),
        )),
        "voice_swap": asdict(VoiceSwap(
            model_basename=str(voice_model or ""),
            transpose_semitones=int(voice_transpose),
        )),
    }
    out = PRESETS_DIR / f"{safe}.json"
    out.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return str(out)


def load_preset(name: str) -> Optional[dict]:
    if not name:
        return None
    p = PRESETS_DIR / f"{_safe_name(name)}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def delete_preset(name: str) -> bool:
    p = PRESETS_DIR / f"{_safe_name(name)}.json"
    try:
        p.unlink()
        return True
    except OSError:
        return False
