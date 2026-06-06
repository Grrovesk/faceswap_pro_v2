"""Vocal isolation (Demucs) — bridges core.lipsync._isolate_vocals.

WHY THIS MODULE EXISTS: v2 originally dropped the isolate_vocals
flag entirely. The result: LatentSync's Whisper feature extractor
saw the full song (vocals + instruments), audio conditioning got
drowned in music, and lip motion collapsed to barely-moving. This
module is the missing pipeline step.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from .paths import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def isolate(song_path: Path,
             log: Callable[[str], None] = print) -> Path:
    """Demucs two-stems separation. Returns the vocals.wav path.

    Falls back to returning the original path on any failure --
    lipsync will still run, just with a weaker audio signal.
    """
    try:
        from core.lipsync import _isolate_vocals as _legacy
    except Exception as exc:
        log(f"[vocal-iso] cannot import legacy helper: {exc}; "
            "passing original audio through")
        return song_path
    try:
        vocals = _legacy(str(song_path), log=log)
        return Path(vocals)
    except Exception as exc:
        log(f"[vocal-iso] Demucs failed ({exc}); "
            "passing original audio through")
        return song_path
