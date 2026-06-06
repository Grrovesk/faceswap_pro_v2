"""RVC voice swap, optional pre-step. Bridges to the proven RVC
pipeline in faceswap_pro/core/voice_clone.py -- that code has the
correct ffmpeg-PATH handling for Demucs and the right .index lookup
for RVC. Re-implementing here would be ~200 lines of identical
subprocess plumbing.

The bridge is a single function call. If voice_clone ever needs to
be replaced, this module is the only thing in v2 that imports it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable

from .config import VoiceSwap
from .ffmpeg_tools import resolve_ffmpeg


def apply_voice_swap(audio_path: Path, swap: VoiceSwap,
                      log: Callable[[str], None] = print) -> Path:
    """If swap.model_basename is set, run RVC. Else return audio_path.

    Returns the path to the (possibly remixed) audio to use downstream.
    """
    if not swap.model_basename or swap.model_basename == "(no voice swap)":
        return audio_path
    # Lazy import so the v2 package doesn't hard-require RVC be set up.
    try:
        # v2 STANDALONE: core/voice_clone.py lives inside v2/core/. The
        # v2/ folder is on sys.path so this import resolves locally
        # without reaching outside v2/. RVC is invoked as a subprocess;
        # the RVC repo location is configured in v2/faceswap/paths.py
        # (EXTERNAL_REPOS_ROOT / "RVC", overridable via env var).
        from core.voice_clone import rvc_convert_song
    except Exception as exc:
        raise RuntimeError(
            f"voice swap requested but core.voice_clone unavailable: {exc}")

    ffmpeg = resolve_ffmpeg()
    log(f"[voice] RVC swap -> model={swap.model_basename}, "
        f"transpose={swap.transpose_semitones}")
    remix_path, _dry = rvc_convert_song(
        str(audio_path), swap.model_basename, ffmpeg,
        transpose=int(swap.transpose_semitones), log=log,
    )
    log(f"[voice] swapped audio -> {remix_path}")
    return Path(remix_path)


def list_available_voices() -> list:
    """For the UI dropdown."""
    try:
        from core.voice_clone import list_voice_models
        return list_voice_models() or []
    except Exception:
        return []
