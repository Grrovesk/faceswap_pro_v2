"""Live preview helpers + render ETA + sidecar JSON.

Features:
  - Frame-0 thumbnail extraction for face clips
  - Audio waveform image (pure ffmpeg via showwavespic)
  - Render-time ETA estimate based on empirical s/frame
  - Sidecar JSON write/read so any past render can be exactly reproduced
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from .config import LipsyncJob
from .ffmpeg_tools import probe_duration_seconds, resolve_ffmpeg

# Empirical LatentSync rate on an A6000 (s/frame at 512px, 20 steps,
# deepcache=on). Used for ETA. Re-measured by render-history if a
# completed render's sidecar has a recorded s/frame.
_DEFAULT_S_PER_FRAME = 1.9
_DEFAULT_FPS = 25.0

_RATE_CACHE = Path(tempfile.gettempdir()) / "fp_v2_rate.txt"


# ---- THUMBNAILS -------------------------------------------------
def extract_first_frame(video_path: Path) -> Optional[Path]:
    """Return path to a JPG of frame 0. Cached in tempdir keyed by
    abspath+mtime+size so re-uploads are instant."""
    src = Path(video_path)
    if not src.is_file():
        return None
    import hashlib
    st = src.stat()
    key = hashlib.sha256(
        f"{src.resolve()}_{int(st.st_mtime)}_{st.st_size}".encode()
    ).hexdigest()[:16]
    out = Path(tempfile.gettempdir()) / f"fp_v2_thumb_{key}.jpg"
    if out.exists() and out.stat().st_size > 1000:
        return out
    ff = resolve_ffmpeg()
    r = subprocess.run([
        ff, "-y", "-loglevel", "error",
        "-i", str(src), "-frames:v", "1",
        "-vf", "scale=320:-2",
        str(out),
    ], capture_output=True, text=True, encoding="utf-8", errors="ignore")
    return out if (r.returncode == 0 and out.exists()) else None


# ---- AUDIO WAVEFORM --------------------------------------------
def extract_audio_waveform(audio_path: Path) -> Optional[Path]:
    """Render a 600x80 PNG of the audio waveform via ffmpeg's
    showwavespic filter. Cached. No matplotlib dependency."""
    src = Path(audio_path)
    if not src.is_file():
        return None
    import hashlib
    st = src.stat()
    key = hashlib.sha256(
        f"{src.resolve()}_{int(st.st_mtime)}_{st.st_size}".encode()
    ).hexdigest()[:16]
    out = Path(tempfile.gettempdir()) / f"fp_v2_wave_{key}.png"
    if out.exists() and out.stat().st_size > 1000:
        return out
    ff = resolve_ffmpeg()
    r = subprocess.run([
        ff, "-y", "-loglevel", "error",
        "-i", str(src),
        "-filter_complex",
        "showwavespic=s=600x80:colors=#4a9eff",
        "-frames:v", "1",
        str(out),
    ], capture_output=True, text=True, encoding="utf-8", errors="ignore")
    return out if (r.returncode == 0 and out.exists()) else None


# ---- ETA --------------------------------------------------------
def _learned_s_per_frame() -> float:
    """Read the last successful render's measured s/frame if cached."""
    try:
        if _RATE_CACHE.exists():
            v = float(_RATE_CACHE.read_text().strip())
            if 0.1 < v < 100.0:
                return v
    except Exception:
        pass
    return _DEFAULT_S_PER_FRAME


def record_actual_s_per_frame(value: float) -> None:
    """Persist for future ETA refinement."""
    try:
        if 0.1 < float(value) < 100.0:
            _RATE_CACHE.write_text(f"{float(value):.4f}")
    except Exception:
        pass


def estimate_render_seconds(face_paths: List[Path],
                              audio_path: Optional[Path],
                              extend_single: bool,
                              enhance_faces: bool,
                              quick_test: bool) -> dict:
    """Return {'frames': N, 'seconds': S, 'label': '...'} for the UI."""
    if not face_paths:
        return {"frames": 0, "seconds": 0,
                "label": "drop a face clip to see ETA"}
    s_per_frame = _learned_s_per_frame()
    fps = _DEFAULT_FPS
    try:
        clip_durs = [probe_duration_seconds(p) for p in face_paths]
    except Exception:
        return {"frames": 0, "seconds": 0,
                "label": "could not probe clip durations"}

    if quick_test:
        total_dur_s = min(12.0, sum(clip_durs))
    elif len(face_paths) > 1:
        total_dur_s = sum(clip_durs)
        if audio_path:
            try:
                a_dur = probe_duration_seconds(audio_path)
                if a_dur > total_dur_s:
                    total_dur_s = a_dur
            except Exception:
                pass
    elif extend_single and audio_path:
        try:
            a_dur = probe_duration_seconds(audio_path)
            total_dur_s = max(clip_durs[0], a_dur)
        except Exception:
            total_dur_s = clip_durs[0]
    else:
        total_dur_s = clip_durs[0]

    frames = int(total_dur_s * fps)
    seconds = int(frames * s_per_frame)
    if enhance_faces:
        seconds = int(seconds * 1.15)
    mins, secs = divmod(seconds, 60)
    label = (f"~{frames} frames · ETA ~{mins}m {secs}s "
             f"(at {s_per_frame:.2f} s/frame; "
             f"refines after each completed render)")
    return {"frames": frames, "seconds": seconds, "label": label}


# ---- SIDECAR JSON ----------------------------------------------
def sidecar_path_for(mp4_path: Path) -> Path:
    return Path(mp4_path).with_suffix(".job.json")


def _maybe_asdict(obj):
    """asdict() if it's a dataclass, else None.  Used so sidecar
    capture survives older LipsyncJob shapes that may not have
    every optional sub-config attached."""
    try:
        return asdict(obj)
    except Exception:
        return None


def write_sidecar(job: LipsyncJob, mp4_path: Path,
                   elapsed_s: float, frames: int) -> Path:
    """Write a JSON describing the lipsync job that produced
    mp4_path.  Any past render can then be exactly reproduced via
    load_sidecar() + History tab "Restore settings"."""
    sp = sidecar_path_for(mp4_path)
    blob = {
        "version": "v2",
        "kind": "lipsync",
        "face_paths": [str(p) for p in job.face_paths],
        "audio_path": str(job.audio_path),
        "isolate_vocals": job.isolate_vocals,
        "enhance_faces": job.enhance_faces,
        "quick_test": job.quick_test,
        "extend_single": job.extend_single,
        "latentsync": asdict(job.latentsync),
        "voice_swap": asdict(job.voice_swap),
        "watermark": _maybe_asdict(getattr(job, "watermark", None)),
        "aspect": _maybe_asdict(getattr(job, "aspect_ratio", None)),
        "maskout": _maybe_asdict(getattr(job, "mask_out", None)),
        "occlusion": _maybe_asdict(getattr(job, "occlusion", None)),
        "elapsed_s": float(elapsed_s),
        "frames": int(frames),
        "s_per_frame": (float(elapsed_s) / max(int(frames), 1)),
    }
    sp.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return sp


def write_video_swap_sidecar(job, mp4_path: Path,
                                elapsed_s: float) -> Path:
    """Sidecar for Face Swap renders (kind='face_swap').  Mirrors the
    lipsync write_sidecar so the History tab "Restore settings"
    button can populate either tab from one button.
    """
    sp = sidecar_path_for(mp4_path)
    blob = {
        "version": "v2",
        "kind": "face_swap",
        "source_image": str(getattr(job, "source_image", "")),
        "target_video": str(getattr(job, "target_video", "")),
        "blend_method": getattr(job, "blend_method", "poisson"),
        "enhance_faces": bool(getattr(job, "enhance_faces", False)),
        "det_threshold": float(getattr(job, "det_threshold", 0.5)),
        "output_quality": getattr(job, "output_quality",
                                    "visually_lossless"),
        "trim_start_frame": int(getattr(job, "trim_start_frame", 0)),
        "trim_end_frame": int(getattr(job, "trim_end_frame", 0)),
        "selector_mode": getattr(job, "selector_mode", "largest"),
        "reference_face_image": str(
            getattr(job, "reference_face_image", "") or ""),
        "reference_distance": float(
            getattr(job, "reference_distance", 0.6)),
        "mask_padding": int(getattr(job, "mask_padding", 0)),
        "mask_blur": float(getattr(job, "mask_blur", 1.0)),
        "swap_strength": float(getattr(job, "swap_strength", 1.0)),
        "enhancer_blend": float(getattr(job, "enhancer_blend", 1.0)),
        "pixel_boost": int(getattr(job, "pixel_boost", 128)),
        "temporal_enabled": bool(
            getattr(job, "temporal_enabled", True)),
        "temporal_ema_decay": float(
            getattr(job, "temporal_ema_decay", 0.85)),
        "temporal_buffer_size": int(
            getattr(job, "temporal_buffer_size", 5)),
        "color_transfer_mode": getattr(job, "color_transfer_mode",
                                          "reinhard"),
        "shadow_correction": bool(
            getattr(job, "shadow_correction", True)),
        "shadow_clamp_min": float(
            getattr(job, "shadow_clamp_min", 0.5)),
        "shadow_clamp_max": float(
            getattr(job, "shadow_clamp_max", 1.5)),
        "source_image_b": str(
            getattr(job, "source_image_b", "") or ""),
        "blend_alpha": float(getattr(job, "blend_alpha", 0.5)),
        "journey_mode": bool(getattr(job, "journey_mode", False)),
        "journey_start_alpha": float(
            getattr(job, "journey_start_alpha", 0.0)),
        "journey_end_alpha": float(
            getattr(job, "journey_end_alpha", 1.0)),
        "journey_curve": getattr(job, "journey_curve", "linear"),
        "elapsed_s": float(elapsed_s),
    }
    sp.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return sp


def load_sidecar(mp4_path: Path) -> Optional[dict]:
    sp = sidecar_path_for(mp4_path)
    if not sp.is_file():
        return None
    try:
        return json.loads(sp.read_text(encoding="utf-8"))
    except Exception:
        return None
