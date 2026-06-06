"""LatentSync 512px inference. Single responsibility: take a face
video + driver audio + knobs, return the raw lipsync mp4 path.

What this module does itself:
  * Stage inputs into a space-free scratch dir (LatentSync's
    internal ffmpeg shell-outs choke on spaces in paths)
  * Convert driver audio to 16 kHz mono PCM WAV (Whisper's
    expected input)
  * Build the argv and run the LatentSync subprocess

What this module BRIDGES from core/lipsync.py:
  * _ensure_latentsync()  -- repo clone + ckpt download (~5 GB)
  * _ffmpeg_bin_env()     -- expose bare-named ffmpeg.exe on PATH
                              for the subprocess (Windows quirk)
  * LS (path constant), LS_CONFIG, _run (streaming subprocess)

Re-implementing the bridged pieces would be ~600 lines of identical
plumbing.  When those legacy helpers eventually need replacing, only
this file needs touching -- nothing else in v2 imports them.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from .config import LatentSyncKnobs
from .ffmpeg_tools import resolve_ffmpeg
from .paths import LATENTSYNC_SCRATCH, PROJECT_ROOT, RECORDINGS_DIR

# Bridge to the legacy install/setup helpers
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from core.lipsync import (                        # noqa: E402
    _ensure_latentsync as _legacy_ensure,
    _ffmpeg_bin_env as _legacy_env,
    _run as _legacy_run,
    LS as _LS_DIR,
    LS_CONFIG as _LS_CONFIG,
)


def run(face_video: Path, audio_driver: Path,
        knobs: LatentSyncKnobs,
        log: Callable[[str], None] = print) -> Path:
    """Run one LatentSync inference. Returns the raw mp4 path.

    PERF: tries the in-process persistent worker first
    (v2/core/lipsync_worker.py) which holds the LatentSync pipeline
    in memory across renders -- saves 60-120s of model reload on
    every render after the first. On ANY exception, falls back to
    the legacy subprocess invocation so we never regress.
    """
    _legacy_ensure(log)
    LATENTSYNC_SCRATCH.mkdir(parents=True, exist_ok=True)
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    out_path = RECORDINGS_DIR / "_latentsync_raw_v2.mp4"
    from .paths import safe_clear_output
    out_path = safe_clear_output(out_path, log=log)

    # In-process worker is opt-in. Default OFF because it crashes with
    # "stack expects a non-empty TensorList" inside LatentSync's
    # pipeline on second-and-later renders (cause not fully diagnosed).
    # The subprocess path below is stable and matches v1 behavior.
    # To re-enable for debugging: set FACESWAP_USE_LIPSYNC_WORKER=1
    if os.environ.get("FACESWAP_USE_LIPSYNC_WORKER"):
        try:
            import time as _time
            from core import lipsync_worker
            _t0 = _time.perf_counter()
            was_loaded = lipsync_worker.is_loaded()
            log(f"[latentsync] trying in-process worker "
                f"(model {'CACHED' if was_loaded else 'will load'})")
            result = lipsync_worker.render(
                face_path=str(face_video),
                audio_path=str(audio_driver),
                out_path=str(out_path),
                inference_steps=int(knobs.inference_steps),
                guidance_scale=float(knobs.guidance_scale),
                enable_deepcache=bool(knobs.enable_deepcache),
                seed=int(knobs.seed),
            )
            _dt = _time.perf_counter() - _t0
            log(f"[latentsync] worker render OK in {_dt:.1f}s "
                f"(model was {'cached' if was_loaded else 'loaded fresh'})")
            return Path(result)
        except Exception as _worker_exc:
            import traceback as _tb
            log(f"[latentsync] worker FAILED ({_worker_exc}); "
                f"falling back to subprocess invocation")
            log("[latentsync] worker traceback (for diagnosis):")
            for _ln in _tb.format_exc().splitlines():
                log(f"[latentsync]   {_ln}")
            # Fall through to legacy subprocess path

    # 1. Stage inputs (space-free; LatentSync's internal ffmpeg cares)
    vid_ext = Path(face_video).suffix or ".mp4"
    staged_video = LATENTSYNC_SCRATCH / f"in_video{vid_ext}"
    staged_audio = LATENTSYNC_SCRATCH / "in_audio.wav"
    shutil.copy2(face_video, staged_video)

    ffmpeg = resolve_ffmpeg()
    r = subprocess.run(
        [ffmpeg, "-y", "-i", str(audio_driver),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
         str(staged_audio)],
        capture_output=True, text=True, encoding="utf-8", errors="ignore",
    )
    if r.returncode != 0 or not staged_audio.exists():
        raise RuntimeError(
            "could not prepare 16 kHz mono WAV for Whisper:\n"
            + (r.stderr or "")[-400:])

    # 2. Output path -- write to RECORDINGS so users see it
    out_path = RECORDINGS_DIR / "_latentsync_raw_v2.mp4"
    from .paths import safe_clear_output
    out_path = safe_clear_output(out_path, log=log)

    # 3. Build argv. Resolution stays fixed at 512 (the only quality
    # we deemed usable in production testing).
    #
    # Per-clip identity fine-tune: if the user has run Phase 2 training
    # on this source clip, swap the base UNet checkpoint for the
    # fine-tuned one. Identity is baked into the model -> face drift
    # against source pose should be greatly reduced.
    try:
        from core.lipsync_finetune import get_finetune_checkpoint
        _ft_ckpt = get_finetune_checkpoint(str(face_video))
    except Exception:
        _ft_ckpt = None
    if _ft_ckpt is not None:
        _inference_ckpt = str(_ft_ckpt)
        log(f"[latentsync] using FINE-TUNED checkpoint for this clip: "
            f"{_ft_ckpt.name}")
    else:
        _inference_ckpt = os.path.join("checkpoints", "latentsync_unet.pt")
    argv = [
        sys.executable, "-m", "scripts.inference",
        "--unet_config_path",    _LS_CONFIG,
        "--inference_ckpt_path", _inference_ckpt,
        "--inference_steps",     str(int(knobs.inference_steps)),
        "--guidance_scale",      str(float(knobs.guidance_scale)),
        "--seed",                str(int(knobs.seed)),
    ]
    if knobs.enable_deepcache:
        argv.append("--enable_deepcache")
    argv += [
        "--video_path",          str(staged_video),
        "--audio_path",          str(staged_audio),
        "--video_out_path",      str(out_path),
    ]

    seed_label = "random" if int(knobs.seed) == -1 else f"fixed {knobs.seed}"
    log(f"[latentsync] 512x512, steps={knobs.inference_steps}, "
        f"guidance={knobs.guidance_scale:.1f}, seed={seed_label}, "
        f"deepcache={'on' if knobs.enable_deepcache else 'off'}")

    env = _legacy_env(ffmpeg, log)
    # Tighten LatentSync's face detector confidence cutoff. Default 0.5
    # = upstream behaviour. Raising filters out marginal cat/animal
    # detections so the largest-face selector inside the detector
    # consistently picks the human.
    env["LATENTSYNC_FACE_DET_THRESHOLD"] = str(float(
        getattr(knobs, "face_det_threshold", 0.5)))
    log(f"[latentsync] face_det_threshold={knobs.face_det_threshold:.2f}")
    _legacy_run(argv, cwd=_LS_DIR, log=log, env=env)

    if not out_path.exists() or out_path.stat().st_size < 100_000:
        raise RuntimeError("LatentSync produced no output -- check the log")
    return out_path
