"""RVC voice-clone pipeline: Demucs vocal isolation -> RVC conversion ->
remix with the instrumental stem.

Used by the Lip-Sync tab when a voice is picked from voice_models/.
Assumes the RVC repo, base models, and fairseq's PyTorch-2.6 patch
have already been set up by running rvc_clone_beta_test.py at least
once -- it is the canonical first-time installer.
"""
from __future__ import annotations
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT     = Path(__file__).resolve().parent.parent
# Voice models live at the user's voice_models/ folder. Same pattern as
# the LatentSync/RVC external repos: default to the peer location
# (../voice_models/, where v1 keeps them), env-overridable for a fully
# portable v2 install.
#   set FACESWAP_VOICE_MODELS=D:\my_voices
import os as _os
VOICE_MODELS_DIR = Path(_os.environ.get(
    "FACESWAP_VOICE_MODELS",
    str(PROJECT_ROOT.parent / "voice_models")))
# RVC + LatentSync are EXTERNAL repos -- they live peer to v2/, NOT
# inside v2/. The standalone refactor flipped PROJECT_ROOT from
# faceswap_pro/ to v2/, so a literal `PROJECT_ROOT / "lipsync_test"`
# now incorrectly resolves to v2/lipsync_test/ (doesn't exist).
# Default to the peer location and let env var FACESWAP_EXTERNAL_REPOS
# override for portable installs. Matches v2/faceswap/paths.py.
EXTERNAL_REPOS_ROOT = Path(_os.environ.get(
    "FACESWAP_EXTERNAL_REPOS",
    str(PROJECT_ROOT.parent / "lipsync_test")))
WORK             = EXTERNAL_REPOS_ROOT
RVC              = WORK / "RVC"
RVC_WEIGHTS      = RVC / "assets" / "weights"
# STEMS and OUT_DIR are v2's OWN work/output dirs (we write to them).
# Keep them inside v2/ even though the upstream repos are external.
STEMS            = PROJECT_ROOT / "lipsync_test" / "stems"
OUT_DIR          = PROJECT_ROOT / "recordings" / "voice"


def list_voice_models() -> list[str]:
    """Return basenames of .pth files in voice_models/, sorted. Used by
    the UI to populate the 'Swap voice' dropdown at tab-build time."""
    if not VOICE_MODELS_DIR.is_dir():
        return []
    return sorted(p.name for p in VOICE_MODELS_DIR.glob("*.pth"))


def _resolve_index(model_path: Path) -> str:
    """Find a matching .index sibling for `model_path`. Same-basename
    first; otherwise the first .index in the same dir; else empty."""
    same = model_path.with_suffix(".index")
    if same.exists():
        return str(same)
    cand = sorted(model_path.parent.glob("*.index"))
    return str(cand[0]) if cand else ""


def _isolate_vocals(song_path: str, log, ffmpeg: str = "") -> tuple[str, str]:
    """Demucs `song_path` (two-stems: vocals / no_vocals). Returns
    (vocals_wav, no_vocals_wav). Raises RuntimeError on failure.

    Env handling:
      - PYTHONIOENCODING=utf-8: Demucs prints the track path to stdout
        and Windows's default cp1252 codec dies on non-ASCII filenames
        (Arabic, CJK, etc.).
      - PATH: prepend a dir containing a bare `ffmpeg.exe`. Demucs's
        audio loader does `shutil.which('ffmpeg')`; imageio-ffmpeg's
        binary is named `ffmpeg-win-x64-v*.exe` which that lookup
        doesn't find. Mirror lipsync.py's _ffmpeg_bin_env helper.
    """
    STEMS.mkdir(parents=True, exist_ok=True)
    log(f"Demucs: isolating vocals from {os.path.basename(song_path)} ...")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if ffmpeg:
        try:
            from core.lipsync import _ffmpeg_bin_env
            bin_env = _ffmpeg_bin_env(ffmpeg, log=log)
            env["PATH"] = bin_env.get("PATH", env.get("PATH", ""))
        except Exception as exc:
            log(f"  WARN: could not expose ffmpeg on PATH for Demucs: {exc}")
    r = subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems", "vocals",
         "-o", str(STEMS), str(song_path)],
        env=env, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"Demucs failed (exit {r.returncode})")
    hits = sorted(STEMS.rglob("vocals.wav"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not hits:
        raise RuntimeError(f"Demucs produced no vocals.wav under {STEMS}")
    vocals = hits[0]
    no_vocals = vocals.parent / "no_vocals.wav"
    if not no_vocals.exists():
        raise RuntimeError(f"Demucs produced no no_vocals.wav alongside {vocals}")
    return str(vocals), str(no_vocals)


def _stage_voice_model(model_path: Path, log) -> tuple[str, str]:
    """Copy .pth into RVC/assets/weights/ (where infer_cli.py looks via
    --model_name); return (basename, .index path or empty string)."""
    RVC_WEIGHTS.mkdir(parents=True, exist_ok=True)
    staged = RVC_WEIGHTS / model_path.name
    if staged.resolve() != model_path.resolve():
        log(f"staging voice model into assets/weights/{model_path.name}")
        shutil.copy2(model_path, staged)
    return model_path.name, _resolve_index(model_path)


def rvc_convert_song(
    song_path: str,
    voice_model_basename: str,
    ffmpeg: str,
    *,
    transpose: int = 0,
    f0_method: str = "rmvpe",
    index_rate: float = 0.75,
    protect: float = 0.33,
    filter_radius: int = 3,
    rms_mix_rate: float = 0.25,
    resample_sr: int = 0,
    is_half: bool = True,
    log=print,
) -> tuple[str, str]:
    """Full voice-swap pipeline. Demucs the song's vocal stem, RVC-
    convert it to the target voice, remix with the instrumental stem.
    Returns (remix_path, dry_vocal_path).

    Assumes the RVC repo, base models, and fairseq patch are already
    set up (by running rvc_clone_beta_test.py at least once).
    Raises RuntimeError on failure.
    """
    if not (RVC / "tools" / "infer_cli.py").exists():
        raise RuntimeError(
            f"RVC repo not set up at {RVC}. Run rvc_clone_beta_test.py "
            "once first -- it does the clone, deps install, base model "
            "download, and fairseq patch.")

    model_full = VOICE_MODELS_DIR / voice_model_basename
    if not model_full.exists():
        raise RuntimeError(f"voice model not found: {model_full}")

    # Stage the input song to an ASCII-only path. Demucs / ffmpeg-python /
    # RVC's internal path passing all interact badly with non-ASCII
    # characters under Windows's default cp1252 stdout/argv codec --
    # the same gap that just bit Arabic / CJK source filenames coming
    # out of Gradio's tmp upload dir. Using a fixed safe name from here
    # on sidesteps every such failure regardless of what the user
    # uploaded.
    stage_dir = OUT_DIR / "_input_staging"
    stage_dir.mkdir(parents=True, exist_ok=True)
    ext = os.path.splitext(str(song_path))[1] or ".mp3"
    staged_song = str(stage_dir / f"_song_in{ext}")
    if os.path.abspath(str(song_path)) != os.path.abspath(staged_song):
        log(f"staging input audio to ASCII path: {os.path.basename(staged_song)}")
        shutil.copy2(song_path, staged_song)
    song_path = staged_song

    # 1. Demucs the song
    vocals_in, no_vocals = _isolate_vocals(song_path, log, ffmpeg=ffmpeg)

    # 2. Stage voice model into RVC's expected layout
    model_name, index_path = _stage_voice_model(model_full, log)
    log(f"voice model: {model_name} (index: "
        f"{os.path.basename(index_path) if index_path else 'none'})")

    # 3. RVC inference
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dry_path = OUT_DIR / f"rvc_{stamp}.wav"
    argv = [
        sys.executable, "tools/infer_cli.py",
        "--f0up_key",      str(int(transpose)),
        "--input_path",    vocals_in,
        "--index_path",    index_path,
        "--f0method",      str(f0_method),
        "--opt_path",      str(dry_path),
        "--model_name",    model_name,
        "--index_rate",    str(float(index_rate)),
        "--device",        "cuda:0",
        "--is_half",       "True" if is_half else "False",
        "--filter_radius", str(int(filter_radius)),
        "--resample_sr",   str(int(resample_sr)),
        "--rms_mix_rate",  str(float(rms_mix_rate)),
        "--protect",       str(float(protect)),
    ]
    # RVC's infer/lib/audio.py uses ffmpeg-python, which shells out to
    # a BARE `ffmpeg` command (not the long-named imageio binary). Use
    # core.lipsync's proven _ffmpeg_bin_env -- it stages a sibling
    # ffmpeg.exe next to imageio's binary and prepends that dir to
    # PATH, so the bare-name call resolves. Without this, RVC inference
    # crashes inside the subprocess with WinError 2.
    from core.lipsync import _ffmpeg_bin_env
    env = _ffmpeg_bin_env(ffmpeg, log)
    env["PYTHONIOENCODING"] = "utf-8"   # defensive: same Windows cp1252 trap
    log(f"RVC inference -> {dry_path.name} "
        f"(transpose={transpose}, f0_method={f0_method}, "
        f"index_rate={index_rate})")
    r = subprocess.run(argv, cwd=str(RVC), env=env)
    if r.returncode != 0 or not dry_path.exists() or dry_path.stat().st_size < 1000:
        raise RuntimeError(f"RVC inference failed (exit {r.returncode})")

    # 4. Remix cloned vocal + instrumental
    remix_path = OUT_DIR / f"rvc_{stamp}_remix.wav"
    log(f"remixing cloned vocal + instrumental -> {remix_path.name}")
    r = subprocess.run(
        [ffmpeg, "-y", "-i", str(dry_path), "-i", no_vocals,
         "-filter_complex",
         "[0:a]volume=1.0[v];[1:a]volume=1.0[i];"
         "[v][i]amix=inputs=2:duration=longest:normalize=0[out]",
         "-map", "[out]", "-c:a", "pcm_s16le", str(remix_path)],
        capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0 or not remix_path.exists():
        raise RuntimeError(
            "remix ffmpeg failed: " + (r.stderr or "")[-400:])
    return str(remix_path), str(dry_path)
