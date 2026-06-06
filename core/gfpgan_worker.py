"""In-memory GFPGAN face-enhance worker.

The legacy path (``v2/faceswap/gfpgan.py`` -> ``core/lipsync._restore_faces``)
constructs a fresh ``GFPGANer`` on every call, which re-reads the ~333 MB
``GFPGANv1.4.pth`` plus the detector / parser weights from disk and re-uploads
them to the GPU. Across a batch of renders that adds many seconds (and a lot
of host->device traffic) for no benefit -- the weights never change.

This module hoists GFPGAN initialisation to module scope and reuses a single
``GFPGANer`` instance across every ``enhance()`` call. The first call pays the
load cost; subsequent calls reuse the warm model. The public API mirrors
``faceswap/gfpgan.py`` exactly (``enhance(video_path: Path, log=print) -> Path``)
so an orchestrator change is a one-line import swap.

Threading note: ``GFPGANer.enhance`` is not thread-safe (face_helper holds
per-call state). Calls are serialised behind a module lock; concurrent renders
will queue rather than corrupt each other's state.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Path / constant resolution
# ---------------------------------------------------------------------------
# v2/core/gfpgan_worker.py -> parents[1] = v2/
_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[1]

# Ensure the v2/ project root is importable so this module can be loaded from
# anywhere (tests, ad-hoc scripts) without configuring sys.path first.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# GFPGAN's facexlib helper looks for detection/parsing weights relative to the
# current working directory under ``gfpgan/weights/``. We point it at the v2/
# tree so the existing detection_Resnet50_Final.pth + parsing_parsenet.pth get
# picked up rather than re-downloaded into a random cwd.
_GFPGAN_WEIGHTS_DIR = PROJECT_ROOT / "gfpgan" / "weights"

# Primary GFPGAN checkpoint. We probe a few known locations rather than
# hard-coding so this works whether the user keeps weights inside v2/ (new
# layout) or in the legacy lipsync_test/models/ scratch dir.
_GFPGAN_MODEL_CANDIDATES = (
    PROJECT_ROOT / "gfpgan" / "weights" / "GFPGANv1.4.pth",
    PROJECT_ROOT / "models" / "GFPGANv1.4.pth",
    PROJECT_ROOT / "lipsync_test" / "models" / "GFPGANv1.4.pth",
    PROJECT_ROOT.parent / "lipsync_test" / "models" / "GFPGANv1.4.pth",
)

# Scratch dir for enhanced mp4s. Lives under v2/ so it gets wiped with the
# rest of the work tree.
_OUT_DIR = PROJECT_ROOT / "lipsync_test" / "output"


def _resolve_model_path() -> Path:
    """Pick the first existing GFPGAN checkpoint. Returns the *expected*
    path (first candidate) if none exist yet -- the caller decides whether
    to surface that as an error at enhance() time, not at import time."""
    for p in _GFPGAN_MODEL_CANDIDATES:
        if p.exists():
            return p
    return _GFPGAN_MODEL_CANDIDATES[0]


# ---------------------------------------------------------------------------
# Module-level singletons -- the whole point of this file.
# ---------------------------------------------------------------------------
# These are populated lazily on the first enhance() call and then reused for
# the lifetime of the Python process. Never reassigned by enhance() once set,
# so concurrent reads (under _LOAD_LOCK on init) are safe.
_RESTORER = None              # type: Optional[object]   # gfpgan.GFPGANer
_EAGER_GFPGAN_MODEL = None    # raw torch model, for compile-fallback parity
_FFMPEG_BIN: Optional[str] = None

_LOAD_LOCK = threading.Lock()    # guards one-time init
_ENHANCE_LOCK = threading.Lock()  # serialises calls (face_helper is stateful)


def _find_ffmpeg() -> str:
    """Locate the ffmpeg binary. Prefers the system one; falls back to
    imageio-ffmpeg's bundled binary so we don't depend on PATH being set."""
    # 1. system ffmpeg
    for cand in ("ffmpeg", "ffmpeg.exe"):
        try:
            r = subprocess.run([cand, "-version"], capture_output=True,
                               text=True, timeout=5)
            if r.returncode == 0:
                return cand
        except Exception:
            pass
    # 2. imageio-ffmpeg fallback
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    # 3. give up; let the caller fail with a clear ffmpeg error
    return "ffmpeg"


def _ensure_loaded(log: Callable[[str], None]) -> None:
    """Initialise the singletons. Idempotent -- a fast no-op on every call
    after the first."""
    global _RESTORER, _EAGER_GFPGAN_MODEL, _FFMPEG_BIN
    if _RESTORER is not None:
        return
    with _LOAD_LOCK:
        if _RESTORER is not None:
            return  # double-checked locking: another thread won the race
        # facexlib looks at cwd for ``gfpgan/weights/`` -- if we're running
        # from somewhere else, cd briefly to PROJECT_ROOT for the init so it
        # finds the bundled detector / parser weights instead of redownloading.
        _GFPGAN_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        _OUT_DIR.mkdir(parents=True, exist_ok=True)

        model_path = _resolve_model_path()
        if not model_path.exists():
            raise FileNotFoundError(
                f"GFPGAN checkpoint not found. Looked in: "
                + ", ".join(str(c) for c in _GFPGAN_MODEL_CANDIDATES))

        log(f"[gfpgan_worker] loading GFPGAN (one-time) from {model_path}")
        from gfpgan import GFPGANer  # noqa: WPS433 -- intentional lazy import

        prev_cwd = os.getcwd()
        try:
            os.chdir(str(PROJECT_ROOT))
            restorer = GFPGANer(model_path=str(model_path),
                                upscale=1,
                                arch="clean",
                                channel_multiplier=2,
                                bg_upsampler=None)
        finally:
            try:
                os.chdir(prev_cwd)
            except Exception:
                pass

        # Stash the eager model so callers (or future torch.compile work)
        # can revert if a compiled path explodes mid-render.
        try:
            _EAGER_GFPGAN_MODEL = restorer.gfpgan
        except Exception:
            _EAGER_GFPGAN_MODEL = None

        # Best-effort CUDA log -- helpful when debugging "is it actually on
        # the GPU?" questions. GFPGAN picks CUDA by default if available.
        try:
            import torch
            if torch.cuda.is_available():
                log(f"[gfpgan_worker] CUDA available -- "
                    f"device={torch.cuda.get_device_name(0)}")
            else:
                log("[gfpgan_worker] CUDA NOT available -- running on CPU "
                    "(per-frame enhance will be slow)")
        except Exception:
            pass

        _FFMPEG_BIN = _find_ffmpeg()
        _RESTORER = restorer  # publish last -- readers see a fully-initialised obj
        log("[gfpgan_worker] GFPGAN ready (singleton; reused on subsequent calls)")


def _enhance_frame(img):
    """Restore one BGR frame with the cached restorer. Returns the restored
    frame or ``None`` on failure (so the caller can fall back to the original
    frame instead of dropping it)."""
    try:
        return _RESTORER.enhance(img,
                                 has_aligned=False,
                                 only_center_face=False,
                                 paste_back=True)[2]
    except Exception:
        return None


def enhance(video_path: Path,
            log: Callable[[str], None] = print) -> Path:
    """Run GFPGAN over every frame of ``video_path``. Returns the path to
    a new mp4 with enhanced frames; on a soft failure returns ``video_path``
    unchanged so the pipeline can still ship something to the user.

    The GFPGAN model is loaded once per process and reused. The output mp4
    has NO audio -- callers are expected to mux the original audio back in
    afterwards (matches the contract of the legacy ``_restore_faces`` helper).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        log(f"[gfpgan_worker] input not found: {video_path}; returning as-is")
        return video_path

    try:
        _ensure_loaded(log)
    except Exception as exc:
        log(f"[gfpgan_worker] init failed ({exc}); returning un-enhanced video")
        return video_path

    try:
        import cv2  # noqa: WPS433 -- lazy so the module imports without opencv
    except Exception as exc:
        log(f"[gfpgan_worker] opencv not importable ({exc}); skipping")
        return video_path

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log(f"[gfpgan_worker] could not open {video_path}; skipping")
        return video_path

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if w <= 0 or h <= 0:
        cap.release()
        log(f"[gfpgan_worker] bad dims {w}x{h}; skipping")
        return video_path

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / f"_restored_{video_path.stem}.mp4"

    enc = subprocess.Popen(
        [_FFMPEG_BIN or "ffmpeg", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-",
         "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-preset", "veryfast", "-crf", "18", str(out_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    log(f"[gfpgan_worker] restoring {total or '?'} frames -> {out_path.name}")
    n = 0
    with _ENHANCE_LOCK:  # face_helper is single-threaded; serialise renders
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                restored = _enhance_frame(frame)
                if restored is not None:
                    if restored.shape[1] != w or restored.shape[0] != h:
                        restored = cv2.resize(restored, (w, h))
                    frame = restored
                try:
                    enc.stdin.write(frame.astype("uint8").tobytes())
                except (BrokenPipeError, OSError):
                    break
                n += 1
                if n % 100 == 0:
                    log(f"  [gfpgan_worker] {n}/{total or '?'} frames")
        finally:
            cap.release()
            try:
                if enc.stdin:
                    enc.stdin.close()
            except Exception:
                pass
            try:
                enc.wait(timeout=120)
            except Exception:
                try:
                    enc.kill()
                except Exception:
                    pass

    if (n == 0
            or not out_path.exists()
            or out_path.stat().st_size < 10_000):
        log(f"[gfpgan_worker] produced empty/short output ({n} frames); "
            f"returning un-enhanced source")
        return video_path

    log(f"[gfpgan_worker] done -- {n} frames -> {out_path}")
    return out_path


# Small smoke hook -- prints the resolved model path without actually loading
# the network. Useful from a one-liner: ``python -m core.gfpgan_worker``.
if __name__ == "__main__":  # pragma: no cover -- diagnostic entry only
    print("PROJECT_ROOT     :", PROJECT_ROOT)
    print("model candidates :")
    for c in _GFPGAN_MODEL_CANDIDATES:
        print("  ", "[OK]" if c.exists() else "[--]", c)
    print("resolved model   :", _resolve_model_path())
    print("weights dir      :", _GFPGAN_WEIGHTS_DIR,
          "(exists)" if _GFPGAN_WEIGHTS_DIR.exists() else "(missing)")
    print("output dir       :", _OUT_DIR)
