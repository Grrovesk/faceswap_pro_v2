"""
core/lipsync.py -- offline audio-driven lip-sync (Wav2Lip + Demucs).

Validated pipeline (proven via lipsync_beta_test.py):

    face video + song
      -> Demucs isolates the vocal stem
      -> Wav2Lip drives the mouth from the clean vocal stem
      -> the full song is muxed back as the output audio track

This is an OFFLINE render -- it is NOT real-time and does not touch the
live webcam transport. The Lip-Sync tab in ui/app.py calls render_lipsync().

Models / assets live under  lipsync_test/  next to the project root:
    lipsync_test/Wav2Lip/                 cloned inference code
    lipsync_test/Wav2Lip/checkpoints/     wav2lip_gan.pth  (~436 MB)
    lipsync_test/stems/                   Demucs vocal stems
    lipsync_test/output/                  scratch / intermediates
    recordings/lipsync/                   final rendered results
"""
from __future__ import annotations

import glob
import os
import re
import ssl
import subprocess
import sys
import time
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK     = os.path.join(PROJECT_ROOT, "lipsync_test")
W2L      = os.path.join(WORK, "Wav2Lip")
CKPT_DIR = os.path.join(W2L, "checkpoints")
OUT_DIR  = os.path.join(WORK, "output")          # scratch / intermediates
RENDER_DIR = os.path.join(PROJECT_ROOT, "recordings", "lipsync")  # final renders
WAV2LIP_REPO = "https://github.com/Rudrabha/Wav2Lip"
MIN_MODEL_BYTES = 50_000_000  # a real .pth is ~436 MB; smaller == error page

MODEL_URLS = {
    "wav2lip_gan.pth": [
        "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth",
        "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth",
        "https://huggingface.co/Non-playing-Character/Wave2lip/resolve/main/wav2lip_gan.pth",
    ],
    "wav2lip.pth": [
        "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip.pth",
        "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip.pth",
    ],
}

# GFPGAN face-restoration model -- re-synthesizes high-frequency detail that
# Wav2Lip's fixed 96x96 generation lacks (the soft look on large faces).
GFPGAN_MODEL_URL = ("https://github.com/TencentARC/GFPGAN/releases/download/"
                    "v1.3.4/GFPGANv1.4.pth")
GFPGAN_MODEL = os.path.join(WORK, "models", "GFPGANv1.4.pth")

# torch.compile the GFPGAN model for a faster per-frame restore. The
# 512x512 face crop is a fixed shape, so the graph compiles once and
# the cost amortises over every frame. Set False to force the eager
# model; the restore loop also self-reverts to eager if a compiled
# call fails (e.g. inductor/Triton unavailable on Windows).
GFPGAN_TORCH_COMPILE = False   # torch.compile experiment OFF -- reverts GFPGAN to the eager path


# -- Diff2Lip: an alternative diffusion-based mouth-sync engine --
# Diff2Lip (WACV 2024) edits the mouth on a 128px crop with an audio-
# conditioned diffusion model -- a sharper mouth and no hard "box" seam
# vs Wav2Lip's 96px GAN. Validated standalone via diff2lip_beta_test.py.
D2L = os.path.join(WORK, "Diff2Lip")
D2L_CKPT = os.path.join(D2L, "checkpoints", "checkpoint.pt")
DIFF2LIP_REPO = "https://github.com/soumik-kanad/diff2lip"
DIFF2LIP_CKPT_URL = ("https://huggingface.co/ameerazam08/diff2lip/"
                     "resolve/main/checkpoints/checkpoint.pt")


# -- LatentSync: ByteDance's audio-conditioned latent-diffusion lip-sync ----
# LatentSync 1.6 (Jun 2025) runs at 512x512 native -- the direct fix for
# Diff2Lip's 128px soft-mouth ceiling. Stable Diffusion 1.5 backbone with
# Whisper audio embeddings cross-attended into the U-Net. ~18 GB VRAM at
# 512; comfortable on the A6000. Repo + checkpoints come from HuggingFace
# ByteDance/LatentSync-1.6. Validated standalone via
# latentsync_beta_test.py before this integration.
LS = os.path.join(WORK, "LatentSync")
LATENTSYNC_REPO = "https://github.com/bytedance/LatentSync"
LATENTSYNC_ZIP_URL = ("https://github.com/bytedance/LatentSync/"
                      "archive/refs/heads/main.zip")
LATENTSYNC_HF_REPO = "ByteDance/LatentSync-1.6"
LS_UNET_CKPT = os.path.join(LS, "checkpoints", "latentsync_unet.pt")
LS_WHISPER_CKPT = os.path.join(LS, "checkpoints", "whisper", "tiny.pt")
LS_CONFIG = os.path.join("configs", "unet", "stage2_512.yaml")  # relative to LS cwd
LS_MIN_UNET_BYTES = 1_000_000_000   # real latentsync_unet.pt is ~5 GB


# MPI-free single-GPU replacement for guided_diffusion/dist_util.py
# (the stock file imports mpi4py, absent on a normal Windows env).
_D2L_DIST_UTIL_SRC = '''"""Single-GPU, MPI-free dist_util -- patched by core/lipsync.py."""
import io
import torch as th


def setup_dist():
    pass


def dev():
    return th.device("cuda" if th.cuda.is_available() else "cpu")


def load_state_dict(path, **kwargs):
    with open(path, "rb") as f:
        data = f.read()
    try:
        return th.load(io.BytesIO(data), **kwargs)
    except Exception:
        kw = dict(kwargs)
        kw["weights_only"] = False
        return th.load(io.BytesIO(data), **kw)


def sync_params(params):
    pass
'''


def checkpoint_path(checkpoint: str = "wav2lip_gan.pth") -> str:
    return os.path.join(CKPT_DIR, checkpoint)


def lipsync_assets_status(checkpoint: str = "wav2lip_gan.pth"):
    """Report whether the lip-sync model is present. Used by the Models tab.

    Returns a dict: {present, path, size_mb, repo_present}.
    """
    p = checkpoint_path(checkpoint)
    present = os.path.exists(p) and os.path.getsize(p) >= MIN_MODEL_BYTES
    return {
        "present": present,
        "path": p,
        "size_mb": (os.path.getsize(p) // 1048576) if os.path.exists(p) else 0,
        "repo_present": os.path.exists(os.path.join(W2L, "inference.py")),
    }


class _StreamedResult:
    """Drop-in replacement for subprocess.CompletedProcess that preserves
    the (.returncode, .stdout, .stderr) attribute interface used by
    callers in this module. stderr is empty because we merge stderr into
    stdout to keep ordering correct in the live stream."""
    __slots__ = ("returncode", "stdout", "stderr")
    def __init__(self, returncode, stdout, stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _run(cmd, cwd=None, log=print, check=True, env=None):
    """Stream subprocess output to `log()` so the terminal shows progress
    live (LatentSync's per-frame tqdm bar, Demucs's track progress,
    GFPGAN's frame counter, etc.) instead of disappearing into a captured
    buffer until the process exits.

    Two important details that v1 missed:
      1. tqdm progress bars use \r (carriage return), not \n.  A naive
         line-iterator (for line in proc.stdout) blocks until \n which
         never comes for a tqdm bar -- the user sees nothing for the
         entire render.  We read in small chunks and split on BOTH
         \r and \n so progress-bar updates flush in real time.
      2. PYTHONUNBUFFERED=1 in the child env forces Python subprocesses
         to line-buffer their stdout immediately instead of waiting for
         the OS's default 4 KB buffer to fill.

    Returns _StreamedResult with the same (.returncode, .stdout, .stderr)
    fields callers already use.  stderr is empty (merged into stdout so
    the live stream preserves the line order the user actually saw).
    """
    log("$ " + " ".join(str(c) for c in cmd))

    # Force the child to flush stdout per-line; otherwise tqdm / print()
    # buffer up to 4 KB before flushing and the user sees nothing.
    child_env = {**(env if env is not None else os.environ)}
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    # tqdm respects this and falls back to plain ascii bars (no ANSI
    # cursor magic) that play nicer with non-tty pipes.
    child_env.setdefault("TQDM_DISABLE_COLORS", "1")

    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=child_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=0, text=False,  # binary mode -- we handle decode + CR
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"could not launch '{cmd[0]}' -- not found ({e}); cwd={cwd}") from e

    buf_lines = []
    MAX_BUFFERED_LINES = 20000

    def _emit(line: str) -> None:
        ln = line.rstrip("\r\n")
        if not ln:
            return
        log(ln)
        if len(buf_lines) < MAX_BUFFERED_LINES:
            buf_lines.append(ln)
        elif len(buf_lines) == MAX_BUFFERED_LINES:
            buf_lines.append("... [output capped at "
                             f"{MAX_BUFFERED_LINES} lines for buffer]")

    try:
        # Read raw bytes in small chunks so a tqdm \r-update is visible
        # within milliseconds.  Maintain a rolling decoder for utf-8
        # split across chunk boundaries.
        import codecs
        decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
        pending = ""
        while True:
            chunk = proc.stdout.read(256)
            if not chunk:
                break
            text = decoder.decode(chunk)
            # Normalize Windows CRLF first, then split on either separator.
            text = pending + text.replace("\r\n", "\n")
            # Walk characters, emitting on each separator.
            start = 0
            for i, ch in enumerate(text):
                if ch == "\n" or ch == "\r":
                    _emit(text[start:i])
                    start = i + 1
            pending = text[start:]
        # Flush any unterminated tail.
        if pending.strip():
            _emit(pending)
    except KeyboardInterrupt:
        proc.kill()
        raise
    finally:
        proc.wait()

    aggregated = "\n".join(buf_lines)
    r = _StreamedResult(proc.returncode, aggregated, "")

    if r.returncode != 0 and check:
        tail = aggregated[-3000:].strip()
        raise RuntimeError(
            f"command failed (exit {r.returncode}):\n"
            f"  {' '.join(str(c) for c in cmd)}\n"
            f"--- output tail ---\n{tail}")
    return r


def _run_soft(cmd):
    try:
        return subprocess.run(cmd, capture_output=True).returncode == 0
    except Exception:
        return False


def _resolve_ffmpeg(log=print) -> str:
    """Absolute path to a working ffmpeg. The FaceSwap Pro app process is
    NOT guaranteed to have ffmpeg on PATH (a user terminal usually does;
    the app process often does not). Resolve it robustly:
    imageio-ffmpeg's bundled binary (pip-installed on demand) -> PATH ->
    common Windows locations. Raises if ffmpeg truly cannot be found."""
    import shutil
    # 1. imageio-ffmpeg ships a known-good ffmpeg build inside its wheel.
    mod = None
    try:
        import imageio_ffmpeg as mod
    except ImportError:
        log("ffmpeg not on PATH; installing imageio-ffmpeg (bundles ffmpeg) ...")
        _run_soft([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"])
        try:
            import imageio_ffmpeg as mod
        except ImportError:
            mod = None
    if mod is not None:
        try:
            exe = mod.get_ffmpeg_exe()
            if exe and os.path.exists(exe):
                return exe
        except Exception:
            pass
    # 2. PATH
    w = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if w:
        return w
    # 3. common Windows locations
    for cand in (r"C:\msys64\mingw64\bin\ffmpeg.exe",
                 r"C:\msys64\usr\bin\ffmpeg.exe",
                 r"C:\ffmpeg\bin\ffmpeg.exe",
                 os.path.join(PROJECT_ROOT, "ffmpeg.exe"),
                 os.path.join(PROJECT_ROOT, "bin", "ffmpeg.exe")):
        if os.path.exists(cand):
            return cand
    raise RuntimeError(
        "ffmpeg could not be located. Install it with:  "
        "pip install imageio-ffmpeg   (or put ffmpeg.exe on PATH).")


def _env_with_ffmpeg(ffmpeg_exe: str) -> dict:
    """os.environ copy with ffmpeg's directory on PATH -- so child processes
    that shell out to a bare `ffmpeg` (e.g. Wav2Lip's inference.py) find it."""
    env = os.environ.copy()
    d = os.path.dirname(os.path.abspath(ffmpeg_exe))
    if d and os.path.isdir(d):
        env["PATH"] = d + os.pathsep + env.get("PATH", "")
    return env

def _ffmpeg_bin_env(ffmpeg_exe: str, log=print) -> dict:
    """Child env where a BARE `ffmpeg` command resolves. Diff2Lip's
    generate.py shells out to a bare `ffmpeg` (subprocess, shell=True);
    the app process often has no ffmpeg on PATH, and the resolved
    binary may be imageio-ffmpeg's `ffmpeg-win-*.exe` -- the wrong name
    for a bare call. Expose a correctly-named ffmpeg.exe on PATH."""
    import shutil
    env = os.environ.copy()
    ffmpeg_exe = os.path.abspath(ffmpeg_exe)
    src_dir = os.path.dirname(ffmpeg_exe)
    base = os.path.basename(ffmpeg_exe).lower()
    if base in ("ffmpeg", "ffmpeg.exe"):
        bin_dir = src_dir
    else:
        bin_dir = os.path.join(OUT_DIR, "_ffmpegbin")
        os.makedirs(bin_dir, exist_ok=True)
        target = os.path.join(bin_dir, "ffmpeg.exe")
        try:
            stale = (not os.path.exists(target)
                     or os.path.getsize(target) != os.path.getsize(ffmpeg_exe))
        except OSError:
            stale = True
        if stale:
            log(f"exposing ffmpeg as ffmpeg.exe for Diff2Lip -> {target}")
            shutil.copy2(ffmpeg_exe, target)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    return env


def _download(urls, dest, log=print):
    if os.path.exists(dest) and os.path.getsize(dest) >= MIN_MODEL_BYTES:
        log(f"model already present ({os.path.getsize(dest) // 1048576} MB)")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    ctx = ssl.create_default_context()
    for url in urls:
        try:
            log(f"downloading {os.path.basename(dest)} <- {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=120) as r, \
                    open(dest, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if os.path.getsize(dest) >= MIN_MODEL_BYTES:
                log(f"saved -> {dest}")
                return
        except Exception as e:
            log(f"  mirror failed: {e}")
        if os.path.exists(dest):
            os.remove(dest)
    raise RuntimeError(f"could not download {os.path.basename(dest)} from any mirror")


def _patch_wav2lip(log=print):
    """Wav2Lip patches: librosa/numpy modernization, plus a feathered
    paste-back so the regenerated face blends in instead of showing a box."""
    audio_py = os.path.join(W2L, "audio.py")
    if os.path.exists(audio_py):
        s = open(audio_py, encoding="utf-8").read()
        fixed = s.replace("librosa.filters.mel(hp.sample_rate, hp.n_fft,",
                          "librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,")
        if fixed != s:
            open(audio_py, "w", encoding="utf-8").write(fixed)
            log("patched audio.py (librosa >= 0.10)")
    for root, _, files in os.walk(W2L):
        if os.sep + ".git" in root:
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            fp = os.path.join(root, fn)
            try:
                s = open(fp, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            fixed = re.sub(r"\bnp\.float\b", "float", s)
            fixed = re.sub(r"\bnp\.int\b", "int", fixed)
            if fixed != s:
                open(fp, "w", encoding="utf-8").write(fixed)
                log(f"patched {os.path.relpath(fp, W2L)} (numpy >= 1.24)")

    # (c) feathered paste-back. Wav2Lip pastes its regenerated face crop as a
    #     HARD rectangle (`f[y1:y2, x1:x2] = p`) with no blending -- that shows
    #     as a visible box on clean / high-fidelity source video. Replace it
    #     with a soft feathered alpha blend so the patch fades into the face.
    inf_py = os.path.join(W2L, "inference.py")
    if os.path.exists(inf_py):
        s = open(inf_py, encoding="utf-8").read()
        anchor = "f[y1:y2, x1:x2] = p"
        if "FSP_FEATHER" not in s and anchor in s:
            new_s = s
            for line in s.splitlines():
                if line.strip() == anchor:
                    ind = line[:len(line) - len(line.lstrip())]
                    feather = "\n".join(ind + x for x in (
                        "# FSP_FEATHER: soft-edged blend, not a hard rectangle",
                        "_mh, _mw = y2 - y1, x2 - x1",
                        "_fr = max(1, min(int(min(_mh, _mw) * 0.18), "
                        "min(_mh, _mw) // 2 - 1))",
                        "_msk = np.zeros((_mh, _mw), np.float32)",
                        "_msk[_fr:_mh - _fr, _fr:_mw - _fr] = 1.0",
                        "_msk = cv2.GaussianBlur(_msk, (_fr * 2 + 1, "
                        "_fr * 2 + 1), 0)[:, :, None]",
                        "f[y1:y2, x1:x2] = (_msk * p.astype(np.float32) + "
                        "(1.0 - _msk) * f[y1:y2, x1:x2].astype(np.float32))"
                        ".astype(np.uint8)",
                    ))
                    new_s = s.replace(line, feather, 1)
                    break
            if new_s != s:
                open(inf_py, "w", encoding="utf-8").write(new_s)
                log("patched inference.py (feathered paste-back -- removes the box)")


def _ensure_demucs(log=print):
    try:
        import demucs  # noqa: F401
        return
    except Exception:
        pass
    log("installing demucs (first time only) ...")
    _run_soft([sys.executable, "-m", "pip", "install", "demucs"])
    try:
        import demucs  # noqa: F401
    except Exception:
        raise RuntimeError("demucs is not available; run: pip install demucs")


def _isolate_vocals(song_path, log=print):
    """Demucs two-stems separation -> path to the isolated vocals .wav.

    Two Windows-specific defences here:
      * Stage the input audio to a fixed ASCII filename inside stems_dir
        before invoking Demucs. Gradio's tmp upload dir can contain non-
        ASCII filenames (Arabic, CJK, etc.) and Windows's C runtime /
        cp1252 stdout encoder both choke on them at different layers.
      * Set PYTHONIOENCODING=utf-8 in the subprocess env so Demucs's own
        `print(f"Separating track {track}")` doesn't crash with
        UnicodeEncodeError on the very first status line.
    """
    _ensure_demucs(log)
    stems_dir = os.path.join(WORK, "stems")
    os.makedirs(stems_dir, exist_ok=True)

    # Stage the input audio to an ASCII path. Demucs / its torchaudio
    # backend / the Windows CRT fopen all interact badly with non-ASCII
    # characters in source filenames. Using a fixed safe name from here
    # on sidesteps every layer.
    stage_dir = os.path.join(WORK, "inputs", "_ascii_stage")
    os.makedirs(stage_dir, exist_ok=True)
    ext = os.path.splitext(str(song_path))[1] or ".mp3"
    staged_song = os.path.join(stage_dir, "_song_in" + ext)
    if os.path.abspath(str(song_path)) != os.path.abspath(staged_song):
        import shutil as _sh
        try:
            _sh.copy2(song_path, staged_song)
            log(f"staged input audio to ASCII path: {os.path.basename(staged_song)}")
            song_path = staged_song
        except Exception as _e:
            log(f"could not stage input audio to ASCII path ({_e}); "
                "continuing with original path -- expect a Demucs crash if "
                "the filename contains non-ASCII characters on Windows.")

    log(f"isolating vocal stem from {os.path.basename(song_path)} (Demucs) ...")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    # Force CUDA explicitly -- Demucs's auto-detection silently falls back
    # to CPU on some envs (mismatched torch/cuda packaging, etc.) and CPU
    # demucs on a 2:30 song is a 10+ minute time sink for no quality gain.
    # Falls back to CPU automatically if CUDA isn't actually available.
    _demucs_device = "cuda"
    try:
        import torch as _torch_probe
        if not _torch_probe.cuda.is_available():
            _demucs_device = "cpu"
            log("Demucs: CUDA not visible to torch -- falling back to CPU "
                "(this will be slow on long songs).")
    except Exception:
        pass
    _run([sys.executable, "-m", "demucs", "--two-stems", "vocals",
          "-d", _demucs_device,
          "-o", stems_dir, song_path], cwd=PROJECT_ROOT, log=log, env=env)
    hits = glob.glob(os.path.join(stems_dir, "**", "vocals.wav"), recursive=True)
    if not hits:
        raise RuntimeError("Demucs produced no vocals.wav")
    return max(hits, key=os.path.getmtime)


def _ensure_gfpgan_weights(log=print) -> str:
    """Make sure the GFPGANv1.4 weights file exists at GFPGAN_MODEL.

    Resolution order:
      1. v2 canonical path (GFPGAN_MODEL = v2/lipsync_test/models/GFPGANv1.4.pth)
      2. Legacy v1 sibling at faceswap_pro/lipsync_test/models/ -- copy
         from there if present (avoids a 350 MB redownload for users who
         already had v1 installed).
      3. Download from the official Tencent ARC release URL.

    Returns the resolved path (always GFPGAN_MODEL once this returns
    without raising).
    """
    if os.path.isfile(GFPGAN_MODEL) and os.path.getsize(GFPGAN_MODEL) > 10_000_000:
        return GFPGAN_MODEL
    os.makedirs(os.path.dirname(GFPGAN_MODEL), exist_ok=True)
    # Try legacy v1 location -- one level up from WORK
    legacy = os.path.join(os.path.dirname(WORK), "lipsync_test",
                          "models", "GFPGANv1.4.pth")
    if (os.path.isfile(legacy)
            and os.path.getsize(legacy) > 10_000_000):
        log(f"copying GFPGAN weights from legacy v1 location "
            f"{legacy} -> {GFPGAN_MODEL}")
        import shutil as _sh
        _sh.copy2(legacy, GFPGAN_MODEL)
        return GFPGAN_MODEL
    # Download from the official Tencent ARC release.
    url = ("https://github.com/TencentARC/GFPGAN/releases/download/"
           "v1.3.0/GFPGANv1.4.pth")
    log(f"downloading GFPGAN weights from {url} (~350 MB; first time only) ...")
    import urllib.request
    tmp = GFPGAN_MODEL + ".part"
    with urllib.request.urlopen(url, timeout=600) as resp, \
            open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, GFPGAN_MODEL)
    sz = os.path.getsize(GFPGAN_MODEL) / (1024 * 1024)
    log(f"GFPGAN weights cached at {GFPGAN_MODEL} ({sz:.1f} MB)")
    return GFPGAN_MODEL


def _ensure_gfpgan(log=print):
    """Make GFPGAN importable: install `gfpgan` if missing, patch
    `basicsr` for the torchvision >= 0.17 breakage (functional_tensor was
    removed), and ensure the GFPGANv1.4.pth weights file exists at
    GFPGAN_MODEL (download if missing, copy from v1 location if present)."""
    try:
        import gfpgan  # noqa: F401
    except ImportError:
        log("installing GFPGAN (gfpgan + basicsr + facexlib; first time only) ...")
        _run_soft([sys.executable, "-m", "pip", "install", "gfpgan"])
    # Patch basicsr without importing it (the bad import would fail) --
    # find_spec locates the package on disk.
    # Make sure the weights file actually exists (the pip install
    # gives us the Python code only; the .pth has to be downloaded
    # or copied separately).
    try:
        _ensure_gfpgan_weights(log=log)
    except Exception as _w_exc:
        log(f"WARNING: GFPGAN weights resolve failed ({_w_exc}); "
            f"face enhancement will be disabled this session")
    try:
        import importlib.util
        spec = importlib.util.find_spec("basicsr")
        if spec and spec.origin:
            deg = os.path.join(os.path.dirname(spec.origin),
                               "data", "degradations.py")
            if os.path.exists(deg):
                ds = open(deg, encoding="utf-8").read()
                fx = ds.replace(
                    "from torchvision.transforms.functional_tensor import "
                    "rgb_to_grayscale",
                    "from torchvision.transforms.functional import "
                    "rgb_to_grayscale")
                if fx != ds:
                    open(deg, "w", encoding="utf-8").write(fx)
                    log("patched basicsr/degradations.py (torchvision compat)")
    except Exception as e:
        log(f"  basicsr patch skipped: {e}")
    try:
        import gfpgan  # noqa: F401
    except Exception as e:
        raise RuntimeError(f"GFPGAN could not be imported after install: {e}")


def _restore_faces(video_in, ffmpeg, log=print):
    """GFPGAN face restoration over every frame of `video_in`. Adds back the
    high-frequency detail the 96/128 px lip-sync crop lacks -- the fix for
    the soft / boxy look on large close-up faces.

    Restored frames are PIPED straight to ffmpeg. cv2.VideoWriter is
    unreliable in this environment -- it failed to open and silently wrote a
    0-byte file. Returns the restored .mp4 (no audio; song muxed afterwards).

    GFPGAN_TORCH_COMPILE wraps the model in torch.compile. If the compile
    cannot build (common on Windows), the loop self-heals -- it reverts to
    the eager model on the first failure and finishes the render normally."""
    _ensure_gfpgan(log)
    import cv2
    from gfpgan import GFPGANer

    _download([GFPGAN_MODEL_URL], GFPGAN_MODEL, log=log)
    log("loading GFPGAN ...")
    restorer = GFPGANer(model_path=GFPGAN_MODEL, upscale=1, arch="clean",
                        channel_multiplier=2, bg_upsampler=None)

    # torch.compile the GFPGAN model. _eager_model is kept so the per-frame
    # loop can fall back if the compiled path fails on first use.
    _eager_model = restorer.gfpgan
    if GFPGAN_TORCH_COMPILE:
        try:
            import torch
            restorer.gfpgan = torch.compile(restorer.gfpgan)
            log("GFPGAN: torch.compile enabled (the first restored frame "
                "pays a one-time compile cost)")
        except Exception as exc:
            log(f"GFPGAN: torch.compile unavailable ({exc}); using eager model")

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video for restoration: {video_in}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError(f"restoration: bad video dimensions {w}x{h}")

    out_path = os.path.join(OUT_DIR, "_restored.mp4")
    enc = subprocess.Popen(
        [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-",
         "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-preset", "veryfast", "-crf", "18", out_path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    log(f"GFPGAN restoring {total or '?'} frames "
        f"(face-detail enhancement -- this is the slow step) ...")

    def _enhance(img):
        """Restore one frame. On a compiled-model failure, revert to the
        eager model once and retry -- so a torch.compile problem costs at
        most one frame instead of silently no-op-ing the whole render."""
        try:
            return restorer.enhance(img, has_aligned=False,
                                    only_center_face=False,
                                    paste_back=True)[2]
        except Exception as exc:
            if restorer.gfpgan is not _eager_model:
                log(f"  GFPGAN compiled path failed ({exc}); reverting to "
                    f"the eager model for the rest of the render")
                restorer.gfpgan = _eager_model
                try:
                    return restorer.enhance(img, has_aligned=False,
                                            only_center_face=False,
                                            paste_back=True)[2]
                except Exception:
                    return None
            return None

    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            restored = _enhance(frame)
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
                log(f"  ... GFPGAN {n}/{total or '?'} frames")
    finally:
        cap.release()
        try:
            enc.stdin.close()
        except Exception:
            pass
        enc.wait()
    if (n == 0 or not os.path.exists(out_path)
            or os.path.getsize(out_path) < 10000):
        raise RuntimeError(
            f"GFPGAN restoration produced an empty video ({n} frames written)")
    log(f"GFPGAN restoration done -- {n} frames -> {out_path}")
    return out_path


def _find_in(root, suffix):
    """First file under `root` whose path ends with `suffix`."""
    suffix = suffix.replace("/", os.sep)
    for dp, _, files in os.walk(root):
        if os.sep + ".git" in dp:
            continue
        for fn in files:
            full = os.path.join(dp, fn)
            if full.replace("/", os.sep).endswith(suffix):
                return full
    return None


def _ensure_diff2lip(log=print):
    """Clone Diff2Lip, fetch its checkpoint, install its extra deps, and
    apply the modern-Python patches (numpy >= 1.24, librosa >= 0.10, and
    an MPI-free dist_util -- the stock one imports mpi4py). Returns the
    directory to put on PYTHONPATH so `import guided_diffusion` resolves."""
    for _pkg in ("tqdm", "av"):
        try:
            __import__(_pkg)
        except Exception:
            log(f"installing {_pkg} (Diff2Lip dependency; first time only) ...")
            _run_soft([sys.executable, "-m", "pip", "install", _pkg])
    if not os.path.exists(os.path.join(D2L, "generate.py")):
        _run(["git", "clone", "--depth", "1", DIFF2LIP_REPO, D2L], log=log)
    _download([DIFF2LIP_CKPT_URL], D2L_CKPT, log=log)
    # (a) numpy >= 1.24 removed np.float / np.int / np.bool
    for root, _, files in os.walk(D2L):
        if os.sep + ".git" in root:
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            fp = os.path.join(root, fn)
            try:
                s = open(fp, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            fixed = re.sub(r"\bnp\.float\b", "float", s)
            fixed = re.sub(r"\bnp\.int\b", "int", fixed)
            fixed = re.sub(r"\bnp\.bool\b", "bool", fixed)
            if fixed != s:
                open(fp, "w", encoding="utf-8").write(fixed)
                log(f"patched {os.path.relpath(fp, D2L)} (numpy >= 1.24)")
    # (b) librosa >= 0.10 needs keyword args for filters.mel
    for cand in glob.glob(os.path.join(D2L, "audio", "*.py")):
        s = open(cand, encoding="utf-8", errors="ignore").read()
        fixed = s.replace(
            "librosa.filters.mel(hp.sample_rate, hp.n_fft,",
            "librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,")
        if fixed != s:
            open(cand, "w", encoding="utf-8").write(fixed)
            log(f"patched {os.path.relpath(cand, D2L)} (librosa >= 0.10)")
    # (c) MPI-free single-GPU dist_util (stock imports mpi4py)
    du = _find_in(D2L, "guided_diffusion/dist_util.py")
    if not du:
        raise RuntimeError(
            "Diff2Lip: guided_diffusion/dist_util.py not found in the clone")
    cur = open(du, encoding="utf-8", errors="ignore").read()
    if "MPI-free dist_util" not in cur:
        if not os.path.exists(du + ".orig"):
            import shutil as _sh
            _sh.copy2(du, du + ".orig")
        open(du, "w", encoding="utf-8").write(_D2L_DIST_UTIL_SRC)
        log("patched guided_diffusion/dist_util.py (MPI-free single-GPU)")
    # (d) feathered paste-back. Diff2Lip's generate.py pastes the regenerated
    #     face crop as a HARD rectangle (`v[y1:y2, x1:x2] = g`) -- that shows
    #     as a tone band at the bounding-box edge (ears / cheeks). Replace it
    #     with a soft feathered alpha blend -- the same fix applied to Wav2Lip.
    gen_py = os.path.join(D2L, "generate.py")
    if os.path.exists(gen_py):
        gs = open(gen_py, encoding="utf-8").read()
        _d2l_anchor = "v[y1:y2, x1:x2] = g"
        if "FSP_FEATHER" not in gs and _d2l_anchor in gs:
            new_gs = gs
            for _line in gs.splitlines():
                if _line.strip() == _d2l_anchor:
                    _ind = _line[:len(_line) - len(_line.lstrip())]
                    _feather = "\n".join(_ind + _x for _x in (
                        "# FSP_FEATHER: soft-edged blend, not a hard rectangle",
                        "_mh, _mw = y2 - y1, x2 - x1",
                        "_fr = max(1, min(int(min(_mh, _mw) * 0.18), "
                        "min(_mh, _mw) // 2 - 1))",
                        "_msk = np.zeros((_mh, _mw), np.float32)",
                        "_msk[_fr:_mh - _fr, _fr:_mw - _fr] = 1.0",
                        "_msk = cv2.GaussianBlur(_msk, (_fr * 2 + 1, "
                        "_fr * 2 + 1), 0)[:, :, None]",
                        "v[y1:y2, x1:x2] = (_msk * g.astype(np.float32) + "
                        "(1.0 - _msk) * v[y1:y2, x1:x2]"
                        ".astype(np.float32)).astype(np.uint8)",
                    ))
                    new_gs = gs.replace(_line, _feather, 1)
                    break
            if new_gs != gs:
                open(gen_py, "w", encoding="utf-8").write(new_gs)
                log("patched generate.py (feathered paste-back -- removes "
                    "the bbox edge band)")
    return os.path.dirname(os.path.dirname(du))


def _ensure_no_broken_bitsandbytes(log=print):
    """diffusers unconditionally imports bitsandbytes (via its quantizers
    module). On many venvs bnb is installed but BROKEN -- its module-level
    decorators call torch.compile, which needs a newer triton API than is
    typically installed alongside torch 2.6/cu126. The result: LatentSync
    inference crashes the moment it does `from diffusers import
    AutoencoderKL`, with 'cannot import name AttrsDescriptor from
    triton.compiler'.

    LatentSync does not use bitsandbytes quantization. The fix is to
    uninstall bnb so diffusers' missing-bnb branch (which IS guarded)
    runs. We probe in a subprocess so any import side-effects do not
    pollute this process, and only uninstall on the specific
    triton/dynamo failure signature -- so a working bnb is never
    removed.
    """
    try:
        import importlib.metadata
        importlib.metadata.version("bitsandbytes")
    except Exception:
        return  # bnb not installed -- diffusers handles missing bnb cleanly
    probe = subprocess.run(
        [sys.executable, "-c", "import bitsandbytes"],
        capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if probe.returncode == 0:
        return  # bnb works
    err = (probe.stderr or "") + (probe.stdout or "")
    triton_signature = (
        "AttrsDescriptor" in err
        or "triton.compiler" in err
        or "triton.backends" in err
        or "torch._dynamo" in err
    )
    if triton_signature:
        log("bitsandbytes is installed but BROKEN in this venv (triton API "
            "mismatch -- its decorators call torch.compile which needs a "
            "newer triton). LatentSync does not use bnb; uninstalling it so "
            "diffusers' missing-bnb branch is taken.")
        _run_soft([sys.executable, "-m", "pip", "uninstall", "-y", "bitsandbytes"])
    else:
        log("bitsandbytes failed to import for an unexpected reason -- "
            "continuing. If LatentSync inference dies on a diffusers "
            "import, run:  pip uninstall -y bitsandbytes")
        log(f"  (stderr tail): {err[-300:]}")


def _ensure_latentsync(log=print):
    """Clone (or zip-download) the LatentSync repo, install its extra
    Python deps, neutralise a broken bitsandbytes if present, and fetch
    the 1.6 checkpoints (latentsync_unet.pt + whisper/tiny.pt) from
    HuggingFace. Returns the clone directory.

    Robust to a venv without git: if `git` is not on PATH, downloads the
    GitHub main.zip and extracts it -- the gap that broke the prior
    integration attempt.
    """
    # 1. Fetch repo. git clone if git is on PATH, else download + extract zip.
    if not os.path.exists(os.path.join(LS, "scripts", "inference.py")):
        import shutil as _sh, zipfile as _zf
        git = _sh.which("git")
        if git:
            log(f"cloning LatentSync into {LS} (via git) ...")
            _run([git, "clone", "--depth", "1", LATENTSYNC_REPO, LS], log=log)
        else:
            log("git not on PATH -- falling back to ZIP download (no git needed).")
            os.makedirs(WORK, exist_ok=True)
            zip_path = os.path.join(WORK, "_latentsync_main.zip")
            _download([LATENTSYNC_ZIP_URL], zip_path, log=log)
            log(f"extracting {os.path.basename(zip_path)} into {WORK} ...")
            with _zf.ZipFile(zip_path, "r") as zf:
                zf.extractall(WORK)
            extracted = os.path.join(WORK, "LatentSync-main")
            if not os.path.exists(extracted):
                raise RuntimeError(
                    f"LatentSync ZIP extracted but {extracted} not found -- "
                    "the GitHub archive layout may have changed.")
            if os.path.exists(LS):
                _sh.rmtree(LS)
            os.rename(extracted, LS)
            try:
                os.remove(zip_path)
            except OSError:
                pass
        if not os.path.exists(os.path.join(LS, "scripts", "inference.py")):
            raise RuntimeError(
                f"LatentSync fetched but scripts/inference.py missing "
                f"under {LS}.")
        log(f"LatentSync repo ready at {LS}")

    # 2. Extra Python deps. We deliberately skip torch / torchvision --
    #    the user's venv has cu126 builds and LatentSync's pinned
    #    requirements would risk replacing them.
    #
    # Per-process cache: once we've verified a module imports cleanly
    # in this Python process, skip the check on subsequent renders.
    # Without this, every render re-runs __import__ for every dep
    # and (worse) the broad `except Exception` used to fire on any
    # import-time error -- including DLL load issues, optional
    # integration probes inside diffusers, etc. -- triggering a
    # spurious pip-install on every render.
    if not hasattr(_ensure_latentsync, "_deps_verified"):
        _ensure_latentsync._deps_verified = set()
    verified = _ensure_latentsync._deps_verified
    needed = [
        ("diffusers",      "diffusers"),
        ("transformers",   "transformers"),
        ("accelerate",     "accelerate"),
        ("omegaconf",      "omegaconf"),
        ("einops",         "einops"),
        ("decord",         "decord"),
        ("imageio_ffmpeg", "imageio-ffmpeg"),
        ("huggingface_hub","huggingface-hub"),
    ]
    for mod, pip_name in needed:
        if mod in verified:
            continue
        try:
            __import__(mod)
            verified.add(mod)
        except (ImportError, ModuleNotFoundError):
            log(f"installing {pip_name} (LatentSync dependency; not found) ...")
            _run_soft([sys.executable, "-m", "pip", "install", pip_name])
            # Re-attempt import; if it still fails, surface the real reason.
            try:
                __import__(mod)
                verified.add(mod)
            except Exception as _post_exc:
                log(f"WARNING: {pip_name} import still failing after "
                    f"install: {type(_post_exc).__name__}: {_post_exc}")
        except Exception as _imp_exc:
            # Package IS installed but its import raised something other
            # than ImportError -- e.g. DLL load failure, optional CUDA
            # probe blowing up, etc. Don't reinstall; that won't help.
            # Log it once so the user sees what's wrong, then mark
            # verified so we don't keep complaining every render.
            log(f"WARNING: {mod} is installed but import raised "
                f"{type(_imp_exc).__name__}: {_imp_exc} -- "
                f"NOT reinstalling (pip won't fix a load-time error)")
            verified.add(mod)

    # 3. Neutralise a broken bitsandbytes (probes + uninstalls if needed).
    _ensure_no_broken_bitsandbytes(log=log)

    # 4. Download checkpoints from HuggingFace if missing.
    need_unet = (not os.path.exists(LS_UNET_CKPT)
                 or os.path.getsize(LS_UNET_CKPT) < LS_MIN_UNET_BYTES)
    need_whisp = not os.path.exists(LS_WHISPER_CKPT)
    if need_unet or need_whisp:
        from huggingface_hub import hf_hub_download
        ckpt_dir = os.path.join(LS, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        if need_unet:
            log(f"downloading {LATENTSYNC_HF_REPO}/latentsync_unet.pt (~5 GB; first time only) ...")
            hf_hub_download(repo_id=LATENTSYNC_HF_REPO,
                            filename="latentsync_unet.pt",
                            local_dir=ckpt_dir)
        if need_whisp:
            log(f"downloading {LATENTSYNC_HF_REPO}/whisper/tiny.pt ...")
            hf_hub_download(repo_id=LATENTSYNC_HF_REPO,
                            filename="whisper/tiny.pt",
                            local_dir=ckpt_dir)

    return LS


def _infer_latentsync(face_path, driver, ffmpeg, log=print, *,
                      inference_steps: int = 20,
                      guidance_scale: float = 1.5,
                      enable_deepcache: bool = True,
                      seed: int = -1,
                      resolution: int = 512,
                      use_finetune: bool = False):
    """Run LatentSync inference. Returns the raw lip-synced video path.

    Stages inputs into a space-free scratch dir (LatentSync internally
    shells out to ffmpeg and is not robust to spaces in source paths --
    same gap that broke the Diff2Lip render until its staging fix).
    The driver audio is converted to a clean 16 kHz mono PCM WAV so the
    Whisper feature extractor sees a predictable input.

    Inference command is `python -m scripts.inference` with the verbatim
    args from upstream inference.sh -- if the upstream changes them,
    this function is the only place that needs updating.
    """
    _ensure_latentsync(log)
    scratch = os.path.join(OUT_DIR, "ls_scratch")
    os.makedirs(scratch, exist_ok=True)
    raw = os.path.join(OUT_DIR, "_latentsync_raw.mp4")
    if os.path.exists(raw):
        os.remove(raw)

    import shutil as _sh
    vid_ext = os.path.splitext(face_path)[1] or ".mp4"
    staged_video = os.path.join(scratch, "ls_in_video" + vid_ext)
    staged_audio = os.path.join(scratch, "ls_in_audio.wav")
    _sh.copy2(face_path, staged_video)
    r = subprocess.run(
        [ffmpeg, "-y", "-i", driver, "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", staged_audio],
        capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0 or not os.path.exists(staged_audio):
        raise RuntimeError(
            "LatentSync: could not prepare 16 kHz mono WAV for the driver "
            f"audio:\n{(r.stderr or '')[-400:]}")
    log(f"staged inputs into space-free scratch: "
        f"{os.path.basename(staged_video)}, {os.path.basename(staged_audio)}")

    # Resolution override: stage2_512.yaml is the upstream default.
    # If the user picked a different resolution, copy the YAML and
    # patch `data.resolution:` to the requested value, then point
    # --unet_config_path at the temp file. 256 = ~4x faster + less
    # VRAM but mouth detail drops; 1024 = sharper but ~3x slower.
    config_path = LS_CONFIG
    if int(resolution) != 512:
        src_yaml = os.path.join(LS, LS_CONFIG)
        try:
            with open(src_yaml, "r", encoding="utf-8") as _yf:
                _txt = _yf.read()
            _new = re.sub(r"(\bresolution:\s*)\d+",
                          rf"\g<1>{int(resolution)}", _txt, count=1)
            if _new == _txt:
                log(f"  WARN: could not find 'resolution:' in "
                    f"{LS_CONFIG}; using upstream 512 anyway")
            else:
                _tmp_cfg_dir = os.path.join("configs", "unet")
                os.makedirs(os.path.join(LS, _tmp_cfg_dir),
                            exist_ok=True)
                _tmp_cfg_name = f"_stage2_{int(resolution)}_custom.yaml"
                _tmp_cfg = os.path.join(_tmp_cfg_dir, _tmp_cfg_name)
                with open(os.path.join(LS, _tmp_cfg), "w",
                          encoding="utf-8") as _tf:
                    _tf.write(_new)
                config_path = _tmp_cfg
                log(f"  using patched config {_tmp_cfg} "
                    f"(resolution={int(resolution)})")
        except Exception as _exc:
            log(f"  WARN: could not patch resolution ({_exc}); "
                "falling back to upstream 512")
    # Per-clip identity fine-tune: if the user opted in AND a fine-tune
    # checkpoint exists for this source clip, point inference at the
    # fine-tuned UNet instead of the base. The subprocess takes an
    # absolute path here, so we don't need to copy the ckpt into the
    # LatentSync repo.
    inf_ckpt = os.path.join("checkpoints", "latentsync_unet.pt")
    if use_finetune:
        try:
            from . import lipsync_finetune as _lsft
            _ft_ckpt = _lsft.get_finetune_checkpoint(face_path)
            if _ft_ckpt is not None:
                inf_ckpt = str(_ft_ckpt)
                log(f"[lipsync] using PER-CLIP fine-tune checkpoint: "
                    f"{_ft_ckpt}")
            else:
                log("[lipsync] use_finetune requested but no fine-tune "
                    "checkpoint found for this clip -- using base UNet")
        except Exception as _exc:
            log(f"[lipsync] WARN: fine-tune lookup failed ({_exc}); "
                "falling back to base UNet")
    argv = [
        sys.executable, "-m", "scripts.inference",
        "--unet_config_path",    config_path,
        "--inference_ckpt_path", inf_ckpt,
        "--inference_steps",     str(int(inference_steps)),
        "--guidance_scale",      str(float(guidance_scale)),
        "--seed",                str(int(seed)),
    ]
    if enable_deepcache:
        argv.append("--enable_deepcache")
    argv += [
        "--video_path",          staged_video,
        "--audio_path",          staged_audio,
        "--video_out_path",      raw,
    ]
    env = _ffmpeg_bin_env(ffmpeg, log)
    _seed_label = "random" if int(seed) == -1 else f"fixed {int(seed)}"
    log(f"running LatentSync inference ({int(resolution)}x{int(resolution)}, "
        f"{int(inference_steps)} ddim steps, "
        f"guidance {float(guidance_scale):.1f}, seed={_seed_label}, "
        f"deepcache={'on' if enable_deepcache else 'off'} -- "
        f"slower than Diff2Lip; first run also fetches InsightFace + "
        f"Whisper assets).")
    # [timing] separate model-load from DDIM inference. The subprocess
    # prints "Loaded checkpoint path: ..." right before inference starts,
    # so we wrap the log callback to capture that moment.
    _ls_t_start = time.perf_counter()
    _ls_t_loaded = [None]  # boxed so the closure can mutate it

    def _ls_log(line):
        if _ls_t_loaded[0] is None and "Loaded checkpoint" in str(line):
            _ls_t_loaded[0] = time.perf_counter()
            _dt = _ls_t_loaded[0] - _ls_t_start
            log(f"[timing] LatentSync model load: {_dt:.1f}s")
        log(line)

    _run(argv, cwd=LS, log=_ls_log, env=env)
    _ls_t_end = time.perf_counter()
    if _ls_t_loaded[0] is not None:
        log(f"[timing] LatentSync inference: "
            f"{_ls_t_end - _ls_t_loaded[0]:.1f}s")
    else:
        log("[timing] LatentSync inference: (no 'Loaded checkpoint' "
            "marker seen)")
    log(f"[timing] LatentSync TOTAL: {_ls_t_end - _ls_t_start:.1f}s")
    if not os.path.exists(raw) or os.path.getsize(raw) < 100_000:
        raise RuntimeError("LatentSync produced no output -- check the log")
    return raw


def _infer_diff2lip(face_path, driver, ffmpeg, log=print):
    """Run Diff2Lip inference. Returns the raw lip-synced video path.

    The model / diffusion / sample / data / tfg flags are verbatim from
    the repo's scripts/inference_single_video.sh (sample_mode "cross",
    NUM_GPUS=1). generate.py runs with cwd = the clone so its `audio`
    and `face_detection` packages resolve; guided_diffusion goes on
    PYTHONPATH."""
    gd_parent = _ensure_diff2lip(log)
    scratch = os.path.join(OUT_DIR, "d2l_scratch")
    os.makedirs(scratch, exist_ok=True)
    raw = os.path.join(OUT_DIR, "_diff2lip_raw.mp4")
    if os.path.exists(raw):
        os.remove(raw)

    # Stage the inputs into the (space-free) scratch dir under safe names.
    # Diff2Lip's generate.py builds its internal ffmpeg commands with
    # unquoted str.format() and runs them via shell=True (load_all_indiv_mels
    # / load_video_frames). Any space in a source path -- e.g. a Demucs stem
    # under ".../Backwards Umbrella/" -- makes the shell split the path,
    # ffmpeg fails silently (the return code is never checked), and the next
    # step crashes on the missing audio.wav. Passing space-free copies
    # sidesteps every such shell command at once, whatever the source is
    # named or where it lives.
    import shutil
    vid_ext = os.path.splitext(face_path)[1] or ".mp4"
    aud_ext = os.path.splitext(driver)[1] or ".wav"
    staged_video = os.path.join(scratch, "d2l_in_video" + vid_ext)
    staged_audio = os.path.join(scratch, "d2l_in_audio" + aud_ext)
    shutil.copy2(face_path, staged_video)
    shutil.copy2(driver, staged_audio)
    log(f"staged inputs into space-free scratch: "
        f"{os.path.basename(staged_video)}, {os.path.basename(staged_audio)}")

    argv = [
        sys.executable, "generate.py",
        "--attention_resolutions", "32,16,8", "--class_cond", "False",
        "--learn_sigma", "True", "--num_channels", "128",
        "--num_head_channels", "64", "--num_res_blocks", "2",
        "--resblock_updown", "True", "--use_fp16", "True",
        "--use_scale_shift_norm", "False",
        "--predict_xstart", "False", "--diffusion_steps", "1000",
        "--noise_schedule", "linear", "--rescale_timesteps", "False",
        "--sampling_seed=7", "--sampling_input_type=gt",
        "--sampling_ref_type=gt", "--timestep_respacing", "ddim25",
        "--use_ddim", "True", f"--model_path={D2L_CKPT}",
        "--nframes", "5", "--nrefer", "1", "--image_size", "128",
        "--sampling_batch_size=32",
        "--face_hide_percentage", "0.5", "--use_ref=True",
        "--use_audio=True", "--audio_as_style=True",
        "--generate_from_filelist", "0",
        f"--video_path={staged_video}", f"--audio_path={staged_audio}",
        f"--out_path={raw}", "--save_orig=False",
        "--face_det_batch_size", "16", "--pads", "0,0,0,0",
        "--is_voxceleb2=False", f"--sample_path={scratch}",
    ]
    env = _ffmpeg_bin_env(ffmpeg, log)
    env["PYTHONPATH"] = gd_parent + os.pathsep + env.get("PYTHONPATH", "")
    log("running Diff2Lip diffusion inference -- slower than Wav2Lip; the "
        "first run also fetches the s3fd detector ...")
    _run(argv, cwd=D2L, log=log, env=env)
    if not os.path.exists(raw) or os.path.getsize(raw) < 100_000:
        raise RuntimeError("Diff2Lip produced no output -- check the log")
    return raw


def render_lipsync(face_path: str, audio_path: str, *,
                   engine: str = "wav2lip",
                   isolate_vocals: bool = True,
                   checkpoint: str = "wav2lip_gan.pth",
                   test_seconds: int = 0,
                   restore_faces: bool = True,
                   latentsync_inference_steps: int = 20,
                   latentsync_guidance_scale: float = 1.5,
                   latentsync_enable_deepcache: bool = True,
                   latentsync_seed: int = -1,
                   latentsync_resolution: int = 512,
                   latentsync_use_finetune: bool = False,
                   log=print) -> str:
    """Render an offline lip-synced video. Returns the output mp4 path.

    engine        -- "wav2lip" (fast GAN, 96px), "diff2lip" (diffusion,
                     128px -- sharper mouth than Wav2Lip), or
                     "latentsync" (SD-1.5 latent diffusion, 512px
                     native -- best mouth quality, slowest, ~18 GB
                     VRAM, validated standalone via
                     latentsync_beta_test.py)
    face_path     -- a video of the (swapped) face to drive
    audio_path    -- the song / audio
    isolate_vocals-- run Demucs and drive the engine with the vocal stem
    checkpoint    -- Wav2Lip checkpoint (ignored when engine="diff2lip")
    test_seconds  -- >0 trims to a quick test; 0 = full track
    restore_faces -- run GFPGAN to re-add facial detail (fixes the soft
                     look); slow -- adds a per-frame pass
    log           -- callable(str) for progress lines
    Raises RuntimeError / OSError on failure.
    """
    engine = (engine or "wav2lip").strip().lower()
    if engine not in ("wav2lip", "diff2lip", "latentsync"):
        raise ValueError(f"unknown lip-sync engine '{engine}'")
    if not face_path or not os.path.exists(face_path):
        raise FileNotFoundError(f"face video not found: {face_path}")
    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(f"audio not found: {audio_path}")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RENDER_DIR, exist_ok=True)
    os.makedirs(os.path.join(WORK, "inputs"), exist_ok=True)
    ffmpeg = _resolve_ffmpeg(log)
    log(f"ffmpeg: {ffmpeg}")
    log(f"lip-sync engine: {engine}")

    # 1. working song (optional quick-test trim)
    #
    # PERF FIX: if the user did NOT set test_seconds but the face video is
    # shorter than the audio, trim the audio to the face video's length
    # anyway. Otherwise Demucs spends time isolating vocals that LatentSync
    # will never use (the final mux uses -shortest, so the output is
    # already capped at the face video length). On a 2:30 song with a 10 s
    # face clip this alone saves ~2 minutes of Demucs wall time.
    song = audio_path
    effective_trim_seconds = int(test_seconds) if test_seconds and test_seconds > 0 else 0
    if effective_trim_seconds == 0:
        try:
            _probe = subprocess.run(
                [ffmpeg, "-i", face_path, "-f", "null", "-"],
                capture_output=True, text=True, encoding="utf-8", errors="ignore")
            _m = re.search(r"Duration:\s+(\d+):(\d+):(\d+\.\d+)", _probe.stderr or "")
            if _m:
                _h, _mn, _s = int(_m.group(1)), int(_m.group(2)), float(_m.group(3))
                face_secs = _h * 3600 + _mn * 60 + _s
                _ap = subprocess.run(
                    [ffmpeg, "-i", audio_path, "-f", "null", "-"],
                    capture_output=True, text=True, encoding="utf-8", errors="ignore")
                _am = re.search(r"Duration:\s+(\d+):(\d+):(\d+\.\d+)", _ap.stderr or "")
                if _am:
                    _ah, _amn, _as = int(_am.group(1)), int(_am.group(2)), float(_am.group(3))
                    audio_secs = _ah * 3600 + _amn * 60 + _as
                    # Trim audio to face length + 1 s slack if face is meaningfully shorter
                    if audio_secs > face_secs + 2.0 and face_secs > 0.5:
                        effective_trim_seconds = int(face_secs) + 1
                        log(f"PERF: face video is {face_secs:.1f}s but audio is "
                            f"{audio_secs:.1f}s -- auto-trimming audio to "
                            f"{effective_trim_seconds}s to skip Demucs work that "
                            "the final mux would discard anyway.")
        except Exception:
            pass

    if effective_trim_seconds > 0:
        clip = os.path.join(WORK, "inputs", "_song_test.wav")
        if _run_soft([ffmpeg, "-y", "-i", audio_path, "-t",
                      str(effective_trim_seconds), clip]) and os.path.exists(clip):
            song = clip

    # 2. driver audio -- isolated vocal stem (cleaner sync) or the full mix
    driver = song
    if isolate_vocals:
        driver = _isolate_vocals(song, log)

    # 3. engine-specific inference -> a raw lip-synced video
    raw = os.path.join(OUT_DIR, "_wav2lip_raw.mp4")
    if engine == "diff2lip":
        video_src = _infer_diff2lip(face_path, driver, ffmpeg, log)
    elif engine == "latentsync":
        video_src = _infer_latentsync(
            face_path, driver, ffmpeg, log,
            inference_steps=latentsync_inference_steps,
            guidance_scale=latentsync_guidance_scale,
            enable_deepcache=latentsync_enable_deepcache,
            seed=latentsync_seed,
            resolution=latentsync_resolution,
            use_finetune=latentsync_use_finetune,
        )
    else:
        if checkpoint not in MODEL_URLS:
            raise ValueError(f"unknown checkpoint '{checkpoint}'")
        if not os.path.exists(os.path.join(W2L, "inference.py")):
            _run(["git", "clone", "--depth", "1", WAV2LIP_REPO, W2L], log=log)
        _download(MODEL_URLS[checkpoint], checkpoint_path(checkpoint), log=log)
        _patch_wav2lip(log)
        result_avi = os.path.join(W2L, "temp", "result.avi")
        for stale in (raw, result_avi):
            if os.path.exists(stale):
                os.remove(stale)
        log("running Wav2Lip inference (first run also fetches the s3fd "
            "detector) ...")
        r = _run([sys.executable, "inference.py",
                  "--checkpoint_path", os.path.join("checkpoints", checkpoint),
                  "--face", face_path, "--audio", driver, "--outfile", raw],
                 cwd=W2L, log=log, check=False, env=_env_with_ffmpeg(ffmpeg))
        video_src = None
        if os.path.exists(raw) and os.path.getsize(raw) > 100_000:
            video_src = raw
        elif (os.path.exists(result_avi)
              and os.path.getsize(result_avi) > 100_000):
            video_src = result_avi
            log("note: Wav2Lip's internal mux did not finish; using its "
                "generated temp/result.avi directly.")
        if video_src is None:
            tail = ((r.stdout or "")[-1000:] + "\n"
                    + (r.stderr or "")[-3000:]).strip()
            raise RuntimeError(
                f"Wav2Lip produced no video (inference exit {r.returncode}).\n"
                f"{tail}")

    # 4. GFPGAN face restoration -- re-adds the facial detail the 96/128 px
    #    generation crop cannot produce. Best-effort: on any failure, fall
    #    through with the un-restored video so the render still completes.
    gfpgan_status = "off (Enhance faces unchecked)"
    if restore_faces:
        try:
            video_src = _restore_faces(video_src, ffmpeg, log=log)
            gfpgan_status = "applied"
        except Exception as exc:
            gfpgan_status = f"FAILED -- {exc}"
            log(f"GFPGAN restoration skipped ({exc}); using un-restored video")

    # 5. final mux: the lip-synced video + the FULL song as the audio track.
    out = os.path.join(
        RENDER_DIR, f"lipsync_{engine}_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    log("muxing final video + full-song audio ...")
    _run([ffmpeg, "-y", "-i", video_src, "-i", song,
          "-map", "0:v:0", "-map", "1:a:0",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
          "-crf", "20", "-c:a", "aac", "-b:a", "192k", "-shortest", out],
         log=log)
    for _tmp in (raw, os.path.join(OUT_DIR, "_diff2lip_raw.mp4"),
                 os.path.join(OUT_DIR, "_restored.avi"),
                 os.path.join(OUT_DIR, "_restored.mp4")):
        if os.path.exists(_tmp):
            os.remove(_tmp)
    if not os.path.exists(out):
        raise RuntimeError("final mux produced no output")
    log(f"GFPGAN restore: {gfpgan_status}")
    log(f"DONE -> {out}")
    return out
