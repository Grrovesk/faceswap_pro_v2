"""SAM2 install + weights resolver, owned by v2 (not KeySync).

Goal: maskout (and any future SAM2 consumer in v2) does not depend on
KeySync being installed at a particular path. Weights live under
v2/checkpoints/sam2/, the sam2 package is expected in the main venv
(same one that runs LatentSync), and Hydra resolves configs via the
installed package's bundled config tree rather than a chdir-relative
path.

Backward-compat: if v2's sam2 install isn't ready, callers can fall
back to KeySync's venv + weights via the env vars KEYSYNC_REPO_DIR /
SAM2_KEYSYNC_FALLBACK=1. That keeps existing setups working until the
user migrates.

The "weights URL" is the public Meta SAM2.1 release. The base+ size
(~81 MB) is the sweet spot for face / object masking on a 24 GB GPU.
Large (~224 MB) is overkill for our use case.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Optional

# v2/core/sam2_install.py -> parents[1] = v2/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Owned-by-v2 location for SAM2 weights.
SAM2_CKPT_DIR = PROJECT_ROOT / "checkpoints" / "sam2"

# Pick the small-but-good default. Override via env vars to use large.
SAM2_CKPT_NAME = os.environ.get(
    "SAM2_CKPT_NAME", "sam2.1_hiera_base_plus.pt")
SAM2_CKPT = SAM2_CKPT_DIR / SAM2_CKPT_NAME

# Hydra config NAME (package-relative path inside sam2 package).
# build_sam2_video_predictor expects this exact format.
SAM2_CONFIG_BY_CKPT = {
    "sam2.1_hiera_tiny.pt":       "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small.pt":      "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus.pt":  "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large.pt":      "configs/sam2.1/sam2.1_hiera_l.yaml",
}
SAM2_DEFAULT_CONFIG = SAM2_CONFIG_BY_CKPT.get(
    SAM2_CKPT_NAME, "configs/sam2.1/sam2.1_hiera_b+.yaml")

# Public Meta release URLs (Sep 28 2024 SAM 2.1 set)
_SAM2_WEIGHT_URLS = {
    "sam2.1_hiera_tiny.pt":
        "https://dl.fbaipublicfiles.com/segment_anything_2/"
        "092824/sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small.pt":
        "https://dl.fbaipublicfiles.com/segment_anything_2/"
        "092824/sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus.pt":
        "https://dl.fbaipublicfiles.com/segment_anything_2/"
        "092824/sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large.pt":
        "https://dl.fbaipublicfiles.com/segment_anything_2/"
        "092824/sam2.1_hiera_large.pt",
}

# KeySync fallback locations (used only if the v2 install isn't ready).
_KEYSYNC_REPO_DIR_DEFAULT = (Path(__file__).resolve().parents[3]
                              / "lipsync_test" / "KeySync")


# ----------------------------------------------------------------------
# Install probe
# ----------------------------------------------------------------------
def is_sam2_importable(python_exe: Optional[str] = None) -> bool:
    """Return True if `import sam2` succeeds in the given interpreter
    (defaults to the current process's). Used to decide whether
    sys.executable can run the worker, vs. needing the KeySync venv
    fallback.
    """
    import subprocess
    exe = python_exe or sys.executable
    try:
        r = subprocess.run(
            [exe, "-c", "import sam2"],
            capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


def keysync_python() -> Path:
    """Path to KeySync's venv Python (where SAM2 was originally
    installed). Used only as a fallback by maskout_pipeline."""
    repo = Path(os.environ.get("KEYSYNC_REPO_DIR",
                               str(_KEYSYNC_REPO_DIR_DEFAULT)))
    return repo / "venv" / "Scripts" / "python.exe"


def keysync_ckpt() -> Path:
    """Path to KeySync's bundled SAM2 weights (legacy location)."""
    repo = Path(os.environ.get("KEYSYNC_REPO_DIR",
                               str(_KEYSYNC_REPO_DIR_DEFAULT)))
    return repo / "pretrained_models" / "checkpoints" / SAM2_CKPT_NAME


# ----------------------------------------------------------------------
# Weights download
# ----------------------------------------------------------------------
def ensure_sam2_weights(log: Callable[[str], None] = print) -> Path:
    """Make sure SAM2_CKPT exists; download from Meta's public CDN if
    missing. Returns the resolved path the caller should pass to
    build_sam2_video_predictor.

    If the v2-local path is missing AND KeySync's legacy weights file
    is present, prefer that to avoid a second copy of a 100+ MB file.
    Set SAM2_NO_LEGACY=1 to force a fresh download into v2.
    """
    if SAM2_CKPT.is_file() and SAM2_CKPT.stat().st_size > 10_000_000:
        return SAM2_CKPT
    legacy = keysync_ckpt()
    if (not os.environ.get("SAM2_NO_LEGACY")
            and legacy.is_file()
            and legacy.stat().st_size > 10_000_000):
        log(f"[sam2] using legacy KeySync weights at {legacy}")
        return legacy
    url = _SAM2_WEIGHT_URLS.get(SAM2_CKPT_NAME)
    if url is None:
        raise RuntimeError(
            f"No known download URL for SAM2_CKPT_NAME={SAM2_CKPT_NAME}; "
            f"set SAM2_CKPT_NAME to one of "
            f"{list(_SAM2_WEIGHT_URLS.keys())}")
    SAM2_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"[sam2] downloading {SAM2_CKPT_NAME} from "
        f"dl.fbaipublicfiles.com (~80-225 MB, first time only) ...")
    import urllib.request
    tmp = SAM2_CKPT.with_suffix(SAM2_CKPT.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as resp, \
            open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, SAM2_CKPT)
    log(f"[sam2] wrote {SAM2_CKPT} "
        f"({SAM2_CKPT.stat().st_size / (1024 * 1024):.1f} MB)")
    return SAM2_CKPT


def pip_install_hint() -> str:
    """User-facing install instruction for getting sam2 into the active venv."""
    return (
        "SAM2 package not importable in this Python interpreter. "
        "Install with:\n"
        "  git clone https://github.com/facebookresearch/sam2.git\n"
        "  pip install -e ./sam2\n"
        "Then restart faceswap_pro."
    )
