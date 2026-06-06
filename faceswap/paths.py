"""All output / scratch paths in one module. Resolved once at import.

v2 STANDALONE: PROJECT_ROOT is v2/ (not the parent faceswap_pro/).
v2 outputs and scratch live inside v2/. External repos (LatentSync, RVC)
that v2 subprocess-launches are referenced via EXTERNAL_REPOS_ROOT, which
defaults to the peer ../lipsync_test/ folder where v1 installed them, but
can be overridden with the FACESWAP_EXTERNAL_REPOS environment variable
for a truly portable v2 install.
"""
from __future__ import annotations

import os
from pathlib import Path

# v2/faceswap/paths.py -> parents[1] = v2/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Outputs (visible to the user)
RECORDINGS_DIR = PROJECT_ROOT / "recordings" / "lipsync"
WEBCAM_RECORDINGS_DIR = PROJECT_ROOT / "recordings" / "webcam"

# Scratch / work (hidden, can be wiped)
WORK_DIR = PROJECT_ROOT / "lipsync_test"
MULTICLIP_WORK = WORK_DIR / "_multiclip_work"
EXTEND_SINGLE_WORK = WORK_DIR / "_extend_single_work"
LATENTSYNC_SCRATCH = WORK_DIR / "_ls_scratch_v2"

# External upstream repos that v2 SUBPROCESS-LAUNCHES (not Python imports).
# These are LatentSync (lipsync inference), RVC (voice clone). Default
# location is the peer ../lipsync_test/ folder (faceswap_pro/lipsync_test/)
# so existing installs keep working. Override via env var for a portable
# v2 install:
#   set FACESWAP_EXTERNAL_REPOS=D:\my_models\lipsync_test
EXTERNAL_REPOS_ROOT = Path(
    os.environ.get("FACESWAP_EXTERNAL_REPOS",
                   str(PROJECT_ROOT.parent / "lipsync_test")))
LATENTSYNC_REPO_DIR = EXTERNAL_REPOS_ROOT / "LatentSync"
RVC_REPO_DIR = EXTERNAL_REPOS_ROOT / "RVC"


def ensure_all() -> None:
    """Create every directory we'll write to. Idempotent."""
    for d in (RECORDINGS_DIR, WEBCAM_RECORDINGS_DIR,
              WORK_DIR, MULTICLIP_WORK,
              EXTEND_SINGLE_WORK, LATENTSYNC_SCRATCH):
        d.mkdir(parents=True, exist_ok=True)


def safe_clear_output(out_path: Path, log=print, retries: int = 5,
                       sleep_s: float = 0.4) -> Path:
    """Make `out_path` writable, dodging Windows file-locks.

    On Windows the Gradio video component holds a file handle to the
    most recently rendered output while it's visible in the UI, and
    Python's `Path.unlink()` raises `PermissionError: [WinError 32] The
    process cannot access the file because it is being used by another
    process` until the player closes. This helper tries:

      1. Plain unlink (works when nothing has it open -- the common case).
      2. Same with N retries + small backoff (lock often clears in
         ~1 second once Gradio is told to swap to a new video).
      3. Rename to a stale name (`<stem>_stale_<ms>.mp4`) and continue.
         Rename frequently succeeds even when delete fails on Windows.
      4. If even rename fails, return a *fresh* timestamped Path
         (`<stem>_<ms>.mp4`) so the render still proceeds; the old
         file stays on disk and the new one lands beside it.

    Returns the Path the caller should ACTUALLY write to.  Callers that
    were already using `out_path = RECORDINGS_DIR / "foo.mp4"` should
    pass `out_path` here and use the return value as their write
    target.
    """
    import time
    out_path = Path(out_path)
    if not out_path.exists():
        return out_path
    # Phase 1: try unlink with retries
    for attempt in range(max(1, int(retries))):
        try:
            out_path.unlink()
            return out_path
        except PermissionError:
            if attempt < retries - 1:
                time.sleep(float(sleep_s))
                continue
            log(f"[paths] unlink locked after {retries} tries: "
                f"{out_path.name} -- attempting rename")
        except FileNotFoundError:
            return out_path
        except OSError as exc:
            log(f"[paths] unlink failed ({exc}); attempting rename")
            break
    # Phase 2: rename out of the way
    stamp_ms = int(time.time() * 1000)
    stale = out_path.with_name(
        f"{out_path.stem}_stale_{stamp_ms}{out_path.suffix}")
    try:
        out_path.rename(stale)
        log(f"[paths] renamed locked output: "
            f"{out_path.name} -> {stale.name}")
        return out_path
    except OSError as exc:
        log(f"[paths] rename also locked ({exc}); "
            f"writing to timestamped path instead")
    # Phase 3: give up, hand back a fresh timestamped path
    fresh = out_path.with_name(
        f"{out_path.stem}_{stamp_ms}{out_path.suffix}")
    log(f"[paths] write target redirected to: {fresh.name}")
    return fresh
