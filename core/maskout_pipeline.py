"""SAM2 multi-click mask-out pipeline for lipsync sources.

Use case: the source video has a non-face object (cat, lizard, animal,
prop) that LatentSync's internal face detector wrongly latches onto and
applies lipsync motion to. Mask the object out before feeding LatentSync
a clean human-only frame, then composite the original masked pixels
back over the lipsync output so the object survives untouched.

Pipeline:
  1. SAM2: user clicks (positive/negative) on an object in a chosen
     reference frame -> single-object multi-click refinement -> binary
     mask propagated to every frame of the video (T, H, W) uint8.
  2. make_void_source: TELEA-inpaint the masked region in every frame
     of the source video, plus optional dilation to cover edge bleed.
     Produces a "void" mp4 with the object removed.
  3. Caller runs LatentSync on the void mp4 -> lipsync output without
     the spurious face.
  4. composite_back: warp the original source's masked pixels onto the
     lipsync output via the SAM2 mask, with optional Gaussian feather
     at the mask edge to hide the seam.

All scripts live inside this module so the orchestrator has a single
import surface.

Clicks: each click is (x, y, frame_idx, label) with label=1 positive
(add to mask) or label=0 negative (subtract from mask). Pass at least
one positive click.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

# SAM2 install resolution: prefer v2-native (sam2 importable in
# sys.executable + weights at v2/checkpoints/sam2/). Fall back to
# KeySync's venv + weights only if the v2-native path isn't ready.
# This decouples mask-out from KeySync so the user can delete KeySync
# without breaking lipsync workflows.
from . import sam2_install as _si

SAM2_WORKER = Path(__file__).parent / "_sam2_worker.py"


# ----------------------------------------------------------------------
# Stream-run helper (mirrors keysync.py)
# ----------------------------------------------------------------------
def _stream_run(argv: list, cwd: str, env: dict,
                log: Callable[[str], None]) -> None:
    proc = subprocess.Popen(
        argv, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="ignore", bufsize=1)
    for line in proc.stdout:
        log(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess exited {proc.returncode}: "
            f"{' '.join(argv[:3])}...")


def _resolve_py_for_sam2() -> str:
    """Pick the Python interpreter that has SAM2 installed.

    Order:
      1. sys.executable (main v2 venv) if `import sam2` works there.
         This is the v2-native path; no KeySync dependency.
      2. KeySync's venv (legacy fallback). Only used if sam2 isn't
         importable in the main venv.

    Caller does not need to know which one fired -- the worker takes
    care of config/weights via CLI args regardless.
    """
    if _si.is_sam2_importable(sys.executable):
        return sys.executable
    ks_py = _si.keysync_python()
    if ks_py.is_file():
        return str(ks_py)
    # Last resort: let the subprocess crash with a clear message.
    return sys.executable


def _build_env() -> dict:
    """Pass through the KEYSYNC_REPO_DIR env var unchanged in case the
    worker falls back to legacy mode (no --sam2_ckpt arg)."""
    env = os.environ.copy()
    env.setdefault("PYTHONWARNINGS", "ignore")
    return env


# ----------------------------------------------------------------------
# Step 1: SAM2 multi-click -> masks .npy
# ----------------------------------------------------------------------
def run_sam2_masks(source_video: Path,
                    clicks: List[Tuple[int, int, int, int]],
                    out_masks: Path,
                    log: Callable[[str], None] = print) -> Path:
    """Invoke _sam2_worker.py in single-object mode with the given
    click list. Writes (T, H, W) uint8 mask to `out_masks`.

    clicks: list of (x, y, frame_idx, label) where label in {0, 1}.
    """
    if not clicks:
        raise ValueError("at least one click required for mask-out")
    pos = sum(1 for c in clicks if c[3] == 1)
    neg = sum(1 for c in clicks if c[3] == 0)
    if pos == 0:
        raise ValueError(
            "need at least one positive click (label=1) for the SAM2 "
            "mask to have something to select")
    log(f"[maskout] SAM2 single-object refinement: "
        f"{pos} positive + {neg} negative click(s)")
    out_masks.parent.mkdir(parents=True, exist_ok=True)
    py = _resolve_py_for_sam2()
    env = _build_env()
    # Resolve weights via v2-native path; falls back to KeySync legacy
    # weights internally if v2 ones aren't present.
    try:
        ckpt_path = _si.ensure_sam2_weights(log=log)
    except Exception as _exc:
        log(f"[maskout] WARN ensure_sam2_weights: {_exc} -- worker "
            f"will use legacy KeySync path")
        ckpt_path = None
    argv = [py, str(SAM2_WORKER),
            "--video", str(source_video),
            "--out", str(out_masks),
            "--single_object"]
    if ckpt_path is not None:
        argv += ["--sam2_ckpt", str(ckpt_path)]
    for (x, y, fidx, label) in clicks:
        argv += ["--click", str(int(x)), str(int(y)),
                  str(int(fidx)), str(int(label))]
    # No more KeySync chdir: the worker resolves Hydra configs via the
    # installed sam2 package when --sam2_ckpt is supplied. If we had to
    # fall back to legacy mode (no ckpt_path), the worker itself will
    # honor KEYSYNC_REPO_DIR env var and chdir there.
    _stream_run(argv, cwd=str(Path(SAM2_WORKER).parent), env=env, log=log)
    if not out_masks.is_file():
        raise RuntimeError(f"SAM2 worker produced no masks at {out_masks}")
    log(f"[maskout] SAM2 masks ready: {out_masks} "
        f"({out_masks.stat().st_size / (1024 * 1024):.1f} MB)")
    return out_masks


# ----------------------------------------------------------------------
# Step 2: TELEA-inpaint the masked region per frame -> void mp4
# ----------------------------------------------------------------------
def make_void_source(source_video: Path,
                      masks_npy: Path,
                      out_video: Path,
                      dilate_px: int = 12,
                      log: Callable[[str], None] = print) -> Path:
    """Per-frame TELEA inpaint of the masked region. Produces a video
    where the SAM2-tracked object has been removed and the hole filled
    with surrounding pixels. The result is what LatentSync will see.
    """
    import cv2
    masks = np.load(str(masks_npy))                    # (T, H, W) uint8
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source video: {source_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if masks.shape[0] < n_frames:
        log(f"[maskout] WARN: masks have {masks.shape[0]} frames but "
            f"video has {n_frames}; truncating to mask length")
        n_frames = masks.shape[0]

    out_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps,
                              (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer for {out_video}")

    kernel = None
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))

    log(f"[maskout] inpainting {n_frames} frames "
        f"(dilate={dilate_px}px) ...")
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        m = masks[i].astype(np.uint8)
        if kernel is not None:
            m = cv2.dilate(m, kernel, iterations=1)
        # TELEA inpaint: fast, reasonable quality
        voided = cv2.inpaint(frame, m, 3, cv2.INPAINT_TELEA)
        writer.write(voided)
        if i % 50 == 0:
            log(f"[maskout]   inpaint frame {i}/{n_frames}")
    cap.release()
    writer.release()
    if not out_video.is_file():
        raise RuntimeError(f"void source not written: {out_video}")
    log(f"[maskout] void source ready: {out_video.name} "
        f"({out_video.stat().st_size / (1024 * 1024):.1f} MB)")
    return out_video


# ----------------------------------------------------------------------
# Step 3 (caller): run LatentSync on the void source
# ----------------------------------------------------------------------
# (Not implemented here -- the orchestrator calls latentsync.run() on
# the void source it gets from this module.)


# ----------------------------------------------------------------------
# Step 4: composite original masked pixels back onto lipsync output
# ----------------------------------------------------------------------
def composite_back(source_video: Path,
                    lipsync_video: Path,
                    masks_npy: Path,
                    out_video: Path,
                    feather: int = 8,
                    log: Callable[[str], None] = print) -> Path:
    """For each frame: where the SAM2 mask is set, paste the ORIGINAL
    source pixel; everywhere else, keep the lipsync output. Optional
    Gaussian feather at the mask edge hides the seam.

    Source / lipsync length handling:
      - If lipsync is longer than source (e.g. extend_single looped a
        10s source up to 139s of audio), the source and masks are
        looped via modulo indexing so the cat / occluder reappears in
        the right place in every loop tile.
      - If lipsync is shorter than source, source/masks are truncated.

    Resolution mismatch (LatentSync's internal 512 resize against an
    HD source) is fixed up by resizing the lipsync frames to match
    the source dims.
    """
    import cv2
    masks = np.load(str(masks_npy))                    # (T, H, W) uint8
    cap_src = cv2.VideoCapture(str(source_video))
    cap_lip = cv2.VideoCapture(str(lipsync_video))
    if not cap_src.isOpened():
        raise RuntimeError(f"cannot open source: {source_video}")
    if not cap_lip.isOpened():
        raise RuntimeError(f"cannot open lipsync: {lipsync_video}")

    fps = cap_src.get(cv2.CAP_PROP_FPS) or 25.0
    # Both fps values are needed for the index correction below. The
    # bug we're fixing: source can be e.g. 30 fps while LatentSync
    # always outputs at 25 fps. If we keep linear i%n_src indexing,
    # source/mask pixels drift relative to lipsync wall-clock time
    # (at lip frame 25 = 1.000s, the old code grabbed src frame 25 =
    # 0.833s on a 30fps source -- 167 ms drift). The composite then
    # paints the wrong moment's occluder pixels over the right
    # moment's lipsync output, looking like lipsync itself broke.
    fps_src = fps
    fps_lip = cap_lip.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap_src.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_src_video = int(cap_src.get(cv2.CAP_PROP_FRAME_COUNT))
    n_lip = int(cap_lip.get(cv2.CAP_PROP_FRAME_COUNT))

    # Buffer ALL source frames + their masks once. We need random-
    # access into them for modulo looping, and cv2.VideoCapture seek
    # is slow + unreliable on mp4. n_src = min(video_frames, masks).
    n_src = min(n_src_video, int(masks.shape[0]))
    src_frames = []
    log(f"[maskout] buffering {n_src} source frames for "
        f"modulo-loop composite (~{n_src * height * width * 3 / (1024**3):.2f} GB)")
    for i in range(n_src):
        ok, frm = cap_src.read()
        if not ok or frm is None:
            break
        src_frames.append(frm)
    cap_src.release()
    n_src = len(src_frames)
    if n_src == 0:
        raise RuntimeError(
            f"could not read any source frames from {source_video}")
    if n_lip > n_src:
        log(f"[maskout] lipsync ({n_lip}f) longer than source ({n_src}f); "
            f"source loops {n_lip / n_src:.2f}x via modulo indexing")
    elif n_lip < n_src:
        log(f"[maskout] lipsync ({n_lip}f) shorter than source ({n_src}f); "
            f"source truncated to lipsync length")

    out_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # CRITICAL: write the output at fps_lip, NOT fps_src. The frames
    # we're writing are LATENTSYNC output frames -- one per LatentSync
    # frame -- so they play at LatentSync's tempo (25 fps). Writing at
    # fps_src stretches/compresses the video relative to the audio,
    # which surfaces as mouth-lag-audio drift when source is 24 fps
    # and lipsync is 25 fps over a long extend_single render
    # (~1.3 s drift over 30 s).
    writer = cv2.VideoWriter(str(out_video), fourcc, float(fps_lip),
                              (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer for {out_video}")

    blur_k = max(int(feather) // 2 * 2 + 1, 1)  # nearest odd
    # Maps a lipsync frame index to the source frame index at the
    # same wall-clock moment. Collapses to 1.0 when both fps are
    # equal, recovering the original linear indexing for 25 fps
    # sources -- so the fix is backward-compatible.
    src_per_lip = float(fps_src) / float(fps_lip)
    log(f"[maskout] compositing {n_lip} frames "
        f"(feather kernel={blur_k}, fps_src={fps_src:.3f}, "
        f"fps_lip={fps_lip:.3f}, src_per_lip={src_per_lip:.4f}) ...")
    for i in range(n_lip):
        ok_l, frm_lip = cap_lip.read()
        if not ok_l or frm_lip is None:
            break
        # fps-aware index: lipsync frame i (at time i/fps_lip) maps
        # to source frame round(i*fps_src/fps_lip), looped modulo.
        idx = int(round(i * src_per_lip)) % n_src
        frm_src = src_frames[idx]
        if frm_lip.shape[:2] != (height, width):
            frm_lip = cv2.resize(frm_lip, (width, height),
                                  interpolation=cv2.INTER_CUBIC)
        m = masks[idx].astype(np.float32)
        m = m / max(float(m.max()), 1.0)  # normalize to [0, 1]
        if feather > 0:
            m = cv2.GaussianBlur(m, (blur_k, blur_k), 0)
        m3 = m[:, :, None]  # (H, W, 1) for broadcast
        merged = (m3 * frm_src.astype(np.float32)
                   + (1.0 - m3) * frm_lip.astype(np.float32))
        writer.write(np.clip(merged, 0, 255).astype(np.uint8))
        if i % 100 == 0:
            log(f"[maskout]   composite frame {i}/{n_lip}")
    cap_lip.release()
    writer.release()
    if not out_video.is_file():
        raise RuntimeError(f"composite not written: {out_video}")
    log(f"[maskout] composite ready: {out_video.name} "
        f"({out_video.stat().st_size / (1024 * 1024):.1f} MB)")
    return out_video


# ----------------------------------------------------------------------
# High-level convenience: run all three steps in sequence
# ----------------------------------------------------------------------
def run_pipeline(source_video: Path,
                  clicks: List[Tuple[int, int, int, int]],
                  workspace: Path,
                  dilate_px: int = 12,
                  feather: int = 8,
                  log: Callable[[str], None] = print) -> dict:
    """Run SAM2 -> void source, return paths the orchestrator needs.

    Returns dict with keys:
      - masks_npy: Path -- (T, H, W) uint8 mask
      - void_video: Path -- inpainted source for LatentSync to render on
    The caller then runs LatentSync on void_video, then calls
    composite_back(source_video, lipsync_video, masks_npy, ...).
    """
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    masks_npy = workspace / "sam2_masks.npy"
    void_video = workspace / "void_source.mp4"
    run_sam2_masks(source_video, clicks, masks_npy, log=log)
    make_void_source(source_video, masks_npy, void_video,
                      dilate_px=dilate_px, log=log)
    return {"masks_npy": masks_npy, "void_video": void_video}
