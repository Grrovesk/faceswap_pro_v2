"""Face-region Reinhard color match for LatentSync output.

Fixes the "stylized source -> orange cheek" artifact: LatentSync's VAE
nudges face skin tone toward its training distribution (warm portrait),
which is invisible on natural sources but breaks on stylized inputs
(cyan-hair, magenta-cool, animated portraits, etc.).

Approach:
  1. Sample N frames evenly from the SOURCE video (pre-lipsync).
  2. Sample the same N frame indices from the LATENTSYNC OUTPUT.
  3. For each sampled frame, detect the face bbox via opencv's Haar
     cascade (no new model deps); collect face-region LAB pixels.
  4. Compute aggregate (mean, std) LAB across all sampled frames for
     both source and lipsync.  Single global stats = no temporal
     flicker.
  5. For every frame in the LatentSync output: detect face, build a
     feathered elliptical mask around the bbox, apply per-channel
     Reinhard correction inside the mask:
         out = (frame_lab - lat_mean) * (src_std / lat_std) + src_mean
     Outside the mask: untouched.  Re-encode the corrected stream.

Failure modes are best-effort: if detection fails on too many sample
frames the function returns the input unchanged with a log line.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[1]


def _haar_face_detector() -> Optional[cv2.CascadeClassifier]:
    """Load OpenCV's bundled Haar frontal face detector.  No new
    model dependency -- bundled with cv2 since forever.
    """
    try:
        path = (Path(cv2.data.haarcascades) /
                  "haarcascade_frontalface_default.xml")
        if not path.is_file():
            return None
        det = cv2.CascadeClassifier(str(path))
        if det.empty():
            return None
        return det
    except Exception as exc:
        logger.warning("Haar face detector load failed: %s", exc)
        return None


def _detect_face_bbox(det, frame_bgr) -> Optional[Tuple[int, int, int, int]]:
    """Return the largest face bbox (x, y, w, h), or None."""
    if det is None:
        return None
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = det.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=4,
            minSize=(60, 60))
        if len(faces) == 0:
            return None
        # Pick the largest
        return tuple(max(faces, key=lambda b: int(b[2]) * int(b[3])))
    except Exception:
        return None


def _build_feathered_face_mask(frame_hw: Tuple[int, int],
                                 bbox: Tuple[int, int, int, int],
                                 feather_px: int = 32) -> np.ndarray:
    """Elliptical mask centered on bbox, feathered to feather_px.
    Returns float32 in [0, 1]."""
    H, W = frame_hw
    x, y, w, h = bbox
    cx, cy = int(x + w / 2), int(y + h / 2)
    # Slightly expand the bbox so the mask covers cheek + jaw, not
    # just the inner face.  Ellipse axes ~ 0.6 * bbox dims.
    ax = int(max(8, 0.6 * w))
    ay = int(max(8, 0.75 * h))
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    if feather_px > 0:
        k = max(1, feather_px * 2 + 1)
        m = cv2.GaussianBlur(m, (k, k), 0)
    return (m.astype(np.float32) / 255.0)


def _collect_face_lab_stats(video_path: Path,
                              det,
                              sample_indices: List[int],
                              log: Callable[[str], None]) -> Optional[
                                  Tuple[np.ndarray, np.ndarray, int]]:
    """Walk sample_indices in video_path.  For each sampled frame
    detect the face, accumulate LAB pixels inside the bbox.  Return
    (mean(3,), std(3,), n_frames_used) -- floats in float32 -- or
    None if too few detections."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log(f"[color-match] cannot open {video_path}")
        return None
    sample_set = set(int(i) for i in sample_indices)
    means: List[np.ndarray] = []
    stds: List[np.ndarray] = []
    n_found = 0
    cur = 0
    try:
        while sample_set:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if cur in sample_set:
                sample_set.discard(cur)
                bbox = _detect_face_bbox(det, frame)
                if bbox is not None:
                    x, y, w, h = bbox
                    H, W = frame.shape[:2]
                    # Pad inward slightly so we sample SKIN, not the
                    # hair / forehead edge.
                    pad = int(0.10 * w)
                    x0 = max(0, x + pad)
                    y0 = max(0, y + pad)
                    x1 = min(W, x + w - pad)
                    y1 = min(H, y + h - pad)
                    if x1 > x0 and y1 > y0:
                        crop = frame[y0:y1, x0:x1]
                        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                        m = lab.reshape(-1, 3).astype(np.float32)
                        means.append(m.mean(axis=0))
                        stds.append(m.std(axis=0))
                        n_found += 1
            cur += 1
    finally:
        cap.release()
    if n_found < 3:
        log(f"[color-match] {video_path.name}: only {n_found} face "
            f"samples (need >=3); aborting match")
        return None
    mean = np.median(np.stack(means, axis=0), axis=0).astype(np.float32)
    std = np.median(np.stack(stds, axis=0), axis=0).astype(np.float32)
    return mean, std, n_found


def _apply_reinhard_to_face(frame_bgr: np.ndarray,
                              bbox: Optional[Tuple[int, int, int, int]],
                              src_mean: np.ndarray, src_std: np.ndarray,
                              lat_mean: np.ndarray, lat_std: np.ndarray,
                              feather_px: int = 32) -> np.ndarray:
    """Apply Reinhard color correction inside an elliptical face
    mask.  Outside the mask: untouched.  Returns BGR uint8."""
    H, W = frame_bgr.shape[:2]
    if bbox is None:
        return frame_bgr
    mask = _build_feathered_face_mask((H, W), bbox, feather_px)
    if mask.max() <= 1e-6:
        return frame_bgr
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Reinhard per-channel.  Guard against lat_std=0.
    safe_lat_std = np.maximum(lat_std, 1e-3)
    corrected = (
        (lab - lat_mean.reshape(1, 1, 3))
        * (src_std.reshape(1, 1, 3) / safe_lat_std.reshape(1, 1, 3))
        + src_mean.reshape(1, 1, 3)
    )
    corrected = np.clip(corrected, 0, 255)
    a = mask[..., None]
    blended_lab = lab * (1.0 - a) + corrected * a
    out_bgr = cv2.cvtColor(blended_lab.astype(np.uint8),
                              cv2.COLOR_LAB2BGR)
    return out_bgr


def color_match_video(source_video: Path,
                        latentsync_output: Path,
                        out_path: Optional[Path] = None,
                        mode: str = "reinhard",
                        n_samples: int = 15,
                        feather_px: int = 32,
                        log: Callable[[str], None] = print) -> Path:
    """Match the face color distribution of ``latentsync_output`` to
    that of ``source_video``.

    Returns the path to the matched mp4.  On any failure returns
    ``latentsync_output`` unchanged so the caller still ships
    something.  Audio is dropped from the output -- the orchestrator
    is expected to remux audio back on a later stage.
    """
    if str(mode).lower() in ("", "none", "off", "false"):
        log("[color-match] mode='none' -- pass-through")
        return Path(latentsync_output)
    if str(mode).lower() != "reinhard":
        log(f"[color-match] unknown mode={mode!r} -- pass-through")
        return Path(latentsync_output)

    source_video = Path(source_video)
    latentsync_output = Path(latentsync_output)
    if not source_video.is_file() or not latentsync_output.is_file():
        log(f"[color-match] missing input(s); pass-through")
        return latentsync_output

    det = _haar_face_detector()
    if det is None:
        log("[color-match] no Haar detector available; pass-through")
        return latentsync_output

    # Sample N indices spaced across the LatentSync output (treat that
    # as the canonical frame index space; source video may be a different
    # length but we just want skin-tone stats, not frame alignment).
    cap = cv2.VideoCapture(str(latentsync_output))
    n_lat = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if n_lat <= 0:
        log("[color-match] zero-frame latentsync output; pass-through")
        return latentsync_output
    cap = cv2.VideoCapture(str(source_video))
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if n_src <= 0:
        log("[color-match] zero-frame source; pass-through")
        return latentsync_output

    n_samples = max(3, min(int(n_samples), n_lat, n_src))
    lat_samples = [int(i * (n_lat - 1) / (n_samples - 1))
                    for i in range(n_samples)]
    src_samples = [int(i * (n_src - 1) / (n_samples - 1))
                    for i in range(n_samples)]

    t0 = time.perf_counter()
    src_stats = _collect_face_lab_stats(source_video, det, src_samples,
                                          log=log)
    lat_stats = _collect_face_lab_stats(latentsync_output, det,
                                          lat_samples, log=log)
    if src_stats is None or lat_stats is None:
        log("[color-match] insufficient face detections; pass-through")
        return latentsync_output
    src_mean, src_std, n_src_found = src_stats
    lat_mean, lat_std, n_lat_found = lat_stats
    delta = src_mean - lat_mean
    log(f"[color-match] stats: src(n={n_src_found}) "
        f"mean_LAB={src_mean.tolist()} std={src_std.tolist()}  "
        f"lat(n={n_lat_found}) mean_LAB={lat_mean.tolist()} "
        f"std={lat_std.tolist()}  delta={delta.tolist()}")
    if float(np.linalg.norm(delta)) < 2.0:
        log("[color-match] |delta|<2 LAB units -- skipping correction")
        return latentsync_output

    if out_path is None:
        out_path = latentsync_output.with_name(
            latentsync_output.stem + "_colormatched.mp4")
    out_path = Path(out_path)

    # Re-encode every frame with face-region Reinhard.  Use mp4v
    # intermediate (matches the existing orchestrator pattern --
    # final remux + container fix-up happens downstream).
    cap = cv2.VideoCapture(str(latentsync_output))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    if not writer.isOpened():
        cap.release()
        log(f"[color-match] cannot open writer at {out_path}; pass-through")
        return latentsync_output

    n_done = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            bbox = _detect_face_bbox(det, frame)
            out_frame = _apply_reinhard_to_face(
                frame, bbox,
                src_mean, src_std, lat_mean, lat_std,
                feather_px=feather_px)
            writer.write(out_frame)
            n_done += 1
            if n_done % 200 == 0:
                log(f"[color-match] frame {n_done}/{n_lat}")
    finally:
        cap.release()
        writer.release()
    elapsed = time.perf_counter() - t0
    log(f"[color-match] DONE {n_done} frames in {elapsed:.1f}s -> {out_path}")
    return out_path
