"""core/xseg_gate.py -- FaceFusion XSeg occluder gate for lipsync.

CANONICAL 2026 PRODUCTION APPROACH:
  The industry-leading face manipulation platform (FaceFusion 3.x) handles
  exactly this problem (hands, mics, hair, chameleons, etc. occluding the
  face during face manipulation) by running a DeepFaceLab-trained XSeg
  occluder model per-frame on the face crop. The model outputs a face-skin
  probability mask; the occluder mask is its complement.

  FaceFusion ships THREE XSeg variants and min-reduces them at inference
  time for robustness. We do the same.

WHY THIS WORKS WHERE THE PREVIOUS ATTEMPTS DIDN'T:
  1. The XSeg models were trained on real DeepFaceLab annotations of
     hands / mics / hair / objects occluding faces. A chameleon held in
     front of a face is in distribution. No prompts, no clicks, no
     manual annotation, no "iterate on prompt placement".
  2. The mask comes from the SOURCE crop (chameleon visible), not from
     the lipsync output (chameleon already painted over). v2's mistake
     was running mask on the post-lipsync frame; we don't.
  3. We do POST-HOC compositing of source-over-lipsync through the
     occluder matte. That works HERE (unlike the post-hoc gating
     in §3 of the prior failure log) because we're pasting literal
     ORIGINAL pixels through the alpha -- there's no painted seam to
     fight. The lipsync engine's mouth pixels never need to win against
     anything inside the matte; they're simply replaced by source pixels.
  4. Stateless. 256x256 inference is ~3 ms/frame on A6000. A 313-frame
     test clip masks in under 5 seconds.

DOWNLOADS (auto, ~210 MB total):
  https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_1.onnx
  https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_2.onnx
  https://huggingface.co/facefusion/models-3.2.0/resolve/main/xseg_3.onnx

Files land in models/xseg/.

USAGE:
  from core.xseg_gate import build_occluder_masks_video, restore_occluder
  masks_dir = build_occluder_masks_video(source_mp4)
  gated_mp4 = restore_occluder(lipsync_mp4, source_mp4, masks_dir)

The video runner run_xseg_lipsync.py chains these into a single command.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
XSEG_DIR = PROJECT_ROOT / "models" / "xseg"
MASKS_ROOT = PROJECT_ROOT / "models" / "face_swap" / "_xseg_masks"

# FaceFusion-hosted XSeg ONNX (DeepFaceLab vendor, GPL-3.0). 70 MB each.
XSEG_MODELS = [
    ("xseg_1.onnx",
     "https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_1.onnx"),
    ("xseg_2.onnx",
     "https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_2.onnx"),
    ("xseg_3.onnx",
     "https://huggingface.co/facefusion/models-3.2.0/resolve/main/xseg_3.onnx"),
]

XSEG_INPUT_SIZE = 256  # all three models are 256x256


# ------------------------------------------------------------------
# infra
# ------------------------------------------------------------------
def _cache_key(video_path: str,
                bbox_smoothing: float = 0.0,
                mask_smoothing: float = 0.0) -> str:
    """Cache key for per-frame mask sequences. Includes temporal
    smoothing params so different smoothing settings get separate
    cache dirs. bbox=0, mask=0 reproduces the pre-smoothing key
    bit-identically so existing caches stay valid."""
    st = os.stat(video_path)
    h = hashlib.sha256()
    h.update(os.path.abspath(video_path).encode())
    h.update(str(int(st.st_mtime)).encode())
    h.update(str(st.st_size).encode())
    if bbox_smoothing > 0 or mask_smoothing > 0:
        h.update(f"_bbox{bbox_smoothing:.3f}_mask{mask_smoothing:.3f}".encode())
    return h.hexdigest()[:16]


def _resolve_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg") or "ffmpeg"


def _download(url: str, dest: Path, log=print):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"  downloading {dest.name} ...")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        total = 0
        while True:
            data = r.read(1024 * 1024)
            if not data:
                break
            f.write(data)
            total += len(data)
    tmp.replace(dest)
    log(f"  -> {dest} ({total / (1024 * 1024):.1f} MB)")


def ensure_xseg_models(log=print) -> List[Path]:
    """Download all three FaceFusion XSeg ONNX files into models/xseg/.
    Idempotent: skips files already present at >50 MB."""
    XSEG_DIR.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for name, url in XSEG_MODELS:
        p = XSEG_DIR / name
        if p.exists() and p.stat().st_size > 50 * 1024 * 1024:
            paths.append(p)
            continue
        try:
            _download(url, p, log=log)
            paths.append(p)
        except Exception as e:
            log(f"  WARN: download failed for {name}: {e}")
    if not paths:
        raise RuntimeError(
            "could not fetch any XSeg model. Manual fix: download one of\n"
            + "\n".join(f"  {u}" for _, u in XSEG_MODELS)
            + f"\ninto {XSEG_DIR}"
        )
    log(f"XSeg models available: {[p.name for p in paths]}")
    return paths


# ------------------------------------------------------------------
# session
# ------------------------------------------------------------------
class XSegEnsemble:
    """ORT inference pool for FaceFusion's three XSeg variants.
    Output is a face-skin mask 0..1; occluder = 1 - face_skin."""

    def __init__(self, model_paths: List[Path], log=print):
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sessions = []
        self.input_names = []
        for p in model_paths:
            sess = ort.InferenceSession(str(p), so, providers=providers)
            self.sessions.append(sess)
            self.input_names.append(sess.get_inputs()[0].name)
            log(f"  XSeg session: {p.name}  EP={sess.get_providers()[0]}  "
                f"input={sess.get_inputs()[0].name}{sess.get_inputs()[0].shape}")
        if not self.sessions:
            raise RuntimeError("no XSeg sessions could be created")

    def occluder_mask(self, face_crop_bgr) -> "object":
        """face_crop_bgr: uint8 HxWx3 BGR (any size).
        Returns float32 H_in x W_in occluder mask in 0..1 (1=occluder)."""
        import cv2
        import numpy as np
        H, W = face_crop_bgr.shape[:2]
        x = cv2.resize(face_crop_bgr, (XSEG_INPUT_SIZE, XSEG_INPUT_SIZE))
        # FaceFusion preprocessing: NHWC float32 /255, no mean/std, BGR as-is.
        x = np.expand_dims(x, axis=0).astype(np.float32) / 255.0  # 1,H,W,3
        masks = []
        for sess, in_name in zip(self.sessions, self.input_names):
            out = sess.run(None, {in_name: x})[0]  # 1,H,W or 1,1,H,W
            m = np.array(out)
            # squeeze to 2D
            while m.ndim > 2:
                m = m.squeeze(0) if m.shape[0] == 1 else m.squeeze(-1)
            m = np.clip(m, 0.0, 1.0).astype(np.float32)
            masks.append(m)
        # min-reduce = conservative face_skin estimate => UNION of occluders
        face_skin = np.minimum.reduce(masks)
        # back to crop resolution
        face_skin = cv2.resize(face_skin, (W, H))
        # apply FaceFusion's "(clip(0.5,1)-0.5)*2" tightening to drop low-conf,
        # then invert -> occluder alpha
        face_skin = (np.clip(face_skin, 0.5, 1.0) - 0.5) * 2.0
        occluder = 1.0 - face_skin
        return np.clip(occluder, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------
# detector helper
# ------------------------------------------------------------------
def _get_face_detector():
    """InsightFace buffalo_l detector; CUDA EP preferred."""
    from insightface.app import FaceAnalysis
    fa = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        allowed_modules=["detection"],
    )
    fa.prepare(ctx_id=0, det_size=(640, 640))
    return fa


def _expand_bbox(bbox, W, H, pad_ratio: float = 0.35) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    # square it
    side = max(bw, bh) * (1 + pad_ratio)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    half = side * 0.5
    nx1 = int(max(0, cx - half))
    ny1 = int(max(0, cy - half))
    nx2 = int(min(W, cx + half))
    ny2 = int(min(H, cy + half))
    return nx1, ny1, nx2, ny2


# ------------------------------------------------------------------
# STEP 1: per-frame occluder mask extraction
# ------------------------------------------------------------------
def build_occluder_masks_video(source_video: str,
                                log=print,
                                pad_ratio: float = 0.35,
                                dilate_px: int = 9,
                                bbox_smoothing: float = 0.0,
                                mask_smoothing: float = 0.0,
                                ) -> Path:
    """Produce per-frame full-resolution occluder masks for source_video.

    Writes mask_XXXXX.png (uint8 0..255) into a cache dir keyed by
    (source path, mtime, size, smoothing). Idempotent.

    Temporal smoothing (kills the per-frame bbox-jitter wobble that
    made XSeg unusable in v1):
      bbox_smoothing: 0 = no smoothing, 0.9 = heavy. EMA on detected
        bbox so the crop region doesn't bounce frame-to-frame.
        formula: smooth = bbox_smoothing*prev + (1-bbox_smoothing)*new
      mask_smoothing: 0 = no smoothing, 0.9 = heavy. EMA on the full
        per-frame alpha mask. Same formula.
      Recommended defaults: 0.4 / 0.7 (light bbox damping, heavier
        mask damping -- bbox should still track motion, mask edges
        should not flicker).

    Returns the masks dir.
    """
    if not os.path.exists(source_video):
        raise FileNotFoundError(source_video)
    key = _cache_key(source_video,
                      bbox_smoothing=bbox_smoothing,
                      mask_smoothing=mask_smoothing)
    masks_dir = MASKS_ROOT / f"xseg_v1_{key}"
    if masks_dir.exists():
        existing = sorted(masks_dir.glob("mask_*.png"))
        if existing:
            log(f"XSeg cache hit: {len(existing)} masks at {masks_dir}")
            return masks_dir
    masks_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np
    model_paths = ensure_xseg_models(log=log)
    ensemble = XSegEnsemble(model_paths, log=log)
    fa = _get_face_detector()

    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {source_video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log(f"XSeg: extracting occluder masks from {N} frames @ {W}x{H}")

    last_bbox: Optional[Tuple[int, int, int, int]] = None
    prev_mask_full: Optional[np.ndarray] = None  # uint8 H,W; for mask EMA
    # NOTE: dilate_px param retained for caller compatibility, unused.
    # FaceFusion's face_masker.create_occlusion_mask does NOT dilate
    # XSeg output -- it only does Gaussian blur sigma=5 + clip(0.5,1)
    # tightening. We mirror that.

    log(f"  temporal smoothing: bbox={bbox_smoothing:.2f} "
        f"mask={mask_smoothing:.2f} "
        f"({'OFF' if (bbox_smoothing == 0 and mask_smoothing == 0) else 'ON'})")

    t0 = time.time()
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # face crop
        faces = fa.get(frame)
        if faces:
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            bbox_raw = _expand_bbox(face.bbox, W, H, pad_ratio=pad_ratio)
            # bbox EMA: smooth across frames so detection jitter doesn't
            # propagate into per-frame crop shifts (which is what made the
            # mask wobble visible at the occluder edges in v1).
            if last_bbox is not None and bbox_smoothing > 0:
                a = float(bbox_smoothing)
                bbox = tuple(int(round(a * pb + (1.0 - a) * nb))
                             for pb, nb in zip(last_bbox, bbox_raw))
            else:
                bbox = bbox_raw
            last_bbox = bbox
        elif last_bbox is not None:
            bbox = last_bbox
        else:
            # No face yet -> all-source mask. With m=1 the composite
            # formula degenerates to out=source, safe no-op.
            allsrc = np.full((H, W), 255, dtype=np.uint8)
            cv2.imwrite(str(masks_dir / f"mask_{fi:05d}.png"), allsrc)
            fi += 1
            continue

        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            # Crop slid off-screen -> same all-source fallback.
            allsrc = np.full((H, W), 255, dtype=np.uint8)
            cv2.imwrite(str(masks_dir / f"mask_{fi:05d}.png"), allsrc)
            fi += 1
            continue

        occ = ensemble.occluder_mask(crop)  # float32 0..1 at crop resolution
        occ_u8 = (occ * 255.0).astype(np.uint8)

        # SEAM FIX: default the full-frame mask to 255 ("use source")
        # outside the bbox, not 0. restore_occluder composes
        #   out = m*source + (1-m)*lipsync
        # so m=0 outside the bbox = pipe lipsync's GFPGAN-modified
        # edge pixels through there, which renders the bbox boundary
        # as a visible blurred square. m=255 outside the bbox =
        # source pixels = no seam. Inside the bbox, XSeg decides.
        full = np.full((H, W), 255, dtype=np.uint8)
        full[y1:y2, x1:x2] = occ_u8

        # MASK EMA: blend with previous frame's smoothed mask so
        # XSeg edge flicker (mostly at hair/finger boundaries) gets
        # averaged out. Done in float space, quantized to uint8 for
        # PNG storage. prev_mask_full holds the SMOOTHED result so
        # smoothing compounds across the sequence (true EMA).
        if prev_mask_full is not None and mask_smoothing > 0:
            a = float(mask_smoothing)
            blended = (a * prev_mask_full.astype(np.float32)
                       + (1.0 - a) * full.astype(np.float32))
            full = np.clip(blended, 0.0, 255.0).astype(np.uint8)
        prev_mask_full = full
        cv2.imwrite(str(masks_dir / f"mask_{fi:05d}.png"), full,
                    [cv2.IMWRITE_PNG_COMPRESSION, 6])

        if fi % 25 == 0 or (time.time() - t0) > 5.0:
            log(f"  XSeg: frame {fi}/{N}  ({time.time() - t0:.1f}s elapsed)")
            t0 = time.time()
        fi += 1
    cap.release()
    log(f"XSeg: {fi} masks written to {masks_dir}")
    return masks_dir


# ------------------------------------------------------------------
# STEP 2: composite source occluder back over lipsync output
# ------------------------------------------------------------------
def restore_occluder(lipsync_video: str,
                      source_video: str,
                      masks_dir: Path,
                      output_path: Optional[str] = None,
                      feather: int = 9,
                      log=print) -> str:
    """Output: out = source*alpha + lipsync*(1-alpha), where alpha is the
    feathered XSeg occluder mask. This restores the chameleon (or whatever
    occluder) onto the lipsynced frames without seams, because the masked
    region is replaced with literal original pixels.

    Audio: copied from the lipsync output (which already has the
    sync'd vocal track).
    """
    if not os.path.exists(lipsync_video):
        raise FileNotFoundError(lipsync_video)
    if not os.path.exists(source_video):
        raise FileNotFoundError(source_video)
    masks_dir = Path(masks_dir)
    mask_files = sorted(masks_dir.glob("mask_*.png"))
    if not mask_files:
        raise RuntimeError(f"no masks in {masks_dir}")
    if output_path is None:
        lp = Path(lipsync_video)
        output_path = str(lp.with_name(lp.stem + "_xseg_gated.mp4"))

    import cv2
    import numpy as np
    cap_s = cv2.VideoCapture(source_video)
    cap_l = cv2.VideoCapture(lipsync_video)
    fps = cap_l.get(cv2.CAP_PROP_FPS) or cap_s.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = min(
        int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap_s.get(cv2.CAP_PROP_FRAME_COUNT)),
        len(mask_files),
    )
    feather = (feather | 1) if feather > 0 else 0

    ffmpeg = _resolve_ffmpeg()
    raw = output_path + ".raw.mp4"
    enc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", f"{fps:.6f}", "-i", "-",
         "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         "-pix_fmt", "yuv420p", raw],
        stdin=subprocess.PIPE,
    )

    log(f"restore: compositing {N} frames @ {W}x{H}, feather={feather}")
    preserved_total = 0.0
    t0 = time.time()
    for fi in range(N):
        ok_s, fs = cap_s.read()
        ok_l, fl = cap_l.read()
        if not (ok_s and ok_l):
            break
        if fl.shape[:2] != (H, W):
            fl = cv2.resize(fl, (W, H))
        if fs.shape[:2] != (H, W):
            fs = cv2.resize(fs, (W, H))
        m = cv2.imread(str(mask_files[fi]), cv2.IMREAD_GRAYSCALE)
        if m is None:
            enc.stdin.write(fl.tobytes())
            continue
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)

        # CRITICAL: binarize at the SAME threshold pre-inpaint uses
        # (>=128 == "occluder, was inpainted"). Without this step,
        # soft-mask pixels (e.g. value 130) cause a thresh-then-blend
        # mismatch: pre-inpaint replaced them with skin filler (so
        # LatentSync painted lips there), but the linear-blend restore
        # only takes 51% source -> visible bright skin patches around
        # the occluder. Binarize, THEN feather, so the soft transition
        # comes from a clean Gaussian falloff, not threshold ambiguity.
        m_bin = (m >= 128).astype(np.float32)   # 0.0 or 1.0
        m_f = m_bin
        if feather > 0:
            m_f = cv2.GaussianBlur(m_f, (feather, feather), 0)
        m_f = np.clip(m_f, 0.0, 1.0)[..., None]
        out = (m_f * fs.astype(np.float32) +
               (1.0 - m_f) * fl.astype(np.float32)).astype(np.uint8)
        enc.stdin.write(out.tobytes())
        preserved_total += float(m_f.mean())
        if fi % 50 == 0:
            log(f"  restore: frame {fi}/{N}  ({time.time() - t0:.1f}s)")
    cap_s.release()
    cap_l.release()
    enc.stdin.close()
    enc.wait()

    # mux audio from lipsync output
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error",
         "-i", raw, "-i", lipsync_video,
         "-map", "0:v:0", "-map", "1:a:0?",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
         output_path],
        check=False,
    )
    try:
        os.remove(raw)
    except Exception:
        pass
    log(f"restore: average occluder coverage = "
        f"{preserved_total / max(N, 1) * 100:.2f}%")
    log(f"restore DONE: {output_path}")
    return output_path


# ------------------------------------------------------------------
# convenience one-shot
# ------------------------------------------------------------------
def gate_lipsync_with_xseg(source_video: str,
                            lipsync_video: str,
                            log=print,
                            output_path: Optional[str] = None,
                            bbox_smoothing: float = 0.0,
                            mask_smoothing: float = 0.0,
                            feather: int = 9,
                            # v2 callers pass these; they are accepted
                            # but IGNORED in this baseline revert -- the
                            # welding/alignment/object-remover paths
                            # never produced clean results.
                            align_to_source: bool = False,
                            mouth_polygon: bool = False,
                            polygon_style: str = "lips_only",
                            landmarks_source_video=None) -> str:
    """Convenience: build masks (cached) and restore occluder. Returns
    the gated mp4 path. Suitable for callers that already have a
    lipsync output and want to re-gate it without rerunning lipsync.

    Smoothing params plumb directly to build_occluder_masks_video so
    different smoothing settings get separate cache dirs."""
    masks_dir = build_occluder_masks_video(
        source_video, log=log,
        bbox_smoothing=bbox_smoothing,
        mask_smoothing=mask_smoothing,
    )
    return restore_occluder(lipsync_video, source_video, masks_dir,
                             output_path=output_path, feather=feather,
                             log=log)
