"""Post-render video transforms: watermark overlay + aspect-ratio
conversion. Both via ffmpeg subprocess (no opencv, no GPU). Each
takes an input mp4 + a typed config and returns an output mp4 path.

Stages run AFTER lipsync + GFPGAN, BEFORE the audio re-mux step.
So watermark is burnt onto every frame; aspect crop/pad reshapes
the canvas before the audio gets muxed.
"""
from __future__ import annotations

import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from .config import AspectRatioConfig, WatermarkConfig
from .ffmpeg_tools import resolve_ffmpeg
from .paths import RECORDINGS_DIR

logger = logging.getLogger(__name__)


# ----- WATERMARK --------------------------------------------------
def _probe_video_width(video_in: Path) -> int:
    """Return the first video stream's width in pixels.

    Uses `ffmpeg -i` (no separate ffprobe binary needed -- ffmpeg
    prints stream info to stderr on null mux). Returns 0 on parse
    failure so the caller can fall back to a sane default.
    """
    ff = resolve_ffmpeg()
    r = subprocess.run(
        [ff, "-hide_banner", "-i", str(video_in), "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="ignore",
    )
    # ffmpeg prints e.g.  "Stream #0:0: Video: h264 ... 1920x1080 ..."
    m = re.search(r"Video:[^\n]*?,\s+(\d{2,5})x(\d{2,5})", r.stderr or "")
    if not m:
        return 0
    return int(m.group(1))


def _overlay_filter(pos: str, target_w_px: int, opacity: float) -> str:
    """Build the ffmpeg -filter_complex string for a watermark overlay.

    target_w_px is the ABSOLUTE pixel width the watermark should be
    scaled to (the caller has already multiplied scale_pct by the
    main video width). We use a plain `scale=W:-1` on the watermark
    input -- no scale2ref, because scale2ref was removed in ffmpeg
    7.0 and silently dropped our video stream on machines with the
    new build, producing audio-only output mp4s.

    Position keys:  TL  TR  BL  BR  CENTER
    """
    alpha = max(0.05, min(1.0, float(opacity) / 100.0))   # 5..100%
    w = max(1, int(target_w_px))

    pos_map = {
        "TL":     "10:10",
        "TR":     "main_w-overlay_w-10:10",
        "BL":     "10:main_h-overlay_h-10",
        "BR":     "main_w-overlay_w-10:main_h-overlay_h-10",
        "CENTER": "(main_w-overlay_w)/2:(main_h-overlay_h)/2",
    }
    where = pos_map.get(pos.upper(), pos_map["BR"])

    # Chain: scale watermark to absolute width, apply alpha, overlay.
    return (
        f"[1:v]scale={w}:-1,"
        f"format=rgba,colorchannelmixer=aa={alpha:.3f}[wm];"
        f"[0:v][wm]overlay={where}[out]"
    )


def apply_watermark(video_in: Path, cfg: WatermarkConfig,
                     log=print) -> Path:
    """Overlay cfg.image_path onto video_in. Returns new mp4 path.
    Skips (returns input) if cfg.enabled is False or image missing."""
    if not cfg.enabled or not cfg.image_path:
        return Path(video_in)
    img_p = Path(cfg.image_path)
    if not img_p.is_file():
        log(f"[watermark] image not found: {img_p}; skipping")
        return Path(video_in)

    ff = resolve_ffmpeg()
    out_path = Path(video_in).with_name(
        Path(video_in).stem + "_wm.mp4")

    # Probe the main video's width so we can compute an ABSOLUTE
    # pixel width for the watermark. This replaces the old scale2ref
    # approach (removed in ffmpeg 7.0).
    main_w_px = _probe_video_width(Path(video_in))
    if main_w_px <= 0:
        # Fallback: assume 1280 px (typical 720p) -- still better than
        # the previous bug (38% of watermark's own width).
        log(f"[watermark] WARN could not probe main video width; "
            f"assuming 1280")
        main_w_px = 1280
    sw = max(0.01, min(0.95, float(cfg.scale_pct) / 100.0))
    target_w_px = max(1, int(round(main_w_px * sw)))

    filt = _overlay_filter(cfg.position, target_w_px, cfg.opacity)
    log(f"[watermark] applying {img_p.name} "
        f"pos={cfg.position} scale={cfg.scale_pct:.0f}% "
        f"(main_w={main_w_px}px -> wm_w={target_w_px}px) "
        f"opacity={cfg.opacity:.0f}%")
    r = subprocess.run([
        ff, "-y", "-loglevel", "error",
        "-i", str(video_in),
        "-i", str(img_p),
        "-filter_complex", filt,
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(out_path),
    ], capture_output=True, text=True, encoding="utf-8",
       errors="ignore")
    if r.returncode != 0 or not out_path.exists():
        log(f"[watermark] FAILED: {(r.stderr or '')[-400:]}; "
            "using un-watermarked output")
        return Path(video_in)
    return out_path


# ----- ASPECT RATIO -----------------------------------------------
_ASPECT_MAP = {
    "16:9":          (1920, 1080),     # YouTube horizontal
    "9:16":          (1080, 1920),     # TikTok / Reels / Shorts
    "1:1":           (1080, 1080),     # Instagram feed
    "3:4 portrait":  (1080, 1440),     # Instagram portrait
    "4:5 portrait":  (1080, 1350),     # Instagram portrait alt
    "4:3":           (1440, 1080),     # classic TV
    "21:9":          (2560, 1080),     # cinematic widescreen
}


def list_aspects() -> list:
    return list(_ASPECT_MAP.keys())


def _aspect_filter(target_w: int, target_h: int, mode: str) -> str:
    """Build ffmpeg -vf to fit input into target_w x target_h.

    mode == 'crop' : scale up to fill, then center-crop the excess
    mode == 'pad'  : scale down to fit, then pad with black bars
    """
    if mode == "pad":
        return (
            f"scale='if(gt(a,{target_w}/{target_h}),"
            f"{target_w},-2)':'"
            f"if(gt(a,{target_w}/{target_h}),-2,{target_h})',"
            f"pad={target_w}:{target_h}:"
            f"(ow-iw)/2:(oh-ih)/2:black"
        )
    # crop (default): scale UP so the smaller axis fills, then crop
    return (
        f"scale='if(gt(a,{target_w}/{target_h}),"
        f"-2,{target_w})':'"
        f"if(gt(a,{target_w}/{target_h}),{target_h},-2)',"
        f"crop={target_w}:{target_h}"
    )


def apply_aspect_ratio(video_in: Path, cfg: AspectRatioConfig,
                       log=print) -> Path:
    """Reshape video_in to cfg.target_aspect via crop or pad.
    Skips if cfg.enabled is False or target unrecognized."""
    if not cfg.enabled or cfg.target_aspect == "(keep original)":
        return Path(video_in)
    dims = _ASPECT_MAP.get(cfg.target_aspect)
    if not dims:
        log(f"[aspect] unknown target {cfg.target_aspect}; skipping")
        return Path(video_in)
    tw, th = dims
    ff = resolve_ffmpeg()
    out_path = Path(video_in).with_name(
        Path(video_in).stem + f"_{cfg.target_aspect.replace(':','x').replace(' ','_')}.mp4")
    vf = _aspect_filter(tw, th, cfg.fill_mode)
    log(f"[aspect] {cfg.target_aspect} ({tw}x{th}) mode={cfg.fill_mode}")
    r = subprocess.run([
        ff, "-y", "-loglevel", "error",
        "-i", str(video_in),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(out_path),
    ], capture_output=True, text=True, encoding="utf-8",
       errors="ignore")
    if r.returncode != 0 or not out_path.exists():
        log(f"[aspect] FAILED: {(r.stderr or '')[-400:]}; "
            "using original aspect")
        return Path(video_in)
    return out_path

# ----- PREVIEW (frame-0 dry run) ----------------------------------
_ASPECT_MAP_W_H = _ASPECT_MAP   # alias for clarity inside preview


def preview_output_frame(face_video_path, watermark_cfg=None,
                          aspect_cfg=None):
    """Return a PIL.Image showing what the FINAL frame will look like
    after the post-process steps: aspect reshape then watermark.
    Reads frame 0 of face_video_path; applies the two transforms
    in-memory (no ffmpeg subprocess) so the preview is sub-second.

    Either cfg may be None or .enabled=False; that step is skipped.
    Returns None if the video can't be read.
    """
    from PIL import Image
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    if not face_video_path:
        return None
    cap = cv2.VideoCapture(str(face_video_path))
    if not cap.isOpened():
        return None
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        return None

    # ----- Stage 1: aspect ratio reshape ---------------------------
    if (aspect_cfg is not None and aspect_cfg.enabled
            and aspect_cfg.target_aspect in _ASPECT_MAP_W_H):
        tw, th = _ASPECT_MAP_W_H[aspect_cfg.target_aspect]
        h, w = frame_bgr.shape[:2]
        src_ar = w / max(1, h)
        dst_ar = tw / th
        if aspect_cfg.fill_mode == "pad":
            # Scale to fit; pad with black
            if src_ar > dst_ar:
                new_w = tw
                new_h = int(round(h * tw / w))
            else:
                new_h = th
                new_w = int(round(w * th / h))
            scaled = cv2.resize(frame_bgr, (new_w, new_h),
                                interpolation=cv2.INTER_AREA)
            canvas = np.zeros((th, tw, 3), dtype=np.uint8)
            y0 = (th - new_h) // 2
            x0 = (tw - new_w) // 2
            canvas[y0:y0 + new_h, x0:x0 + new_w] = scaled
            frame_bgr = canvas
        else:
            # crop: scale up so the smaller axis fills, then center-crop
            if src_ar > dst_ar:
                new_h = th
                new_w = int(round(w * th / h))
            else:
                new_w = tw
                new_h = int(round(h * tw / w))
            scaled = cv2.resize(frame_bgr, (new_w, new_h),
                                interpolation=cv2.INTER_AREA)
            y0 = max(0, (new_h - th) // 2)
            x0 = max(0, (new_w - tw) // 2)
            frame_bgr = scaled[y0:y0 + th, x0:x0 + tw]

    # ----- Convert to PIL for the watermark overlay ----------------
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(rgb).convert("RGBA")

    # ----- Stage 2: watermark overlay ------------------------------
    # NOTE: any failure here is LOGGED -- no silent swallowing. Silent
    # passes here cost a debug session ("watermark doesn't show up but
    # no error either" => preview-broken-but-looks-fine).
    if watermark_cfg is None or not watermark_cfg.enabled:
        print(f"[preview] watermark skipped: cfg={watermark_cfg!r}",
              flush=True)
    elif not watermark_cfg.image_path:
        print("[preview] watermark skipped: image_path is empty "
              "(did you upload a PNG?)", flush=True)
    else:
        wm_path = Path(watermark_cfg.image_path)
        if not wm_path.is_file():
            print(f"[preview] watermark skipped: file not found at "
                  f"{wm_path}", flush=True)
        else:
            try:
                wm = Image.open(wm_path).convert("RGBA")
                sw = max(0.01, min(0.95,
                                   float(watermark_cfg.scale_pct) / 100.0))
                new_w = max(8, int(base.width * sw))
                new_h = max(8, int(wm.height * new_w / wm.width))
                wm = wm.resize((new_w, new_h), Image.LANCZOS)
                alpha_mult = max(0.05, min(1.0,
                                           float(watermark_cfg.opacity) / 100.0))
                if alpha_mult < 1.0:
                    a = wm.split()[-1]
                    a = a.point(lambda v, m=alpha_mult: int(v * m))
                    wm.putalpha(a)
                pos = (watermark_cfg.position or "BR").upper()
                margin = 10
                if pos == "TL":
                    x, y = margin, margin
                elif pos == "TR":
                    x, y = base.width - new_w - margin, margin
                elif pos == "BL":
                    x, y = margin, base.height - new_h - margin
                elif pos == "CENTER":
                    x, y = ((base.width - new_w) // 2,
                            (base.height - new_h) // 2)
                else:    # BR (default)
                    x = base.width - new_w - margin
                    y = base.height - new_h - margin
                base.alpha_composite(wm, dest=(x, y))
                print(f"[preview] watermark composited OK: "
                      f"src={wm_path.name} size={new_w}x{new_h} "
                      f"pos={pos} opacity={alpha_mult:.2f} at ({x},{y})",
                      flush=True)
            except Exception:
                import traceback
                print("[preview] watermark composite FAILED:", flush=True)
                traceback.print_exc()

    return base.convert("RGB")
