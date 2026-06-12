"""Video face-swap pipeline (paste SOURCE identity onto TARGET video).

Thin typed wrapper over core.pipeline.FaceSwapPipeline. The legacy app
calls FaceSwapPipeline directly with a dict; v2 builds that dict from
a VideoSwapJob dataclass so callers don't have to know the dict shape.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

from .config import VideoSwapJob
from .paths import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "recordings" / "video_swap"


# Same codes legacy v1 sent (see core.pipeline OUTPUT_QUALITY).
# Other strings get rejected / silently downgraded by FaceSwapPipeline.
_QUALITY_MAP = {
    "visually_lossless": "visually_lossless",
    "balanced":          "balanced",
    "lossless":          "lossless",
}


def _build_cfg(job: VideoSwapJob) -> dict:
    """Translate VideoSwapJob -> the config dict FaceSwapPipeline takes."""
    cfg = {
        "blending":     {
            "method": job.blend_method,
            "mask_padding": int(job.mask_padding),
            "mask_blur": float(job.mask_blur),
        },
        "optimization": {"cuda_device": int(job.gpu_id)},
        "identity": {
            "drift_threshold": float(job.det_threshold),
            "swap_strength": float(job.swap_strength),
        },
        # NOTE: legacy v1 passed only {"threshold": ...}. Adding
        # other keys (like input_size) can silently mis-configure
        # FaceSwapPipeline's detector and cause blurry output.
        "detection":    {"threshold": float(job.det_threshold)},
        "output": {
            "quality": _QUALITY_MAP.get(job.output_quality,
                                          "visually_lossless"),
        },
        "mask_gate": {
            # T2-NEW: rotoscope mask region restriction.  Pipeline
            # loads this stack at init time and filters detected faces
            # per frame by bbox-centroid containment.
            "npy_path": str(getattr(job, "mask_npy_path", "") or ""),
        },
        "enhancement": {
            # T2-2: only request in-pipeline GFPGAN if the chosen
            # restorer IS GFPGAN.  Other restorers run as a video-
            # level post-process after the pipeline returns.
            "method": (
                "gfpgan"
                if (job.enhance_faces
                    and str(getattr(job, "face_restorer", "gfpgan"))
                        .lower() == "gfpgan")
                else "none"
            ),
            "blend": float(job.enhancer_blend),
        },
        "selector": {
            "mode": str(job.selector_mode),
            "reference_path": (str(job.reference_face_image)
                                if job.reference_face_image else None),
            "reference_distance": float(job.reference_distance),
        },
        "swap": {
            "pixel_boost": int(job.pixel_boost),
        },
        # Temporal smoothing runs by default in core/pipeline._stage_temporal.
        # Knobs are surfaced via VideoSwapJob so the UI can tune them.
        "temporal": {
            "enabled":     bool(job.temporal_enabled),
            "ema_decay":   float(job.temporal_ema_decay),
            "buffer_size": int(job.temporal_buffer_size),
        },
        # Lighting / color match runs by default in pipeline._stage_lighting.
        # Reinhard LAB color transfer + SH-relit shadow correction.
        "lighting": {
            "color_transfer":    str(job.color_transfer_mode),
            "shadow_correction": bool(job.shadow_correction),
            "shadow_clamp_min":  float(job.shadow_clamp_min),
            "shadow_clamp_max":  float(job.shadow_clamp_max),
        },
    }
    if int(job.trim_end_frame) > 0:
        cfg["trim"] = {
            "start_frame": int(job.trim_start_frame),
            "end_frame":   int(job.trim_end_frame),
        }
    return cfg


def run(job: VideoSwapJob,
        log: Callable[[str], None] = print) -> Path:
    """Execute the swap. Returns the output mp4 path."""
    job.validate()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)
    out_path = OUT_DIR / f"faceswap_video_v2_{stamp}.mp4"

    from core.pipeline import FaceSwapPipeline
    cfg = _build_cfg(job)
    log(f"[video-swap] source={Path(job.source_image).name} "
        f"target={Path(job.target_video).name}")
    log(f"[video-swap] blend={job.blend_method} enhance={job.enhance_faces} "
        f"det={job.det_size_px}px thresh={job.det_threshold}")
    if int(job.trim_end_frame) > 0:
        log(f"[video-swap] trim frames "
            f"{job.trim_start_frame}-{job.trim_end_frame}")

    if job.source_image_b is not None:
        # Identity blend or embedding journey via pipeline_blend.
        from core import pipeline_blend
        if job.journey_mode:
            log(f"[video-swap] mode = embedding JOURNEY "
                f"(alpha {job.journey_start_alpha:.2f}->"
                f"{job.journey_end_alpha:.2f}, "
                f"curve={job.journey_curve})")
            pipeline_blend.run_journey(
                source_a=str(job.source_image),
                source_b=str(job.source_image_b),
                target_video=str(job.target_video),
                output_path=str(out_path),
                cfg=cfg,
                start_alpha=float(job.journey_start_alpha),
                end_alpha=float(job.journey_end_alpha),
                curve=str(job.journey_curve),
                log=log,
            )
        else:
            log(f"[video-swap] mode = identity BLEND "
                f"(alpha={job.blend_alpha:.2f})")
            pipeline_blend.run_blend(
                source_a=str(job.source_image),
                source_b=str(job.source_image_b),
                alpha=float(job.blend_alpha),
                target_video=str(job.target_video),
                output_path=str(out_path),
                cfg=cfg,
                log=log,
            )
    else:
        pipeline = FaceSwapPipeline(cfg)
        pipeline.run(
            source_path=str(job.source_image),
            input_path=str(job.target_video),
            output_path=str(out_path),
            verbose=False,
        )
    if not out_path.exists() or out_path.stat().st_size < 10_000:
        raise RuntimeError(
            f"video swap produced no output at {out_path}")

    # T2-2 post-process restoration if user picked a non-GFPGAN backend.
    restorer = str(getattr(job, "face_restorer", "gfpgan")).lower()
    if job.enhance_faces and restorer not in ("none", "gfpgan"):
        try:
            from core import face_restoration as _fr
            log(f"[video-swap] post-process restoration backend={restorer}")
            out_path = _fr.enhance(restorer, out_path, log=log)
        except Exception as exc:
            log(f"[video-swap] post-process restoration FAILED "
                f"({exc}); shipping un-restored output")

    log(f"[video-swap] DONE -> {out_path}")
    return out_path


def preview_one_frame(source_image: Path, target_video: Path,
                       frame_idx: int, blend_method: str = "poisson",
                       enhance_faces: bool = False,
                       gpu_id: int = 0, det_threshold: float = 0.5,
                       log: Callable[[str], None] = print):
    """Return a BGR numpy array of one swapped frame for live preview."""
    if not Path(source_image).is_file():
        raise FileNotFoundError(source_image)
    if not Path(target_video).is_file():
        raise FileNotFoundError(target_video)

    from core.pipeline import FaceSwapPipeline
    cfg = {
        "blending":     {"method": blend_method},
        "optimization": {"cuda_device": int(gpu_id)},
        "identity":     {"drift_threshold": float(det_threshold)},
        "detection":    {"threshold": float(det_threshold)},
        "output":       {"quality": "visually_lossless"},
        "enhancement":  {"method": "gfpgan" if enhance_faces else "none"},
    }
    log(f"[video-swap-preview] frame={frame_idx}")
    pipeline = FaceSwapPipeline(cfg)
    return pipeline.preview_frame(
        source_path=str(source_image),
        target_video_path=str(target_video),
        frame_idx=int(frame_idx),
    )
