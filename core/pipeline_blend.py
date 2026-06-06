"""Embedding-blend and embedding-journey face-swap modes.

Both modes work in InsightFace's ArcFace embedding space (512-d). The
embedding space is approximately linear with respect to perceptual
identity, so weighted averages of normalized embeddings produce hybrid
identities that the inswapper_128 decoder renders coherently.

Two modes:
  - BLEND (static alpha): output = inswap((1-alpha)*emb_A + alpha*emb_B)
    Same identity used for every frame. Use case: pick the best fixed
    mix of two source images.
  - JOURNEY (alpha ramps across timeline): alpha varies per frame from
    0 to 1 (or any range). Subject morphs from source_a to source_b
    continuously across the clip. Use case: creative effects, identity
    crossfade.

This module is a thin orchestration layer; the heavy lifting still
runs through core.pipeline.FaceSwapPipeline via the
source_embedding_override and per_frame_embedding_fn kwargs added in
that module's surgical patch.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Embedding math
# ----------------------------------------------------------------------
def _normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a (512,) vector. Returns input unchanged if zero-norm."""
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v
    return (v / n).astype(np.float32)


def blend_embeddings(emb_a: np.ndarray, emb_b: np.ndarray,
                      alpha: float) -> np.ndarray:
    """Weighted average of two ArcFace embeddings.

    alpha=0 -> pure A, alpha=1 -> pure B, 0.5 -> 50/50 hybrid.
    Result is L2-normalized to keep it on the embedding manifold
    inswapper_128 was trained against.
    """
    alpha = float(max(0.0, min(1.0, alpha)))
    mix = (1.0 - alpha) * emb_a + alpha * emb_b
    return _normalize(mix)


def _smoothstep(t: float) -> float:
    """Hermite ease-in-out S-curve on [0, 1]. 3t^2 - 2t^3."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def make_journey_fn(emb_a: np.ndarray, emb_b: np.ndarray,
                     start_alpha: float = 0.0,
                     end_alpha: float = 1.0,
                     curve: str = "linear") -> Callable[[int, int], np.ndarray]:
    """Build the per_frame_embedding_fn that pipeline.run consumes.

    Args:
        emb_a, emb_b: the two anchor embeddings.
        start_alpha: alpha at frame 0 (default 0.0 = pure A).
        end_alpha:   alpha at last frame (default 1.0 = pure B).
        curve: "linear" or "smoothstep" (ease-in-out).

    Returns:
        fn(frame_idx, total_frames) -> (512,) np.float32 embedding.
    """
    if curve not in ("linear", "smoothstep"):
        raise ValueError(f"unknown curve: {curve}")

    def fn(frame_idx: int, total_frames: int) -> np.ndarray:
        denom = max(int(total_frames) - 1, 1)
        t = max(0, int(frame_idx)) / denom
        if curve == "smoothstep":
            t = _smoothstep(t)
        alpha = (1.0 - t) * start_alpha + t * end_alpha
        return blend_embeddings(emb_a, emb_b, alpha)

    return fn


# ----------------------------------------------------------------------
# High-level entry points
# ----------------------------------------------------------------------
def _build_pipeline(cfg: dict):
    """Construct + initialize a FaceSwapPipeline so we can call its
    _extract_source_embedding method on both source images."""
    from core.pipeline import FaceSwapPipeline
    pipeline = FaceSwapPipeline(cfg)
    pipeline._init_modules()  # noqa: SLF001 -- private but needed
    return pipeline


def extract_embedding(pipeline, source_path: str) -> np.ndarray:
    """Re-use FaceSwapPipeline's detector to pull a (512,) ArcFace
    embedding out of a single source image. Wraps the private
    _extract_source_embedding method for callers outside the pipeline.
    """
    return pipeline._extract_source_embedding(str(source_path))  # noqa: SLF001


def run_blend(source_a: str, source_b: str, alpha: float,
              target_video: str, output_path: str,
              cfg: dict,
              log: Callable[[str], None] = print) -> Path:
    """Static-blend face-swap.

    The blended embedding is computed once and held constant for every
    frame. alpha=0 yields A, alpha=1 yields B.
    """
    log(f"[blend] source_a={Path(source_a).name} "
        f"source_b={Path(source_b).name} alpha={alpha:.3f}")
    pipeline = _build_pipeline(cfg)
    emb_a = extract_embedding(pipeline, source_a)
    emb_b = extract_embedding(pipeline, source_b)
    log(f"[blend] embeddings extracted "
        f"(norms A={float(np.linalg.norm(emb_a)):.3f} "
        f"B={float(np.linalg.norm(emb_b)):.3f})")
    blended = blend_embeddings(emb_a, emb_b, alpha)
    log(f"[blend] blended embedding norm={float(np.linalg.norm(blended)):.3f}")
    pipeline.run(
        source_path=str(source_a),  # used only for logging now
        input_path=str(target_video),
        output_path=str(output_path),
        source_embedding_override=blended,
    )
    return Path(output_path)


def run_journey(source_a: str, source_b: str,
                target_video: str, output_path: str,
                cfg: dict,
                start_alpha: float = 0.0,
                end_alpha: float = 1.0,
                curve: str = "linear",
                log: Callable[[str], None] = print) -> Path:
    """Embedding-journey face-swap.

    alpha ramps from start_alpha to end_alpha across the timeline.
    With defaults (0 -> 1, linear), the subject morphs cleanly from
    source_a at frame 0 to source_b at the final frame.
    """
    log(f"[journey] source_a={Path(source_a).name} "
        f"source_b={Path(source_b).name} "
        f"alpha=[{start_alpha:.3f}..{end_alpha:.3f}] curve={curve}")
    pipeline = _build_pipeline(cfg)
    emb_a = extract_embedding(pipeline, source_a)
    emb_b = extract_embedding(pipeline, source_b)
    log(f"[journey] embeddings extracted "
        f"(norms A={float(np.linalg.norm(emb_a)):.3f} "
        f"B={float(np.linalg.norm(emb_b)):.3f})")
    fn = make_journey_fn(emb_a, emb_b,
                          start_alpha=start_alpha,
                          end_alpha=end_alpha,
                          curve=curve)
    # Seed source identity with the start-frame embedding so the
    # pipeline's identity_tracker isn't initialized blank.
    seed_emb = blend_embeddings(emb_a, emb_b, start_alpha)
    pipeline.run(
        source_path=str(source_a),
        input_path=str(target_video),
        output_path=str(output_path),
        source_embedding_override=seed_emb,
        per_frame_embedding_fn=fn,
    )
    return Path(output_path)
