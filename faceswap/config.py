"""LipsyncJob -- the typed config object that flows through every stage.

Replaces the old kwarg-soup pattern where ~12 args were threaded through
render_lipsync, run_multiclip, _infer_latentsync, etc. Every pipeline
stage takes a LipsyncJob and returns a LipsyncJob or output path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class LatentSyncKnobs:
    """Inference-time knobs for the LatentSync subprocess."""
    inference_steps: int = 20
    guidance_scale: float = 1.5
    enable_deepcache: bool = True
    seed: int = -1          # -1 = random per render; any int = reproducible
    # Face detector confidence threshold passed through to LatentSync's
    # FaceDetector via the LATENTSYNC_FACE_DET_THRESHOLD env var. 0.5 is
    # upstream default. Raise to ~0.7-0.85 when the source has nearby
    # face-like distractors (animals, statues) to filter them out so the
    # largest-face selector picks the human consistently.
    face_det_threshold: float = 0.5
    # T1-NEW: post-LatentSync face-region Reinhard color match.  Fixes
    # the cyan-source -> orange-cheek VAE drift artifact.  "reinhard"
    # is default-on; "none" disables.  Runs as Stage 2.4 (after
    # optional maskout composite, before audio remux).
    color_match_mode: str = "reinhard"
    # Resolution is FIXED at 512 in v2 -- other values gave unusable
    # quality in production testing.


@dataclass
class VoiceSwap:
    """Optional RVC voice swap applied to the audio before lipsync."""
    model_basename: str = ""        # "" = no swap
    transpose_semitones: int = 0    # pitch shift after conversion


@dataclass
class KeySyncKnobs:
    """KeySync-specific inference knobs.

    face_click_xy: pixel coords in the SOURCE frame where the
                   character's face is. SAM2 propagates from this
                   click to track the face across the entire video,
                   bypassing the human-trained face detector that
                   fails on cartoons / non-human characters.
                   (0, 0) = disabled (auto detection still used).
    """
    face_click_x: int = 0
    face_click_y: int = 0
    face_click_frame: int = 0
    compute_until: int = 45
    # skip_crop=True bypasses KeySync's crop_video.py preprocessing
    # step entirely. Use this to test the model on native-resolution
    # source video (no face-tight crop). Helpful for verifying whether
    # KeySync's lipsync works on a clean human face at original
    # resolution before paying the crop-jitter tax on stylized chars.
    skip_crop: bool = False


@dataclass
class LipsyncJob:
    """Everything one render needs. Build once, pass through stages."""
    face_paths: List[Path]                  # primary + extras (>= 1)
    audio_path: Path

    # Engine choice:
    #   'latentsync' - 512px diffusion, optimized for HUMAN speech
    #   'keysync'    - Imperial College 2025 model, works on NON-HUMAN
    #                  characters (cartoons, stylized art, animals).
    #                  Slower (~15min) and runs in its own venv.
    engine: str = "latentsync"

    # Pipeline switches
    isolate_vocals: bool = True             # Demucs vocal stem
    enhance_faces: bool = True              # GFPGAN post-step
    quick_test: bool = False                # 12s trim for smoke tests
    extend_single: bool = False             # stream_loop a short clip

    # Engine knobs
    latentsync: LatentSyncKnobs = field(default_factory=LatentSyncKnobs)
    keysync: KeySyncKnobs = field(default_factory=KeySyncKnobs)
    voice_swap: VoiceSwap = field(default_factory=VoiceSwap)
    # Occlusion gate (applied AFTER lipsync render, BEFORE GFPGAN
    # enhance + aspect/watermark). FaceFusion-style XSeg matte
    # composites source pixels back over occluded regions (hands,
    # mics, hair). v2 adds temporal smoothing (EMA on bbox + mask)
    # to kill the per-frame wobble that made v1's XSeg unusable.
    occlusion: "OcclusionConfig" = field(
        default_factory=lambda: OcclusionConfig())
    # Mask-out non-face regions via SAM2 (multi-click). See
    # MaskOutConfig below. Forward reference (string annotation +
    # lambda factory) matches the OcclusionConfig pattern because
    # MaskOutConfig is declared after LipsyncJob in this file.
    maskout: "MaskOutConfig" = field(
        default_factory=lambda: MaskOutConfig())
    # Post-process (applied AFTER lipsync + GFPGAN, BEFORE audio remux)
    watermark: "WatermarkConfig" = field(
        default_factory=lambda: WatermarkConfig())
    aspect: "AspectRatioConfig" = field(
        default_factory=lambda: AspectRatioConfig())

    # Derived (populated by stages as they run)
    # effective_audio: the CLEAN audio fed to LatentSync as conditioning
    # (vocals-only when isolation is on; full song otherwise). Used
    # only to drive the model.
    effective_audio: Optional[Path] = None
    # final_audio: the audio that gets MUXED onto the output mp4 (the
    # full song the user wants to hear, including instruments). Set
    # before isolation strips the instrumental from effective_audio.
    final_audio: Optional[Path] = None

    # ---- shape predicates: orchestrator branches on these ----
    @property
    def is_multi_clip(self) -> bool:
        return len(self.face_paths) > 1

    @property
    def is_single_clip(self) -> bool:
        return len(self.face_paths) == 1

    def validate(self) -> None:
        if not self.face_paths:
            raise ValueError("face_paths is empty")
        for p in self.face_paths:
            if not Path(p).is_file():
                raise FileNotFoundError(f"face clip not found: {p}")
        if not Path(self.audio_path).is_file():
            raise FileNotFoundError(f"audio not found: {self.audio_path}")


@dataclass
class VideoSwapJob:
    """One face-swap job: paste SOURCE identity onto every face in TARGET."""
    source_image: Path                       # face to transfer (jpg/png)
    target_video: Path                       # video to swap face in
    # Pipeline knobs
    blend_method: str = "poisson"            # poisson | alpha | feather | none
    enhance_faces: bool = False              # GFPGAN post-process
    gpu_id: int = 0
    # NOTE: det_size_px is kept for back-compat but NOT passed to
    # FaceSwapPipeline -- the pipeline picks its own optimal detector
    # input size. Sending input_size in the config corrupted output
    # quality in v2.0/v2.1 (blurry renders).
    det_size_px: int = 640
    det_threshold: float = 0.5
    output_quality: str = "visually_lossless"
    # Optional trim
    trim_start_frame: int = 0
    trim_end_frame: int = 0                  # 0 = render to end

    # ---- Face selector (FaceFusion #4) ----
    # selector_mode = "largest": legacy behavior, swap the biggest face.
    # selector_mode = "reference": only swap the detected face whose
    #   ArcFace embedding is closest (and within reference_distance)
    #   to the embedding of reference_face_image. Required for any
    #   multi-person video where the user only wants one person swapped.
    selector_mode: str = "largest"               # "largest" | "reference"
    reference_face_image: Optional[Path] = None  # required for "reference" mode
    reference_distance: float = 0.6              # cosine distance threshold

    # ---- Face mask / swap strength / enhancer blend (FaceFusion-style knobs) ----
    # mask_padding: + erodes mask inward, - grows mask outward
    # mask_blur: scale on the auto-computed gaussian feather (1.0 = stock)
    # swap_strength: 0..1 LERP between original face (0) and full swap (1)
    # enhancer_blend: 0..1 LERP between un-enhanced (0) and full GFPGAN (1)
    mask_padding: int = 0
    mask_blur: float = 1.0
    swap_strength: float = 1.0
    enhancer_blend: float = 1.0
    # pixel_boost: 128 (native, off) | 256 | 384 | 512 | 768.
    # Post-swap GFPGAN upscale on the swap crop before paste-back.
    # Higher = finer detail at close-up shots, but slower (extra
    # GFPGAN call per face). 128 is identical to the legacy path.
    pixel_boost: int = 128

    # ---- Temporal smoothing (face-region EMA in core/pipeline._stage_temporal) ----
    # Already runs by default with ema_decay=0.85 (pipeline.py defaults);
    # surfaced here so the user can dial it to taste from the UI.
    # temporal_enabled:    False = skip temporal stage entirely (no smoothing).
    # temporal_ema_decay:  0..0.99; higher = stronger frame-to-frame smoothing
    #   (kills flicker but adds lag on fast motion).  0.85 = stock.
    # temporal_buffer_size: number of frames kept for optical-flow context.
    temporal_enabled: bool = True
    temporal_ema_decay: float = 0.85
    temporal_buffer_size: int = 5

    # ---- Lighting / color match (lighting/*, runs in pipeline._stage_lighting) ----
    # All three sub-stages run by default; surfaced for per-render tuning.
    # color_transfer_mode:
    #   "reinhard" = match swap LAB-color stats to original face region (stock)
    #   "none"     = skip color matching entirely
    # shadow_correction: apply SH-relit shadow map from original face to swap.
    # shadow_clamp_min/max: multiplicative shadow-strength bounds.  1.0=neutral.
    color_transfer_mode: str = "reinhard"
    shadow_correction: bool = True
    shadow_clamp_min: float = 0.5
    shadow_clamp_max: float = 1.5

    # ---- Identity blend / journey (idea #3 + #4) ----
    # When source_image_b is set, the swap runs in embedding-blend mode.
    # The ArcFace embedding fed to inswapper_128 becomes
    #   (1-alpha)*emb_A + alpha*emb_B  (L2 normalized).
    # journey_mode=True turns this into an embedding-journey: alpha
    # ramps from journey_start_alpha at frame 0 to journey_end_alpha
    # at the final frame, producing a continuous identity morph
    # across the timeline. curve="linear" or "smoothstep".
    source_image_b: Optional[Path] = None     # None = no blend (legacy)
    blend_alpha: float = 0.5                  # static blend mix; ignored when journey_mode=True
    journey_mode: bool = False
    journey_start_alpha: float = 0.0
    journey_end_alpha: float = 1.0
    journey_curve: str = "linear"             # "linear" or "smoothstep"

    # ---- Region restriction via rotoscope mask (T2-NEW, 2026-06-11) ----
    # Path to a (N, H, W) uint8 NPY stack produced by the Rotoscoping tab
    # ("Send masks" button).  When set + the file is valid, the face
    # detector's output is filtered per frame: faces whose bbox centroid
    # falls outside mask[frame_idx]>0 are dropped before the selector
    # picks one.  This is the real fix for the multi-face problem
    # (e.g. girl's face also applied to a gorilla beside her).
    # None or empty string -> no gating (legacy behavior).
    mask_npy_path: Optional[str] = None

    # ---- Face restoration backend (T2-2) ----
    # "gfpgan"        : default; in-pipeline GFPGAN at the per-frame stage
    # "codeformer"    : CodeFormer post-process (skips in-pipeline GFPGAN)
    # "restoreformer" : RestoreFormer++ post-process (stub until installed)
    # "none"          : no restoration at all (overrides enhance_faces=True)
    face_restorer: str = "gfpgan"

    def validate(self) -> None:
        if not Path(self.source_image).is_file():
            raise FileNotFoundError(
                f"source image not found: {self.source_image}")
        if not Path(self.target_video).is_file():
            raise FileNotFoundError(
                f"target video not found: {self.target_video}")
        if self.blend_method not in ("poisson", "alpha", "feather",
                                      "neural", "none"):
            raise ValueError(f"unknown blend_method: {self.blend_method}")
        if self.source_image_b is not None:
            if not Path(self.source_image_b).is_file():
                raise FileNotFoundError(
                    f"source image B not found: {self.source_image_b}")
        if self.journey_curve not in ("linear", "smoothstep"):
            raise ValueError(
                f"unknown journey_curve: {self.journey_curve}")
        if self.selector_mode not in ("largest", "reference"):
            raise ValueError(
                f"unknown selector_mode: {self.selector_mode}")
        if self.selector_mode == "reference":
            if self.reference_face_image is None:
                raise ValueError(
                    "selector_mode='reference' requires "
                    "reference_face_image")
            if not Path(self.reference_face_image).is_file():
                raise FileNotFoundError(
                    f"reference_face_image not found: "
                    f"{self.reference_face_image}")
        if self.pixel_boost not in (128, 256, 384, 512, 768):
            raise ValueError(
                f"pixel_boost must be one of 128/256/384/512/768, "
                f"got {self.pixel_boost}")

# ============================================================
# Post-process configs (apply AFTER lipsync, before audio remux)
# ============================================================

@dataclass
class WatermarkConfig:
    """Overlay an image on every frame (logo / handle / branding)."""
    enabled: bool = False
    image_path: str = ""               # absolute path to PNG/JPG
    position: str = "BR"               # TL | TR | BL | BR | CENTER
    scale_pct: float = 15.0            # 1..95 (% of frame width)
    opacity: float = 80.0              # 5..100 (%)


@dataclass
class AspectRatioConfig:
    """Reshape output to a target aspect ratio. Crop = scale-up + cut;
    Pad = scale-down + black bars."""
    enabled: bool = False
    target_aspect: str = "(keep original)"
    fill_mode: str = "crop"            # "crop" or "pad"


@dataclass
class MaskOutConfig:
    """SAM2-driven mask-out for non-face objects in the source.

    When the source has a non-human face-like object (cat, animal,
    statue, prop), LatentSync's internal face detector latches onto
    it and applies spurious lipsync. This config wires the
    core/maskout_pipeline module:
      - clicks: list of (x, y, frame_idx, label) refinement clicks
        on a single object. label=1 positive, label=0 negative.
        Need at least one positive click.
      - dilate_px: how much to dilate the SAM2 mask before inpaint
        (covers TELEA edge bleed).
      - feather: Gaussian feather kernel applied to the mask at the
        composite-back stage (hides the seam between original-object
        and lipsync regions).
    """
    enabled: bool = False
    clicks: List = field(default_factory=list)
    dilate_px: int = 12
    feather: int = 8
    # Phase 1.5 (Rotoscoping handoff): when this points at an existing
    # (T, H, W) uint8 .npy, Stage 1.76 SKIPS the SAM2 worker invocation
    # and uses these masks directly.  Saves 30-60+ seconds per render
    # by reusing rotoscope-produced masks instead of re-segmenting.
    mask_npy_path: str = ""


@dataclass
class OcclusionConfig:
    """FaceFusion-style XSeg occluder gate (single-clip only for now).

    After lipsync render, runs the 3-model XSeg ensemble on the
    ORIGINAL source video to detect "what's covering the face" per
    frame, then composites those source pixels back over the lipsync
    output. Hands / mics / hair / glasses stop disappearing behind
    the lipsynced mouth.

    Temporal smoothing kills the per-frame mask flicker that made
    v1's XSeg gate unusable: bbox EMA damps detector jitter, mask
    EMA damps XSeg edge noise. Recommended start: 0.4 / 0.7.
    """
    enabled: bool = False
    bbox_smoothing: float = 0.4        # 0..0.9; higher = more damping
    mask_smoothing: float = 0.7        # 0..0.9; higher = more damping
    align_to_source: bool = False      # WORKING BASELINE: alignment OFF.
                                       # Every welding/homography variant
                                       # I tried introduced more
                                       # artifacts than the LatentSync
                                       # baseline drift it tried to fix.
                                       # The v1 xseg_gate has no
                                       # alignment path; this flag is
                                       # accepted but ignored.
    feather: int = 9                   # gaussian blur kernel on the matte
    mouth_polygon: bool = False        # Off in the lizard-only design.
                                       # The new mask convention puts
                                       # the lipsync output everywhere
                                       # except the occluder, so face
                                       # never needs a polygon. The
                                       # mouth_polygon path remains in
                                       # xseg_gate.py for experiments
                                       # but should stay OFF in normal
                                       # use because LatentSync's face
                                       # drift makes any face-boundary
                                       # polygon (lips, full-face, etc.)
                                       # produce a visible seam.
