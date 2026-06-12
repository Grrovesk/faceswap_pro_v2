"""FaceSwap Pro -- main pipeline orchestrator.

Drives the frame-by-frame processing loop with async stages to overlap
I/O, detection, swap, and post-processing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
from typing import Optional, Callable

import cv2
import numpy as np

from typing import TYPE_CHECKING

# Lightweight, always-needed imports remain at module level.
from core.frame_packet import FramePacket
from utils.io_utils import VideoReader, VideoWriter, extract_audio, finalize_video_h264
from utils.metrics import cosine_similarity

# Heavy modules (audio, blending, lighting, temporal, identity tracking,
# face landmarks, swap engine) are imported lazily inside FaceSwapPipeline.__init__
# so importing this module does not eagerly load ~half the codebase.
# These imports are exposed to static type checkers only.
if TYPE_CHECKING:
    from audio_sync.lip_corrector import LipCorrector
    from audio_sync.phoneme_extractor import PhonemeExtractor
    from audio_sync.viseme_mapper import VisemeMapper
    from blending.edge_refiner import EdgeRefiner
    from blending.mask_debug import AlphaCorrector
    from blending.mask_generator import MaskGenerator
    from blending.poisson_blend import PoissonBlender
    from core.swap_engine import SwapEngine
    from lighting.color_transfer import ReinhardColorTransfer
    from lighting.illumination_est import IlluminationEstimator
    from lighting.shadow_corrector import ShadowCorrector
    from temporal.frame_buffer import FrameBuffer
    from temporal.optical_flow import OpticalFlowEngine
    from temporal.temporal_smoother import TemporalSmoother
    from tracking.detector import FaceDetector
    from tracking.face_landmarks import FaceLandmarkExtractor
    from tracking.identity_tracker import IdentityTracker

logger = logging.getLogger(__name__)

FACE_ANALYSIS_ROOT = Path(__file__).resolve().parents[1] / "models" / "face_analysis"
DEFAULT_FACE_ANALYSIS_PACK = "buffalo_l"
FALLBACK_FACE_ANALYSIS_PACKS = ("antelopev2",)


class FaceSwapPipeline:
    """End-to-end face-swap pipeline with all quality-improvement stages.

    Usage
    -----
    >>> pipeline = FaceSwapPipeline(config)
    >>> pipeline.run(source_path="source.jpg", input_path="input.mp4", output_path="out.mp4")
    """

    def __init__(self, config: dict):
        # ─── Lazy imports (Fix #6): pulled inside __init__ so module load is cheap. ───
        from audio_sync.lip_corrector import LipCorrector
        from audio_sync.phoneme_extractor import PhonemeExtractor
        from audio_sync.viseme_mapper import VisemeMapper
        from blending.edge_refiner import EdgeRefiner
        from blending.mask_generator import MaskGenerator
        from blending.poisson_blend import PoissonBlender
        from core.swap_engine import SwapEngine
        from lighting.color_transfer import ReinhardColorTransfer
        from lighting.illumination_est import IlluminationEstimator
        from lighting.shadow_corrector import ShadowCorrector
        from temporal.frame_buffer import FrameBuffer
        from temporal.optical_flow import OpticalFlowEngine
        from temporal.temporal_smoother import TemporalSmoother
        from tracking.detector import FaceDetector
        from tracking.face_landmarks import FaceLandmarkExtractor
        from tracking.identity_tracker import IdentityTracker

        self.cfg = config

        # -- Sub-modules (lazy-initialised on first ``run``) --
        self.detector: Optional[FaceDetector] = None
        self.landmark_extractor: Optional[FaceLandmarkExtractor] = None
        self.identity_tracker: Optional[IdentityTracker] = None
        self.swap_engine: Optional[SwapEngine] = None

        self.phoneme_extractor: Optional[PhonemeExtractor] = None
        self.viseme_mapper: Optional[VisemeMapper] = None
        self.lip_corrector: Optional[LipCorrector] = None

        self.color_transfer: Optional[ReinhardColorTransfer] = None
        self.illumination_est: Optional[IlluminationEstimator] = None
        self.shadow_corrector: Optional[ShadowCorrector] = None

        self.mask_generator: Optional[MaskGenerator] = None
        self.poisson_blender: Optional[PoissonBlender] = None
        self.edge_refiner: Optional[EdgeRefiner] = None

        self.flow_engine: Optional[OpticalFlowEngine] = None
        self.temporal_smoother: Optional[TemporalSmoother] = None
        self.frame_buffer: Optional[FrameBuffer] = None

        # InsightFace FaceAnalysis fallback (used if custom detector fails)
        self._insightface_app = None
        self._face_analysis_pack_name: str | None = None
        self._track_source_drift: bool = False

        self._initialised = False

    # ==================================================================
    # Initialisation
    # ==================================================================

    def _init_modules(self) -> None:
        """Instantiate all sub-modules based on config."""
        # ─── Lazy submodule imports (Fix #6 v2): heavy classes loaded only when
        #     a pipeline run is actually started. Importing core.pipeline by
        #     itself remains cheap. ─────────────────────────────────────────
        from audio_sync.lip_corrector import LipCorrector
        from audio_sync.phoneme_extractor import PhonemeExtractor
        from audio_sync.viseme_mapper import VisemeMapper
        from blending.edge_refiner import EdgeRefiner
        from blending.mask_generator import MaskGenerator
        from blending.poisson_blend import PoissonBlender
        from core.swap_engine import SwapEngine
        from lighting.color_transfer import ReinhardColorTransfer
        from lighting.illumination_est import IlluminationEstimator
        from lighting.shadow_corrector import ShadowCorrector
        from temporal.frame_buffer import FrameBuffer
        from temporal.optical_flow import OpticalFlowEngine
        from temporal.temporal_smoother import TemporalSmoother
        from tracking.detector import FaceDetector
        from tracking.face_landmarks import FaceLandmarkExtractor
        from tracking.identity_tracker import IdentityTracker

        det_cfg = self.cfg.get("detection", {})
        id_cfg = self.cfg.get("identity", {})
        swap_cfg = self.cfg.get("swap", {})
        audio_cfg = self.cfg.get("audio_sync", {})
        light_cfg = self.cfg.get("lighting", {})
        blend_cfg = self.cfg.get("blending", {})
        temp_cfg = self.cfg.get("temporal", {})
        opt_cfg = self.cfg.get("optimization", {})
        fa_cfg = self.cfg.get("face_analysis", {})

        gpu_id = opt_cfg.get("cuda_device", 0)
        self._prefer_paste_back = bool(swap_cfg.get("prefer_paste_back", True))
        self._face_analysis_pack_name = self._resolve_face_analysis_pack(fa_cfg)
        self._track_source_drift = bool(id_cfg.get("track_source_drift", False))

        # Detection & tracking -- try InsightFace first, fall back to custom
        self._init_insightface(det_cfg, gpu_id)

        # Try custom detector as well
        try:
            self.detector = FaceDetector(
                model_name=det_cfg.get("model", "scrfd_10g_bnkps"),
                threshold=det_cfg.get("threshold", 0.5),
                nms_iou=det_cfg.get("nms_iou", 0.4),
                gpu_id=gpu_id,
            )
            self.landmark_extractor = FaceLandmarkExtractor(
                gpu_id=gpu_id,
                pack_name=self._face_analysis_pack_name,
                fallback_packs=tuple(fa_cfg.get("fallback_packs", list(FALLBACK_FACE_ANALYSIS_PACKS))),
            )
        except Exception as exc:
            logger.warning("Custom detector init failed: %s -- using InsightFace only", exc)

        # Swap model -- use SwapEngine (not raw ONNX session)
        swap_model = swap_cfg.get("model", "inswapper_128")
        model_path = self._find_model(swap_model)
        # Phase 1 of multi-backend roadmap (Tier 1 #1): route through
        # the swap_backends factory so GHOST-A / SimSwap-512 backends
        # can register and the UI can pick between them.  Default is
        # "inswapper_128" which resolves to InswapperBackend, a thin
        # subclass of SwapEngine -- behavior is byte-identical when
        # no backend override is set in config.
        from core.swap_backends import get_backend
        swap_backend_name = swap_cfg.get("backend", "inswapper_128")
        BackendCls = get_backend(swap_backend_name)
        self.swap_engine = BackendCls(
            model_path=model_path,
            device_id=gpu_id,
            use_tensorrt=opt_cfg.get("use_tensorrt", False),
        )

        # Identity tracker (source embedding set later)
        self.identity_tracker = IdentityTracker(
            drift_threshold=id_cfg.get("drift_threshold", 0.45),
            ema_decay=id_cfg.get("ema_decay", 0.9),
        )
        if not self._track_source_drift:
            logger.info("Source drift tracking disabled for pre-swap target frames.")

        # Audio-lip sync (optional)
        if audio_cfg.get("enabled", False):
            self.phoneme_extractor = PhonemeExtractor(
                model_name=audio_cfg.get("phoneme_model", "facebook/wav2vec2-base-960h"),
            )
            self.viseme_mapper = VisemeMapper()
            self.lip_corrector = LipCorrector()

        # Lighting
        if light_cfg.get("color_transfer", "reinhard") != "none":
            self.color_transfer = ReinhardColorTransfer()
        self.illumination_est = IlluminationEstimator(
            sh_order=light_cfg.get("sh_order", 1),
        )
        self.shadow_corrector = ShadowCorrector(
            clamp_range=(
                light_cfg.get("shadow_clamp_min", 0.5),
                light_cfg.get("shadow_clamp_max", 1.5),
            ),
        ) if light_cfg.get("shadow_correction", True) else None

        # Blending
        self.mask_generator = MaskGenerator(
            feather_px=blend_cfg.get("feather_px", 20),
        )
        self.poisson_blender = PoissonBlender() if blend_cfg.get("method", "poisson") == "poisson" else None
        self.edge_refiner = EdgeRefiner(
            radius=blend_cfg.get("guided_filter_radius", 5),
            eps=blend_cfg.get("guided_filter_eps", 0.01),
        )

        # Temporal
        if temp_cfg.get("enabled", True):
            self.flow_engine = OpticalFlowEngine(
                model_name=temp_cfg.get("flow_model", "raft_small"),
                gpu_id=gpu_id,
            )
            self.temporal_smoother = TemporalSmoother(
                ema_decay=temp_cfg.get("ema_decay", 0.85),
            )
            self.frame_buffer = FrameBuffer(size=temp_cfg.get("buffer_size", 5))

        self._initialised = True
        logger.info("All pipeline modules initialised.")

    def _init_insightface(self, det_cfg: dict, gpu_id: int) -> None:
        """Initialize InsightFace FaceAnalysis as primary or fallback detector."""
        try:
            from insightface.app import FaceAnalysis

            pack_name = self._face_analysis_pack_name or DEFAULT_FACE_ANALYSIS_PACK
            self._insightface_app = FaceAnalysis(
                name=pack_name,
                root=str(FACE_ANALYSIS_ROOT),
                providers=[
                    ("CUDAExecutionProvider", {"device_id": gpu_id}),
                    "CPUExecutionProvider",
                ],
            )
            self._insightface_app.prepare(
                ctx_id=gpu_id,
                det_size=(640, 640),
                det_thresh=det_cfg.get("threshold", 0.5),
            )
            logger.info(
                "InsightFace FaceAnalysis initialized with %s from %s",
                pack_name,
                FACE_ANALYSIS_ROOT,
            )
        except Exception as exc:
            logger.warning("InsightFace FaceAnalysis init failed: %s", exc)
            self._insightface_app = None

    # ==================================================================
    # Public API
    # ==================================================================

    def run(
        self,
        source_path: str,
        input_path: str,
        output_path: str,
        verbose: bool = False,
        source_embedding_override=None,
        per_frame_embedding_fn=None,
    ) -> None:
        """Process an entire video end-to-end.

        source_embedding_override: optional (512,) np.ndarray. If provided,
            this embedding is used instead of extracting one from
            source_path. The source_path arg is still required (used for
            logging/UI metadata) but no face is detected on it.

        per_frame_embedding_fn: optional callable
            (frame_idx: int, total_frames: int) -> np.ndarray (512,)
            Called inside the frame loop before the swap stage; the
            returned embedding becomes the source identity for THAT
            frame only. Enables embedding-journey (LERP A->B across
            time) without modifying the swap engine's persistent state
            outside the loop.
        """
        if not self._initialised:
            self._init_modules()

        # -- (Selector mode) extract reference face embedding if set --
        # cfg["selector"]["mode"] = "largest" (default, legacy behavior:
        #   pick the biggest detected face) or "reference" (only swap
        #   the detected face whose ArcFace embedding is closest to
        #   the reference image's embedding, and only if cosine
        #   distance <= reference_distance).
        sel_cfg = self.cfg.get("selector", {}) or {}
        sel_mode = str(sel_cfg.get("mode", "largest") or "largest").lower()
        ref_path = sel_cfg.get("reference_path") or None
        ref_distance = float(sel_cfg.get("reference_distance", 0.6))
        self._selector_mode = sel_mode
        self._reference_distance = ref_distance
        self._reference_embedding = None
        if sel_mode == "reference" and ref_path:
            try:
                self._reference_embedding = \
                    self._extract_source_embedding(str(ref_path))
                logger.info(
                    "selector_mode=reference, reference embedding "
                    "extracted (norm=%.3f, distance_threshold=%.2f)",
                    float(np.linalg.norm(self._reference_embedding)),
                    ref_distance,
                )
            except Exception as exc:
                logger.warning(
                    "Could not extract reference embedding from %s: "
                    "%s -- falling back to largest-face mode",
                    ref_path, exc)
                self._selector_mode = "largest"

        # ---- T2-NEW Region restriction via rotoscope mask ----
        # Load the (N, H, W) uint8 stack if a path was provided.  On any
        # failure, log + null out -> no gating, same as legacy.
        self._mask_stack = None
        mg_cfg = self.cfg.get("mask_gate", {}) or {}
        mg_path = mg_cfg.get("npy_path")
        if mg_path:
            try:
                from pathlib import Path as _P
                _mp = _P(mg_path)
                if _mp.is_file():
                    arr = np.load(str(_mp), mmap_mode="r")
                    if arr.ndim == 3 and arr.dtype == np.uint8:
                        self._mask_stack = arr
                        logger.info(
                            "Region-restriction mask loaded: %s shape=%s",
                            mg_path, tuple(arr.shape))
                    else:
                        logger.warning(
                            "Region-restriction mask ignored: "
                            "expected (N,H,W) uint8, got %s %s",
                            arr.shape, arr.dtype)
                else:
                    logger.warning(
                        "Region-restriction mask path does not exist: %s",
                        mg_path)
            except Exception as exc:
                logger.warning(
                    "Region-restriction mask load failed: %s -- gating disabled",
                    exc)

        # -- Extract source embedding and set it on the swap engine --
        if source_embedding_override is not None:
            source_embedding = source_embedding_override
            logger.info(
                "Using caller-provided source_embedding_override "
                "(norm=%.3f); skipping detection on source_path",
                float(np.linalg.norm(source_embedding)),
            )
        else:
            source_embedding = self._extract_source_embedding(source_path)
        self.identity_tracker.set_source(source_embedding)
        self.swap_engine.set_source_embedding(source_embedding)

        # -- Pre-extract phonemes if audio sync is enabled --
        phoneme_map: dict[int, str] = {}
        if self.phoneme_extractor is not None:
            audio_path = extract_audio(input_path)
            phoneme_map = self.phoneme_extractor.extract(audio_path, fps=0)

        # -- Open video --
        reader = VideoReader(input_path)
        out_cfg = self.cfg.get("output", {})
        final_output_path = output_path
        requested_codec = out_cfg.get("codec", "mp4v")
        use_ffmpeg_finalize = bool(out_cfg.get("ffmpeg_finalize", True))
        keep_audio = bool(out_cfg.get("keep_audio", True))

        if use_ffmpeg_finalize:
            output_dir = os.path.dirname(output_path) or "."
            output_stem = Path(output_path).stem
            intermediate_ext = out_cfg.get("intermediate_ext", ".mkv")
            intermediate_codec = out_cfg.get("intermediate_codec", "FFV1")
            intermediate_path = os.path.join(output_dir, f"{output_stem}.intermediate{intermediate_ext}")
            writer_path = intermediate_path
            writer_codec = intermediate_codec
        else:
            intermediate_path = None
            writer_path = output_path
            writer_codec = requested_codec

        writer = VideoWriter(
            writer_path,
            reader.fps,
            (reader.width, reader.height),
            codec=writer_codec,
        )

        # Set fps for phoneme map (now that we know it)
        if self.phoneme_extractor is not None and phoneme_map:
            audio_path = extract_audio(input_path)
            phoneme_map = self.phoneme_extractor.extract(audio_path, fps=reader.fps)

        # -- Process frames --
        prev_frame = None
        total_start = time.perf_counter()

        for batch in reader.read_batches(batch_size=1):
            for frame_idx, frame_bgr in batch:
                packet = FramePacket(frame_idx=frame_idx, frame_bgr=frame_bgr)

                # Stage 1: Detection & alignment
                self._stage_detect(packet)

                if not packet.face_found:
                    writer.write(frame_bgr)
                    continue

                # Stage 2: Identity tracking
                self._stage_identity(packet, verbose=verbose)

                # Stage 2.5 (OPTIONAL): per-frame embedding override.
                # Used by embedding-journey to LERP A->B across time.
                # Setting packet.source_embedding here propagates into
                # _stage_swap; we also update the swap engine's cached
                # projection (set_source_embedding short-circuits when
                # the embedding hasn't changed, so the constant-blend
                # case is free).
                if per_frame_embedding_fn is not None:
                    try:
                        emb = per_frame_embedding_fn(
                            frame_idx, reader.total)
                        packet.source_embedding = emb
                        self.swap_engine.set_source_embedding(emb)
                    except Exception as _hook_exc:
                        logger.warning(
                            "per_frame_embedding_fn raised at frame %d: "
                            "%s (using last embedding)",
                            frame_idx, _hook_exc)

                # Stage 3: Swap
                self._stage_swap(packet)

                # Stage 4: Audio-lip sync
                self._stage_audio_sync(packet, phoneme_map)

                # Stage 5: Lighting correction
                self._stage_lighting(packet)

                # Stage 6: Blending
                self._stage_blend(packet)

                # Stage 7: Temporal smoothing
                self._stage_temporal(packet, prev_frame)

                # Stage 8: Optional GFPGAN face restoration
                self._stage_enhance(packet)

                # Write output
                out = packet.output_frame if packet.output_frame is not None else frame_bgr
                writer.write(out)

                prev_frame = frame_bgr.copy()

                # Store in temporal buffer
                if self.frame_buffer is not None:
                    self.frame_buffer.push(packet)

        elapsed = time.perf_counter() - total_start
        n_frames = reader.total
        fps_proc = n_frames / elapsed if elapsed > 0 else 0
        logger.info(
            "Done. %d frames in %.1fs (%.1f fps processed)",
            n_frames, elapsed, fps_proc,
        )

        # Log swap engine stats
        if self.swap_engine is not None:
            logger.info("Swap engine stats: %s", self.swap_engine.stats)

        writer.close()
        reader.close()

        if intermediate_path is not None:
            # Output encoding quality. The intermediate is already lossless
            # (FFV1); this only controls the distributable MP4. The default,
            # "visually_lossless" (libx264 yuv420p CRF 17), is indistinguish-
            # able by eye from true lossless but ~10-15x smaller. "lossless"
            # keeps the old libx264rgb CRF 0 path (very large files).
            _QUALITY = {
                "lossless":          {"lossless_rgb": True,  "crf": 0},
                "visually_lossless": {"lossless_rgb": False, "crf": 17},
                "balanced":          {"lossless_rgb": False, "crf": 21},
            }
            _q = _QUALITY.get(
                str(out_cfg.get("quality", "visually_lossless")).lower(),
                _QUALITY["visually_lossless"],
            )
            try:
                finalize_video_h264(
                    intermediate_path,
                    input_path,
                    final_output_path,
                    crf=int(out_cfg.get("final_crf", _q["crf"])),
                    preset=out_cfg.get("final_preset", "slow"),
                    audio_bitrate=str(out_cfg.get("audio_bitrate", "192k")),
                    keep_audio=keep_audio,
                    lossless_rgb=_q["lossless_rgb"],
                )
                logger.info(
                    "Finalized H.264 output path=%s from intermediate=%s",
                    final_output_path,
                    intermediate_path,
                )
            finally:
                try:
                    os.remove(intermediate_path)
                except OSError:
                    pass

    # ==================================================================
    # Single-frame preview (for the Video Swap tab's live preview)
    # ==================================================================

    def preview_frame(
        self,
        source_path: str,
        target_video_path: str,
        frame_idx: int,
    ) -> "np.ndarray":
        """Run the full per-frame swap pipeline on ONE target frame and
        return the resulting BGR ndarray. The video file is opened, the
        requested frame is seeked + read, the pipeline stages run as in
        run(), but no writer is opened and no audio / phoneme / GFPGAN
        post-pass happens. Intended for the Video Swap tab's preview
        button -- fast (~1 s on the A6000 after the first call warms up
        the models); does not touch the full render path.

        Raises RuntimeError / IOError on setup failure. If no face is
        detected in the requested frame, returns the original frame
        unchanged (same behaviour as run()).
        """
        import cv2 as _cv2

        if not self._initialised:
            self._init_modules()

        # Source embedding -- same path run() uses. The swap engine
        # internally caches the source so back-to-back previews on the
        # same source skip the embedding cost.
        source_embedding = self._extract_source_embedding(source_path)
        self.identity_tracker.set_source(source_embedding)
        self.swap_engine.set_source_embedding(source_embedding)

        cap = _cv2.VideoCapture(target_video_path)
        if not cap.isOpened():
            raise IOError(f"could not open target video: {target_video_path}")
        total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            cap.release()
            raise IOError(f"target video reports 0 frames: {target_video_path}")
        frame_idx = max(0, min(int(frame_idx), total - 1))

        # Robust seek: some codecs land on the nearest keyframe rather
        # than the exact frame. Jump a few frames before and read
        # forward until we land on the target.
        backoff = 6
        cap.set(_cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - backoff))
        frame_bgr = None
        for _ in range(backoff + 1):
            ok, frame_bgr = cap.read()
            if not ok:
                frame_bgr = None
                break
            pos = int(cap.get(_cv2.CAP_PROP_POS_FRAMES)) - 1
            if pos >= frame_idx:
                break
        cap.release()
        if frame_bgr is None:
            raise RuntimeError(f"could not read frame {frame_idx} from {target_video_path}")

        packet = FramePacket(frame_idx=frame_idx, frame_bgr=frame_bgr)
        self._stage_detect(packet)
        if not packet.face_found:
            return frame_bgr  # no face -> return source frame as-is
        self._stage_identity(packet, verbose=False)
        self._stage_swap(packet)
        # Audio-sync stage takes a phoneme_map; preview is single-frame
        # with no audio context -> pass an empty dict.
        self._stage_audio_sync(packet, {})
        self._stage_lighting(packet)
        self._stage_blend(packet)
        # Temporal stage smooths across prev_frame; no prev frame for a
        # one-shot preview.
        self._stage_temporal(packet, None)

        # Optional GFPGAN restoration -- now wired so the preview's
        # Enhance Faces dropdown actually does something.
        self._stage_enhance(packet)

        return packet.output_frame if packet.output_frame is not None else frame_bgr


    # ==================================================================
    # Pipeline stages
    # ==================================================================

    def _pick_face(self, faces, frame_idx):
        """Return the single face the swap should operate on for this
        frame, or None to skip the swap entirely.

        Modes:
          - "largest" (default, legacy behavior): biggest bbox area.
          - "reference": cosine-closest to self._reference_embedding,
            but only if 1 - cos_sim <= self._reference_distance.
            (cosine distance = 1 - cosine similarity.)
        """
        if not faces:
            return None
        mode = getattr(self, "_selector_mode", "largest")
        ref = getattr(self, "_reference_embedding", None)
        if mode == "reference" and ref is not None:
            ref_n = ref / (float(np.linalg.norm(ref)) + 1e-8)
            best = None
            best_dist = float("inf")
            for f in faces:
                emb = getattr(f, "normed_embedding", None)
                if emb is None:
                    continue
                emb_n = emb.astype(np.float32)
                emb_n = emb_n / (float(np.linalg.norm(emb_n)) + 1e-8)
                cos_sim = float(np.dot(ref_n, emb_n))
                dist = 1.0 - cos_sim
                if dist < best_dist:
                    best_dist = dist
                    best = f
            if best is None:
                return None
            if best_dist > float(self._reference_distance):
                # Closest detected face is still too dissimilar -> skip.
                if frame_idx % 25 == 0:
                    logger.debug(
                        "frame %d: no detected face within distance "
                        "(closest=%.3f, threshold=%.3f)",
                        frame_idx, best_dist, self._reference_distance)
                return None
            return best
        # Default / "largest": biggest bbox area
        return max(faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def _filter_faces_by_mask(self, faces, frame_idx, frame_hw):
        """T2-NEW. Drop detected faces whose bbox centroid is OUTSIDE
        the mask for this frame.  Returns the filtered list (possibly
        empty).  No-op if mask stack isn't loaded.

        Diagnostic logging: at frame 0 + every 25th frame, log how many
        faces went in, how many came out, and the (cx, cy, mask_value)
        of every detected face.  Lets the user see immediately whether
        the rotoscope mask is actually gating the unwanted faces or
        letting them through (mask is over-generous -> tighten the
        rotoscope click; or the gorilla bbox centroid happens to land
        inside the girl-mask -> add a negative click).
        """
        if self._mask_stack is None or not faces:
            return faces
        n, mh, mw = self._mask_stack.shape
        if int(frame_idx) >= n:
            return faces
        mask = np.asarray(self._mask_stack[int(frame_idx)])
        H, W = frame_hw
        if (mh, mw) != (H, W):
            try:
                import cv2 as _cv
                mask = _cv.resize(mask, (W, H),
                                    interpolation=_cv.INTER_NEAREST)
            except Exception:
                return faces  # be permissive on resize failure

        diag = (int(frame_idx) == 0 or int(frame_idx) % 25 == 0)
        decisions = []
        out = []
        for f in faces:
            try:
                x1, y1, x2, y2 = f.bbox
                cx = int(max(0, min(W - 1, (x1 + x2) * 0.5)))
                cy = int(max(0, min(H - 1, (y1 + y2) * 0.5)))
            except Exception:
                out.append(f)
                if diag:
                    decisions.append("(bbox-bad, kept)")
                continue
            mv = int(mask[cy, cx])
            keep = mv > 0
            if keep:
                out.append(f)
            if diag:
                bbox_area = (
                    (float(x2) - float(x1))
                    * (float(y2) - float(y1)))
                decisions.append(
                    f"({cx},{cy}) mask_v={mv} "
                    f"bbox={int(bbox_area)}px "
                    f"-> {'KEEP' if keep else 'DROP'}")

        if diag:
            mask_white_pct = float((mask > 0).mean()) * 100.0
            logger.info(
                "[mask-gate] frame %d: %d faces in -> %d out  "
                "(mask coverage %.1f%% of frame)  %s",
                int(frame_idx), len(faces), len(out),
                mask_white_pct, " | ".join(decisions))
        return out

    def _stage_detect(self, pkt: FramePacket) -> None:
        """Detect face, extract landmarks, compute alignment matrix.

        Tries InsightFace FaceAnalysis first (most reliable with buffalo_l),
        falls back to custom FaceDetector if InsightFace is unavailable.
        """
        # -- Try InsightFace FaceAnalysis first --
        if self._insightface_app is not None:
            try:
                ifaces = self._insightface_app.get(pkt.frame_bgr)
                if ifaces:
                    # T2-NEW: drop faces outside the rotoscope mask for
                    # this frame (if a mask stack was loaded).  Done BEFORE
                    # selector_mode picks one, so reference/largest see
                    # only the gated set.
                    if self._mask_stack is not None:
                        ifaces = self._filter_faces_by_mask(
                            ifaces, pkt.frame_idx,
                            pkt.frame_bgr.shape[:2])
                        if not ifaces:
                            return  # all faces outside mask -> skip frame
                    face = self._pick_face(ifaces, pkt.frame_idx)
                    if face is None:
                        return  # reference mode: nothing matched, skip swap
                    pkt.face_bbox = tuple(int(x) for x in face.bbox)
                    pkt.face_kps = face.kps

                    # Alignment matrix -> 112x112 using InsightFace's standard template
                    if self.detector is not None:
                        pkt.face_align_mat = self.detector.compute_align_mat(face.kps)
                    else:
                        pkt.face_align_mat = self._compute_align_mat_standalone(face.kps)

                    pkt.aligned_face = self._warp_align_standalone(
                        pkt.frame_bgr, pkt.face_align_mat
                    )

                    # Use InsightFace's embedding directly if available
                    if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
                        pkt.current_embedding = face.normed_embedding.astype(np.float32)

                    # Dense landmarks
                    if self.landmark_extractor is not None:
                        lms_106 = self.landmark_extractor.extract(pkt.aligned_face)
                        if lms_106 is not None:
                            M_inv = cv2.invertAffineTransform(pkt.face_align_mat)
                            lms_orig = cv2.transform(lms_106.reshape(-1, 1, 2).astype(np.float32), M_inv)
                            pkt.face_landmarks_106 = lms_orig.reshape(-1, 2)

                    return  # Success with InsightFace
            except Exception as exc:
                logger.debug("InsightFace detection failed: %s -- trying custom", exc)

        # -- Fallback to custom FaceDetector --
        if self.detector is not None:
            faces = self.detector.detect(pkt.frame_bgr)
            if not faces:
                return
            face = self._pick_face(faces, pkt.frame_idx)
            if face is None:
                return
            pkt.face_bbox = face.bbox
            pkt.face_kps = face.kps

            pkt.face_align_mat = self.detector.compute_align_mat(face.kps)
            pkt.aligned_face = self.detector.warp_align(pkt.frame_bgr, pkt.face_align_mat)

            lms_106 = self.landmark_extractor.extract(pkt.aligned_face)
            if lms_106 is not None:
                M_inv = cv2.invertAffineTransform(pkt.face_align_mat)
                lms_orig = cv2.transform(lms_106.reshape(-1, 1, 2).astype(np.float32), M_inv)
                pkt.face_landmarks_106 = lms_orig.reshape(-1, 2)

    def _stage_identity(self, pkt: FramePacket, verbose: bool = False) -> None:
        """Extract current embedding, check for identity drift."""
        if pkt.aligned_face is None:
            return

        # Use InsightFace embedding if already extracted during detection
        if pkt.current_embedding is None:
            if self._insightface_app is not None:
                # Re-extract embedding using InsightFace
                try:
                    ifaces = self._insightface_app.get(pkt.aligned_face)
                    if ifaces and hasattr(ifaces[0], "normed_embedding"):
                        pkt.current_embedding = ifaces[0].normed_embedding.astype(np.float32)
                except Exception:
                    pass

            # Fall back to custom extractor
            if pkt.current_embedding is None and self.landmark_extractor is not None:
                pkt.current_embedding = self.landmark_extractor.extract_embedding(pkt.aligned_face)

        pkt.source_embedding = self.identity_tracker.source.copy()

        if pkt.current_embedding is not None and self._track_source_drift:
            sim, drifted = self.identity_tracker.update(pkt.current_embedding)
            pkt.identity_score = sim
            pkt.identity_drifted = drifted

            if drifted:
                logger.warning(
                    "Frame %d: identity drift detected (sim=%.3f). Re-initialising.",
                    pkt.frame_idx, sim,
                )

            if verbose:
                print(f"  Frame {pkt.frame_idx}: identity_score={sim:.4f}  drifted={drifted}")
        elif pkt.current_embedding is not None:
            pkt.identity_score = cosine_similarity(pkt.source_embedding, pkt.current_embedding)

    def _stage_swap(self, pkt: FramePacket) -> None:
        """Run the inswapper model via SwapEngine to produce the swapped face.

        When the official InsightFace INSwapper path is available, prefer its
        native paste-back composition over the repo's custom lighting/blending
        stack. That path is the one we validated directly against this model.

        Reads three knobs from cfg["blending"] and one from cfg["identity"]:
          - mask_padding (int, default 0)  -- erode kernel adjustment
          - mask_blur    (float, default 1.0) -- blur kernel scale
          - swap_strength (float, default 1.0) -- 1.0 = full swap,
                              0.0 = original face, lerp in between
        """
        if pkt.source_embedding is None:
            return

        blend_cfg = self.cfg.get("blending", {}) or {}
        identity_cfg = self.cfg.get("identity", {}) or {}
        swap_cfg = self.cfg.get("swap", {}) or {}
        mask_padding = int(blend_cfg.get("mask_padding", 0))
        mask_blur = float(blend_cfg.get("mask_blur", 1.0))
        swap_strength = float(identity_cfg.get("swap_strength", 1.0))
        swap_strength = max(0.0, min(1.0, swap_strength))
        # pixel_boost: post-swap upscale via GFPGAN before paste-back.
        # inswapper_128 hallucinates at 128x128 and the stock paste-back
        # warps that to bbox size via INTER_CUBIC. With pixel_boost set
        # to 256/512/768 we upsize the aligned + swapped face crops to
        # the boost size, run GFPGAN restore on the upsized swap
        # (has_aligned=True), scale the warp matrix accordingly, then
        # paste back. Result: finer skin / eye detail on close-ups.
        # pixel_boost=128 (or unset) -> stock 128px path unchanged.
        pixel_boost = int(swap_cfg.get("pixel_boost", 128))
        if pixel_boost not in (128, 256, 384, 512, 768):
            pixel_boost = 128

        if self._prefer_paste_back and pkt.face_kps is not None and hasattr(self.swap_engine, "swap_paste_back"):
            try:
                if pixel_boost <= 128:
                    swapped_frame = self.swap_engine.swap_paste_back(
                        pkt.frame_bgr,
                        pkt.face_kps,
                        pkt.source_embedding,
                        mask_padding_px=mask_padding,
                        mask_blur_scale=mask_blur,
                    )
                else:
                    # ---- pixel_boost path ----
                    swap_result = self.swap_engine.swap_aligned(
                        pkt.frame_bgr,
                        pkt.face_kps,
                        pkt.source_embedding,
                    )
                    base_size = int(swap_result.aligned_face.shape[0])
                    if pixel_boost <= base_size:
                        # Boost <= native; skip the upscale, use stock paste.
                        swapped_frame = self.swap_engine.paste_back(
                            pkt.frame_bgr,
                            swap_result.aligned_face,
                            swap_result.swapped_face,
                            swap_result.align_mat,
                            mask_padding_px=mask_padding,
                            mask_blur_scale=mask_blur,
                        )
                    else:
                        scale = float(pixel_boost) / float(base_size)
                        up_aligned = cv2.resize(
                            swap_result.aligned_face,
                            (pixel_boost, pixel_boost),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        up_swapped = cv2.resize(
                            swap_result.swapped_face,
                            (pixel_boost, pixel_boost),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        # Try GFPGAN on the upsized swap crop. The face
                        # is already aligned (it came from the swap path),
                        # so we pass has_aligned=True / paste_back=False
                        # to get just the enhanced crop back.
                        try:
                            restorer = self._init_gfpgan_restorer()
                        except Exception:
                            restorer = None
                        if restorer is not None:
                            try:
                                _, _, enhanced = restorer.enhance(
                                    up_swapped,
                                    has_aligned=True,
                                    only_center_face=True,
                                    paste_back=False,
                                )
                                # GFPGAN returns either a list of crops
                                # or a single crop depending on version.
                                if isinstance(enhanced, list) and enhanced:
                                    enhanced = enhanced[0]
                                if (enhanced is not None
                                        and hasattr(enhanced, "shape")):
                                    # Make sure shape matches pixel_boost
                                    if (enhanced.shape[0] != pixel_boost
                                            or enhanced.shape[1] != pixel_boost):
                                        enhanced = cv2.resize(
                                            enhanced,
                                            (pixel_boost, pixel_boost),
                                            interpolation=cv2.INTER_CUBIC)
                                    up_swapped = enhanced
                            except Exception as _gfp_exc:
                                logger.debug(
                                    "pixel_boost GFPGAN restore failed: "
                                    "%s -- using cubic-upscaled swap",
                                    _gfp_exc)
                        # Scale the warp matrix so it maps to the larger
                        # source dimensions. M is 2x3 affine mapping
                        # frame_xy -> aligned_uv at base_size; multiply
                        # the (u,v) rows by `scale` to map to pixel_boost
                        # space.
                        scaled_M = swap_result.align_mat.copy().astype(np.float32)
                        scaled_M[:2, :] = scaled_M[:2, :] * scale
                        swapped_frame = self.swap_engine.paste_back(
                            pkt.frame_bgr,
                            up_aligned,
                            up_swapped,
                            scaled_M,
                            mask_padding_px=mask_padding,
                            mask_blur_scale=mask_blur,
                        )
                if swap_strength < 1.0:
                    # Alpha-blend with the original frame. Outside the
                    # face mask, swapped == original so the blend is a
                    # no-op there; inside, this dials the swap intensity.
                    swapped_frame = (
                        swap_strength * swapped_frame.astype(np.float32)
                        + (1.0 - swap_strength) * pkt.frame_bgr.astype(np.float32)
                    ).clip(0, 255).astype(np.uint8)


                pkt.output_frame = swapped_frame
                return
            except Exception as exc:
                logger.warning("swap_paste_back failed: %s -- falling back to crop swap", exc)

        if pkt.aligned_face is None:
            return

        result = self.swap_engine.swap(pkt.aligned_face, pkt.source_embedding)
        pkt.swapped_face = result.swapped_face

        if result.confidence < 0.1:
            logger.warning(
                "Frame %d: swap confidence very low (%.3f). Output may be degraded.",
                pkt.frame_idx, result.confidence,
            )

    def _stage_audio_sync(self, pkt: FramePacket, phoneme_map: dict[int, str]) -> None:
        """Apply lip-shape correction based on audio phonemes."""
        if pkt.output_frame is not None or self.lip_corrector is None or pkt.swapped_face is None:
            return

        phoneme = phoneme_map.get(pkt.frame_idx)
        if phoneme is None:
            return

        viseme = self.viseme_mapper.map(phoneme)
        pkt.viseme_label = viseme

        if viseme is None or pkt.face_landmarks_106 is None:
            return

        target_lms = self.viseme_mapper.get_target_landmarks(viseme, pkt.face_landmarks_106)
        current_lms = pkt.face_landmarks_106

        pkt.lip_corrected_face = self.lip_corrector.correct(
            pkt.swapped_face, current_lms, target_lms, pkt.lip_indices
        )

    def _stage_lighting(self, pkt: FramePacket) -> None:
        """Colour-transfer, relight, and shadow-correct the swapped face.

        This stage applies three lighting corrections in sequence:
        1. Reinhard color transfer -- match swapped face colors to the original
        2. SH illumination estimation & relighting -- match lighting direction
        3. Shadow correction -- transfer shadow map from original to swapped
        """
        if pkt.output_frame is not None:
            return

        face = pkt.lip_corrected_face if pkt.lip_corrected_face is not None else pkt.swapped_face
        ref_aligned = pkt.aligned_face
        if face is None or ref_aligned is None:
            return

        # Ensure face is in the same aligned-face coordinate space as the active path.
        ah, aw = ref_aligned.shape[:2]
        if face.shape[:2] != (ah, aw):
            face = cv2.resize(face, (aw, ah))

        # -- Step 1: Reinhard color transfer --
        if self.color_transfer is not None:
            x1, y1, x2, y2 = pkt.face_bbox
            h, w = pkt.frame_bgr.shape[:2]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            orig_face_crop = pkt.frame_bgr[y1c:y2c, x1c:x2c]

            if orig_face_crop.size > 0:
                orig_resized = cv2.resize(orig_face_crop, (aw, ah))
                face = self.color_transfer.transfer(face, orig_resized)

        # -- Step 2: SH illumination estimation & relighting --
        if self.illumination_est is not None and pkt.face_landmarks_106 is not None:
            lms_aligned = self.landmark_extractor.extract(ref_aligned) if self.landmark_extractor else None
            if lms_aligned is not None:
                sh_source = self.illumination_est.estimate(ref_aligned, lms_aligned)
                sh_target = self.illumination_est.estimate(face, lms_aligned)
                face = self.illumination_est.relight(face, lms_aligned, sh_source, sh_target)

        # -- Step 3: Shadow correction --
        if self.shadow_corrector is not None and pkt.face_landmarks_106 is not None:
            lms_aligned = self.landmark_extractor.extract(ref_aligned) if self.landmark_extractor else None
            if lms_aligned is not None:
                illum_map = self.illumination_est.render(
                    lms_aligned, ah, aw,
                    sh_coeff=None,
                ) if self.illumination_est else ref_aligned
                shadow_map = self.shadow_corrector.extract_shadow_map(ref_aligned, illum_map)
                face = self.shadow_corrector.apply_shadow(face, shadow_map)

        pkt.lighting_corrected_face = face

    def _stage_blend(self, pkt: FramePacket) -> None:
        """Generate mask, refine edges, and composite the face."""
        if pkt.output_frame is not None:
            return

        face = pkt.lighting_corrected_face
        if face is None or pkt.face_align_mat is None or pkt.face_bbox is None:
            return

        h, w = pkt.frame_bgr.shape[:2]

        # -- Robust mask generation with three-tier fallback --
        mask_generated = False

        # Tier 1: 106-pt landmarks -> convex hull mask
        if pkt.face_landmarks_106 is not None and len(pkt.face_landmarks_106) >= 72:
            pkt.blend_mask = self.mask_generator.generate(pkt.face_landmarks_106, h, w)
            if pkt.blend_mask is not None and pkt.blend_mask.sum() > 100:
                mask_generated = True

        # Tier 2: 5-point keypoints -> dilated convex hull mask
        if not mask_generated and pkt.face_kps is not None:
            kps_hull = cv2.convexHull(pkt.face_kps.astype(np.int32))
            mask_uint8 = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask_uint8, kps_hull, 255)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=2)
            dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
            feather = self.mask_generator.feather_px
            pkt.blend_mask = np.clip(dist / (feather + 1e-6), 0, 1).astype(np.float32)
            mask_generated = True

        # Tier 3: bbox -> elliptical mask with generous margin
        if not mask_generated:
            pkt.blend_mask = self.mask_generator.generate_from_bbox(
                pkt.face_bbox, h, w, margin=0.15
            )

        # Apply alpha correction to fix holes, spill, and jagged edges
        try:
            from blending.mask_debug import AlphaCorrector  # lazy
            pkt.blend_mask = AlphaCorrector.fix_mask(pkt.blend_mask)
        except Exception:
            pass  # AlphaCorrector is optional

        # Refine edges with guided filter
        guide = cv2.cvtColor(pkt.frame_bgr, cv2.COLOR_BGR2GRAY)
        pkt.refined_mask = self.edge_refiner.refine(pkt.blend_mask, guide)

        # Warp corrected face back to original frame coordinates
        if pkt.aligned_face is not None:
            ah, aw = pkt.aligned_face.shape[:2]
            if face.shape[:2] != (ah, aw):
                face = cv2.resize(face, (aw, ah))
        M_inv = cv2.invertAffineTransform(pkt.face_align_mat)
        face_warped = cv2.warpAffine(face, M_inv, (w, h), flags=cv2.INTER_CUBIC)

        # Blend
        mask_uint8 = (pkt.refined_mask * 255).astype(np.uint8)

        if self.poisson_blender is not None and pkt.face_bbox is not None:
            x1, y1, x2, y2 = pkt.face_bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            try:
                pkt.blended_frame = self.poisson_blender.blend(face_warped, pkt.frame_bgr, mask_uint8, (cx, cy))
            except cv2.error:
                pkt.blended_frame = self._alpha_blend(face_warped, pkt.frame_bgr, pkt.refined_mask)
        else:
            pkt.blended_frame = self._alpha_blend(face_warped, pkt.frame_bgr, pkt.refined_mask)

    def _stage_temporal(self, pkt: FramePacket, prev_frame: Optional[np.ndarray]) -> None:
        """Apply temporal smoothing using optical flow and EMA.

        Only smooths the face region (not the entire frame) to prevent
        background blur / ghosting artifacts.
        """
        if pkt.output_frame is not None:
            return

        if self.temporal_smoother is None or pkt.blended_frame is None:
            pkt.output_frame = pkt.blended_frame
            return

        # Compute optical flow if we have a previous frame
        if prev_frame is not None and self.flow_engine is not None:
            pkt.optical_flow = self.flow_engine.compute(prev_frame, pkt.frame_bgr)

        # Smooth only the face region using the blend mask
        if pkt.refined_mask is not None and self.temporal_smoother.ema is not None:
            smoothed_full = self.temporal_smoother.smooth_face(pkt.blended_frame)
            mask3 = pkt.refined_mask[:, :, np.newaxis]
            pkt.output_frame = np.clip(
                smoothed_full.astype(np.float32) * mask3
                + pkt.blended_frame.astype(np.float32) * (1 - mask3),
                0, 255,
            ).astype(np.uint8)
        else:
            # First frame or no mask: just smooth directly
            pkt.output_frame = self.temporal_smoother.smooth_face(pkt.blended_frame)

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _alpha_blend(
        face: np.ndarray, background: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Simple alpha blend: face * mask + background * (1 - mask)."""
        mask3 = mask[:, :, np.newaxis]
        blended = (face.astype(np.float32) * mask3 + background.astype(np.float32) * (1 - mask3))
        return np.clip(blended, 0, 255).astype(np.uint8)

    # ==================================================================
    # Optional GFPGAN face restoration (post-blend per-frame stage)
    # ==================================================================

    def _init_gfpgan_restorer(self):
        """Lazy-init a GFPGANer instance. Cached so subsequent frames
        don't re-init. Returns the restorer or None if init failed
        (failure is sticky -- we don't retry every frame, just skip)."""
        cached = getattr(self, "_gfpgan_restorer", "unset")
        if cached != "unset":
            return cached
        try:
            # Disable cuDNN BEFORE touching torch/gfpgan. Windows DLL
            # search order can pick up an older system-installed
            # cudnn_cnn64_8.dll while PyTorch's bundled
            # cudnn_cnn64_9.dll waits behind it, producing WinError 127
            # ("specified procedure could not be found"). With cuDNN
            # disabled, PyTorch falls back to non-cuDNN CUDA kernels
            # (cuBLAS / native conv impls). Marginally slower than full
            # cuDNN but works on every Windows config we've seen.
            try:
                import torch as _torch_cudnn_off
                _torch_cudnn_off.backends.cudnn.enabled = False
            except Exception:
                pass
            # Reuse the proven setup from core.lipsync: it installs gfpgan,
            # patches basicsr's torchvision.functional_tensor import, and
            # gives us the canonical GFPGANv1.4 model path.
            from core.lipsync import _ensure_gfpgan, GFPGAN_MODEL
            _ensure_gfpgan(log=lambda *_a, **_k: None)
            from gfpgan import GFPGANer
            self._gfpgan_restorer = GFPGANer(
                model_path=GFPGAN_MODEL,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
            logger.info("GFPGAN restorer initialized (model=%s)", GFPGAN_MODEL)
        except Exception as exc:
            logger.warning("GFPGAN init failed; enhancement disabled this "
                           "session: %s", exc)
            self._gfpgan_restorer = None
        return self._gfpgan_restorer

    def _stage_enhance(self, pkt: FramePacket) -> None:
        """Post-blend face restoration. Runs after _stage_temporal.

        cfg["enhancement"]["method"] = "gfpgan"  -> GFPGAN v1.4 enhance
        cfg["enhancement"]["method"] = anything else (including "none"
        or missing) -> no-op.

        Operates on pkt.output_frame (the blended/temporally-smoothed
        result) and overwrites it in place. GFPGAN handles detect +
        enhance + paste-back internally so the face region is restored
        wherever it appears in the frame.
        """
        enh_cfg = self.cfg.get("enhancement", {}) or {}
        method = str(enh_cfg.get("method", "none") or "none").lower()
        if method != "gfpgan":
            return
        frame = pkt.output_frame
        if frame is None:
            return
        restorer = self._init_gfpgan_restorer()
        if restorer is None:
            return
        try:
            _, _, restored = restorer.enhance(
                frame, has_aligned=False, only_center_face=False,
                paste_back=True,
            )
            if restored is not None:
                # Optional alpha blend with the un-enhanced frame.
                # blend=1.0 (default) -> full GFPGAN; blend=0.0 ->
                # bypass GFPGAN entirely (same as method="none"); in
                # between -> partial restoration ("less plasticky").
                blend = float(enh_cfg.get("blend", 1.0))
                blend = max(0.0, min(1.0, blend))
                if blend >= 0.999:
                    pkt.output_frame = restored
                else:
                    pkt.output_frame = (
                        blend * restored.astype(np.float32)
                        + (1.0 - blend) * frame.astype(np.float32)
                    ).clip(0, 255).astype(np.uint8)
        except Exception as exc:
            logger.warning("GFPGAN enhance failed on frame %d: %s",
                           pkt.frame_idx, exc)


    def _extract_source_embedding(self, source_path: str) -> np.ndarray:
        """Detect face in source image, extract ArcFace embedding.

        Tries InsightFace first (most reliable), falls back to custom detector.
        """
        img = cv2.imread(source_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read source image: {source_path}")

        # -- Try InsightFace FaceAnalysis --
        if self._insightface_app is not None:
            try:
                ifaces = self._insightface_app.get(img)
                if ifaces:
                    face = max(ifaces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    if self.detector is not None:
                        align_mat = self.detector.compute_align_mat(face.kps)
                    else:
                        align_mat = self._compute_align_mat_standalone(face.kps)
                    if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
                        embedding = face.normed_embedding.astype(np.float32)
                        logger.info(
                            "Source embedding extracted via InsightFace (norm=%.3f)",
                            np.linalg.norm(embedding),
                        )
                        return embedding
            except Exception as exc:
                logger.debug("InsightFace source extraction failed: %s", exc)

        # -- Fallback to custom detector --
        if self.detector is not None and self.landmark_extractor is not None:
            faces = self.detector.detect(img)
            if not faces:
                raise ValueError(f"No face detected in source image: {source_path}")

            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            align_mat = self.detector.compute_align_mat(face.kps)
            aligned = self.detector.warp_align(img, align_mat)

            embedding = self.landmark_extractor.extract_embedding(aligned)
            if embedding is None:
                raise ValueError("Failed to extract source embedding.")

            logger.info("Source embedding extracted (norm=%.3f)", np.linalg.norm(embedding))
            return embedding

        raise ValueError("No face detection method available. Install insightface or fix custom detector.")

    @staticmethod
    def _compute_align_mat_standalone(kps: np.ndarray) -> np.ndarray:
        """Compute alignment matrix without requiring FaceDetector instance.

        Uses the InsightFace standard 5-point template for 112x112 alignment.
        """
        TEMPLATE_5PT = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        M, _ = cv2.estimateAffinePartial2D(
            kps.astype(np.float64),
            TEMPLATE_5PT.astype(np.float64),
        )
        if M is None:
            # Fallback similarity transform
            n = kps.shape[0]
            A = np.zeros((2 * n, 4), dtype=np.float64)
            b = np.zeros((2 * n, 1), dtype=np.float64)
            for i in range(n):
                A[2 * i] = [kps[i, 0], -kps[i, 1], 1, 0]
                A[2 * i + 1] = [kps[i, 1], kps[i, 0], 0, 1]
                b[2 * i] = TEMPLATE_5PT[i, 0]
                b[2 * i + 1] = TEMPLATE_5PT[i, 1]
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            M = np.array([[x[0, 0], -x[1, 0], x[2, 0]],
                          [x[1, 0], x[0, 0], x[3, 0]]], dtype=np.float32)
        else:
            M = M.astype(np.float32)
        return M

    @staticmethod
    def _warp_align_standalone(img: np.ndarray, M: np.ndarray, size: int = 112) -> np.ndarray:
        """Warp the face region to 112x112 without requiring FaceDetector instance."""
        return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR)

    @staticmethod
    def _find_model(model_name: str) -> str:
        """Search for the ONNX model in common locations."""
        root = Path(__file__).resolve().parents[1]
        face_analysis_root = root / "models" / "face_analysis" / "models"

        candidates = [
            str(root / "models" / "face_swap" / f"{model_name}.onnx"),
            str(root / "models" / f"{model_name}.onnx"),
            str(root / "models" / "face_analysis" / f"{model_name}.onnx"),
            str(face_analysis_root / "antelopev2" / f"{model_name}.onnx"),
            str(face_analysis_root / "buffalo_l" / f"{model_name}.onnx"),
            f"./models/face_swap/{model_name}.onnx",
            f"./models/{model_name}.onnx",
            f"./models/face_analysis/buffalo_l/{model_name}.onnx",
            model_name,
            f"{model_name}.onnx",
        ]

        # Also check InsightFace default location
        import os
        home = os.path.expanduser("~")
        candidates.extend([
            os.path.join(home, ".insightface", "models", "antelopev2", f"{model_name}.onnx"),
            os.path.join(home, ".insightface", "models", "buffalo_l", f"{model_name}.onnx"),
            os.path.join(home, ".insightface", "models", f"{model_name}.onnx"),
        ])

        # Try the package's models directory
        try:
            from models import get_registry
            registry = get_registry()
            info = registry.find(model_name, "face_swap") or registry.find(model_name)
            if info is not None:
                return str(info.path)
        except Exception:
            pass

        for path in candidates:
            if Path(path).exists():
                return path

        # Return first candidate -- onnxruntime will raise a clear error if not found
        logger.warning("Model '%s' not found in any search path. Tried: %s", model_name, candidates)
        return candidates[0]

    @staticmethod
    def _resolve_face_analysis_pack(face_analysis_cfg: dict) -> str:
        models_root = FACE_ANALYSIS_ROOT / "models"
        requested = face_analysis_cfg.get("pack", DEFAULT_FACE_ANALYSIS_PACK)
        fallback_packs = tuple(face_analysis_cfg.get("fallback_packs", list(FALLBACK_FACE_ANALYSIS_PACKS)))
        for pack in (requested, *fallback_packs):
            if pack and (models_root / pack).exists():
                return pack
        return requested or DEFAULT_FACE_ANALYSIS_PACK
