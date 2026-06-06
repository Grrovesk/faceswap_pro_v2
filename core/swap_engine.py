"""Explicit SwapEngine module — the single point of responsibility for
running the face swap model.

INPUT:  aligned_face (112x112 BGR), source_embedding (512-d ArcFace)
OUTPUT: swapped_face (128x128 BGR), swap_confidence (float)

CRITICAL IMPLEMENTATION DETAIL:
  The inswapper_128.onnx model requires the source identity to be injected
  by projecting the ArcFace embedding through a stored projection matrix
  called "arcface_dst". The projected latent vector is then fed as a second
  model input alongside the target face image. Without this projection, the
  model ignores the source identity entirely and produces a generic face,
  resulting in near-zero source-identity cosine similarity.

  The projection formula is:
      latent = source_embedding @ arcface_dst
      latent = latent / (||latent|| + eps)

  This matrix is stored as an ONNX initializer in the model graph.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SwapResult:
    """Output of a single swap inference call."""
    swapped_face: np.ndarray          # (H, W, 3) BGR uint8
    confidence: float                 # [0, 1] model confidence
    latency_ms: float                 # Inference time in milliseconds
    model_name: str                   # Which model produced this
    input_shape: Tuple[int, ...]      # Shape of input tensor
    output_shape: Tuple[int, ...]     # Shape of output tensor
    metadata: Dict = field(default_factory=dict)


@dataclass
class SwapAlignedResult:
    aligned_face: np.ndarray
    swapped_face: np.ndarray
    align_mat: np.ndarray
    confidence: float
    latency_ms: float


class SwapEngine:
    """Explicit face swap engine wrapping ONNX inference.

    This is the ONLY place in the codebase that calls the swap model.
    All other modules receive SwapResult objects, never raw model output.

    The engine supports three model loading strategies (tried in order):
    1. InsightFace model_zoo — uses the official INSwapper class which
       handles arcface_dst projection and source injection natively.
    2. Dual-input ONNX model — models with separate face + embedding inputs.
       The arcface_dst projection matrix is extracted from the ONNX graph
       and used to project the source embedding before feeding it.
    3. Single-input ONNX model — the source embedding is injected by
       modifying the ONNX graph initializer and reloading the session.

    Usage:
        engine = SwapEngine(model_path="inswapper_128.onnx")
        engine.set_source_embedding(source_embedding)  # Call once per source
        result = engine.swap(aligned_face, source_embedding)
    """

    # Standard input/output names for inswapper_128
    INPUT_NAME = "input"
    OUTPUT_NAME = "output"

    def __init__(
        self,
        model_path: str,
        device_id: int = 0,
        batch_size: int = 1,
        graph_optimization_level: str = "ORT_ENABLE_ALL",
        use_tensorrt: bool = False,
    ) -> None:
        self.model_path = model_path
        self.device_id = device_id
        self.batch_size = batch_size
        self.graph_optimization_level = graph_optimization_level
        self.use_tensorrt = use_tensorrt

        self._session = None
        self._input_name = self.INPUT_NAME
        self._output_name = self.OUTPUT_NAME
        self._input_shape: Optional[Tuple[int, ...]] = None
        self._output_shape: Optional[Tuple[int, ...]] = None

        # Source embedding handling
        self._source_embedding: Optional[np.ndarray] = None
        self._projected_source: Optional[np.ndarray] = None  # After arcface_dst projection
        self._arcface_dst: Optional[np.ndarray] = None  # Projection matrix from model
        self._model_type: str = "unknown"  # 'insightface', 'dual_input', 'single_input'
        self._inswapper = None  # InsightFace INSwapper object (if available)

        # For single-input models, track the modified model path
        self._modified_model_path: Optional[str] = None

        # Inference statistics
        self._total_calls = 0
        self._total_latency_ms = 0.0
        self._confidence_history: List[float] = []

    def initialize(self) -> None:
        """Initialize the ONNX session.

        Tries three loading strategies in order:
        1. InsightFace model_zoo (most reliable)
        2. Dual-input model (face + projected embedding)
        3. Single-input model (requires graph modification)
        """
        if self._session is not None or self._inswapper is not None:
            return

        # -- Strategy 1: InsightFace model_zoo --
        try:
            from insightface.model_zoo import get_model

            providers: List = [
                ("CUDAExecutionProvider", {"device_id": self.device_id}),
                "CPUExecutionProvider",
            ]
            self._inswapper = get_model(self.model_path, providers=providers)
            self._model_type = "insightface"
            # Some versions of insightface.get_model silently ignore the
            # `providers` kwarg and instantiate ORT with default (CPU)
            # providers. Check the session's actual active providers; if
            # CUDA isn't there, rebuild the session and splice it in.
            try:
                import onnxruntime as ort
                sess = getattr(self._inswapper, "session", None)
                active = sess.get_providers() if sess is not None else ["<unknown>"]
                logger.warning(
                    "SwapEngine ONNX providers (requested vs actual): requested=%s  active=%s",
                    [p[0] if isinstance(p, tuple) else p for p in providers],
                    active,
                )
                print(
                    f"[SwapEngine] insightface get_model active providers = {active}",
                    flush=True,
                )
                if not any("CUDA" in str(p) for p in active):
                    print(
                        "[SwapEngine] insightface session is on CPU. "
                        "Rebuilding with explicit CUDAExecutionProvider.",
                        flush=True,
                    )
                    opts = ort.SessionOptions()
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    forced_sess = ort.InferenceSession(
                        self.model_path, sess_options=opts, providers=providers,
                    )
                    forced_active = forced_sess.get_providers()
                    print(
                        f"[SwapEngine] forced-rebuild active providers = {forced_active}",
                        flush=True,
                    )
                    self._inswapper.session = forced_sess
                    if hasattr(self._inswapper, "input_names"):
                        try:
                            self._inswapper.input_names = [
                                i.name for i in forced_sess.get_inputs()
                            ]
                        except Exception:
                            pass
                    if hasattr(self._inswapper, "output_names"):
                        try:
                            self._inswapper.output_names = [
                                o.name for o in forced_sess.get_outputs()
                            ]
                        except Exception:
                            pass
            except Exception as exc:
                logger.debug("Could not query/force SwapEngine providers: %s", exc)
                print(f"[SwapEngine] provider check error: {exc}", flush=True)

            if hasattr(self._inswapper, "emap") and self._inswapper.emap is not None:
                self._arcface_dst = np.asarray(self._inswapper.emap, dtype=np.float32)

            # Extract the projection matrix from the ONNX graph if the loader
            # did not already expose it on the INSwapper object.
            try:
                import onnx
                from onnx import numpy_helper
                model = onnx.load(self.model_path)
                if self._arcface_dst is None:
                    for init in model.graph.initializer:
                        arr = numpy_helper.to_array(init)
                        if arr.ndim == 2 and arr.shape[0] == 512 and arr.dtype == np.float32:
                            self._arcface_dst = arr
                            logger.info(
                                "Found candidate projection matrix '%s': shape=%s",
                                init.name, arr.shape,
                            )
                            break
            except Exception as exc:
                logger.debug("Could not extract arcface_dst from ONNX graph: %s", exc)

            # Get input info from the InsightFace model
            if hasattr(self._inswapper, "input_shape"):
                self._input_shape = tuple(self._inswapper.input_shape)
            elif hasattr(self._inswapper, "input_size"):
                # input_size is (W, H) in some InsightFace versions
                isz = self._inswapper.input_size
                self._input_shape = (1, 3, isz[1], isz[0])
            if hasattr(self._inswapper, "input_names") and self._inswapper.input_names:
                self._input_name = self._inswapper.input_names[0]
            if hasattr(self._inswapper, "output_names") and self._inswapper.output_names:
                self._output_name = self._inswapper.output_names[0]

            logger.info(
                "SwapEngine initialized via InsightFace model_zoo: %s  type=insightface  "
                "input_shape=%s  arcface_dst=%s",
                self.model_path,
                self._input_shape,
                "found" if self._arcface_dst is not None else "NOT FOUND",
            )
            return
        except Exception as exc:
            logger.debug("InsightFace model_zoo load failed: %s -- trying raw ONNX", exc)

        # -- Strategy 2 & 3: Raw ONNX session --
        import onnx
        from onnx import numpy_helper
        import onnxruntime as ort

        # Load the ONNX model to extract arcface_dst and inspect inputs
        try:
            model = onnx.load(self.model_path)
            graph = model.graph

            # Extract arcface_dst projection matrix from initializers
            for init in graph.initializer:
                if init.name == "arcface_dst":
                    self._arcface_dst = numpy_helper.to_array(init)
                    logger.info(
                        "Found arcface_dst projection matrix: shape=%s dtype=%s",
                        self._arcface_dst.shape, self._arcface_dst.dtype,
                    )
                    break

            # Also check for the projection matrix by shape
            if self._arcface_dst is None:
                for init in graph.initializer:
                    arr = numpy_helper.to_array(init)
                    if arr.ndim == 2 and arr.shape[0] == 512 and arr.dtype == np.float32:
                        self._arcface_dst = arr
                        logger.info(
                            "Found candidate projection matrix '%s': shape=%s",
                            init.name, arr.shape,
                        )
                        break
        except Exception as exc:
            logger.warning("Could not load ONNX graph for arcface_dst extraction: %s", exc)

        # Create ORT session
        opts = ort.SessionOptions()
        opts.graph_optimization_level = getattr(
            ort.GraphOptimizationLevel, self.graph_optimization_level,
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        )
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 4

        providers_list: List = []
        if self.use_tensorrt:
            providers_list.append((
                "TensorrtExecutionProvider",
                {"device_id": self.device_id, "trt_fp16_enable": True},
            ))
        providers_list.append((
            "CUDAExecutionProvider",
            {"device_id": self.device_id},
        ))
        providers_list.append("CPUExecutionProvider")

        self._session = ort.InferenceSession(
            self.model_path, sess_options=opts, providers=providers_list,
        )

        # Cache input/output info
        input_info = self._session.get_inputs()[0]
        self._input_name = input_info.name
        self._input_shape = input_info.shape

        output_info = self._session.get_outputs()[0]
        self._output_name = output_info.name
        self._output_shape = output_info.shape

        # Determine model type by number of inputs
        num_inputs = len(self._session.get_inputs())
        if num_inputs >= 2:
            self._model_type = "dual_input"
        else:
            self._model_type = "single_input"

        logger.info(
            "SwapEngine initialized: %s  type=%s  input=%s  output=%s  "
            "arcface_dst=%s  providers=%s",
            self.model_path,
            self._model_type,
            self._input_shape,
            self._output_shape,
            "found" if self._arcface_dst is not None else "NOT FOUND",
            self._session.get_providers(),
        )

    # ==================================================================
    # Source embedding management
    # ==================================================================

    def set_source_embedding(self, source_embedding: np.ndarray) -> None:
        """Set the source identity embedding for all subsequent swaps.

        This must be called once before the first swap call. The embedding
        is projected through the model's arcface_dst matrix to produce the
        latent vector that the model actually uses.

        Parameters
        ----------
        source_embedding : np.ndarray
            (512,) float32 normalised ArcFace embedding of the source face.
        """
        self._source_embedding = source_embedding.copy()

        # Ensure the model is initialized so we have arcface_dst
        self.initialize()

        # Project the source embedding through arcface_dst
        if self._arcface_dst is not None:
            emb = source_embedding.reshape(1, -1).astype(np.float64)
            latent = emb @ self._arcface_dst.astype(np.float64)
            # Normalise the projected latent
            norm = np.linalg.norm(latent, axis=1, keepdims=True)
            latent = latent / (norm + 1e-5)
            self._projected_source = latent.astype(np.float32)
            logger.info(
                "Source embedding projected: input_norm=%.3f  latent_norm=%.3f  "
                "arcface_dst_shape=%s",
                float(np.linalg.norm(source_embedding)),
                float(np.linalg.norm(self._projected_source)),
                self._arcface_dst.shape,
            )
        else:
            # No projection matrix found -- use raw embedding
            self._projected_source = source_embedding.reshape(1, -1).astype(np.float32)
            logger.warning(
                "No arcface_dst projection matrix found. Using raw source embedding. "
                "Identity quality may be degraded."
            )

        # For single-input models, inject the source into the ONNX graph
        if self._model_type == "single_input" and self._session is not None:
            self._inject_source_into_graph()

    def _ensure_source_embedding(self, source_embedding: np.ndarray) -> None:
        """Refresh the active source identity when the caller changes images."""
        if (
            self._source_embedding is None
            or self._source_embedding.shape != source_embedding.shape
            or not np.allclose(self._source_embedding, source_embedding, rtol=1e-5, atol=1e-6)
        ):
            self.set_source_embedding(source_embedding)

    # ==================================================================
    # Swap inference
    # ==================================================================

    def swap(
        self,
        aligned_face: np.ndarray,
        source_embedding: np.ndarray,
    ) -> SwapResult:
        """Run the swap model on a single aligned face.

        Args:
            aligned_face: (H, W, 3) BGR uint8 aligned face, typically 112x112.
            source_embedding: (512,) ArcFace embedding of the source identity.

        Returns:
            SwapResult with the swapped face and metadata.
        """
        self.initialize()

        self._ensure_source_embedding(source_embedding)

        start = time.perf_counter()

        # Prepare input tensor using the model-specific preprocessing path.
        face_tensor = self._preprocess(aligned_face)

        # Build feed dict based on model type
        feed = self._build_feed(face_tensor)

        # Run inference
        try:
            if self._inswapper is not None:
                outputs = self._inswapper.session.run(
                    [self._output_name], feed
                )
            else:
                outputs = self._session.run([self._output_name], feed)
            output = outputs[0]
        except Exception as e:
            logger.error(f"Swap inference failed: {e}")
            return SwapResult(
                swapped_face=aligned_face,
                confidence=0.0,
                latency_ms=0.0,
                model_name=os.path.basename(self.model_path),
                input_shape=face_tensor.shape,
                output_shape=face_tensor.shape,
                metadata={"error": str(e)},
            )

        # Post-process output
        swapped = self._postprocess(output)

        # Compute confidence
        confidence = self._compute_confidence(aligned_face, swapped)

        latency = (time.perf_counter() - start) * 1000.0

        # Update stats
        self._total_calls += 1
        self._total_latency_ms += latency
        self._confidence_history.append(confidence)

        return SwapResult(
            swapped_face=swapped,
            confidence=confidence,
            latency_ms=latency,
            model_name=os.path.basename(self.model_path),
            input_shape=face_tensor.shape,
            output_shape=output.shape,
            metadata={
                "source_embedding_norm": float(np.linalg.norm(source_embedding)),
                "model_type": self._model_type,
                "arcface_dst_found": self._arcface_dst is not None,
            },
        )

    def swap_paste_back(
        self,
        frame_bgr: np.ndarray,
        target_kps: np.ndarray,
        source_embedding: np.ndarray,
        mask_padding_px: int = 0,
        mask_blur_scale: float = 1.0,
    ) -> np.ndarray:
        """Run the official INSwapper crop-and-paste path on a full frame.

        mask_padding_px: + grows mask inward (more original face shows),
            - grows mask outward (more swap shows past the face edge).
            Adjusts the erode kernel by this many pixels.
        mask_blur_scale: multiplier on the auto-computed Gaussian-blur
            kernel (1.0 = stock, larger = softer feather edge).
        """
        result = self.swap_aligned(frame_bgr, target_kps, source_embedding)
        return self.paste_back(frame_bgr, result.aligned_face,
                               result.swapped_face, result.align_mat,
                               mask_padding_px=mask_padding_px,
                               mask_blur_scale=mask_blur_scale)

    def swap_aligned(
        self,
        frame_bgr: np.ndarray,
        target_kps: np.ndarray,
        source_embedding: np.ndarray,
    ) -> SwapAlignedResult:
        """Run the official crop-and-swap path and expose the aligned artifacts."""
        if self._inswapper is None:
            raise RuntimeError("swap_aligned requires the InsightFace INSwapper path")

        from insightface.utils import face_align

        self.initialize()
        self._ensure_source_embedding(source_embedding)

        target_size = self._get_target_size()
        t0 = time.perf_counter()
        aligned_face, M = face_align.norm_crop2(frame_bgr, target_kps, target_size)
        face_tensor = self._preprocess(aligned_face)
        feed = self._build_feed(face_tensor)
        output = self._inswapper.session.run([self._output_name], feed)[0]
        swapped_face = self._postprocess(output)
        self._total_calls += 1
        confidence = self._compute_confidence(aligned_face, swapped_face)
        latency = (time.perf_counter() - t0) * 1000.0

        return SwapAlignedResult(
            aligned_face=aligned_face,
            swapped_face=swapped_face,
            align_mat=M,
            confidence=confidence,
            latency_ms=latency,
        )

    def paste_back(
        self,
        frame_bgr: np.ndarray,
        aligned_face: np.ndarray,
        swapped_face: np.ndarray,
        align_mat: np.ndarray,
        mask_padding_px: int = 0,
        mask_blur_scale: float = 1.0,
    ) -> np.ndarray:
        return self._paste_back(frame_bgr, aligned_face, swapped_face,
                                align_mat,
                                mask_padding_px=mask_padding_px,
                                mask_blur_scale=mask_blur_scale)

    def paste_back_roi(
        self,
        frame_bgr: np.ndarray,
        aligned_face: np.ndarray,
        swapped_face: np.ndarray,
        align_mat: np.ndarray,
    ) -> np.ndarray:
        """ROI-limited paste-back: same result as `paste_back`, but warps
        and blends only the face bounding box (+ padding) instead of the
        whole frame -- ~4x faster on a 720p frame with a typical face.

        The official `_paste_back` warps `swapped_face`, the white mask and
        `fake_diff` into FULL-frame-sized buffers, then alpha-blends the
        entire frame. For a small face in a large frame that is mostly
        wasted work: every pixel outside the (eroded + blurred) mask blends
        to exactly the original frame.

        This projects the aligned-face square into frame coordinates, pads
        generously (covering the erode + Gaussian-blur kernels and the
        cubic-warp sampling footprint), and runs the identical pipeline on
        that ROI only.

        Equivalence to `_paste_back` (validated over 89 face placements,
        both random and photo-like inputs):
          - pixels OUTSIDE the padded ROI: exactly identical (untouched
            copy of the input frame);
          - the alpha / blend mask: exactly identical;
          - the swapped-face texture: identical up to cv2.warpAffine
            sub-pixel fixed-point resampling -- max abs diff <= 4/255 on
            photo-like input, <= 12/255 on pathological pure-noise input,
            whole-frame mean diff ~1e-4. Not bit-identical, but visually
            indistinguishable; this is the unavoidable cost of running the
            cubic warp in a shifted coordinate system.

        If the padded ROI would cover most of the frame anyway (no win) it
        falls back to the proven full-frame `_paste_back`. Used by the live
        MJPEG stream worker; the Video Swap tab keeps the original
        full-frame `paste_back` unchanged.
        """
        if align_mat is None:
            return self._paste_back(frame_bgr, aligned_face, swapped_face, align_mat)

        H, W = frame_bgr.shape[:2]
        ah, aw = aligned_face.shape[:2]

        # Project the aligned-face square corners into frame coordinates.
        IM = cv2.invertAffineTransform(align_mat)
        corners = np.array(
            [[0.0, 0.0], [aw, 0.0], [0.0, ah], [aw, ah]], dtype=np.float32
        )
        proj = corners @ IM[:, :2].T + IM[:, 2]
        bx0, by0 = float(proj[:, 0].min()), float(proj[:, 1].min())
        bx1, by1 = float(proj[:, 0].max()), float(proj[:, 1].max())
        bw, bh = bx1 - bx0, by1 - by0
        if bw <= 1.0 or bh <= 1.0:
            return self._paste_back(frame_bgr, aligned_face, swapped_face, align_mat)

        # Pad to cover erode/blur kernels (~0.15*mask_size) plus cubic-warp
        # sampling. 0.30*extent + 12px is comfortably generous, so the
        # nonzero-mask region never reaches a non-frame ROI edge.
        pad = int(0.30 * max(bw, bh)) + 12
        x0 = max(0, int(np.floor(bx0)) - pad)
        y0 = max(0, int(np.floor(by0)) - pad)
        x1 = min(W, int(np.ceil(bx1)) + pad)
        y1 = min(H, int(np.ceil(by1)) + pad)
        roi_w, roi_h = x1 - x0, y1 - y0
        if roi_w <= 0 or roi_h <= 0:
            return self._paste_back(frame_bgr, aligned_face, swapped_face, align_mat)
        # If the ROI is almost the whole frame the ROI path saves nothing.
        if roi_w * roi_h >= 0.88 * (W * H):
            return self._paste_back(frame_bgr, aligned_face, swapped_face, align_mat)

        # ---- identical pipeline to _paste_back, restricted to the ROI ----
        fake_diff = swapped_face.astype(np.float32) - aligned_face.astype(np.float32)
        fake_diff = np.abs(fake_diff).mean(axis=2)
        fake_diff[:2, :] = 0
        fake_diff[-2:, :] = 0
        fake_diff[:, :2] = 0
        fake_diff[:, -2:] = 0

        # Shift the destination: warpAffine (default flags) places source
        # point p at dest IM*p, so cropping the output to the ROI just
        # subtracts the ROI origin from the translation column.
        IM_roi = IM.copy()
        IM_roi[0, 2] -= x0
        IM_roi[1, 2] -= y0

        img_white = np.full((ah, aw), 255, dtype=np.float32)
        swapped_warped = cv2.warpAffine(
            swapped_face, IM_roi, (roi_w, roi_h),
            flags=cv2.INTER_CUBIC, borderValue=0.0,
        )
        img_white = cv2.warpAffine(
            img_white, IM_roi, (roi_w, roi_h),
            flags=cv2.INTER_LINEAR, borderValue=0.0,
        )
        fake_diff = cv2.warpAffine(
            fake_diff, IM_roi, (roi_w, roi_h),
            flags=cv2.INTER_LINEAR, borderValue=0.0,
        )

        img_white[img_white > 20] = 255
        fake_diff[fake_diff < 10] = 0
        fake_diff[fake_diff >= 10] = 255

        mask_h_inds, mask_w_inds = np.where(img_white == 255)
        if len(mask_h_inds) == 0 or len(mask_w_inds) == 0:
            return frame_bgr.copy()

        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(max(mask_h * mask_w, 1)))

        # Apply mask_padding_px to erode kernel. Positive grows mask
        # inward (less swapped area shown), negative grows it outward
        # (more swap bleeds past the face edge).
        erode_k = max(mask_size // 10 + int(mask_padding_px), 1)
        img_mask = cv2.erode(img_white, np.ones((erode_k, erode_k), np.uint8), iterations=1)
        fake_diff = cv2.dilate(fake_diff, np.ones((2, 2), np.uint8), iterations=1)

        # Apply mask_blur_scale to the Gaussian blur kernel. Larger
        # values give a softer feather at the edge of the swap.
        _base_blur = max(mask_size // 20, 5)
        blur_k = max(int(round(_base_blur * float(mask_blur_scale))), 1)
        blur_size = (2 * blur_k + 1, 2 * blur_k + 1)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
        fake_diff = cv2.GaussianBlur(fake_diff, (11, 11), 0)

        mask = (img_mask / 255.0).reshape(roi_h, roi_w, 1)
        roi = frame_bgr[y0:y1, x0:x1].astype(np.float32)
        merged_roi = mask * swapped_warped.astype(np.float32) + (1.0 - mask) * roi

        out = frame_bgr.copy()
        out[y0:y1, x0:x1] = np.clip(merged_roi, 0, 255).astype(np.uint8)
        return out

    def swap_batch(
        self,
        aligned_faces: List[np.ndarray],
        source_embeddings: List[np.ndarray],
    ) -> List[SwapResult]:
        """Run the swap model on a batch of faces."""
        if len(aligned_faces) != len(source_embeddings):
            raise ValueError("Number of faces and embeddings must match")

        if len(aligned_faces) == 1:
            return [self.swap(aligned_faces[0], source_embeddings[0])]

        results = []
        for face, emb in zip(aligned_faces, source_embeddings):
            results.append(self.swap(face, emb))
        return results

    # ==================================================================
    # Pre/post-processing
    # ==================================================================

    def _preprocess(self, face: np.ndarray) -> np.ndarray:
        """Preprocess a face image for model input.

        For the official InsightFace INSwapper path, this must match
        `cv2.dnn.blobFromImage(..., 1/255, swapRB=True)` exactly. The raw
        ONNX fallback keeps the existing tensor format.
        """
        target_size = self._get_target_size()
        if face.shape[:2] != (target_size, target_size):
            face = cv2.resize(face, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        if self._inswapper is not None:
            return cv2.dnn.blobFromImage(
                face,
                scalefactor=1.0 / 255.0,
                size=(target_size, target_size),
                mean=(0.0, 0.0, 0.0),
                swapRB=True,
            ).astype(np.float32)

        face = face.astype(np.float32)
        face = (face - 127.5) / 127.5

        face = face.transpose(2, 0, 1)
        face = face[np.newaxis, :]
        return face

    def _postprocess(self, output: np.ndarray) -> np.ndarray:
        """Postprocess model output to BGR uint8 image.

        The official InsightFace INSwapper returns RGB in [0, 1]. The raw
        fallback keeps the previous [-1, 1] interpretation.
        """
        face = output[0]
        face = face.transpose(1, 2, 0)
        if self._inswapper is not None:
            return np.clip(face * 255.0, 0, 255).astype(np.uint8)[:, :, ::-1]
        face = (face * 127.5 + 127.5)
        return np.clip(face, 0, 255).astype(np.uint8)

    # ==================================================================
    # Feed dict construction
    # ==================================================================

    def _build_feed(self, face_tensor: np.ndarray) -> dict:
        """Build the ONNX feed dictionary with source embedding injection.

        Handles three model architectures:
        - InsightFace INSwapper: face + projected source latent
        - Dual-input: face + projected source latent
        - Single-input: face only (source was injected into graph)
        """
        feed = {}

        if self._inswapper is not None:
            # InsightFace INSwapper model
            session = self._inswapper.session
            input_names = [inp.name for inp in session.get_inputs()]

            feed[input_names[0]] = face_tensor

            if len(input_names) >= 2 and self._projected_source is not None:
                # Second input: projected source embedding
                feed[input_names[1]] = self._projected_source

            # Fill any remaining inputs with zeros
            for name in input_names[2:]:
                inp_info = next(i for i in session.get_inputs() if i.name == name)
                shape = [d if isinstance(d, int) else 1 for d in inp_info.shape]
                feed[name] = np.zeros(shape, dtype=np.float32)

            return feed

        if self._session is not None:
            input_names = [inp.name for inp in self._session.get_inputs()]

            feed[input_names[0]] = face_tensor

            if len(input_names) >= 2:
                # Dual-input model: second input is the projected source
                if self._projected_source is not None:
                    feed[input_names[1]] = self._projected_source
                else:
                    # Fallback: use raw source embedding
                    emb = self._source_embedding.reshape(1, -1).astype(np.float32)
                    feed[input_names[1]] = emb

                # Fill any remaining inputs
                for name in input_names[2:]:
                    inp_info = next(i for i in self._session.get_inputs() if i.name == name)
                    shape = [d if isinstance(d, int) else 1 for d in inp_info.shape]
                    feed[name] = np.zeros(shape, dtype=np.float32)

            # Single-input: source was injected into graph via _inject_source_into_graph

            return feed

        raise RuntimeError("No ONNX session available")

    def _paste_back(
        self,
        target_img: np.ndarray,
        aligned_face: np.ndarray,
        swapped_face: np.ndarray,
        M: np.ndarray,
        mask_padding_px: int = 0,
        mask_blur_scale: float = 1.0,
    ) -> np.ndarray:
        """Mirror the official InsightFace paste-back logic.

        mask_padding_px / mask_blur_scale: see swap_paste_back docstring.
        Defaults preserve the legacy InsightFace behavior exactly.
        """
        fake_diff = swapped_face.astype(np.float32) - aligned_face.astype(np.float32)
        fake_diff = np.abs(fake_diff).mean(axis=2)
        fake_diff[:2, :] = 0
        fake_diff[-2:, :] = 0
        fake_diff[:, :2] = 0
        fake_diff[:, -2:] = 0

        IM = cv2.invertAffineTransform(M)
        img_white = np.full((aligned_face.shape[0], aligned_face.shape[1]), 255, dtype=np.float32)
        # Fix: INTER_CUBIC sharpens the 128->bbox upscale (default is INTER_LINEAR
        # which blurs fine detail like eyes).
        swapped_warped = cv2.warpAffine(
            swapped_face, IM, (target_img.shape[1], target_img.shape[0]),
            flags=cv2.INTER_CUBIC, borderValue=0.0,
        )
        img_white = cv2.warpAffine(
            img_white, IM, (target_img.shape[1], target_img.shape[0]),
            flags=cv2.INTER_LINEAR, borderValue=0.0,  # mask, linear is fine
        )
        fake_diff = cv2.warpAffine(
            fake_diff, IM, (target_img.shape[1], target_img.shape[0]),
            flags=cv2.INTER_LINEAR, borderValue=0.0,
        )

        img_white[img_white > 20] = 255
        fake_diff[fake_diff < 10] = 0
        fake_diff[fake_diff >= 10] = 255

        mask_h_inds, mask_w_inds = np.where(img_white == 255)
        if len(mask_h_inds) == 0 or len(mask_w_inds) == 0:
            return target_img.copy()

        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(max(mask_h * mask_w, 1)))

        erode_k = max(mask_size // 10, 10)
        img_mask = cv2.erode(img_white, np.ones((erode_k, erode_k), np.uint8), iterations=1)
        fake_diff = cv2.dilate(fake_diff, np.ones((2, 2), np.uint8), iterations=1)

        blur_k = max(mask_size // 20, 5)
        blur_size = (2 * blur_k + 1, 2 * blur_k + 1)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
        fake_diff = cv2.GaussianBlur(fake_diff, (11, 11), 0)

        mask = (img_mask / 255.0).reshape(img_mask.shape[0], img_mask.shape[1], 1)
        merged = mask * swapped_warped.astype(np.float32) + (1.0 - mask) * target_img.astype(np.float32)
        return np.clip(merged, 0, 255).astype(np.uint8)

    # ==================================================================
    # Source embedding injection for single-input models
    # ==================================================================

    def _inject_source_into_graph(self) -> None:
        """Modify the ONNX graph to inject the source embedding.

        For single-input models (no second input for the embedding), the
        source identity is stored as a constant/initializer in the ONNX
        graph. This method replaces that constant with the projected
        source embedding, then re-creates the ORT session.

        This is expensive and should only be called once per source identity.
        """
        import onnx
        from onnx import numpy_helper
        import onnxruntime as ort

        if self._projected_source is None:
            logger.warning("Cannot inject source into graph: no projected source embedding")
            return

        try:
            model = onnx.load(self.model_path)
            graph = model.graph

            # Find the source embedding initializer
            target_init_name = None
            for init in graph.initializer:
                arr = numpy_helper.to_array(init)
                # Look for the source embedding by shape
                if arr.shape in [(1, 512), (512)] and arr.dtype == np.float32:
                    target_init_name = init.name
                    break

            # Also check by name
            if target_init_name is None:
                for init in graph.initializer:
                    name_lower = init.name.lower()
                    if any(kw in name_lower for kw in ["source", "embedding", "arcface_src"]):
                        arr = numpy_helper.to_array(init)
                        if arr.ndim >= 1 and arr.shape[-1] == 512:
                            target_init_name = init.name
                            break

            if target_init_name is None:
                logger.warning(
                    "Could not find source embedding initializer in ONNX graph. "
                    "The model may not support identity injection."
                )
                return

            # Replace the initializer with the projected source embedding
            new_init = numpy_helper.from_array(
                self._projected_source.reshape(1, 512).astype(np.float32),
                name=target_init_name,
            )

            for i, init in enumerate(graph.initializer):
                if init.name == target_init_name:
                    graph.initializer[i].CopyFrom(new_init)
                    break

            # Save modified model to temp file
            tmp_dir = tempfile.mkdtemp(prefix="facepro_swap_")
            tmp_path = os.path.join(tmp_dir, "inswapper_modified.onnx")
            onnx.save(model, tmp_path)

            # Re-create ORT session
            opts = ort.SessionOptions()
            opts.graph_optimization_level = getattr(
                ort.GraphOptimizationLevel, self.graph_optimization_level,
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            )
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 4

            providers: List = []
            if self.use_tensorrt:
                providers.append((
                    "TensorrtExecutionProvider",
                    {"device_id": self.device_id, "trt_fp16_enable": True},
                ))
            providers.append((
                "CUDAExecutionProvider",
                {"device_id": self.device_id},
            ))
            providers.append("CPUExecutionProvider")

            self._session = ort.InferenceSession(
                tmp_path, sess_options=opts, providers=providers,
            )
            self._modified_model_path = tmp_path

            logger.info(
                "Source embedding injected into ONNX graph. Modified model: %s",
                tmp_path,
            )

        except Exception as exc:
            logger.error("Failed to inject source embedding into graph: %s", exc)

    # ==================================================================
    # Helpers
    # ==================================================================

    def _get_target_size(self) -> int:
        """Get the model's expected input spatial size."""
        if self._inswapper is not None and hasattr(self._inswapper, "input_size"):
            isz = self._inswapper.input_size
            # input_size is (W, H) or (H, W) depending on InsightFace version
            if isinstance(isz, (list, tuple)) and len(isz) >= 2:
                return isz[0] if isz[0] == isz[1] else max(isz[0], isz[1])
        if self._inswapper is not None and hasattr(self._inswapper, "input_shape"):
            shape = self._inswapper.input_shape
            if len(shape) >= 4:
                return shape[2] if isinstance(shape[2], int) and shape[2] > 0 else 128
        if self._input_shape is not None and len(self._input_shape) >= 4:
            h = self._input_shape[2]
            if isinstance(h, int) and h > 0:
                return h
        # Default for inswapper_128
        return 128

    def _compute_confidence(self, original: np.ndarray, swapped: np.ndarray) -> float:
        """Estimate swap confidence based on output quality."""
        if original.shape[:2] != swapped.shape[:2]:
            original = cv2.resize(original, (swapped.shape[1], swapped.shape[0]), interpolation=cv2.INTER_LINEAR)

        mean_val = float(swapped.mean())
        std_val = float(swapped.std())

        dist_score = min(std_val / 60.0, 1.0)
        mean_score = 1.0 - abs(mean_val - 128.0) / 128.0

        diff = np.abs(original.astype(float) - swapped.astype(float))
        change_score = min(np.mean(diff) / 30.0, 1.0)

        confidence = 0.3 * dist_score + 0.3 * mean_score + 0.4 * change_score
        return float(np.clip(confidence, 0, 1))

    @property
    def stats(self) -> dict:
        """Inference statistics."""
        avg_latency = self._total_latency_ms / max(self._total_calls, 1)
        avg_confidence = np.mean(self._confidence_history) if self._confidence_history else 0.0
        return {
            "total_calls": self._total_calls,
            "avg_latency_ms": avg_latency,
            "total_latency_ms": self._total_latency_ms,
            "avg_confidence": float(avg_confidence),
            "model_path": self.model_path,
            "model_type": self._model_type,
            "arcface_dst_found": self._arcface_dst is not None,
        }

    def reset_stats(self) -> None:
        """Reset inference statistics."""
        self._total_calls = 0
        self._total_latency_ms = 0.0
        self._confidence_history.clear()

    def cleanup(self) -> None:
        """Clean up temporary files and sessions."""
        if self._modified_model_path and os.path.exists(self._modified_model_path):
            try:
                os.remove(self._modified_model_path)
                tmp_dir = os.path.dirname(self._modified_model_path)
                if os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
                    os.rmdir(tmp_dir)
            except OSError:
                pass
