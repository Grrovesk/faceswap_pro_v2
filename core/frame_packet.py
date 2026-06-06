"""FramePacket — shared data carrier that flows through the pipeline.

Every module reads from and writes to fields on this dataclass.  This keeps
modules decoupled: no module needs to know about another module's internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class FramePacket:
    """All intermediate results for a single frame.

    Fields are populated progressively as the frame moves through pipeline
    stages.  A ``None`` value means that stage has not yet run.
    """

    # ── Identifiers ──────────────────────────────────────────────────
    frame_idx: int
    frame_bgr: np.ndarray  # Original BGR frame (H, W, 3) uint8

    # ── Detection & alignment ────────────────────────────────────────
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    face_kps: Optional[np.ndarray] = None  # (5, 2) float32 five-point keypoints
    face_landmarks_106: Optional[np.ndarray] = None  # (106, 2) float32
    face_align_mat: Optional[np.ndarray] = None  # (2, 3) affine matrix
    aligned_face: Optional[np.ndarray] = None  # (112, 112, 3) uint8

    # ── Identity tracking ────────────────────────────────────────────
    source_embedding: Optional[np.ndarray] = None  # (512,) float32 ArcFace
    current_embedding: Optional[np.ndarray] = None  # (512,) float32
    identity_score: float = 1.0  # cosine similarity to source
    identity_drifted: bool = False

    # ── Swap output ──────────────────────────────────────────────────
    swapped_face: Optional[np.ndarray] = None  # (128, 128, 3) uint8

    # ── Audio-lip sync ───────────────────────────────────────────────
    viseme_label: Optional[str] = None  # e.g. "viseme_open_wide"
    lip_corrected_face: Optional[np.ndarray] = None  # (128, 128, 3) uint8

    # ── Lighting correction ──────────────────────────────────────────
    color_matched_face: Optional[np.ndarray] = None
    relit_face: Optional[np.ndarray] = None
    shadow_map: Optional[np.ndarray] = None  # (H, W, 3) float32 multiply map
    lighting_corrected_face: Optional[np.ndarray] = None

    # ── Blending ─────────────────────────────────────────────────────
    blend_mask: Optional[np.ndarray] = None  # (H, W) float32 [0, 1]
    refined_mask: Optional[np.ndarray] = None
    blended_frame: Optional[np.ndarray] = None  # (H, W, 3) uint8

    # ── Temporal smoothing ───────────────────────────────────────────
    optical_flow: Optional[np.ndarray] = None  # (H, W, 2) float32
    output_frame: Optional[np.ndarray] = None  # Final (H, W, 3) uint8

    # ── Misc ─────────────────────────────────────────────────────────
    metadata: Dict = field(default_factory=dict)

    # -----------------------------------------------------------------
    # Convenience helpers
    # -----------------------------------------------------------------

    @property
    def face_found(self) -> bool:
        return self.face_bbox is not None

    @property
    def lip_indices(self) -> list[int]:
        """Indices into 106-pt landmark array that correspond to the lip region.

        Based on the InsightFace 106-pt scheme: outer lip = 52-61, inner = 62-71.
        """
        return list(range(52, 72))

    def face_region(self) -> Optional[np.ndarray]:
        """Crop the face region from the original frame using the bounding box."""
        if self.face_bbox is None:
            return None
        x1, y1, x2, y2 = self.face_bbox
        return self.frame_bgr[y1:y2, x1:x2].copy()
