"""Image utility functions: normalization, resizing, affine transforms."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# InsightFace standard alignment template (112×112, 5-point)
# Reference: insightface/python-package/insightface/model_zoo/model_zoo.py
_ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def normalize_face(
    face_img: np.ndarray,
    mean: Tuple[float, ...] = (127.5, 127.5, 127.5),
    std: Tuple[float, ...] = (127.5, 127.5, 127.5),
) -> np.ndarray:
    """Normalize a face image from [0, 255] uint8 to [-1, 1] float32.

    Default normalization matches the InsightFace/ArcFace training regime
    (subtract 127.5, divide by 127.5). Other mean/std combinations can be
    passed for different model requirements.

    Args:
        face_img: BGR uint8 image, shape (H, W, 3).
        mean: Per-channel mean to subtract.
        std: Per-channel std to divide by.

    Returns:
        Normalized float32 image, shape (H, W, 3), range roughly [-1, 1].
    """
    face = face_img.astype(np.float32)
    face -= np.array(mean, dtype=np.float32)
    face /= np.array(std, dtype=np.float32)
    return face


def denormalize_face(
    face_img: np.ndarray,
    mean: Tuple[float, ...] = (127.5, 127.5, 127.5),
    std: Tuple[float, ...] = (127.5, 127.5, 127.5),
) -> np.ndarray:
    """Inverse of normalize_face: [-1, 1] float32 → [0, 255] uint8."""
    face = face_img.copy()
    face *= np.array(std, dtype=np.float32)
    face += np.array(mean, dtype=np.float32)
    return np.clip(face, 0, 255).astype(np.uint8)


def resize_pad(
    img: np.ndarray,
    target_size: Tuple[int, int],
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize image to fit within target_size, preserving aspect ratio, with
    padding to fill the remaining area.

    This is used before feeding faces into models that require a fixed input
    shape but the detected face bbox may not be square.

    Args:
        img: BGR uint8 image.
        target_size: (width, height) of the output.
        pad_color: BGR padding color.

    Returns:
        (resized_img, scale_factor, (pad_x, pad_y))
    """
    h, w = img.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((th, tw, 3), pad_color, dtype=np.uint8)
    pad_x = (tw - new_w) // 2
    pad_y = (th - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    return canvas, scale, (pad_x, pad_y)


def compute_alignment_matrix(
    keypoints: np.ndarray,
    template: np.ndarray | None = None,
    output_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """Compute a 2×3 affine matrix that aligns 5-point facial keypoints to
    the ArcFace standard template.

    The matrix M satisfies:  aligned = M @ [x, y, 1]^T

    Args:
        keypoints: (5, 2) array of (x, y) facial keypoints.
        template: (5, 2) target template. Defaults to _ARCFACE_TEMPLATE_112.
        output_size: (width, height) for the aligned face crop.

    Returns:
        2×3 affine transformation matrix.
    """
    if template is None:
        template = _ARCFACE_TEMPLATE_112.copy()
    M, _ = cv2.estimateAffinePartial2D(
        keypoints.astype(np.float32),
        template.astype(np.float32),
        method=cv2.LMEDS,
    )
    if M is None:
        # Fallback to least-squares
        M, _ = cv2.estimateAffinePartial2D(
            keypoints.astype(np.float32),
            template.astype(np.float32),
        )
    return M


def affine_transform_image(
    img: np.ndarray,
    M: np.ndarray,
    output_size: Tuple[int, int] = (112, 112),
    flags: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Apply an affine transformation to an image.

    Args:
        img: Source BGR image.
        M: 2×3 affine matrix (from compute_alignment_matrix).
        output_size: (width, height) of the output.
        flags: Interpolation flag.
        border_mode: Border extrapolation mode.
        border_value: Fill value for constant border.

    Returns:
        Warped image of shape (output_size[1], output_size[0], 3).
    """
    return cv2.warpAffine(
        img,
        M,
        output_size,
        flags=flags,
        borderMode=border_mode,
        borderValue=border_value,
    )


def inverse_affine_transform_image(
    img: np.ndarray,
    M: np.ndarray,
    output_size: Tuple[int, int],
    flags: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Apply the inverse of an affine transformation.

    Used to map the swapped face from the aligned (112×112) space back to
    the original frame coordinate system.

    Args:
        img: Aligned face image (e.g., 112×112).
        M: The *forward* 2×3 affine matrix that was used for alignment.
        output_size: (width, height) of the original frame.
        flags: Interpolation flag.
        border_mode: Border mode.
        border_value: Fill value.

    Returns:
        Inverse-warped image in the original frame coordinate space.
    """
    M_inv = cv2.invertAffineTransform(M)
    return cv2.warpAffine(
        img,
        M_inv,
        output_size,
        flags=flags,
        borderMode=border_mode,
        borderValue=border_value,
    )


def crop_face_region(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: float = 0.2,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop a face region from a frame with optional margin expansion.

    The margin is expressed as a fraction of the bounding box dimensions and
    is applied symmetrically. The crop coordinates are clamped to the frame
    boundaries.

    Args:
        frame: Full BGR frame.
        bbox: (x1, y1, x2, y2) bounding box.
        margin: Fraction of bbox size to add as margin (default 20%).

    Returns:
        (cropped_face, adjusted_bbox) — the cropped image and the new bbox
        in original frame coordinates.
    """
    x1, y1, x2, y2 = bbox
    h_bbox = y2 - y1
    w_bbox = x2 - x1
    mx = int(w_bbox * margin)
    my = int(h_bbox * margin)

    frame_h, frame_w = frame.shape[:2]
    nx1 = max(0, x1 - mx)
    ny1 = max(0, y1 - my)
    nx2 = min(frame_w, x2 + mx)
    ny2 = min(frame_h, y2 + my)

    cropped = frame[ny1:ny2, nx1:nx2]
    return cropped, (nx1, ny1, nx2, ny2)


def draw_face_info(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    identity_score: float | None = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding box and optional identity score on a frame (in-place).

    Args:
        frame: BGR frame (modified in-place).
        bbox: (x1, y1, x2, y2).
        identity_score: Optional cosine similarity to display.
        color: Box color in BGR.
        thickness: Line thickness.

    Returns:
        The same frame reference.
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if identity_score is not None:
        label = f"id: {identity_score:.3f}"
        cv2.putText(
            frame,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame
