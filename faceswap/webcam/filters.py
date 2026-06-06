"""Output filters applied to each swapped frame BEFORE encoding.

Pure cv2 / numpy, no external deps. Called from worker._loop right
after swap_fn returns. Cheap enough to run at 30 fps:
  brightness/contrast via cv2.convertScaleAbs (single linear op)
  saturation         via HSV multiply on S channel
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


def apply_filters(frame_bgr: "np.ndarray",
                   brightness: int = 0,
                   contrast: float = 1.0,
                   saturation: float = 1.0) -> "np.ndarray":
    """Apply brightness/contrast/saturation in-place-friendly order.

    brightness: -100..+100   (additive offset)
    contrast:   0.5..2.0     (multiplicative gain)
    saturation: 0.0..2.0     (HSV S-channel multiplier; 1.0 = no op)
    """
    if cv2 is None or frame_bgr is None:
        return frame_bgr
    out = frame_bgr
    if int(brightness) != 0 or abs(float(contrast) - 1.0) > 1e-3:
        out = cv2.convertScaleAbs(out, alpha=float(contrast),
                                   beta=int(brightness))
    if abs(float(saturation) - 1.0) > 1e-3:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= float(saturation)
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out
