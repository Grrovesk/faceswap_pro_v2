"""SwapBackend marker class + BackendInfo dataclass.

A backend is a class with:

  * an ``info`` class attribute of type :class:`BackendInfo`
  * a ``SwapEngine``-compatible call surface (duck-typed):

      __init__(self, model_path, device_id=0, batch_size=1,
               graph_optimization_level="ORT_ENABLE_ALL",
               use_tensorrt=False)
      initialize(self) -> None
      set_source_embedding(self, embedding: np.ndarray) -> None
      swap(self, aligned_face, source_embedding) -> SwapResult

  * Optional (callers ``hasattr``-check before invoking):

      swap_paste_back(self, frame_bgr, face_kps, source_embedding,
                      mask_padding_px=0, mask_blur_scale=1.0) -> np.ndarray
      swap_aligned(self, frame_bgr, face_kps, source_embedding)
                                                  -> SwapAlignedResult
      paste_back(self, frame_bgr, aligned_face, swapped_face, M,
                 mask_padding_px=0, mask_blur_scale=1.0) -> np.ndarray

The base class itself does NOT declare these as abstract.  Abstract
methods would force every future backend (GHOST-A, SimSwap-512,
PuLID, …) to implement the inswapper-specific paste-back machinery,
which is the wrong shape for diffusion-based backends.  Pipeline
already ``hasattr``-checks (see ``core/pipeline.py``
``_stage_swap`` → ``hasattr(self.swap_engine, "swap_paste_back")``),
so a backend that omits an optional method falls back to the manual
crop/composite path automatically.

License labels follow the SPDX vocabulary
(https://spdx.org/licenses/) so machine parsing is stable.  The
``commercial_safe`` flag is a conservative summary, not legal
advice; consumers should still read the license itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class BackendInfo:
    """Static metadata about a swap backend.

    Attributes
    ----------
    name : str
        Stable machine identifier, used as the registry key and the
        ``config["swap"]["backend"]`` value.  Lowercase, snake_case.
    display_name : str
        Human-readable label for the UI dropdown.
    license_spdx : str
        SPDX short identifier.  Use ``"Custom"`` for licenses outside
        the SPDX list.  Examples: ``"Apache-2.0"``, ``"CC-BY-NC-4.0"``.
    license_label : str
        One-line plain-English license summary suitable for UI display.
    commercial_safe : bool
        Conservative summary: ``True`` only when the upstream license
        explicitly permits commercial redistribution of the weights
        AND the inference pipeline.  When in doubt, leave ``False``.
    native_resolution : int
        Side length (px) of the model's native aligned-face output.
        Used by the UI to disable ``pixel_boost`` when it would be a
        no-op (``pixel_boost <= native_resolution``).
    requires_weights : Tuple[str, ...]
        Filenames the backend expects under the project's weights
        directory.  Used by ``detect_system.py`` to surface a missing-
        weights WARN.
    upstream_url : str
        Canonical upstream repo URL (for THIRD_PARTY_NOTICES.md).
    notes : str
        Long-form caveats.  Shown in the UI tooltip / INSTALL.md.
    """

    name: str
    display_name: str
    license_spdx: str
    license_label: str
    commercial_safe: bool
    native_resolution: int
    requires_weights: Tuple[str, ...] = field(default_factory=tuple)
    upstream_url: str = ""
    notes: str = ""


class SwapBackend:
    """Marker base class for swap backends.

    Concrete backends MUST set ``info`` as a class attribute of
    type :class:`BackendInfo`.  The factory in
    ``core.swap_backends.__init__`` validates this on registration.
    """

    info: Optional[BackendInfo] = None
