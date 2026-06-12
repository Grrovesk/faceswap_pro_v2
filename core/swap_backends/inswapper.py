"""InsightFace ``inswapper_128`` backend.

Thin wrapper around :class:`core.swap_engine.SwapEngine`.  The
engine class already implements the full inswapper call surface
(``initialize`` / ``set_source_embedding`` / ``swap`` /
``swap_paste_back`` / ``swap_aligned`` / ``paste_back``); this
module only adds the :class:`BackendInfo` metadata and a stable
``SwapBackend`` superclass so the factory in
``core.swap_backends.__init__`` can register it.

Behavior is byte-identical to direct ``SwapEngine(...)``
instantiation — the same class, the same MRO, the same methods.
"""

from __future__ import annotations

from core.swap_engine import SwapEngine

from .base import BackendInfo, SwapBackend


class InswapperBackend(SwapEngine, SwapBackend):
    """``inswapper_128.onnx`` from InsightFace.

    Default backend.  Native 128 × 128 aligned-face output.  The
    repo combines this with optional GFPGAN restoration and the
    ``pixel_boost`` upscale path in
    ``core/pipeline.py::_stage_swap`` to ship 256 / 512 / 768 crops
    when the user opts in.

    License is community-distributed; commercial deployment should
    be reviewed against InsightFace's terms for your jurisdiction.
    We mark ``commercial_safe=False`` as a conservative default so
    the UI surfaces the caveat to the user.
    """

    info = BackendInfo(
        name="inswapper_128",
        display_name="inswapper_128 (default, fast, 128 native)",
        license_spdx="Custom",
        license_label=(
            "InsightFace community weights — verify license for your "
            "deployment before commercial use"
        ),
        commercial_safe=False,
        native_resolution=128,
        requires_weights=("inswapper_128.onnx",),
        upstream_url="https://github.com/deepinsight/insightface",
        notes=(
            "The original inswapper_128 weights are widely redistributed "
            "but ship under terms whose commercial-use status is not "
            "explicitly stated.  Review with counsel before shipping a "
            "commercial product."
        ),
    )
