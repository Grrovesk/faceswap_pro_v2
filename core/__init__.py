"""Core package exports with lazy imports to avoid circular import traps."""

from __future__ import annotations

__all__ = [
    "FaceSwapPipeline",
    "FramePacket",
    "create_ort_session",
    "optimize_onnx_graph",
    "SwapEngine",
    "SwapResult",
    "SwapAlignedResult",
    "SwapBackend",
    "BackendInfo",
    "get_backend",
    "list_backends",
    "FrameMetrics",
    "FrameMetricsCollector",
    "FailureLog",
    "FailureCategory",
    "FailureSeverity",
]


def __getattr__(name: str):
    if name == "FaceSwapPipeline":
        from .pipeline import FaceSwapPipeline
        return FaceSwapPipeline
    if name == "FramePacket":
        from .frame_packet import FramePacket
        return FramePacket
    if name in {"create_ort_session", "optimize_onnx_graph"}:
        from .onnx_opt import create_ort_session, optimize_onnx_graph
        return {"create_ort_session": create_ort_session, "optimize_onnx_graph": optimize_onnx_graph}[name]
    if name in {"SwapEngine", "SwapResult", "SwapAlignedResult"}:
        from .swap_engine import SwapEngine, SwapResult, SwapAlignedResult
        return {"SwapEngine": SwapEngine, "SwapResult": SwapResult, "SwapAlignedResult": SwapAlignedResult}[name]
    if name in {"SwapBackend", "BackendInfo", "get_backend", "list_backends"}:
        from .swap_backends import SwapBackend, BackendInfo, get_backend, list_backends
        return {
            "SwapBackend":   SwapBackend,
            "BackendInfo":   BackendInfo,
            "get_backend":   get_backend,
            "list_backends": list_backends,
        }[name]
    if name in {"FrameMetrics", "FrameMetricsCollector"}:
        from .frame_metrics import FrameMetrics, FrameMetricsCollector
        return {"FrameMetrics": FrameMetrics, "FrameMetricsCollector": FrameMetricsCollector}[name]
    if name in {"FailureLog", "FailureCategory", "FailureSeverity"}:
        from .failure_log import FailureLog, FailureCategory, FailureSeverity
        return {
            "FailureLog": FailureLog,
            "FailureCategory": FailureCategory,
            "FailureSeverity": FailureSeverity,
        }[name]
    raise AttributeError(name)
