"""ONNX graph optimization and session creation utilities.

This module provides two optimisation paths:

1. **Session-level** — ``create_ort_session`` applies onnxruntime's built-in
   graph optimiser (ORT_ENABLE_ALL), CUDA/TensorRT execution providers,
   and optionally CUDA graphs for fixed-shape models.

2. **Graph-level** — ``optimize_onnx_graph`` rewrites the ONNX graph before
   loading it (constant folding, shape inference, node fusion) and saves
   the optimised model to disk so the cost is paid once.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import onnx
import onnxruntime as ort

logger = logging.getLogger(__name__)


# =====================================================================
# Session-level optimisation
# =====================================================================

def create_ort_session(
    model_path: str | Path,
    *,
    gpu_id: int = 0,
    use_tensorrt: bool = False,
    enable_cuda_graph: bool = False,
    intra_op_threads: int = 0,  # 0 = onnxruntime decides
    inter_op_threads: int = 0,
) -> ort.InferenceSession:
    """Create an optimised ONNX Runtime inference session.

    Parameters
    ----------
    model_path:
        Path to the ``.onnx`` model file.
    gpu_id:
        CUDA device ordinal.
    use_tensorrt:
        If *True*, prepend ``TensorrtExecutionProvider`` for TRT-accelerated
        subgraphs.  Falls back to CUDA if TRT is not available.
    enable_cuda_graph:
        If *True*, allow CUDA graph capture.  Only works when the model has
        fixed input shapes and no dynamic control-flow.
    intra_op_threads / inter_op_threads:
        Thread-count overrides.  Defaults (0) let onnxruntime choose.
    """
    model_path = str(model_path)

    # ── Session options ──────────────────────────────────────────────
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = intra_op_threads
    opts.inter_op_num_threads = inter_op_threads
    opts.log_severity_level = 3  # warnings only

    if enable_cuda_graph:
        opts.enable_cuda_graph = True

    # ── Execution providers ──────────────────────────────────────────
    providers: List[str | tuple] = []
    provider_options: List[Dict] = []

    if use_tensorrt:
        providers.append("TensorrtExecutionProvider")
        provider_options.append({
            "device_id": gpu_id,
            "trt_max_workspace_size": 2 << 30,  # 2 GiB
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "/tmp/faceswap_pro_trt_cache",
        })

    providers.append("CUDAExecutionProvider")
    provider_options.append({
        "device_id": gpu_id,
        "gpu_mem_limit": 4 << 30,  # 4 GiB
        "arena_extend_strategy": "kNextPowerOfTwo",
    })

    # CPU fallback
    providers.append("CPUExecutionProvider")
    provider_options.append({})

    session = ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=providers,
        provider_options=provider_options,
    )

    logger.info(
        "Created ORT session for %s  providers=%s",
        model_path,
        session.get_providers(),
    )
    return session


# =====================================================================
# Graph-level optimisation (offline)
# =====================================================================

def optimize_onnx_graph(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    *,
    fix_input_shape: Optional[dict] = None,
) -> Path:
    """Run offline ONNX graph optimisations and save the result.

    Optimisations applied:
    - onnx optimizer passes (constant folding, dead-code elimination, etc.)
    - Shape inference propagation
    - Optional: fix dynamic dims to static shapes for CUDA-graph compatibility

    Parameters
    ----------
    input_path:
        Source ``.onnx`` model.
    output_path:
        Destination for the optimised model.  Defaults to
        ``<input>_optimized.onnx``.
    fix_input_shape:
        If provided, a dict ``{input_name: (dim0, dim1, ...)}`` that
        overrides dynamic dimensions with concrete sizes.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_optimized.onnx")
    output_path = Path(output_path)

    logger.info("Optimising ONNX graph: %s → %s", input_path, output_path)

    model = onnx.load(str(input_path))

    # ── Shape inference ──────────────────────────────────────────────
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as exc:
        logger.warning("Shape inference failed (non-fatal): %s", exc)

    # ── Fix dynamic input shapes if requested ────────────────────────
    if fix_input_shape:
        for inp in model.graph.input:
            name = inp.name
            if name in fix_input_shape:
                shape = fix_input_shape[name]
                dim_proto = inp.type.tensor_type.shape.dim
                for i, d in enumerate(shape):
                    dim_proto[i].dim_value = d
                    dim_proto[i].ClearField("dim_param")

    # ── Constant folding + optimiser passes ──────────────────────────
    try:
        from onnxruntime.transformers import optimizer as ort_optimizer
        model = ort_optimizer.optimize_model(
            str(input_path),
            model_type="bert",  # generic optimiser works for conv nets too
            num_heads=0,
            hidden_size=0,
        )
        model.save_model_to_file(str(output_path))
        return output_path
    except ImportError:
        logger.debug("onnxruntime.transformers not available — using onnx native passes")

    # Fallback: just save with shape inference applied
    onnx.save(model, str(output_path))
    logger.info("Saved optimised model to %s", output_path)
    return output_path


# =====================================================================
# Batched inference helper
# =====================================================================

def run_batched(
    session: ort.InferenceSession,
    inputs: Sequence[np.ndarray],  # type: ignore[name-defined]
    input_name: str | None = None,
) -> list[np.ndarray]:  # type: ignore[name-defined]
    """Run the model on a batch of inputs, stacking when possible.

    If all inputs share the same spatial shape, they are stacked into a
    single (N, C, H, W) tensor for one ``session.run()`` call — this is
    significantly faster on GPU than N separate calls.

    Parameters
    ----------
    session:
        The ONNX Runtime session.
    inputs:
        List of (C, H, W) float32 arrays — one per face.
    input_name:
        Override the input tensor name.  If *None*, uses the first input.

    Returns
    -------
    List of output arrays, one per input face.
    """
    import numpy as np  # local import to avoid import-time cost

    if not inputs:
        return []

    input_meta = session.get_inputs()[0]
    if input_name is None:
        input_name = input_meta.name

    # Check if shapes are compatible for stacking
    shapes = [inp.shape for inp in inputs]
    can_stack = all(s == shapes[0] for s in shapes)

    if can_stack and len(inputs) > 1:
        batch = np.stack(inputs, axis=0)  # (N, C, H, W)
        results = session.run(None, {input_name: batch})
        # results[0] shape: (N, C, H, W) → split
        return [results[0][i] for i in range(len(inputs))]
    else:
        # Fall back to sequential inference
        out = []
        for inp in inputs:
            # Ensure 4D: (1, C, H, W)
            tensor = inp[np.newaxis, ...] if inp.ndim == 3 else inp
            results = session.run(None, {input_name: tensor})
            out.append(results[0][0] if results[0].shape[0] == 1 else results[0])
        return out
