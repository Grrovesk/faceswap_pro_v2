"""
GPU DLL bootstrap.

Import this BEFORE onnxruntime (or anything that imports onnxruntime,
which includes torch when CUDA is involved). It adds the cuDNN 9 +
CUDA 12 bin directories to Windows's DLL search path so that
`onnxruntime_providers_cuda.dll` can find its dependencies and ORT
can actually use the CUDA execution provider.

Without this, ORT silently falls back to CPU and inswapper_128 runs
in ~275 ms warm median instead of ~13 ms on an RTX A6000.

Verified 2026-05-17: warm-median dropped from 0.275 s -> 0.013 s
on this exact box with this exact bootstrap applied.

This module is idempotent: importing twice is harmless.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Order matters: the dirs are searched in the order they're added, and
# we want cuDNN 9.8 + CUDA 12.8 to win over PyTorch's bundled cuDNN
# (which is also cuDNN 9 but a different patch and would conflict).
#
# The third entry (PyTorch's bundled libs) used to be hardcoded to a
# specific user's venv. We resolve it from sys.prefix at module-import
# time -- BEFORE any `import torch` happens elsewhere. This is critical:
# importing torch here would defeat the purpose of this whole file,
# which is to set up CUDA/cuDNN DLL paths BEFORE torch loads.
def _resolve_torch_lib_dir() -> str:
    """Find the installed torch's lib directory from sys.prefix only.
    No `import torch` -- that would force torch to load BEFORE the
    CUDA/cuDNN paths below are on PATH, which is the entire bug this
    file exists to prevent."""
    candidates = [
        # Windows venv layout
        Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib",
        # Linux/macOS venv layout
        Path(sys.prefix) / "lib" / "python3.10" / "site-packages"
            / "torch" / "lib",
        Path(sys.prefix) / "lib" / "python3.11" / "site-packages"
            / "torch" / "lib",
    ]
    for c in candidates:
        try:
            if c.is_dir():
                return str(c)
        except OSError:
            continue
    return ""


_TORCH_LIB_DIR = _resolve_torch_lib_dir()

_CANDIDATE_DIRS: Tuple[str, ...] = (
    r"C:\Program Files\NVIDIA\CUDNN\v9.8\bin\12.8",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
)
# Append PyTorch's bundled libs LAST (if we found them). System cuDNN
# and CUDA above are searched first, so they win for the executors
# that actually do inference (ORT, etc.), but having torch's dir on
# the path means torch's own DLL loads also resolve cleanly.
if _TORCH_LIB_DIR:
    _CANDIDATE_DIRS = _CANDIDATE_DIRS + (_TORCH_LIB_DIR,)

_added_dirs: List[str] = []
_done = False


def add_gpu_dll_dirs() -> List[str]:
    """Add cuDNN/CUDA bin dirs to Windows DLL search path.

    Uses BOTH `os.add_dll_directory()` (for Python's own LoadLibrary
    calls) AND prepends them to `os.environ["PATH"]` (for transitive
    dependency loads: when ORT's CUDA EP DLL loads cuDNN, cuDNN loads
    cuBLAS, etc., the OS uses the process PATH not Python's per-dir
    list). Without the PATH prepend you still get error 126 from
    LoadLibrary deep inside the dependency chain.

    Returns the list of directories actually added. Safe to call
    multiple times.
    """
    global _done
    if _done:
        return list(_added_dirs)
    if sys.platform != "win32":
        _done = True
        return []

    # Always prepend to PATH first; this is the reliable mechanism for
    # transitive DLL loads on Windows. We do it before add_dll_directory
    # so that if add_dll_directory isn't available we still got coverage.
    path_parts: List[str] = []
    existing_path = os.environ.get("PATH", "")
    for d in _CANDIDATE_DIRS:
        if not Path(d).is_dir():
            logger.debug("GPU DLL dir not present, skipping: %s", d)
            continue
        path_parts.append(d)
        _added_dirs.append(d)
    if path_parts:
        os.environ["PATH"] = os.pathsep.join(path_parts) + os.pathsep + existing_path
        logger.info("Prepended to PATH: %s", path_parts)

    if hasattr(os, "add_dll_directory"):
        for d in _added_dirs:
            try:
                os.add_dll_directory(d)
                logger.info("os.add_dll_directory: %s", d)
            except OSError as e:
                logger.warning("Failed to add GPU DLL dir %s: %s", d, e)
    else:
        logger.warning(
            "os.add_dll_directory unavailable (Python < 3.8); relying on PATH only."
        )

    _done = True
    return list(_added_dirs)


def preload_cuda_cudnn_dlls() -> List[str]:
    """Pre-load PyTorch's bundled CUDA runtime + cuDNN DLLs via
    ``ctypes.WinDLL`` with ABSOLUTE paths.

    The PATH / add_dll_directory approach above tells Windows where to
    SEARCH for DLLs.  But on some configs (different CUDA toolkit version
    on system vs the one PyTorch was built against), the search still
    fails on transitive dependencies of cudnn_cnn64_9.dll like
    cudart64_12.dll or one of the cudnn_engines_* helpers.

    The fix: load each DLL eagerly with an absolute path, in dependency
    order.  After the load succeeds, the DLL is in the process loaded-
    module table and any subsequent ``LoadLibrary`` from GFPGAN /
    PyTorch / etc. returns the already-loaded handle without searching.

    Idempotent: re-calling is harmless.  Failures are tolerated -- if a
    file doesn't exist we just skip it.  Returns the list of DLLs that
    actually loaded.
    """
    if sys.platform != "win32" or not _TORCH_LIB_DIR:
        return []

    import ctypes

    # Order matters.  PyTorch bundles all of these in torch/lib.
    # cuDNN's cudnn_cnn64_9.dll depends on the others -- load them
    # FIRST so when cuDNN's loader probes for its dependencies they
    # are already resolved.
    dll_order = (
        # CUDA runtime + math libs (cuDNN depends on these).
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cufft64_11.dll",
        "curand64_10.dll",
        "cusolver64_11.dll",
        "cusparse64_12.dll",
        "nvrtc64_120_0.dll",
        "nvrtc-builtins64_124.dll",
        # cuDNN graph layer (loaded by all cuDNN sub-libs).
        "cudnn64_9.dll",
        "cudnn_graph64_9.dll",
        # cuDNN engine + heuristics (cnn depends on these).
        "cudnn_engines_precompiled64_9.dll",
        "cudnn_engines_runtime_compiled64_9.dll",
        "cudnn_heuristic64_9.dll",
        # cuDNN convolution + ops (these are what GFPGAN ultimately needs).
        "cudnn_cnn64_9.dll",
        "cudnn_ops64_9.dll",
        "cudnn_adv64_9.dll",
    )
    loaded: List[str] = []
    skipped: List[str] = []
    for name in dll_order:
        full = os.path.join(_TORCH_LIB_DIR, name)
        if not os.path.isfile(full):
            skipped.append(name)
            continue
        try:
            ctypes.WinDLL(full)
            loaded.append(name)
        except OSError as exc:
            logger.warning("preload failed for %s: %s", name, exc)
    if loaded:
        logger.info("Preloaded %d CUDA/cuDNN DLLs from %s: %s",
                    len(loaded), _TORCH_LIB_DIR, loaded)
    if skipped:
        logger.debug("Skipped (not present in torch/lib): %s", skipped)
    return loaded


# Auto-apply at import. Callers that prefer explicit control can still
# call add_gpu_dll_dirs() and preload_cuda_cudnn_dlls() themselves.
add_gpu_dll_dirs()
preload_cuda_cudnn_dlls()
