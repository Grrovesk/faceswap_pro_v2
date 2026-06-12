"""Entry point. Run: `python launch.py` from v2/.

v2.5 (standalone -- no v1 imports): builds FastAPI app, registers
v2's own webcam stream routes (/webcam_stream/*), mounts Gradio at
'/'. The webcam subsystem lives entirely in v2/faceswap/webcam/.
ZERO imports from ui.app or ui.stream_server -- v2 is its own world.
"""
import sys
import time
import tempfile
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# v2 is STANDALONE: core/ and utils/ live INSIDE v2/. We do NOT add
# v2's parent folder to sys.path. The hard contract is no Python
# imports reach outside v2/. External repos (LatentSync, RVC)
# are SUBPROCESS-launched via configurable paths, not imported.

# CRITICAL: add cuDNN + CUDA DLL dirs to the Windows DLL search path
# BEFORE anything else loads torch.  Without this, GFPGAN init fails
# with WinError 127 ("specified procedure could not be found")
# because PyTorch's bundled cudnn_cnn64_9.dll can't resolve its CUDA
# runtime dependencies.  The module side-effect on import calls
# add_gpu_dll_dirs() once; idempotent if other code imports it later.
import core.gpu_dll_bootstrap  # noqa: F401 -- side-effects only

from faceswap.ui import build
from faceswap.paths import PROJECT_ROOT


def _purge_stale_gradio_cache(max_age_hours: float = 1.0) -> None:
    g = Path(tempfile.gettempdir()) / "gradio"
    if not g.exists():
        return
    cutoff = time.time() - (float(max_age_hours) * 3600)
    removed = 0
    for f in g.rglob("*"):
        if f.is_file():
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink(); removed += 1
            except OSError:
                pass
    for d in sorted([p for p in g.rglob("*") if p.is_dir()],
                    key=lambda p: -len(str(p))):
        try: d.rmdir()
        except OSError: pass
    if removed:
        print(f"[v2 startup] purged {removed} stale Gradio cache files "
              f"(>{max_age_hours:g}h old) from {g}", flush=True)


def _build_fastapi_app(blocks):
    """Build the FastAPI app v2-only. Uses v2's own webcam streaming
    routes, no imports from ui.app or ui.stream_server."""
    import gradio as gr
    from fastapi import FastAPI
    from faceswap.webcam.streaming import register_stream_routes

    fast_app = FastAPI(title="faceswap_pro v2")
    stream_worker = register_stream_routes(fast_app)

    @fast_app.on_event("shutdown")
    def _stop_stream_worker():
        try:
            if stream_worker is not None:
                stream_worker.stop()
        except Exception:
            pass

    gr.mount_gradio_app(
        fast_app, blocks, path="/",
        allowed_paths=[str(PROJECT_ROOT)],
    )
    return fast_app


def _run_prechecks() -> None:
    """Fail-fast guards added 2026-06-11 to catch the Edit-tool
    truncation / null-byte / dropped-symbol patterns that bit Restyle,
    Creature Swap, and faceswap/ui.py during that day's session.

    Two checks:
      * tools/check_tail_integrity.py -- null bytes + AST parse over
        every .py file in the production tree.
      * tools/check_public_symbols.py -- regression check against the
        captured public-symbol baseline.

    Either failing aborts launch with a clear error message.  Set
    env var FACESWAP_SKIP_PRECHECK=1 to bypass (don't check that env
    var into version control -- it should be ephemeral).
    """
    import os
    import subprocess
    if os.environ.get("FACESWAP_SKIP_PRECHECK"):
        print("[v2 startup] FACESWAP_SKIP_PRECHECK set -- skipping "
              "prechecks", flush=True)
        return
    tools_dir = Path(__file__).resolve().parent / "tools"
    for script in ("check_tail_integrity.py", "check_public_symbols.py"):
        p = tools_dir / script
        if not p.is_file():
            print(f"[v2 startup] WARN: precheck {script} not found at "
                  f"{p} -- skipping", flush=True)
            continue
        try:
            r = subprocess.run([sys.executable, str(p)],
                                  capture_output=True, text=True,
                                  timeout=30)
        except Exception as exc:
            print(f"[v2 startup] precheck {script} failed to run: "
                  f"{exc}", flush=True)
            continue
        if r.returncode != 0:
            print("=" * 64, flush=True)
            print(f"[v2 startup] PRECHECK FAILED: {script}", flush=True)
            if r.stdout:
                print(r.stdout, flush=True)
            if r.stderr:
                print(r.stderr, flush=True)
            print("=" * 64, flush=True)
            print(f"Refusing to start.  Fix the issue above, or "
                  f"override with FACESWAP_SKIP_PRECHECK=1", flush=True)
            sys.exit(1)
        else:
            if r.stdout:
                print(f"[v2 startup] {script}: "
                      f"{r.stdout.strip().splitlines()[0]}", flush=True)


def main():
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    _purge_stale_gradio_cache(max_age_hours=1.0)
    _run_prechecks()

    blocks = build()
    fast_app = _build_fastapi_app(blocks)

    import uvicorn
    uvicorn.run(fast_app, host="0.0.0.0", port=7861)


if __name__ == "__main__":
    main()
