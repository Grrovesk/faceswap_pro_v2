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


def main():
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    _purge_stale_gradio_cache(max_age_hours=1.0)

    blocks = build()
    fast_app = _build_fastapi_app(blocks)

    import uvicorn
    uvicorn.run(fast_app, host="0.0.0.0", port=7861)


if __name__ == "__main__":
    main()
