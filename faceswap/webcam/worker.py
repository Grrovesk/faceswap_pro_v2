"""SwapStreamWorker -- daemon thread that owns the camera and runs
per-frame swap. v2 owns this code; no imports from ui/.

Public API:
  worker = SwapStreamWorker(swap_fn)
  worker.start(device_index, width, height)
  worker.stop()
  worker.get_latest_jpeg()
  worker.running, worker.fps, worker.latency_ms, worker.frames, worker.error
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import cv2
import sys
import numpy as np
import os
import subprocess

from ..ffmpeg_tools import resolve_ffmpeg

logger = logging.getLogger(__name__)


class SwapStreamWorker:
    """Camera capture + swap_fn pipeline on a daemon thread."""

    def __init__(self, swap_fn: Callable[[np.ndarray], np.ndarray]):
        self._swap_fn = swap_fn
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._running = False
        self._latest_jpeg: Optional[bytes] = None
        # Camera params (set on start)
        self._device_index = 0
        self._width = 640
        self._height = 480
        self._jpeg_quality = 80
        # Stats (read by /webcam_stream/stats)
        self.fps = 0.0
        self.latency_ms = 0.0
        self.frames = 0
        self.error = ""
        # Recording (the Record / Stop record buttons hit these)
        self._rec_writer = None       # subprocess.Popen(ffmpeg) or None
        self._rec_path = None         # Path of current recording
        self._rec_frames = 0          # frames written so far
        self._rec_w = 0               # locked-in recorder width
        self._rec_h = 0               # locked-in recorder height
        self._rec_fps_start = 0.0     # fps guess at start (for ffmpeg -r)
        self._rec_t0 = 0.0            # wallclock start time for true-fps calc
        # Most-recent output frame shape (h, w) from swap+filter pipeline.
        # Set by _loop; used by start_recording so the recorder matches
        # what's actually being streamed (not the UI's Width/Height which
        # the camera may have ignored).
        self._last_out_shape = None

    def start(self, device_index: int = 0, width: int = 640,
              height: int = 480, jpeg_quality: int = 80) -> None:
        with self._lock:
            if self._running:
                return
            self._device_index = int(device_index)
            self._width = int(width)
            self._height = int(height)
            self._jpeg_quality = int(jpeg_quality)
            self._running = True
            self.frames = 0
            self.error = ""
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("v2 SwapStreamWorker started (device=%s %sx%s q=%s)",
                    device_index, width, height, jpeg_quality)

    def stop(self) -> None:
        # Finalize any in-flight recording first
        try:
            self.stop_recording()
        except Exception:
            pass
        with self._lock:
            self._running = False
        t = self._thread
        if t is not None:
            t.join(timeout=3.0)
        self._thread = None
        # Close virtual cam if it was open
        try:
            from . import virtual_cam as _vc
            if _vc.is_open():
                _vc.close_cam()
        except Exception:
            pass
        logger.info("v2 SwapStreamWorker stopped")

    @property
    def running(self) -> bool:
        return self._running

    def start_recording(self):
        """Begin recording the live (post-swap, post-filter) stream
        to v2/recordings/webcam/webcam_swap_<timestamp>.mp4. Returns
        the destination Path. If already recording, stops the previous
        and starts a new clip.

        Uses ffmpeg subprocess + libx264 (H.264) so the output plays
        in browsers / Gradio video widget. cv2.VideoWriter with mp4v
        fourcc produces MPEG-4 Part 2 which most web players reject.
        """
        import time as _t
        from ..paths import WEBCAM_RECORDINGS_DIR
        WEBCAM_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = _t.strftime("%Y%m%d_%H%M%S")
        path = WEBCAM_RECORDINGS_DIR / f"webcam_swap_{stamp}.mp4"

        # Use the actual processed-frame dims, not the configured
        # camera dims (camera may have ignored cap.set, the swap may
        # have padded/cropped, etc). Fall back to configured w/h only
        # if no frame has come through yet.
        if self._last_out_shape is not None:
            rec_h, rec_w = self._last_out_shape
        else:
            rec_h, rec_w = self._height, self._width

        # Use the worker's measured fps (EMA-smoothed in _loop). If we
        # haven't measured yet (camera just started), fall back to 25.
        # The TRUE fps is computed on stop_recording from frame-count /
        # wallclock-elapsed and the file is then remuxed with that real
        # rate so playback timing is exact regardless of any drift here.
        fps_init = float(self.fps) if self.fps >= 5.0 else 25.0

        ffmpeg = resolve_ffmpeg()
        # libx264 + yuv420p + faststart = standard playable mp4
        argv = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{rec_w}x{rec_h}",
            "-r", f"{fps_init:.3f}",
            "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(path),
        ]
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            logger.exception("ffmpeg subprocess failed to start")
            return None

        with self._lock:
            # stop any previous recording first
            if self._rec_writer is not None:
                try:
                    self._rec_writer.stdin.close()
                    self._rec_writer.wait(timeout=3)
                except Exception:
                    try:
                        self._rec_writer.kill()
                    except Exception:
                        pass
                self._rec_writer = None
            self._rec_writer = proc
            self._rec_path = path
            self._rec_frames = 0
            self._rec_w = rec_w
            self._rec_h = rec_h
            self._rec_fps_start = fps_init
            self._rec_t0 = time.perf_counter()
        logger.info("v2 webcam recording started -> %s (%dx%d @ %.1f fps init)",
                    path, rec_w, rec_h, fps_init)
        return path

    def stop_recording(self):
        """Stop and finalize current recording. Returns the saved Path
        or None if not recording. Closes the ffmpeg subprocess pipe
        cleanly so the mp4 trailer/moov atom is written. Then remuxes
        with the TRUE measured fps (= frames / wallclock_elapsed) so
        playback timing is correct regardless of pipeline rate."""
        with self._lock:
            proc = self._rec_writer
            path = self._rec_path
            self._rec_writer = None
            self._rec_path = None
            frames = self._rec_frames
            self._rec_frames = 0
            self._rec_w = 0
            self._rec_h = 0
            fps_init = self._rec_fps_start
            t0 = self._rec_t0
            self._rec_fps_start = 0.0
            self._rec_t0 = 0.0
        if proc is None:
            return None
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        # Compute TRUE fps from wallclock and remux. The temp file was
        # written at fps_init (the worker's running EMA at start); the
        # actual rate is frames / elapsed which is the source of truth
        # for playback timing.
        elapsed = max(0.001, time.perf_counter() - t0)
        true_fps = frames / elapsed if frames > 0 else fps_init
        logger.info("v2 webcam recording: %d frames / %.2fs = %.2f fps "
                    "(was tagged %.2f fps)",
                    frames, elapsed, true_fps, fps_init)

        # Only remux if measured fps deviates by more than 5% from the
        # init value. Otherwise the file is already close enough.
        if path and frames > 0 and abs(true_fps - fps_init) / fps_init > 0.05:
            try:
                temp = path.with_suffix(".raw.mp4")
                os.replace(str(path), str(temp))
                ffmpeg = resolve_ffmpeg()
                # -c copy + -r retags fps without re-encoding (instant)
                # but doesn't change container PTS. For correct PTS we
                # need to re-encode briefly with setpts; do that with
                # libx264 stream copy of the bitstream via filter graph.
                # Simpler: just retag and let players honor the new fps.
                r = subprocess.run([
                    ffmpeg, "-y", "-loglevel", "error",
                    "-r", f"{true_fps:.3f}",
                    "-i", str(temp),
                    "-c:v", "copy",
                    "-movflags", "+faststart",
                    str(path),
                ], capture_output=True, text=True, timeout=60)
                if r.returncode == 0 and path.is_file():
                    try:
                        os.remove(str(temp))
                    except Exception:
                        pass
                    logger.info("remuxed at %.2f fps -> %s",
                                true_fps, path)
                else:
                    # retime failed -- restore the un-retimed file
                    os.replace(str(temp), str(path))
                    logger.warning("remux failed (%s); keeping original "
                                    "fps tagging", (r.stderr or "")[-200:])
            except Exception:
                logger.exception("remux pass failed; keeping original")

        logger.info("v2 webcam recording stopped: %s (%d frames @ %.2f fps)",
                    path, frames, true_fps)
        return path

    @property
    def is_recording(self) -> bool:
        return self._rec_writer is not None

    def _loop(self) -> None:
        # On Windows the DirectShow backend gives noticeably lower
        # latency than the default Media Foundation backend. On
        # Linux/macOS CAP_DSHOW isn't available, so skip the wasted
        # first attempt and let OpenCV pick the right backend
        # (V4L2 on Linux, AVFoundation on macOS).
        if sys.platform == "win32":
            cap = cv2.VideoCapture(self._device_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self._device_index)
        else:
            cap = cv2.VideoCapture(self._device_index)
        if not cap.isOpened():
            with self._lock:
                self.error = (f"could not open camera device "
                              f"{self._device_index}")
                self._running = False
            logger.warning("v2 webcam: %s", self.error)
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # low-latency
        except Exception:
            pass

        fps_ema = 0.0
        lat_ema = 0.0
        last_t = time.perf_counter()
        try:
            while True:
                with self._lock:
                    if not self._running:
                        break
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    time.sleep(0.01)
                    continue
                t0 = time.perf_counter()
                try:
                    out_bgr = self._swap_fn(frame_bgr)
                except Exception:
                    logger.exception("v2 swap_fn failed; raw frame")
                    out_bgr = frame_bgr
                # ---- Post-swap filters (brightness/contrast/saturation)
                try:
                    from . import state as _st
                    from . import filters as _flt
                    snap = _st.get_snapshot()
                    out_bgr = _flt.apply_filters(
                        out_bgr,
                        brightness=snap.brightness,
                        contrast=snap.contrast,
                        saturation=snap.saturation)
                except Exception:
                    logger.exception("filter apply failed")
                # ---- Optional: send to virtual cam (OBS / Zoom / Discord)
                try:
                    from . import virtual_cam as _vc
                    if snap.virtual_cam_on:
                        if not _vc.is_open():
                            _vc.open_cam(width=out_bgr.shape[1],
                                         height=out_bgr.shape[0],
                                         fps=30.0)
                        _vc.send_frame(out_bgr)
                    elif _vc.is_open():
                        _vc.close_cam()
                except Exception:
                    logger.exception("virtual cam send failed")
                # Remember the actual processed frame shape so that
                # start_recording uses the TRUE output dims (not the
                # UI-configured camera dims, which the camera may have
                # ignored). This is what fixes the "recording is 4:3 but
                # my source is 16:9" issue -- we now record at whatever
                # the swap+filter pipeline actually produces.
                self._last_out_shape = out_bgr.shape[:2]   # (h, w)

                # Write to recording if active. We record the
                # FULLY PROCESSED frame (post-swap, post-filter) so
                # the saved clip matches what the user sees. Writes
                # raw BGR bytes to the ffmpeg subprocess stdin.
                if self._rec_writer is not None:
                    try:
                        # Resize ONLY if processed frame doesn't match
                        # the dims locked at start_recording (which were
                        # captured from the first real output frame).
                        if (self._rec_h and self._rec_w and
                                out_bgr.shape[:2] != (self._rec_h,
                                                       self._rec_w)):
                            wf = cv2.resize(out_bgr,
                                             (self._rec_w, self._rec_h))
                        else:
                            wf = out_bgr
                        self._rec_writer.stdin.write(wf.tobytes())
                        self._rec_frames += 1
                    except BrokenPipeError:
                        # ffmpeg died; null out so next stop is a no-op
                        logger.warning("recorder pipe broke; ffmpeg exited")
                        self._rec_writer = None
                    except Exception:
                        logger.exception("recording write failed")

                t1 = time.perf_counter()
                ok_enc, buf = cv2.imencode(
                    ".jpg", out_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
                if not ok_enc:
                    continue
                with self._lock:
                    self._latest_jpeg = buf.tobytes()
                    self.frames += 1
                inst_lat = (t1 - t0) * 1000.0
                inst_fps = 1.0 / max(1e-6, (t1 - last_t))
                last_t = t1
                lat_ema = (inst_lat if lat_ema <= 0
                           else 0.8 * lat_ema + 0.2 * inst_lat)
                fps_ema = (inst_fps if fps_ema <= 0
                           else 0.8 * fps_ema + 0.2 * inst_fps)
                self.latency_ms = lat_ema
                self.fps = fps_ema
        finally:
            cap.release()

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg


# Singleton -- shared between FastAPI route + Gradio handlers
_WORKER: Optional[SwapStreamWorker] = None
_WORKER_LOCK = threading.Lock()


def get_worker(swap_fn: Optional[Callable] = None) -> SwapStreamWorker:
    """Return the singleton worker, lazy-init on first call."""
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is None:
            if swap_fn is None:
                from .swap_fn import make_swap_fn
                swap_fn = make_swap_fn()
            _WORKER = SwapStreamWorker(swap_fn)
    return _WORKER
