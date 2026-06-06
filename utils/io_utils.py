"""Video I/O utilities backed by OpenCV with optional ffmpeg finalization."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader:
    """Read video frames via OpenCV VideoCapture with batch read support.

    Supports standard video containers (mp4, avi, mkv, mov). For formats that
    OpenCV struggles with, an ffmpeg-pipe fallback is available.

    Args:
        path: Path to the input video file.
        use_ffmpeg: If True, read via ffmpeg subprocess pipe instead of
            OpenCV. Useful for codecs that OpenCV cannot decode.
    """

    def __init__(self, path: str, use_ffmpeg: bool = False) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        self.path = path
        self.use_ffmpeg = use_ffmpeg
        self._ffmpeg_process: Optional[subprocess.Popen] = None
        self._cap: Optional[cv2.VideoCapture] = None

        # Probe metadata with ffprobe when available, otherwise fall back to OpenCV.
        self.fps: float = self._probe_fps()
        self.width: int = self._probe_width()
        self.height: int = self._probe_height()
        self.total_frames: int = self._probe_frame_count()
        self.duration: float = self.total_frames / max(self.fps, 1e-6)

        if use_ffmpeg:
            pass
        else:
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                raise IOError(f"Cannot open video: {path}")

    # ------------------------------------------------------------------
    # Metadata probing via ffprobe
    # ------------------------------------------------------------------

    def _run_ffprobe(self, *args: str) -> str:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "csv=p=0",
            *args,
            self.path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except (FileNotFoundError, PermissionError, OSError):
            return ""
        return result.stdout.strip()

    def _fallback_capture_value(self, prop: int, default: float) -> float:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            return default
        try:
            value = cap.get(prop)
            if value and value > 0:
                return value
            return default
        finally:
            cap.release()

    def _probe_fps(self) -> float:
        out = self._run_ffprobe(
            "-show_entries", "stream=r_frame_rate",
            "-select_streams", "v:0",
        )
        if "/" in out:
            num, den = out.split("/")
            return float(num) / max(float(den), 1e-6)
        try:
            return float(out)
        except (ValueError, TypeError):
            return float(self._fallback_capture_value(cv2.CAP_PROP_FPS, 30.0))

    def _probe_width(self) -> int:
        out = self._run_ffprobe(
            "-show_entries", "stream=width",
            "-select_streams", "v:0",
        )
        try:
            return int(out)
        except (ValueError, TypeError):
            return int(self._fallback_capture_value(cv2.CAP_PROP_FRAME_WIDTH, 1920))

    def _probe_height(self) -> int:
        out = self._run_ffprobe(
            "-show_entries", "stream=height",
            "-select_streams", "v:0",
        )
        try:
            return int(out)
        except (ValueError, TypeError):
            return int(self._fallback_capture_value(cv2.CAP_PROP_FRAME_HEIGHT, 1080))

    def _probe_frame_count(self) -> int:
        out = self._run_ffprobe(
            "-show_entries", "stream=nb_frames",
            "-select_streams", "v:0",
        )
        try:
            return int(out)
        except (ValueError, TypeError):
            cap_frames = int(self._fallback_capture_value(cv2.CAP_PROP_FRAME_COUNT, 0))
            if cap_frames > 0:
                return cap_frames
            # Some containers don't report nb_frames; estimate from duration
            duration_out = self._run_ffprobe(
                "-show_entries", "format=duration",
            )
            try:
                return int(float(duration_out) * self.fps)
            except (ValueError, TypeError):
                return 0

    # ------------------------------------------------------------------
    # Frame reading — OpenCV path
    # ------------------------------------------------------------------

    def read_frame(self) -> Optional[Tuple[int, np.ndarray]]:
        """Read a single frame. Returns (frame_idx, bgr_array) or None at EOF."""
        if self._cap is None:
            raise RuntimeError("VideoReader not initialized")
        ok, frame = self._cap.read()
        if not ok:
            return None
        idx = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        return idx, frame

    def read_batch(self, n: int) -> List[Tuple[int, np.ndarray]]:
        """Read up to *n* frames. Returns list of (frame_idx, bgr_array)."""
        frames: List[Tuple[int, np.ndarray]] = []
        for _ in range(n):
            result = self.read_frame()
            if result is None:
                break
            frames.append(result)
        return frames

    def iter_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yield (frame_idx, bgr_array) for every frame in the video."""
        while True:
            result = self.read_frame()
            if result is None:
                break
            yield result

    def read_batches(
        self, batch_size: int
    ) -> Generator[List[Tuple[int, np.ndarray]], None, None]:
        """Yield successive frame batches for pipeline compatibility."""
        while True:
            batch = self.read_batch(batch_size)
            if not batch:
                break
            yield batch

    # ------------------------------------------------------------------
    # Frame reading — ffmpeg pipe path
    # ------------------------------------------------------------------

    def _start_ffmpeg(self) -> subprocess.Popen:
        cmd = [
            "ffmpeg",
            "-i", self.path,
            "-loglevel", "quiet",
            "-f", "image2pipe",
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo",
            "-",
        ]
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def read_frame_ffmpeg(self) -> Optional[np.ndarray]:
        """Read one frame via ffmpeg pipe. Returns bgr_array or None at EOF."""
        if self._ffmpeg_process is None:
            self._ffmpeg_process = self._start_ffmpeg()
        frame_size = self.width * self.height * 3
        raw = self._ffmpeg_process.stdout.read(frame_size)
        if len(raw) != frame_size:
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.height, self.width, 3)
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._ffmpeg_process is not None:
            self._ffmpeg_process.terminate()
            self._ffmpeg_process.wait(timeout=5)
            self._ffmpeg_process = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @property
    def total(self) -> int:
        """Compatibility alias for older pipeline code."""
        return self.total_frames


class VideoWriter:
    """Write video frames to disk using OpenCV VideoWriter.

    Supports mp4 (H.264) and other common containers. Automatically selects
    the best available codec.

    Args:
        path: Output video file path.
        fps: Frames per second.
        size: (width, height) tuple.
        codec: FourCC codec string. Default 'mp4v'. Use 'avc1' or 'H264' if
            available for better compression.
    """

    def __init__(
        self,
        path: str,
        fps: float,
        size: Tuple[int, int],
        codec: str = "mp4v",
    ) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.fps = fps
        self.size = size
        self.requested_codec = (codec or "mp4v").strip() or "mp4v"
        self.codec = self._open_writer()
        self._frame_count = 0

    def write(self, frame: np.ndarray) -> None:
        """Write a single BGR frame."""
        self.writer.write(frame)
        self._frame_count += 1

    def write_batch(self, frames: List[np.ndarray]) -> None:
        """Write multiple frames."""
        for frame in frames:
            self.write(frame)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _open_writer(self) -> str:
        candidates = self._codec_candidates()
        errors: List[str] = []

        for candidate in candidates:
            fourcc = cv2.VideoWriter_fourcc(*candidate)
            writer = cv2.VideoWriter(self.path, fourcc, self.fps, self.size)
            if writer.isOpened():
                self.writer = writer
                logger.info(
                    "VideoWriter opened path=%s codec=%s fps=%.3f size=%sx%s",
                    self.path,
                    candidate,
                    self.fps,
                    self.size[0],
                    self.size[1],
                )
                if candidate != self.requested_codec:
                    logger.warning(
                        "Requested codec %s unavailable for %s; using %s instead.",
                        self.requested_codec,
                        self.path,
                        candidate,
                    )
                return candidate
            writer.release()
            errors.append(candidate)

        raise IOError(
            f"Cannot open VideoWriter for: {self.path} (tried codecs: {', '.join(errors)})"
        )

    def _codec_candidates(self) -> List[str]:
        ext = os.path.splitext(self.path)[1].lower()
        if ext in {".avi"}:
            defaults = ["FFV1", "HFYU", "MJPG", "XVID", "mp4v"]
        elif ext in {".mkv"}:
            defaults = ["FFV1", "HFYU", "MJPG", "XVID", "mp4v"]
        else:
            # Keep mp4v as the tested MP4-safe baseline on this machine.
            defaults = ["mp4v", "XVID", "MJPG", "avc1"]

        ordered = [self.requested_codec, *defaults]
        deduped: List[str] = []
        for candidate in ordered:
            normalized = (candidate or "").strip()
            if len(normalized) != 4 or normalized in deduped:
                continue
            deduped.append(normalized)
        return deduped

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def _resolve_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def finalize_video_h264(
    intermediate_path: str,
    source_video_path: str,
    output_path: str,
    *,
    crf: int = 18,
    preset: str = "slow",
    audio_bitrate: str = "192k",
    keep_audio: bool = True,
    lossless_rgb: bool = False,
) -> str:
    """Encode a final MP4 from an intermediate render using ffmpeg.

    The pipeline renders a lossless FFV1 intermediate; this re-encodes it
    to a distributable MP4. Modes (driven by core.pipeline's output cfg):
      * lossless_rgb=True          -> libx264rgb CRF 0, mathematically
                                      lossless RGB. Huge files (120+
                                      Mbit/s); only when every pixel must
                                      be bit-exact.
      * lossless_rgb=False, crf~17 -> libx264 yuv420p, visually lossless:
                                      no difference to the eye, ~10-15x
                                      smaller. The default.
      * lossless_rgb=False, crf~21 -> libx264 yuv420p, balanced/smaller.
    Audio is transcoded to AAC at `audio_bitrate` so the MP4 is portable
    (a Vorbis track copied from a .webm source is not valid MP4 audio).
    """
    ffmpeg_exe = _resolve_ffmpeg_exe()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        intermediate_path,
    ]
    if keep_audio:
        cmd.extend(["-i", source_video_path])

    cmd.extend(
        [
            "-map",
            "0:v:0",
            "-movflags",
            "+faststart",
        ]
    )

    if lossless_rgb:
        cmd.extend(
            [
                "-c:v",
                "libx264rgb",
                "-preset",
                str(preset),
                "-crf",
                "0",
            ]
        )
    else:
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                str(preset),
                "-crf",
                str(int(crf)),
                "-pix_fmt",
                "yuv420p",
            ]
        )

    if keep_audio:
        cmd.extend(
            [
                "-map",
                "1:a:0?",
                "-c:a",
                "aac",
                "-b:a",
                str(audio_bitrate),
                "-shortest",
            ]
        )
    else:
        cmd.append("-an")

    cmd.append(output_path)

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg final encode failed:\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return output_path


class AudioExtractor:
    """Extract audio track from a video file using ffmpeg.

    Args:
        video_path: Path to the source video.
        output_path: Path for the extracted WAV audio.
        sample_rate: Target sample rate (default 16000 for wav2vec2).
    """

    def __init__(
        self,
        video_path: str,
        output_path: str,
        sample_rate: int = 16000,
    ) -> None:
        self.video_path = video_path
        self.output_path = output_path
        self.sample_rate = sample_rate

    def extract(self) -> str:
        """Run ffmpeg to extract audio as mono WAV. Returns output path."""
        cmd = [
            "ffmpeg",
            "-i", self.video_path,
            "-vn",                    # no video
            "-ac", "1",               # mono
            "-ar", str(self.sample_rate),
            "-acodec", "pcm_s16le",   # 16-bit PCM WAV
            "-y",                     # overwrite
            "-loglevel", "quiet",
            self.output_path,
        ]
        subprocess.run(cmd, check=True)
        return self.output_path


def extract_audio(
    video_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 16000,
) -> str:
    """Extract audio and return the generated WAV path."""
    if output_path is None:
        root, _ = os.path.splitext(video_path)
        output_path = f"{root}_audio.wav"
    return AudioExtractor(video_path, output_path, sample_rate=sample_rate).extract()
