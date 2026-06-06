"""Thin, typed wrappers over the ffmpeg CLI. One module for ALL the
shell-outs the pipeline needs, so subprocess construction lives in
exactly one place instead of being copy-pasted into 5 callers."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional


def resolve_ffmpeg() -> str:
    """Find an ffmpeg binary, preferring imageio's bundled one."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg") or "ffmpeg"


def probe_duration_seconds(path: Path) -> float:
    """Extract duration from ffmpeg's stderr probe."""
    ff = resolve_ffmpeg()
    r = subprocess.run(
        [ff, "-i", str(path), "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="ignore",
    )
    m = re.search(r"Duration:\s+(\d+):(\d+):(\d+\.\d+)", r.stderr or "")
    if not m:
        raise RuntimeError(f"could not probe duration: {path}")
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])


def slice_audio_to_wav(audio_in: Path, start_s: float, duration_s: float,
                        out_path: Path) -> Path:
    """Cut [start, start+duration] from audio_in into a 44.1k stereo
    PCM WAV. Re-encodes so MP3/OGG/WAV inputs all become predictable."""
    ff = resolve_ffmpeg()
    r = subprocess.run([
        ff, "-y", "-loglevel", "error",
        "-ss", f"{start_s:.3f}", "-t", f"{duration_s:.3f}",
        "-i", str(audio_in),
        "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le",
        str(out_path),
    ], capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"audio slice failed: {(r.stderr or '')[-400:]}")
    return out_path


def loop_video_to_duration(video_in: Path, target_duration_s: float,
                            out_path: Path) -> Path:
    """stream_loop the input until target_duration_s, re-encode h.264."""
    ff = resolve_ffmpeg()
    r = subprocess.run([
        ff, "-y", "-loglevel", "error",
        "-stream_loop", "-1",
        "-i", str(video_in),
        "-t", f"{target_duration_s:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an",
        str(out_path),
    ], capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"video loop failed: {(r.stderr or '')[-400:]}")
    return out_path


def concat_videos(paths: List[Path], out_path: Path) -> Path:
    """Concat demuxer + re-encode so container/codec mismatch is OK."""
    ff = resolve_ffmpeg()
    list_path = out_path.with_suffix(".concat.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in paths:
            ap = str(Path(p).resolve()).replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{ap}'\n")
    try:
        r = subprocess.run([
            ff, "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            str(out_path),
        ], capture_output=True, text=True, encoding="utf-8", errors="ignore")
    finally:
        try: list_path.unlink()
        except OSError: pass
    if r.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"concat failed: {(r.stderr or '')[-400:]}")
    return out_path


def replace_audio_with_stems_mix(video_in: Path, vocals_wav: Path,
                                  no_vocals_wav: Path,
                                  out_path: Path) -> Path:
    """Mux a SUMMED Demucs-stems mix onto the lipsync video.

    Why this instead of replace_audio_track(original_song):
        The lipsync was timed against vocals.wav (Demucs WAV, sample-0
        aligned). The original upload (often MP3) has encoder lead-in
        padding (LAME inserts ~1152 samples ~= 26 ms of silence per the
        format spec). Muxing the original MP3 back onto a vocals-timed
        video shifts the audio behind the video by that padding, which
        reads as "lipsync doesn't match" -- the mouth opens ~3 frames
        before the vocal you hear.
        vocals + no_vocals share the SAME time reference as the lipsync
        conditioning (both are Demucs WAV outputs from the same run),
        so summing them and muxing the sum preserves perfect sync.
        Loudness: Demucs stems are normalized such that the sum has
        the same envelope as the original. amix with normalize=0
        keeps the unscaled sum.
    """
    ff = resolve_ffmpeg()
    r = subprocess.run([
        ff, "-y", "-loglevel", "error",
        "-i", str(video_in),
        "-i", str(vocals_wav),
        "-i", str(no_vocals_wav),
        "-filter_complex",
        "[1:a][2:a]amix=inputs=2:duration=longest:normalize=0[a]",
        "-map", "0:v:0", "-map", "[a]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(out_path),
    ], capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"stems-mix remux failed: {(r.stderr or '')[-400:]}")
    return out_path


def replace_audio_track(video_in: Path, audio_in: Path,
                         out_path: Path) -> Path:
    """Take the video stream from video_in, replace its audio with
    audio_in's audio stream. Re-encodes audio to AAC for browser
    playback. Used to put the full song (with instruments) back onto
    a lipsync render that was driven by isolated vocals only.

    NOTE: this introduces sample-level desync if audio_in's time
    reference doesn't match the video's frame timing (e.g. MP3 lead-in
    padding). Prefer replace_audio_with_stems_mix() when Demucs stems
    are available.
    """
    ff = resolve_ffmpeg()
    r = subprocess.run([
        ff, "-y", "-loglevel", "error",
        "-i", str(video_in), "-i", str(audio_in),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(out_path),
    ], capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"audio remux failed: {(r.stderr or '')[-400:]}")
    return out_path
