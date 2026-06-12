"""Transcode uploaded videos to browser-playable H.264/AAC + faststart
MP4 for Gradio's HTML5 video widget.

Many user clips arrive in containers/codecs Gradio's <video> tag can't
decode (H.265, ProRes, MOV with weird audio, etc.).  The widget
silently overlays "Error" and the user has no idea why.  Cache the
transcode by content hash so re-uploads / .change re-fires are O(1).
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREVIEW_CACHE_DIR = PROJECT_ROOT / "models" / "rotoscope" / "_preview_cache"


def _content_hash(path: Path) -> str:
    """Cheap content hash: size + 4 MB head + 4 MB tail.  Same scheme
    as rotoscope_cache.compute_video_hash -- both use it independently
    here to avoid an import cycle."""
    sz = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(sz).encode("utf-8"))
    head_n = min(4 * 1024 * 1024, sz)
    tail_n = min(4 * 1024 * 1024, max(0, sz - head_n))
    with open(path, "rb") as f:
        h.update(f.read(head_n))
        if tail_n:
            f.seek(-tail_n, 2)
            h.update(f.read(tail_n))
    return h.hexdigest()[:16]


def _ffprobe(path: Path) -> Optional[dict]:
    """Return ffprobe JSON for the file, or None on failure."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", str(path)],
            stderr=subprocess.DEVNULL, timeout=10)
        return json.loads(out)
    except Exception as exc:
        logger.debug("ffprobe failed on %s: %s", path, exc)
        return None


def _is_already_browser_safe(probe: dict) -> bool:
    """Return True if the container+codecs look like Gradio can play
    them natively in the browser.  Conservative -- we'd rather
    transcode unnecessarily than ship "Error" overlays.
    """
    if not probe:
        return False
    fmt = (probe.get("format", {}).get("format_name") or "").lower()
    # Container must be MP4-family.
    if not any(x in fmt for x in ("mp4", "mov", "m4v", "isom")):
        return False
    streams = probe.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None or v.get("codec_name") != "h264":
        return False
    # H.264 profile 'high 10' or 'high 4:4:4 predictive' won't play in
    # most browsers; require yuv420p pixel format.
    if v.get("pix_fmt") and v["pix_fmt"] != "yuv420p":
        return False
    # Audio is optional, but if present must be AAC or empty.
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if a is not None and a.get("codec_name") not in ("aac", None):
        return False
    return True


def ensure_browser_safe(video_path: str,
                          log=logger.info) -> str:
    """Return a path to a browser-playable copy of ``video_path``.

    If ``video_path`` is already browser-safe (mp4 + h264 yuv420p +
    aac/no-audio), it's returned as-is.  Otherwise we transcode to
    PREVIEW_CACHE_DIR/<hash>.preview.mp4 and return that path.

    The cache key is a cheap content hash (size + 4 MB head + 4 MB
    tail) so renames / re-uploads of the same bytes are O(1).
    """
    p = Path(video_path)
    if not p.is_file():
        return video_path

    # Already in our cache?  Skip the probe and return as-is.
    try:
        if PREVIEW_CACHE_DIR in p.parents:
            return str(p)
    except Exception:
        pass

    probe = _ffprobe(p)
    if _is_already_browser_safe(probe or {}):
        return str(p)

    PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = _content_hash(p)
    cached = PREVIEW_CACHE_DIR / f"{h}.preview.mp4"
    if cached.is_file() and cached.stat().st_size > 0:
        log(f"[browser-safe] cache hit {p.name} -> {cached.name}")
        return str(cached)

    # Transcode.  veryfast + crf 23 is the cost/quality sweet spot for
    # preview -- this is NOT the final render.
    log(f"[browser-safe] transcoding {p.name} -> H.264/AAC/faststart "
        f"({cached.name}) ...")
    tmp = cached.with_suffix(".preview.mp4.tmp")
    cmd = [
        "ffmpeg", "-y", "-i", str(p),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(tmp),
    ]
    try:
        subprocess.check_call(cmd,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=600)
    except subprocess.CalledProcessError as exc:
        log(f"[browser-safe] transcode FAILED (rc={exc.returncode}) "
            f"-- returning original path; widget may show error overlay")
        try:
            tmp.unlink()
        except OSError:
            pass
        return str(p)
    except subprocess.TimeoutExpired:
        log("[browser-safe] transcode TIMED OUT -- returning original")
        try:
            tmp.unlink()
        except OSError:
            pass
        return str(p)
    try:
        tmp.replace(cached)
    except OSError as exc:
        log(f"[browser-safe] could not finalize cache file: {exc}")
        return str(tmp) if tmp.is_file() else str(p)
    log(f"[browser-safe] transcoded -> {cached}")
    return str(cached)
