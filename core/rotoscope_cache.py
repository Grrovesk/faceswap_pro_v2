"""Rotoscope frame extraction + caching.

Given a target video, extract it to a PNG sequence on disk so the
Rotoscoping tab's scrubber can do instant random-access frame reads
instead of paying a cv2.VideoCapture seek per scrub event (slow,
especially on long videos).

Layout
------
Each video gets its own directory under ``CACHE_ROOT``:

    v2/models/rotoscope/<video_hash>/
        meta.json          # source path, hash, frame_count, dims, fps
        frames/
            frame_000000.png
            frame_000001.png
            ...
        masks/             # populated by SAM2 daemon's propagate() later
            obj_1/
                frame_000000.png
                ...

Cache key
---------
A truncated SHA-256 of (source path resolve + size + mtime).  This is
fast enough that even multi-GB videos hash in <1 second, and stable
across renames as long as size+mtime are unchanged.

Eviction
--------
Not implemented in Phase 1.2.  Cache grows indefinitely.  Future:
LRU prune past a total size threshold.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


CACHE_ROOT = (Path(__file__).resolve().parents[1]
              / "models" / "rotoscope")


@dataclass
class FrameCacheInfo:
    """Metadata about an extracted-frame cache directory."""
    video_path: str            # original source path
    video_hash: str            # cache key
    cache_dir: str             # absolute path to the cache dir
    frames_dir: str            # absolute path to the frames subdir
    masks_dir: str             # absolute path to the masks subdir
    frame_count: int           # int from cv2.CAP_PROP_FRAME_COUNT
    width: int
    height: int
    fps: float                 # source fps as float
    duration_s: float          # frame_count / fps (or 0.0 if no fps)


# ---------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------
def compute_video_hash(video_path: Path,
                       hash_len: int = 16) -> str:
    """Stable short hash of the video's actual byte content.

    We previously seeded with (resolved path, size, mtime), which
    looked stable but isn't on Windows: when Gradio stages an upload
    and the front-end re-touches the file (preview load, websocket
    re-fetch, etc.) the OS updates mtime.  Same file -> different
    hash -> cache miss -> ffmpeg re-extracts every interaction.  So
    we hash the file contents instead.

    For a multi-GB clip we'd kill startup if we read the whole file,
    so the hash combines (size, head 4 MB, tail 4 MB) which is
    essentially as unique as a full content hash for media files
    and is O(8 MB) regardless of clip length.
    """
    p = Path(video_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    st = p.stat()
    HEAD = TAIL = 4 * 1024 * 1024
    size = int(st.st_size)
    h = hashlib.sha256()
    h.update(str(size).encode())
    h.update(b"\0")
    try:
        with open(p, "rb") as f:
            h.update(f.read(min(HEAD, size)))
            if size > HEAD + TAIL:
                f.seek(size - TAIL)
                h.update(f.read(TAIL))
            elif size > HEAD:
                f.seek(HEAD)
                h.update(f.read(size - HEAD))
    except OSError as exc:
        # Fall back to the legacy seed if we can't read the file.
        seed = f"{str(p)}\0{size}\0{int(st.st_mtime)}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:hash_len]
    return h.hexdigest()[:hash_len]


# ---------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------
def _extract_frames_with_ffmpeg(video_path: Path,
                                  out_dir: Path,
                                  on_progress: Optional[Callable[[int, int], None]],
                                  ) -> int:
    """Extract every frame to PNG via ffmpeg.  Returns frame count.

    ffmpeg is dramatically faster than cv2 for sequential extraction
    AND it writes lossless PNGs deterministically named per frame.

    Fallback: if ffmpeg is missing, the cv2 path below handles it.
    """
    # Probe frame count first via cv2 (cheap, no decode).
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        raise RuntimeError(f"video has 0 frames or could not be opened: "
                            f"{video_path}")

    pattern = str(out_dir / "frame_%06d.png")
    # ``-vsync 0`` so ffmpeg writes exactly the frames it reads (no
    # duplicates or drops for cfr->cfr).  ``-start_number 0`` so the
    # first output is frame_000000.png matching our scrubber's 0-index.
    argv = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-i", str(video_path),
        "-vsync", "0",
        "-start_number", "0",
        "-pix_fmt", "rgb24",
        pattern,
    ]
    logger.info("ffmpeg extract: %s", argv)
    proc = subprocess.Popen(argv,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE,
                              text=True)
    last_progress = 0
    # ffmpeg stderr contains "frame=N" lines we can parse for progress.
    if proc.stderr is not None:
        for line in proc.stderr:
            line = line.strip()
            if "frame=" in line:
                # very loose parse: pick the integer after "frame="
                try:
                    parts = line.split("frame=", 1)[1].split()
                    n = int(parts[0])
                    if on_progress and n - last_progress >= 50:
                        on_progress(n, total)
                        last_progress = n
                except Exception:
                    pass
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg extract exited {rc}")
    written = len(list(out_dir.glob("frame_*.png")))
    if on_progress:
        on_progress(written, total)
    return written


def _extract_frames_with_cv2(video_path: Path,
                              out_dir: Path,
                              on_progress: Optional[Callable[[int, int], None]],
                              ) -> int:
    """cv2 fallback when ffmpeg isn't on PATH."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0
    last_progress = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        out_path = out_dir / f"frame_{idx:06d}.png"
        cv2.imwrite(str(out_path), frame)
        idx += 1
        if on_progress and (idx - last_progress) >= 50:
            on_progress(idx, total)
            last_progress = idx
    cap.release()
    if on_progress:
        on_progress(idx, total)
    return idx


def extract_and_cache(video_path: Path,
                       *,
                       force: bool = False,
                       on_progress: Optional[Callable[[int, int], None]] = None,
                       log: Callable[[str], None] = print,
                       ) -> FrameCacheInfo:
    """Extract ``video_path`` to a PNG sequence + write meta.json.

    Idempotent: if the cache already exists and ``force=False``, this
    is just a quick metadata load.

    Args
    ----
    video_path : Path
        Source target video.  Must exist.
    force : bool, default False
        Delete and re-extract the cache.  Use after a re-encode or if
        you suspect a partial / corrupted previous extract.
    on_progress : callable(written:int, total:int), optional
        Called every ~50 frames during extraction.
    log : callable(str), default print
        Diagnostic log sink.

    Returns
    -------
    FrameCacheInfo
        Includes the cache dir + meta needed by the UI.
    """
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    video_hash = compute_video_hash(video_path)
    cache_dir = CACHE_ROOT / video_hash
    frames_dir = cache_dir / "frames"
    masks_dir = cache_dir / "masks"
    meta_path = cache_dir / "meta.json"

    if force and cache_dir.exists():
        log(f"[rotoscope-cache] force=True, removing {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)

    # Quick path: if meta.json is valid and frame count matches the
    # frames on disk, just load + return.
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            existing_pngs = sum(1 for _ in frames_dir.glob("frame_*.png"))
            if existing_pngs == int(meta.get("frame_count", -1)):
                log(f"[rotoscope-cache] HIT  {video_hash}  "
                    f"{existing_pngs} frames @ {meta['width']}x{meta['height']}")
                masks_dir.mkdir(parents=True, exist_ok=True)
                return FrameCacheInfo(
                    video_path=str(video_path),
                    video_hash=video_hash,
                    cache_dir=str(cache_dir),
                    frames_dir=str(frames_dir),
                    masks_dir=str(masks_dir),
                    frame_count=int(meta["frame_count"]),
                    width=int(meta["width"]),
                    height=int(meta["height"]),
                    fps=float(meta.get("fps", 30.0)),
                    duration_s=float(meta.get("duration_s", 0.0)),
                )
            log(f"[rotoscope-cache] STALE  expected "
                f"{meta.get('frame_count')} frames, found {existing_pngs}")
        except Exception as exc:
            log(f"[rotoscope-cache] bad meta.json ({exc}); re-extracting")

    # Extract.
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    log(f"[rotoscope-cache] MISS  extracting -> {frames_dir}")

    # Probe dims + fps first (we want them whether ffmpeg or cv2 wins).
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    expected_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    t0 = time.perf_counter()
    used = "?"
    try:
        written = _extract_frames_with_ffmpeg(video_path, frames_dir,
                                                on_progress)
        used = "ffmpeg"
    except Exception as exc:
        log(f"[rotoscope-cache] ffmpeg failed ({exc}); using cv2")
        # Wipe any partial output before retrying.
        for f in frames_dir.glob("frame_*.png"):
            try: f.unlink()
            except Exception: pass
        written = _extract_frames_with_cv2(video_path, frames_dir,
                                             on_progress)
        used = "cv2"
    dt = time.perf_counter() - t0
    log(f"[rotoscope-cache] EXTRACT done via {used} -- {written} "
        f"frames in {dt:.1f}s ({written / max(dt, 1e-6):.1f} fps)")

    duration_s = float(written) / fps if fps > 0 else 0.0
    info = FrameCacheInfo(
        video_path=str(video_path),
        video_hash=video_hash,
        cache_dir=str(cache_dir),
        frames_dir=str(frames_dir),
        masks_dir=str(masks_dir),
        frame_count=written,
        width=width,
        height=height,
        fps=fps,
        duration_s=duration_s,
    )
    meta = asdict(info)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return info


# ---------------------------------------------------------------------
# Quick lookups for the UI
# ---------------------------------------------------------------------
def load_cache_info(video_path: Path) -> Optional[FrameCacheInfo]:
    """Return cached info if a complete extract exists, else None."""
    try:
        video_path = Path(video_path).expanduser().resolve()
        h = compute_video_hash(video_path)
    except Exception:
        return None
    meta_path = CACHE_ROOT / h / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        d = json.loads(meta_path.read_text(encoding="utf-8"))
        return FrameCacheInfo(**d)
    except Exception:
        return None


def frame_path(info: FrameCacheInfo, frame_idx: int) -> Path:
    """Path to a single cached frame PNG.  Does NOT check existence."""
    return Path(info.frames_dir) / f"frame_{int(frame_idx):06d}.png"


def mask_dir(info: FrameCacheInfo, obj_id: int = 1) -> Path:
    """Where SAM2 daemon should write masks for the given object."""
    d = Path(info.masks_dir) / f"obj_{int(obj_id)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


__all__ = [
    "FrameCacheInfo",
    "CACHE_ROOT",
    "compute_video_hash",
    "extract_and_cache",
    "load_cache_info",
    "frame_path",
    "mask_dir",
]
