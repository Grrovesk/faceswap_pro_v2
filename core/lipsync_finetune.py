"""core/lipsync_finetune.py -- per-clip identity fine-tune for LatentSync.

Plan in 4 phases:
  Phase 1 (THIS FILE, scaffolding): face extraction + preprocessing
  Phase 2 (TODO): minimal training loop (recon loss only, no SyncNet)
  Phase 3 (TODO): inference uses per-clip checkpoint
  Phase 4 (TODO): UI integration + cache

Per-clip workspace layout under PROJECT_ROOT/models/lipsync_finetune/:
  clip_<hash>/
      preprocessed_512.mp4         <- output of Phase 1
      checkpoint/                   <- output of Phase 2
          latentsync_unet_finetuned.pt
      train_log.txt
      meta.json
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINETUNE_ROOT = PROJECT_ROOT / "models" / "lipsync_finetune"
FINETUNE_ROOT.mkdir(parents=True, exist_ok=True)

# Match LatentSync stage2_512 expected input format
TRAIN_RESOLUTION = 512
TRAIN_FPS = 25
TRAIN_AUDIO_HZ = 16000


def _clip_hash(clip_path: str) -> str:
    """Stable hash of (abs_path, mtime, size). Different copies of the
    same content land in the same workspace dir."""
    st = os.stat(clip_path)
    h = hashlib.sha256()
    h.update(os.path.abspath(clip_path).encode())
    h.update(str(int(st.st_mtime)).encode())
    h.update(str(st.st_size).encode())
    return h.hexdigest()[:16]


def get_finetune_dir(clip_path: str) -> Path:
    """Return (and create) per-clip workspace directory."""
    h = _clip_hash(clip_path)
    d = FINETUNE_ROOT / f"clip_{h}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------
# Phase 1: face crop + FPS/audio normalize
# ------------------------------------------------------------------
def _resolve_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg") or "ffmpeg"


def _have_nvenc(ffmpeg: str) -> bool:
    try:
        r = subprocess.run([ffmpeg, "-encoders"],
                           capture_output=True, text=True, timeout=5)
        return "h264_nvenc" in (r.stdout or "")
    except Exception:
        return False


def prepare_clip_for_training(source_video: str, log=print,
                                source_audio: str = None) -> Path:
    """Phase 1: normalize the source clip to LatentSync's training
    requirements WITHOUT transforming the video pixels.

    LatentSync's UNetDataset reads the video with `affine_transform=
    False` -- the dataset does NOT warp/crop/align frames; it expects
    the input to already match what the model was pre-trained on.

    All this function does:
      1. Resample video to 25 FPS (LatentSync's expected fps)
      2. Take audio from `source_audio` (if given) else from
         `source_video`; resample to 16 kHz mono.
         If neither has audio, synthesize a silent 16 kHz track of
         matching duration so Whisper has something to read -- this
         degrades audio-driven mouth motion but keeps identity
         fine-tune viable when the user only has video.
      3. NO face cropping, NO resize, NO affine warp -- the video
         pixels are passed through identically.

    The original ratio is preserved; LatentSync handles any internal
    resize during training.

    Idempotent: returns cached output on second call for the same clip.
    """
    if not os.path.isfile(source_video):
        raise FileNotFoundError(source_video)

    work = get_finetune_dir(source_video)
    out_path = work / "preprocessed_512.mp4"
    if out_path.exists() and out_path.stat().st_size > 1024:
        log(f"[finetune] preprocessing cache hit: {out_path}")
        return out_path

    log(f"[finetune] preparing training clip from {Path(source_video).name}")
    log("[finetune] NO face crop, NO resize -- only fps/audio normalize")

    ffmpeg = _resolve_ffmpeg()

    # ---- pick the audio source ----
    # Priority: explicit source_audio kwarg -> probe source_video for
    # an audio stream -> fall back to synthesized silence.
    audio_input_path = None
    audio_source_label = "none"
    if source_audio and os.path.isfile(source_audio):
        audio_input_path = source_audio
        audio_source_label = f"explicit ({Path(source_audio).name})"
    else:
        # Probe source_video for an audio stream
        try:
            probe = subprocess.run(
                [ffmpeg, "-loglevel", "error", "-i", str(source_video)],
                capture_output=True, text=True, timeout=10,
                encoding="utf-8", errors="ignore")
            probe_txt = (probe.stderr or "") + (probe.stdout or "")
            if "Audio:" in probe_txt:
                audio_input_path = str(source_video)
                audio_source_label = "source video"
        except Exception:
            pass
    log(f"[finetune] audio source: {audio_source_label}")

    if _have_nvenc(ffmpeg):
        v_enc = ["-c:v", "h264_nvenc", "-preset", "p4",
                  "-rc", "vbr", "-cq", "20"]
    else:
        v_enc = ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]

    if audio_input_path is not None:
        # Mux video from source + audio from audio_input_path. Map
        # streams explicitly so an audio-less source doesn't pull a
        # phantom audio track. `-shortest` clips to whichever stream
        # ends first so we don't pad the video with frozen frames.
        argv = [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(source_video),
            "-i", str(audio_input_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-r", str(TRAIN_FPS),
            "-pix_fmt", "yuv420p",
        ] + v_enc + [
            "-ac", "1",
            "-ar", str(TRAIN_AUDIO_HZ),
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(out_path),
        ]
    else:
        # No audio anywhere -> synthesize 16 kHz mono silence the same
        # length as the source. Whisper will produce constant features
        # but at least the dataset will read successfully. Warn loudly
        # because audio-driven mouth motion will degrade.
        log("[finetune] WARNING: no audio in source_video AND no "
            "source_audio supplied. Synthesizing silence for training "
            "audio track. Audio-driven mouth motion will degrade -- "
            "pass source_audio for a real identity+audio fine-tune.")
        argv = [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(source_video),
            "-f", "lavfi",
            "-i", f"anullsrc=channel_layout=mono:sample_rate={TRAIN_AUDIO_HZ}",
            "-map", "0:v:0", "-map", "1:a:0",
            "-r", str(TRAIN_FPS),
            "-pix_fmt", "yuv420p",
        ] + v_enc + [
            "-ac", "1",
            "-ar", str(TRAIN_AUDIO_HZ),
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(out_path),
        ]
    r = subprocess.run(argv, capture_output=True, text=True,
                       encoding="utf-8", errors="ignore", timeout=300)
    if r.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"ffmpeg fps/audio normalize failed:\n{(r.stderr or '')[-800:]}")

    # Read back actual dimensions for the meta file
    import cv2
    cap = cv2.VideoCapture(str(out_path))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    log(f"[finetune] output: {src_w}x{src_h} @ {out_fps:.2f} fps, "
        f"{out_n} frames (source dims preserved)")

    # Sidecar meta
    meta = {
        "source": str(source_video),
        "out_size": [src_w, src_h],
        "out_fps": out_fps,
        "out_frames": out_n,
        "out_audio_hz": TRAIN_AUDIO_HZ,
        "out_path": str(out_path),
        "prepared_at": int(time.time()),
        "transform": "none (fps + audio only)",
    }
    (work / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"[finetune] preprocessing DONE: {out_path} "
        f"({out_path.stat().st_size / (1024 * 1024):.1f} MB)")
    return out_path


# Old face-cropping code preserved here for reference, NOT called.
def _legacy_face_crop_prepare(source_video, log=print):
    """DO NOT USE. The old Phase 1 that face-cropped to 512x512.
    Caused warping when the face was near the frame edge -- the crop
    rect got clamped asymmetrically and the non-square crop got
    stretched in the 512x512 resize. Kept here in case the canonical
    crop approach is ever needed again, but the new prepare_clip_for_
    training does NOT call it."""
    import cv2
    import numpy as np
    work = get_finetune_dir(source_video)
    from .xseg_gate import _get_face_detector
    fa = _get_face_detector()
    cap = cv2.VideoCapture(source_video)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    crops_xyxy = []
    last = None
    bbox_alpha = 0.85
    for fi in range(src_n):
        ok, frame = cap.read()
        if not ok:
            crops_xyxy.append(last)
            continue
        faces = fa.get(frame)
        if not faces:
            crops_xyxy.append(last)
            continue
        f = max(faces,
                key=lambda x: (x.bbox[2] - x.bbox[0])
                              * (x.bbox[3] - x.bbox[1]))
        x1, y1, x2, y2 = f.bbox
        bw, bh = x2 - x1, y2 - y1
        # Square + 40% pad so hair and chin fit comfortably
        side = max(bw, bh) * 1.40
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        half = side * 0.5
        # NO CLAMPING -- square crop, may extend past image bounds.
        rect = (
            float(cx - half), float(cy - half),
            float(cx + half), float(cy + half),
        )
        if last is None:
            sm = rect
        else:
            a = bbox_alpha
            sm = tuple(a * lp + (1.0 - a) * np_ for lp, np_ in zip(last, rect))
        last = sm
        crops_xyxy.append(sm)

    cap.release()

    if not any(c is not None for c in crops_xyxy):
        raise RuntimeError(
            "no faces detected in source clip; cannot prepare for training")

    log(f"[finetune] detected face in "
        f"{sum(1 for c in crops_xyxy if c is not None)}/{src_n} frames")

    # Second pass: read frames, affine-crop to 512x512, pipe to ffmpeg.
    ffmpeg = _resolve_ffmpeg()
    if _have_nvenc(ffmpeg):
        enc_args = ["-c:v", "h264_nvenc", "-preset", "p4",
                    "-rc", "vbr", "-cq", "20"]
    else:
        enc_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]

    enc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{TRAIN_RESOLUTION}x{TRAIN_RESOLUTION}",
         "-r", f"{TRAIN_FPS:.6f}", "-i", "-"]
        + enc_args
        + ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(raw_path)],
        stdin=subprocess.PIPE,
    )

    cap = cv2.VideoCapture(source_video)
    t0 = time.time()
    written = 0
    for fi, crop_rect in enumerate(crops_xyxy):
        ok, frame = cap.read()
        if not ok or crop_rect is None:
            continue
        # Round to integer pixel coords -- still square (rounding both
        # ends symmetrically preserves squareness within 1px).
        x1, y1, x2, y2 = [int(round(v)) for v in crop_rect]
        # Enforce strict squareness in case rounding made it 1px off.
        cw = x2 - x1
        ch = y2 - y1
        if cw != ch:
            s = max(cw, ch)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            half = s // 2
            x1, y1, x2, y2 = cx - half, cy - half, cx - half + s, cy - half + s

        # Build a square canvas at (s x s) and copy the in-bounds
        # portion of the source frame into it. Out-of-bounds pixels
        # stay black. This keeps the aspect ratio EXACTLY 1:1 so the
        # 512x512 resize doesn't stretch.
        s = x2 - x1
        canvas = np.zeros((s, s, 3), dtype=np.uint8)
        src_x1 = max(0, x1); src_y1 = max(0, y1)
        src_x2 = min(src_w, x2); src_y2 = min(src_h, y2)
        if src_x2 > src_x1 and src_y2 > src_y1:
            dst_x1 = src_x1 - x1
            dst_y1 = src_y1 - y1
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = \
                frame[src_y1:src_y2, src_x1:src_x2]
        # Square in -> square out, no stretch.
        crop_512 = cv2.resize(canvas,
                              (TRAIN_RESOLUTION, TRAIN_RESOLUTION),
                              interpolation=cv2.INTER_AREA)
        enc.stdin.write(crop_512.tobytes())
        written += 1
        if fi % 25 == 0:
            log(f"[finetune] cropped {fi}/{src_n} "
                f"({time.time() - t0:.1f}s)")
    cap.release()
    enc.stdin.close()
    enc.wait()

    log(f"[finetune] wrote {written} face-crop frames -> {raw_path.name}")

    # FPS resample if needed + mux audio at 16 kHz mono.
    audio_args = ["-i", str(source_video), "-map", "0:v:0",
                  "-map", "1:a:0?",
                  "-ac", "1", "-ar", str(TRAIN_AUDIO_HZ),
                  "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                  "-shortest"]
    # If source fps differs from 25, ffmpeg above already retimed the
    # video stream (we wrote at TRAIN_FPS), so just mux audio.
    mux = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error",
         "-i", str(raw_path)] + audio_args + [str(out_path)],
        capture_output=True, text=True, timeout=120,
    )
    if mux.returncode != 0:
        log(f"[finetune] audio mux failed: {mux.stderr[-400:]}")
        # Fallback: ship the silent video
        shutil.copy2(raw_path, out_path)

    try:
        os.remove(raw_path)
    except Exception:
        pass

    # Sidecar meta
    meta = {
        "source": str(source_video),
        "source_size": [src_w, src_h],
        "source_fps": src_fps,
        "source_frames": src_n,
        "out_size": [TRAIN_RESOLUTION, TRAIN_RESOLUTION],
        "out_fps": TRAIN_FPS,
        "out_frames": written,
        "out_audio_hz": TRAIN_AUDIO_HZ,
        "out_path": str(out_path),
        "prepared_at": int(time.time()),
    }
    (work / "meta.json").write_text(json.dumps(meta, indent=2))

    log(f"[finetune] preprocessing DONE: {out_path} "
        f"({out_path.stat().st_size / (1024 * 1024):.1f} MB)")
    return out_path


# ------------------------------------------------------------------
# Phase 2: training (TODO -- next turn)
# ------------------------------------------------------------------
def train_identity_finetune(source_video: str, num_steps: int = 1000,
                              source_audio: str = None,
                              log=print) -> Path:
    """Phase 2: brief identity-only fine-tune of LatentSync's UNet on
    the preprocessed clip. Recon loss only (no SyncNet, no perceptual,
    no TREPA).

    Invokes scripts/train_oneshot.py inside LatentSync's repo via
    subprocess. Saves the fine-tuned UNet to:
        models/lipsync_finetune/clip_<hash>/checkpoint/
            latentsync_unet_finetuned.pt

    Returns the path to the saved checkpoint. Idempotent: returns the
    cached checkpoint on second call.
    """
    work = get_finetune_dir(source_video)
    pre_path = work / "preprocessed_512.mp4"
    if not pre_path.is_file():
        log("[finetune] preprocessing missing; running Phase 1 first")
        prepare_clip_for_training(source_video, log=log,
                                  source_audio=source_audio)

    ckpt_dir = work / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = ckpt_dir / "latentsync_unet_finetuned.pt"

    if out_ckpt.is_file() and out_ckpt.stat().st_size > 100 * 1024 * 1024:
        log(f"[finetune] training cache hit: {out_ckpt} "
            f"({out_ckpt.stat().st_size / (1024 * 1024):.0f} MB)")
        return out_ckpt

    # Locate LatentSync repo. Default to v2/lipsync_test/LatentSync so
    # we share the SAME repo + ckpt that core/lipsync.py uses at
    # inference time. Previously this defaulted to
    # PROJECT_ROOT.parent/lipsync_test (= faceswap_pro/lipsync_test),
    # which pointed at a SECOND LatentSync clone with a stale
    # train_oneshot.py (SingleClipDataset.__len__ inflated to 100000
    # while video_paths held 1 entry -> IndexError -> upstream
    # __getitem__'s except handler crashed printing an unbound
    # video_path). FACESWAP_EXTERNAL_REPOS still overrides if set.
    ext_root = Path(os.environ.get(
        "FACESWAP_EXTERNAL_REPOS",
        str(PROJECT_ROOT / "lipsync_test")))
    ls_dir = ext_root / "LatentSync"
    if not (ls_dir / "scripts" / "train_oneshot.py").is_file():
        raise RuntimeError(
            f"scripts/train_oneshot.py not found at {ls_dir}/scripts/. "
            "Phase 2 fine-tune cannot run.")
    base_ckpt = ls_dir / "checkpoints" / "latentsync_unet.pt"
    if not base_ckpt.is_file():
        raise RuntimeError(
            f"base UNet checkpoint not found at {base_ckpt}.")
    argv = [
        sys.executable, "-m", "scripts.train_oneshot",
        "--unet_config_path", "configs/unet/stage2_512.yaml",
        "--base_ckpt_path",   str(base_ckpt),
        "--clip_path",        str(pre_path),
        "--output_ckpt_path", str(out_ckpt),
        "--num_steps",        str(int(num_steps)),
    ]
    log(f"[finetune] Phase 2 training: {num_steps} steps")
    proc = subprocess.Popen(
        argv, cwd=str(ls_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="ignore", bufsize=1)
    for line in proc.stdout:
        log(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"train_oneshot exited {proc.returncode}")
    if not out_ckpt.is_file():
        raise RuntimeError(f"checkpoint missing: {out_ckpt}")
    log(f"[finetune] DONE: {out_ckpt}")
    return out_ckpt


def get_finetune_checkpoint(source_video):
    if not os.path.isfile(source_video):
        return None
    work = get_finetune_dir(source_video)
    ckpt = work / "checkpoint" / "latentsync_unet_finetuned.pt"
    if ckpt.is_file() and ckpt.stat().st_size > 100 * 1024 * 1024:
        return ckpt
    return None
