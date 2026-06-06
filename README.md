# faceswap_pro v2

A desktop Gradio app for AI face-swap and lip-sync on video, built on
LatentSync 1.6 (lip-sync), InsightFace inswapper_128 (face-swap),
GFPGAN (face restoration), Demucs (vocal isolation), SAM 2 (mask
propagation), and RVC (voice cloning).

> **⚠️ Synthetic media notice.** This software produces realistic
> face-swapped and lip-synced video. Before you use it, read
> [USAGE_POLICY.md](USAGE_POLICY.md). Non-consensual sexual imagery,
> CSAM, fraudulent impersonation, and political disinformation
> are prohibited.

---

## What it does

- **Lip-sync tab** — drive a face video with new audio (LatentSync
  1.6). Supports per-clip identity fine-tune, SAM 2 mask-out for
  non-face objects in frame, vocal isolation, voice swap, optional
  GFPGAN enhancement, watermark, aspect-ratio reshape.
- **Face-swap tab** — paste a source identity onto every face in a
  target video (inswapper_128). Includes identity blend / embedding
  journey (LERP two source faces over time), reference-face
  selector for multi-person videos, mask padding / blur, identity
  strength, enhancer blend, pixel boost (256/512/768 GFPGAN upscale
  on the swap crop before paste-back).
- **Webcam tab** — real-time face-swap on a webcam feed with optional
  virtual camera output.
- **Job queue** — render multiple jobs sequentially in the
  background; foreground and queue serialize on a single GPU lock.
- **History** — every render gets a JSON sidecar with knobs + timing
  so you can reproduce results.

---

## Quick start

### 1. Prerequisites

- **Windows 10/11** (Linux/macOS untested but plausibly works)
- **NVIDIA GPU** with 16 GB+ VRAM (24 GB recommended for fine-tune)
- **Python 3.10** in a venv (LatentSync 1.6 + diffusers 0.32.x +
  PyTorch 2.4 cu121 is the validated combo)
- **Git** and **ffmpeg** on PATH (optional — bundled fallback via
  `imageio-ffmpeg`)

### 2. Clone + install

```bash
git clone https://github.com/seedhunterai/faceswap_pro_v2.git
cd faceswap_pro_v2

python -m venv venv_new
venv_new\Scripts\activate     # Windows
# source venv_new/bin/activate # Linux/macOS

pip install -r requirements.txt
```

### 3. Acquire the inswapper_128 model

This file is **not bundled** for licensing reasons. See
[INSTALL.md § inswapper_128](INSTALL.md#inswapper_128) for sourcing.
Place it at:

```
checkpoints/inswapper_128.onnx
```

### 4. Launch

```bash
python launch.py
# or on Windows:
launch.bat
```

First-run downloads:
- LatentSync 1.6 UNet (~5 GB, from HuggingFace `ByteDance/LatentSync-1.6`)
- Whisper tiny (~78 MB)
- GFPGAN v1.4 + facial landmarks (~350 MB)
- Demucs htdemucs (~200 MB)
- SAM 2.1 base_plus (~81 MB, on first mask-out)
- Sub-models for InsightFace buffalo_l (~250 MB)

Plan for ~6 GB of model downloads on first launch. Subsequent launches
are cached.

Browser opens at `http://localhost:7860`.

---

## Features

### Lip-sync (LatentSync 1.6)

- 512×512 native resolution (1.6 retrained at 512 to fix the blurry
  teeth/lips problem of 1.5)
- DeepCache 2× speedup, deterministic seeds for reproducibility
- 20-step DDIM default; tunable
- Vocal isolation (Demucs) so the model conditions on clean vocals
  even when the input track is a song with instruments
- Per-clip identity fine-tune (~10-30 min on a 4090) overfits motion
  and attention layers to one source video for much better likeness
  retention
- SAM 2 mask-out: click on a non-face object (cat, animal, prop,
  statue) that LatentSync would wrongly latch onto; the region is
  inpainted before lip-sync and composited back after
- Per-clip extension (loop-pad short clips to audio length)

### Face-swap (inswapper_128)

- 128-native inswapper with stable face alignment
- **Pixel boost** (256/384/512/768) — post-swap GFPGAN upscale on the
  aligned face crop with scaled warp matrix; cleans up close-ups
- **Face mask padding + blur** — fix jaw-line seams without touching
  the model
- **Identity strength** — LERP between full swap (1.0) and original
  face (0.0)
- **Enhancer blend** — control GFPGAN intensity (anti-plasticky-skin
  slider)
- **Face selector mode**: `largest` (default) or `reference` —
  upload a reference image to swap only the matching face in
  multi-person videos
- **Identity blend** — two source images, alpha slider, hybrid
  identity in ArcFace space
- **Embedding journey** — alpha ramps A→B across the timeline for a
  continuous identity morph (linear or smoothstep curve)
- Trim by frame, output quality (visually_lossless / balanced /
  lossless)

### Webcam

- Real-time face-swap on webcam input
- Optional Windows virtual camera output (OBS / pyvirtualcam)

### Job queue

- Submit lip-sync jobs from the lip-sync tab's "Queue" button
- Queue worker drains them serially; foreground "Run" button and
  queue worker share a single render lock so two renders never share
  one GPU
- Cancel honored both during and before render (wakes a queued job
  to abort before model loads)

---

## Documentation

- **[INSTALL.md](INSTALL.md)** — detailed install + model download
  + Windows quirks
- **[USAGE.md](USAGE.md)** — per-tab user manual with screenshots
- **[USAGE_POLICY.md](USAGE_POLICY.md)** — what you may and may not
  use this software for
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — contributor workflow,
  code style, testing
- **[CHANGELOG.md](CHANGELOG.md)** — release notes
- **[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)** — upstream
  licenses and citations
- **[CITATION.cff](CITATION.cff)** — cite this project in academic
  work

---

## Architecture overview

```
                         ┌──────────────────┐
                         │   Gradio UI      │
                         │ (faceswap/ui.py) │
                         └────────┬─────────┘
                                  │
                  ┌───────────────┼───────────────┐
                  │               │               │
            ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
            │ Lip-sync  │   │ Face-swap │   │  Webcam   │
            │  job      │   │   job     │   │  worker   │
            └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
                  │               │               │
            ┌─────▼─────┐   ┌─────▼─────────────┐ │
            │ Orchestr- │   │ FaceSwapPipeline  │ │
            │ ator      │   │ (core/pipeline.py)│ │
            │ (Stages   │   └──────────┬────────┘ │
            │  1..5)    │              │          │
            └─────┬─────┘     ┌────────▼────────┐ │
                  │           │  SwapEngine     │◄┘
        ┌─────────┴─────────┐ │ (inswapper_128) │
        │                   │ └─────────────────┘
   ┌────▼────┐  ┌───────────▼─────┐
   │ Demucs  │  │  LatentSync 1.6 │
   │ (vocal) │  │  (subprocess)   │
   └─────────┘  └─────────────────┘
```

Key modules:
- `faceswap/orchestrator.py` — stages a render through voice swap →
  vocal isolation → optional mask-out → lip-sync → optional
  composite-back → optional GFPGAN → aspect → watermark → mux. Single
  global render lock.
- `core/pipeline.py` — face-swap pipeline (detect, identity, swap,
  audio-sync, lighting, blend, temporal, enhance per frame).
- `core/swap_engine.py` — inswapper_128 ONNX wrapper with InsightFace
  paste-back path.
- `core/lipsync.py` — LatentSync subprocess invocation, dep probe,
  HuggingFace download, checkpoint resolution.
- `core/maskout_pipeline.py` — SAM 2 multi-click → TELEA inpaint →
  void source → caller runs lip-sync → composite-back.
- `core/sam2_install.py` — SAM 2 weights resolver.
- `core/lipsync_finetune.py` — per-clip identity fine-tune via
  `LatentSync/scripts/train_oneshot.py` subprocess.

---

## Citing this project

If you use faceswap_pro in academic work, please cite the project
metadata in [CITATION.cff](CITATION.cff). If you use any of the
upstream models (LatentSync, inswapper_128, GFPGAN, SAM 2, Whisper,
Demucs), please cite their original papers — see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Additional usage restrictions apply — see [USAGE_POLICY.md](USAGE_POLICY.md).
