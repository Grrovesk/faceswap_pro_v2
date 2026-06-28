# Install guide

## Contents

- [Hardware requirements](#hardware-requirements)
- [Python + venv](#python--venv)
- [Step 1 — clone](#step-1--clone)
- [Step 2 — Python deps](#step-2--python-deps)
- [Step 3 — PyTorch CUDA build](#step-3--pytorch-cuda-build)
- [Step 4 — inswapper_128 (manual)](#inswapper_128)
- [Step 5 — first launch (auto-downloads)](#step-5--first-launch)
- [Optional: voice swap (RVC)](#optional-voice-swap-rvc)
- [Optional: virtual camera](#optional-virtual-camera)
- [Verify your install (`detect_system.py`)](#verify-your-install-detect_systempy)
- [Troubleshooting](#troubleshooting)

---

## Hardware requirements

| Workload | VRAM | RAM | Disk |
|---|---|---|---|
| Lip-sync render (LatentSync 1.6, 512px) | 8 GB | 16 GB | 15 GB free |
| Face-swap render (inswapper_128) | 4 GB | 8 GB | 5 GB free |
| Webcam swap (real-time) | 4 GB | 8 GB | 2 GB free |
| Per-clip lip-sync fine-tune | 20 GB | 32 GB | 30 GB free |
| All features comfortably | 24 GB | 32 GB | 50 GB free |

CPU-only execution is **not supported** for lip-sync (LatentSync hard-
requires CUDA). Face-swap and webcam may run on CPU but speeds are
not usable.

GPU: NVIDIA CUDA 12.1+ recommended. Tested on RTX 3090 / 4090 /
A6000.

---

## Python + venv

This project is pinned to **Python 3.10**. PyTorch 2.4 cu121 +
diffusers 0.32.x + the LatentSync repo combine cleanly on 3.10;
newer Python versions have not been validated.

```bash
# Windows (PowerShell or cmd):
py -3.10 -m venv venv_new
venv_new\Scripts\activate

# Linux/macOS:
python3.10 -m venv venv_new
source venv_new/bin/activate

# Verify
python --version  # should be 3.10.x
```

---

## Step 1 — clone

```bash
git clone https://github.com/seedhunterai/faceswap_pro_v2.git
cd faceswap_pro_v2
```

---

## Step 2 — Python deps

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Note: `requirements.txt` deliberately does **not** pin torch /
torchvision — install those manually in step 3 so we control which
CUDA build you get. If your environment already has them, `pip
install -r requirements.txt` skips them.

---

## Step 3 — PyTorch CUDA build

```bash
# CUDA 12.1 (validated combo):
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (alternative):
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

Verify CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True NVIDIA GeForce RTX 4090
```

---

## inswapper_128

The **inswapper_128.onnx** face-swap model (~554 MB) is not
distributed in this repo. InsightFace removed it from official
distribution in mid-2023 over ethical concerns about deepfake misuse.

You must source the file yourself, place it at:

```
checkpoints/inswapper_128.onnx
```

and accept the ethical-use terms in [USAGE_POLICY.md](USAGE_POLICY.md)
before using the face-swap features. Lip-sync features do **not**
require this file.

The webcam tab and face-swap tab will refuse to start without the
model present, with a clear error message.

If you can't or won't source the model, the lip-sync tab still
works as a standalone feature.

---

## Step 5 — first launch

```bash
python launch.py
# or on Windows:
launch.bat
```

The first launch downloads the rest of the model assets automatically:

| Asset | Size | Source | Triggered by |
|---|---|---|---|
| LatentSync 1.6 UNet | ~5 GB | `huggingface.co/ByteDance/LatentSync-1.6` | First lip-sync render |
| Whisper tiny | ~78 MB | same HF repo | First lip-sync render |
| GFPGAN v1.4 | ~350 MB | Tencent release | First GFPGAN-enabled render |
| Demucs htdemucs | ~200 MB | Meta release | First vocal-isolate render |
| InsightFace buffalo_l | ~250 MB | onnxruntime model zoo | First face-swap render |
| SAM 2.1 base_plus | ~81 MB | `dl.fbaipublicfiles.com` | First mask-out render |

All downloads land under `checkpoints/` and `models/` and are cached
across launches. Total disk: ~6 GB on top of inswapper_128.

The Gradio UI opens at `http://localhost:7860` once everything is
ready. Watch the console for progress on the first run.

---

## Required: clone LatentSync (lip-sync engine)

LatentSync is upstream code we don't redistribute -- you clone it
yourself, the same way as RVC below. The wrapper code in `core/`
expects it at `lipsync_test/LatentSync/`:

```bash
git clone https://github.com/bytedance/LatentSync.git \
    lipsync_test/LatentSync
```

The LatentSync model weights (~5 GB) download automatically into
`lipsync_test/LatentSync/checkpoints/` on first lip-sync render --
nothing to fetch manually. SAM 2 weights (~81 MB) likewise auto-
download into `checkpoints/sam2/` on first mask-out render.

---

## Optional: voice swap (RVC)

To enable the voice-swap feature, install the RVC repo:

```bash
# Clone RVC peer of LatentSync:
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git \
    ../lipsync_test/RVC
cd ../lipsync_test/RVC
python -m pip install -r requirements.txt
```

Place your voice models at:

```
faceswap_pro_v2/voice_models/<voice_name>/<voice_name>.pth
faceswap_pro_v2/voice_models/<voice_name>/<voice_name>.index
```

They appear in the Voice swap dropdown automatically.

---

## Optional: virtual camera

For Windows OBS-style virtual camera output from the webcam tab:

```bash
pip install pyvirtualcam
```

Then install OBS Studio (which registers the virtual camera driver).
The webcam tab will show a "Send to virtual camera" toggle.

---

## Verify your install (`detect_system.py`)

Before your first render, run the bundled environment audit:

```bash
python detect_system.py
```

It detects your hardware (CPU, RAM, GPU, VRAM, CUDA, cuDNN), checks
Python + PyTorch + ffmpeg, verifies every pinned package in
`requirements.txt` is installed at the right version, and looks for
the optional weights the app uses (SAM2, inswapper_128.onnx,
GFPGAN). Each finding is tagged:

- **BLOCKER** — the app will not run until you fix it
- **WARN** — the app runs, but a feature will OOM / be unavailable
- **INFO** — informational (cuDNN version, FP8-capable compute, etc.)
- **PASS** — working as expected

Every blocker and warning ships with a `fix_hint` — the exact
command to run to fix it.

Two artifacts are written next to the script:

- `PROJECT_ENV.md` — human-readable hardware + compatibility report
- `PROJECT_ENV.json` — same data, machine-readable

Exit code is `0` if no blockers, `1` if any. Safe to call from CI
or wrapper scripts.

---

## Troubleshooting

> **Before opening an issue:** run `python detect_system.py` and
> attach the generated `PROJECT_ENV.md`. Most "it won't run"
> reports resolve themselves from the BLOCKER / WARN entries — and
> when they don't, the report tells us your exact environment so
> we don't need a back-and-forth to triage.

### "ImportError: cannot import name X from diffusers"

You have a diffusers version mismatch. LatentSync 1.6 wants
diffusers 0.32.x. Pin it:

```bash
pip install "diffusers==0.32.2"
```

### "RuntimeError: CUDA out of memory" during fine-tune

Per-clip fine-tune needs ~20 GB on a 512-resolution UNet. Either:
- Use a card with 24 GB+
- Lower the fine-tune `num_steps` slider (won't reduce peak VRAM,
  just total time)
- Skip fine-tune; stock LatentSync 1.6 is already a strong baseline

### "WinError 32: process cannot access the file"

Gradio's video player is holding a handle to the last render output
while the next render tries to overwrite it. The orchestrator has a
retry + timestamped-fallback safety net for this — no user action
needed; renders proceed under a timestamped filename when contention
persists.

### "Video not playable" on upload

The browser can't decode the file. Most common cause: the source is
encoded in HEVC/H.265 which not all browsers support. Re-encode to
H.264:

```bash
ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4
```

### Mask-out fails with "SAM2 weights not found"

The auto-download couldn't reach `dl.fbaipublicfiles.com`. Either
download `sam2.1_hiera_base_plus.pt` manually to
`checkpoints/sam2/sam2.1_hiera_base_plus.pt`, or set the
`SAM2_CKPT_NAME` env var to one of `tiny / small / base_plus / large`
and retry.

### LatentSync render starts but fps is very low

Check that `torch.cuda.is_available()` returns True in your venv.
The most common cause is that PyTorch was installed without CUDA
(CPU build); LatentSync will fall back to CPU and run at ~0.01 fps.

### First face-swap render says "InsightFace model not found"

You skipped step 4. inswapper_128.onnx is required for face-swap and
webcam features. Place it at `checkpoints/inswapper_128.onnx`.

---

## Verifying the install

The fastest verification is the [`detect_system.py`](#verify-your-install-detect_systempy)
audit above — it covers Python, CUDA, every pinned package, and
optional weights in one shot. The lower-level smoke tests below
remain useful when you want to confirm the **Python import graph**
itself is clean (i.e. your editor didn't half-save a file mid-edit).

Smoke test (no model loads):

```bash
python -c "from faceswap.config import LipsyncJob, VideoSwapJob; print('configs OK')"
python -c "from faceswap.orchestrator import render; print('orchestrator OK')"
python -c "from core.pipeline import FaceSwapPipeline; print('pipeline OK')"
```

Smoke test (face-swap, ~30 s):

```bash
# Drop a face photo at sample_face.jpg and a short clip at sample_target.mp4
python -c "
from faceswap.config import VideoSwapJob
from faceswap import video_swap
job = VideoSwapJob(source_image='sample_face.jpg',
                    target_video='sample_target.mp4',
                    enhance_faces=False)
out = video_swap.run(job)
print('OK ->', out)
"
```

If all three smoke tests pass, the install is good.
