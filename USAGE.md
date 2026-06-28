# User manual

This is the end-user manual for faceswap_pro v2. For install
instructions see [INSTALL.md](INSTALL.md). For the contributor
workflow see [CONTRIBUTING.md](CONTRIBUTING.md).

## Contents

- [Launching the app](#launching-the-app)
- [Lip-sync tab](#lip-sync-tab)
  - [LatentSync knobs](#latentsync-knobs)
  - [Per-clip identity fine-tune](#per-clip-identity-fine-tune)
  - [Mask out non-face regions (SAM 2)](#mask-out-non-face-regions-sam-2)
  - [Voice swap](#voice-swap)
  - [Watermark & aspect](#watermark--aspect)
  - [Job queue](#job-queue)
- [Face-swap tab](#face-swap-tab)
  - [Source + target](#source--target)
  - [Face selector](#face-selector)
  - [Face mask & identity strength](#face-mask--identity-strength)
  - [Identity blend / journey](#identity-blend--journey)
- [Webcam tab](#webcam-tab)
- [Presets](#presets)
- [History](#history)
- [Reproducibility](#reproducibility)

---

## Launching the app

```bash
python launch.py
# or:
launch.bat
```

The browser opens at `http://localhost:7860`. Wait until the console
shows `Running on local URL:` before clicking anything — first-run
model downloads happen in the background.

---

## Lip-sync tab

The lip-sync tab takes a **face clip** (a video of someone you want
to lip-sync) and an **audio file** (the new audio you want them to
"speak") and produces a video where the face mouths along to the
audio.

### LatentSync knobs

- **Inference steps** (10–50, default 20): DDIM steps. More steps =
  marginally cleaner output, linear time penalty. 20 is the sweet
  spot.
- **Lip strength (guidance scale)** (1.0–5.0, default 1.5): how
  aggressively the model follows the audio. 1.5–2.0 for clean
  speech, 3–5 for songs with quiet vocals or mumbly speech.
- **DeepCache** (checkbox, default ON): ~2× speedup via diffusion
  feature caching. Tiny quality hit; almost always worth it.
- **Seed** (–1 = random, any int = reproducible).

### Per-clip identity fine-tune

For repeated work on one face video, you can fine-tune the
LatentSync UNet's attention + motion layers on that clip. Result:
much stronger identity retention and finer mouth motion on
characteristic vowels.

1. Upload your face clip + an audio file (the audio is required —
   we need it to make the training data, but it's also the audio
   you'll lip-sync with later)
2. Open the **Per-clip identity fine-tune** accordion
3. The status banner shows whether a fine-tune already exists for
   this clip:
   - `_no clip loaded_` — pick a face video first
   - `none yet (click *Train* to create one)` — fresh state
   - `READY (XXX MB) — will be auto-used on next render of this clip` —
     a fine-tune is already on disk and renders will use it
4. Adjust **Fine-tune steps** (100–5000, default 1000)
5. Click **Train per-clip fine-tune (slow)**. The log box streams
   progress. Expect 10–30 min on a 4090; longer if you raised steps.
6. When the status flips to `READY`, render normally with the
   Run button — the fine-tuned UNet is used automatically.

**Costs**: per-clip ckpt is ~3 GB and stored under
`models/lipsync_finetune/clip_<hash>/checkpoint/`. Delete the
clip workspace dir to free disk.

### Mask out non-face regions (SAM 2)

When your source video has a non-face object that has face-like
features (a cat, statue, animal, prop), LatentSync's internal
detector will sometimes latch onto it and try to lip-sync those
fake "eyes" and "mouths". The mask-out feature lets you click
the object away.

1. Open the **Mask out non-face regions (SAM 2)** accordion
2. Tick **Enable mask-out**
3. Set **Click frame index** (default 0) and click **load preview
   frame** — frame 0 of the face clip appears in the preview area
4. Set the **+/- radio** to `+ positive` and click on the object
   you want to remove. The click is appended to the visible list
5. Switch the radio to `- negative` and click anywhere SAM 2
   over-grabbed (e.g. extended into the person's shoulder)
6. Add more positive clicks to extend the mask if it under-grabs
7. **Mask dilate** (0–40 px, default 12): grows the mask before
   inpaint to cover TELEA edge bleed
8. **Composite feather** (0–24 px, default 8): Gaussian edge of
   the final paste-back; hides the seam

Under the hood (orchestrator Stage 1.76 → Stage 2.05):
1. SAM 2 propagates your clicks to every frame → binary mask
2. Source video is TELEA-inpainted in the masked region → void
   source where the object is gone
3. LatentSync runs on the void source — no spurious face detected
4. After lip-sync, the original masked pixels are composited back
   over the lip-sync output with a feathered edge

### Voice swap

Optional RVC-based voice conversion applied to the audio *before*
it's used to drive LatentSync. Pick a voice model from the dropdown
(must be at `voice_models/<name>/<name>.pth`) and a transpose
amount in semitones.

The vocal stem is also separated from instrumental via Demucs
*before* RVC and lip-sync, then the original instrumental is
re-muxed onto the final output. So you can lip-sync to a full
song and still hear the music.

### Watermark & aspect

- **Watermark**: overlay an image on every frame. Position TL/TR/
  BL/BR/center, scale percent of frame width, opacity.
- **Aspect ratio**: crop or pad to a target ratio (9:16, 16:9, 1:1,
  4:5, etc).

### Job queue

Click **Queue** instead of **Run** to submit a job to the
background queue. The Queue tab shows status. The queue worker
drains jobs serially and shares a single render lock with the
foreground Run button — they never collide on GPU.

---

## Face-swap tab

### Source + target

- **Source face image**: a .jpg or .png of the face you want to
  paste into the target. Drop any image type — invalid uploads
  no longer wedge the picker.
- **Target video**: any video. The first frame's largest face is
  identified, swapped, and the rest of the video tracks identity
  drift to stay consistent.
- **Blend method**: `poisson` (seamless skin tone, default),
  `alpha`, `feather`, or `none` (hard paste).
- **Enhance faces (GFPGAN)**: post-swap face restoration.
- **Detection threshold** (0.1–0.9): higher rejects faces detected
  with low confidence.

### Face selector

For multi-person videos, the default `largest` mode swaps the
biggest face in every frame. Switch to `reference` and upload a
**reference face image** to swap only the face that matches that
identity in cosine distance.

- **Reference distance threshold** (0.05–1.5, default 0.6): lower
  = stricter. 0.3–0.4 for tight matches; 0.6 sane default; >1.0
  effectively disables the filter.

### Face mask & identity strength

- **Pixel boost** (128/256/384/512/768): runs GFPGAN on the swap
  crop scaled up to this resolution before paste-back. 128 = stock
  (off). 512 is the sweet spot for close-ups; 768 is heavier.
- **Mask padding** (–30 to +30 px): + shrinks the mask inward
  (more original face shows around jaw/forehead), – pushes it
  outward (swap extends past the edge).
- **Mask blur scale** (0.0–4.0): multiplier on the auto-computed
  edge feather. Larger = softer seam.
- **Identity strength** (0.0–1.0): LERP between original face
  (0.0) and full swap (1.0). Use 0.7–0.85 for "kinda looks like"
  effects.
- **Enhancer blend** (0.0–1.0): only active when GFPGAN is on.
  Lower = less plasticky skin.

### Identity blend / journey

Drop a **Source face B** image to enable blend / journey modes.

- **Blend alpha** (0.0–1.0): static mix in ArcFace embedding space.
  0 = pure A, 1 = pure B, 0.5 = 50/50 hybrid. Same identity used
  on every frame.
- **Journey mode** (checkbox): instead of a static alpha, the alpha
  ramps from `journey_start_alpha` at frame 0 to `journey_end_alpha`
  at the last frame.
- **Journey curve**: `linear` (constant rate) or `smoothstep`
  (S-curve: slow start, slow end).

---

## Webcam tab

Real-time face-swap from your webcam.

1. **Source face image**: same as the Face-swap tab — drop the face
   you want to wear.
2. Click **Start webcam**. The browser will prompt for camera
   permission.
3. Optional: tick **Send to virtual camera** to publish the swapped
   feed as a Windows virtual camera (requires `pyvirtualcam` and OBS).

Throughput: ~25 fps at 720p on a 4090, ~15 fps on a 3090.

---

## Presets

Save common knob configurations in the **Preset** dropdown on the
lip-sync tab. Stored under `presets/<name>.json` as a serialized
LipsyncJob dataclass.

---

## History

Every successful render writes a JSON sidecar (`<output>.job.json`)
under `recordings/lipsync/`. The History dropdown lets you preview
any past render and re-load its knobs into the UI.

---

## Reproducibility

If you supply a fixed `seed` for LatentSync, the lip-sync output is
deterministic per (source clip + audio + seed) tuple on the same
GPU model. Face-swap is deterministic given the same inswapper_128
checkpoint and detector threshold.

Per-clip fine-tune is also deterministic given a fixed seed (default
1247), so collaborators can reproduce your fine-tune exactly.

---

## Where files live

```
recordings/lipsync/    final renders (mp4) + JSON sidecars
recordings/video_swap/ face-swap outputs
recordings/webcam/     webcam captures
models/lipsync_finetune/clip_<hash>/   per-clip fine-tune workspace
checkpoints/           all auto-downloaded model weights
checkpoints/sam2/      SAM 2 weights (downloaded on first mask-out)
checkpoints/inswapper_128.onnx     YOU place this manually
voice_models/<name>/   RVC voice models (optional)
presets/               saved knob configurations
```

If you need to free disk: deleting `recordings/`, individual `models/
lipsync_finetune/clip_*/` workspaces, or specific lines in
`checkpoints/` is all safe (auto-redownloads on next use).
