# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [2.0.0] -- 2026-06-06

### Added
- Per-clip lip-sync identity fine-tune (Phase 1 preprocessing,
  Phase 2 training, Phase 3 auto-use at inference, Phase 4 UI)
- SAM 2 multi-click mask-out for non-face objects in lip-sync sources
  (`Mask out non-face regions (SAM2)` accordion)
- Identity blend + embedding journey for face-swap
- Face mask padding + blur, identity strength, enhancer blend
  sliders
- Face selector mode (largest vs reference) with reference-face
  picker and cosine-distance threshold
- Pixel boost (256/384/512/768) for face-swap output detail
- Global render lock — foreground Run button and queue worker
  serialize on a single GPU

### Changed
- LatentSync default upgraded to 1.6 (retrained at 512×512 for
  sharper teeth/lips than 1.5)
- `_ensure_latentsync()` per-process cache eliminates spurious
  "installing diffusers/accelerate" messages on every render
- `gr.File` widgets no longer use `file_types=["image"]` (Gradio
  validation cached invalid paths, wedging the upload widget)

### Fixed
- Windows file-lock retry/rename/timestamped-fallback for output
  mp4 unlink contention with Gradio's video player
- Maskout SAM 2 worker `WinError 267` (path resolution)
- Fine-tune path mismatch when two LatentSync clones existed
- Diffusers API drift on `enable_gradient_checkpointing` (newer
  diffusers passes `enable=True` kwarg the LatentSync override
  doesn't accept)
- `MaskOutConfig` forward-reference NameError on launch
- Video preview "Video not playable" race between cv2.VideoCapture
  and Gradio's HTML5 player
