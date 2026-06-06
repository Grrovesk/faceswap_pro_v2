"""faceswap v2 -- typed, modular lipsync pipeline.

New in v2.1:
  * Live thumbnail / waveform previews on upload
  * Live render-time ETA estimate (refines after each render)
  * Save/load named presets persisted to v2/presets/*.json
  * Render history tab with sidecar JSON per past render
  * Click-to-replay on any past render
"""
from .config import LatentSyncKnobs, LipsyncJob, VoiceSwap
from . import history, orchestrator, presets, previews

__all__ = [
    "LatentSyncKnobs", "LipsyncJob", "VoiceSwap",
    "orchestrator", "presets", "previews", "history",
]
