"""
core/lipsync_worker.py -- in-process LatentSync diffusion worker.

Holds the LatentSync pipeline as a module-level singleton so multiple
renders in the same Python process share one warm copy of the UNet,
VAE, Whisper encoder, and scheduler. This eliminates the 60-120 s
model-reload cost that the legacy implementation paid every time it
subprocess-launched `python -m scripts.inference`.

Layout assumed on disk:

    faceswap_pro/
      lipsync_test/
        LatentSync/                  cloned upstream repo (added to sys.path)
          configs/                   scheduler + unet yaml configs
          checkpoints/
            latentsync_unet.pt       inference UNet checkpoint
            whisper/                 small.pt / tiny.pt
          scripts/inference.py       reference implementation we mirror
      v2/core/lipsync_worker.py      <-- this file

Design notes:
    * Module import is SAFE even if the LatentSync repo or checkpoints
      are missing -- we only touch the filesystem and import LatentSync
      modules inside _load_pipeline(), which is called lazily by
      render(). This lets the orchestrator import this module on
      startup without paying any GPU / disk cost.
    * The first render() call builds the pipeline and caches it in
      _PIPELINE. Subsequent calls reuse it.
    * LatentSync's inference.py uses several relative paths
      ("configs", "checkpoints/whisper/...", config.data.mask_image_path)
      so we chdir into the LatentSync repo for both load and call,
      restoring the previous cwd in a finally block.
    * fp16 is enabled when the device compute capability is > 7
      (matches the upstream reference script).

Public surface:
    render(face_path, audio_path, out_path, *,
           inference_steps=20, guidance_scale=1.5,
           enable_deepcache=True, seed=-1) -> out_path
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Optional

# ---------------------------------------------------------------------------
# Paths -- resolved at import time but NOT validated until first render().
# ---------------------------------------------------------------------------
_THIS_FILE   = os.path.abspath(__file__)
_V2_DIR      = os.path.dirname(os.path.dirname(_THIS_FILE))
_PROJECT_ROOT = os.path.dirname(_V2_DIR)
_LATENTSYNC_DIR = os.path.join(_PROJECT_ROOT, "lipsync_test", "LatentSync")
_UNET_CONFIG_PATH = os.path.join(_LATENTSYNC_DIR, "configs", "unet", "stage2.yaml")
_INFERENCE_CKPT_PATH = os.path.join(_LATENTSYNC_DIR, "checkpoints", "latentsync_unet.pt")

# ---------------------------------------------------------------------------
# Singleton state.
# ---------------------------------------------------------------------------
_PIPELINE = None          # LipsyncPipeline (warm, on CUDA)
_PIPELINE_DTYPE = None    # torch.float16 or torch.float32
_PIPELINE_CONFIG = None   # OmegaConf object for the chosen UNet yaml
_DEEPCACHE_HELPER = None  # DeepCacheSDHelper (None if disabled)
_DEEPCACHE_ENABLED = None # bool the helper was built with
_LOAD_LOCK = threading.Lock()


def _ensure_repo_on_path() -> None:
    """Add the LatentSync repo to sys.path so its `latentsync` package
    is importable. Idempotent."""
    if not os.path.isdir(_LATENTSYNC_DIR):
        raise RuntimeError(
            f"LatentSync repo not found at '{_LATENTSYNC_DIR}'. "
            "Clone https://github.com/bytedance/LatentSync into "
            "faceswap_pro/lipsync_test/LatentSync first."
        )
    if _LATENTSYNC_DIR not in sys.path:
        sys.path.insert(0, _LATENTSYNC_DIR)


def _pick_unet_config_path() -> str:
    """Return a UNet yaml that exists on disk. Prefers stage2.yaml,
    falls back to the first .yaml in configs/unet/."""
    if os.path.isfile(_UNET_CONFIG_PATH):
        return _UNET_CONFIG_PATH
    unet_cfg_dir = os.path.join(_LATENTSYNC_DIR, "configs", "unet")
    if os.path.isdir(unet_cfg_dir):
        for name in sorted(os.listdir(unet_cfg_dir)):
            if name.endswith(".yaml") and not name.startswith("_"):
                return os.path.join(unet_cfg_dir, name)
    raise RuntimeError(
        f"No UNet yaml config found under '{unet_cfg_dir}'."
    )


def _load_pipeline(enable_deepcache: bool) -> None:
    """One-time pipeline build. Mirrors LatentSync's inference.main()
    but caches every heavy object in module globals.

    Safe to call multiple times -- subsequent calls are no-ops unless
    the DeepCache toggle changed (in which case we rebuild the helper).
    """
    global _PIPELINE, _PIPELINE_DTYPE, _PIPELINE_CONFIG
    global _DEEPCACHE_HELPER, _DEEPCACHE_ENABLED

    with _LOAD_LOCK:
        if _PIPELINE is not None:
            # Pipeline already warm -- only adjust DeepCache if requested.
            if enable_deepcache != _DEEPCACHE_ENABLED:
                _toggle_deepcache(enable_deepcache)
            return

        _ensure_repo_on_path()

        # Imports are deferred so a missing checkpoint dir doesn't
        # break `import lipsync_worker`.
        import torch
        from omegaconf import OmegaConf
        from diffusers import AutoencoderKL, DDIMScheduler
        from latentsync.models.unet import UNet3DConditionModel
        from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        from latentsync.whisper.audio2feature import Audio2Feature

        if not os.path.isfile(_INFERENCE_CKPT_PATH):
            raise RuntimeError(
                f"LatentSync UNet checkpoint missing at "
                f"'{_INFERENCE_CKPT_PATH}'."
            )

        unet_yaml = _pick_unet_config_path()
        config = OmegaConf.load(unet_yaml)

        # Pick precision the same way upstream does.
        is_fp16_supported = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] > 7
        )
        dtype = torch.float16 if is_fp16_supported else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # LatentSync uses several repo-relative paths internally
        # ("configs", "checkpoints/whisper/...", mask images). Run the
        # whole load inside the repo dir, restore cwd afterwards.
        prev_cwd = os.getcwd()
        try:
            os.chdir(_LATENTSYNC_DIR)

            scheduler = DDIMScheduler.from_pretrained("configs")

            cad = int(config.model.cross_attention_dim)
            if cad == 768:
                whisper_model_path = "checkpoints/whisper/small.pt"
            elif cad == 384:
                whisper_model_path = "checkpoints/whisper/tiny.pt"
            else:
                raise NotImplementedError(
                    "cross_attention_dim must be 768 or 384"
                )

            audio_encoder = Audio2Feature(
                model_path=whisper_model_path,
                device=device,
                num_frames=config.data.num_frames,
                audio_feat_length=config.data.audio_feat_length,
            )

            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse", torch_dtype=dtype
            )
            vae.config.scaling_factor = 0.18215
            vae.config.shift_factor = 0

            unet, _meta = UNet3DConditionModel.from_pretrained(
                OmegaConf.to_container(config.model),
                _INFERENCE_CKPT_PATH,
                device="cpu",
            )
            unet = unet.to(dtype=dtype)

            pipeline = LipsyncPipeline(
                vae=vae,
                audio_encoder=audio_encoder,
                unet=unet,
                scheduler=scheduler,
            ).to(device)
        finally:
            os.chdir(prev_cwd)

        _PIPELINE = pipeline
        _PIPELINE_DTYPE = dtype
        _PIPELINE_CONFIG = config
        _DEEPCACHE_HELPER = None
        _DEEPCACHE_ENABLED = False

        if enable_deepcache:
            _toggle_deepcache(True)


def _toggle_deepcache(enable: bool) -> None:
    """Attach or detach a DeepCacheSDHelper on the cached pipeline."""
    global _DEEPCACHE_HELPER, _DEEPCACHE_ENABLED
    if _PIPELINE is None:
        return
    if enable:
        from DeepCache import DeepCacheSDHelper
        helper = DeepCacheSDHelper(pipe=_PIPELINE)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()
        _DEEPCACHE_HELPER = helper
        _DEEPCACHE_ENABLED = True
    else:
        if _DEEPCACHE_HELPER is not None:
            try:
                _DEEPCACHE_HELPER.disable()
            except Exception:
                pass
        _DEEPCACHE_HELPER = None
        _DEEPCACHE_ENABLED = False


def is_loaded() -> bool:
    """True iff the pipeline is currently warm in memory."""
    return _PIPELINE is not None


def render(
    face_path: str,
    audio_path: str,
    out_path: str,
    *,
    inference_steps: int = 20,
    guidance_scale: float = 1.5,
    enable_deepcache: bool = True,
    seed: int = -1,
    temp_dir: Optional[str] = None,
) -> str:
    """Run one LatentSync render against the cached pipeline.

    Parameters
    ----------
    face_path : str
        Path to the input face video (mp4 etc.).
    audio_path : str
        Path to the driving audio (wav/mp3).
    out_path : str
        Where to write the rendered mp4. Parent dirs are created.
    inference_steps : int
        DDIM steps. 20 is the upstream default.
    guidance_scale : float
        Classifier-free guidance scale. Upstream uses 1.0; we default
        to 1.5 (a stronger lip-sync signal that's still stable).
    enable_deepcache : bool
        If True (default) attach DeepCacheSDHelper for ~2x speedup.
    seed : int
        -1 means re-roll a fresh seed every call (upstream behaviour);
        any other value is forwarded to accelerate.set_seed().
    temp_dir : str | None
        Scratch dir for intermediates. Defaults to a 'temp' folder
        beside out_path.

    Returns
    -------
    out_path : str
        The same path that was written.
    """
    if not os.path.isfile(face_path):
        raise RuntimeError(f"Video path '{face_path}' not found")
    if not os.path.isfile(audio_path):
        raise RuntimeError(f"Audio path '{audio_path}' not found")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    if temp_dir is None:
        temp_dir = os.path.join(
            os.path.dirname(os.path.abspath(out_path)) or ".", "temp"
        )
    os.makedirs(temp_dir, exist_ok=True)

    _load_pipeline(enable_deepcache=enable_deepcache)

    # Re-sync DeepCache toggle in case caller flipped it after warmup.
    if enable_deepcache != _DEEPCACHE_ENABLED:
        _toggle_deepcache(enable_deepcache)

    import torch
    from accelerate.utils import set_seed

    if seed != -1:
        set_seed(int(seed))
    else:
        torch.seed()

    config = _PIPELINE_CONFIG
    dtype = _PIPELINE_DTYPE

    # DeepCache's forward-hook cache accumulates stale state across
    # renders -> 2nd render fails with "stack expects a non-empty
    # TensorList". Force-recycle the helper before each render so the
    # hooks are fresh.
    if enable_deepcache:
        _toggle_deepcache(False)
        _toggle_deepcache(True)

    prev_cwd = os.getcwd()
    try:
        os.chdir(_LATENTSYNC_DIR)
        _PIPELINE(
            video_path=face_path,
            audio_path=audio_path,
            video_out_path=out_path,
            num_frames=config.data.num_frames,
            num_inference_steps=int(inference_steps),
            guidance_scale=float(guidance_scale),
            weight_dtype=dtype,
            width=config.data.resolution,
            height=config.data.resolution,
            mask_image_path=config.data.mask_image_path,
            temp_dir=temp_dir,
        )
    finally:
        os.chdir(prev_cwd)

    return out_path
