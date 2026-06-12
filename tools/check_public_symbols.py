"""Public-symbol regression check.

Auto-generated baseline as of 2026-06-11.  The EXPECTED dict below
captures every public symbol that existed in this tree's production
modules at baseline.  Any missing symbol on a future run means a
module was truncated, edited destructively, or had a symbol renamed
without updating this baseline.

When you intentionally rename or remove a public symbol, update the
EXPECTED entry to match -- this checker surfaces unintended changes,
not policy.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


EXPECTED = {
    "core/frame_packet.py": {
        "FramePacket",
        "FramePacket.face_found",
        "FramePacket.face_region",
        "FramePacket.lip_indices",
    },
    "core/gpu_dll_bootstrap.py": {
        "add_gpu_dll_dirs",
        "logger",
        "preload_cuda_cudnn_dlls",
    },
    "core/lipsync.py": {
        "CKPT_DIR",
        "D2L",
        "D2L_CKPT",
        "DIFF2LIP_CKPT_URL",
        "DIFF2LIP_REPO",
        "GFPGAN_MODEL",
        "GFPGAN_MODEL_URL",
        "GFPGAN_TORCH_COMPILE",
        "LATENTSYNC_HF_REPO",
        "LATENTSYNC_REPO",
        "LATENTSYNC_ZIP_URL",
        "LS",
        "LS_CONFIG",
        "LS_MIN_UNET_BYTES",
        "LS_UNET_CKPT",
        "LS_WHISPER_CKPT",
        "MIN_MODEL_BYTES",
        "MODEL_URLS",
        "OUT_DIR",
        "PROJECT_ROOT",
        "RENDER_DIR",
        "W2L",
        "WAV2LIP_REPO",
        "WORK",
        "checkpoint_path",
        "lipsync_assets_status",
        "render_lipsync",
    },
    "core/lipsync_color_match.py": {
        "PROJECT_ROOT",
        "color_match_video",
        "logger",
    },
    "core/lipsync_finetune.py": {
        "FINETUNE_ROOT",
        "PROJECT_ROOT",
        "TRAIN_AUDIO_HZ",
        "TRAIN_FPS",
        "TRAIN_RESOLUTION",
        "get_finetune_checkpoint",
        "get_finetune_dir",
        "prepare_clip_for_training",
        "train_identity_finetune",
    },
    "core/maskout_pipeline.py": {
        "SAM2_WORKER",
        "composite_back",
        "make_void_source",
        "run_pipeline",
        "run_sam2_masks",
    },
    "core/onnx_opt.py": {
        "create_ort_session",
        "logger",
        "optimize_onnx_graph",
        "run_batched",
    },
    "core/pipeline.py": {
        "DEFAULT_FACE_ANALYSIS_PACK",
        "FACE_ANALYSIS_ROOT",
        "FALLBACK_FACE_ANALYSIS_PACKS",
        "FaceSwapPipeline",
        "FaceSwapPipeline.preview_frame",
        "FaceSwapPipeline.run",
        "PROJECT_ROOT",
        "logger",
    },
    "core/pipeline_blend.py": {
        "blend_embeddings",
        "extract_embedding",
        "logger",
        "make_journey_fn",
        "run_blend",
        "run_journey",
    },
    "core/rotoscope_cache.py": {
        "CACHE_ROOT",
        "FrameCacheInfo",
        "compute_video_hash",
        "extract_and_cache",
        "frame_path",
        "load_cache_info",
        "logger",
        "mask_dir",
    },
    "core/sam2_daemon.py": {
        "DEFAULT_CLICK_TIMEOUT_S",
        "DEFAULT_LOAD_TIMEOUT_S",
        "DEFAULT_PROPAGATE_TIMEOUT_S",
        "DEFAULT_VIDEO_LOAD_TIMEOUT_S",
        "SAM2Daemon",
        "SAM2Daemon.apply_click",
        "SAM2Daemon.clear",
        "SAM2Daemon.click",
        "SAM2Daemon.is_running",
        "SAM2Daemon.load_video",
        "SAM2Daemon.ping",
        "SAM2Daemon.propagate",
        "SAM2Daemon.set_all_prompts",
        "SAM2Daemon.set_prompts",
        "SAM2Daemon.shutdown",
        "SAM2Daemon.singleton",
        "SAM2Daemon.start",
        "SAM2DaemonError",
        "get_or_start_daemon",
        "logger",
    },
    "core/sam2_install.py": {
        "PROJECT_ROOT",
        "SAM2_CKPT",
        "SAM2_CKPT_DIR",
        "SAM2_CKPT_NAME",
        "SAM2_CONFIG_BY_CKPT",
        "SAM2_DEFAULT_CONFIG",
        "ensure_sam2_weights",
        "is_sam2_importable",
        "keysync_ckpt",
        "keysync_python",
        "pip_install_hint",
    },
    "core/swap_backends/base.py": {
        "BackendInfo",
        "SwapBackend",
    },
    "core/swap_backends/inswapper.py": {
        "InswapperBackend",
    },
    "core/swap_engine.py": {
        "SwapAlignedResult",
        "SwapEngine",
        "SwapEngine.cleanup",
        "SwapEngine.initialize",
        "SwapEngine.paste_back",
        "SwapEngine.paste_back_roi",
        "SwapEngine.reset_stats",
        "SwapEngine.set_source_embedding",
        "SwapEngine.stats",
        "SwapEngine.swap",
        "SwapEngine.swap_aligned",
        "SwapEngine.swap_batch",
        "SwapEngine.swap_paste_back",
        "SwapResult",
        "logger",
    },
    "core/voice_clone.py": {
        "EXTERNAL_REPOS_ROOT",
        "OUT_DIR",
        "PROJECT_ROOT",
        "RVC",
        "RVC_WEIGHTS",
        "STEMS",
        "VOICE_MODELS_DIR",
        "WORK",
        "list_voice_models",
        "rvc_convert_song",
    },
    "core/xseg_gate.py": {
        "MASKS_ROOT",
        "PROJECT_ROOT",
        "XSEG_DIR",
        "XSEG_INPUT_SIZE",
        "XSEG_MODELS",
        "XSegEnsemble",
        "XSegEnsemble.occluder_mask",
        "build_occluder_masks_video",
        "ensure_xseg_models",
        "gate_lipsync_with_xseg",
        "restore_occluder",
    },
    "faceswap/config.py": {
        "AspectRatioConfig",
        "LatentSyncKnobs",
        "LipsyncJob",
        "LipsyncJob.is_multi_clip",
        "LipsyncJob.is_single_clip",
        "LipsyncJob.validate",
        "MaskOutConfig",
        "OcclusionConfig",
        "VideoSwapJob",
        "VideoSwapJob.validate",
        "VoiceSwap",
        "WatermarkConfig",
    },
    "faceswap/ffmpeg_tools.py": {
        "concat_videos",
        "loop_video_to_duration",
        "probe_duration_seconds",
        "replace_audio_track",
        "replace_audio_with_stems_mix",
        "resolve_ffmpeg",
        "slice_audio_to_wav",
    },
    "faceswap/gfpgan.py": {
        "enhance",
    },
    "faceswap/history.py": {
        "list_renders",
        "list_renders_for_dropdown",
    },
    "faceswap/job_queue.py": {
        "Job",
        "Job.elapsed_s",
        "JobQueue",
        "JobQueue.cancel",
        "JobQueue.clear_completed",
        "JobQueue.get",
        "JobQueue.list_all",
        "JobQueue.stop",
        "JobQueue.submit",
        "get_queue",
        "jobs_as_rows",
        "logger",
    },
    "faceswap/latentsync.py": {
        "run",
    },
    "faceswap/orchestrator.py": {
        "RenderCancelled",
        "render",
    },
    "faceswap/paths.py": {
        "EXTEND_SINGLE_WORK",
        "EXTERNAL_REPOS_ROOT",
        "LATENTSYNC_REPO_DIR",
        "LATENTSYNC_SCRATCH",
        "MULTICLIP_WORK",
        "PROJECT_ROOT",
        "RECORDINGS_DIR",
        "RVC_REPO_DIR",
        "WEBCAM_RECORDINGS_DIR",
        "WORK_DIR",
        "ensure_all",
        "safe_clear_output",
    },
    "faceswap/post_process.py": {
        "add_timecode_overlay",
        "apply_aspect_ratio",
        "apply_watermark",
        "list_aspects",
        "logger",
        "preview_output_frame",
    },
    "faceswap/presets.py": {
        "PRESETS_DIR",
        "delete_preset",
        "list_presets",
        "load_preset",
        "save_preset",
    },
    "faceswap/previews.py": {
        "estimate_render_seconds",
        "extract_audio_waveform",
        "extract_first_frame",
        "load_sidecar",
        "record_actual_s_per_frame",
        "sidecar_path_for",
        "write_sidecar",
        "write_video_swap_sidecar",
    },
    "faceswap/projects.py": {
        "PROJECTS_DIR",
        "delete_project",
        "list_projects",
        "load_project",
        "project_dir",
        "save_project",
    },
    "faceswap/rotoscope/ui.py": {
        "REPLACE_RADIUS",
        "build_tab",
        "logger",
    },
    "faceswap/ui.py": {
        "build",
    },
    "faceswap/video_swap.py": {
        "OUT_DIR",
        "preview_one_frame",
        "run",
    },
    "faceswap/vocal_isolation.py": {
        "isolate",
    },
    "faceswap/voice_swap.py": {
        "apply_voice_swap",
        "list_available_voices",
    },
    "faceswap/webcam/filters.py": {
        "apply_filters",
    },
    "faceswap/webcam/state.py": {
        "WebcamState",
        "get_snapshot",
        "set_options",
        "set_source",
    },
    "faceswap/webcam/streaming.py": {
        "logger",
        "mjpeg_generator",
        "register_stream_routes",
    },
    "faceswap/webcam/swap_fn.py": {
        "logger",
        "make_swap_fn",
    },
    "faceswap/webcam/ui.py": {
        "build_tab",
    },
    "faceswap/webcam/virtual_cam.py": {
        "close_cam",
        "is_open",
        "logger",
        "open_cam",
        "send_frame",
        "status",
    },
    "faceswap/webcam/worker.py": {
        "SwapStreamWorker",
        "SwapStreamWorker.get_latest_jpeg",
        "SwapStreamWorker.is_recording",
        "SwapStreamWorker.running",
        "SwapStreamWorker.start",
        "SwapStreamWorker.start_recording",
        "SwapStreamWorker.stop",
        "SwapStreamWorker.stop_recording",
        "get_worker",
        "logger",
    },
}


def _public_symbols(path):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            names.add(node.name)
            for sub in node.body:
                if not isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if sub.name == "__init__":
                    continue
                if sub.name.startswith("_"):
                    continue
                names.add(node.name + "." + sub.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and not tgt.id.startswith("_"):
                    names.add(tgt.id)
        elif isinstance(node, ast.AnnAssign):
            tgt = node.target
            if isinstance(tgt, ast.Name) and not tgt.id.startswith("_"):
                names.add(tgt.id)
    return names


def main():
    missing = {}
    for rel, expected in EXPECTED.items():
        path = PROJECT_ROOT / rel
        if not path.is_file():
            missing[rel] = {"FILE MISSING: " + str(path)}
            continue
        actual = _public_symbols(path)
        gone = expected - actual
        if gone:
            missing[rel] = gone
    if missing:
        print("PUBLIC SYMBOL REGRESSION DETECTED:", file=sys.stderr)
        for rel, names in missing.items():
            print("  " + rel + ": missing " + str(sorted(names)),
                  file=sys.stderr)
        return 1
    total_symbols = sum(len(v) for v in EXPECTED.values())
    print("OK -- all " + str(total_symbols) + " public symbols across "
          + str(len(EXPECTED)) + " modules present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
