"""Gradio binding layer. Single responsibility: build the widget tree
and translate the click event into a typed LipsyncJob, then hand it
to the orchestrator. Zero business logic lives here.

v2.2 layout:
  - Lip-Sync tab is a real 3-column grid: Inputs | Pipeline | Output
  - Presets and Cache promoted from accordions to top-level tabs
  - History tab unchanged (was already useful)
  - About tab added (scope, what's in, what's out)
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Optional

import gradio as gr

from . import history, job_queue, orchestrator, presets, previews, video_swap
from .config import (KeySyncKnobs, LatentSyncKnobs, LipsyncJob,
                       MaskOutConfig,
                       VideoSwapJob, VoiceSwap, WatermarkConfig,
                       AspectRatioConfig, OcclusionConfig)
from .post_process import list_aspects as _list_aspects
from .voice_swap import list_available_voices

_VOICE_NONE = "(no voice swap)"

# Module-level cancel signal for the in-flight render. Cleared at the
# start of each _run_render, set by the Cancel button handler. The
# orchestrator polls this at stage boundaries.
import threading as _th
_RENDER_CANCEL = _th.Event()


# ============================================================
# render handler
# ============================================================
def _run_render(face, audio, face_extras,
                 engine_choice,
                 isolate, quick, enhance, extend_single,
                 ls_steps, ls_guidance, ls_deepcache, ls_seed, ls_color_match,
                 ls_face_det,
                 voice_model, voice_transpose,
                 wm_enabled, wm_image, wm_position, wm_scale, wm_opacity,
                 ar_enabled, ar_target, ar_fill,
                 occ_enabled, occ_bbox_smooth, occ_mask_smooth,
                 occ_align, occ_feather, occ_mouth_polygon,
                 ks_face_click_x, ks_face_click_y, ks_face_click_frame,
                 ks_skip_crop,
                 mout_enabled_v, mout_clicks_v,
                 mout_dilate_v, mout_feather_v,
                 mout_npy_override_v):
    """Generator. Yields (video_path, status_md, history_update) every
    ~1.5s while the render is running so the UI shows live progress
    instead of a silent spinner. Final yield carries the result."""
    if not face:
        yield None, "**Error:** no face clip selected.", gr.update(); return
    if not audio:
        yield None, "**Error:** no audio selected.", gr.update(); return

    face_paths = [Path(face)]
    if face_extras:
        for x in face_extras:
            p = getattr(x, "name", x)
            if p and Path(str(p)).is_file():
                face_paths.append(Path(str(p)))

    job = LipsyncJob(
        face_paths=face_paths, audio_path=Path(audio),
        engine=str(engine_choice or "latentsync").lower(),
        isolate_vocals=bool(isolate), enhance_faces=bool(enhance),
        quick_test=bool(quick), extend_single=bool(extend_single),
        latentsync=LatentSyncKnobs(
            inference_steps=int(ls_steps),
            guidance_scale=float(ls_guidance),
            enable_deepcache=bool(ls_deepcache),
            seed=int(ls_seed),
            face_det_threshold=float(ls_face_det),
            color_match_mode=str(ls_color_match or "reinhard"),
        ),
        keysync=KeySyncKnobs(
            face_click_x=int(ks_face_click_x or 0),
            face_click_y=int(ks_face_click_y or 0),
            face_click_frame=int(ks_face_click_frame or 0),
            skip_crop=bool(ks_skip_crop),
        ),
        maskout=MaskOutConfig(
            enabled=bool(mout_enabled_v),
            clicks=list(mout_clicks_v or []),
            dilate_px=int(mout_dilate_v),
            feather=int(mout_feather_v),
            mask_npy_path=str(mout_npy_override_v or ""),
        ),
        voice_swap=VoiceSwap(
            model_basename=("" if (not voice_model or
                                    voice_model == _VOICE_NONE)
                            else str(voice_model)),
            transpose_semitones=int(voice_transpose),
        ),
        watermark=WatermarkConfig(
            enabled=bool(wm_enabled),
            image_path=(getattr(wm_image, "name", wm_image)
                        if wm_image else ""),
            position=str(wm_position),
            scale_pct=float(wm_scale),
            opacity=float(wm_opacity),
        ),
        aspect=AspectRatioConfig(
            enabled=bool(ar_enabled),
            target_aspect=str(ar_target),
            fill_mode=str(ar_fill),
        ),
        occlusion=OcclusionConfig(
            enabled=bool(occ_enabled),
            bbox_smoothing=float(occ_bbox_smooth),
            mask_smoothing=float(occ_mask_smooth),
            align_to_source=bool(occ_align),
            feather=int(occ_feather),
            mouth_polygon=bool(occ_mouth_polygon),
        ),
    )

    import threading
    log_lines = []
    log_lock = threading.Lock()
    def log(msg):
        s = str(msg)
        with log_lock:
            log_lines.append(s)
        print(s, flush=True)

    # Clear any prior cancel state before starting this render
    _RENDER_CANCEL.clear()

    result = {"path": None, "exc": None, "tb": None, "cancelled": False}
    done_evt = threading.Event()

    def worker():
        try:
            result["path"] = orchestrator.render(
                job, log=log, cancel_event=_RENDER_CANCEL)
        except orchestrator.RenderCancelled:
            result["cancelled"] = True
        except Exception as exc:
            result["exc"] = exc
            result["tb"] = traceback.format_exc()
        finally:
            done_evt.set()

    t0 = time.time()
    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # POLL LOOP: yield live progress every 1.5s while render runs
    yield None, "**Starting render...**", gr.update()
    while not done_evt.is_set():
        with log_lock:
            tail = "\n".join(log_lines[-25:])
        elapsed = time.time() - t0
        status = (f"**Rendering... ({elapsed:.0f}s elapsed)**\n\n"
                  f"```\n{tail or '(initializing)'}\n```")
        yield None, status, gr.update()
        done_evt.wait(timeout=1.5)

    # RESULT
    if result.get("cancelled"):
        with log_lock:
            tail = "\n".join(log_lines[-30:])
        yield None, (f"**Render cancelled by user.**\n\n```\n"
                     + tail + "\n```"), gr.update()
        return
    if result["exc"] is not None:
        yield None, (f"**Render failed:** {result['exc']}\n\n```\n"
                     + (result["tb"] or "")[-2000:]
                     + "\n```"), gr.update()
        return

    out_path = result["path"]
    elapsed = time.time() - t0
    try: size_mb = Path(out_path).stat().st_size / (1024 * 1024)
    except OSError: size_mb = 0
    eta = previews.estimate_render_seconds(
        job.face_paths, job.effective_audio or job.audio_path,
        job.extend_single, job.enhance_faces, job.quick_test)
    frames = max(eta["frames"], 1)
    if elapsed > 5:
        previews.record_actual_s_per_frame(elapsed / frames)
    try:
        previews.write_sidecar(job, Path(out_path),
                               elapsed_s=elapsed, frames=frames)
    except Exception as exc:
        log(f"[sidecar] WARN: {exc}")

    with log_lock:
        full_log = "\n".join(log_lines[-120:])
    status = (
        f"**Done** -- `{Path(out_path).name}`  \n"
        f"Time: **{elapsed:.1f}s**  Size: **{size_mb:.1f} MB**  \n"
        f"\n<details open><summary>full log</summary>\n\n```\n"
        + full_log + "\n```\n</details>"
    )
    new_hist = gr.update(choices=history.list_renders_for_dropdown(),
                         value=str(out_path))
    yield str(out_path), status, new_hist
def _on_render_cancel():
    """Signal the in-flight render to stop at the next stage boundary.
    Does NOT kill in-progress LatentSync inference (PyTorch CUDA ops
    can't be cleanly interrupted from outside). User sees 'Cancelling'
    until the current stage finishes, then 'Render cancelled.'"""
    _RENDER_CANCEL.set()
    return "**Cancelling...** (will stop at next stage boundary)"


# ============================================================
# preview / ETA handlers
# ============================================================
def _on_face_change(face, audio, face_extras,
                     extend_single, enhance, quick):
    # face_thumb widget removed -- gr.Video shows its own preview.
    # This handler now only refreshes the ETA on face-clip change.
    return _format_eta(face, audio, face_extras, extend_single,
                       enhance, quick)


def _on_audio_change(audio, face, face_extras,
                      extend_single, enhance, quick):
    wave = previews.extract_audio_waveform(Path(audio)) if audio else None
    return (str(wave) if wave else None,
            _format_eta(face, audio, face_extras, extend_single,
                        enhance, quick))


def _on_options_change(face, audio, face_extras,
                        extend_single, enhance, quick):
    return _format_eta(face, audio, face_extras, extend_single,
                       enhance, quick)


def _format_eta(face, audio, face_extras, extend_single, enhance, quick):
    fps = []
    if face: fps.append(Path(face))
    if face_extras:
        for x in face_extras:
            p = getattr(x, "name", x)
            if p and Path(str(p)).is_file():
                fps.append(Path(str(p)))
    eta = previews.estimate_render_seconds(
        fps, Path(audio) if audio else None,
        bool(extend_single), bool(enhance), bool(quick))
    return f"**ETA:** {eta['label']}"


# ============================================================
# preset handlers
# ============================================================
def _on_save_preset(name, isolate, quick, enhance, extend_single,
                     ls_steps, ls_guidance, ls_deepcache, ls_seed, ls_color_match,
                 ls_face_det,
                     voice_model, voice_transpose):
    if not name or not name.strip():
        return ("**Error:** preset name required.",
                gr.update(choices=presets.list_presets()))
    out = presets.save_preset(
        name, isolate, quick, enhance, extend_single,
        ls_steps, ls_guidance, ls_deepcache, ls_seed, ls_color_match,
        voice_model, voice_transpose)
    return (f"saved preset `{Path(out).stem}`",
            gr.update(choices=presets.list_presets(),
                      value=presets._safe_name(name)))


def _on_load_preset(name):
    p = presets.load_preset(name) if name else None
    if not p:
        return [gr.update()] * 10 + [f"preset `{name}` not found"]
    ls = p.get("latentsync", {}); vs = p.get("voice_swap", {})
    voice_value = vs.get("model_basename") or _VOICE_NONE
    return [
        gr.update(value=bool(p.get("isolate_vocals", True))),
        gr.update(value=bool(p.get("quick_test", False))),
        gr.update(value=bool(p.get("enhance_faces", True))),
        gr.update(value=bool(p.get("extend_single", False))),
        gr.update(value=int(ls.get("inference_steps", 20))),
        gr.update(value=float(ls.get("guidance_scale", 1.5))),
        gr.update(value=bool(ls.get("enable_deepcache", True))),
        gr.update(value=int(ls.get("seed", -1))),
        gr.update(value=voice_value),
        gr.update(value=int(vs.get("transpose_semitones", 0))),
        f"loaded preset `{name}`",
    ]


def _on_delete_preset(name):
    if not name: return ("nothing selected", gr.update())
    ok = presets.delete_preset(name)
    return (("deleted" if ok else "delete failed"),
            gr.update(choices=presets.list_presets(), value=None))


# ============================================================
# history handlers
# ============================================================
def _refresh_history():
    return gr.update(choices=history.list_renders_for_dropdown(),
                     value=None)


def _on_history_select(path):
    if not path: return None, "_(none selected)_"
    sc = previews.load_sidecar(Path(path))
    if sc:
        import json as _j
        md = ("**Sidecar** for `" + Path(path).name + "`:\n\n```json\n"
              + _j.dumps(sc, indent=2) + "\n```")
    else:
        md = f"_(no sidecar for {Path(path).name})_"
    return path, md


# ============================================================
# cache cleaner
# ============================================================
def _clear_gradio_cache(min_age_min):
    import tempfile
    g = Path(tempfile.gettempdir()) / "gradio"
    if not g.exists(): return f"no gradio cache at {g}"
    cutoff = time.time() - (int(min_age_min) * 60)
    removed, freed = 0, 0
    for f in g.rglob("*"):
        if f.is_file():
            try:
                if f.stat().st_mtime < cutoff:
                    sz = f.stat().st_size; f.unlink()
                    removed += 1; freed += sz
            except OSError: pass
    for d in sorted([p for p in g.rglob("*") if p.is_dir()],
                    key=lambda p: -len(str(p))):
        try: d.rmdir()
        except OSError: pass
    return (f"cleared {removed} files ({freed/(1024*1024):.1f} MB) "
            f"older than {int(min_age_min)} min from {g}")


# ============================================================
# build
# ============================================================

# ============================================================
# face-swap handler (Tab 2: Face Swap)
# ============================================================
def _run_video_swap(source_img, target_vid,
                     blend_method, enhance_faces,
                     det_threshold,
                     output_quality, trim_start, trim_end,
                     source_img_b, blend_alpha,
                     journey_mode, journey_start_alpha,
                     journey_end_alpha, journey_curve,
                     mask_padding, mask_blur,
                     swap_strength, enhancer_blend,
                     selector_mode, reference_face_img,
                     reference_distance, pixel_boost,
                     temporal_enabled, temporal_ema_decay,
                     temporal_buffer_size,
                     color_transfer_mode, shadow_correction,
                     shadow_clamp_min, shadow_clamp_max,
                     face_restorer="gfpgan",
                     mask_npy_path=""):
    """Generator. Yields live progress while video_swap.run() executes
    in a daemon thread."""
    if not source_img:
        yield None, "**Error:** no source image."; return
    if not target_vid:
        yield None, "**Error:** no target video."; return

    # gr.File returns either a string path OR an object with .name.
    src_path = getattr(source_img, "name", source_img)
    if not src_path or not Path(str(src_path)).is_file():
        yield None, f"**Error:** source path invalid: {src_path}"; return

    # Optional source B for blend / journey modes
    src_path_b = getattr(source_img_b, "name", source_img_b) \
        if source_img_b else None
    if src_path_b and not Path(str(src_path_b)).is_file():
        src_path_b = None

    # Reference face path (optional; required only for "reference" mode).
    ref_path = getattr(reference_face_img, "name", reference_face_img) \
        if reference_face_img else None
    if ref_path and not Path(str(ref_path)).is_file():
        ref_path = None
    # If user picked "reference" but didn't upload a ref image, fall
    # back to legacy "largest" so we don't trip VideoSwapJob.validate().
    _selector_mode = str(selector_mode or "largest").lower()
    if _selector_mode == "reference" and not ref_path:
        _selector_mode = "largest"
    job = VideoSwapJob(
        source_image=Path(src_path),
        target_video=Path(target_vid),
        blend_method=str(blend_method),
        enhance_faces=bool(enhance_faces),
        det_threshold=float(det_threshold),
        output_quality=str(output_quality),
        trim_start_frame=int(trim_start),
        trim_end_frame=int(trim_end),
        source_image_b=(Path(src_path_b) if src_path_b else None),
        blend_alpha=float(blend_alpha),
        journey_mode=bool(journey_mode),
        journey_start_alpha=float(journey_start_alpha),
        journey_end_alpha=float(journey_end_alpha),
        journey_curve=str(journey_curve),
        mask_padding=int(mask_padding),
        mask_blur=float(mask_blur),
        swap_strength=float(swap_strength),
        enhancer_blend=float(enhancer_blend),
        selector_mode=_selector_mode,
        reference_face_image=(Path(ref_path) if ref_path else None),
        reference_distance=float(reference_distance),
        pixel_boost=int(pixel_boost),
        temporal_enabled=bool(temporal_enabled),
        temporal_ema_decay=float(temporal_ema_decay),
        temporal_buffer_size=int(temporal_buffer_size),
        color_transfer_mode=str(color_transfer_mode),
        shadow_correction=bool(shadow_correction),
        shadow_clamp_min=float(shadow_clamp_min),
        shadow_clamp_max=float(shadow_clamp_max),
        face_restorer=str(face_restorer or "gfpgan"),
        mask_npy_path=(str(mask_npy_path).strip() or None),
    )

    import threading
    log_lines = []
    log_lock = threading.Lock()
    def log(msg):
        s = str(msg)
        with log_lock: log_lines.append(s)
        print(s, flush=True)

    result = {"path": None, "exc": None, "tb": None}
    done_evt = threading.Event()

    def worker():
        try:
            result["path"] = video_swap.run(job, log=log)
        except Exception as exc:
            result["exc"] = exc
            result["tb"] = traceback.format_exc()
        finally:
            done_evt.set()

    t0 = time.time()
    threading.Thread(target=worker, daemon=True).start()

    yield None, "**Starting face swap...**"
    while not done_evt.is_set():
        with log_lock: tail = "\n".join(log_lines[-25:])
        elapsed = time.time() - t0
        yield None, (f"**Swapping... ({elapsed:.0f}s elapsed)**\n\n"
                     f"```\n{tail or '(initializing)'}\n```")
        done_evt.wait(timeout=1.5)

    if result["exc"] is not None:
        yield None, (f"**Face swap failed:** {result['exc']}\n\n```\n"
                     + (result["tb"] or "")[-2000:] + "\n```")
        return

    out_path = result["path"]
    elapsed = time.time() - t0
    try: size_mb = Path(out_path).stat().st_size / (1024 * 1024)
    except OSError: size_mb = 0
    with log_lock: full_log = "\n".join(log_lines[-120:])
    status = (
        f"**Done** -- `{Path(out_path).name}`  \n"
        f"Time: **{elapsed:.1f}s**  Size: **{size_mb:.1f} MB**  \n"
        f"\n<details open><summary>full log</summary>\n\n```\n"
        + full_log + "\n```\n</details>"
    )
    # T1-2: write the JSON sidecar so History tab "Restore settings"
    # can populate the Face Swap tab from this render's knobs.
    try:
        previews.write_video_swap_sidecar(job, Path(out_path), elapsed)
    except Exception as exc:
        print(f"[T1-2] sidecar write failed: {exc}", flush=True)

    yield str(out_path), status

# ============================================================
# Queue handlers (Tab: Queue)
# ============================================================
def _enqueue_render(face, audio, face_extras,
                     engine_choice,
                     isolate, quick, enhance, extend_single,
                     ls_steps, ls_guidance, ls_deepcache, ls_seed, ls_color_match,
                     ls_face_det,
                     voice_model, voice_transpose,
                     wm_enabled, wm_image, wm_position, wm_scale, wm_opacity,
                     ar_enabled, ar_target, ar_fill,
                     occ_enabled, occ_bbox_smooth, occ_mask_smooth,
                     occ_align, occ_feather, occ_mouth_polygon,
                     ks_face_click_x, ks_face_click_y, ks_face_click_frame,
                 ks_skip_crop,
                 mout_enabled_v, mout_clicks_v,
                 mout_dilate_v, mout_feather_v,
                 mout_npy_override_v):
    """Build a LipsyncJob from the current Lip-Sync tab settings and
    submit it to the queue. Returns a status string."""
    if not face:
        return "**Error:** no face clip selected."
    if not audio:
        return "**Error:** no audio selected."
    face_paths = [Path(face)]
    if face_extras:
        for x in face_extras:
            p = getattr(x, "name", x)
            if p and Path(str(p)).is_file():
                face_paths.append(Path(str(p)))
    lj = LipsyncJob(
        face_paths=face_paths, audio_path=Path(audio),
        engine=str(engine_choice or "latentsync").lower(),
        isolate_vocals=bool(isolate), enhance_faces=bool(enhance),
        quick_test=bool(quick), extend_single=bool(extend_single),
        latentsync=LatentSyncKnobs(
            inference_steps=int(ls_steps),
            guidance_scale=float(ls_guidance),
            enable_deepcache=bool(ls_deepcache),
            seed=int(ls_seed),
            face_det_threshold=float(ls_face_det),
            color_match_mode=str(ls_color_match or "reinhard")),
        keysync=KeySyncKnobs(
            face_click_x=int(ks_face_click_x or 0),
            face_click_y=int(ks_face_click_y or 0),
            face_click_frame=int(ks_face_click_frame or 0),
            skip_crop=bool(ks_skip_crop)),
        maskout=MaskOutConfig(
            enabled=bool(mout_enabled_v),
            clicks=list(mout_clicks_v or []),
            dilate_px=int(mout_dilate_v),
            feather=int(mout_feather_v),
            mask_npy_path=str(mout_npy_override_v or "")),
        voice_swap=VoiceSwap(
            model_basename=("" if (not voice_model
                                    or voice_model == _VOICE_NONE)
                            else str(voice_model)),
            transpose_semitones=int(voice_transpose)),
        watermark=WatermarkConfig(
            enabled=bool(wm_enabled),
            image_path=(getattr(wm_image, "name", wm_image)
                        if wm_image else ""),
            position=str(wm_position),
            scale_pct=float(wm_scale),
            opacity=float(wm_opacity)),
        occlusion=OcclusionConfig(
            enabled=bool(occ_enabled),
            bbox_smoothing=float(occ_bbox_smooth),
            mask_smoothing=float(occ_mask_smooth),
            align_to_source=bool(occ_align),
            feather=int(occ_feather),
            mouth_polygon=bool(occ_mouth_polygon)),
        aspect=AspectRatioConfig(
            enabled=bool(ar_enabled),
            target_aspect=str(ar_target),
            fill_mode=str(ar_fill)),
    )
    j = job_queue.get_queue().submit(lj)
    return (f"queued job `{j.id}` ({j.label}). Switch to the Queue "
            "tab to watch progress.")


def _refresh_queue():
    return job_queue.jobs_as_rows()


def _cancel_queue_job(job_id):
    if not job_id:
        return "(provide a job id)", job_queue.jobs_as_rows()
    ok = job_queue.get_queue().cancel(str(job_id).strip())
    msg = (f"cancelled `{job_id}`"
           if ok else f"could not cancel `{job_id}` (not queued?)")
    return msg, job_queue.jobs_as_rows()


def _preview_post_output(face, wm_enabled, wm_image, wm_position,
                          wm_scale, wm_opacity,
                          ar_enabled, ar_target, ar_fill):
    """Apply watermark + aspect to frame 0 of the face video and
    return the result as a PIL Image. Sub-second dry run."""
    if not face:
        return None
    try:
        from .post_process import preview_output_frame
        wcfg = WatermarkConfig(
            enabled=bool(wm_enabled),
            image_path=(getattr(wm_image, "name", wm_image)
                        if wm_image else ""),
            position=str(wm_position),
            scale_pct=float(wm_scale),
            opacity=float(wm_opacity))
        acfg = AspectRatioConfig(
            enabled=bool(ar_enabled),
            target_aspect=str(ar_target),
            fill_mode=str(ar_fill))
        return preview_output_frame(face, watermark_cfg=wcfg,
                                     aspect_cfg=acfg)
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def _clear_completed():
    n = job_queue.get_queue().clear_completed()
    return (f"cleared {n} completed/failed/cancelled jobs",
            job_queue.jobs_as_rows())


def build() -> gr.Blocks:
    voices = [_VOICE_NONE] + list_available_voices()
    preset_choices = presets.list_presets()
    hist_choices = history.list_renders_for_dropdown()

    with gr.Blocks(
        title="faceswap_pro v2",
        theme=gr.themes.Soft(),
        # Load html2canvas for the full-page screenshot button.
        head=("<script src='https://cdnjs.cloudflare.com/ajax/libs/"
              "html2canvas/1.4.1/html2canvas.min.js'></script>"),
        # On page load: force dark theme, install screenshot button.
        js="""
() => {
  try { document.querySelector('html').classList.add('dark'); } catch(e) {}
  if (document.getElementById('fp-screenshot-btn')) return;
  const btn = document.createElement('button');
  btn.id = 'fp-screenshot-btn';
  btn.innerHTML = '\\u{1F4F7} Capture full page';
  btn.title = 'Save a PNG of the entire scrollable UI';
  btn.style.cssText = (
    'position:fixed;bottom:12px;right:12px;z-index:10000;' +
    'background:rgba(40,40,40,0.92);color:#fff;border:1px solid #555;' +
    'padding:8px 14px;border-radius:6px;cursor:pointer;' +
    'font:13px system-ui,sans-serif;box-shadow:0 2px 8px rgba(0,0,0,0.3)'
  );
  btn.onmouseenter = () => { btn.style.background='rgba(70,70,70,0.95)'; };
  btn.onmouseleave = () => { btn.style.background='rgba(40,40,40,0.92)'; };
  btn.onclick = async () => {
    if (typeof html2canvas !== 'function') {
      alert('html2canvas failed to load (network?)'); return;
    }
    const orig = btn.innerHTML;
    btn.innerHTML = '\\u23F3 capturing...'; btn.disabled = true;
    try {
      btn.style.visibility = 'hidden';
      const canvas = await html2canvas(document.body, {
        useCORS: true,
        backgroundColor: getComputedStyle(document.body).backgroundColor,
        scale: Math.min(2, window.devicePixelRatio || 1),
        windowWidth: document.documentElement.scrollWidth,
        windowHeight: document.documentElement.scrollHeight,
      });
      btn.style.visibility = 'visible';
      canvas.toBlob((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        const ts = new Date().toISOString().replace(/[:.]/g,'-').slice(0,19);
        a.href = url; a.download = 'faceswap_pro_v2_ui_' + ts + '.png';
        document.body.appendChild(a); a.click(); a.remove();
        URL.revokeObjectURL(url);
      }, 'image/png');
    } catch(e) {
      console.error(e); alert('Screenshot failed: ' + (e.message || e));
    } finally {
      btn.style.visibility = 'visible';
      btn.innerHTML = orig; btn.disabled = false;
    }
  };
  document.body.appendChild(btn);
}
""",
        css="""
        /* Show only the logo matching the active theme. Gradio toggles
           the 'dark' class on <html> via ?__theme=dark / light. */
        html.dark .fp-logo-light { display: none !important; }
        html:not(.dark) .fp-logo-dark { display: none !important; }
        """,
    ) as app:
        # Optional dual header logo: drop two PNGs at
        #   v2/assets/logo.png        (DARK theme version, light-on-dark)
        #   v2/assets/logo_light.png  (LIGHT theme version, dark-on-light)
        # Either or both may exist. CSS classes below hide the wrong
        # one based on the .dark class on <html> (Gradio toggles it
        # when the user flips themes via ?__theme=dark|light).
        _assets = Path(__file__).resolve().parent.parent / "assets"
        _logo_dark  = _assets / "logo.png"
        _logo_light = _assets / "logo_light.png"
        if _logo_dark.is_file() or _logo_light.is_file():
            with gr.Row():
                if _logo_dark.is_file():
                    gr.Image(
                        value=str(_logo_dark),
                        show_label=False, interactive=False,
                        show_download_button=False, container=False,
                        height=140, elem_classes=["fp-logo-dark"],
                    )
                if _logo_light.is_file():
                    gr.Image(
                        value=str(_logo_light),
                        show_label=False, interactive=False,
                        show_download_button=False, container=False,
                        height=140, elem_classes=["fp-logo-light"],
                    )
        gr.Markdown(
            "# faceswap_pro v2 — Lip-Sync\n"
            "_LatentSync 512 · 3-col layout · live previews · ETA · "
            "named presets · render history._"
        )

        with gr.Tabs():
            # ====================================================
            # TAB 1: Lip-Sync  (3-column layout)
            # ====================================================
            with gr.Tab("🎤 Lip-Sync"):
                with gr.Row(equal_height=False):

                    # ---- COLUMN 1: Inputs + Previews ----
                    with gr.Column(scale=1):
                        gr.Markdown("### 1. Inputs")
                        face = gr.Video(label="Face clip (primary)")
                        # frame-0 thumbnail removed: gr.Video above already
                        # shows a built-in preview of the uploaded clip.
                        face_extras = gr.File(
                            label="Additional face clips "
                                  "(multi-clip mode)",
                            file_count="multiple",
                            file_types=["video"], type="filepath",
                        )
                        audio = gr.Audio(label="Audio", type="filepath")
                        audio_wave = gr.Image(
                            interactive=False, height=80,
                            type="filepath", show_label=False,
                            show_download_button=False,
                            container=False,
                        )

                    # ---- COLUMN 2: Pipeline + Knobs ----
                    with gr.Column(scale=1):
                        gr.Markdown("### 2. Pipeline")
                        with gr.Group():
                            isolate = gr.Checkbox(
                                label="Isolate vocals (Demucs)",
                                value=True)
                            quick = gr.Checkbox(
                                label="Quick 12-second test",
                                value=False)
                            enhance = gr.Checkbox(
                                label="Enhance faces (GFPGAN)",
                                value=True)
                            extend_single = gr.Checkbox(
                                label="Extend single clip to "
                                      "audio length",
                                value=False,
                            )

                        gr.Markdown("### Engine")
                        with gr.Group():
                            engine = gr.Dropdown(
                                choices=[
                                    "latentsync",
                                    "keysync",
                                ],
                                value="latentsync",
                                label="Lipsync engine",
                                info="latentsync = HUMAN speech / singing "
                                     "(fast, ~5 min). keysync = NON-HUMAN "
                                     "characters (cartoons, stylized art, "
                                     "animals; ~15 min, runs in its own "
                                     "venv).",
                            )

                        # ---- KeySync non-human face ROI (SAM2 click) ----
                        # Visible only when engine == "keysync". User
                        # clicks on the character's face in the preview
                        # image -- SAM2 propagates that click across the
                        # whole video so KeySync sees a clean face crop
                        # regardless of the SFD detector failing on
                        # cartoons / animals. (0,0) = auto detect.
                        with gr.Group(visible=False) as ks_face_group:
                            gr.Markdown(
                                "### KeySync non-human face ROI\n"
                                "Click once on the character's face in "
                                "the preview below. SAM2 will track "
                                "that point across the whole video. "
                                "Use this when the face detector "
                                "can't find a face (cartoons, statues, "
                                "stylized art, animals).")
                            ks_face_preview = gr.Image(
                                label="Preview frame (click the face)",
                                type="filepath",
                                interactive=True,
                                height=320,
                            )
                            with gr.Row():
                                ks_face_click_x = gr.Number(
                                    value=0, precision=0,
                                    label="face click X (px)")
                                ks_face_click_y = gr.Number(
                                    value=0, precision=0,
                                    label="face click Y (px)")
                                ks_face_click_frame = gr.Number(
                                    value=0, precision=0,
                                    label="preview frame index")
                            ks_face_reset_btn = gr.Button(
                                "reset face click", size="sm")
                            ks_skip_crop = gr.Checkbox(
                                label="Skip KeySync crop_video.py "
                                      "(use source as-is, no face crop)",
                                value=False,
                                info="Bypass KeySync's landmark-based "
                                     "crop step entirely. Required test "
                                     "path for human face at native "
                                     "resolution. Auto-on when a face "
                                     "click is supplied (SAM2 already "
                                     "cropped). Tick manually to test "
                                     "without any cropping at all.",
                            )

                        # ---- SAM2 multi-click mask-out (non-face objects) ----
                        # RE-ENABLED 2026-06-05 after fixing the
                        # fps-ratio drift bug in composite_back.
                        with gr.Accordion(
                                "Mask out non-face regions (SAM2)",
                                open=False, visible=True):
                            gr.Markdown(
                                "Click on a non-face object that "
                                "LatentSync wrongly lipsyncs (cat, "
                                "animal, statue, prop). Positive "
                                "clicks add to the mask; negative "
                                "clicks subtract. The masked region "
                                "is inpainted before LatentSync sees "
                                "it, then the original pixels are "
                                "composited back over the lipsync "
                                "output.")
                            mout_enabled = gr.Checkbox(
                                label="Enable mask-out",
                                value=False)
                            mout_preview = gr.Image(
                                label="Click on object to mask "
                                      "(frame 0 of face clip)",
                                type="filepath",
                                interactive=True,
                                height=320,
                            )
                            with gr.Row():
                                mout_click_label = gr.Radio(
                                    choices=["+ positive", "- negative"],
                                    value="+ positive",
                                    label="Next click is:")
                                mout_frame_idx = gr.Number(
                                    value=0, precision=0,
                                    label="Click frame index")
                            mout_clicks_state = gr.State([])
                            mout_clicks_md = gr.Markdown(
                                "**Clicks so far:** (none)")
                            with gr.Row():
                                mout_clear_btn = gr.Button(
                                    "clear all clicks", size="sm")
                                mout_load_preview_btn = gr.Button(
                                    "load preview frame", size="sm")
                            with gr.Row():
                                mout_dilate = gr.Slider(
                                    0, 40, value=12, step=1,
                                    label="Mask dilate (px)",
                                    info="Grows the mask before inpaint "
                                         "to cover edge bleed.")
                                mout_feather = gr.Slider(
                                    0, 24, value=8, step=1,
                                    label="Composite feather (px)",
                                    info="Gaussian feather at the mask "
                                         "edge during paste-back.")
                            mout_npy_override = gr.Textbox(
                                value="",
                                label="Pre-computed mask NPY (optional)",
                                placeholder="Paste a path from the "
                                            "Rotoscoping tab, or leave "
                                            "empty to run SAM2 from "
                                            "clicks above.",
                                info="When set + file exists, Stage "
                                     "1.76 skips the SAM2 worker and "
                                     "uses this mask directly.")

                        gr.Markdown("### LatentSync knobs")
                        with gr.Group():
                            ls_steps = gr.Slider(
                                10, 50, value=20, step=1,
                                label="Inference steps")
                            ls_guidance = gr.Slider(
                                1.0, 5.0, value=1.5, step=0.05,
                                label="Lip strength (guidance scale)",
                                info="Higher = more pronounced lip "
                                     "movement (model follows the "
                                     "audio more aggressively). "
                                     "Lower = subtler, closer to "
                                     "source pixels. Crank to 3-5 "
                                     "for songs with quiet vocals "
                                     "or mumbly speech; keep at "
                                     "1.5-2 for clean spoken-word.")
                            ls_deepcache = gr.Checkbox(
                                label="DeepCache (~2× faster)",
                                value=True)
                            ls_seed = gr.Number(
                                value=-1, precision=0, label="Seed",
                                info="-1 = random; any int = "
                                     "reproducible.")
                            # T1-NEW color match dropdown
                            ls_color_match = gr.Dropdown(
                                label="Color match (post-LatentSync)",
                                choices=["reinhard", "none"],
                                value="reinhard",
                                info="Fixes stylized-source -> orange-cheek "
                                     "VAE drift.  Reinhard = match face LAB "
                                     "to source.  None = disable.")
                            ls_face_det = gr.Slider(
                                0.30, 0.95, value=0.5, step=0.05,
                                label="Face detection threshold",
                                info="Upstream default 0.5. Raise to "
                                     "0.7-0.85 when the source has "
                                     "nearby face-like distractors "
                                     "(animals, statues) so LatentSync's "
                                     "detector filters them out and "
                                     "consistently picks the human face.")

                        with gr.Accordion(
                                "Per-clip identity fine-tune",
                                open=False):
                            gr.Markdown(
                                "Fine-tune LatentSync on the currently "
                                "selected face clip. Trains only "
                                "attention + motion modules with recon "
                                "loss for the specified number of steps. "
                                "Once trained, the fine-tuned ckpt is "
                                "auto-used for renders of THIS clip. "
                                "Slow (minutes), GPU-only.")
                            ft_status = gr.Markdown(
                                "**fine-tune status:** _no clip loaded_")
                            ft_steps = gr.Slider(
                                100, 5000, value=1000, step=100,
                                label="Fine-tune steps",
                                info="More steps = more identity "
                                     "specialization, but diminishing "
                                     "returns past ~1500.")
                            ft_train_btn = gr.Button(
                                "Train per-clip fine-tune (slow)",
                                size="sm")
                            ft_log = gr.Textbox(
                                label="fine-tune log",
                                value="(idle)",
                                lines=8, max_lines=8,
                                interactive=False)

                        with gr.Accordion(
                                "Voice swap (optional)",
                                open=False):
                            voice_model = gr.Dropdown(
                                choices=voices, value=voices[0],
                                label="Voice model")
                            voice_transpose = gr.Slider(
                                -12, 12, value=0, step=1,
                                label="Transpose (semitones)")

                        # Occlusion gate REMOVED from pipeline. The
                        # widgets below are kept as hidden inputs so
                        # the click handlers' positional argument
                        # contracts stay valid -- but the user can't
                        # see or change them, and the orchestrator
                        # ignores them.
                        with gr.Group(visible=False):
                            occ_enabled = gr.Checkbox(
                                label="(removed)", value=False)
                            occ_bbox_smooth = gr.Slider(
                                0.0, 0.9, value=0.4, step=0.05,
                                label="Bbox smoothing (EMA)",
                                info="Damps detector jitter. 0 = none, "
                                     "0.9 = heavy. Start at 0.4.")
                            occ_mask_smooth = gr.Slider(
                                0.0, 0.9, value=0.7, step=0.05,
                                label="Mask smoothing (EMA)",
                                info="Damps per-frame XSeg edge flicker. "
                                     "0 = none, 0.9 = heavy. Start 0.7.")
                            occ_align = gr.Checkbox(
                                label="Weld face to head (DISABLED in "
                                      "working baseline)",
                                value=False,
                                info="OFF. Every welding/alignment "
                                     "variant produced more artifacts "
                                     "than the baseline LatentSync "
                                     "drift it was trying to fix. The "
                                     "current xseg_gate is the v1 "
                                     "working baseline -- pure XSeg "
                                     "occluder restore, no warping. "
                                     "Flag is accepted but ignored.")
                            occ_mouth_polygon = gr.Checkbox(
                                label="Mouth-polygon paste-back "
                                      "(EXPERIMENTAL -- usually OFF)",
                                value=False,
                                info="OFF (default, recommended): the "
                                     "occluder gate uses the lizard-only "
                                     "mask -- lipsync output everywhere, "
                                     "source pixels only where there's "
                                     "an occluder. No face boundary in "
                                     "the mask, no drift seam. "
                                     "ON: experimental polygon paste-"
                                     "back path. Currently shows a "
                                     "double-anatomy seam at the polygon "
                                     "edge on clips with LatentSync face "
                                     "drift. Keep OFF unless debugging.")
                            occ_feather = gr.Slider(
                                1, 31, value=9, step=2,
                                label="Matte feather (px)",
                                info="Gaussian blur kernel on the "
                                     "composite alpha. Must be odd.")

                        with gr.Accordion(
                                "Watermark (optional)",
                                open=False):
                            wm_enabled = gr.Checkbox(
                                label="Burn watermark onto every frame",
                                value=False)
                            wm_image = gr.File(
                                label="Watermark image (PNG w/ alpha "
                                      "recommended)",
                                file_types=["image"],
                                type="filepath")
                            wm_position = gr.Dropdown(
                                choices=["TL","TR","BL","BR","CENTER"],
                                value="BR", label="Position")
                            wm_scale = gr.Slider(
                                1, 50, value=15, step=1,
                                label="Scale (% of frame width)")
                            wm_opacity = gr.Slider(
                                5, 100, value=80, step=5,
                                label="Opacity (%)")

                        with gr.Accordion(
                                "Output aspect ratio (optional)",
                                open=False):
                            ar_enabled = gr.Checkbox(
                                label="Reshape output canvas",
                                value=False)
                            ar_target = gr.Dropdown(
                                choices=(["(keep original)"]
                                         + _list_aspects()),
                                value="(keep original)",
                                label="Target aspect")
                            ar_fill = gr.Radio(
                                choices=["crop", "pad"], value="crop",
                                label="Fill mode",
                                info="crop = scale up + cut excess; "
                                     "pad = scale down + black bars.")

                        with gr.Accordion(
                                "Preview output frame",
                                open=False):
                            preview_pp_btn = gr.Button(
                                "\U0001F50D Preview output (watermark "
                                "+ aspect on frame 0)",
                                variant="secondary", size="sm")
                            preview_pp_img = gr.Image(
                                interactive=False, height=260,
                                show_label=False,
                                show_download_button=False,
                                container=False)
                            gr.Markdown(
                                "_Sub-second dry run on the first "
                                "frame of the primary face clip. "
                                "Click any time you change the "
                                "watermark or aspect settings._")

                        with gr.Accordion(
                                "Quick presets",
                                open=False):
                            preset_dd_inline = gr.Dropdown(
                                choices=preset_choices, value=None,
                                label="Load preset",
                                allow_custom_value=True,
                                interactive=True)
                            load_btn_inline = gr.Button(
                                "📂 Apply preset",
                                variant="secondary", size="sm")

                    # ---- COLUMN 3: Output + Render ----
                    with gr.Column(scale=1):
                        gr.Markdown("### 3. Output")
                        out_video = gr.Video(
                            label="Result", interactive=False,
                            show_download_button=True, height=320)
                        # T1-1 cross-tab handoff: take the lipsync output
                        # and use it as the target video on Face Swap.
                        ls_send_to_fs_btn = gr.Button(
                            "Send result to Face Swap →",
                            variant="secondary", size="sm")
                        eta_md = gr.Markdown(
                            "**ETA:** drop a face clip to see ETA")
                        with gr.Row():
                            render_btn = gr.Button(
                                "▶ Render now", variant="primary",
                                size="lg")
                            queue_btn = gr.Button(
                                "+ Add to queue", variant="secondary",
                                size="lg")
                            cancel_btn = gr.Button(
                                "✖ Cancel",
                                variant="stop", size="lg")
                        out_status = gr.Markdown(
                            "Ready. Drop a face clip + audio on the "
                            "left, tune in the middle, render here.")

                # ---- wire previews + ETA ----
                _opt_inputs = [face, audio, face_extras,
                               extend_single, enhance, quick]
                face.change(fn=_on_face_change, inputs=_opt_inputs,
                            outputs=[eta_md])
                audio.change(fn=_on_audio_change,
                             inputs=[audio, face, face_extras,
                                     extend_single, enhance, quick],
                             outputs=[audio_wave, eta_md])
                for opt in (face_extras, extend_single, enhance, quick):
                    opt.change(fn=_on_options_change,
                               inputs=_opt_inputs, outputs=[eta_md])

                # ---- wire render ----
                render_btn.click(
                    fn=_run_render,
                    inputs=[face, audio, face_extras,
                            engine,
                            isolate, quick, enhance, extend_single,
                            ls_steps, ls_guidance, ls_deepcache, ls_seed, ls_color_match,
                            ls_face_det,
                            voice_model, voice_transpose,
                            wm_enabled, wm_image, wm_position,
                            wm_scale, wm_opacity,
                            ar_enabled, ar_target, ar_fill,
                            occ_enabled, occ_bbox_smooth,
                            occ_mask_smooth, occ_align,
                            occ_feather, occ_mouth_polygon,
                            ks_face_click_x, ks_face_click_y,
                            ks_face_click_frame, ks_skip_crop,
                            mout_enabled, mout_clicks_state,
                            mout_dilate, mout_feather,
                            mout_npy_override],
                    outputs=[out_video, out_status, gr.State()],
                )
                cancel_btn.click(
                    fn=_on_render_cancel,
                    inputs=[],
                    outputs=[out_status],
                )
                queue_btn.click(
                    fn=_enqueue_render,
                    inputs=[face, audio, face_extras,
                            engine,
                            isolate, quick, enhance, extend_single,
                            ls_steps, ls_guidance, ls_deepcache, ls_seed, ls_color_match,
                            ls_face_det,
                            voice_model, voice_transpose,
                            wm_enabled, wm_image, wm_position,
                            wm_scale, wm_opacity,
                            ar_enabled, ar_target, ar_fill,
                            occ_enabled, occ_bbox_smooth,
                            occ_mask_smooth, occ_align,
                            occ_feather, occ_mouth_polygon,
                            ks_face_click_x, ks_face_click_y,
                            ks_face_click_frame, ks_skip_crop,
                            mout_enabled, mout_clicks_state,
                            mout_dilate, mout_feather,
                            mout_npy_override],
                    outputs=[out_status],
                )

                # ---- KeySync face-ROI widget plumbing ----
                # (a) toggle the ks_face_group on engine selection
                # (b) on face_video change, extract frame 0 to use as
                #     the click target preview
                # (c) capture (x,y) from a click on the preview into
                #     the hidden Number widgets
                # (d) reset button clears the click coords
                def _on_engine_change(eng):
                    is_ks = (str(eng or "").lower() == "keysync")
                    return gr.update(visible=is_ks)
                engine.change(
                    fn=_on_engine_change,
                    inputs=[engine],
                    outputs=[ks_face_group],
                )

                def _on_face_change_for_ks(face_path, eng):
                    """Reset KeySync click coords on face change.

                    NOTE: We deliberately do NOT read the video here.
                    Previously this opened the file with cv2 to extract
                    frame 0 as a PNG for the click target. On Windows
                    that left a brief file handle that could race with
                    Gradio's HTML5 video preview server, surfacing as
                    "Error - Video not playable" in the upload widget.
                    The frame extraction now runs lazily inside
                    _on_engine_change when the user actually switches
                    to KeySync mode (see ks_load_frame_now below).
                    """
                    return gr.update(), 0, 0, 0
                face.change(
                    fn=_on_face_change_for_ks,
                    inputs=[face, engine],
                    outputs=[ks_face_preview, ks_face_click_x,
                             ks_face_click_y, ks_face_click_frame],
                )

                def _ks_load_frame_now(face_path, eng):
                    """Extract frame 0 from the face video as a PNG for
                    the KeySync click-target preview. Triggered when
                    the engine dropdown is set to 'keysync' (i.e. the
                    user actually opens the KeySync workflow), so we
                    don't pay the cv2 read cost on the default LatentSync
                    path. Wrapped in try/finally so the cv2 handle is
                    always released even on exceptions, eliminating
                    the file-lock race with Gradio's video player.
                    """
                    if not face_path:
                        return gr.update(value=None)
                    if str(eng or "").lower() != "keysync":
                        return gr.update()
                    cap = None
                    try:
                        import cv2 as _cv2, tempfile as _tmp, os as _os
                        cap = _cv2.VideoCapture(str(face_path))
                        ok, frm = cap.read()
                        if not ok or frm is None:
                            return gr.update(value=None)
                        fd, png = _tmp.mkstemp(
                            suffix="_ks_preview.png")
                        _os.close(fd)
                        _cv2.imwrite(png, frm)
                        return gr.update(value=png)
                    except Exception:
                        return gr.update(value=None)
                    finally:
                        # Release ALWAYS so no handle lingers on Windows.
                        try:
                            if cap is not None:
                                cap.release()
                        except Exception:
                            pass
                # Wire on engine change AND once on initial keysync entry.
                engine.change(
                    fn=_ks_load_frame_now,
                    inputs=[face, engine],
                    outputs=[ks_face_preview],
                )

                # ---- SAM2 mask-out wiring ----
                def _mout_load_preview(face_path, frame_idx):
                    if not face_path:
                        return gr.update(value=None)
                    cap = None
                    try:
                        import cv2 as _cv2, tempfile as _tmp, os as _os
                        cap = _cv2.VideoCapture(str(face_path))
                        cap.set(_cv2.CAP_PROP_POS_FRAMES,
                                max(0, int(frame_idx or 0)))
                        ok, frm = cap.read()
                        if not ok or frm is None:
                            return gr.update(value=None)
                        fd, png = _tmp.mkstemp(
                            suffix="_mout_preview.png")
                        _os.close(fd)
                        _cv2.imwrite(png, frm)
                        return gr.update(value=png)
                    except Exception:
                        return gr.update(value=None)
                    finally:
                        try:
                            if cap is not None:
                                cap.release()
                        except Exception:
                            pass
                mout_load_preview_btn.click(
                    fn=_mout_load_preview,
                    inputs=[face, mout_frame_idx],
                    outputs=[mout_preview],
                )

                def _mout_render_clicks(clicks):
                    if not clicks:
                        return "**Clicks so far:** (none)"
                    lines = ["**Clicks so far:**"]
                    for i, c in enumerate(clicks):
                        sign = "+" if c.get("label", 1) == 1 else "-"
                        lines.append(
                            f"- {i+1}. {sign} ({c.get('x',0)}, "
                            f"{c.get('y',0)}) @ frame "
                            f"{c.get('frame',0)}")
                    return "\n".join(lines)

                def _mout_on_image_click(clicks, label_choice,
                                          frame_idx, evt: gr.SelectData):
                    try:
                        x, y = int(evt.index[0]), int(evt.index[1])
                    except Exception:
                        return clicks, _mout_render_clicks(clicks)
                    lab = 1 if str(label_choice or "").startswith("+") else 0
                    fidx = int(frame_idx or 0)
                    new_clicks = list(clicks or [])
                    new_clicks.append(
                        {"x": x, "y": y, "frame": fidx, "label": lab})
                    return new_clicks, _mout_render_clicks(new_clicks)
                mout_preview.select(
                    fn=_mout_on_image_click,
                    inputs=[mout_clicks_state, mout_click_label,
                            mout_frame_idx],
                    outputs=[mout_clicks_state, mout_clicks_md],
                )

                def _mout_clear():
                    return [], "**Clicks so far:** (none)"
                mout_clear_btn.click(
                    fn=_mout_clear,
                    inputs=None,
                    outputs=[mout_clicks_state, mout_clicks_md],
                )

                # ---- per-clip fine-tune wiring ----
                def _ft_status_for(face_path):
                    """Compute fine-tune readiness banner. Wrapped in
                    a hard try/except that absolutely cannot raise; any
                    failure returns 'unknown' so this handler can't
                    interfere with other face.change handlers (e.g.
                    Gradio's video preview generator) firing on the
                    same upload event."""
                    try:
                        if not face_path:
                            return "**fine-tune status:** _no clip loaded_"
                        try:
                            from core import lipsync_finetune as _lsft
                        except Exception:
                            return "**fine-tune status:** _module unavailable_"
                        try:
                            ck = _lsft.get_finetune_checkpoint(str(face_path))
                        except Exception:
                            return "**fine-tune status:** unknown"
                        if ck is None:
                            return ("**fine-tune status:** none yet "
                                    "(click *Train* to create one)")
                        try:
                            mb = ck.stat().st_size / (1024 * 1024)
                        except Exception:
                            mb = 0
                        return (f"**fine-tune status:** READY "
                                f"({mb:.0f} MB) -- will be auto-used "
                                f"on next render of this clip")
                    except Exception:
                        # Absolutely never let this raise.
                        return "**fine-tune status:** unknown"
                face.change(
                    fn=_ft_status_for,
                    inputs=[face],
                    outputs=[ft_status],
                )

                def _ft_train_clicked(face_path, audio_path, steps):
                    """Generator: streams training progress lines into
                    ft_log + flips status when done. Runs the training
                    in a worker thread; the main thread polls the log
                    buffer every ~1.5s and yields incremental UI
                    updates."""
                    if not face_path:
                        yield ("**fine-tune status:** _no clip loaded_",
                               "no face clip selected")
                        return
                    import threading as _th
                    import time as _time
                    log_lines = []
                    log_lock = _th.Lock()
                    done_evt = _th.Event()
                    result = {"path": None, "exc": None}

                    def _log(msg):
                        s = str(msg).rstrip()
                        with log_lock:
                            log_lines.append(s)
                        print(s, flush=True)

                    def _worker():
                        try:
                            from core import lipsync_finetune as _lsft
                            _aud = str(audio_path) if audio_path else None
                            _log(f"[ui] preparing clip "
                                 f"(Phase 1; audio="
                                 f"{Path(_aud).name if _aud else 'NONE'})")
                            _lsft.prepare_clip_for_training(
                                str(face_path), log=_log,
                                source_audio=_aud)
                            _log(f"[ui] launching training "
                                 f"({int(steps)} steps) ...")
                            ck = _lsft.train_identity_finetune(
                                str(face_path),
                                num_steps=int(steps),
                                source_audio=_aud,
                                log=_log)
                            result["path"] = str(ck)
                        except Exception as _exc:
                            import traceback as _tb
                            result["exc"] = _exc
                            _log("[ui] FAILED: " + str(_exc))
                            for ln in _tb.format_exc().splitlines():
                                _log("[ui]   " + ln)
                        finally:
                            done_evt.set()

                    t = _th.Thread(target=_worker, daemon=True)
                    t.start()
                    yield ("**fine-tune status:** TRAINING ...",
                           "(starting)")
                    while not done_evt.is_set():
                        with log_lock:
                            tail = "\n".join(log_lines[-8:])
                        yield (gr.update(),
                               tail or "(initializing)")
                        done_evt.wait(timeout=1.5)
                    with log_lock:
                        tail = "\n".join(log_lines[-12:])
                    if result["exc"] is not None:
                        yield (f"**fine-tune status:** FAILED "
                               f"({result['exc']})", tail)
                    else:
                        yield (_ft_status_for(face_path), tail)
                ft_train_btn.click(
                    fn=_ft_train_clicked,
                    inputs=[face, audio, ft_steps],
                    outputs=[ft_status, ft_log],
                )

                def _on_ks_face_click(evt: gr.SelectData):
                    """Gradio Image .select() callback. evt.index is
                    (x, y) in image pixel coords."""
                    try:
                        x, y = int(evt.index[0]), int(evt.index[1])
                    except Exception:
                        return 0, 0
                    return x, y
                ks_face_preview.select(
                    fn=_on_ks_face_click,
                    inputs=None,
                    outputs=[ks_face_click_x, ks_face_click_y],
                )

                def _on_ks_reset():
                    return 0, 0, 0
                ks_face_reset_btn.click(
                    fn=_on_ks_reset,
                    inputs=None,
                    outputs=[ks_face_click_x, ks_face_click_y,
                             ks_face_click_frame],
                )
                preview_pp_btn.click(
                    fn=_preview_post_output,
                    inputs=[face, wm_enabled, wm_image, wm_position,
                            wm_scale, wm_opacity,
                            ar_enabled, ar_target, ar_fill],
                    outputs=[preview_pp_img],
                )

                # ---- inline preset apply ----
                load_btn_inline.click(
                    fn=_on_load_preset,
                    inputs=[preset_dd_inline],
                    outputs=[isolate, quick, enhance, extend_single,
                             ls_steps, ls_guidance, ls_deepcache,
                             ls_seed, ls_color_match, voice_model, voice_transpose,
                             out_status])

            # ====================================================
            # TAB 2: Face Swap (Source -> Target video)
            # ====================================================
            with gr.Tab("\U0001F3DE\uFE0F Face Swap"):
                with gr.Row(equal_height=False):
                    # ---- COLUMN 1: SOURCE ----
                    with gr.Column(scale=1):
                        gr.Markdown("### 1. Source face")
                        # NOTE: legacy v1 uses gr.File for SOURCE, not
                        # gr.Image. gr.Image re-encodes / downscales
                        # uploads which destroys ArcFace identity
                        # detail and causes waxy face-swap output.
                        # We match v1's widget exactly here, then
                        # render a passthrough preview in a separate
                        # gr.Image that NEVER touches the underlying
                        # source file.
                        # file_types=["image"] DELIBERATELY OMITTED:
                        # gr.File's preprocess validates against
                        # file_types and on rejection caches the
                        # invalid path in component state -- every
                        # subsequent upload (even valid ones) then
                        # re-raises "Invalid file type" forever.
                        # Downstream InsightFace face detection
                        # cleanly rejects non-images with a clear
                        # error; that's better UX than a wedged UI.
                        vs_source = gr.File(
                            label="Source face image (.jpg / .png)",
                            type="filepath",
                        )
                        vs_source_preview = gr.Image(
                            interactive=False, height=240,
                            type="filepath", show_label=False,
                            show_download_button=False,
                            container=False,
                        )
                        gr.Markdown("_Face identity to paste onto "
                                    "every detected face in the "
                                    "target video._")
                    # ---- COLUMN 2: TARGET + options ----
                    with gr.Column(scale=1):
                        gr.Markdown("### 2. Target video & options")
                        vs_target = gr.Video(label="Target video")
                        with gr.Group():
                            vs_blend = gr.Dropdown(
                                choices=["poisson", "alpha",
                                         "feather", "none"],
                                value="poisson",
                                label="Blend method",
                                info="poisson = seamless skin tone "
                                     "blend. alpha/feather = soft "
                                     "alpha. none = hard paste.")
                            vs_enhance = gr.Checkbox(
                                label="Enhance faces",
                                value=False)
                            # T2-2 face-restoration backend dropdown
                            vs_restorer = gr.Dropdown(
                                label="Face restoration backend",
                                choices=[
                                    "none",
                                    "gfpgan",
                                    "codeformer",
                                    "restoreformer",
                                ],
                                value="gfpgan",
                                info="GFPGAN = in-pipeline (default). "
                                     "CodeFormer = post-process, often "
                                     "sharper eyes. RestoreFormer = "
                                     "NOT installed yet (stub).  Only "
                                     "runs when Enhance faces is on.")
                            # T2-NEW: rotoscope mask region restriction
                            vs_mask_npy = gr.Textbox(
                                label="Restrict swap to rotoscope mask (NPY, optional)",
                                placeholder=(
                                    "e.g. .../rotoscope/<hash>/masks/"
                                    "obj_1_combined.npy"),
                                info=("Drops detected faces whose "
                                       "bbox centroid lands outside "
                                       "the mask.  Use this when "
                                       "multi-face frames pick up "
                                       "unwanted subjects (animals, "
                                       "extras, statues).  Empty = "
                                       "no gating."))
                            vs_det_thresh = gr.Slider(
                                0.1, 0.9, value=0.5, step=0.05,
                                label="Detection threshold")
                            # Exact values FaceSwapPipeline accepts
                            # (see core.pipeline OUTPUT_QUALITY).
                            vs_quality = gr.Dropdown(
                                choices=["visually_lossless",
                                         "balanced", "lossless"],
                                value="visually_lossless",
                                label="Output quality",
                                info="visually_lossless = ~10x smaller "
                                     "than true lossless, looks the "
                                     "same. balanced = smaller file. "
                                     "lossless = largest file.")
                        with gr.Accordion("Trim frames (optional)",
                                          open=False):
                            vs_trim_start = gr.Number(
                                value=0, precision=0,
                                label="Start frame")
                            vs_trim_end = gr.Number(
                                value=0, precision=0,
                                label="End frame (0 = end of video)")

                        with gr.Accordion(
                                "Face selector (which face to swap)",
                                open=False):
                            gr.Markdown(
                                "Default: swap the largest detected "
                                "face in every frame. **Reference** "
                                "mode swaps only the detected face "
                                "matching the reference image you "
                                "upload below -- required for "
                                "multi-person videos.")
                            vs_selector_mode = gr.Dropdown(
                                choices=["largest", "reference"],
                                value="largest",
                                label="Selector mode")
                            # file_types deliberately omitted -- see
                            # comment on vs_source above.
                            vs_reference_face = gr.File(
                                label="Reference face image "
                                      "(used when mode='reference')",
                                type="filepath",
                            )
                            vs_reference_distance = gr.Slider(
                                0.05, 1.5, value=0.6, step=0.05,
                                label="Reference distance threshold",
                                info="Cosine distance (1 - cos_sim). "
                                     "Lower = stricter match. 0.6 is "
                                     "a sensible default; 0.3-0.4 for "
                                     "tight identity match; >1.0 = "
                                     "match almost anyone (effectively "
                                     "disables filtering).")
                            vs_reference_preview = gr.Image(
                                interactive=False, height=180,
                                type="filepath", show_label=False,
                                show_download_button=False,
                                container=False,
                            )

                        with gr.Accordion(
                                "Face mask & identity strength",
                                open=False):
                            gr.Markdown(
                                "Tune the mask edge and identity "
                                "intensity. Defaults match the stock "
                                "InsightFace paste-back; deviate as "
                                "needed for jaw seams or over-saturated "
                                "identity swaps.")
                            vs_pixel_boost = gr.Dropdown(
                                choices=[128, 256, 384, 512, 768],
                                value=128,
                                label="Pixel boost (output detail)",
                                info="inswapper hallucinates at 128. "
                                     "Higher = post-swap GFPGAN upscale "
                                     "to that resolution before paste-"
                                     "back. Adds one GFPGAN pass per "
                                     "face per frame; 512 is the sweet "
                                     "spot for close-ups.")
                            vs_mask_padding = gr.Slider(
                                -30, 30, value=0, step=1,
                                label="Mask padding (px)",
                                info="+ grows mask inward (more original "
                                     "face shows through). - grows "
                                     "outward (swap extends past the "
                                     "edge). 0 = stock.")
                            vs_mask_blur = gr.Slider(
                                0.0, 4.0, value=1.0, step=0.05,
                                label="Mask blur scale",
                                info="Multiplier on the auto-computed "
                                     "Gaussian feather kernel. Larger = "
                                     "softer edge. 1.0 = stock.")
                            vs_swap_strength = gr.Slider(
                                0.0, 1.0, value=1.0, step=0.05,
                                label="Identity strength",
                                info="1.0 = full swap. 0.0 = original "
                                     "face untouched. Lower for subtle "
                                     "look-alike effect.")
                            vs_enhancer_blend = gr.Slider(
                                0.0, 1.0, value=1.0, step=0.05,
                                label="Enhancer blend (GFPGAN)",
                                info="Effective only when Enhance "
                                     "faces is ON. 1.0 = full GFPGAN "
                                     "restoration; lower for less "
                                     "plasticky skin.")

                        with gr.Accordion(
                                "Temporal smoothing (face flicker)",
                                open=False):
                            gr.Markdown(
                                "Reduce frame-to-frame face flicker. "
                                "Already on by default (EMA 0.85 on the "
                                "face region). Tune here to dial the "
                                "strength up for static shots or down "
                                "for fast motion.")
                            vs_temporal_enabled = gr.Checkbox(
                                label="Enable temporal smoothing",
                                value=True)
                            vs_temporal_ema = gr.Slider(
                                0.0, 0.98, value=0.85, step=0.01,
                                label="EMA decay (smoothing strength)",
                                info="Higher = more smoothing, more "
                                     "lag on motion. 0.85 = stock. "
                                     "Try 0.92+ for static shots, "
                                     "0.6-0.75 for fast motion.")
                            vs_temporal_buffer = gr.Slider(
                                1, 15, value=5, step=1,
                                label="Buffer size (frames)",
                                info="History depth for optical-flow "
                                     "context. 5 = stock. Larger "
                                     "buffers use more RAM but help "
                                     "on long jitter periods.")

                        with gr.Accordion(
                                "Lighting / color match",
                                open=False):
                            gr.Markdown(
                                "Match the swapped face to the scene\'s "
                                "color + shadow. Reinhard color transfer "
                                "and SH-relit shadow correction both run "
                                "by default; tune here if the swap looks "
                                "'pasted on'.")
                            vs_color_transfer_mode = gr.Dropdown(
                                choices=["reinhard", "none"],
                                value="reinhard",
                                label="Color transfer mode",
                                info="reinhard = match swap LAB-color "
                                     "stats to the original face region. "
                                     "Set to none if the source already "
                                     "matches the scene and the "
                                     "transfer is making colors worse.")
                            vs_shadow_correction = gr.Checkbox(
                                label="Shadow correction",
                                value=True,
                                info="Apply the original face's lighting "
                                     "envelope to the swap. Helps in "
                                     "harsh side-lit or dim scenes.")
                            with gr.Row():
                                vs_shadow_clamp_min = gr.Slider(
                                    0.1, 1.0, value=0.5, step=0.05,
                                    label="Shadow clamp min",
                                    info="Lower = darker possible "
                                         "shadows.")
                                vs_shadow_clamp_max = gr.Slider(
                                    1.0, 3.0, value=1.5, step=0.05,
                                    label="Shadow clamp max",
                                    info="Higher = brighter possible "
                                         "highlights.")

                        with gr.Accordion(
                                "Identity blend / journey (optional)",
                                open=False):
                            gr.Markdown(
                                "**Blend** two source identities in "
                                "ArcFace embedding space, or **journey** "
                                "between them across the timeline. "
                                "Drop a second source to enable.")
                            # file_types deliberately omitted -- see
                            # comment on vs_source above.
                            vs_source_b = gr.File(
                                label="Source face B (.jpg / .png)",
                                type="filepath",
                            )
                            vs_blend_alpha = gr.Slider(
                                0.0, 1.0, value=0.5, step=0.05,
                                label="Blend alpha "
                                      "(0 = pure A, 1 = pure B)",
                                info="Static blend: same identity on "
                                     "every frame. Ignored when "
                                     "Journey is on.")
                            vs_journey_mode = gr.Checkbox(
                                label="Journey mode "
                                      "(alpha ramps A -> B across video)",
                                value=False)
                            with gr.Row():
                                vs_journey_start = gr.Slider(
                                    0.0, 1.0, value=0.0, step=0.05,
                                    label="Journey start alpha")
                                vs_journey_end = gr.Slider(
                                    0.0, 1.0, value=1.0, step=0.05,
                                    label="Journey end alpha")
                            vs_journey_curve = gr.Dropdown(
                                choices=["linear", "smoothstep"],
                                value="linear",
                                label="Journey curve",
                                info="linear = constant rate. "
                                     "smoothstep = ease-in/out S-curve "
                                     "(slow start, slow end).")
                            vs_source_b_preview = gr.Image(
                                interactive=False, height=200,
                                type="filepath", show_label=False,
                                show_download_button=False,
                                container=False,
                            )
                    # ---- COLUMN 3: OUTPUT ----
                    with gr.Column(scale=1):
                        gr.Markdown("### 3. Output")
                        vs_out_video = gr.Video(
                            label="Result", interactive=False,
                            show_download_button=True, height=320)
                        # T1-1 cross-tab handoff: take the face-swap
                        # output and use it as the face clip on Lip-Sync.
                        fs_send_to_ls_btn = gr.Button(
                            "Send result to Lip-Sync →",
                            variant="secondary", size="sm")
                        vs_btn = gr.Button(
                            "\u25b6 Run face swap",
                            variant="primary", size="lg")
                        vs_status = gr.Markdown(
                            "Drop a source face + target video.")

                vs_btn.click(
                    fn=_run_video_swap,
                    inputs=[vs_source, vs_target,
                            vs_blend, vs_enhance,
                            vs_det_thresh,
                            vs_quality, vs_trim_start, vs_trim_end,
                            vs_source_b, vs_blend_alpha,
                            vs_journey_mode, vs_journey_start,
                            vs_journey_end, vs_journey_curve,
                            vs_mask_padding, vs_mask_blur,
                            vs_swap_strength, vs_enhancer_blend,
                            vs_selector_mode, vs_reference_face,
                            vs_reference_distance, vs_pixel_boost,
                            vs_temporal_enabled, vs_temporal_ema,
                            vs_temporal_buffer,
                            vs_color_transfer_mode, vs_shadow_correction,
                            vs_shadow_clamp_min, vs_shadow_clamp_max,
                            vs_restorer, vs_mask_npy],
                    outputs=[vs_out_video, vs_status])

                # Mirror reference face upload into its thumbnail.
                vs_reference_face.change(
                    fn=lambda p: p,
                    inputs=[vs_reference_face],
                    outputs=[vs_reference_preview])

                # Mirror Source B upload into its preview thumbnail.
                vs_source_b.change(
                    fn=lambda p: p,
                    inputs=[vs_source_b],
                    outputs=[vs_source_b_preview])

                # Passthrough preview: just mirror the uploaded path
                # into the preview Image. Zero re-encoding of the
                # actual source file.
                vs_source.change(
                    fn=lambda p: (p if (p and Path(str(p)).is_file())
                                  else None),
                    inputs=[vs_source],
                    outputs=[vs_source_preview])

            # ====================================================
            # TAB 3: Webcam (v2-OWNED -- zero v1 imports)
            # ====================================================
            with gr.Tab("📷 Webcam"):
                # v2's own webcam tab. Source: faceswap/webcam/ui.py.
                # The /webcam_stream/* FastAPI routes are registered
                # in v2/launch.py via faceswap.webcam.streaming.
                # Heavy ML (face detection + swap) routes through
                # core.pipeline only -- same module v2 Face Swap tab
                # already uses. NO imports from ui.app or
                # ui.stream_server anywhere.
                from .webcam.ui import build_tab as _v2_webcam_tab
                _v2_webcam_tab()
            # TAB 3.75: Rotoscoping (Phase 1 MVP)
            # ====================================================
            # SAM2-backed segmentation + propagation.  Produces a
            # mask sequence that the Lip-Sync tab can consume via
            # the existing mask-out / composite-back pipeline.
            # Source: faceswap/rotoscope/ui.py.  Uses the SAM2
            # daemon (core/sam2_daemon.py) for sub-second clicks
            # after warmup, and the frame cache
            # (core/rotoscope_cache.py) for instant scrub.
            with gr.Tab("🪒 Rotoscoping"):
                from .rotoscope.ui import build_tab as _roto_tab
                # T1-1 + T2-NEW handoff: rotoscope's "Send masks" button
                # writes the NPY path into BOTH the Lip-Sync mask-out
                # textbox AND the Face Swap region-restrict textbox.
                _roto_tab(lipsync_npy_target=mout_npy_override,
                            faceswap_npy_target=vs_mask_npy)
            # (Creature Swap tab ripped out 2026-06-11 -- pipeline
            # could not handle non-human anatomies; both SAM2 mask
            # centroid and Lucas-Kanade point tracking failed on
            # stylized creatures.  See WORKLOG.)
            # TAB 3.5: Queue (batch render)
            # ====================================================
            with gr.Tab("📦 Queue"):
                gr.Markdown("### Render queue")
                gr.Markdown(
                    "_Drop N jobs (each via Add to queue on the "
                    "Lip-Sync tab), walk away. Jobs run serially on "
                    "the GPU. Click Refresh to update; cancel by id._"
                )
                with gr.Row():
                    q_refresh = gr.Button("\U0001F504 Refresh",
                                          variant="primary")
                    q_clear   = gr.Button("\U0001F9F9 Clear done",
                                          variant="secondary")
                q_table = gr.Dataframe(
                    headers=["id", "label", "status",
                             "elapsed", "result/error"],
                    datatype=["str","str","str","str","str"],
                    value=job_queue.jobs_as_rows(),
                    interactive=False, wrap=True)
                with gr.Row():
                    q_cancel_id = gr.Textbox(
                        label="Cancel job by id",
                        placeholder="e.g. abc12345", scale=2)
                    q_cancel_btn = gr.Button("\u2716 Cancel",
                                              variant="stop",
                                              scale=1)
                q_status = gr.Markdown("")

                q_refresh.click(fn=_refresh_queue, outputs=[q_table])
                q_clear.click(fn=_clear_completed,
                              outputs=[q_status, q_table])
                q_cancel_btn.click(fn=_cancel_queue_job,
                                    inputs=[q_cancel_id],
                                    outputs=[q_status, q_table])

            # TAB 4: History
            # ====================================================
            with gr.Tab("📜 History"):
                gr.Markdown("### Past renders")
                gr.Markdown(
                    "_Every successful render writes a sidecar "
                    "JSON with its full job config. Click any past "
                    "render to replay it and view its sidecar._"
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        hist_dd = gr.Dropdown(
                            choices=hist_choices, value=None,
                            label="Past renders (newest first)",
                            interactive=True)
                        refresh_btn = gr.Button("🔄 Refresh",
                                                variant="secondary")
                        hist_video = gr.Video(
                            label="Replay", interactive=False,
                            show_download_button=True, height=380)
                    with gr.Column(scale=1):
                        hist_meta = gr.Markdown("_(none selected)_")
                # T1-2 restore button: read the selected mp4's sidecar
                # and dispatch every saved knob value back into the
                # source tab's widgets.
                hist_restore_btn = gr.Button(
                    "🔧 Restore settings to source tab",
                    variant="secondary", size="sm")
                hist_restore_status = gr.Markdown(
                    "_Select a past render above, then click Restore "
                    "to push its knob values back into the Lip-Sync "
                    "or Face Swap tab (based on the render kind)._")
                refresh_btn.click(fn=_refresh_history,
                                  outputs=[hist_dd])
                hist_dd.change(fn=_on_history_select,
                               inputs=[hist_dd],
                               outputs=[hist_video, hist_meta])

            # ====================================================
            # TAB 4.5: Projects (T1-3) -- session bundle: every knob
            # value across Lip-Sync + Face Swap saved to a single
            # project.json, restorable in one click across both tabs.
            # ====================================================
            with gr.Tab("📂 Projects"):
                gr.Markdown("### Project sessions")
                gr.Markdown(
                    "_Save the **current Lip-Sync + Face Swap** "
                    "settings as a named project.  Open it later to "
                    "restore every knob value across both tabs in one "
                    "click.  Projects live under `v2/projects/`._\n\n"
                    "_File paths are noted in project.json for "
                    "reference but files are NOT auto-copied (MVP).  "
                    "If you move source / target files later, "
                    "re-upload them after Open._")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Save current settings**")
                        proj_save_name = gr.Textbox(
                            label="Save as",
                            placeholder="e.g. acme-promo-spot-q3")
                        proj_save_btn = gr.Button(
                            "💾 Save current settings as project",
                            variant="primary")
                        proj_save_status = gr.Markdown("")
                    with gr.Column(scale=1):
                        gr.Markdown("**Open / manage existing projects**")
                        from . import projects as _projects_module
                        proj_dd = gr.Dropdown(
                            choices=_projects_module.list_projects(),
                            value=None,
                            label="Existing projects (newest first)",
                            allow_custom_value=False, interactive=True)
                        with gr.Row():
                            proj_refresh = gr.Button("🔄 Refresh",
                                                       size="sm")
                            proj_open_btn = gr.Button(
                                "📂 Open project",
                                variant="secondary")
                            proj_delete_btn = gr.Button(
                                "🗑 Delete", variant="stop", size="sm")
                        proj_manage_status = gr.Markdown("")

            # ====================================================
            # TAB 5: Presets (full management)
            # ====================================================
            with gr.Tab("⚙️ Presets"):
                gr.Markdown("### Named-preset manager")
                gr.Markdown(
                    "_Save the current Lip-Sync tab settings under "
                    "a name. Load later to restore in one click. "
                    "JSON files land in `v2/presets/`._"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Save current settings**")
                        preset_name = gr.Textbox(
                            label="Save as",
                            placeholder="e.g. fast-test, best-quality")
                        save_btn = gr.Button(
                            "💾 Save current settings as preset",
                            variant="primary")
                        preset_save_status = gr.Markdown("")

                    with gr.Column(scale=1):
                        gr.Markdown("**Manage existing presets**")
                        preset_dd = gr.Dropdown(
                            choices=preset_choices, value=None,
                            label="Existing presets",
                            allow_custom_value=False, interactive=True)
                        with gr.Row():
                            preset_refresh = gr.Button("🔄 Refresh")
                            del_btn = gr.Button("🗑 Delete",
                                                variant="stop")
                        preset_manage_status = gr.Markdown("")

                # NOTE: save_btn here needs Lip-Sync tab widget refs;
                # the inline preset apply on the Lip-Sync tab covers
                # load. For save, we read from the inline dropdown's
                # OWN current state via a tiny shim. To keep things
                # simple in v2.2 the SAVE here writes a preset named
                # whatever you typed, using the inline dropdown's
                # current loaded preset values OR safe defaults.
                # (Full cross-tab state wiring would require Gradio
                # State -- see v2.3 roadmap in About tab.)
                def _save_from_defaults(name):
                    if not name or not name.strip():
                        return ("**Error:** preset name required.",
                                gr.update())
                    out = presets.save_preset(
                        name, isolate=True, quick=False, enhance=True,
                        extend_single=False, ls_steps=20,
                        ls_guidance=1.5, ls_deepcache=True, ls_seed=-1,
                        voice_model="", voice_transpose=0)
                    return (f"saved `{Path(out).stem}` "
                            "(defaults — go to Lip-Sync tab, "
                            "tweak, and save from there for a custom "
                            "preset in v2.3)",
                            gr.update(choices=presets.list_presets(),
                                      value=presets._safe_name(name)))

                save_btn.click(
                    fn=_save_from_defaults,
                    inputs=[preset_name],
                    outputs=[preset_save_status, preset_dd])
                del_btn.click(
                    fn=_on_delete_preset,
                    inputs=[preset_dd],
                    outputs=[preset_manage_status, preset_dd])
                preset_refresh.click(
                    fn=lambda: gr.update(
                        choices=presets.list_presets(), value=None),
                    outputs=[preset_dd])

            # ====================================================
            # TAB 6: Cache & Cleanup
            # ====================================================
            with gr.Tab("Cache"):
                gr.Markdown("### Gradio cache cleaner")
                gr.Markdown(
                    "_Gradio stages every upload under "
                    "`%TEMP%/gradio/` and never auto-prunes. Delete "
                    "files older than the cutoff. Newer files are "
                    "kept in case an upload is in flight._"
                )
                with gr.Row():
                    min_age = gr.Number(
                        value=10, precision=0,
                        label="Min age (minutes)")
                    clear_btn = gr.Button(
                        "Clear cache", variant="secondary")
                cache_out = gr.Textbox(
                    label="Status", interactive=False, lines=3)
                clear_btn.click(
                    fn=_clear_gradio_cache,
                    inputs=[min_age],
                    outputs=[cache_out],
                )

        # ---- T1-2 History "Restore settings" wiring (defined after all
        # tabs so widget refs from Lip-Sync + Face Swap are in scope).
        # The function reads the selected mp4's sidecar JSON and returns
        # a flat tuple of gr.update() calls -- one per output widget in
        # the order they're bound below.
        def _on_history_restore(selected_path):
            import gradio as _gr
            n_outs = 33  # MUST match outputs= list below
            noop = [_gr.update() for _ in range(n_outs)]
            if not selected_path:
                return tuple(noop) + ("_Restore: nothing selected._",)
            try:
                sc = previews.load_sidecar(Path(selected_path))
            except Exception as exc:
                return tuple(noop) + (f"_Restore: load failed: {exc}_",)
            if sc is None:
                return tuple(noop) + (
                    "_Restore: no sidecar found for that render._",)
            kind = sc.get("kind", "lipsync")
            updates = list(noop)
            # Output order (must match outputs= list below):
            #   0..7  -- lipsync widgets
            #   8..32 -- face-swap widgets
            if kind == "lipsync":
                ls = sc.get("latentsync", {}) or {}
                updates[0] = _gr.update(value=bool(sc.get("isolate_vocals", True)))
                updates[1] = _gr.update(value=bool(sc.get("quick_test", False)))
                updates[2] = _gr.update(value=bool(sc.get("enhance_faces", True)))
                updates[3] = _gr.update(value=bool(sc.get("extend_single", False)))
                updates[4] = _gr.update(value=int(ls.get("inference_steps", 20)))
                updates[5] = _gr.update(value=float(ls.get("guidance_scale", 1.5)))
                updates[6] = _gr.update(value=bool(ls.get("deepcache", True)))
                updates[7] = _gr.update(value=int(ls.get("seed", -1)))
                status = (f"_Restored Lip-Sync settings from "
                           f"`{Path(selected_path).name}` "
                           f"(steps={ls.get('inference_steps', '?')}, "
                           f"seed={ls.get('seed', '?')})._")
            elif kind == "face_swap":
                g = lambda k, d=None: sc.get(k, d)
                updates[8]  = _gr.update(value=str(g("blend_method", "poisson")))
                updates[9]  = _gr.update(value=bool(g("enhance_faces", False)))
                updates[10] = _gr.update(value=float(g("det_threshold", 0.5)))
                updates[11] = _gr.update(value=str(g("output_quality",
                                                        "visually_lossless")))
                updates[12] = _gr.update(value=int(g("trim_start_frame", 0)))
                updates[13] = _gr.update(value=int(g("trim_end_frame", 0)))
                updates[14] = _gr.update(value=str(g("selector_mode", "largest")))
                updates[15] = _gr.update(value=float(g("reference_distance", 0.6)))
                updates[16] = _gr.update(value=int(g("pixel_boost", 128)))
                updates[17] = _gr.update(value=int(g("mask_padding", 0)))
                updates[18] = _gr.update(value=float(g("mask_blur", 1.0)))
                updates[19] = _gr.update(value=float(g("swap_strength", 1.0)))
                updates[20] = _gr.update(value=float(g("enhancer_blend", 1.0)))
                updates[21] = _gr.update(value=bool(g("temporal_enabled", True)))
                updates[22] = _gr.update(value=float(g("temporal_ema_decay", 0.85)))
                updates[23] = _gr.update(value=int(g("temporal_buffer_size", 5)))
                updates[24] = _gr.update(value=str(g("color_transfer_mode",
                                                        "reinhard")))
                updates[25] = _gr.update(value=bool(g("shadow_correction", True)))
                updates[26] = _gr.update(value=float(g("shadow_clamp_min", 0.5)))
                updates[27] = _gr.update(value=float(g("shadow_clamp_max", 1.5)))
                updates[28] = _gr.update(value=float(g("blend_alpha", 0.5)))
                updates[29] = _gr.update(value=bool(g("journey_mode", False)))
                updates[30] = _gr.update(value=float(g("journey_start_alpha", 0.0)))
                updates[31] = _gr.update(value=float(g("journey_end_alpha", 1.0)))
                updates[32] = _gr.update(value=str(g("journey_curve", "linear")))
                status = (f"_Restored Face Swap settings from "
                           f"`{Path(selected_path).name}`._")
            else:
                return tuple(noop) + (f"_Restore: unknown kind={kind!r}_",)
            return tuple(updates) + (status,)

        hist_restore_btn.click(
            fn=_on_history_restore,
            inputs=[hist_dd],
            outputs=[
                # lipsync (0..7)
                isolate, quick, enhance, extend_single,
                ls_steps, ls_guidance, ls_deepcache, ls_seed,
                # face-swap (8..32)
                vs_blend, vs_enhance, vs_det_thresh, vs_quality,
                vs_trim_start, vs_trim_end,
                vs_selector_mode, vs_reference_distance, vs_pixel_boost,
                vs_mask_padding, vs_mask_blur, vs_swap_strength, vs_enhancer_blend,
                vs_temporal_enabled, vs_temporal_ema, vs_temporal_buffer,
                vs_color_transfer_mode, vs_shadow_correction,
                vs_shadow_clamp_min, vs_shadow_clamp_max,
                vs_blend_alpha,
                vs_journey_mode, vs_journey_start, vs_journey_end, vs_journey_curve,
                # status (33)
                hist_restore_status,
            ],
        )

        # ---- T1-3 Projects: Save / Open / Delete / Refresh.
        # Defined after all tabs so every Lip-Sync + Face Swap widget ref
        # is in scope.  Save reads 33 inputs; Open returns 33 gr.updates.
        from . import projects as _projects_mod

        def _on_project_save(name,
                               # lipsync inputs:
                               iso, qk, eh, ex,
                               lss, lsg, lsd, lse,
                               # face-swap inputs:
                               vsb, vse, vsdt, vsq,
                               vsts, vste,
                               vsm, vsrd, vsbp,
                               vsmp, vsmb, vsss, vseb,
                               vste_en, vste_em, vste_bf,
                               vsct, vssc, vsscn, vsscx,
                               vsba,
                               vsjm, vsjs, vsje, vsjc):
            if not str(name or "").strip():
                return ("_Save: name required._",
                         gr.update())
            blob = {
                "lipsync": {
                    "isolate_vocals": bool(iso),
                    "quick_test": bool(qk),
                    "enhance_faces": bool(eh),
                    "extend_single": bool(ex),
                    "latentsync": {
                        "inference_steps": int(lss),
                        "guidance_scale": float(lsg),
                        "deepcache": bool(lsd),
                        "seed": int(lse),
                    },
                },
                "face_swap": {
                    "blend_method": str(vsb),
                    "enhance_faces": bool(vse),
                    "det_threshold": float(vsdt),
                    "output_quality": str(vsq),
                    "trim_start_frame": int(vsts),
                    "trim_end_frame": int(vste),
                    "selector_mode": str(vsm),
                    "reference_distance": float(vsrd),
                    "pixel_boost": int(vsbp),
                    "mask_padding": int(vsmp),
                    "mask_blur": float(vsmb),
                    "swap_strength": float(vsss),
                    "enhancer_blend": float(vseb),
                    "temporal_enabled": bool(vste_en),
                    "temporal_ema_decay": float(vste_em),
                    "temporal_buffer_size": int(vste_bf),
                    "color_transfer_mode": str(vsct),
                    "shadow_correction": bool(vssc),
                    "shadow_clamp_min": float(vsscn),
                    "shadow_clamp_max": float(vsscx),
                    "blend_alpha": float(vsba),
                    "journey_mode": bool(vsjm),
                    "journey_start_alpha": float(vsjs),
                    "journey_end_alpha": float(vsje),
                    "journey_curve": str(vsjc),
                },
            }
            try:
                manifest = _projects_mod.save_project(str(name), blob)
            except Exception as exc:
                return (f"_Save failed: {exc}_", gr.update())
            choices = _projects_mod.list_projects()
            # Auto-select the newly-saved project in the dropdown.
            saved_name = _projects_mod._safe_name(str(name))
            status = f"_Saved `{saved_name}` → `{manifest}`_"
            return (status,
                     gr.update(choices=choices, value=saved_name))

        proj_save_btn.click(
            fn=_on_project_save,
            inputs=[proj_save_name,
                    # lipsync (8)
                    isolate, quick, enhance, extend_single,
                    ls_steps, ls_guidance, ls_deepcache, ls_seed,
                    # face-swap (25)
                    vs_blend, vs_enhance, vs_det_thresh, vs_quality,
                    vs_trim_start, vs_trim_end,
                    vs_selector_mode, vs_reference_distance, vs_pixel_boost,
                    vs_mask_padding, vs_mask_blur, vs_swap_strength, vs_enhancer_blend,
                    vs_temporal_enabled, vs_temporal_ema, vs_temporal_buffer,
                    vs_color_transfer_mode, vs_shadow_correction,
                    vs_shadow_clamp_min, vs_shadow_clamp_max,
                    vs_blend_alpha,
                    vs_journey_mode, vs_journey_start, vs_journey_end, vs_journey_curve],
            outputs=[proj_save_status, proj_dd],
        )

        def _on_project_open(name):
            import gradio as _gr
            n_outs = 33
            noop = [_gr.update() for _ in range(n_outs)]
            if not name:
                return tuple(noop) + ("_Open: select a project first._",)
            blob = _projects_mod.load_project(str(name))
            if not blob:
                return tuple(noop) + (
                    f"_Open: project `{name}` not found / unreadable._",)
            ls = (blob.get("lipsync") or {}) if isinstance(blob, dict) else {}
            ls_inner = ls.get("latentsync") or {}
            fs = (blob.get("face_swap") or {}) if isinstance(blob, dict) else {}
            updates = list(noop)
            # Lipsync (0..7)
            updates[0] = _gr.update(value=bool(ls.get("isolate_vocals", True)))
            updates[1] = _gr.update(value=bool(ls.get("quick_test", False)))
            updates[2] = _gr.update(value=bool(ls.get("enhance_faces", True)))
            updates[3] = _gr.update(value=bool(ls.get("extend_single", False)))
            updates[4] = _gr.update(value=int(ls_inner.get("inference_steps", 20)))
            updates[5] = _gr.update(value=float(ls_inner.get("guidance_scale", 1.5)))
            updates[6] = _gr.update(value=bool(ls_inner.get("deepcache", True)))
            updates[7] = _gr.update(value=int(ls_inner.get("seed", -1)))
            # Face-swap (8..32)
            g = lambda k, d=None: fs.get(k, d)
            updates[8]  = _gr.update(value=str(g("blend_method", "poisson")))
            updates[9]  = _gr.update(value=bool(g("enhance_faces", False)))
            updates[10] = _gr.update(value=float(g("det_threshold", 0.5)))
            updates[11] = _gr.update(value=str(g("output_quality", "visually_lossless")))
            updates[12] = _gr.update(value=int(g("trim_start_frame", 0)))
            updates[13] = _gr.update(value=int(g("trim_end_frame", 0)))
            updates[14] = _gr.update(value=str(g("selector_mode", "largest")))
            updates[15] = _gr.update(value=float(g("reference_distance", 0.6)))
            updates[16] = _gr.update(value=int(g("pixel_boost", 128)))
            updates[17] = _gr.update(value=int(g("mask_padding", 0)))
            updates[18] = _gr.update(value=float(g("mask_blur", 1.0)))
            updates[19] = _gr.update(value=float(g("swap_strength", 1.0)))
            updates[20] = _gr.update(value=float(g("enhancer_blend", 1.0)))
            updates[21] = _gr.update(value=bool(g("temporal_enabled", True)))
            updates[22] = _gr.update(value=float(g("temporal_ema_decay", 0.85)))
            updates[23] = _gr.update(value=int(g("temporal_buffer_size", 5)))
            updates[24] = _gr.update(value=str(g("color_transfer_mode", "reinhard")))
            updates[25] = _gr.update(value=bool(g("shadow_correction", True)))
            updates[26] = _gr.update(value=float(g("shadow_clamp_min", 0.5)))
            updates[27] = _gr.update(value=float(g("shadow_clamp_max", 1.5)))
            updates[28] = _gr.update(value=float(g("blend_alpha", 0.5)))
            updates[29] = _gr.update(value=bool(g("journey_mode", False)))
            updates[30] = _gr.update(value=float(g("journey_start_alpha", 0.0)))
            updates[31] = _gr.update(value=float(g("journey_end_alpha", 1.0)))
            updates[32] = _gr.update(value=str(g("journey_curve", "linear")))
            status = (f"_Opened **{name}** → restored "
                       f"8 Lip-Sync + 25 Face Swap knobs._")
            return tuple(updates) + (status,)

        proj_open_btn.click(
            fn=_on_project_open,
            inputs=[proj_dd],
            outputs=[
                # lipsync (0..7)
                isolate, quick, enhance, extend_single,
                ls_steps, ls_guidance, ls_deepcache, ls_seed,
                # face-swap (8..32)
                vs_blend, vs_enhance, vs_det_thresh, vs_quality,
                vs_trim_start, vs_trim_end,
                vs_selector_mode, vs_reference_distance, vs_pixel_boost,
                vs_mask_padding, vs_mask_blur, vs_swap_strength, vs_enhancer_blend,
                vs_temporal_enabled, vs_temporal_ema, vs_temporal_buffer,
                vs_color_transfer_mode, vs_shadow_correction,
                vs_shadow_clamp_min, vs_shadow_clamp_max,
                vs_blend_alpha,
                vs_journey_mode, vs_journey_start, vs_journey_end, vs_journey_curve,
                # status (33)
                proj_manage_status,
            ],
        )

        def _on_project_refresh():
            return gr.update(choices=_projects_mod.list_projects())

        proj_refresh.click(fn=_on_project_refresh, outputs=[proj_dd])

        def _on_project_delete(name):
            if not name:
                return ("_Delete: select a project first._",
                         gr.update())
            ok = _projects_mod.delete_project(str(name))
            msg = (f"_Deleted `{name}`._" if ok
                    else f"_Delete failed: `{name}` not found._")
            return (msg, gr.update(
                choices=_projects_mod.list_projects(), value=None))

        proj_delete_btn.click(
            fn=_on_project_delete,
            inputs=[proj_dd],
            outputs=[proj_manage_status, proj_dd],
        )

        # ---- T1-1 cross-tab handoff wirings (defined after all tabs
        # so widget refs are in scope).  Click → pass video path as-is.
        ls_send_to_fs_btn.click(
            fn=lambda v: v,
            inputs=[out_video],
            outputs=[vs_target],
        )
        fs_send_to_ls_btn.click(
            fn=lambda v: v,
            inputs=[vs_out_video],
            outputs=[face],
        )

    app.queue()
    return app


__all__ = ["build"]
