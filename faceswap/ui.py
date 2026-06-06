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
from .config import (LatentSyncKnobs, LipsyncJob,
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
                 isolate, quick, enhance, extend_single,
                 ls_steps, ls_guidance, ls_deepcache, ls_seed,
                 ls_face_det,
                 voice_model, voice_transpose,
                 wm_enabled, wm_image, wm_position, wm_scale, wm_opacity,
                 ar_enabled, ar_target, ar_fill,
                 occ_enabled, occ_bbox_smooth, occ_mask_smooth,
                 occ_align, occ_feather, occ_mouth_polygon,
                 mout_enabled_v, mout_clicks_v,
                 mout_dilate_v, mout_feather_v):
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
        isolate_vocals=bool(isolate), enhance_faces=bool(enhance),
        quick_test=bool(quick), extend_single=bool(extend_single),
        latentsync=LatentSyncKnobs(
            inference_steps=int(ls_steps),
            guidance_scale=float(ls_guidance),
            enable_deepcache=bool(ls_deepcache),
            seed=int(ls_seed),
            face_det_threshold=float(ls_face_det),
        ),
        maskout=MaskOutConfig(
            enabled=bool(mout_enabled_v),
            clicks=list(mout_clicks_v or []),
            dilate_px=int(mout_dilate_v),
            feather=int(mout_feather_v),
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
                     ls_steps, ls_guidance, ls_deepcache, ls_seed,
                 ls_face_det,
                     voice_model, voice_transpose):
    if not name or not name.strip():
        return ("**Error:** preset name required.",
                gr.update(choices=presets.list_presets()))
    out = presets.save_preset(
        name, isolate, quick, enhance, extend_single,
        ls_steps, ls_guidance, ls_deepcache, ls_seed,
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
                     reference_distance, pixel_boost):
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
    yield str(out_path), status

# ============================================================
# Queue handlers (Tab: Queue)
# ============================================================
def _enqueue_render(face, audio, face_extras,
                     isolate, quick, enhance, extend_single,
                     ls_steps, ls_guidance, ls_deepcache, ls_seed,
                     voice_model, voice_transpose,
                     wm_enabled, wm_image, wm_position, wm_scale, wm_opacity,
                     ar_enabled, ar_target, ar_fill,
                     occ_enabled, occ_bbox_smooth, occ_mask_smooth,
                     occ_align, occ_feather, occ_mouth_polygon,
                 mout_enabled_v, mout_clicks_v,
                 mout_dilate_v, mout_feather_v):
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
        isolate_vocals=bool(isolate), enhance_faces=bool(enhance),
        quick_test=bool(quick), extend_single=bool(extend_single),
        latentsync=LatentSyncKnobs(
            inference_steps=int(ls_steps),
            guidance_scale=float(ls_guidance),
            enable_deepcache=bool(ls_deepcache),
            seed=int(ls_seed),
            face_det_threshold=float(ls_face_det)),
        maskout=MaskOutConfig(
            enabled=bool(mout_enabled_v),
            clicks=list(mout_clicks_v or []),
            dilate_px=int(mout_dilate_v),
            feather=int(mout_feather_v)),
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
    'position:fixed;top:12px;right:12px;z-index:10000;' +
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
                            isolate, quick, enhance, extend_single,
                            ls_steps, ls_guidance, ls_deepcache, ls_seed,
                            ls_face_det,
                            voice_model, voice_transpose,
                            wm_enabled, wm_image, wm_position,
                            wm_scale, wm_opacity,
                            ar_enabled, ar_target, ar_fill,
                            occ_enabled, occ_bbox_smooth,
                            occ_mask_smooth, occ_align,
                            occ_feather, occ_mouth_polygon,
                            mout_enabled, mout_clicks_state,
                            mout_dilate, mout_feather],
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
                            isolate, quick, enhance, extend_single,
                            ls_steps, ls_guidance, ls_deepcache, ls_seed,
                            ls_face_det,
                            voice_model, voice_transpose,
                            wm_enabled, wm_image, wm_position,
                            wm_scale, wm_opacity,
                            ar_enabled, ar_target, ar_fill,
                            occ_enabled, occ_bbox_smooth,
                            occ_mask_smooth, occ_align,
                            occ_feather, occ_mouth_polygon,
                            mout_enabled, mout_clicks_state,
                            mout_dilate, mout_feather],
                    outputs=[out_status],
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
                             ls_seed, voice_model, voice_transpose,
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
                                label="Enhance faces (GFPGAN)",
                                value=False)
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
                            vs_reference_distance, vs_pixel_boost],
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
                refresh_btn.click(fn=_refresh_history,
                                  outputs=[hist_dd])
                hist_dd.change(fn=_on_history_select,
                               inputs=[hist_dd],
                               outputs=[hist_video, hist_meta])

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

    app.queue()
    return app


__all__ = ["build"]
