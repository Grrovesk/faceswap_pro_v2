"""Orchestrator -- the only module that decides which render shape
runs for a given job. Stages flow as pure functions taking a
LipsyncJob and returning a Path.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, List

from . import latentsync
from .config import LipsyncJob

# ---- Global render serializer ----
# orchestrator.render() can be invoked from two paths simultaneously:
#   (1) the Run button in the lipsync tab (foreground worker thread)
#   (2) the queue worker draining job_queue
# Without serialization both fire two LatentSync subprocesses
# concurrently, sharing GPU / colliding on canonical output paths.
# This non-reentrant lock guarantees one render at a time across
# every caller. The contended caller logs "waiting on in-flight
# render" once and then blocks until the lock releases.
_RENDER_LOCK = threading.Lock()
from .ffmpeg_tools import (
    concat_videos, loop_video_to_duration, probe_duration_seconds,
    replace_audio_track, slice_audio_to_wav,
)
from .gfpgan import enhance as gfpgan_enhance
from .paths import (
    EXTEND_SINGLE_WORK, LATENTSYNC_SCRATCH, MULTICLIP_WORK,
    RECORDINGS_DIR, ensure_all,
)
from .vocal_isolation import isolate as isolate_vocals_fn
from .voice_swap import apply_voice_swap


class RenderCancelled(Exception):
    """Raised when the user clicks Cancel mid-render. Caught by the
    UI layer to show 'Cancelled' status instead of an error trace."""


def _check_cancel(cancel_event, log):
    """Stage-boundary cancellation check. Called before/after each
    stage so an in-progress render can stop at the next safe point."""
    if cancel_event is not None and cancel_event.is_set():
        log("[orchestrator] CANCEL signal received -- stopping render")
        raise RenderCancelled("user cancelled")


def render(job: LipsyncJob,
           log: Callable[[str], None] = print,
           cancel_event=None) -> Path:
    """Top-level entry. Serialized against other in-flight renders
    via _RENDER_LOCK -- if another render is currently running
    (foreground or queue worker), this call logs the wait once and
    blocks until the in-flight render completes. The cancel_event
    is honored DURING the wait too, so a queued job can be cancelled
    before it ever starts.
    """
    if not _RENDER_LOCK.acquire(blocking=False):
        log("[orchestrator] another render is in progress -- "
            "waiting (queue + foreground are serialized to one "
            "render at a time to avoid GPU + output-path collision)")
        _wait_start = time.perf_counter()
        while True:
            if cancel_event is not None and cancel_event.is_set():
                log("[orchestrator] CANCEL while waiting on render "
                    "lock -- aborting before this job ever started")
                raise RenderCancelled("user cancelled while waiting")
            if _RENDER_LOCK.acquire(timeout=0.5):
                _waited = time.perf_counter() - _wait_start
                log(f"[orchestrator] render lock acquired after "
                    f"{_waited:.1f}s wait; proceeding")
                break
    try:
        return _render_locked(job, log=log, cancel_event=cancel_event)
    finally:
        try:
            _RENDER_LOCK.release()
        except RuntimeError:
            pass


def _render_locked(job: LipsyncJob,
           log: Callable[[str], None] = print,
           cancel_event=None) -> Path:
    """Top-level entry. Validates, branches, runs stages, returns the
    final mp4 path inside RECORDINGS_DIR."""
    job.validate()
    ensure_all()

    render_t0 = time.perf_counter()
    stage_times: List = []

    _check_cancel(cancel_event, log)
    # Stage 1 (always): optional voice swap. effective_audio after
    # this is the full song with cloned vocals if swap is on, else
    # the original song. THIS is what should end up on the output mp4.
    t0 = time.perf_counter()
    job.effective_audio = apply_voice_swap(
        job.audio_path, job.voice_swap, log=log)
    dt = time.perf_counter() - t0
    log(f"[timing] Stage 1 voice_swap took {dt:.2f}s")
    stage_times.append(("Stage 1 voice_swap", dt))

    # Save the FINAL audio (what gets muxed onto the output mp4)
    # BEFORE isolation strips the instrumental.
    job.final_audio = job.effective_audio

    # Stage 1.5: vocal isolation BEFORE lipsync. Drives LatentSync's
    # Whisper feature extractor with clean vocals so the audio
    # conditioning doesn't collapse under instrument noise.
    # effective_audio is updated to vocals.wav (driver only); the
    # ORIGINAL song is preserved in final_audio for the post-render
    # mux step.
    _check_cancel(cancel_event, log)
    if job.isolate_vocals:
        log("[orchestrator] isolating vocals (Demucs) -- DRIVER only; "
            "full song preserved for output mux ...")
        t0 = time.perf_counter()
        job.effective_audio = isolate_vocals_fn(
            job.effective_audio, log=log)
        dt = time.perf_counter() - t0
        log(f"[timing] Stage 1.5 vocal_isolation took {dt:.2f}s")
        stage_times.append(("Stage 1.5 vocal_isolation", dt))

    # Stage 1.75 REMOVED: XSeg occluder pre-inpaint is gone.
    # If you re-enable in the future, restore from
    # core/xseg_gate.py.broken_2026_06_02 backup.
    inpainted_face_paths = None

    # Stage 1.76 RE-ENABLED 2026-06-05 after root-causing the prior
    # break. The bug was NOT audio desync (measured 0.19 ms offset);
    # it was source/mask frame-index drift in composite_back. Source
    # at 30fps + LatentSync output at 25fps + linear i%n_src indexing
    # painted source pixels from the wrong wall-clock moment over
    # each lipsync frame. Fixed in core/maskout_pipeline.py via
    # i*(fps_src/fps_lip) % n_src so lookups happen at the lipsync
    # frame's true time. If you need to disable again to A/B, flip
    # this flag.
    DISABLE_MASKOUT_PIPELINE = False
    _check_cancel(cancel_event, log)
    maskout_artifacts = None
    _gate_disabled = DISABLE_MASKOUT_PIPELINE
    _gate_enabled  = bool(job.maskout.enabled)
    _gate_nclicks  = len(job.maskout.clicks) if job.maskout.clicks else 0
    log(f"[orchestrator] Stage 1.76 gate: disable_flag={_gate_disabled} "
        f"job.maskout.enabled={_gate_enabled} "
        f"job.maskout.clicks={_gate_nclicks}")
    if (not _gate_disabled and _gate_enabled and _gate_nclicks > 0):
        log("[orchestrator] Stage 1.76: SAM2 mask-out pre-stage")
        t0 = time.perf_counter()
        try:
            from core import maskout_pipeline as _mout
            ws = MULTICLIP_WORK / f"_maskout_{int(time.time())}"
            ws.mkdir(parents=True, exist_ok=True)
            # Normalize clicks to 4-tuples for the SAM2 worker.
            _normalized = []
            for c in job.maskout.clicks:
                if isinstance(c, dict):
                    _normalized.append((
                        int(c.get("x", 0)), int(c.get("y", 0)),
                        int(c.get("frame", 0)), int(c.get("label", 1))))
                else:
                    seq = list(c)
                    if len(seq) == 3:
                        seq.append(1)
                    _normalized.append(tuple(int(v) for v in seq[:4]))
            primary_src = Path(job.face_paths[0])
            result = _mout.run_pipeline(
                source_video=primary_src,
                clicks=_normalized,
                workspace=ws,
                dilate_px=int(job.maskout.dilate_px),
                feather=int(job.maskout.feather),
                log=log,
            )
            # Swap face_paths so Stage 2 (LatentSync) reads the void.
            job.face_paths = [Path(result["void_video"])] + \
                              list(job.face_paths[1:])
            maskout_artifacts = {
                "original_source": primary_src,
                "masks_npy": result["masks_npy"],
                "feather": int(job.maskout.feather),
            }
            dt = time.perf_counter() - t0
            log(f"[timing] Stage 1.76 maskout_prep took {dt:.2f}s "
                f"(void source: {Path(result['void_video']).name})")
            stage_times.append(("Stage 1.76 maskout_prep", dt))
        except Exception as _mout_exc:
            log(f"[orchestrator] Stage 1.76 maskout FAILED "
                f"({_mout_exc}); proceeding with original source")
            maskout_artifacts = None

    # Stage 1.9 (OPTIONAL): quick_test audio trim. The quick-test
    # checkbox was supposed to mean "render a 12-second smoke test".
    # Before this fix, it only skipped pre-inpaint + occlusion-gate;
    # LatentSync still processed the FULL audio (thousands of frames).
    # Trim effective_audio to first 12 seconds so LatentSync only
    # renders that many frames worth of video.
    _check_cancel(cancel_event, log)
    QUICK_TEST_SECONDS = 12
    if job.quick_test:
        try:
            t_qt = time.perf_counter()
            qt_path = LATENTSYNC_SCRATCH / "quick_test_audio.wav"
            LATENTSYNC_SCRATCH.mkdir(parents=True, exist_ok=True)
            slice_audio_to_wav(job.effective_audio, 0.0,
                                float(QUICK_TEST_SECONDS),
                                qt_path)
            job.effective_audio = qt_path
            dt_qt = time.perf_counter() - t_qt
            log(f"[orchestrator] quick_test ON: trimmed driver audio "
                f"to first {QUICK_TEST_SECONDS}s ({dt_qt:.2f}s)")
            stage_times.append(("Stage 1.9 quick_test_trim", dt_qt))
        except Exception as exc:
            log(f"[orchestrator] quick_test trim FAILED ({exc}); "
                "LatentSync will process the full audio")

    _check_cancel(cancel_event, log)
    # Stage 2: lipsync render (LatentSync).
    t0 = time.perf_counter()
    if job.is_multi_clip:
        raw = _render_multi(job, log,
                            face_override=inpainted_face_paths)
    elif job.extend_single:
        raw = _render_single_extended(
            job, log, face_override=inpainted_face_paths)
    else:
        raw = _render_single(job, log,
                              face_override=inpainted_face_paths)
    dt = time.perf_counter() - t0
    log(f"[timing] Stage 2 latentsync_render took {dt:.2f}s")
    stage_times.append(("Stage 2 latentsync_render", dt))

    _check_cancel(cancel_event, log)
    # Stage 2.05 (OPTIONAL): SAM2 mask-out composite-back. Replaces the
    # masked region in the lipsync output with the ORIGINAL source
    # pixels via the SAM2 mask, hiding the seam with a Gaussian feather.
    # Runs only when Stage 1.76 succeeded.
    if maskout_artifacts is not None:
        log("[orchestrator] Stage 2.05: SAM2 mask-out composite-back")
        t0 = time.perf_counter()
        try:
            from core import maskout_pipeline as _mout
            # Local import: Stage 2.5 (below) also imports RECORDINGS_DIR
            # inside the function, which makes Python treat the name as
            # local across the entire function. Re-importing here keeps
            # both references happy.
            from .paths import RECORDINGS_DIR as _REC_DIR
            composited = _REC_DIR / (
                f"_maskout_composited_{int(time.time()*1000)}.mp4")
            _mout.composite_back(
                source_video=maskout_artifacts["original_source"],
                lipsync_video=Path(raw),
                masks_npy=maskout_artifacts["masks_npy"],
                out_video=composited,
                feather=maskout_artifacts["feather"],
                log=log,
            )
            raw = composited
            dt = time.perf_counter() - t0
            log(f"[timing] Stage 2.05 maskout_composite took "
                f"{dt:.2f}s -> {Path(raw).name}")
            stage_times.append(("Stage 2.05 maskout_composite", dt))
        except Exception as _mc_exc:
            log(f"[orchestrator] Stage 2.05 composite-back FAILED "
                f"({_mc_exc}); keeping raw lipsync output")

    _check_cancel(cancel_event, log)
    # Stage 2.5: re-mux the FINAL audio onto the lipsync video. There
    # are two triggers that make this necessary:
    #   (a) isolate_vocals=True: LatentSync drove on vocals-only, so
    #       the output mp4's audio is vocals-only and we need the full
    #       song (with instruments) put back.
    #   (b) maskout pipeline ran: Stage 2.05 composite_back writes via
    #       cv2.VideoWriter which is video-only and ALWAYS strips
    #       audio. If we skip remux here the final mp4 is silent,
    #       even if isolate_vocals was False.
    # Either trigger fires the remux as long as we have final_audio.
    _maskout_stripped_audio = maskout_artifacts is not None
    _isolate_swapped_track = (
        job.isolate_vocals and job.final_audio
        and Path(job.final_audio).resolve()
        != Path(job.effective_audio).resolve())
    if job.final_audio and (
            _isolate_swapped_track or _maskout_stripped_audio):
        if _maskout_stripped_audio and not _isolate_swapped_track:
            log("[orchestrator] re-muxing audio onto lipsync video "
                "(maskout composite stripped audio) ...")
        else:
            log("[orchestrator] re-muxing FULL audio (with instruments) "
                "back onto lipsync video ...")
        from .paths import RECORDINGS_DIR
        import time as _t
        remixed = RECORDINGS_DIR / f"_remuxed_{int(_t.time()*1000)}.mp4"
        try:
            t0 = time.perf_counter()
            replace_audio_track(Path(raw), Path(job.final_audio),
                                remixed)
            raw = remixed
            dt = time.perf_counter() - t0
            log(f"[orchestrator] remuxed -> {remixed.name}")
            log(f"[timing] Stage 2.5 audio_remux took {dt:.2f}s")
            stage_times.append(("Stage 2.5 audio_remux", dt))
        except Exception as exc:
            log(f"[orchestrator] WARN remux failed ({exc}); "
                "keeping vocals-only audio on output")

    # Stage 2.75 REMOVED: XSeg occluder gate is gone.

    _check_cancel(cancel_event, log)
    # Stage 3 (optional): GFPGAN
    if job.enhance_faces:
        log("[orchestrator] GFPGAN post-step starting...")
        t0 = time.perf_counter()
        raw = gfpgan_enhance(raw, log=log)
        dt = time.perf_counter() - t0
        log(f"[timing] Stage 3 gfpgan took {dt:.2f}s")
        stage_times.append(("Stage 3 gfpgan", dt))

    _check_cancel(cancel_event, log)
    # Stage 4 (optional): aspect-ratio reshape (crop/pad to target)
    if job.aspect.enabled:
        from .post_process import apply_aspect_ratio
        log("[orchestrator] aspect ratio post-step ...")
        t0 = time.perf_counter()
        raw = Path(apply_aspect_ratio(Path(raw), job.aspect, log=log))
        dt = time.perf_counter() - t0
        log(f"[timing] Stage 4 aspect_ratio took {dt:.2f}s")
        stage_times.append(("Stage 4 aspect_ratio", dt))

    _check_cancel(cancel_event, log)
    # Stage 5 (optional): watermark overlay (logo / handle)
    if job.watermark.enabled and job.watermark.image_path:
        from .post_process import apply_watermark
        log("[orchestrator] watermark post-step ...")
        t0 = time.perf_counter()
        raw = Path(apply_watermark(Path(raw), job.watermark, log=log))
        dt = time.perf_counter() - t0
        log(f"[timing] Stage 5 watermark took {dt:.2f}s")
        stage_times.append(("Stage 5 watermark", dt))

    total = time.perf_counter() - render_t0
    log("="*60)
    log(f"[timing] RENDER SUMMARY (total {total:.1f}s):")
    for name, dt in stage_times:
        pct = 100 * dt / max(total, 0.001)
        log(f"  {name:30s} {dt:7.2f}s  ({pct:5.1f}%)")
    log("="*60)

    return raw


# ---- branch: plain single clip ----
def _render_single(job: LipsyncJob,
                    log: Callable[[str], None],
                    face_override=None) -> Path:
    log("[orchestrator] mode = single clip")
    face = (face_override[0] if face_override
            else job.face_paths[0])
    if face_override:
        log(f"[orchestrator] using PRE-INPAINTED source for "
            f"LatentSync: {Path(face).name}")
    return latentsync.run(
        face_video=face,
        audio_driver=job.effective_audio,
        knobs=job.latentsync,
        log=log,
    )


# ---- branch: single clip stream_loop'd to audio length ----
def _render_single_extended(job: LipsyncJob,
                              log: Callable[[str], None],
                              face_override=None) -> Path:
    log("[orchestrator] mode = single clip, extend to audio length")
    face = (face_override[0] if face_override
            else job.face_paths[0])
    if face_override:
        log(f"[orchestrator] using PRE-INPAINTED source for "
            f"LatentSync: {Path(face).name}")
    a_dur = probe_duration_seconds(job.effective_audio)
    c_dur = probe_duration_seconds(face)

    if a_dur <= c_dur + 0.05:
        log(f"[orchestrator] audio ({a_dur:.1f}s) not longer than clip "
            f"({c_dur:.1f}s); skipping loop")
        return _render_single(job, log)

    EXTEND_SINGLE_WORK.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    looped = EXTEND_SINGLE_WORK / f"loop_{stamp}.mp4"
    loop_video_to_duration(face, a_dur, looped)
    log(f"[orchestrator] looped {c_dur:.1f}s -> {a_dur:.1f}s -> "
        f"{looped.name}")

    return latentsync.run(
        face_video=looped,
        audio_driver=job.effective_audio,
        knobs=job.latentsync,
        log=log,
    )


# ---- branch: multi-clip orchestration ----
def _render_multi(job: LipsyncJob,
                   log: Callable[[str], None],
                   face_override=None) -> Path:
    """N clips. Each gets its slice of audio. Last clip extends if
    audio over-hangs. Per-clip lipsync, then concat.

    face_override: list of pre-inpainted Path objects, one per clip.
    When provided, LatentSync receives the cleaned source per clip
    while job.face_paths (ORIGINAL) is still used by the per-clip
    occlusion gate to restore occluder pixels."""
    log(f"[orchestrator] mode = multi-clip ({len(job.face_paths)})")
    if face_override:
        log(f"[orchestrator] using {len(face_override)} PRE-INPAINTED "
            f"sources for LatentSync (originals used for occluder restore)")
    clip_durs = [probe_duration_seconds(p) for p in job.face_paths]
    audio_dur = probe_duration_seconds(job.effective_audio)
    log(f"[orchestrator] clip_durs={clip_durs}, audio={audio_dur:.1f}s")

    MULTICLIP_WORK.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    job_dir = MULTICLIP_WORK / f"job_{stamp}"
    job_dir.mkdir(parents=True, exist_ok=True)

    rendered: List[Path] = []
    audio_cursor = 0.0
    for i, (clip_path, clip_dur) in enumerate(zip(job.face_paths,
                                                    clip_durs)):
        is_last = (i == len(job.face_paths) - 1)
        remaining = max(0.0, audio_dur - audio_cursor)
        if remaining <= 0.05:
            log(f"[orchestrator] clip {i+1}: audio exhausted, skip")
            break

        # Last clip stream_loops if audio over-hangs
        if is_last and remaining > clip_dur + 0.05:
            target = remaining
            extended = job_dir / f"clip_{i:02d}_extended.mp4"
            loop_video_to_duration(clip_path, target, extended)
            log(f"[orchestrator] clip {i+1} LAST: looped to {target:.1f}s")
            face_in = extended
        else:
            target = min(clip_dur, remaining)
            face_in = clip_path

        # Audio slice
        slice_path = job_dir / f"clip_{i:02d}_audio.wav"
        slice_audio_to_wav(job.effective_audio, audio_cursor, target,
                            slice_path)
        log(f"[orchestrator] clip {i+1}/{len(job.face_paths)}: "
            f"face={Path(clip_path).name} dur={target:.1f}s "
            f"audio=[{audio_cursor:.1f}..{audio_cursor+target:.1f}]s")

        # If pre-inpainted clips are available, feed the CLEANED clip
        # to LatentSync. Otherwise feed the original.
        ls_input_face = (face_override[i] if face_override
                         and i < len(face_override) else face_in)
        if face_override:
            log(f"[orchestrator] clip {i+1}: LatentSync input = "
                f"PRE-INPAINTED {Path(ls_input_face).name}")
        out = latentsync.run(
            face_video=ls_input_face,
            audio_driver=slice_path,
            knobs=job.latentsync,
            log=log,
        )

        # Per-clip occlusion gate REMOVED.

        # Stage rendered clip in job_dir so concat list is stable
        staged = job_dir / f"clip_{i:02d}_rendered.mp4"
        if Path(out) != staged:
            import shutil
            shutil.copy2(out, staged)
        rendered.append(staged)
        audio_cursor += target

    if not rendered:
        raise RuntimeError("no clips were rendered")

    final = RECORDINGS_DIR / f"lipsync_multiclip_v2_{stamp}.mp4"
    concat_videos(rendered, final)
    log(f"[orchestrator] DONE: {final}")
    if final.stat().st_size < 1024:
        raise RuntimeError(
            f"multi-clip concat produced near-empty output: "
            f"{final} ({final.stat().st_size} bytes)")
    return final


