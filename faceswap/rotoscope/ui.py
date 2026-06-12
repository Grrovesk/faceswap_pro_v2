"""Rotoscoping tab UI -- deterministic single-object segmentation.

Behaviour (the only behaviour that matters):

    - You click on the frame.  The full current click list is sent to
      SAM2 via ``set_prompts``.  The daemon does a reset_state and one
      batched ``add_new_points_or_box`` with every point in one call.
      So the displayed mask is a PURE function of the click list --
      no stale prompts from earlier clicks accumulate; no first-click
      quirks; nothing.

    - A negative click within 25 px of an existing positive REPLACES
      that positive (and vice versa) BEFORE the submission.  This is
      what makes "click positive, then click negative on the same
      object" actually carve into the mask instead of doing nothing.

    - Clear clicks wipes the UI list AND tells the daemon to forget
      everything.

    - Speed is ~one SAM2 inference per click regardless of how many
      clicks you have, because every point fits in a single batched
      add_new_points_or_box call.  No N-clicks = N-round-trip blow up.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import cv2
import gradio as gr
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------
def _import_cache():
    from core import rotoscope_cache
    return rotoscope_cache


def _import_daemon():
    from core import sam2_daemon
    return sam2_daemon


# ---------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------
_GREEN_BGR = (0, 255, 0)
_RED_BGR = (0, 0, 255)
_TINT_ALPHA = 0.45
REPLACE_RADIUS = 25


def _read_frame_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    return img if img is not None else np.zeros((256, 256, 3),
                                                  dtype=np.uint8)


def _read_mask_gray(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def _union_disk_masks_for_frame(cache_info,
                                  frame_idx: int) -> Optional[np.ndarray]:
    """Read every \`masks/obj_*/frame_{idx:06d}.png\` file under the
    cache's masks_dir and union them into a single binary mask.  Used
    when there is no inline mask (e.g. after scrub) so the viewer
    reflects every tracked object, not just obj_1.
    """
    try:
        masks_root = Path(cache_info.masks_dir)
    except Exception:
        return None
    if not masks_root.is_dir():
        return None
    union = None
    pattern = f"frame_{int(frame_idx):06d}.png"
    for obj_dir in sorted(masks_root.glob("obj_*")):
        if not obj_dir.is_dir():
            continue
        mp = obj_dir / pattern
        if not mp.is_file():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if union is None:
            union = m.copy()
        else:
            if m.shape != union.shape:
                continue
            union = np.maximum(union, m)
    return union


def _wipe_frame_masks(cache_info, frame_idx: Optional[int] = None) -> None:
    """Delete \`masks/obj_*/frame_{idx:06d}.png\` for a specific frame,
    or every frame's masks if \`frame_idx\` is None.  Used by Clear so
    the disk no longer holds stale mask data after a reset.
    """
    try:
        masks_root = Path(cache_info.masks_dir)
    except Exception:
        return
    if not masks_root.is_dir():
        return
    if frame_idx is None:
        for obj_dir in masks_root.glob("obj_*"):
            if not obj_dir.is_dir():
                continue
            for mp in obj_dir.glob("frame_*.png"):
                try:
                    mp.unlink()
                except Exception:
                    pass
        return
    pattern = f"frame_{int(frame_idx):06d}.png"
    for obj_dir in masks_root.glob("obj_*"):
        if not obj_dir.is_dir():
            continue
        mp = obj_dir / pattern
        try:
            if mp.is_file():
                mp.unlink()
        except Exception:
            pass


def _decode_mask_b64(b64: str, h: int, w: int) -> Optional[np.ndarray]:
    if not b64:
        return None
    try:
        import base64
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        m = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        return m
    except Exception:
        return None


def _overlay_mask(frame_bgr: np.ndarray,
                    mask_gray):
    if mask_gray is None or mask_gray.max() == 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if mask_gray.shape != (h, w):
        mask_gray = cv2.resize(mask_gray, (w, h),
                                 interpolation=cv2.INTER_LINEAR)
    alpha = (mask_gray.astype(np.float32) / 255.0)[..., None] * _TINT_ALPHA
    colour = np.array(_GREEN_BGR, dtype=np.float32)
    out = (frame_bgr.astype(np.float32) * (1.0 - alpha)
            + colour[None, None] * alpha)
    return out.clip(0, 255).astype(np.uint8)


def _draw_clicks(frame_bgr: np.ndarray,
                   clicks: List[dict],
                   frame_idx: int) -> np.ndarray:
    out = frame_bgr.copy()
    for c in clicks:
        if int(c.get("frame", -1)) != int(frame_idx):
            continue
        # Brush strokes have no positive/negative -- the carve in the
        # mask itself is the visual; nothing to draw on top.
        if str(c.get("type", "click")) == "brush":
            continue
        x, y = int(c["x"]), int(c["y"])
        is_pos = int(c["label"]) == 1
        colour = _GREEN_BGR if is_pos else _RED_BGR
        # white halo so the dot is readable on any background
        cv2.circle(out, (x, y), 10, (255, 255, 255), 2,
                    lineType=cv2.LINE_AA)
        if is_pos:
            cv2.circle(out, (x, y), 8, colour, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 3, (255, 255, 255), -1,
                        lineType=cv2.LINE_AA)
        else:
            cv2.circle(out, (x, y), 8, colour, 3, lineType=cv2.LINE_AA)
            # diagonal slash so neg is unambiguous
            cv2.line(out, (x - 5, y - 5), (x + 5, y + 5),
                      colour, 2, lineType=cv2.LINE_AA)
    return out


def _apply_manual_erase(mask: Optional[np.ndarray],
                          erase: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Subtract the manual erase mask from the SAM2 mask.  Both are
    expected to be uint8 with 0/255.  Returns a uint8 mask of the
    same shape as ``mask``.  ``None``-safe.
    """
    if mask is None:
        return None
    if erase is None:
        return mask
    if erase.shape != mask.shape:
        try:
            erase = cv2.resize(erase, (mask.shape[1], mask.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        except Exception:
            return mask
    out = mask.copy()
    out[erase > 127] = 0
    return out


def _rebuild_erase_for_frame(clicks: List[dict],
                                frame_idx: int,
                                h: int, w: int) -> Optional[np.ndarray]:
    """Recompute the manual erase mask for one frame from every
    brush stroke in the click list that targets this frame.  Each
    brush stroke paints a filled disc of its own radius.  Returns
    ``None`` if no brush strokes target this frame.
    """
    have = False
    mask = np.zeros((int(h), int(w)), dtype=np.uint8)
    for c in (clicks or []):
        if str(c.get("type", "click")) != "brush":
            continue
        if int(c.get("frame", -1)) != int(frame_idx):
            continue
        try:
            x = int(c["x"]); y = int(c["y"])
            r = int(c.get("radius", 40))
        except Exception:
            continue
        if r <= 0:
            continue
        cv2.circle(mask, (x, y), r, 255, -1, lineType=cv2.LINE_AA)
        have = True
    return mask if have else None


def _render(cache_info,
              frame_idx: int,
              clicks: List[dict],
              inline_mask: Optional[np.ndarray] = None,
              manual_erase: Optional[dict] = None) -> np.ndarray:
    cache = _import_cache()
    bgr = _read_frame_bgr(cache.frame_path(cache_info, frame_idx))
    m = inline_mask
    if m is None:
        m = _union_disk_masks_for_frame(cache_info, frame_idx)
    # Apply per-frame manual erase (from brush strokes) on top of SAM2.
    erase = None
    if isinstance(manual_erase, dict):
        erase = manual_erase.get(int(frame_idx))
    m = _apply_manual_erase(m, erase)
    bgr = _overlay_mask(bgr, m)
    bgr = _draw_clicks(bgr, clicks, frame_idx)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _format_clicks_md(clicks: List[dict]) -> str:
    if not clicks:
        return "**Clicks:** none yet — click on the frame above"
    lines = ["**Clicks so far:**"]
    for i, c in enumerate(clicks, start=1):
        if str(c.get("type", "click")) == "brush":
            r = int(c.get("radius", 40))
            lines.append(
                f"  {i}. 🧽 erase r={r}px ({c['x']}, {c['y']})  @ frame {c['frame']}")
            continue
        sign = "+" if int(c["label"]) == 1 else "−"
        oid = int(c.get("obj_id", 1))
        lines.append(
            f"  {i}. obj{oid} {sign} ({c['x']}, {c['y']})  @ frame {c['frame']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Tab
# ---------------------------------------------------------------------
def build_tab(lipsync_npy_target=None, faceswap_npy_target=None) -> None:
    gr.Markdown(
        "### 🪒 Rotoscoping\n"
        "Click on the object to seed SAM2.  Positive includes, negative "
        "excludes.  Clicking the opposite label within ~25 px of an "
        "existing click REPLACES it, so the mask actually changes when "
        "you correct a spot.  Propagate across every frame when ready."
    )

    # session state
    roto_video_path = gr.State("")
    roto_cache_info = gr.State(None)
    roto_clicks: gr.State = gr.State([])           # list[dict]
    roto_inline_mask: gr.State = gr.State(None)    # latest mask numpy
    # Per-frame manual erase mask painted by the brush tool.  Dict of
    # ``frame_idx -> np.ndarray (uint8 0/255)``.  Applied as a
    # boolean subtract on top of every SAM2 mask before display and
    # before lipsync handoff.  Persisted across scrubs.
    roto_manual_erase: gr.State = gr.State({})
    roto_daemon_loaded = gr.State("")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### 1. Source video")
            roto_video = gr.Video(label="Target video", height=240)
            roto_status = gr.Markdown("_Upload a video to begin._")

            gr.Markdown("#### 2. Click mode")
            roto_click_mode = gr.Radio(
                choices=["+ positive (add to mask)",
                         "- negative (remove from mask)",
                         "🧽 erase brush (paint pixels OUT of mask)"],
                value="+ positive (add to mask)",
                label="Next click is",
            )
            roto_obj_picker = gr.Radio(
                choices=[1, 2, 3],
                value=1,
                label="Object (clicks attach to this object id)",
            )
            roto_brush_size = gr.Slider(
                minimum=5, maximum=200, value=40, step=1,
                label="Brush radius (px, used in erase brush mode)",
            )
            with gr.Row():
                roto_undo_btn = gr.Button("Undo last click", size="sm")
                roto_clear_btn = gr.Button("Clear clicks", size="sm")
                roto_unload_btn = gr.Button("Unload SAM2 (free GPU)",
                                              size="sm")
            roto_clicks_md = gr.Markdown(
                "**Clicks:** none yet — click on the frame above")

        with gr.Column(scale=2):
            gr.Markdown("#### Frame viewer")
            roto_canvas = gr.Image(
                label="Click anywhere on the object to segment it.",
                interactive=True,
                height=520,
                show_label=True,
                show_download_button=False,
                show_share_button=False,
            )
            roto_scrubber = gr.Slider(0, 0, value=0, step=1,
                                        label="Frame", interactive=False)
            with gr.Row():
                roto_prev10 = gr.Button("◀◀ -10")
                roto_prev1 = gr.Button("◀ -1")
                roto_next1 = gr.Button("+1 ▶")
                roto_next10 = gr.Button("+10 ▶▶")
                roto_frame_label = gr.Markdown("Frame 0 / 0")

            gr.Markdown("#### 3. Propagate + handoff")
            with gr.Row():
                roto_propagate_btn = gr.Button(
                    "▶ Propagate across all frames",
                    variant="primary", size="lg")
                roto_send_btn = gr.Button(
                    "🚀 Send masks to Lip-Sync tab",
                    variant="secondary", size="lg")
            roto_progress_md = gr.Markdown(
                "_Status / progress will appear here._")

    # =====================================================
    # Handlers
    # =====================================================
    def _on_video_upload(video_path, cur_cache_info=None,
                          cur_video_path=""):
        if not video_path:
            return ("", None, [], None,
                    gr.update(maximum=0, value=0, interactive=False),
                    "_Upload a video to begin._",
                    "**Clicks:** none yet — click on the frame above",
                    "Frame 0 / 0", None, "")
        # Browser-safe preview (T0-2 / #129).  If the user uploaded
        # H.265 / ProRes / weird MOV, Gradio's <video> tag overlays
        # "Error".  Transcode to H.264/AAC/faststart in a content-
        # hashed cache; re-uploads of the same bytes are O(1).
        try:
            from core.browser_safe_preview import ensure_browser_safe
            video_path = ensure_browser_safe(video_path, log=logger.info)
        except Exception as exc:
            logger.warning("[rotoscope] browser_safe_preview failed: %s "
                            "-- using original; widget may show Error "
                            "overlay", exc)
        # Gradio fires `.change` more than once per upload (preview
        # render, websocket re-fetch, etc.).  If the incoming path
        # matches what we already cached, skip the full extract path
        # entirely -- avoids re-running ffmpeg on every spurious
        # change event.
        try:
            if (cur_cache_info is not None
                    and str(cur_video_path or "")
                    and str(video_path) == str(cur_video_path)):
                rgb = _render(cur_cache_info, 0, [])
                status = (
                    f"**{Path(cur_cache_info.video_path).name}** -- "
                    f"{cur_cache_info.frame_count} frames @ "
                    f"{cur_cache_info.width}x{cur_cache_info.height} "
                    f"@ {cur_cache_info.fps:.2f} fps\n\n"
                    f"_Cache:_ `{cur_cache_info.cache_dir}` (cached)")
                return (str(cur_video_path), cur_cache_info, [], None,
                        gr.update(
                            maximum=max(0, cur_cache_info.frame_count - 1),
                            value=0, interactive=True),
                        status,
                        "**Clicks:** none yet -- click on the frame above",
                        f"Frame 0 / {cur_cache_info.frame_count - 1}",
                        rgb, "")
        except Exception as _dx:
            logger.debug("upload dedup check failed: %s", _dx)
        cache = _import_cache()
        try:
            info = cache.extract_and_cache(video_path, log=logger.info)
        except Exception as exc:
            return ("", None, [], None,
                    gr.update(maximum=0, value=0, interactive=False),
                    f"**Extract failed:** {exc}",
                    "**Clicks:** none yet — click on the frame above",
                    "Frame 0 / 0", None, "")
        rgb = _render(info, 0, [])
        status = (
            f"**{Path(info.video_path).name}** — {info.frame_count} "
            f"frames @ {info.width}×{info.height} @ "
            f"{info.fps:.2f} fps\n\n"
            f"_Cache:_ `{info.cache_dir}`")
        return (info.video_path, info, [], None,
                gr.update(maximum=max(0, info.frame_count - 1), value=0,
                            interactive=True),
                status,
                "**Clicks:** none yet — click on the frame above",
                f"Frame 0 / {info.frame_count - 1}",
                rgb, "")

    roto_video.change(
        fn=_on_video_upload,
        inputs=[roto_video, roto_cache_info, roto_video_path],
        outputs=[roto_video_path, roto_cache_info, roto_clicks,
                  roto_inline_mask, roto_scrubber, roto_status,
                  roto_clicks_md, roto_frame_label, roto_canvas,
                  roto_daemon_loaded],
    )

    def _on_scrub(frame_idx, cache_info, clicks, manual_erase):
        if cache_info is None:
            return None, None, "Frame 0 / 0"
        f = int(frame_idx or 0)
        rgb = _render(cache_info, f, clicks or [], inline_mask=None,
                       manual_erase=manual_erase)
        return (rgb, None,
                f"Frame {f} / {int(cache_info.frame_count) - 1}")

    roto_scrubber.change(
        fn=_on_scrub,
        inputs=[roto_scrubber, roto_cache_info, roto_clicks,
                roto_manual_erase],
        outputs=[roto_canvas, roto_inline_mask, roto_frame_label],
    )

    def _jump(delta, scrub, cache_info):
        if cache_info is None:
            return 0
        return max(0, min(int(cache_info.frame_count) - 1,
                            int(scrub or 0) + int(delta)))

    for btn, d in ((roto_prev10, -10), (roto_prev1, -1),
                    (roto_next1, +1), (roto_next10, +10)):
        btn.click(fn=_jump,
                   inputs=[gr.State(d), roto_scrubber, roto_cache_info],
                   outputs=[roto_scrubber])

    def _union_masks_from_b64(obj_masks_b64, h, w):
        """Decode the daemon's per-obj b64 dict into a single union
        mask (uint8 255 where any obj is set).  Used as the inline
        mask for ``_render`` so green tint covers every tracked obj.
        """
        union = None
        for _oid, b64 in (obj_masks_b64 or {}).items():
            m = _decode_mask_b64(b64, h, w)
            if m is None:
                continue
            if union is None:
                union = m.copy()
            else:
                if m.shape != union.shape:
                    continue
                union = np.maximum(union, m)
        return union

    def _on_canvas_click(cache_info, clicks, click_mode, obj_pick,
                          brush_size, manual_erase,
                          scrub, daemon_loaded, evt: gr.SelectData):
        if cache_info is None:
            return (clicks or [], None,
                    "**Clicks:** load a video first",
                    None, daemon_loaded, "_no video loaded_",
                    manual_erase)
        try:
            x = int(evt.index[0]); y = int(evt.index[1])
        except Exception:
            return (clicks or [], None,
                    _format_clicks_md(clicks or []),
                    None, daemon_loaded, "_bad click coords_",
                    manual_erase)

        frame_idx = int(scrub or 0)
        # Detect brush mode FIRST -- distinct UTF-8 codepoint, no overlap
        # with "+ positive" or "- negative".
        is_brush_mode = "erase brush" in str(click_mode or "")
        label = 1 if str(click_mode or "").startswith("+") else 0
        try:
            obj_id = int(obj_pick) if obj_pick is not None else 1
        except Exception:
            obj_id = 1
        if obj_id < 1:
            obj_id = 1

        # ----- BRUSH ERASE BRANCH -----
        # Paint a disc into the per-frame manual_erase mask, record the
        # brush stroke in the click list (so Undo can roll it back),
        # and re-render WITHOUT touching SAM2.
        if is_brush_mode:
            try:
                radius = int(brush_size) if brush_size is not None else 40
            except Exception:
                radius = 40
            if radius <= 0:
                radius = 1
            # mutate the manual_erase dict (Gradio State stores by ref).
            erase_dict = dict(manual_erase or {})
            cur = erase_dict.get(frame_idx)
            if cur is None or cur.shape != (cache_info.height,
                                              cache_info.width):
                cur = np.zeros((cache_info.height, cache_info.width),
                                dtype=np.uint8)
            cv2.circle(cur, (x, y), radius, 255, -1,
                        lineType=cv2.LINE_AA)
            erase_dict[frame_idx] = cur
            stroke = {"type": "brush", "x": x, "y": y,
                       "frame": frame_idx, "radius": radius}
            new_clicks = list(clicks or []) + [stroke]
            # Pull the existing SAM2 union from disk so we can render
            # with the brush carve already applied.
            sam_union = _union_disk_masks_for_frame(cache_info, frame_idx)
            rgb = _render(cache_info, frame_idx, new_clicks,
                           inline_mask=sam_union,
                           manual_erase=erase_dict)
            progress = (f"_brush erase ({x}, {y}) r={radius}px  "
                        f"frame {frame_idx}_")
            return (new_clicks, sam_union,
                    _format_clicks_md(new_clicks),
                    rgb, daemon_loaded, progress, erase_dict)

        # ----- proximity replace WITHIN this obj_id -----
        # SAM2 demo style: each obj_id tracks its own set of prompts.
        # A negative on top of an existing positive of the SAME obj_id
        # replaces it (or vice versa).  Other obj_ids are untouched.
        new_clicks = []
        for c in (clicks or []):
            if (int(c.get("frame", -1)) == frame_idx
                    and int(c.get("obj_id", 1)) == obj_id
                    and int(c.get("label", 1)) != label):
                dx = int(c["x"]) - x
                dy = int(c["y"]) - y
                if dx * dx + dy * dy <= REPLACE_RADIUS * REPLACE_RADIUS:
                    continue
            new_clicks.append(c)
        new_clicks.append({"x": x, "y": y, "frame": frame_idx,
                            "label": label, "obj_id": obj_id})

        # ----- daemon -----
        try:
            daemon = _import_daemon().get_or_start_daemon()
        except Exception as exc:
            return (new_clicks, None,
                    _format_clicks_md(new_clicks),
                    None, daemon_loaded, f"_daemon error: {exc}_",
                    manual_erase)
        if daemon_loaded != cache_info.video_path:
            try:
                daemon.load_video(cache_info.video_path)
                daemon_loaded = cache_info.video_path
            except Exception as exc:
                return (new_clicks, None,
                        _format_clicks_md(new_clicks),
                        None, daemon_loaded,
                        f"_load_video error: {exc}_",
                        manual_erase)

        # ----- send THIS obj_id's point set for this frame -----
        # Single add_new_points_or_box call on the daemon side; other
        # obj_ids' state in inference_state is preserved.
        pts_this_obj = [
            (int(c["x"]), int(c["y"]))
            for c in new_clicks
            if int(c["frame"]) == frame_idx
            and int(c.get("obj_id", 1)) == obj_id
        ]
        labs_this_obj = [
            int(c["label"])
            for c in new_clicks
            if int(c["frame"]) == frame_idx
            and int(c.get("obj_id", 1)) == obj_id
        ]
        try:
            masks_root = Path(cache_info.masks_dir)
            masks_root.mkdir(parents=True, exist_ok=True)
            res = daemon.apply_click(
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=pts_this_obj,
                labels=labs_this_obj,
                masks_out_root=str(masks_root),
                return_b64=True,
            )
        except Exception as exc:
            return (new_clicks, None,
                    _format_clicks_md(new_clicks),
                    None, daemon_loaded,
                    f"_apply_click error: {exc}_",
                    manual_erase)

        union = _union_masks_from_b64(res.get("obj_masks", {}),
                                        cache_info.height,
                                        cache_info.width)
        rgb = _render(cache_info, frame_idx, new_clicks,
                       inline_mask=union,
                       manual_erase=manual_erase)
        obj_ids_seen = res.get("obj_ids", []) or []
        progress = (f"_click ({x}, {y}) frame {frame_idx} "
                    f"obj={obj_id} label={'+' if label else '-'}  "
                    f"tracked objs={obj_ids_seen}_")
        return (new_clicks, union,
                _format_clicks_md(new_clicks),
                rgb, daemon_loaded, progress, manual_erase)

    roto_canvas.select(
        fn=_on_canvas_click,
        inputs=[roto_cache_info, roto_clicks, roto_click_mode,
                roto_obj_picker, roto_brush_size, roto_manual_erase,
                roto_scrubber, roto_daemon_loaded],
        outputs=[roto_clicks, roto_inline_mask, roto_clicks_md,
                  roto_canvas, roto_daemon_loaded, roto_progress_md,
                  roto_manual_erase],
    )

    def _on_undo(cache_info, clicks, manual_erase, scrub, daemon_loaded):
        """Pop the last entry off the click list.  If it's a brush
        stroke, recompute the manual erase mask for that frame from
        the remaining brush strokes -- no SAM2 call.  If it's a SAM2
        click, re-send the affected obj_id's remaining points via
        apply_click (or remove the obj if empty).  If the whole list
        becomes empty, clear the daemon + disk + manual erase.
        """
        if cache_info is None:
            return ([], None,
                    "**Clicks:** none yet -- click on the frame above",
                    None, daemon_loaded, "_nothing to undo_",
                    manual_erase)
        if not clicks:
            rgb = _render(cache_info, int(scrub or 0), [],
                           manual_erase=manual_erase)
            return ([], None,
                    _format_clicks_md([]),
                    rgb, daemon_loaded, "_nothing to undo_",
                    manual_erase)
        popped = clicks[-1]
        new_clicks = list(clicks)[:-1]
        frame_idx = int(scrub or 0)

        # ----- BRUSH UNDO BRANCH -----
        if str(popped.get("type", "click")) == "brush":
            popped_frame = int(popped.get("frame", frame_idx))
            erase_dict = dict(manual_erase or {})
            new_erase = _rebuild_erase_for_frame(
                new_clicks, popped_frame,
                cache_info.height, cache_info.width)
            if new_erase is None:
                erase_dict.pop(popped_frame, None)
            else:
                erase_dict[popped_frame] = new_erase
            sam_union = _union_disk_masks_for_frame(cache_info, frame_idx)
            rgb = _render(cache_info, frame_idx, new_clicks,
                           inline_mask=sam_union,
                           manual_erase=erase_dict)
            return (new_clicks, sam_union,
                    _format_clicks_md(new_clicks),
                    rgb, daemon_loaded,
                    f"_undid brush stroke ({len(new_clicks)} remaining)_",
                    erase_dict)

        popped_obj = int(popped.get("obj_id", 1))

        # whole list empty -> full daemon reset + wipe manual erase
        if not new_clicks:
            try:
                d = _import_daemon().SAM2Daemon.singleton()
                if d.is_running():
                    d.clear()
            except Exception as exc:
                logger.warning("daemon clear failed during undo: %s", exc)
            _wipe_frame_masks(cache_info, frame_idx=None)
            rgb = _render(cache_info, frame_idx, [], manual_erase={})
            return ([], None,
                    _format_clicks_md([]),
                    rgb, daemon_loaded,
                    "_undid last click -- no clicks remaining_",
                    {})

        # re-send the popped obj_id's remaining points
        try:
            daemon = _import_daemon().get_or_start_daemon()
        except Exception as exc:
            return (new_clicks, None,
                    _format_clicks_md(new_clicks),
                    None, daemon_loaded,
                    f"_daemon error: {exc}_",
                    manual_erase)
        if daemon_loaded != cache_info.video_path:
            try:
                daemon.load_video(cache_info.video_path)
                daemon_loaded = cache_info.video_path
            except Exception as exc:
                return (new_clicks, None,
                        _format_clicks_md(new_clicks),
                        None, daemon_loaded,
                        f"_load_video error: {exc}_",
                        manual_erase)

        pts_this = [(int(c["x"]), int(c["y"]))
                     for c in new_clicks
                     if str(c.get("type", "click")) != "brush"
                     and int(c["frame"]) == frame_idx
                     and int(c.get("obj_id", 1)) == popped_obj]
        labs_this = [int(c["label"])
                      for c in new_clicks
                      if str(c.get("type", "click")) != "brush"
                      and int(c["frame"]) == frame_idx
                      and int(c.get("obj_id", 1)) == popped_obj]
        try:
            masks_root = Path(cache_info.masks_dir)
            masks_root.mkdir(parents=True, exist_ok=True)
            res = daemon.apply_click(
                frame_idx=frame_idx,
                obj_id=popped_obj,
                points=pts_this,
                labels=labs_this,
                masks_out_root=str(masks_root),
                return_b64=True,
            )
        except Exception as exc:
            return (new_clicks, None,
                    _format_clicks_md(new_clicks),
                    None, daemon_loaded,
                    f"_undo apply_click error: {exc}_",
                    manual_erase)

        union = _union_masks_from_b64(res.get("obj_masks", {}),
                                        cache_info.height,
                                        cache_info.width)
        rgb = _render(cache_info, frame_idx, new_clicks,
                       inline_mask=union,
                       manual_erase=manual_erase)
        return (new_clicks, union,
                _format_clicks_md(new_clicks),
                rgb, daemon_loaded,
                f"_undid last click ({len(new_clicks)} remaining)_",
                manual_erase)

    roto_undo_btn.click(
        fn=_on_undo,
        inputs=[roto_cache_info, roto_clicks, roto_manual_erase,
                roto_scrubber, roto_daemon_loaded],
        outputs=[roto_clicks, roto_inline_mask, roto_clicks_md,
                  roto_canvas, roto_daemon_loaded, roto_progress_md,
                  roto_manual_erase],
    )

    def _on_clear(cache_info, daemon_loaded, scrub):
        if daemon_loaded:
            try:
                d = _import_daemon().SAM2Daemon.singleton()
                if d.is_running():
                    d.clear()
            except Exception as exc:
                logger.warning("daemon clear failed: %s", exc)
        frame_idx = int(scrub or 0)
        if cache_info is not None:
            # Wipe ALL per-obj mask files for this video so _render's
            # disk fallback can't resurrect stale segmentations.
            _wipe_frame_masks(cache_info, frame_idx=None)
            rgb = _render(cache_info, frame_idx, [], manual_erase={})
        else:
            rgb = None
        return ([], None,
                "**Clicks:** none yet — click on the frame above",
                rgb, "_cleared every click; SAM2 state reset_", {})

    roto_clear_btn.click(
        fn=_on_clear,
        inputs=[roto_cache_info, roto_daemon_loaded, roto_scrubber],
        outputs=[roto_clicks, roto_inline_mask, roto_clicks_md,
                  roto_canvas, roto_progress_md, roto_manual_erase],
    )

    def _on_unload():
        try:
            d = _import_daemon().SAM2Daemon.singleton()
            d.shutdown()
        except Exception as exc:
            return f"_unload error: {exc}_"
        return "_SAM2 daemon shut down — GPU memory freed._"

    roto_unload_btn.click(fn=_on_unload, inputs=None,
                            outputs=[roto_progress_md])

    def _on_propagate(cache_info, clicks, scrub, daemon_loaded):
        if cache_info is None:
            yield "_propagate: no video loaded._", gr.update()
            return
        if not clicks:
            yield ("_propagate: add at least one click first._",
                   gr.update())
            return
        try:
            daemon = _import_daemon().get_or_start_daemon()
            if daemon_loaded != cache_info.video_path:
                daemon.load_video(cache_info.video_path)
                daemon_loaded = cache_info.video_path
            # Resync SAM2's state with the UI's click list right before
            # propagation so the propagated masks match what the user
            # sees on the seed frame.
            prompts = [
                {"x": int(c["x"]), "y": int(c["y"]),
                  "label": int(c["label"]),
                  "frame_idx": int(c["frame"])}
                for c in clicks
            ]
            daemon.set_prompts(
                frame_idx=int(scrub or 0),
                prompts=prompts,
                obj_id=1,
                return_b64=False,
            )
        except Exception as exc:
            yield f"_propagate prep failed: {exc}_", gr.update()
            return

        cache_mod = _import_cache()
        masks_dir = cache_mod.mask_dir(cache_info, obj_id=1)
        masks_dir.mkdir(parents=True, exist_ok=True)

        progress = {"frame_idx": 0, "total": int(cache_info.frame_count)}
        def _cb(d):
            try:
                if d.get("status") == "progress":
                    progress["frame_idx"] = int(d.get("frame_idx", 0))
                    progress["total"] = int(d.get("total",
                                                    progress["total"]))
            except Exception:
                pass

        import threading
        done = threading.Event()
        result = {"err": None}
        def _runner():
            try:
                daemon.propagate(str(masks_dir), on_progress=_cb)
            except Exception as exc:
                result["err"] = exc
            finally:
                done.set()
        t = threading.Thread(target=_runner, daemon=True)
        t0 = time.perf_counter()
        t.start()
        while not done.is_set():
            done.wait(timeout=0.8)
            f, tot = progress["frame_idx"], max(progress["total"], 1)
            pct = int(round(100.0 * f / tot))
            yield (f"_propagating … frame {f} / {tot} ({pct} %) — "
                    f"{time.perf_counter() - t0:.0f} s_",
                    gr.update())
        if result["err"]:
            yield f"_propagate FAILED: {result['err']}_", gr.update()
            return
        rgb = _render(cache_info, int(scrub or 0), clicks)
        yield (f"_propagate done in {time.perf_counter() - t0:.1f} s — "
                f"masks in `{masks_dir}`.  Click **Send masks to "
                f"Lip-Sync tab** to finish._",
                rgb)

    roto_propagate_btn.click(
        fn=_on_propagate,
        inputs=[roto_cache_info, roto_clicks, roto_scrubber,
                roto_daemon_loaded],
        outputs=[roto_progress_md, roto_canvas],
    )

    def _on_send(cache_info):
        """Returns (npy_path_str_or_empty, status_md).

        When ``lipsync_npy_target`` is wired, the click handler routes
        ``npy_path_str`` straight into the Lip-Sync NPY textbox so the
        user doesn't have to copy-paste.  Without it (old behavior),
        only ``status_md`` is shown in the progress markdown.
        """
        if cache_info is None:
            return ("", "_no video loaded._")
        masks_dir = Path(cache_info.masks_dir) / "obj_1"
        if not masks_dir.is_dir():
            return ("", "_no masks yet — run **Propagate across all "
                         "frames** first._")
        png_paths = sorted(masks_dir.glob("frame_*.png"))
        if not png_paths:
            return ("", "_no masks yet — run **Propagate across all "
                         "frames** first._")
        total = int(cache_info.frame_count)
        h, w = int(cache_info.height), int(cache_info.width)
        stack = np.zeros((total, h, w), dtype=np.uint8)
        for p in png_paths:
            try:
                fi = int(p.stem.split("_")[-1])
            except Exception:
                continue
            if 0 <= fi < total:
                m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h),
                                         interpolation=cv2.INTER_LINEAR)
                    stack[fi] = m
        npy_path = Path(cache_info.masks_dir) / "obj_1_combined.npy"
        try:
            np.save(str(npy_path), stack)
        except Exception as exc:
            return ("", f"_could not write NPY: {exc}_")
        n_nonempty = int((stack.max(axis=(1, 2)) > 0).sum())
        if lipsync_npy_target is not None:
            status_md = (
                f"_**Masks sent to Lip-Sync.** "
                f"({n_nonempty} / {total} non-empty frames at "
                f"{w}x{h}.)  Open the **Mask out non-face regions "
                f"(SAM2)** accordion on the Lip-Sync tab and tick "
                f"**Enable mask-out** -- the NPY path is already "
                f"filled in._\n\n`{npy_path}`"
            )
        else:
            status_md = (
                f"_**Masks ready.** {len(png_paths)} PNGs combined into "
                f"a ({total}, {h}, {w}) NPY  "
                f"({n_nonempty} non-empty frames)._\n\n"
                f"**NPY path:**\n\n`{npy_path}`\n\n"
                f"Switch to the **Lip-Sync** tab, open the "
                f"**Mask out non-face regions (SAM2)** accordion, tick "
                f"**Enable mask-out**, and paste the path above into "
                f"**Pre-computed mask NPY (optional)**.  Stage 1.76 will "
                f"skip SAM2 entirely and use these masks directly."
            )
        return (str(npy_path), status_md)

    # Cross-tab handoff (T1-1 + T2-NEW, 2026-06-11).
    # _on_send returns (npy_path, status_md).  We wire the click's
    # outputs to whichever targets the caller provided so the NPY
    # path lands in the right widgets in one click:
    #   - lipsync_npy_target  -> Lip-Sync mask-out NPY textbox
    #   - faceswap_npy_target -> Face Swap region-restrict NPY textbox
    targets = [t for t in (lipsync_npy_target, faceswap_npy_target)
               if t is not None]
    if not targets:
        roto_send_btn.click(
            fn=lambda ci: _on_send(ci)[1],
            inputs=[roto_cache_info],
            outputs=[roto_progress_md],
        )
    elif len(targets) == 1:
        roto_send_btn.click(
            fn=_on_send,
            inputs=[roto_cache_info],
            outputs=[targets[0], roto_progress_md],
        )
    else:
        # Two targets: write the same NPY path to both, plus status.
        def _on_send_dual(ci):
            npy, md = _on_send(ci)
            return (npy, npy, md)
        roto_send_btn.click(
            fn=_on_send_dual,
            inputs=[roto_cache_info],
            outputs=[targets[0], targets[1], roto_progress_md],
        )
