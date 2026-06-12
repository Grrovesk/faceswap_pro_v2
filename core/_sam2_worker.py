"""_sam2_worker.py -- runs SAM2 video segmentation in the KeySync venv.

Tracks ONE OR MORE objects from a single frame's clicks, propagates each
across the video, and outputs the UNION mask as a (T, H, W) uint8 .npy.

This file is invoked as a SUBPROCESS by sam2_occlusion_composite.py.

CLI:
    python _sam2_worker.py --video <mp4> \
        --click x1 y1 frame1 --click x2 y2 frame2 ... \
        --out <npy>

Each --click takes THREE ints: x, y, frame_idx. You can pass --click
multiple times to track multiple objects (e.g. lizard + left hand +
right hand). All masks are union-combined into a single (T, H, W)
output where 255 = any tracked object, 0 = elsewhere.

Output .npy: uint8 (T, H, W). 255 = preserve-from-source. 0 = take-from-lipsync.
"""
import argparse
import os
import shutil
import sys

import numpy as np


# ============================================================
# Daemon mode (Phase 1.1 of Rotoscoping tab) -- long-running process.
# Reads JSON-lines requests from stdin, writes JSON-lines responses to
# stdout.  Loads SAM2 weights once at startup.  Sub-second clicks
# after warmup.  Used by core/sam2_daemon.py.
# ============================================================
def daemon_main() -> None:
    """Run as a long-running daemon: load model once, then accept
    click / propagate / clear requests over stdin until shutdown.
    """
    import json
    import base64
    import traceback

    def emit(payload: dict) -> None:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()

    # Parse the daemon-mode args (just need ckpt path).
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--sam2_ckpt", type=str, default=None)
    ap.add_argument("--sam2_config", type=str, default=None)
    args, _unused = ap.parse_known_args()

    emit({"event": "starting"})

    # Resolve ckpt + cfg (same logic as CLI main).
    _DEFAULT_CFG_BY_NAME = {
        "sam2.1_hiera_tiny.pt":       "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_small.pt":      "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_base_plus.pt":  "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_large.pt":      "configs/sam2.1/sam2.1_hiera_l.yaml",
    }
    if not args.sam2_ckpt:
        emit({"event": "error",
              "message": "daemon mode requires --sam2_ckpt"})
        return
    sam2_ckpt = os.path.abspath(args.sam2_ckpt)
    if not os.path.isfile(sam2_ckpt):
        emit({"event": "error",
              "message": f"weights not found: {sam2_ckpt}"})
        return
    model_cfg = args.sam2_config or _DEFAULT_CFG_BY_NAME.get(
        os.path.basename(sam2_ckpt),
        "configs/sam2.1/sam2.1_hiera_b+.yaml")

    try:
        from hydra.core.global_hydra import GlobalHydra
        from hydra import initialize_config_module
        if not GlobalHydra.instance().is_initialized():
            initialize_config_module(
                config_module="sam2", version_base=None)
    except Exception as exc:
        emit({"event": "warn",
              "message": f"hydra init: {exc}"})

    try:
        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor(model_cfg, sam2_ckpt,
                                                 device="cuda")
        emit({"event": "model_loaded",
              "ckpt": sam2_ckpt, "config": model_cfg})
    except Exception as exc:
        emit({"event": "error",
              "message": f"model load failed: {exc}",
              "traceback": traceback.format_exc()})
        return

    # Per-video state (rebuilt on each load_video request).
    import cv2  # noqa: E402
    state = None
    video_path = None
    video_dims = None  # (frame_count, height, width)
    # Flag: has this state ever served a real user click successfully?
    # On the FIRST real click after load_video, SAM2 returns a wonky
    # or empty mask.  The user's known-working workaround is to clear
    # and click the same coordinate again.  We do that internally on
    # the first click so the user never sees the wonky result.
    first_click_done = False
    # Per-obj warmup tracking.  add_new_points_or_box returns a wonky
    # mask on the FIRST call for any new obj_id within an
    # inference_state, even after the global first-call warmup has
    # fired.  We pump each new obj_id once internally before its real
    # submit so the user does not have to re-click.
    seen_obj_ids = set()

    emit({"event": "ready"})

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            emit({"event": "error",
                  "message": f"bad json: {exc}"})
            continue

        op = req.get("op", "")
        req_id = req.get("request_id", "")

        try:
            if op == "load_video":
                vp = req["video_path"]
                if not os.path.isfile(vp):
                    raise FileNotFoundError(vp)
                state = predictor.init_state(
                    video_path=vp,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                )
                cap = cv2.VideoCapture(vp)
                fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                w_ = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h_ = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                cap.release()
                video_path = vp
                video_dims = (fc, h_, w_)

                # No load-time warmup.  The click handler does an
                # internal "click + reset + click again" cycle on the
                # FIRST real click of a new video, which directly
                # mimics the user's known-working manual workflow
                # (click -> clear -> click).  See first_click_done
                # logic in the click handler below.
                first_click_done = False
                seen_obj_ids = set()

                emit({"request_id": req_id, "op": "load_video",
                      "status": "ok", "frame_count": fc,
                      "width": w_, "height": h_, "fps": fps})

            elif op == "click":
                if state is None:
                    raise RuntimeError("load_video must be called first")
                x = int(req["x"])
                y = int(req["y"])
                frame_idx = int(req["frame_idx"])
                label = int(req.get("label", 1))
                obj_id = int(req.get("obj_id", 1))
                pts = np.array([[x, y]], dtype=np.float32)
                labels = np.array([label], dtype=np.int32)
                # First-click retry: SAM2's first add_new_points_or_box
                # call on a freshly init_state'd state returns wonky or
                # empty mask logits.  The user's known-working manual
                # workaround is to click, clear, click again at the
                # SAME coordinate -- the second click is correct.  We
                # automate that here so the user never sees the wonky
                # first result.  Only runs once per load_video.
                if not first_click_done:
                    print(f"[sam2_worker] first-click retry: clicking "
                          f"({x},{y}) twice with reset_state between",
                          file=sys.stderr, flush=True)
                    # First (throwaway) add at the user's coordinate.
                    predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=frame_idx,
                        obj_id=obj_id, points=pts, labels=labels,
                    )
                    # Reset and click again -- this is the call whose
                    # mask we return.
                    if hasattr(predictor, "reset_state"):
                        predictor.reset_state(state)
                    first_click_done = True
                # clear_old_points=False so that a 2nd, 3rd, ... click
                # ACCUMULATES with the existing prompts for this obj_id
                # instead of replacing them.  Without this, the SAM2
                # default of True throws away the positive click as
                # soon as the user adds a refining negative click,
                # leaving SAM2 with only the negative prompt and
                # returning a stale mask.
                _f, _o, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=frame_idx,
                    obj_id=obj_id, points=pts, labels=labels,
                    clear_old_points=False,
                )
                # Extract the mask for THIS frame, for THIS object id.
                fc, h_, w_ = video_dims
                idx = 0
                if hasattr(_o, "__iter__"):
                    try:
                        idx = list(_o).index(obj_id)
                    except ValueError:
                        idx = 0
                m = (out_mask_logits[idx] > 0.0).cpu().numpy()
                m = np.squeeze(m).astype(np.uint8) * 255
                if m.shape != (h_, w_):
                    m = cv2.resize(m, (w_, h_),
                                    interpolation=cv2.INTER_LINEAR)
                # Save to caller-provided path AND/OR return inline b64.
                mask_out = req.get("mask_out_path")
                if mask_out:
                    os.makedirs(os.path.dirname(os.path.abspath(mask_out)),
                                 exist_ok=True)
                    cv2.imwrite(mask_out, m)
                mask_b64 = ""
                if req.get("return_b64", False):
                    ok, png = cv2.imencode(".png", m)
                    if ok:
                        mask_b64 = base64.b64encode(png.tobytes()).decode("ascii")
                emit({"request_id": req_id, "op": "click",
                      "status": "ok",
                      "obj_id": obj_id, "frame_idx": frame_idx,
                      "mask_out_path": mask_out,
                      "mask_b64": mask_b64,
                      "nonzero_pixels": int(m.sum() // 255)})

            elif op == "set_prompts":
                # Deterministic prompt-set model: caller sends the FULL
                # list of clicks for an obj_id and we return the mask
                # that SAM2 produces from exactly that set.  Replaces
                # the incremental click+accumulate model whose state
                # could drift across calls.  Every interaction is a
                # full reset_state + one (or per-frame) submit, so the
                # mask is a pure function of (prompts).
                if state is None:
                    raise RuntimeError("load_video must be called first")
                obj_id = int(req.get("obj_id", 1))
                display_frame = int(req["frame_idx"])
                prompts = list(req.get("prompts", []))
                mask_out = req.get("mask_out_path")
                return_b64 = bool(req.get("return_b64", False))

                fc, h_, w_ = video_dims
                if hasattr(predictor, "reset_state"):
                    predictor.reset_state(state)
                else:
                    state = predictor.init_state(
                        video_path=video_path,
                        offload_video_to_cpu=True,
                        offload_state_to_cpu=True,
                    )

                if not prompts:
                    # No prompts -> empty mask.
                    m = np.zeros((h_, w_), dtype=np.uint8)
                else:
                    # Group by frame (Phase 1 UI puts everything on the
                    # currently-displayed frame, but the protocol allows
                    # multi-frame prompts for future use).
                    by_frame = {}
                    for p in prompts:
                        fi = int(p.get("frame_idx",
                                          p.get("frame", display_frame)))
                        by_frame.setdefault(fi, []).append(p)
                    # SAM2's first add_new_points_or_box call on a freshly
                    # reset_state'd state still has a JIT-warmup quirk.
                    # Double-pump only on the very first set_prompts of
                    # this load_video so subsequent UI interactions stay
                    # snappy.
                    if not first_click_done and display_frame in by_frame:
                        warmup_prompts = by_frame[display_frame]
                        wpts = np.array(
                            [[wp["x"], wp["y"]] for wp in warmup_prompts],
                            dtype=np.float32)
                        wlabels = np.array(
                            [int(wp.get("label", 1))
                             for wp in warmup_prompts],
                            dtype=np.int32)
                        predictor.add_new_points_or_box(
                            inference_state=state, frame_idx=display_frame,
                            obj_id=obj_id, points=wpts, labels=wlabels,
                        )
                        if hasattr(predictor, "reset_state"):
                            predictor.reset_state(state)
                        first_click_done = True
                        print(f"[sam2_worker] set_prompts warmup pump on "
                              f"first call ({len(warmup_prompts)} prompts "
                              f"at frame {display_frame})",
                              file=sys.stderr, flush=True)
                    # Real submit.  Each frame's add call carries that
                    # frame's COMPLETE prompt list, which lets SAM2's
                    # default clear_old_points=True do the right thing
                    # (we wiped state above and we're populating it
                    # afresh from authoritative data).
                    out_mask_logits = None
                    out_obj_ids = []
                    for fi, fprompts in by_frame.items():
                        pts = np.array(
                            [[p["x"], p["y"]] for p in fprompts],
                            dtype=np.float32)
                        labels = np.array(
                            [int(p.get("label", 1)) for p in fprompts],
                            dtype=np.int32)
                        _f, _o, _logits = predictor.add_new_points_or_box(
                            inference_state=state, frame_idx=fi,
                            obj_id=obj_id, points=pts, labels=labels,
                        )
                        if fi == display_frame:
                            out_mask_logits = _logits
                            out_obj_ids = (list(_o)
                                            if hasattr(_o, "__iter__")
                                            else [obj_id])
                    if out_mask_logits is None:
                        # display_frame had no prompts; return empty.
                        m = np.zeros((h_, w_), dtype=np.uint8)
                    else:
                        idx = 0
                        try:
                            idx = out_obj_ids.index(obj_id)
                        except (ValueError, AttributeError):
                            idx = 0
                        m = (out_mask_logits[idx] > 0.0).cpu().numpy()
                        m = np.squeeze(m).astype(np.uint8) * 255
                        if m.shape != (h_, w_):
                            m = cv2.resize(m, (w_, h_),
                                            interpolation=cv2.INTER_LINEAR)
                # ---- guaranteed negative-click carve ----------
                # SAM2 sometimes ignores a negative click that lands
                # inside a high-confidence positive region (the whole
                # object stays masked).  To guarantee the negative
                # click does SOMETHING visible, punch a disc of radius
                # `neg_carve_radius` to 0 at every negative point on
                # the display frame.  Caller controls the radius;
                # 0 disables the override.
                try:
                    neg_carve_radius = int(req.get("neg_carve_radius", 0))
                except Exception:
                    neg_carve_radius = 0
                if neg_carve_radius > 0 and prompts:
                    for p in prompts:
                        try:
                            fi_p = int(p.get("frame_idx",
                                              p.get("frame", display_frame)))
                            lab_p = int(p.get("label", 1))
                        except Exception:
                            continue
                        if fi_p != display_frame or lab_p != 0:
                            continue
                        try:
                            cx, cy = int(p["x"]), int(p["y"])
                        except Exception:
                            continue
                        cv2.circle(m, (cx, cy), neg_carve_radius, 0,
                                    -1, lineType=cv2.LINE_AA)
                if mask_out:
                    os.makedirs(
                        os.path.dirname(os.path.abspath(mask_out)),
                        exist_ok=True)
                    cv2.imwrite(mask_out, m)
                mask_b64 = ""
                if return_b64:
                    ok, png = cv2.imencode(".png", m)
                    if ok:
                        mask_b64 = base64.b64encode(
                            png.tobytes()).decode("ascii")
                emit({"request_id": req_id, "op": "set_prompts",
                      "status": "ok",
                      "obj_id": obj_id, "frame_idx": display_frame,
                      "n_prompts": len(prompts),
                      "mask_out_path": mask_out,
                      "mask_b64": mask_b64,
                      "nonzero_pixels": int(m.sum() // 255)})

            elif op == "apply_click":
                # Demo-style multi-object incremental update.  ONE
                # add_new_points_or_box call for the named obj_id
                # carrying that obj_id's COMPLETE point list for this
                # frame.  No reset_state -- other obj_ids' state is
                # preserved.  Returns per-obj_id masks for the display
                # frame so the UI can union/composite them.
                if state is None:
                    raise RuntimeError("load_video must be called first")
                frame_idx = int(req["frame_idx"])
                obj_id = int(req["obj_id"])
                points = req.get("points") or []
                labels = req.get("labels") or []
                return_b64 = bool(req.get("return_b64", False))
                masks_out_root = req.get("masks_out_root")

                fc, h_, w_ = video_dims

                # Empty point list = remove this obj from tracking.
                if not points:
                    removed = False
                    try:
                        predictor.remove_object(
                            inference_state=state, obj_id=obj_id)
                        removed = True
                    except Exception as exc:
                        print(f"[sam2_worker] remove_object({obj_id}) "
                              f"failed: {exc}",
                              file=sys.stderr, flush=True)
                    # Drop the obj's mask directory on disk so the UI's
                    # disk-union fallback no longer paints it.
                    if masks_out_root:
                        odir = os.path.join(masks_out_root,
                                              f"obj_{int(obj_id)}")
                        if os.path.isdir(odir):
                            try:
                                shutil.rmtree(odir, ignore_errors=True)
                            except Exception as exc:
                                print(f"[sam2_worker] rmtree({odir}) "
                                      f"failed: {exc}",
                                      file=sys.stderr, flush=True)
                    # Forget this obj from the warmup-tracker so a
                    # later re-click on the same obj_id re-arms its
                    # per-obj warmup.
                    try:
                        seen_obj_ids.discard(int(obj_id))
                    except Exception:
                        pass
                    emit({"request_id": req_id, "op": "apply_click",
                          "status": "ok",
                          "frame_idx": frame_idx,
                          "removed_obj_id": obj_id if removed else None,
                          "obj_ids": [], "obj_masks": {}})
                    continue

                pts = np.array(points, dtype=np.float32)
                labs = np.array(labels, dtype=np.int32)

                # Warmup logic.  Two flavours:
                #   * GLOBAL warmup (first apply_click of this
                #     load_video): add -> reset_state -> add.  The
                #     reset is safe here because no other obj_ids
                #     exist yet.
                #   * PER-OBJ warmup (first time a given obj_id is
                #     seen, AFTER the global warmup): add -> add.
                #     No reset_state, so other obj_ids' state in
                #     inference_state is preserved.  The default
                #     clear_old_points=True on the second add
                #     replaces the first add's points so no
                #     contradictory accumulation happens.
                if not first_click_done:
                    try:
                        predictor.add_new_points_or_box(
                            inference_state=state, frame_idx=frame_idx,
                            obj_id=obj_id, points=pts, labels=labs,
                        )
                    except Exception as exc:
                        print(f"[sam2_worker] apply_click global "
                              f"warmup add failed: {exc}",
                              file=sys.stderr, flush=True)
                    if hasattr(predictor, "reset_state"):
                        predictor.reset_state(state)
                    first_click_done = True
                    seen_obj_ids = set()
                    print(f"[sam2_worker] apply_click GLOBAL warmup "
                          f"done (obj={obj_id})",
                          file=sys.stderr, flush=True)
                elif obj_id not in seen_obj_ids:
                    try:
                        predictor.add_new_points_or_box(
                            inference_state=state, frame_idx=frame_idx,
                            obj_id=obj_id, points=pts, labels=labs,
                        )
                    except Exception as exc:
                        print(f"[sam2_worker] apply_click per-obj "
                              f"warmup add failed (obj={obj_id}): "
                              f"{exc}",
                              file=sys.stderr, flush=True)
                    print(f"[sam2_worker] apply_click PER-OBJ warmup "
                          f"done (obj={obj_id})",
                          file=sys.stderr, flush=True)

                # Real submit -- only this obj_id is touched; default
                # clear_old_points=True replaces obj_id's points on
                # this frame; other obj_ids unaffected.
                _f, out_obj_ids, out_mask_logits = (
                    predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=frame_idx,
                        obj_id=obj_id, points=pts, labels=labs,
                    )
                )
                seen_obj_ids.add(int(obj_id))

                obj_ids_list = (list(out_obj_ids)
                                  if hasattr(out_obj_ids, "__iter__")
                                  else [obj_id])
                obj_masks_b64 = {}
                obj_nonzero = {}
                for i, oid in enumerate(obj_ids_list):
                    try:
                        m = (out_mask_logits[i] > 0.0).cpu().numpy()
                        m = np.squeeze(m).astype(np.uint8) * 255
                    except Exception:
                        m = np.zeros((h_, w_), dtype=np.uint8)
                    if m.shape != (h_, w_):
                        m = cv2.resize(m, (w_, h_),
                                        interpolation=cv2.INTER_LINEAR)
                    if masks_out_root:
                        odir = os.path.join(masks_out_root,
                                              f"obj_{int(oid)}")
                        try:
                            os.makedirs(odir, exist_ok=True)
                            cv2.imwrite(os.path.join(
                                odir,
                                f"frame_{frame_idx:06d}.png"), m)
                        except Exception as exc:
                            print(f"[sam2_worker] mask save failed "
                                  f"(obj {oid}): {exc}",
                                  file=sys.stderr, flush=True)
                    if return_b64:
                        ok, png = cv2.imencode(".png", m)
                        if ok:
                            obj_masks_b64[str(int(oid))] = (
                                base64.b64encode(
                                    png.tobytes()).decode("ascii"))
                    obj_nonzero[str(int(oid))] = int(m.sum() // 255)

                emit({"request_id": req_id, "op": "apply_click",
                      "status": "ok",
                      "frame_idx": frame_idx,
                      "obj_ids": [int(o) for o in obj_ids_list],
                      "obj_masks": obj_masks_b64,
                      "obj_nonzero": obj_nonzero})

            elif op == "set_all_prompts":
                # Multi-object deterministic prompt-set submission.
                # Caller sends the COMPLETE list of every object's
                # prompts; daemon does ONE reset_state then populates
                # every object atomically.  Returns per-object masks
                # for the display frame so the UI can compose a
                # multi-colored overlay.  This is the multi-object
                # peer of set_prompts -- use it whenever there is more
                # than one tracked object so different objects do not
                # wipe each other on consecutive set_prompts calls.
                if state is None:
                    raise RuntimeError("load_video must be called first")
                display_frame = int(req["frame_idx"])
                obj_specs = list(req.get("objects", []))
                return_b64 = bool(req.get("return_b64", False))
                masks_root = req.get("masks_root")  # optional

                fc, h_, w_ = video_dims
                if hasattr(predictor, "reset_state"):
                    predictor.reset_state(state)

                # Warmup pump on the very first set_all_prompts of
                # this load_video, mirroring the set_prompts logic.
                if not first_click_done and obj_specs:
                    first = obj_specs[0]
                    fprompts = [p for p in first.get("prompts", [])
                                 if int(p.get("frame_idx",
                                                p.get("frame",
                                                    display_frame)))
                                    == display_frame]
                    if fprompts:
                        wpts = np.array([[p["x"], p["y"]] for p in fprompts],
                                          dtype=np.float32)
                        wlabels = np.array(
                            [int(p.get("label", 1)) for p in fprompts],
                            dtype=np.int32)
                        predictor.add_new_points_or_box(
                            inference_state=state, frame_idx=display_frame,
                            obj_id=int(first.get("obj_id", 1)),
                            points=wpts, labels=wlabels,
                        )
                        if hasattr(predictor, "reset_state"):
                            predictor.reset_state(state)
                        first_click_done = True
                        print("[sam2_worker] set_all_prompts warmup pump",
                              file=sys.stderr, flush=True)

                # Submit each object's prompts.  Group per object per
                # frame; pass each (obj_id, frame) group as one
                # add_new_points_or_box call with all its points.
                per_obj_logits = {}  # obj_id -> mask logits at display_frame
                per_obj_oidlist = {}
                for spec in obj_specs:
                    oid = int(spec.get("obj_id", 1))
                    prompts = list(spec.get("prompts", []))
                    if not prompts:
                        continue
                    by_frame = {}
                    for p in prompts:
                        fi = int(p.get("frame_idx",
                                         p.get("frame", display_frame)))
                        by_frame.setdefault(fi, []).append(p)
                    for fi, fprompts in by_frame.items():
                        pts = np.array(
                            [[p["x"], p["y"]] for p in fprompts],
                            dtype=np.float32)
                        labels = np.array(
                            [int(p.get("label", 1)) for p in fprompts],
                            dtype=np.int32)
                        _f, _o, _logits = predictor.add_new_points_or_box(
                            inference_state=state, frame_idx=fi,
                            obj_id=oid, points=pts, labels=labels,
                        )
                        if fi == display_frame:
                            per_obj_logits[oid] = _logits
                            per_obj_oidlist[oid] = (
                                list(_o) if hasattr(_o, "__iter__")
                                else [oid])

                # Build per-object masks for display frame.
                obj_results = []  # [{"obj_id": int, "mask_b64": str, "nonzero": int}]
                for spec in obj_specs:
                    oid = int(spec.get("obj_id", 1))
                    if oid not in per_obj_logits:
                        m = np.zeros((h_, w_), dtype=np.uint8)
                    else:
                        oids_seen = per_obj_oidlist.get(oid, [oid])
                        try:
                            idx = oids_seen.index(oid)
                        except ValueError:
                            idx = 0
                        m = (per_obj_logits[oid][idx] > 0.0).cpu().numpy()
                        m = np.squeeze(m).astype(np.uint8) * 255
                        if m.shape != (h_, w_):
                            m = cv2.resize(m, (w_, h_),
                                            interpolation=cv2.INTER_LINEAR)
                    # Optional per-object PNG on disk (handy for propagation)
                    if masks_root:
                        out_dir = os.path.join(masks_root,
                                                 f"obj_{oid}")
                        os.makedirs(out_dir, exist_ok=True)
                        cv2.imwrite(
                            os.path.join(out_dir,
                                          f"frame_{display_frame:06d}.png"),
                            m)
                    entry = {"obj_id": oid,
                              "nonzero_pixels": int(m.sum() // 255)}
                    if return_b64:
                        ok, png = cv2.imencode(".png", m)
                        if ok:
                            entry["mask_b64"] = base64.b64encode(
                                png.tobytes()).decode("ascii")
                    obj_results.append(entry)

                emit({"request_id": req_id, "op": "set_all_prompts",
                      "status": "ok",
                      "frame_idx": display_frame,
                      "objects": obj_results})

            elif op == "propagate":
                if state is None:
                    raise RuntimeError("load_video must be called first")
                masks_dir = req["masks_dir"]
                os.makedirs(masks_dir, exist_ok=True)
                fc, h_, w_ = video_dims
                written = 0
                last_progress = 0
                for fi, oids, mls in predictor.propagate_in_video(state):
                    union = np.zeros((h_, w_), dtype=np.uint8)
                    for i, _oid in enumerate(oids):
                        m = (mls[i] > 0.0).cpu().numpy()
                        m = np.squeeze(m).astype(np.uint8) * 255
                        if m.shape != (h_, w_):
                            m = cv2.resize(m, (w_, h_),
                                            interpolation=cv2.INTER_LINEAR)
                        np.maximum(union, m, out=union)
                    cv2.imwrite(os.path.join(masks_dir,
                                              f"frame_{fi:06d}.png"),
                                 union)
                    written += 1
                    if fi - last_progress >= 25:
                        emit({"request_id": req_id, "op": "propagate",
                              "status": "progress",
                              "frame_idx": int(fi), "total": int(fc)})
                        last_progress = fi
                emit({"request_id": req_id, "op": "propagate",
                      "status": "ok", "masks_dir": masks_dir,
                      "frames_written": written})

            elif op == "clear":
                obj_id = req.get("obj_id")
                if state is not None:
                    if obj_id is None:
                        # Reset entire state (all objects).
                        if hasattr(predictor, "reset_state"):
                            predictor.reset_state(state)
                        else:
                            state = predictor.init_state(
                                video_path=video_path,
                                offload_video_to_cpu=True,
                                offload_state_to_cpu=True,
                            )
                    else:
                        # SAM2 doesn't expose per-object clear; fall
                        # back to reset_state for the whole graph.  The
                        # caller is expected to re-submit clicks for
                        # objects to keep.
                        if hasattr(predictor, "reset_state"):
                            predictor.reset_state(state)
                        else:
                            state = predictor.init_state(
                                video_path=video_path,
                                offload_video_to_cpu=True,
                                offload_state_to_cpu=True,
                            )
                # After a reset_state, the very next add_new_points_or_box
                # hits SAM2's wonky-first-call quirk again.  Re-arm the
                # warmup pump so the next click is correct.
                first_click_done = False
                seen_obj_ids = set()
                emit({"request_id": req_id, "op": "clear",
                      "status": "ok"})

            elif op == "ping":
                emit({"request_id": req_id, "op": "ping",
                      "status": "ok"})

            elif op == "shutdown":
                emit({"request_id": req_id, "op": "shutdown",
                      "status": "ok"})
                break

            else:
                raise ValueError(f"unknown op: {op}")

        except Exception as exc:
            emit({"request_id": req_id, "op": op or "?",
                  "status": "error", "message": str(exc),
                  "traceback": traceback.format_exc()})


def main() -> None:
    # Daemon-mode dispatch: if --daemon is on argv, run the daemon
    # event loop instead of the legacy CLI.
    if "--daemon" in sys.argv:
        daemon_main()
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument(
        "--click", action="append", nargs="+", type=int, required=True,
        help="A click as (x, y, frame) or (x, y, frame, label). "
             "label=1 = positive (default), label=0 = negative refinement. "
             "Without --single_object, each --click becomes its own object. "
             "With --single_object, all --click args refine the SAME object "
             "(allows mixing positive + negative clicks to sculpt one mask).")
    ap.add_argument("--single_object", action="store_true",
                    help="Treat all --click args as refinement clicks on a "
                         "single object (obj_id=1). The label (4th value of "
                         "--click) determines + or - refinement.")
    ap.add_argument("--out", required=True,
                    help="path to write the (T, H, W) uint8 .npy")
    ap.add_argument("--sam2_ckpt", type=str, default=None,
                    help="Absolute path to SAM2 weights (.pt). When "
                         "supplied, the worker uses Hydra's package-"
                         "based config resolution and NO chdir is "
                         "performed -- decouples from KeySync. "
                         "When omitted, falls back to legacy "
                         "KeySync-chdir behavior.")
    ap.add_argument("--sam2_config", type=str, default=None,
                    help="Hydra config name, e.g. "
                         "'configs/sam2.1/sam2.1_hiera_b+.yaml'. "
                         "Defaults based on the weights filename.")
    args = ap.parse_args()

    # Normalize clicks to 4-tuples (x, y, frame, label) with label
    # defaulting to 1 if absent (positive).
    clicks_raw = args.click
    clicks = []
    for c in clicks_raw:
        if len(c) == 3:
            clicks.append((c[0], c[1], c[2], 1))
        elif len(c) == 4:
            clicks.append((c[0], c[1], c[2], c[3]))
        else:
            sys.exit(f"ERROR: --click takes 3 or 4 ints, got {c}")
    print(f"[sam2_worker] {len(clicks)} click(s) "
          f"({'single object' if args.single_object else 'one per object'})")

    # ---- SAM2 install resolution ----
    # Two paths:
    #   (A) --sam2_ckpt supplied: NO chdir. Hydra resolves configs via
    #       the installed sam2 package's bundled config tree using
    #       initialize_config_module. This is the v2-native path and
    #       does not require KeySync.
    #   (B) --sam2_ckpt omitted: legacy KeySync-chdir behavior.
    #       Worker chdirs into KeySync repo so Hydra's default CWD
    #       search finds configs/sam2.1/*.yaml there. Weights load
    #       from KeySync/pretrained_models/checkpoints/.
    _DEFAULT_CFG_BY_NAME = {
        "sam2.1_hiera_tiny.pt":       "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_small.pt":      "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_base_plus.pt":  "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_large.pt":      "configs/sam2.1/sam2.1_hiera_l.yaml",
    }
    if args.sam2_ckpt:
        sam2_ckpt = os.path.abspath(args.sam2_ckpt)
        if not os.path.isfile(sam2_ckpt):
            sys.exit(f"ERROR: SAM2 weights not found at {sam2_ckpt}")
        model_cfg = args.sam2_config or _DEFAULT_CFG_BY_NAME.get(
            os.path.basename(sam2_ckpt),
            "configs/sam2.1/sam2.1_hiera_b+.yaml")
        # Package-init Hydra so it can find configs inside the sam2
        # pip-install rather than CWD-relative. SAM2 ships its configs
        # as package data under sam2/configs/. Recent SAM2 versions do
        # this transparently in build_sam2_video_predictor; older
        # versions need an explicit initialize_config_module.
        try:
            from hydra.core.global_hydra import GlobalHydra
            from hydra import initialize_config_module
            if not GlobalHydra.instance().is_initialized():
                initialize_config_module(
                    config_module="sam2", version_base=None)
        except Exception as _hydra_exc:
            print(f"[sam2_worker] WARN hydra init: {_hydra_exc} "
                  f"(may still work if sam2 handles it internally)")
        from sam2.build_sam import build_sam2_video_predictor  # noqa: E402
        print(f"[sam2_worker] v2 path: ckpt={sam2_ckpt} cfg={model_cfg}")
    else:
        # Legacy KeySync-chdir path (kept for backward compat).
        keysync_dir = os.environ.get(
            "KEYSYNC_REPO_DIR",
            r"F:\faceprodpbraw\lipsync_test\KeySync")
        if not os.path.isdir(keysync_dir):
            sys.exit(f"ERROR: KEYSYNC_REPO_DIR not found: {keysync_dir}")
        os.chdir(keysync_dir)
        from sam2.build_sam import build_sam2_video_predictor  # noqa: E402
        sam2_ckpt = os.path.join(keysync_dir, "pretrained_models",
                                  "checkpoints", "sam2.1_hiera_large.pt")
        if not os.path.isfile(sam2_ckpt):
            sys.exit(f"ERROR: SAM2 weights not found at {sam2_ckpt}")
        model_cfg = args.sam2_config or "configs/sam2.1/sam2.1_hiera_l.yaml"
        print(f"[sam2_worker] legacy KeySync path: cwd={keysync_dir} "
              f"cfg={model_cfg}")

    print(f"[sam2_worker] building SAM2 predictor")
    predictor = build_sam2_video_predictor(model_cfg, sam2_ckpt,
                                             device="cuda")
    state = predictor.init_state(
        video_path=args.video,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )

    if args.single_object:
        # All clicks refine object_id=1. Group by frame_idx so each
        # frame's clicks are submitted as one add_new_points_or_box call
        # (SAM2 expects all clicks for a given frame to be batched).
        from collections import defaultdict
        by_frame = defaultdict(list)
        for x, y, frame, label in clicks:
            by_frame[int(frame)].append((x, y, label))
        for frame_idx, lst in sorted(by_frame.items()):
            pts = np.array([[x, y] for (x, y, _) in lst], dtype=np.float32)
            labels = np.array([lab for (_, _, lab) in lst], dtype=np.int32)
            sign_summary = "/".join(("+" if lab else "-") for lab in labels)
            print(f"[sam2_worker] obj#1 frame {frame_idx}: "
                  f"{len(lst)} clicks ({sign_summary})")
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=1,
                points=pts,
                labels=labels,
            )
    else:
        # Backward-compat: one click per object_id.
        for obj_id, (x, y, frame, label) in enumerate(clicks, start=1):
            pts = np.array([[x, y]], dtype=np.float32)
            labels = np.array([label], dtype=np.int32)
            print(f"[sam2_worker] obj#{obj_id}: click ({x},{y}) frame {frame} "
                  f"label={label}")
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame,
                obj_id=obj_id,
                points=pts,
                labels=labels,
            )

    # Determine output shape
    import cv2  # noqa: E402
    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    print(f"[sam2_worker] video has {total} frames at {w}x{h}")

    # Allocate union mask
    masks = np.zeros((total, h, w), dtype=np.uint8)

    print("[sam2_worker] propagating masks across video...")
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        # mask_logits is a tensor of shape (n_objs, 1, H_low, W_low)
        for i, _oid in enumerate(obj_ids):
            m = (mask_logits[i] > 0.0).cpu().numpy().squeeze().astype(np.uint8) * 255
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            # Union with existing mask at this frame
            np.maximum(masks[frame_idx], m, out=masks[frame_idx])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, masks)
    pixels = int(masks.sum())
    avg = pixels // (total * 255) if total else 0
    print(f"[sam2_worker] DONE: {args.out} "
          f"shape={masks.shape} total_pixels={pixels} avg_per_frame={avg}")


if __name__ == "__main__":
    main()
