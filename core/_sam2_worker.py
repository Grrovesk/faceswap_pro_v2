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
import sys

import numpy as np


def main() -> None:
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
        keysync_dir = os.environ.get("KEYSYNC_REPO_DIR", "")
        if not keysync_dir or not os.path.isdir(keysync_dir):
            sys.exit(
                "ERROR: legacy SAM2 path requires KEYSYNC_REPO_DIR env "
                "var pointing at a KeySync clone. Set it explicitly, or "
                "pass --sam2_ckpt to use the v2 native path.")
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
