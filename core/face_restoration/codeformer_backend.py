"""CodeFormer face-restoration backend.

Loads CodeFormer via basicsr's architecture registry (basicsr is already
a GFPGAN dependency, so the install footprint is zero new packages
beyond the .pth weight file we auto-download on first use).

Mirrors core/gfpgan_worker.py structure -- module-scope cached model,
serialised per-call, paste-back through facexlib's FaceRestoreHelper.

Weight provenance: official CodeFormer release at
https://github.com/sczhou/CodeFormer/releases (codeformer.pth, ~360 MB).
"""
from __future__ import annotations

import logging
import sys
import threading
import urllib.request
from pathlib import Path
from typing import Callable, Optional

_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]   # core/face_restoration/file -> parents[2] = v2/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


_WEIGHTS_DIR = PROJECT_ROOT / "models" / "face_restoration"
_CODEFORMER_PTH = _WEIGHTS_DIR / "codeformer.pth"
_CODEFORMER_URL = (
    "https://github.com/sczhou/CodeFormer/releases/download/"
    "v0.1.0/codeformer.pth"
)

logger = logging.getLogger(__name__)

# Cached objects across calls
_model = None             # CodeFormer torch.nn.Module
_face_helper = None       # facexlib FaceRestoreHelper
_lock = threading.Lock()


def _ensure_weight(log: Callable[[str], None]) -> Path:
    """Download codeformer.pth on first use.  ~360 MB."""
    _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if _CODEFORMER_PTH.is_file() and _CODEFORMER_PTH.stat().st_size > 100_000_000:
        return _CODEFORMER_PTH
    log(f"[codeformer] downloading weights from {_CODEFORMER_URL}")
    log("            (one-time, ~360 MB)")
    tmp = _CODEFORMER_PTH.with_suffix(".pth.tmp")
    try:
        urllib.request.urlretrieve(_CODEFORMER_URL, str(tmp))
        tmp.replace(_CODEFORMER_PTH)
        log(f"[codeformer] weight ready at {_CODEFORMER_PTH}")
    except Exception as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"[codeformer] weight download FAILED: {exc}.  "
            f"Manually download {_CODEFORMER_URL} to {_CODEFORMER_PTH}"
        ) from exc
    return _CODEFORMER_PTH


def _ensure_loaded(log: Callable[[str], None]) -> None:
    """Lazy-load the CodeFormer model + face helper.  Cached on
    module scope so subsequent renders skip the load cost."""
    global _model, _face_helper
    if _model is not None and _face_helper is not None:
        return
    weight = _ensure_weight(log)

    import torch
    from basicsr.archs.codeformer_arch import CodeFormer
    from basicsr.utils.registry import ARCH_REGISTRY
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[codeformer] init on device={device}")

    # Standard CodeFormer config used by the official inference script.
    net = CodeFormer(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(device)
    ck = torch.load(str(weight), map_location=device)
    net.load_state_dict(ck["params_ema"], strict=True)
    net.eval()
    _model = net

    # Face helper used by CodeFormer's official inference -- detects,
    # aligns to 512x512 template, pastes back into the original frame.
    _face_helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        use_parse=True,
        device=device,
    )
    log("[codeformer] model + face helper ready")


def _enhance_frame(img,
                     fidelity: float = 0.7,
                     log: Callable[[str], None] = print):
    """Run one BGR frame through CodeFormer.  Returns the restored BGR frame.

    ``fidelity`` (0..1) is CodeFormer's W parameter: higher = more
    identity-preserving but lower restoration; lower = more
    aggressive restoration but more identity drift.  0.7 is the
    documented sweet spot for portraits.
    """
    import cv2
    import numpy as np
    import torch
    from basicsr.utils import img2tensor, tensor2img
    from torchvision.transforms.functional import normalize

    assert _model is not None and _face_helper is not None

    _face_helper.clean_all()
    _face_helper.read_image(img)
    n = _face_helper.get_face_landmarks_5(
        only_center_face=False, resize=640, eye_dist_threshold=5)
    if n == 0:
        return img  # no faces -> return unchanged
    _face_helper.align_warp_face()

    device = next(_model.parameters()).device
    for cropped in _face_helper.cropped_faces:
        ct = img2tensor(cropped / 255.0, bgr2rgb=True, float32=True)
        normalize(ct, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        ct = ct.unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                output = _model(ct, w=float(fidelity), adain=True)[0]
            restored_face = tensor2img(output, rgb2bgr=True,
                                          min_max=(-1, 1))
        except Exception as exc:
            log(f"[codeformer] frame restore FAILED: {exc}")
            restored_face = cropped
        restored_face = restored_face.astype("uint8")
        _face_helper.add_restored_face(restored_face)

    _face_helper.get_inverse_affine(None)
    out = _face_helper.paste_faces_to_input_image(upsample_img=img)
    return out


def enhance(video_path: Path,
             log: Callable[[str], None] = print,
             fidelity: float = 0.7) -> Path:
    """Run CodeFormer over every frame of ``video_path``.  Returns
    path to a new mp4 with restored frames.  Audio is dropped --
    callers re-mux via ffmpeg (matches gfpgan_worker.enhance contract).

    On any failure returns the input path so the pipeline still ships.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        log(f"[codeformer] input not found: {video_path}; returning as-is")
        return video_path

    try:
        with _lock:
            _ensure_loaded(log)
    except Exception as exc:
        log(f"[codeformer] init failed ({exc}); returning un-enhanced video")
        return video_path

    try:
        import cv2
    except Exception as exc:
        log(f"[codeformer] opencv not importable ({exc}); skipping")
        return video_path

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log(f"[codeformer] cannot open {video_path}; skipping")
        return video_path
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = video_path.with_name(video_path.stem + "_codeformer.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    if not writer.isOpened():
        cap.release()
        log(f"[codeformer] cannot open writer at {out_path}; skipping")
        return video_path

    import time
    t0 = time.perf_counter()
    n_done = 0
    try:
        with _lock:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                restored = _enhance_frame(frame, fidelity=fidelity,
                                            log=log)
                writer.write(restored)
                n_done += 1
                if n_done % 25 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = n_done / max(elapsed, 1e-6)
                    log(f"[codeformer] frame {n_done}/{n_frames} "
                        f"({rate:.1f} fps)")
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - t0
    log(f"[codeformer] DONE {n_done} frames in {elapsed:.1f}s "
        f"-> {out_path}")
    return out_path
