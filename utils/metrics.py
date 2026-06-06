"""Quality metrics: cosine similarity, PSNR, SSIM, LPIPS."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors.

    Typically used with ArcFace 512-d embeddings to measure identity
    consistency across frames.

    Args:
        a: First vector (1-D).
        b: Second vector (1-D), same shape as *a*.

    Returns:
        Cosine similarity in [-1, 1]. Higher = more similar.
    """
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / denom)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images.

    Both images must have the same shape and dtype. For uint8 images the
    data range is 255; for float images the range is assumed to be [0, 1].

    Args:
        img1: First image (BGR or grayscale).
        img2: Second image, same shape as img1.

    Returns:
        PSNR value in dB. Higher = more similar. 40+ dB is excellent.
    """
    if img1.shape != img2.shape:
        raise ValueError(
            f"Shape mismatch: {img1.shape} vs {img2.shape}"
        )
    if img1.dtype == np.uint8:
        data_range = 255.0
        diff = img1.astype(np.float64) - img2.astype(np.float64)
    else:
        data_range = 1.0
        diff = img1.astype(np.float64) - img2.astype(np.float64)

    mse = np.mean(diff ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def compute_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int = 11,
    data_range: Optional[float] = None,
) -> float:
    """Compute Structural Similarity Index (SSIM) between two images.

    Uses a Gaussian window of the specified size. This is a pure NumPy
    implementation — no scikit-image dependency required.

    Args:
        img1: First image (grayscale or single channel).
        img2: Second image, same shape as img1.
        window_size: Size of the Gaussian window.
        data_range: Dynamic range of pixel values. Auto-detected if None.

    Returns:
        Mean SSIM value in [-1, 1]. Higher = more similar.
    """
    if img1.shape != img2.shape:
        raise ValueError(
            f"Shape mismatch: {img1.shape} vs {img2.shape}"
        )

    # Convert to float64
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if data_range is None:
        if img1.max() <= 1.0 and img2.max() <= 1.0:
            data_range = 1.0
        else:
            data_range = 255.0

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Gaussian window via cv2.GaussianBlur
    mu1 = cv2.GaussianBlur(img1, (window_size, window_size), 1.5)
    mu2 = cv2.GaussianBlur(img2, (window_size, window_size), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (window_size, window_size), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (window_size, window_size), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (window_size, window_size), 1.5) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / (denominator + 1e-10)
    return float(ssim_map.mean())


def compute_lpips(
    img1: np.ndarray,
    img2: np.ndarray,
    net: str = "alex",
) -> float:
    """Compute Learned Perceptual Image Patch Similarity (LPIPS).

    Requires the `lpips` package (pip install lpips). Falls back to a
    PSNR-based approximation if the package is unavailable.

    Args:
        img1: First BGR uint8 image (H, W, 3).
        img2: Second BGR uint8 image (H, W, 3).
        net: Backbone network — "alex", "vgg", or "squeeze".

    Returns:
        LPIPS distance. Lower = more similar. 0 = identical.
    """
    try:
        import lpips as _lpips
        import torch

        # BGR uint8 → RGB float [-1, 1]
        def _to_tensor(img: np.ndarray) -> torch.Tensor:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 127.5 - 1.0
            return t.unsqueeze(0)

        t1 = _to_tensor(img1)
        t2 = _to_tensor(img2)

        # Use GPU if available, else CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_fn = _lpips.LPIPS(net=net).to(device)
        t1 = t1.to(device)
        t2 = t2.to(device)

        with torch.no_grad():
            dist = loss_fn(t1, t2)

        return float(dist.item())

    except ImportError:
        # Fallback: approximate with 1 - SSIM (rough but no extra deps)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ssim = compute_ssim(gray1, gray2)
        return 1.0 - ssim


def compute_identity_drift_series(
    embeddings: list[np.ndarray],
    reference: np.ndarray,
) -> list[float]:
    """Compute per-frame cosine similarity against a reference embedding.

    Useful for plotting identity drift over time. A monotonically decreasing
    trend indicates progressive identity loss.

    Args:
        embeddings: List of per-frame ArcFace embeddings (each 512-d).
        reference: Reference (source) embedding (512-d).

    Returns:
        List of cosine similarity values, one per frame.
    """
    return [cosine_similarity(emb, reference) for emb in embeddings]


def compute_reconstruction_error(
    original: np.ndarray,
    swapped: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute reconstruction quality metrics in the face region only.

    Args:
        original: Original frame (BGR uint8).
        swapped: Swapped frame (BGR uint8), same shape.
        mask: Optional binary mask (uint8, 255=face region). If None, the
            entire frame is compared.

    Returns:
        Dict with 'psnr', 'ssim', 'mse' keys.
    """
    if original.shape != swapped.shape:
        raise ValueError("Shape mismatch between original and swapped")

    if mask is not None:
        mask_bool = mask > 127
        orig_masked = original[mask_bool].reshape(-1, 3)
        swap_masked = swapped[mask_bool].reshape(-1, 3)
        mse = float(np.mean((orig_masked.astype(np.float64) - swap_masked.astype(np.float64)) ** 2))
        # For PSNR/SSIM with mask, crop to mask bbox
        ys, xs = np.where(mask > 127)
        if len(ys) == 0:
            return {"psnr": 0.0, "ssim": 0.0, "mse": mse}
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        orig_crop = original[y1:y2, x1:x2]
        swap_crop = swapped[y1:y2, x1:x2]
        psnr = compute_psnr(orig_crop, swap_crop)
        gray1 = cv2.cvtColor(orig_crop, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(swap_crop, cv2.COLOR_BGR2GRAY)
        ssim = compute_ssim(gray1, gray2)
    else:
        mse = float(np.mean((original.astype(np.float64) - swapped.astype(np.float64)) ** 2))
        psnr = compute_psnr(original, swapped)
        gray1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(swapped, cv2.COLOR_BGR2GRAY)
        ssim = compute_ssim(gray1, gray2)

    return {"psnr": psnr, "ssim": ssim, "mse": mse}
