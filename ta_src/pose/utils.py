"""Geometry helpers shared by the ViTPose wrapper.

Implements the SimpleBaseline / ViTPose crop convention:
  - axis-aligned affine transforms (no rotation)
  - heatmap argmax decode with sub-pixel dark-side correction
"""
from __future__ import annotations

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Affine transform (crop ↔ full-frame)
# --------------------------------------------------------------------------- #

def bbox_to_center_dims(
    x1: float, y1: float, x2: float, y2: float,
    input_w: int, input_h: int,
    padding: float = 1.25,
) -> tuple[np.ndarray, float, float]:
    """Compute crop center, padded width, and padded height from a bbox.

    Aspect-ratio correction ensures the padded region matches the model's
    input aspect ratio (input_w / input_h) before applying the padding factor.

    Returns:
        center:  (2,) float32 [cx, cy] in full-frame pixels
        crop_w:  padded crop width  in full-frame pixels
        crop_h:  padded crop height in full-frame pixels
    """
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1

    aspect = input_w / input_h  # e.g. 192/256 = 0.75
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect

    w *= padding
    h *= padding

    return np.array([cx, cy], dtype=np.float32), float(w), float(h)


def get_affine_transform(
    center: np.ndarray,
    crop_w: float,
    crop_h: float,
    dst_w: int,
    dst_h: int,
    inv: bool = False,
) -> np.ndarray:
    """Build a 2×3 affine matrix mapping a crop_w×crop_h region (centred at
    *center* in full-frame space) to/from a dst_w×dst_h image.

    Three non-collinear control points uniquely determine the transform:
    crop centre, right-centre, and top-centre.  No rotation is applied because
    our inputs are axis-aligned bounding boxes.
    """
    src = np.array([
        [center[0],                center[1]],           # centre
        [center[0] + crop_w / 2.0, center[1]],           # right-centre
        [center[0],                center[1] - crop_h / 2.0],  # top-centre
    ], dtype=np.float32)

    dst = np.array([
        [dst_w / 2.0, dst_h / 2.0],
        [dst_w,       dst_h / 2.0],
        [dst_w / 2.0, 0.0],
    ], dtype=np.float32)

    if inv:
        return cv2.getAffineTransform(dst, src)
    return cv2.getAffineTransform(src, dst)


# --------------------------------------------------------------------------- #
# Heatmap decode
# --------------------------------------------------------------------------- #

def get_max_preds(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Argmax decode of heatmaps.

    Args:
        heatmaps: (N, K, H, W) float32

    Returns:
        preds:   (N, K, 2) float32  — (col, row) in heatmap pixel space
        maxvals: (N, K)   float32  — peak value (proxy for confidence)
    """
    N, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(N, K, -1)
    idx = np.argmax(flat, axis=2)          # (N, K)
    maxvals = flat[np.arange(N)[:, None], np.arange(K)[None, :], idx]  # (N, K)

    preds = np.zeros((N, K, 2), dtype=np.float32)
    preds[:, :, 0] = idx % W              # col
    preds[:, :, 1] = idx // W             # row

    # Zero out predictions at background (argmax at position 0 with low value)
    maxvals[maxvals < 0.0] = 0.0
    preds[maxvals == 0.0] = 0.0

    return preds, maxvals


def dark_postprocess(heatmaps: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Sub-pixel refinement via the DARK (Distribution-Aware coordinate
    Representation) method: shift each peak by the gradient of a local
    Gaussian fit.  Improves accuracy ~0.5 AP with negligible cost.

    Args:
        heatmaps: (N, K, H, W) float32
        preds:    (N, K, 2) float32  — integer (col, row) from get_max_preds

    Returns:
        refined (N, K, 2) float32 still in heatmap pixel space
    """
    N, K, H, W = heatmaps.shape
    refined = preds.copy()

    # Apply Gaussian smoothing to suppress noise before computing gradient
    smoothed = heatmaps.copy()
    for n in range(N):
        for k in range(K):
            smoothed[n, k] = cv2.GaussianBlur(heatmaps[n, k], (11, 11), 0)

    for n in range(N):
        for k in range(K):
            col = int(preds[n, k, 0])
            row = int(preds[n, k, 1])
            if col < 1 or col >= W - 1 or row < 1 or row >= H - 1:
                continue
            h = smoothed[n, k]
            # Prevent log(0)
            h = np.clip(h, 1e-10, None)
            lh = np.log(h)
            # 1-D second derivatives along col and row
            dx  = 0.5 * (lh[row, col + 1] - lh[row, col - 1])
            dy  = 0.5 * (lh[row + 1, col] - lh[row - 1, col])
            dxx = lh[row, col + 1] - 2.0 * lh[row, col] + lh[row, col - 1]
            dyy = lh[row + 1, col] - 2.0 * lh[row, col] + lh[row - 1, col]
            dxy = 0.25 * (lh[row + 1, col + 1] - lh[row + 1, col - 1]
                          - lh[row - 1, col + 1] + lh[row - 1, col - 1])
            # Hessian and its inverse
            det = dxx * dyy - dxy * dxy
            if abs(det) < 1e-12:
                continue
            inv_dxx =  dyy / det
            inv_dyy =  dxx / det
            inv_dxy = -dxy / det
            # Newton step: -H^{-1} grad
            refined[n, k, 0] = col - (inv_dxx * dx + inv_dxy * dy)
            refined[n, k, 1] = row - (inv_dxy * dx + inv_dyy * dy)

    return refined


def heatmaps_to_keypoints(
    heatmaps: np.ndarray,
    inv_transforms: list[np.ndarray],
    heatmap_w: int,
    heatmap_h: int,
    input_w: int,
    input_h: int,
    use_dark: bool = True,
) -> list[np.ndarray]:
    """Convert raw heatmap outputs to full-frame keypoint coordinates.

    Args:
        heatmaps:       (N, K, heatmap_h, heatmap_w) float32
        inv_transforms: list of N 2×3 affine matrices (heatmap-space → full-frame)
        heatmap_w/h:    spatial dims of the heatmap output (e.g. 48, 64)
        input_w/h:      model input spatial dims (e.g. 192, 256)
        use_dark:       apply DARK sub-pixel correction

    Returns:
        list of N arrays, each (K, 3) float32: [x, y, confidence]
        coordinates are in full-frame pixel space.
    """
    preds, maxvals = get_max_preds(heatmaps)  # (N,K,2), (N,K)

    if use_dark:
        preds = dark_postprocess(heatmaps, preds)

    # Scale from heatmap space to model-input space
    preds[:, :, 0] *= input_w  / heatmap_w   # col  → x in [0, input_w]
    preds[:, :, 1] *= input_h  / heatmap_h   # row  → y in [0, input_h]

    result = []
    for n in range(len(inv_transforms)):
        t = inv_transforms[n]    # 2×3 float64
        pts = preds[n]           # (K, 2)
        # Apply inverse affine: [x', y'] = t @ [x, y, 1]^T
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        pts_h = np.concatenate([pts, ones], axis=1)   # (K, 3)
        full_frame = (t @ pts_h.T).T                  # (K, 2)
        kps = np.concatenate([full_frame, maxvals[n:n+1].T], axis=1)  # (K, 3)
        result.append(kps.astype(np.float32))

    return result
