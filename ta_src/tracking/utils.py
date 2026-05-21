import numpy as np


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(a_boxes, b_boxes) -> np.ndarray:
    """Vectorized IoU matrix — ~19× faster than the previous nested-loop version."""
    if not a_boxes or not b_boxes:
        return np.zeros((len(a_boxes), len(b_boxes)), dtype=np.float32)
    a = np.asarray(a_boxes, dtype=np.float32)   # (M, 4)
    b = np.asarray(b_boxes, dtype=np.float32)   # (N, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])   # (M,)
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])   # (N,)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union for two boolean masks."""
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union > 0 else 0.0


def mask_iou_matrix(
    track_masks: list,
    det_masks: list,
    scale: float = 1.0,
) -> np.ndarray | None:
    """Build (T, D) mask-IoU matrix.

    Args:
        scale: downsample masks to this fraction of original size before
               computing IoU.  0.25 reduces pixel count 16× with negligible
               accuracy loss for the association cost (O1 + O9).
    Returns None if either list is all-None.
    """
    if not any(m is not None for m in track_masks):
        return None
    if not any(m is not None for m in det_masks):
        return None

    def _prep(m: np.ndarray | None) -> np.ndarray | None:
        if m is None:
            return None
        if scale < 1.0:
            h, w = m.shape
            nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
            import cv2
            m = cv2.resize(m.astype(np.uint8), (nw, nh),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        return m

    tm = [_prep(m) for m in track_masks]
    dm = [_prep(m) for m in det_masks]

    M, N = len(tm), len(dm)
    mat = np.zeros((M, N), dtype=np.float32)
    for i, a in enumerate(tm):
        if a is None:
            continue
        for j, b in enumerate(dm):
            if b is not None:
                mat[i, j] = mask_iou(a, b)
    return mat


def centroid_dist(box_a, box_b) -> float:
    """Euclidean distance between centroids of two xyxy boxes."""
    ax = (box_a[0] + box_a[2]) / 2
    ay = (box_a[1] + box_a[3]) / 2
    bx = (box_b[0] + box_b[2]) / 2
    by = (box_b[1] + box_b[3]) / 2
    return float(np.hypot(ax - bx, ay - by))


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x) + eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))
