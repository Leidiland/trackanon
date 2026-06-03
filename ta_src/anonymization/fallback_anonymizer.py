from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


class FallbackAnonymizer:
    """Scale-aware Gaussian blur with mask dilation + alpha feathering.

    The legacy `frame[m] = blurred[m]` binary copy leaked at three failure
    modes: (1) hard mask boundary visible as a contour, (2) hair/beard/nose
    that fall outside SAM 3's tight skin mask, (3) σ ≈ 8 px too weak for 4K
    subjects. The three knobs below address each in turn — dilation extends
    coverage beyond the SAM 3 mask, feather softens the boundary, and the
    kernel scales with bbox height so 4K subjects get a σ proportional to
    head size.
    """

    def __init__(
        self,
        kernel_min: int = 51,
        kernel_frac: float = 0.20,
        dilate_min: int = 5,
        dilate_frac: float = 0.025,
        feather_min: int = 5,
        feather_frac: float = 0.020,
    ):
        if kernel_min % 2 == 0 or kernel_min < 1:
            raise ValueError(f"kernel_min must be a positive odd integer, got {kernel_min}")
        self._kernel_min = kernel_min
        self._kernel_frac = float(kernel_frac)
        self._dilate_min = int(dilate_min)
        self._dilate_frac = float(dilate_frac)
        self._feather_min = int(feather_min)
        self._feather_frac = float(feather_frac)

    # ---- scale helpers ------------------------------------------------------

    def _kernel_for_height(self, bbox_height: int) -> int:
        k = max(self._kernel_min, int(round(self._kernel_frac * bbox_height)))
        if k % 2 == 0:
            k += 1
        return k

    def _dilate_for_short(self, bbox_short_side: int) -> int:
        return max(self._dilate_min, int(round(self._dilate_frac * bbox_short_side)))

    def _feather_for_short(self, bbox_short_side: int) -> int:
        return max(self._feather_min, int(round(self._feather_frac * bbox_short_side)))

    # ---- dispatch -----------------------------------------------------------

    def apply(
        self,
        frame: np.ndarray,
        mask_or_bbox: np.ndarray | Sequence[float],
    ) -> None:
        if isinstance(mask_or_bbox, np.ndarray) and mask_or_bbox.ndim == 2:
            self._apply_mask(frame, mask_or_bbox)
        else:
            self._apply_bbox(frame, mask_or_bbox)

    # ---- mask path ----------------------------------------------------------

    def _apply_mask(self, frame: np.ndarray, mask: np.ndarray) -> None:
        ys, xs = np.where(mask)
        if ys.size == 0:
            return
        h, w = frame.shape[:2]
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        bh = max(1, y2 - y1)
        short = max(1, min(bh, x2 - x1))
        kernel = self._kernel_for_height(bh)
        dilate_px = self._dilate_for_short(short)
        feather_px = self._feather_for_short(short)

        pad = dilate_px + feather_px
        ry1 = max(0, y1 - pad); ry2 = min(h, y2 + pad)
        rx1 = max(0, x1 - pad); rx2 = min(w, x2 + pad)

        sub_mask = (mask[ry1:ry2, rx1:rx2].astype(np.uint8)) * 255
        sub_mask = self._dilate_and_feather(sub_mask, dilate_px, feather_px)
        self._blend(frame, ry1, ry2, rx1, rx2, sub_mask, kernel)

    # ---- bbox path ----------------------------------------------------------

    def _apply_bbox(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return
        bh = max(1, y2 - y1)
        short = max(1, min(bh, x2 - x1))
        kernel = self._kernel_for_height(bh)
        dilate_px = self._dilate_for_short(short)
        feather_px = self._feather_for_short(short)

        pad = dilate_px + feather_px
        ry1 = max(0, y1 - pad); ry2 = min(h, y2 + pad)
        rx1 = max(0, x1 - pad); rx2 = min(w, x2 + pad)
        sub_mask = np.zeros((ry2 - ry1, rx2 - rx1), dtype=np.uint8)
        sub_mask[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = 255
        sub_mask = self._dilate_and_feather(sub_mask, dilate_px, feather_px)
        self._blend(frame, ry1, ry2, rx1, rx2, sub_mask, kernel)

    # ---- shared core --------------------------------------------------------

    @staticmethod
    def _dilate_and_feather(mask8: np.ndarray, dilate_px: int, feather_px: int) -> np.ndarray:
        if dilate_px > 0:
            k = 2 * dilate_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask8 = cv2.dilate(mask8, kernel)
        if feather_px > 0:
            fk = 2 * feather_px + 1
            mask8 = cv2.GaussianBlur(mask8, (fk, fk), 0)
        return mask8

    @staticmethod
    def _blend(
        frame: np.ndarray,
        ry1: int, ry2: int, rx1: int, rx2: int,
        sub_mask: np.ndarray,
        kernel: int,
    ) -> None:
        region = frame[ry1:ry2, rx1:rx2]
        blurred = cv2.GaussianBlur(region, (kernel, kernel), 0)
        alpha = (sub_mask.astype(np.float32) / 255.0)[..., None]
        out = region.astype(np.float32) * (1.0 - alpha) + blurred.astype(np.float32) * alpha
        frame[ry1:ry2, rx1:rx2] = out.astype(frame.dtype)
