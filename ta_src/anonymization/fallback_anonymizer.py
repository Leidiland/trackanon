from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


class FallbackAnonymizer:
    def __init__(self, kernel_size: int = 51, sigma: float = 0.0):
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        self._kernel = (kernel_size, kernel_size)
        self._sigma = sigma

    def apply(
        self,
        frame: np.ndarray,
        mask_or_bbox: np.ndarray | Sequence[float],
    ) -> None:
        if isinstance(mask_or_bbox, np.ndarray) and mask_or_bbox.ndim == 2:
            self._apply_mask(frame, mask_or_bbox)
        else:
            self._apply_bbox(frame, mask_or_bbox)

    def _apply_mask(self, frame: np.ndarray, mask: np.ndarray) -> None:
        blurred = cv2.GaussianBlur(frame, self._kernel, self._sigma)
        m = mask.astype(bool)
        frame[m] = blurred[m]

    def _apply_bbox(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return
        region = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, self._kernel, self._sigma)
