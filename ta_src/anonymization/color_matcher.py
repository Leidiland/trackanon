"""LAB color match of a generated crop to the surrounding scene.

The match target is sampled from a ring of *real-frame* pixels around the
mask (`dilate(mask, ring_width_px) - mask`) — never from the mask interior,
which contains the real person's pixels. See ADR-0017.
"""
from __future__ import annotations

import cv2
import numpy as np


class ColorMatcher:
    def __init__(
        self,
        *,
        ring_width_px: int = 16,
        match_std: bool = False,
    ):
        self.ring_width_px = int(ring_width_px)
        self.match_std = bool(match_std)

    def compute_ring(self, mask: np.ndarray) -> np.ndarray:
        """Return the ring of scene pixels around the mask: dilate(mask) − mask."""
        mask_bool = mask.astype(bool)
        k = 2 * self.ring_width_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(mask_bool.astype(np.uint8), kernel).astype(bool)
        return dilated & ~mask_bool

    def match(
        self,
        gen_rgb: np.ndarray,
        frame_rgb: np.ndarray,
        mask: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        alpha = float(alpha)
        if alpha == 0.0:
            return gen_rgb.copy()

        mask_bool = mask.astype(bool)
        if not mask_bool.any():
            return gen_rgb.copy()

        ring = self.compute_ring(mask_bool)
        if not ring.any():
            return gen_rgb.copy()

        gen_lab = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        frame_lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        gen_interior = gen_lab[mask_bool]
        target_pixels = frame_lab[ring]

        gen_mean = gen_interior.mean(axis=0)
        target_mean = target_pixels.mean(axis=0)

        if self.match_std:
            gen_std = gen_interior.std(axis=0) + 1e-6
            target_std = target_pixels.std(axis=0)
            scale = (1.0 - alpha) + alpha * (target_std / gen_std)
            shifted = (gen_interior - gen_mean) * scale + (
                gen_mean + alpha * (target_mean - gen_mean)
            )
        else:
            shifted = gen_interior + alpha * (target_mean - gen_mean)

        out_lab = gen_lab.copy()
        out_lab[mask_bool] = shifted
        out_lab = np.clip(out_lab, 0.0, 255.0).astype(np.uint8)
        rgb_round = cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)
        # Outside the mask, preserve the original gen_rgb bit-exact —
        # LAB↔RGB roundtrip introduces ±1 quantisation noise we don't want
        # to leak into pixels we're not actually adjusting.
        out_rgb = gen_rgb.copy()
        out_rgb[mask_bool] = rgb_round[mask_bool]
        return out_rgb

    @classmethod
    def from_config(cls, cfg) -> "ColorMatcher | None":
        if cfg is None:
            return None
        cm = cfg.get("color_match", None) if hasattr(cfg, "get") else None
        if cm is None:
            return None
        if not bool(cm.get("enabled", False)):
            return None
        return cls(
            ring_width_px=int(cm.get("ring_width_px", 16)),
            match_std=bool(cm.get("match_std", False)),
        )
