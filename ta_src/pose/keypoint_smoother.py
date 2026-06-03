"""Temporal smoothing of per-frame DWpose keypoints (One Euro filter).

Runs in full-frame pixel coordinates, keyed per track, so a near-stationary
person stops jittering while a moving one keeps up without lag. Time is the
frame index, so cutoff frequencies are in cycles-per-frame (sub-Nyquist
< 0.5). Feeds the TPS warp / pose gate / visualization; the ControlNet
conditioning skeleton is rendered elsewhere and untouched.
"""
from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class _OneEuroBank:
    """Vectorized One Euro filter over an (N, 2) keypoint array."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._init: np.ndarray | None = None  # per-keypoint, seen-valid-once

    def filter(self, xy: np.ndarray, valid: np.ndarray, dt: float) -> np.ndarray:
        """Smooth the valid rows; hold the invalid ones at their last value
        (and leave their filter state untouched so an occluded/low-conf
        reading can never poison the running estimate)."""
        xy = xy.astype(np.float64)
        if self._x_prev is None:
            self._x_prev = xy.copy()
            self._dx_prev = np.zeros_like(xy)
            self._init = valid.copy()
            return xy.copy()

        dx = (xy - self._x_prev) / dt
        a_d = _alpha(self._d_cutoff, dt)
        edx = self._dx_prev + a_d * (dx - self._dx_prev)
        cutoff = self._min_cutoff + self._beta * np.abs(edx)
        a = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * cutoff)) / dt)
        x_hat = self._x_prev + a * (xy - self._x_prev)

        out = self._x_prev.copy()
        filt = valid & self._init           # valid & seen before → smooth
        out[filt] = x_hat[filt]
        self._x_prev[filt] = x_hat[filt]
        self._dx_prev[filt] = edx[filt]
        seed = valid & ~self._init          # first valid sighting → seed
        out[seed] = xy[seed]
        self._x_prev[seed] = xy[seed]
        self._dx_prev[seed] = 0.0
        self._init[seed] = True
        return out


class KeypointSmoother:
    def __init__(
        self,
        *,
        min_cutoff: float,
        beta: float,
        d_cutoff: float,
        conf_floor: float,
        reset_after_missing_frames: int,
    ):
        self._min_cutoff = float(min_cutoff)
        self._beta = float(beta)
        self._d_cutoff = float(d_cutoff)
        self._conf_floor = float(conf_floor)
        self._reset_gap = int(reset_after_missing_frames)
        self._banks: dict[tuple, _OneEuroBank] = {}
        self._last_frame: dict[tuple, int] = {}

    @classmethod
    def from_config(cls, cfg) -> "KeypointSmoother | None":
        """Build from the pose `smoothing:` block. None when absent or
        explicitly disabled (feature-flag on by default)."""
        if cfg is None or not bool(cfg.get("enabled", False)):
            return None
        return cls(
            min_cutoff=float(cfg.get("min_cutoff", 0.1)),
            beta=float(cfg.get("beta", 0.02)),
            d_cutoff=float(cfg.get("d_cutoff", 1.0)),
            conf_floor=float(cfg.get("conf_floor", 0.3)),
            reset_after_missing_frames=int(cfg.get("reset_after_missing_frames", 5)),
        )

    def smooth(self, track_key, keypoints: np.ndarray, frame_idx: int) -> np.ndarray:
        bank_key = (track_key, keypoints.shape[0])
        last = self._last_frame.get(bank_key)
        if last is not None and (frame_idx - last) > self._reset_gap:
            self._banks.pop(bank_key, None)  # re-entry → start fresh
        bank = self._banks.get(bank_key)
        if bank is None:
            bank = _OneEuroBank(self._min_cutoff, self._beta, self._d_cutoff)
            self._banks[bank_key] = bank
        dt = float(frame_idx - last) if last is not None else 1.0
        if dt <= 0:
            dt = 1.0
        self._last_frame[bank_key] = frame_idx

        valid = keypoints[:, 2] >= self._conf_floor
        out = keypoints.copy()
        out[:, :2] = bank.filter(keypoints[:, :2], valid, dt)
        return out

    def apply(self, results: list[dict], detections: list[dict], frame_idx: int) -> None:
        """Smooth `poser.run()` results in place, keyed per track. Confirmed
        tracks key on global_id; unmatched (gid=-1) key on sam3_obj_id so they
        don't share filter state. Entries with no keypoints are left as-is."""
        for res, det in zip(results, detections):
            gid = det.get("global_id", -1)
            key = gid if gid != -1 else ("obj", det.get("sam3_obj_id"))
            for field in ("keypoints", "keypoints_full"):
                kps = res.get(field)
                if kps is not None:
                    res[field] = self.smooth((key, field), kps, frame_idx)
