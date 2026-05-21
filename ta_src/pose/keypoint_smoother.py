"""1€ filter for ViTPose keypoints — per-`global_id` per-joint smoothing.

Velocity-adaptive cutoff: low velocity → low cutoff (smooth), high velocity →
high cutoff (responsive). Same per-frame cost as EMA. See ADR-0014.
"""
from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    """1st-order low-pass smoothing factor at the given cutoff frequency."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class _JointFilter:
    """Per-joint 1€ filter state."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None
        self.t_prev: int | None = None

    def step(self, xy: np.ndarray, t: int) -> np.ndarray:
        if self.t_prev is None:
            self.x_prev = xy.astype(np.float64).copy()
            self.dx_prev = np.zeros(2, dtype=np.float64)
            self.t_prev = int(t)
            return xy.astype(np.float32)
        dt = max(1, int(t) - int(self.t_prev))
        raw_dx = (xy.astype(np.float64) - self.x_prev) / float(dt)
        a_d = _alpha(self.d_cutoff, float(dt))
        smoothed_dx = a_d * raw_dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(smoothed_dx))
        a = _alpha(cutoff, float(dt))
        smoothed_x = a * xy.astype(np.float64) + (1.0 - a) * self.x_prev
        self.x_prev = smoothed_x
        self.dx_prev = smoothed_dx
        self.t_prev = int(t)
        return smoothed_x.astype(np.float32)

    def predict(self) -> np.ndarray | None:
        if self.x_prev is None:
            return None
        return self.x_prev.astype(np.float32)


class KeypointSmoother:
    def __init__(
        self,
        *,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        conf_floor: float = 0.3,
        gap_reset_frames: int = 30,
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.conf_floor = float(conf_floor)
        self.gap_reset_frames = int(gap_reset_frames)
        self._states: dict[tuple[int, int], _JointFilter] = {}
        self._last_frame: dict[int, int] = {}

    def smooth(
        self,
        global_id: int,
        keypoints: np.ndarray,
        frame_idx: int,
    ) -> np.ndarray:
        out = np.asarray(keypoints, dtype=np.float32).copy()
        if global_id in self._last_frame:
            gap = int(frame_idx) - int(self._last_frame[global_id])
            if gap > self.gap_reset_frames:
                self.reset(global_id)
        self._last_frame[global_id] = int(frame_idx)
        for k in range(out.shape[0]):
            conf = float(out[k, 2])
            key = (int(global_id), int(k))
            state = self._states.get(key)
            if state is None:
                state = _JointFilter(self.min_cutoff, self.beta, self.d_cutoff)
                self._states[key] = state
            if conf >= self.conf_floor:
                xy = np.array([out[k, 0], out[k, 1]], dtype=np.float64)
                smoothed = state.step(xy, int(frame_idx))
                out[k, 0] = float(smoothed[0])
                out[k, 1] = float(smoothed[1])
            else:
                predicted = state.predict()
                if predicted is not None:
                    out[k, 0] = float(predicted[0])
                    out[k, 1] = float(predicted[1])
        return out

    def reset(self, global_id: int) -> None:
        gid = int(global_id)
        self._states = {k: v for k, v in self._states.items() if k[0] != gid}
        self._last_frame.pop(gid, None)

    def reset_all(self) -> None:
        self._states.clear()
        self._last_frame.clear()

    @classmethod
    def from_config(cls, cfg) -> "KeypointSmoother | None":
        if cfg is None:
            return None
        sm = cfg.get("smoother", None) if hasattr(cfg, "get") else None
        if sm is None:
            return None
        if not bool(sm.get("enabled", False)):
            return None
        return cls(
            min_cutoff=float(sm.get("min_cutoff", 1.0)),
            beta=float(sm.get("beta", 0.007)),
            d_cutoff=float(sm.get("d_cutoff", 1.0)),
            conf_floor=float(sm.get("conf_floor", 0.3)),
            gap_reset_frames=int(sm.get("gap_reset_frames", 30)),
        )
