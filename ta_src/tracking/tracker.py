from __future__ import annotations

import numpy as np


class KalmanBox:
    """Constant-velocity Kalman filter for a single bounding box.

    State: [cx, cy, w, h, vx, vy, vw, vh]
    Obs:   [cx, cy, w, h]
    """
    _F = np.eye(8, dtype=np.float64)
    for _i in range(4):
        _F[_i, _i + 4] = 1.0

    _H = np.eye(4, 8, dtype=np.float64)

    _Q = np.diag([1., 1., 1., 1., 0.01, 0.01, 0.0001, 0.0001])
    _R = np.diag([1., 1., 10., 10.])

    def __init__(self, box_xyxy):
        cx, cy, w, h = self._to_cxcywh(box_xyxy)
        self.x = np.array([cx, cy, w, h, 0., 0., 0., 0.], dtype=np.float64)
        self.P = np.diag([10., 10., 10., 10., 1e4, 1e4, 1e4, 1e4])

    @staticmethod
    def _to_cxcywh(box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1

    @staticmethod
    def _to_xyxy(cx, cy, w, h):
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    def predict(self):
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self._Q
        return self._to_xyxy(*self.x[:4])

    def update(self, box_xyxy):
        z = np.array(self._to_cxcywh(box_xyxy), dtype=np.float64)
        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + self._R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self._H) @ self.P

    def box(self):
        return self._to_xyxy(*self.x[:4])

    def peek_forward(self):
        """One-step-ahead prediction WITHOUT advancing state.

        Used by the Stage-1 spatial fast-path to gate detections against the
        Identity's expected position this frame, while leaving Kalman state
        for the subsequent `observe()` call (which runs predict+update).
        """
        x_next = self._F @ self.x
        return self._to_xyxy(*x_next[:4])

    def shift(self, dx: float, dy: float):
        self.x[0] += dx
        self.x[1] += dy
