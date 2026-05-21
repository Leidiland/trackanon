from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RegenReason(Enum):
    NO_CACHE = "no_cache"
    INTERVAL_CEILING = "interval_ceiling"
    IOU_DROP = "iou_drop"
    REUSE = "reuse"


@dataclass(frozen=True)
class RegenDecision:
    regenerate: bool
    reason: RegenReason


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


class RegenerationGate:
    def __init__(self, max_interval: int, motion_iou_threshold: float):
        self._max_interval = int(max_interval)
        self._motion_iou_threshold = float(motion_iou_threshold)
        self._last_bbox: dict[int, tuple] = {}
        self._last_frame: dict[int, int] = {}

    def evaluate(self, gid: int, bbox, frame_idx: int) -> RegenDecision:
        if gid not in self._last_bbox:
            return RegenDecision(True, RegenReason.NO_CACHE)
        if frame_idx - self._last_frame[gid] >= self._max_interval:
            return RegenDecision(True, RegenReason.INTERVAL_CEILING)
        if _bbox_iou(bbox, self._last_bbox[gid]) < self._motion_iou_threshold:
            return RegenDecision(True, RegenReason.IOU_DROP)
        return RegenDecision(False, RegenReason.REUSE)

    def record_generation(self, gid: int, bbox, frame_idx: int) -> None:
        self._last_bbox[gid] = tuple(bbox)
        self._last_frame[gid] = int(frame_idx)

    def reset(self) -> None:
        self._last_bbox.clear()
        self._last_frame.clear()
