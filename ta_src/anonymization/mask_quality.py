"""Mask-quality dispatch gate — routes detections to diffusion or Fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import numpy as np


@dataclass(frozen=True)
class Pass:
    """Mask quality is sufficient to dispatch to diffusion."""


@dataclass(frozen=True)
class FailNone:
    """No mask available (SAM2 outright failure). Route to Fallback."""


@dataclass(frozen=True)
class FailRatio:
    ratio: float
    floor: float


@dataclass(frozen=True)
class FailScore:
    score: float
    floor: float


DispatchDecision = Union[Pass, FailNone, FailRatio, FailScore]


def mask_quality_check(
    mask: np.ndarray | None,
    mask_score: float | None,
    bbox: tuple[float, float, float, float],
    mask_source: Literal["track", "detection"],
    *,
    ratio_floor: float,
    score_floor: float,
    score_pass_override: float | None = None,
) -> DispatchDecision:
    """Decide whether ``mask`` is good enough to dispatch to diffusion.

    ``score_floor`` applies only when ``mask_source == "detection"``.
    ``ratio_floor`` applies in all non-None cases. ``mask is None`` → FailNone.

    ``score_pass_override`` (optional, detection-source only): if the mask
    score is at or above this value, a sub-``ratio_floor`` ratio is rescued
    to Pass. Defends crowded-scene occlusions where SAM 3 emits a high-
    confidence narrow mask covering only the visible portion of a person.

    ``bbox`` is the un-padded detection bbox ``(x1, y1, x2, y2)``. The mask is
    expected to be a full-frame 2D array (bool or uint8); ``mask_area`` is
    counted within the bbox region.
    """
    if mask is None:
        return FailNone()

    if mask_source == "detection" and mask_score is not None:
        if mask_score < score_floor:
            return FailScore(score=float(mask_score), floor=float(score_floor))

    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    bbox_area = max(0, (x2 - x1) * (y2 - y1))
    if bbox_area == 0:
        return FailRatio(ratio=0.0, floor=float(ratio_floor))
    mask_area = int(mask[y1:y2, x1:x2].astype(bool).sum())
    ratio = mask_area / bbox_area
    if ratio < ratio_floor:
        score_rescues = (
            score_pass_override is not None
            and mask_source == "detection"
            and mask_score is not None
            and mask_score >= score_pass_override
        )
        if not score_rescues:
            return FailRatio(ratio=ratio, floor=float(ratio_floor))

    return Pass()
