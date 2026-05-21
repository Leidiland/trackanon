from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameContext:
    """Per-frame state mutated in-place by each pipeline stage."""

    frame: np.ndarray
    detections: list[dict] = field(default_factory=list)
    keypoints: list[dict] = field(default_factory=list)
    tracks: list[dict] = field(default_factory=list)
