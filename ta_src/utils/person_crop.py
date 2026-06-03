"""Person-centered uniform crop — normalize differing images to one canvas.

Scale each image so the person occupies a fixed fraction of the output height,
then crop/pad a fixed canvas centred on the person. The person bbox is supplied
by the caller; detection runs upstream.
"""
from __future__ import annotations

import cv2
import numpy as np


def center_on_person(
    image_rgb: np.ndarray,
    bbox,
    *,
    out_w: int,
    out_h: int,
    person_height_frac: float = 0.8,
    pad_value: int = 255,
) -> np.ndarray:
    """Return an out_h×out_w canvas with the person scaled so its bbox height is
    `person_height_frac` of the canvas height and its centre at the canvas
    centre. Regions falling outside the source are filled with `pad_value`."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    person_h = max(1.0, y2 - y1)
    scale = (person_height_frac * out_h) / person_h

    src_h, src_w = image_rgb.shape[:2]
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    scaled = cv2.resize(image_rgb, (new_w, new_h), interpolation=interp)

    cx = (x1 + x2) / 2.0 * scale
    cy = (y1 + y2) / 2.0 * scale
    left = int(round(cx - out_w / 2.0))
    top = int(round(cy - out_h / 2.0))

    canvas = np.full((out_h, out_w, 3), pad_value, dtype=image_rgb.dtype)
    sx0, sy0 = max(0, left), max(0, top)
    sx1, sy1 = min(new_w, left + out_w), min(new_h, top + out_h)
    if sx1 > sx0 and sy1 > sy0:
        dx0, dy0 = sx0 - left, sy0 - top
        canvas[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = scaled[sy0:sy1, sx0:sx1]
    return canvas
