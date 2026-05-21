"""On-demand face-mask preparation for the face-polish pass.

Pure function: takes the current frame + body bbox + the person mask
(already in inpaint-crop coords) + the inpaint-crop bounds + a face-polish
config, returns a face mask in crop coords intersected with the person
mask, or a status explaining why no mask was produced. No caching, no
pipeline state.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np


class FaceMaskStatus(Enum):
    OK = "ok"
    NO_FACE = "no_face"
    SUB_RESOLUTION = "sub_resolution"


@dataclass(frozen=True)
class FaceMaskResult:
    status: FaceMaskStatus
    mask: np.ndarray | None
    bbox_in_crop: tuple[int, int, int, int] | None


class _FaceDetector(Protocol):
    def detect_face_bbox(
        self, image: np.ndarray
    ) -> tuple[float, float, float, float] | None: ...


def build_face_mask(
    *,
    face_id_wrapper: _FaceDetector,
    frame_rgb: np.ndarray,
    body_bbox: tuple[float, float, float, float],
    person_mask_in_crop: np.ndarray,
    crop_bounds: tuple[int, int, int, int],
    pad_ratio: float,
    min_face_size_px: int,
) -> FaceMaskResult:
    bx1, by1, bx2, by2 = (int(round(v)) for v in body_bbox)
    fh, fw = frame_rgb.shape[:2]
    bx1 = max(0, min(bx1, fw))
    by1 = max(0, min(by1, fh))
    bx2 = max(0, min(bx2, fw))
    by2 = max(0, min(by2, fh))
    body_crop = frame_rgb[by1:by2, bx1:bx2]

    face_bbox_body = face_id_wrapper.detect_face_bbox(body_crop)
    if face_bbox_body is None:
        return FaceMaskResult(status=FaceMaskStatus.NO_FACE, mask=None, bbox_in_crop=None)

    # Frame coords
    fx1 = bx1 + face_bbox_body[0]
    fy1 = by1 + face_bbox_body[1]
    fx2 = bx1 + face_bbox_body[2]
    fy2 = by1 + face_bbox_body[3]

    # Pad
    fw_bb = fx2 - fx1
    fh_bb = fy2 - fy1
    pad_x = fw_bb * pad_ratio
    pad_y = fh_bb * pad_ratio
    fx1 -= pad_x
    fy1 -= pad_y
    fx2 += pad_x
    fy2 += pad_y

    # Crop coords
    cx1_b, cy1_b, _, _ = crop_bounds
    crop_h, crop_w = person_mask_in_crop.shape[:2]
    ix1 = int(round(fx1 - cx1_b))
    iy1 = int(round(fy1 - cy1_b))
    ix2 = int(round(fx2 - cx1_b))
    iy2 = int(round(fy2 - cy1_b))
    ix1 = max(0, min(ix1, crop_w))
    iy1 = max(0, min(iy1, crop_h))
    ix2 = max(0, min(ix2, crop_w))
    iy2 = max(0, min(iy2, crop_h))

    if min(ix2 - ix1, iy2 - iy1) < min_face_size_px:
        return FaceMaskResult(status=FaceMaskStatus.SUB_RESOLUTION, mask=None, bbox_in_crop=None)

    mask = np.zeros_like(person_mask_in_crop, dtype=bool)
    mask[iy1:iy2, ix1:ix2] = True
    mask &= person_mask_in_crop.astype(bool)

    return FaceMaskResult(
        status=FaceMaskStatus.OK,
        mask=mask,
        bbox_in_crop=(ix1, iy1, ix2, iy2),
    )
