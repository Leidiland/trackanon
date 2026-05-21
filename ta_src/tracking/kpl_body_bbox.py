"""Heuristic body-bbox expansion from a face bbox, used at KPL build time.

The runtime OSNet path crops persons tightly (sam3 mask bbox or rtdetr
person box). If we feed full reference images to OSNet at KPL build time,
embeddings encode the reference background — different distribution from
what the runtime sees, narrower inter-person margins. Expanding a detected
face bbox into a body-sized region gives OSNet a similar crop at both
build and runtime. The proportions below are standard anthropometric
ratios: head height ≈ 1/7 of body height, shoulder width ≈ 3× face width.
"""
from __future__ import annotations

from typing import Optional

# Body sizing as multiples of face size. Generous on the height side so a
# face near the top of the frame doesn't truncate the legs; generous on
# width so shoulders/torso aren't clipped.
_BODY_HEIGHT_FACTOR = 7.5
_BODY_WIDTH_FACTOR = 3.0


def expand_face_to_body_bbox(
    face_bbox: Optional[tuple[float, float, float, float]],
    *,
    img_h: int,
    img_w: int,
) -> tuple[int, int, int, int]:
    """Return an integer xyxy bbox approximating the person's body region
    given the detected face bbox. Clips to image bounds. Falls back to the
    whole image when the face bbox is None so KPL extraction still proceeds."""
    if face_bbox is None:
        return (0, 0, img_w, img_h)

    fx1, fy1, fx2, fy2 = (float(v) for v in face_bbox)
    fw, fh = max(1.0, fx2 - fx1), max(1.0, fy2 - fy1)
    face_cx = (fx1 + fx2) / 2

    body_w = fw * _BODY_WIDTH_FACTOR
    body_h = fh * _BODY_HEIGHT_FACTOR
    x1 = int(round(face_cx - body_w / 2))
    x2 = int(round(face_cx + body_w / 2))
    # Body extends downward from face top. Pad a small margin above to
    # keep the head intact when the detected face is tight.
    y1 = int(round(fy1 - fh * 0.2))
    y2 = int(round(fy1 + body_h))

    x1 = max(0, min(img_w, x1))
    x2 = max(0, min(img_w, x2))
    y1 = max(0, min(img_h, y1))
    y2 = max(0, min(img_h, y2))
    return (x1, y1, x2, y2)
