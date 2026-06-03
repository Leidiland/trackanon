"""Matte the generated persona out of VACE's hallucinated background (the
pose_gen generate-then-composite path). There the masked region carries a pose
skeleton, not grey scene, and the companion mask is all-white, so VACE invents a
full background; rembg's person matting recovers just the persona, which the
stitch intersects with the original SAM3 silhouette (the persona stays bounded by
the scene-correct silhouette; the residual inherits the fallback blur)."""
from __future__ import annotations

import cv2
import numpy as np

_session = None


def _get_session():
    # u2net_human_seg: rembg's person-tuned matting model. Cached process-wide —
    # one model load per worker, not per frame.
    global _session
    if _session is None:
        from rembg import new_session
        _session = new_session("u2net_human_seg")
    return _session


def persona_matte_canvas(canvas_rgb, *, thr: int = 128, dilate_px: int = 0) -> np.ndarray:
    """Boolean persona matte (H,W) for a rendered canvas — True where rembg
    judges persona, False on the hallucinated background. `dilate_px` grows the
    matte outward: rembg cuts conservatively at the persona edge, so a dilation
    lets the matte∩silhouette paste fill the silhouette completely instead of
    leaving a blurred fallback rim where the matte undercut it."""
    from rembg import remove
    alpha = remove(np.asarray(canvas_rgb), session=_get_session(), only_mask=True)
    alpha = np.asarray(alpha)
    if alpha.ndim == 3:
        alpha = alpha[..., 0]
    m = alpha >= thr
    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        m = cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    return m


def fill_nonpersona(canvas_rgb, persona_mask, *, radius: int = 3) -> np.ndarray:
    """Replace every non-persona pixel with its NEAREST persona pixel's colour
    (Voronoi edge-extension), so a full-silhouette paste of the result contains
    ONLY persona-derived colour — never the generated background. The persona's
    free-form shape rarely matches the silhouette exactly; this extends the
    persona edge cleanly into the gap (no Telea radial smear, no bg, no blur)."""
    persona_mask = np.asarray(persona_mask).astype(bool)
    canvas = np.ascontiguousarray(np.asarray(canvas_rgb, dtype=np.uint8))
    if not persona_mask.any() or persona_mask.all():
        return canvas
    # distanceTransformWithLabels: every non-persona pixel gets the label of the
    # nearest persona pixel; map label -> persona coord -> colour.
    nonpersona = (~persona_mask).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        nonpersona, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL,
    )
    ys, xs = np.where(persona_mask)
    plabels = labels[ys, xs]
    lut_y = np.zeros(int(labels.max()) + 1, np.int64)
    lut_x = np.zeros(int(labels.max()) + 1, np.int64)
    lut_y[plabels] = ys
    lut_x[plabels] = xs
    out = canvas.copy()
    fill = ~persona_mask
    lbl = labels[fill]
    out[fill] = canvas[lut_y[lbl], lut_x[lbl]]
    return out


def canvas_mask_to_frame(canvas_mask, crop_box, frame_hw) -> np.ndarray:
    """Place a canvas-space (H,W) bool mask into a full-frame bool mask at
    crop_box (mirrors compose_frame's resize+placement), so it can be ANDed with
    the full-frame SAM3 silhouette."""
    canvas_mask = np.asarray(canvas_mask).astype(np.uint8)
    ix0, iy0 = int(np.floor(crop_box[0])), int(np.floor(crop_box[1]))
    ix1, iy1 = int(np.ceil(crop_box[2])), int(np.ceil(crop_box[3]))
    resized = cv2.resize(canvas_mask, (ix1 - ix0, iy1 - iy0), interpolation=cv2.INTER_NEAREST)
    fh, fw = frame_hw
    out = np.zeros((fh, fw), bool)
    cx0, cy0 = max(0, ix0), max(0, iy0)
    cx1, cy1 = min(fw, ix1), min(fh, iy1)
    out[cy0:cy1, cx0:cx1] = resized[cy0 - iy0:cy0 - iy0 + (cy1 - cy0),
                                    cx0 - ix0:cx0 - ix0 + (cx1 - cx0)] > 0
    return out
