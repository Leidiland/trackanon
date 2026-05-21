import numpy as np
import cv2

from ta_src.pipeline.frame_context import FrameContext

# Reference width for scale calculations (1080p baseline).
_REF_WIDTH = 1920


def _vis_scale(image: np.ndarray) -> tuple[float, int, int]:
    """Return (font_scale, text_thickness, rect_thickness) scaled to frame width."""
    factor = max(1.0, image.shape[1] / _REF_WIDTH)
    return 0.7 * factor, max(1, round(1.5 * factor)), max(1, round(2 * factor))


def _put_text_with_bg(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    thickness: int,
    color: tuple[int, int, int],
):
    """Draw text with a semi-transparent dark background for legibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    pad = max(2, round(3 * font_scale))
    bg_pt1 = (x - pad, y - th - pad)
    bg_pt2 = (x + tw + pad, y + baseline + pad)
    # Clamp background rect to image bounds
    h, w = image.shape[:2]
    bg_pt1 = (max(0, bg_pt1[0]), max(0, bg_pt1[1]))
    bg_pt2 = (min(w - 1, bg_pt2[0]), min(h - 1, bg_pt2[1]))
    roi = image[bg_pt1[1]:bg_pt2[1], bg_pt1[0]:bg_pt2[0]]
    if roi.size:
        dark = np.zeros_like(roi)
        image[bg_pt1[1]:bg_pt2[1], bg_pt1[0]:bg_pt2[0]] = cv2.addWeighted(roi, 0.4, dark, 0.6, 0)
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def overlay_all(image_rgb: np.ndarray, ctx: FrameContext) -> np.ndarray:
    """Draw detections, tracks, masks, and keypoints onto a copy of image_rgb."""
    vis = image_rgb.copy()
    _draw_masks(vis, ctx.detections)
    _draw_detections(vis, ctx.detections)
    _draw_tracks(vis, ctx.tracks)
    _draw_keypoints(vis, ctx.keypoints)
    return vis


def _draw_masks(image: np.ndarray, detections: list[dict]):
    overlay = image.copy()
    for det in detections:
        mask = det.get("mask")
        if mask is None:
            continue
        overlay[mask] = (overlay[mask] * 0.5 + np.array([0, 200, 100], dtype=np.uint8) * 0.5).astype(np.uint8)
    np.copyto(image, overlay)


def _draw_detections(image: np.ndarray, detections: list[dict]):
    fs, tt, rt = _vis_scale(image)
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        score = det.get("score", 0.0)
        label = det.get("label", "")
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), rt)
        _put_text_with_bg(image, f"{label} {score:.2f}", (x1, max(y1 - 4, round(fs * 20))),
                          fs, tt, (0, 255, 0))


def _draw_tracks(image: np.ndarray, tracks: list[dict]):
    fs, tt, rt = _vis_scale(image)
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t["bbox"])
        gid = t.get("global_id", -1)
        obj_id = t.get("sam3_obj_id")
        name = t.get("name", "") or ""
        prefix = f"{obj_id:02d} - " if obj_id is not None else ""
        if gid == -1:
            label = f"{prefix}unmatched"
        elif name:
            label = f"{prefix}G{gid} {name}"
        else:
            label = f"{prefix}G{gid}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 128, 0), rt)
        h = image.shape[0]
        _put_text_with_bg(image, label, (x1, min(y2 + round(fs * 22), h - 4)),
                          fs, tt, (255, 128, 0))


def _draw_keypoints(image: np.ndarray, keypoints_list: list[dict]):
    _, _, rt = _vis_scale(image)
    radius = max(3, rt * 2)
    for person in keypoints_list:
        kps = person.get("keypoints")
        if kps is None:
            continue
        for x, y, conf in kps:
            if conf > 0.3:
                cv2.circle(image, (int(x), int(y)), radius, (0, 0, 255), -1)
