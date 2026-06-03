import numpy as np
import cv2

from ta_src.pipeline.frame_context import FrameContext

# Reference width for scale calculations (1080p baseline).
_REF_WIDTH = 1920


def _vis_scale(image: np.ndarray) -> tuple[float, int, int]:
    """Return (font_scale, text_thickness, rect_thickness) scaled to frame width."""
    factor = max(1.0, image.shape[1] / _REF_WIDTH)
    return 0.7 * factor, max(1, round(1.5 * factor)), max(1, round(2 * factor))


def _label_metrics(image: np.ndarray, label_scale: float = 1.0) -> tuple[float, int, int]:
    """(font_scale, text_thickness, box_line_width) scaled to the image's larger
    dimension AND label_scale. scale=1.0 reproduces the 1080p video baseline
    (thin: font 0.7, text 2 px, box 2 px); higher scales (e.g. 4.0 for hi-res
    figures) bump text and box together — one knob for both."""
    m = max(image.shape[:2])
    font = (m / 2800.0) * label_scale
    text_thickness = max(1, round(m / 900.0 * label_scale))
    # Box width: linear in label_scale so video stays thin while figures get
    # the thicker outline. Tuned to land on 2 px at 1080p/scale=1 and 5 px at
    # 1080p/scale=4 (the pre-change video and figure looks).
    line_width = max(1, round(m * (1.0 + label_scale) / 1920.0))
    return font, text_thickness, line_width


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


def _draw_detections(image: np.ndarray, detections: list[dict], label_scale: float = 1.0):
    fs, tt, rt = _label_metrics(image, label_scale)
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        score = det.get("score", 0.0)
        label = det.get("label", "")
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), rt)
        _put_text_with_bg(image, f"{label} {score:.2f}", (x1, max(y1 - 4, round(fs * 20))),
                          fs, tt, (0, 255, 0))


def _track_label(track: dict) -> str:
    """Track box caption: obj-id prefix, gid/name, and the face-match cosine
    when present (image mode). Video tracks carry no match_cos → no cosine."""
    gid = track.get("global_id", -1)
    obj_id = track.get("sam3_obj_id")
    name = track.get("name", "") or ""
    cos = track.get("match_cos")
    prefix = f"{obj_id:02d} - " if obj_id is not None else ""
    suffix = f" {cos:.2f}" if cos is not None else ""
    if gid == -1:
        core = "unmatched"
    elif name:
        core = f"G{gid} {name}"
    else:
        core = f"G{gid}"
    return f"{prefix}{core}{suffix}"


def _label_bottom_y(y2: int, fs: float, h: int) -> int:
    """Baseline y for a track's label below its box, clamped to stay a
    font-proportional margin off the image's bottom edge so the text isn't
    pinned to the very edge."""
    return min(y2 + round(fs * 22), h - round(fs * 14))


def _draw_tracks(image: np.ndarray, tracks: list[dict], label_scale: float = 1.0):
    fs, tt, rt = _label_metrics(image, label_scale)
    h = image.shape[0]
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t["bbox"])
        label = _track_label(t)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 128, 0), rt)
        _put_text_with_bg(image, label, (x1, _label_bottom_y(y2, fs, h)),
                          fs, tt, (255, 128, 0))


# COCO-Wholebody 133-kpt layout: [0:23) body+feet, [23:91) face-68,
# [91:112) left hand, [112:133) right hand. Dense face/hand clusters get a
# finer marker so they read as clusters instead of one blob (community DWpose
# look). RGB buffer — colors are authored in RGB, not BGR.
_WHOLEBODY_LEN = 133
_FACE_START, _FACE_END = 23, 91
_LHAND_START, _LHAND_END = 91, 112
_RHAND_START, _RHAND_END = 112, 133

# One color per section, each with green as a non-dominant channel so dots stay
# legible on top of the green SAM3 mask, and none reusing the orange track box.
_BODY_COLOR = (255, 0, 0)       # red — complementary to the green mask
_FACE_COLOR = (255, 255, 255)   # white — DWpose convention, high contrast
_LHAND_COLOR = (255, 0, 255)    # magenta
_RHAND_COLOR = (0, 80, 255)     # azure


# Limb edges. Body uses the COCO-17 skeleton (shared by the 17- and 133-kpt
# layouts); feet/hands only exist in the 133-kpt Wholebody set. Hand topology is
# the 21-point layout, listed relative to each hand's start index.
_BODY_EDGES = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
               (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
               (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
_FEET_EDGES = [(15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)]
_HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
               (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def _draw_limbs(image: np.ndarray, kps, wholebody: bool, thickness: int):
    """Connect confident, adjacent joints with colored limb lines (under the
    dots): body/feet red, left hand magenta, right hand azure. Face has no
    skeleton — its 68 points stay dotted."""
    n = len(kps)

    def seg(i: int, j: int, color):
        if i >= n or j >= n or kps[i][2] <= 0.3 or kps[j][2] <= 0.3:
            return
        p1 = (int(kps[i][0]), int(kps[i][1]))
        p2 = (int(kps[j][0]), int(kps[j][1]))
        cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)

    for i, j in _BODY_EDGES:
        seg(i, j, _BODY_COLOR)
    if wholebody:
        for i, j in _FEET_EDGES:
            seg(i, j, _BODY_COLOR)
        for i, j in _HAND_EDGES:
            seg(_LHAND_START + i, _LHAND_START + j, _LHAND_COLOR)
            seg(_RHAND_START + i, _RHAND_START + j, _RHAND_COLOR)


def _kp_style(i: int, wholebody: bool, body_radius: int, fine_radius: int):
    """(radius, RGB color) for keypoint index `i`."""
    if wholebody and _FACE_START <= i < _FACE_END:
        return fine_radius, _FACE_COLOR
    if wholebody and _LHAND_START <= i < _LHAND_END:
        return fine_radius, _LHAND_COLOR
    if wholebody and _RHAND_START <= i < _RHAND_END:
        return fine_radius, _RHAND_COLOR
    return body_radius, _BODY_COLOR


# OpenPose-style limb palette (RGB) — the multicolor DWpose skeleton look,
# one color per body edge / joint (cycled across _BODY_EDGES).
_OPENPOSE_PALETTE = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85), (255, 0, 0),
]


def _pose_scale(image: np.ndarray) -> tuple[int, int]:
    """(limb line width, joint radius) scaled to the image's larger dimension,
    so the skeleton stays proportionally visible instead of a fixed pixel size
    that looks tiny on high-resolution images."""
    m = max(image.shape[:2])
    return max(2, round(m / 350)), max(2, round(m / 300))


def draw_pose_overlay(image: np.ndarray, keypoints_list: list[dict]):
    """Draw a DWpose/OpenPose-style skeleton: multicolor body limbs + joints,
    white face-68 dots, per-hand finger skeletons — all sized to the image
    resolution. Limbs go under the joints. Prefers the full 133-kpt set."""
    line_w, joint_r = _pose_scale(image)
    fine_r = max(1, joint_r // 2)
    hand_w = max(1, line_w // 2)
    for person in keypoints_list:
        kps = person.get("keypoints_full")
        if kps is None:
            kps = person.get("keypoints")
        if kps is None:
            continue
        kps = np.asarray(kps)
        n = len(kps)
        wholebody = n >= _WHOLEBODY_LEN

        def vis(i: int) -> bool:
            return i < n and kps[i][2] > 0.3

        def pt(i: int) -> tuple[int, int]:
            return int(kps[i][0]), int(kps[i][1])

        for idx, (i, j) in enumerate(_BODY_EDGES):
            if vis(i) and vis(j):
                color = _OPENPOSE_PALETTE[idx % len(_OPENPOSE_PALETTE)]
                cv2.line(image, pt(i), pt(j), color, line_w, cv2.LINE_AA)
        if wholebody:
            for i, j in _FEET_EDGES:
                if vis(i) and vis(j):
                    cv2.line(image, pt(i), pt(j), _OPENPOSE_PALETTE[0], line_w, cv2.LINE_AA)
            for i, j in _HAND_EDGES:
                li, lj, ri, rj = _LHAND_START + i, _LHAND_START + j, _RHAND_START + i, _RHAND_START + j
                if vis(li) and vis(lj):
                    cv2.line(image, pt(li), pt(lj), _LHAND_COLOR, hand_w, cv2.LINE_AA)
                if vis(ri) and vis(rj):
                    cv2.line(image, pt(ri), pt(rj), _RHAND_COLOR, hand_w, cv2.LINE_AA)

        for i in range(n):
            if kps[i][2] <= 0.3:
                continue
            if wholebody and _FACE_START <= i < _FACE_END:
                r, color = fine_r, _FACE_COLOR
            elif wholebody and _LHAND_START <= i < _LHAND_END:
                r, color = fine_r, _LHAND_COLOR
            elif wholebody and _RHAND_START <= i < _RHAND_END:
                r, color = fine_r, _RHAND_COLOR
            else:
                r, color = joint_r, _OPENPOSE_PALETTE[i % len(_OPENPOSE_PALETTE)]
            cv2.circle(image, pt(i), r, color, -1)


def _draw_keypoints(image: np.ndarray, keypoints_list: list[dict]):
    _, _, rt = _vis_scale(image)
    body_radius = max(3, rt * 2)
    fine_radius = max(1, body_radius // 2)
    for person in keypoints_list:
        # Prefer the full DWpose set (body + face-68 + hands) when present;
        # fall back to the COCO-17 tracking slice.
        kps = person.get("keypoints_full")
        if kps is None:
            kps = person.get("keypoints")
        if kps is None:
            continue
        wholebody = len(kps) >= _WHOLEBODY_LEN
        _draw_limbs(image, kps, wholebody, max(1, rt))
        for i, (x, y, conf) in enumerate(kps):
            if conf <= 0.3:
                continue
            radius, color = _kp_style(i, wholebody, body_radius, fine_radius)
            cv2.circle(image, (int(x), int(y)), radius, color, -1)
