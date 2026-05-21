"""OpenPose-style skeleton rendering for ControlNet pose conditioning.

Converts COCO-17 keypoints from the pose stage into an RGB skeleton image
that ControlNet OpenPose expects as its conditioning input.

Graceful degradation: returns a black image (no conditioning) when
keypoints are absent or all confidence scores fall below the threshold.
ControlNet sees a blank canvas and effectively ignores pose conditioning,
falling back to prompt-only inpainting.

Note: the COCO-17 layout and index-keyed limb colours are out-of-distribution
relative to the OpenPose ControlNet training data (BODY-25, anatomical colour
mapping). An A/B against the canonical controlnet_aux OpenposeDetector showed
ControlNet absorbs the deviation at current workflow settings — the renderer
is OOD-but-inert in practice.
"""
from __future__ import annotations

import cv2
import numpy as np

# COCO-17 skeleton: (joint_a, joint_b) index pairs.
# Joints: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
#         5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
#         9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
#         13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12),   # ankles → knees → hips
    (11, 12),                                   # hip bar
    (5, 11),  (6, 12),                          # shoulders → hips
    (5, 6),                                     # shoulder bar
    (5, 7),   (6, 8),                           # shoulders → elbows
    (7, 9),   (8, 10),                          # elbows → wrists
    (1, 2),                                     # eye bar
    (0, 1),   (0, 2),                           # nose → eyes
    (1, 3),   (2, 4),                           # eyes → ears
    (3, 5),   (4, 6),                           # ears → shoulders
]

# Distinct RGB colour per limb (matching the standard OpenPose palette).
_LIMB_COLORS: list[tuple[int, int, int]] = [
    (255,   0,   0), (255,  85,   0), (255, 170,   0), (255, 255,   0),
    (170, 255,   0), ( 85, 255,   0), (  0, 255,   0), (  0, 255,  85),
    (  0, 255, 170), (  0, 255, 255), (  0, 170, 255), (  0,  85, 255),
    (  0,   0, 255), ( 85,   0, 255), (170,   0, 255), (255,   0, 255),
    (255,   0, 170), (255,   0,  85),
]


def render_skeleton(
    keypoints: np.ndarray | None,
    crop_x1: int,
    crop_y1: int,
    crop_w: int,
    crop_h: int,
    target_w: int,
    target_h: int,
    conf_threshold: float = 0.3,
    joint_radius: int = 4,
    limb_thickness: int = 3,
) -> np.ndarray:
    """Render COCO-17 keypoints as an OpenPose skeleton image.

    Args:
        keypoints:        (K, 3) array of [x, y, confidence] in full-frame
                          pixel coordinates, or None.
        crop_x1, crop_y1: top-left corner of the ROI crop in the full frame.
        crop_w, crop_h:   size of the ROI crop before diffusion resizing.
        target_w, target_h: output image dimensions (= diffusion input size).
        conf_threshold:   joints below this confidence are treated as absent.
        joint_radius:     circle radius for joint visualisation (in target px).
        limb_thickness:   line thickness for limb visualisation (in target px).

    Returns:
        (target_h, target_w, 3) uint8 RGB skeleton image.
        All-black when keypoints is None or no joint exceeds conf_threshold.
    """
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    if keypoints is None or len(keypoints) == 0:
        return canvas

    kps = np.asarray(keypoints, dtype=np.float32)  # (K, 3)
    n_joints = kps.shape[0]

    # Scale factors: full-frame → crop-local → target-size
    sx = target_w / max(crop_w, 1)
    sy = target_h / max(crop_h, 1)

    def _project(j: int) -> tuple[int, int] | None:
        if j >= n_joints or kps[j, 2] < conf_threshold:
            return None
        x = int((kps[j, 0] - crop_x1) * sx)
        y = int((kps[j, 1] - crop_y1) * sy)
        if 0 <= x < target_w and 0 <= y < target_h:
            return x, y
        return None

    # Draw limbs first so joints render on top.
    for idx, (ja, jb) in enumerate(_SKELETON):
        pa = _project(ja)
        pb = _project(jb)
        if pa is not None and pb is not None:
            color = _LIMB_COLORS[idx % len(_LIMB_COLORS)]
            cv2.line(canvas, pa, pb, color, limb_thickness, cv2.LINE_AA)

    # Draw joints.
    for j in range(n_joints):
        pt = _project(j)
        if pt is not None:
            cv2.circle(canvas, pt, joint_radius, (255, 255, 255), -1, cv2.LINE_AA)

    return canvas


def has_valid_keypoints(
    keypoints: np.ndarray | None, conf_threshold: float = 0.3, min_joints: int = 3
) -> bool:
    """Return True if enough joints are above the confidence threshold."""
    if keypoints is None or len(keypoints) == 0:
        return False
    kps = np.asarray(keypoints)
    return int((kps[:, 2] >= conf_threshold).sum()) >= min_joints


# Stable body landmarks used for crop-warp alignment. Wrists/ankles/elbows
# swing wildly even when the torso is still, so they're excluded.
_WARP_JOINTS: tuple[int, ...] = (0, 1, 2, 5, 6, 11, 12)  # nose, eyes, shoulders, hips


def face_orientation_factor(
    keypoints: np.ndarray | None,
    conf_threshold: float = 0.3,
    min_factor: float = 0.2,
    face_emb_present: bool | None = None,
) -> float:
    """Multiplier in ``[min_factor, 1.0]`` to scale IPAdapter face_strength.

    Reads COCO-17 face keypoints — nose (0), left/right eye (1, 2) — and
    softens the face conditioning when the person is in profile or facing
    away. The IPAdapter Reference Crop is a frontal portrait; injecting
    it at full weight forces a frontal face even when the original head
    is turned. The returned factor scales face_strength down on those
    frames so the ControlNet pose can drive head orientation instead.

    ``face_emb_present`` overrides the keypoint heuristic when set: COCO pose
    estimators report high keypoint confidence for hallucinated nose/eye
    positions on back-of-head views, so InsightFace's presence/absence signal
    is more reliable when available.
    """
    if face_emb_present is False:
        return float(min_factor)
    if keypoints is None or len(keypoints) < 3:
        return 1.0
    kps = np.asarray(keypoints, dtype=np.float32)
    nose_ok = bool(kps[0, 2] >= conf_threshold)
    eyes_visible = int(kps[1, 2] >= conf_threshold) + int(kps[2, 2] >= conf_threshold)
    if nose_ok and eyes_visible == 2:
        return 1.0
    if nose_ok and eyes_visible == 1:
        return 0.7
    if not nose_ok and eyes_visible == 2:
        return 0.5
    return float(min_factor)


def _project_to_crop(
    full_xy: np.ndarray,
    roi: tuple[int, int, int, int],
    crop_wh: tuple[int, int],
) -> np.ndarray:
    """Project full-frame ``(x, y)`` points into a cached crop's pixel space."""
    x1, y1, x2, y2 = roi
    cw, ch = crop_wh
    sx = cw / max(1, x2 - x1)
    sy = ch / max(1, y2 - y1)
    out = np.asarray(full_xy, dtype=np.float32).copy()
    out[:, 0] = (out[:, 0] - x1) * sx
    out[:, 1] = (out[:, 1] - y1) * sy
    return out


def warp_crop_to_pose(
    cached_crop: np.ndarray,
    prev_kps: np.ndarray | None,
    prev_roi: tuple[int, int, int, int] | None,
    cur_kps: np.ndarray | None,
    cur_roi: tuple[int, int, int, int],
    conf_threshold: float = 0.3,
    min_points: int = 3,
    max_abs_scale: float = 2.0,
) -> np.ndarray | None:
    """Affine-warp a regen-time crop so its body parts line up with the
    current frame's keypoints.

    Mitigates "ghosting" on the regen-gate skip path: cached crops are
    generated for a past frame's bbox/keypoints, and pasting them under
    the current mask/bbox stretches features by a few pixels per frame.

    Returns ``None`` when the warp can't be computed (missing keypoints,
    insufficient stable joints, degenerate transform) — caller falls back
    to the unwarped cached crop.
    """
    if cached_crop is None or prev_kps is None or prev_roi is None or cur_kps is None:
        return None
    H, W = cached_crop.shape[:2]
    p = np.asarray(prev_kps, dtype=np.float32)
    c = np.asarray(cur_kps, dtype=np.float32)
    if p.ndim != 2 or c.ndim != 2 or p.shape[0] == 0 or c.shape[0] == 0:
        return None

    valid: list[int] = []
    for j in _WARP_JOINTS:
        if j >= p.shape[0] or j >= c.shape[0]:
            continue
        if p[j, 2] >= conf_threshold and c[j, 2] >= conf_threshold:
            valid.append(j)
    if len(valid) < min_points:
        return None

    src = _project_to_crop(p[valid, :2], prev_roi, (W, H))
    dst = _project_to_crop(c[valid, :2], cur_roi, (W, H))
    M, _inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=8.0,
    )
    if M is None:
        return None
    scale = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
    if not (1.0 / max_abs_scale <= scale <= max_abs_scale):
        return None

    return cv2.warpAffine(
        cached_crop, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
