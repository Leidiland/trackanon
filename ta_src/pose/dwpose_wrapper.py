"""DWpose (rtmlib RTMW-DW-x-l) — the pipeline's sole pose backend.

Two roles: `run()` returns COCO-17 keypoints per detection for tracking and
the diffusion keypoint gates; `render()` returns the RGB OpenPose skeleton
(body + face-68 + hands-21×2) that ControlNet conditions on.

Backed by rtmlib (`Wholebody`): YOLOX-m HumanArt for person detection +
RTMW-DW-x-l (cocktail14) for 133-keypoint whole-body inference. Pure
onnxruntime — same runtime as InsightFace. Weights are auto-managed by
rtmlib's MMPose CDN cache on first use.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from rtmlib import Wholebody, draw_skeleton
from rtmlib.tools.pose_estimation.post_processings import convert_coco_to_openpose

_log = logging.getLogger(__name__)


def _fullframe_kps_to_crop(
    keypoints_full: np.ndarray,
    roi: tuple[int, int, int, int],
    out_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Map a (N, 3) full-frame keypoint array into the resized crop's pixel
    space (the inverse of extract_roi's crop+resize). Returns ((1, N, 2) xy,
    (1, N) scores) — the shape rtmlib's convert/draw expect."""
    x1, y1, x2, y2 = roi
    th, tw = out_hw
    sx = tw / max(1, (x2 - x1))
    sy = th / max(1, (y2 - y1))
    xy = keypoints_full[:, :2].astype(np.float64).copy()
    xy[:, 0] = (xy[:, 0] - x1) * sx
    xy[:, 1] = (xy[:, 1] - y1) * sy
    scores = keypoints_full[:, 2].astype(np.float64)
    return xy[None, ...], scores[None, ...]


# COCO-Wholebody 133-kpt layout: [0:23) body+feet, [23:91) face-68, [91:133) hands.
_BODY_FEET_KPTS = 23


def _gate_scores_by_mask(
    xy: np.ndarray,
    scores: np.ndarray,
    mask_crop: np.ndarray,
    th: int,
    tw: int,
    dilate_px: int,
) -> np.ndarray:
    """Zero the score of any keypoint projecting outside the (dilated) mask.
    Drops bones whose endpoint sits off the person's silhouette — the phantom
    legs RTMW hallucinates on seated/occluded subjects and any neighbour
    keypoint that fell inside the bbox but outside this object's mask."""
    m = np.asarray(mask_crop)
    if m.ndim == 3:
        m = m[..., 0]
    m = (m > 0).astype(np.uint8)
    if m.shape != (th, tw):
        m = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)
    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    mb = m.astype(bool)
    pts = xy[0]
    xi = np.round(pts[:, 0]).astype(int)
    yi = np.round(pts[:, 1]).astype(int)
    inb = (xi >= 0) & (xi < tw) & (yi >= 0) & (yi < th)
    inside = np.zeros(pts.shape[0], dtype=bool)
    inside[inb] = mb[yi[inb], xi[inb]]
    scores = scores.copy()
    scores[0, ~inside] = 0.0
    return scores


def render_skeleton_from_keypoints(
    keypoints_full: np.ndarray,
    roi: tuple[int, int, int, int],
    out_hw: tuple[int, int],
    kpt_thr: float = 0.3,
    draw_face_hands: bool = True,
    mask_crop: np.ndarray | None = None,
    mask_dilate_px: int = 0,
) -> np.ndarray:
    """Draw the OpenPose skeleton (body + face-68 + hands) for one person from
    provided full-frame keypoints, in the crop's pixel space. Mirrors render()
    but skips re-detection — used to condition ControlNet on the smoothed
    keypoints instead of a fresh per-crop detection.

    draw_face_hands=False zeroes the face-68 mesh and hand confidences before
    the draw — RTMW reports high confidence for hallucinated frontal face
    keypoints on back-of-head views, so drawing them makes ControlNet enforce a
    forward-facing face on a person turned away. Suppressing the dense mesh lets
    the body pose drive head orientation. Coarse body head points are untouched.

    mask_crop (the object's crop-space SAM3 mask) gates keypoints to the
    silhouette: any keypoint projecting outside the dilated mask is dropped
    before the draw, so off-silhouette hallucinations don't reach ControlNet."""
    th, tw = out_hw
    xy, scores = _fullframe_kps_to_crop(keypoints_full, roi, out_hw)
    if not draw_face_hands:
        scores = scores.copy()
        scores[:, _BODY_FEET_KPTS:] = 0.0
    if mask_crop is not None:
        scores = _gate_scores_by_mask(xy, scores, mask_crop, th, tw, mask_dilate_px)
    xy, scores = convert_coco_to_openpose(xy, scores)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    skeleton = draw_skeleton(
        canvas, xy, scores, openpose_skeleton=True, kpt_thr=kpt_thr,
    )
    return np.asarray(skeleton, dtype=np.uint8)


class DWposeWrapper:
    def __init__(
        self,
        device: str = "cuda",
        mode: str = "balanced",
        conf_threshold: float = 0.3,
    ):
        _log.info("Loading RTMW Wholebody (mode=%s, device=%s)", mode, device)
        self._conf_threshold = float(conf_threshold)
        # to_openpose=False keeps the raw 133-keypoint COCO-Wholebody output
        # available for infer()'s COCO-17 slice. render() applies the OpenPose
        # conversion itself before draw_skeleton, preserving the ControlNet
        # OpenPose skeleton layout.
        self._detector = Wholebody(
            mode=mode,
            to_openpose=False,
            backend="onnxruntime",
            device=device,
        )

    @classmethod
    def from_config(cls, cfg, device: str) -> "DWposeWrapper | None":
        if cfg is None:
            return None
        return cls(
            device=device,
            mode=str(cfg.get("mode", "balanced")),
            conf_threshold=float(cfg.get("conf_threshold", 0.3)),
        )

    def render(self, crop_rgb: np.ndarray) -> np.ndarray:
        """Detect whole-body keypoints on `crop_rgb` and return the RGB
        OpenPose skeleton image, sized to the input crop.
        """
        keypoints, scores = self._detector(crop_rgb)
        keypoints, scores = convert_coco_to_openpose(keypoints, scores)
        canvas = np.zeros_like(crop_rgb)
        skeleton = draw_skeleton(
            canvas, keypoints, scores,
            openpose_skeleton=True,
            kpt_thr=0.3,
        )
        skeleton = np.asarray(skeleton, dtype=np.uint8)
        h, w = crop_rgb.shape[:2]
        if skeleton.shape[:2] != (h, w):
            skeleton = cv2.resize(skeleton, (w, h), interpolation=cv2.INTER_LINEAR)
        return skeleton

    def render_from_keypoints(
        self,
        keypoints_full: np.ndarray,
        roi: tuple[int, int, int, int],
        out_hw: tuple[int, int],
        draw_face_hands: bool = True,
        mask_crop: np.ndarray | None = None,
        mask_dilate_px: int = 0,
    ) -> np.ndarray:
        """Skeleton image from provided full-frame keypoints (e.g. the smoothed
        run() output), instead of re-detecting on the crop."""
        return render_skeleton_from_keypoints(
            keypoints_full, roi, out_hw, kpt_thr=self._conf_threshold,
            draw_face_hands=draw_face_hands,
            mask_crop=mask_crop, mask_dilate_px=mask_dilate_px,
        )

    def run(self, frame_rgb: np.ndarray, detections: list[dict]) -> list[dict]:
        """Run RTMW on every upstream person detection and return COCO-17
        keypoints in full-frame pixel coordinates — the contract tracking
        and the diffusion keypoint gates consume.

        Args:
            frame_rgb:  (H, W, 3) uint8 RGB full frame.
            detections: list of dicts, each with key "bbox": [x1, y1, x2, y2].

        Returns:
            list of dicts (one per detection):
              {"keypoints": np.ndarray (17, 3) [x, y, conf], "score": float}
            Keypoints are in full-frame pixel coordinates.
            Entries with no valid box get {"keypoints": None, "score": 0.0}.
        """
        if not detections:
            return []

        fh, fw = frame_rgb.shape[:2]
        valid_bboxes: list[list[float]] = []
        valid_idx: list[int] = []
        for di, det in enumerate(detections):
            bbox = det.get("bbox") or det.get("box")
            if bbox is None:
                continue
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            x1 = max(0.0, min(x1, fw - 1))
            y1 = max(0.0, min(y1, fh - 1))
            x2 = max(0.0, min(x2, fw - 1))
            y2 = max(0.0, min(y2, fh - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            valid_bboxes.append([x1, y1, x2, y2])
            valid_idx.append(di)

        results: list[dict] = [{"keypoints": None, "score": 0.0} for _ in detections]
        if not valid_bboxes:
            return results

        # Skip the built-in YOLOX detector — use upstream bboxes directly.
        # RTMPose returns full-frame pixel coords via center+scale rescale.
        kps_all, scores_all = self._detector.pose_model(frame_rgb, bboxes=valid_bboxes)

        for i, vi in enumerate(valid_idx):
            kps17 = kps_all[i, :17]
            sc17 = scores_all[i, :17].copy()
            sc17[sc17 < self._conf_threshold] = 0.0
            out = np.empty((17, 3), dtype=np.float32)
            out[:, :2] = kps17
            out[:, 2] = sc17
            # Full COCO-Wholebody (body + face-68 + hands) with raw confidences —
            # tracking ignores it; the visualization draws it to show DWpose.
            full = np.empty((kps_all[i].shape[0], 3), dtype=np.float32)
            full[:, :2] = kps_all[i]
            full[:, 2] = scores_all[i]
            results[vi] = {
                "keypoints": out,
                "score": float(sc17.mean()),
                "keypoints_full": full,
            }
        return results
