"""ViTPose-B inference via ONNX Runtime.

No mmpose dependency — uses the official ONNX export directly.
Input:  (N, 3, 256, 192) float32, ImageNet-normalised.
Output: (N, 17, 64, 48)  float32, heatmaps.
Returns COCO-17 keypoints in full-frame pixel coordinates.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ta_src.pose.utils import (
    bbox_to_center_dims,
    get_affine_transform,
    heatmaps_to_keypoints,
)

_log = logging.getLogger(__name__)

# ImageNet normalisation (same as OSNet)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ViTPose-B heatmap stride = 4  (192→48, 256→64)
_HEATMAP_W = 48
_HEATMAP_H = 64


class ViTPoseWrapper:
    """Run ViTPose-B on person crops and return COCO-17 keypoints.

    Args:
        checkpoint:     Path to the ONNX model file.
        device:         "cuda" or "cpu".
        input_w/h:      Model crop size (192×256 for ViTPose-B).
        padding:        Bbox expansion factor before cropping (1.25 = +25 %).
        conf_threshold: Heatmap peak threshold; joints below this get conf=0.
        use_dark:       Apply DARK sub-pixel refinement (recommended).
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        input_w: int = 192,
        input_h: int = 256,
        padding: float = 1.25,
        conf_threshold: float = 0.3,
        use_dark: bool = True,
    ):
        self._input_w = input_w
        self._input_h = input_h
        self._padding = padding
        self._conf_threshold = conf_threshold
        self._use_dark = use_dark

        import onnxruntime as ort

        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Suppress verbose ONNX Runtime logs
        sess_opts.log_severity_level = 3

        self._session = ort.InferenceSession(
            checkpoint,
            sess_options=sess_opts,
            providers=providers,
        )
        self._input_name = self._session.get_inputs()[0].name

        # Confirm the model accepted our preferred provider
        active = self._session.get_providers()[0]
        _log.info(
            "ViTPose loaded (%s, input %d×%d, provider=%s)",
            checkpoint, input_w, input_h, active,
        )

    @classmethod
    def from_config(cls, cfg, device: str) -> "ViTPoseWrapper | None":
        if cfg is None:
            return None
        checkpoint = Path(cfg.checkpoint)
        if not checkpoint.exists():
            _log.warning(
                "ViTPose checkpoint not found at %s — pose stage disabled. "
                "See configs/pose/vitpose.yaml for download instructions.",
                checkpoint,
            )
            return None
        return cls(
            checkpoint=str(checkpoint),
            device=device,
            input_w=int(cfg.get("input_w", 192)),
            input_h=int(cfg.get("input_h", 256)),
            padding=float(cfg.get("padding", 1.25)),
            conf_threshold=float(cfg.get("conf_threshold", 0.3)),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        frame_rgb: np.ndarray,
        detections: list[dict],
    ) -> list[dict]:
        """Run ViTPose on every person detection.

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

        batch, inv_transforms, valid_idx = self._preprocess(frame_rgb, detections)

        if batch is None or len(valid_idx) == 0:
            return [{"keypoints": None, "score": 0.0} for _ in detections]

        heatmaps = self._session.run(None, {self._input_name: batch})[0]  # (N, 17, 64, 48)

        keypoints_list = heatmaps_to_keypoints(
            heatmaps,
            inv_transforms,
            heatmap_w=_HEATMAP_W,
            heatmap_h=_HEATMAP_H,
            input_w=self._input_w,
            input_h=self._input_h,
            use_dark=self._use_dark,
        )

        # Gate low-confidence joints
        for kps in keypoints_list:
            kps[kps[:, 2] < self._conf_threshold, 2] = 0.0

        results: list[dict] = [{"keypoints": None, "score": 0.0} for _ in detections]
        for i, vi in enumerate(valid_idx):
            kps = keypoints_list[i]
            score = float(kps[:, 2].mean())
            results[vi] = {"keypoints": kps, "score": score}

        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _preprocess(
        self,
        frame_rgb: np.ndarray,
        detections: list[dict],
    ) -> tuple[np.ndarray | None, list[np.ndarray], list[int]]:
        """Crop and normalise each person bbox into a model-input tensor.

        Returns:
            batch:          (N_valid, 3, H, W) float32 or None if no valid boxes
            inv_transforms: list of N_valid 2×3 float64 affine matrices
            valid_idx:      indices into `detections` that were valid
        """
        crops: list[np.ndarray] = []
        inv_transforms: list[np.ndarray] = []
        valid_idx: list[int] = []
        fh, fw = frame_rgb.shape[:2]

        for di, det in enumerate(detections):
            bbox = det.get("bbox") or det.get("box")
            if bbox is None:
                continue
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            # Clamp to frame
            x1 = max(0.0, min(x1, fw - 1))
            y1 = max(0.0, min(y1, fh - 1))
            x2 = max(0.0, min(x2, fw - 1))
            y2 = max(0.0, min(y2, fh - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            center, crop_w, crop_h = bbox_to_center_dims(
                x1, y1, x2, y2,
                self._input_w, self._input_h,
                self._padding,
            )
            fwd = get_affine_transform(center, crop_w, crop_h, self._input_w, self._input_h)
            inv = get_affine_transform(center, crop_w, crop_h, self._input_w, self._input_h, inv=True)

            crop = cv2.warpAffine(
                frame_rgb,
                fwd,
                (self._input_w, self._input_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(128, 128, 128),
            )

            # Normalise: uint8 RGB → float32, ImageNet stats
            crop_f = crop.astype(np.float32) / 255.0
            crop_f = (crop_f - _MEAN) / _STD        # (H, W, 3)
            crops.append(crop_f.transpose(2, 0, 1))  # (3, H, W)
            inv_transforms.append(inv)
            valid_idx.append(di)

        if not crops:
            return None, [], []

        batch = np.stack(crops, axis=0).astype(np.float32)   # (N, 3, H, W)
        return batch, inv_transforms, valid_idx
