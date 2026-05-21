"""SAM 3 chunked video segmentation — caller drives session lifecycle explicitly.

Session flags: async_loading_frames, offload_video_to_cpu, offload_state_to_cpu
(GPU state ratchets ~16 MB/frame at 4K, so offload keeps a chunk under ~1 GB).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def apply_zero_frame_carry_forward(
    per_frame_rows: list[list[dict]],
    max_consecutive_carry: int = 3,
    ghost_score_factor: float = 0.9,
) -> list[list[dict]]:
    """Replace zero-detection mid-chunk frames with ghost rows from the prior
    real frame. Caps the carry to `max_consecutive_carry` consecutive ghosts so
    a genuine disappearance does not get masked forever. Ghost rows are tagged
    `is_carry_forward=True` and own a copy of the mask array.
    """
    out: list[list[dict]] = []
    last_real: list[dict] | None = None
    carry_count = 0
    for rows in per_frame_rows:
        if rows:
            out.append(rows)
            last_real = rows
            carry_count = 0
            continue
        if last_real is None or carry_count >= max_consecutive_carry:
            out.append([])
            continue
        ghosts = [_make_ghost_row(r, ghost_score_factor) for r in last_real]
        out.append(ghosts)
        carry_count += 1
    return out


def _make_ghost_row(src: dict, score_factor: float) -> dict:
    ghost = dict(src)
    mask = src.get("mask")
    if mask is not None:
        ghost["mask"] = mask.copy()
    if "bbox" in src:
        ghost["bbox"] = list(src["bbox"])
    src_score = float(src.get("score", 0.0))
    ghost["score"] = src_score * score_factor
    ghost["mask_score"] = float(src.get("mask_score", src_score)) * score_factor
    ghost["is_carry_forward"] = True
    return ghost


def scale_rows_to_frame(
    rows: list[dict], frame_hw: tuple[int, int]
) -> list[dict]:
    """Upscale mask + bbox in-place to frame resolution (SAM 3 outputs at lower res)."""
    if not rows:
        return rows
    mh, mw = rows[0]["mask"].shape[:2]
    fh, fw = frame_hw
    if (mh, mw) == (fh, fw):
        return rows
    sx, sy = fw / mw, fh / mh
    for r in rows:
        m = r["mask"]
        r["mask"] = cv2.resize(
            m.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        x1, y1, x2, y2 = r["bbox"]
        r["bbox"] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
    return rows


class SAM3ChunkedStage:
    def __init__(
        self,
        predictor,
        prompt: str = "person",
        zero_frame_max_carry: int = 3,
        zero_frame_ghost_score_factor: float = 0.9,
    ) -> None:
        self._predictor = predictor
        self._prompt = prompt
        self._zero_frame_max_carry = int(zero_frame_max_carry)
        self._zero_frame_ghost_score_factor = float(zero_frame_ghost_score_factor)
        self._session_id: str | None = None

    @classmethod
    def from_config(cls, cfg, device: str) -> "SAM3ChunkedStage":
        # Lazy import: production CLI may run without sam3 installed (e.g.
        # rtdetr_sam2 backend); only the sam3 path needs the dependency.
        from sam3.model_builder import build_sam3_video_predictor

        if device.startswith("cuda"):
            parts = device.split(":")
            gpu_idx = int(parts[1]) if len(parts) > 1 else 0
            gpus = [gpu_idx]
        else:
            raise ValueError(
                f"SAM 3 requires a CUDA device; got {device!r}. "
                "CPU inference is not supported."
            )

        predictor = build_sam3_video_predictor(gpus_to_use=gpus)
        get = getattr(cfg, "get", None)
        if get is None:
            get = lambda k, d=None: getattr(cfg, k, d)
        return cls(
            predictor=predictor,
            prompt=cfg.prompt,
            zero_frame_max_carry=int(get("zero_frame_max_carry", 3)),
            zero_frame_ghost_score_factor=float(
                get("zero_frame_ghost_score_factor", 0.9),
            ),
        )

    def process_chunk(
        self, chunk_dir: Path, base_frame_idx: int
    ) -> list[list[dict]]:
        start = self._predictor.handle_request({
            "type": "start_session",
            "resource_path": str(chunk_dir),
            "async_loading_frames": True,
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": True,
        })
        self._session_id = start["session_id"]
        self._predictor.handle_request({
            "type": "add_prompt",
            "session_id": self._session_id,
            "frame_index": 0,
            "text": self._prompt,
        })

        rows_per_frame: list[list[dict]] = []
        for response in self._predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": self._session_id,
        }):
            rows_per_frame.append(_rows_from_outputs(response["outputs"]))
        # Zero-detection frames mid-chunk get ghost rows from the prior frame
        # so bindings keep their sam3_obj_id keys and anonymization stays
        # applied. Cap prevents masking a genuine disappearance.
        return apply_zero_frame_carry_forward(
            rows_per_frame,
            max_consecutive_carry=self._zero_frame_max_carry,
            ghost_score_factor=self._zero_frame_ghost_score_factor,
        )

    def close_session_and_empty_cache(self) -> None:
        """Close the current SAM 3 session and release CUDA cache.

        Caller invokes after process_chunk so ComfyUI can claim the GPU
        for anonymisation. No-op if no session is open.
        """
        if self._session_id is None:
            return
        self._predictor.handle_request({
            "type": "close_session",
            "session_id": self._session_id,
        })
        self._session_id = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _rows_from_outputs(outputs: dict) -> list[dict]:
    rows: list[dict] = []
    for prob, oid, mask in zip(
        outputs["out_probs"],
        outputs["out_obj_ids"],
        outputs["out_binary_masks"],
    ):
        rows.append({
            "bbox": _bbox_from_mask(mask),
            "score": float(prob),
            "label": "person",
            "mask": mask,
            "mask_score": float(prob),
            "sam3_obj_id": int(oid),
        })
    return rows


def _bbox_from_mask(mask: np.ndarray) -> list[float]:
    """Tight xyxy bbox over True pixels (half-open upper bound, COCO style)."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()),
            float(xs.max()) + 1.0, float(ys.max()) + 1.0]
