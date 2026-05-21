"""Policy gate around WarmIdentityGallery. Decides when a TrackBinding's
face embedding is trustworthy enough to retain.

Reads TrackBinding only via narrow predicates so the gallery's drift
defenses share the same signals as the live binding's drift defenses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ta_src.tracking.utils import centroid_dist, iou_xyxy
from ta_src.tracking.warm_identity_gallery import WarmIdentityGallery


@dataclass(frozen=True)
class MaskSanityInputs:
    """Per-write context the operator-bypass path uses to detect SAM3
    segmentation drift before retaining a polluted embedding."""

    current_bbox: tuple[float, float, float, float]
    prev_bbox: Optional[tuple[float, float, float, float]] = None
    other_live_bboxes: list[tuple[float, float, float, float]] = field(
        default_factory=list,
    )
    frame_height: int = 0


class WarmGalleryWriter:
    def __init__(
        self,
        gallery: WarmIdentityGallery,
        min_face_det_score: float,
        low_confidence_face_cosine: float,
        osnet_trust_window_frames: int = 30,
        anchor_count: int = 0,
        mask_sanity_iou_floor: float = 0.0,
        mask_sanity_jump_px_frac: float = 0.0,
        confidence_log=None,
    ) -> None:
        self._gallery = gallery
        self._min_face_det_score = min_face_det_score
        self._low_conf_cos = low_confidence_face_cosine
        self._osnet_trust_window = osnet_trust_window_frames
        self._anchor_count = anchor_count
        self._mask_sanity_iou_floor = mask_sanity_iou_floor
        self._mask_sanity_jump_px_frac = mask_sanity_jump_px_frac
        self._confidence_log = confidence_log
        self._last_face_confirm_by_track: dict[int, int] = {}
        self._accepted_writes: dict[tuple[int, str], int] = {}

    def set_confidence_log(self, log) -> None:
        """Late-wire the shared ConfidenceLog so blocked-write telemetry can
        route through the same sink as `pool_stats` / `rebind`."""
        self._confidence_log = log

    def last_face_confirm_frame(self, track_id: int) -> int | None:
        return self._last_face_confirm_by_track.get(track_id)

    def maybe_write_face(
        self,
        binding,
        gid: int,
        face_emb: np.ndarray,
        det_score: float,
        matched_cosine: float,
        frame_idx: int,
        *,
        mask_sanity: Optional[MaskSanityInputs] = None,
    ) -> bool:
        if gid < 0:
            return False
        if det_score < self._min_face_det_score:
            return False
        if not binding.operator_assigned and matched_cosine < self._low_conf_cos:
            return False
        if binding.low_confidence_streak > 0:
            return False
        if binding.operator_assigned and mask_sanity is not None:
            reason = self._mask_sanity_blocked(mask_sanity)
            if reason is not None:
                self._emit_write_blocked(gid, "face", reason, frame_idx)
                return False
        is_anchor = self._accepted_writes.get((gid, "face"), 0) < self._anchor_count
        retained = self._gallery.write(
            gid=gid, kind="face", emb=face_emb,
            source_cosine=matched_cosine, frame_idx=frame_idx,
            is_anchor=is_anchor,
        )
        if retained:
            self._last_face_confirm_by_track[binding.track_id] = frame_idx
            self._accepted_writes[(gid, "face")] = (
                self._accepted_writes.get((gid, "face"), 0) + 1
            )
        return retained

    def maybe_write_osnet(
        self,
        binding,
        gid: int,
        osnet_emb: np.ndarray,
        frame_idx: int,
        *,
        mask_sanity: Optional[MaskSanityInputs] = None,
    ) -> bool:
        if gid < 0:
            return False
        if binding.low_confidence_streak > 0:
            return False
        if binding.osnet_column_loss_streak > 0:
            return False
        if not binding.operator_assigned:
            last_face = self._last_face_confirm_by_track.get(binding.track_id)
            if last_face is None:
                return False
            if frame_idx - last_face > self._osnet_trust_window:
                return False
        if binding.operator_assigned and mask_sanity is not None:
            reason = self._mask_sanity_blocked(mask_sanity)
            if reason is not None:
                self._emit_write_blocked(gid, "osnet", reason, frame_idx)
                return False
        is_anchor = self._accepted_writes.get((gid, "osnet"), 0) < self._anchor_count
        retained = self._gallery.write(
            gid=gid, kind="osnet", emb=osnet_emb,
            source_cosine=1.0, frame_idx=frame_idx, is_anchor=is_anchor,
        )
        if retained:
            self._accepted_writes[(gid, "osnet")] = (
                self._accepted_writes.get((gid, "osnet"), 0) + 1
            )
        return retained

    def _emit_write_blocked(
        self, gid: int, kind: str, reason: str, frame_idx: int,
    ) -> None:
        if self._confidence_log is None:
            return
        log = getattr(self._confidence_log, "log_warm_write_blocked", None)
        if log is None:
            return
        log(gid=int(gid), kind=kind, reason=reason, frame_idx=int(frame_idx))

    def _mask_sanity_blocked(
        self, mask_sanity: MaskSanityInputs,
    ) -> Optional[str]:
        if self._mask_sanity_iou_floor > 0.0:
            for other in mask_sanity.other_live_bboxes:
                if iou_xyxy(mask_sanity.current_bbox, other) > self._mask_sanity_iou_floor:
                    return "iou_overlap"
        if (
            self._mask_sanity_jump_px_frac > 0.0
            and mask_sanity.prev_bbox is not None
        ):
            threshold = self._mask_sanity_jump_px_frac * mask_sanity.frame_height
            if centroid_dist(mask_sanity.current_bbox, mask_sanity.prev_bbox) > threshold:
                return "bbox_jump"
        return None
