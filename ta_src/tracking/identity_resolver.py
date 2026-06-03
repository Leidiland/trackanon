"""Maps SAM 3 obj_ids to KPL global_ids via face evidence + sparse Hungarian.

Per-frame: face evidence accumulates at K-frame cadence; Hungarian fires once
a track reaches M observations or a high-quality face. At chunk boundaries,
active tracks snapshot full state and are matched into the next chunk by bbox IoU.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_INHERIT_INFEASIBLE = 1e6
# Cross-chunk inherit appearance blend: face is trusted enough to reorder
# feasible IoU pairs (breaks the boundary identity swap); OSNet stays a weak
# tiebreak only, since body cost is near-inert.
_INHERIT_FACE_WEIGHT = 0.5
_INHERIT_OSNET_WEIGHT = 0.1


_INHERIT_MASK_FLOOR = 0.1  # min mask IoU for a mask-disambiguated inherit
_INHERIT_MASK_CONTINUITY_FLOOR = 0.5  # mask IoU that rescues a sub-bbox-floor inherit
_OSNET_TIE_MARGIN = 0.03  # a free-gid osnet confirm must win KPL by this margin


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    return inter / int(np.logical_or(a, b).sum())


def _unit_mean(emb_sum: np.ndarray, weight_total: float) -> Optional[np.ndarray]:
    if weight_total <= 0:
        return None
    mean = emb_sum / weight_total
    n = float(np.linalg.norm(mean))
    return mean / n if n > 0 else mean


def _inherit_appearance_bonus(
    carry: "_ChunkBoundarySnapshot",
    new_face: Optional[np.ndarray],
    new_osnet: Optional[np.ndarray],
) -> float:
    # Face when both sides have it (trusted); else OSNet as a weak tiebreak.
    snap_face = _unit_mean(carry.face_emb_sum, carry.face_weight_total)
    if new_face is not None and snap_face is not None:
        return _INHERIT_FACE_WEIGHT * float(np.dot(new_face, snap_face))
    snap_osnet = _unit_mean(carry.osnet_emb_sum, carry.osnet_weight_total)
    if new_osnet is not None and snap_osnet is not None:
        return _INHERIT_OSNET_WEIGHT * float(np.dot(new_osnet, snap_osnet))
    return 0.0


def _module_log() -> logging.Logger:
    return logging.getLogger(__name__)

import numpy as np
from scipy.optimize import linear_sum_assignment

from ta_src.anonymization.confidence_gate import evaluate as evaluate_gate
from ta_src.tracking.hungarian_assigner import AssignmentInfo, Detection
from ta_src.tracking.utils import iou_xyxy
from ta_src.tracking.warm_gallery_writer import MaskSanityInputs, WarmGalleryWriter
from ta_src.tracking.warm_identity_gallery import WarmIdentityGallery


@dataclass
class TrackBinding:
    sam3_obj_id: int
    chunk_id: int
    global_id: int = -1
    face_emb_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(512, dtype=np.float32)
    )
    face_weight_total: float = 0.0
    n_face_observations: int = 0
    osnet_emb_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(512, dtype=np.float32)
    )
    osnet_weight_total: float = 0.0
    n_osnet_observations: int = 0
    confirmed: bool = False
    # Set by IdentityResolver.apply_operator_overrides — manual frame-0 gid
    # assignment treated as ground truth, bypasses every demotion path.
    operator_assigned: bool = False
    low_confidence_streak: int = 0  # face-path drift streak
    osnet_column_loss_streak: int = 0  # cross-track OSNet column-loss streak
    # gids demoted from this chunk — blocks the demote → re-confirm-same-gid cycle.
    blocked_gids: set[int] = field(default_factory=set)
    frames_seen: int = 0  # controls K-sampling cadence
    last_bbox: Optional[tuple] = None
    # Last full-frame silhouette — disambiguates the chunk-start inherit when two
    # bodies' bboxes overlap but their masks are disjoint (the duplicate-gid swap).
    last_mask: Optional[np.ndarray] = None
    last_seen_frame_idx: int = -1
    # Frame index of the most recent frame InsightFace sampled a face for this
    # binding. Debounces the K-cadence so the skeleton face-gate sees sustained
    # orientation, not per-frame sampling noise. -1 = no face ever sampled.
    last_face_frame_idx: int = -1
    # Fresh binding that did not inherit a snapshot: bbox carries no signal,
    # so the assigner drops the spatial blend on no-face confirmation cost.
    long_gap: bool = True
    # Most recent InsightFace per-detection confidence for this binding, used
    # by HungarianAssigner.face_quality_floor to gate the face term per row.
    # 0.0 is the safe default — at floor=0.0 (today) the gate never fires.
    last_face_det_score: float = 0.0

    @property
    def track_id(self) -> int:
        return self.chunk_id * 100_000 + self.sam3_obj_id

    @property
    def face_emb_running(self) -> Optional[np.ndarray]:
        if self.face_weight_total <= 0:
            return None
        mean = self.face_emb_sum / self.face_weight_total
        n = float(np.linalg.norm(mean))
        return mean / n if n > 0 else mean

    @property
    def osnet_emb_running(self) -> Optional[np.ndarray]:
        if self.osnet_weight_total <= 0:
            return None
        mean = self.osnet_emb_sum / self.osnet_weight_total
        n = float(np.linalg.norm(mean))
        return mean / n if n > 0 else mean


@dataclass
class _ChunkBoundarySnapshot:
    """Full binding state captured at chunk end — matched into the next chunk by
    bbox IoU (mask IoU when silhouettes are available) so a confirmed gid
    survives the SAM3 obj_id renumber."""
    last_bbox: tuple[float, float, float, float]
    last_mask: Optional[np.ndarray]
    face_emb_sum: np.ndarray
    face_weight_total: float
    n_face_observations: int
    osnet_emb_sum: np.ndarray
    osnet_weight_total: float
    n_osnet_observations: int
    confirmed: bool
    operator_assigned: bool
    global_id: int
    osnet_column_loss_streak: int
    blocked_gids: set[int]


class IdentityResolver:
    def __init__(
        self,
        face_wrapper,
        gallery,
        assigner,
        thresholds,
        face_sampling_K: int = 5,
        face_confirm_M: int = 3,
        low_confidence_face_cosine: float = 0.4,
        low_confidence_streak_max: int = 5,
        osnet_column_loss_streak_max: int = 5,
        osnet_column_loss_margin: float = 0.05,
        osnet_wrapper=None,
        warm_gallery_enabled: bool = False,
        warm_face_size: int = 10,
        warm_osnet_size: int = 20,
        warm_anchor_count: int = 3,
        warm_dedup_cosine: float = 0.95,
        warm_osnet_trust_window_frames: int = 30,
        warm_mask_sanity_iou_floor: float = 0.0,
        warm_mask_sanity_jump_px_frac: float = 0.0,
        min_face_det_score: float = 0.6,
        confidence_log=None,
        intra_chunk_revival_iou_floor: float = 0.5,
        intra_chunk_revival_max_age_frames: int = 15,
        partial_carry_iou_floor: float = 0.5,
        face_consistency_cos_floor: float = 0.3,
        face_consistency_det_score_floor: float = 0.7,
        locality_max_jump_px: float = 300.0,
        locality_max_stale_frames: int = 5,
        locality_max_speed_px: float = 100.0,
    ) -> None:
        self._face = face_wrapper
        self._gallery = gallery
        self._assigner = assigner
        self._thresholds = thresholds
        self._K = face_sampling_K
        self._M = face_confirm_M
        self._low_conf_cos = low_confidence_face_cosine
        self._streak_max = low_confidence_streak_max
        self._osnet_loss_max = osnet_column_loss_streak_max
        self._osnet_loss_margin = osnet_column_loss_margin
        self._osnet = osnet_wrapper
        self._min_face_det_score = min_face_det_score
        self._face_consistency_cos_floor = float(face_consistency_cos_floor)
        self._face_consistency_det_score_floor = float(face_consistency_det_score_floor)
        # Locality gate: a gid seen <= N frames ago can't (re)bind to a box more
        # than M px from where it was last seen — physically impossible motion.
        self._locality_max_jump_px = float(locality_max_jump_px)
        self._locality_max_stale_frames = int(locality_max_stale_frames)
        self._locality_max_speed_px = float(locality_max_speed_px)
        # gid -> (frame_idx, (cx, cy)) last frame each gid held a detection.
        self._gid_last_seen: dict[int, tuple[int, tuple[float, float]]] = {}
        self._current_frame_idx: int = -1

        if warm_gallery_enabled:
            self._warm_gallery: WarmIdentityGallery | None = WarmIdentityGallery(
                face_size=warm_face_size,
                osnet_size=warm_osnet_size,
                dedup_cosine=warm_dedup_cosine,
            )
            self._warm_writer: WarmGalleryWriter | None = WarmGalleryWriter(
                gallery=self._warm_gallery,
                min_face_det_score=min_face_det_score,
                low_confidence_face_cosine=low_confidence_face_cosine,
                osnet_trust_window_frames=warm_osnet_trust_window_frames,
                anchor_count=warm_anchor_count,
                mask_sanity_iou_floor=warm_mask_sanity_iou_floor,
                mask_sanity_jump_px_frac=warm_mask_sanity_jump_px_frac,
            )
        else:
            self._warm_gallery = None
            self._warm_writer = None

        self._confidence_log = confidence_log

        # When SAM 3 mid-chunk drops an obj_id and re-emits the same person
        # under a new id, IoU-match against recently-died bindings preserves
        # operator_assigned status + accumulated evidence (saves the cascade
        # until the next chunk boundary). Floor > 1 disables.
        self._intra_revival_iou_floor = float(intra_chunk_revival_iou_floor)
        self._intra_revival_max_age = int(intra_chunk_revival_max_age_frames)
        # Cross-chunk carry: leaves margin for jitter without admitting unrelated tracks.
        self._partial_carry_iou_floor = float(partial_carry_iou_floor)
        # gids the operator assigned this video; any track that re-binds to one
        # re-acquires operator stickiness (survives warm-rebind handoffs) without
        # loosening any geometric gate.
        self._operator_gids: set[int] = set()

        self._chunk_id: int = -1
        self._tracks: dict[int, TrackBinding] = {}  # keyed by sam3_obj_id
        # Cross-chunk state snapshots; matched by bbox IoU on the next chunk's first frame.
        self._carryover: list[_ChunkBoundarySnapshot] = []
        self._frames_into_chunk: int = 0

        self._trace_log: list[dict] | None = None
        self._gid_prev_by_track: dict[int, int] = {}
        self._first_frame_by_track: dict[int, int] = {}
        self._chunk_boundary_pending: bool = False
        # obj_id -> why a confirm-eligible track did not bind this frame (trace).
        self._confirm_abstain: dict[int, str] = {}

    def set_confidence_log(self, log) -> None:
        """Late-wire the shared ConfidenceLog (constructed by AnonymizationStage)."""
        self._confidence_log = log
        if self._warm_writer is not None:
            self._warm_writer.set_confidence_log(log)

    def reset_video(self) -> None:
        """Clear per-video state at a video boundary. The resolver is reused
        across every video in a batch run; without this, chunk snapshots and
        per-track history bleed from one video into the next (and the trace
        buffer grows unbounded across the whole run)."""
        if self._trace_log is not None:
            self._trace_log = []
        self._gid_prev_by_track.clear()
        self._first_frame_by_track.clear()
        self._chunk_id = -1
        self._tracks = {}
        self._carryover = []
        self._frames_into_chunk = 0
        self._chunk_boundary_pending = False
        self._gid_last_seen = {}
        self._current_frame_idx = -1
        self._operator_gids = set()
        if self._warm_writer is not None:
            self._warm_writer.reset_video()

    def start_chunk(self, chunk_id: int) -> None:
        if self._chunk_id >= 0:
            self._emit_pool_stats(closing_chunk_id=self._chunk_id)
        if self._chunk_id >= 0 and self._tracks:
            # Snapshot most-recent-frame bindings only — mid-chunk dropouts
            # shouldn't match against similarly-positioned new tracks.
            last_frame = max(
                b.last_seen_frame_idx for b in self._tracks.values()
            )
            self._carryover = [
                _ChunkBoundarySnapshot(
                    last_bbox=tuple(b.last_bbox),
                    last_mask=b.last_mask,
                    face_emb_sum=b.face_emb_sum.copy(),
                    face_weight_total=b.face_weight_total,
                    n_face_observations=b.n_face_observations,
                    osnet_emb_sum=b.osnet_emb_sum.copy(),
                    osnet_weight_total=b.osnet_weight_total,
                    n_osnet_observations=b.n_osnet_observations,
                    confirmed=b.confirmed,
                    operator_assigned=b.operator_assigned,
                    global_id=b.global_id,
                    osnet_column_loss_streak=b.osnet_column_loss_streak,
                    blocked_gids=set(b.blocked_gids),
                )
                for b in self._tracks.values()
                if b.last_bbox is not None
                and b.last_seen_frame_idx == last_frame
                and (
                    b.confirmed
                    or b.n_face_observations > 0
                    or b.n_osnet_observations > 0
                )
            ]
        self._chunk_id = chunk_id
        self._tracks = {}
        self._frames_into_chunk = 0
        self._chunk_boundary_pending = True

    def update(
        self,
        frame_rgb: np.ndarray,
        sam3_rows: list[dict],
        frame_idx: int,
    ) -> list[dict]:
        self._current_frame_idx = frame_idx
        if self._chunk_boundary_pending:
            self._record_chunk_boundary(frame_idx)
            self._chunk_boundary_pending = False

        # Pass 0: revive recently-died bindings whose last_bbox aligns with a
        # new sam3_obj_id this frame. SAM 3 occasionally drops an obj_id then
        # re-emits the same person under a fresh id (memory-attention quirk);
        # transferring the live binding to the new key keeps operator_assigned
        # and confirmed gid intact without waiting for the chunk boundary.
        self._revive_died_bindings(sam3_rows, frame_idx)

        # Neighbour-subtracted face crops — shared by the inherit's on-demand
        # appearance sample and Pass 1's accumulation (keeps a crosser's face
        # out of both).
        face_remove_masks = self._neighbour_remove_masks(sam3_rows)

        # Pass 0b: chunk-start one-to-one Hungarian over bbox IoU, with a face
        # cosine blend so a face-consistent snapshot beats a higher-IoU
        # stranger (the boundary identity swap). Greedy best-per-row also let
        # an interloping bbox steal a gid; the Hungarian fixes that.
        inherit_by_obj_id: dict[int, _ChunkBoundarySnapshot] = (
            self._compute_chunk_start_inherits(
                sam3_rows, frame_rgb, face_remove_masks
            )
            if self._frames_into_chunk == 0 and self._carryover
            else {}
        )

        # Pass 1: refresh bindings + accumulate face evidence. Streak update
        # is deferred — it needs the column-winner view across all bindings.
        bindings_this_frame: list[TrackBinding] = []
        sampled_this_frame: list[tuple[TrackBinding, np.ndarray]] = []
        sampled_osnet_this_frame: list[tuple[TrackBinding, np.ndarray]] = []
        # Snapshot prev-frame bboxes before any binding's last_bbox is
        # overwritten this frame — feeds the op_edit mask-sanity jump gate.
        prev_bbox_by_obj_id: dict[int, tuple] = {
            oid: tuple(b.last_bbox) for oid, b in self._tracks.items()
            if b.last_bbox is not None
        }
        for row in sam3_rows:
            sam3_obj_id = int(row["sam3_obj_id"])
            existing = self._tracks.get(sam3_obj_id)
            # SAM 3 obj_ids stay stable across brief gaps; keep the binding
            # alive so flicker doesn't cause flash-to-unmatched cycles.
            if existing is None:
                binding = TrackBinding(
                    sam3_obj_id=sam3_obj_id, chunk_id=self._chunk_id
                )
                self._tracks[sam3_obj_id] = binding
                carry = inherit_by_obj_id.get(sam3_obj_id)
                if carry is not None:
                    self._apply_carryover_snapshot(binding, carry)
            else:
                binding = existing
            binding.last_bbox = tuple(row["bbox"])
            binding.last_mask = row.get("mask")
            binding.last_seen_frame_idx = frame_idx
            new_face = self._maybe_accumulate_face(
                frame_rgb, row, binding,
                remove_mask=face_remove_masks.get(sam3_obj_id),
            )
            if new_face is not None:
                sampled_this_frame.append((binding, new_face))
                binding.last_face_frame_idx = frame_idx
            new_osnet = self._maybe_accumulate_osnet(frame_rgb, row, binding)
            if new_osnet is not None:
                sampled_osnet_this_frame.append((binding, new_osnet))
            binding.frames_seen += 1
            bindings_this_frame.append(binding)

        # Pass 2: streak gate over bindings with a new face sample this frame.
        self._update_streaks(sampled_this_frame)
        # Pass 2b: OSNet column-loss demotion (face-confirmed tracks exempt).
        self._update_osnet_column_loss_streaks(sampled_osnet_this_frame)

        # Run confirmation Hungarian on bindings that hit M observations.
        self._maybe_confirm()

        # Operator stickiness is identity-level: any binding re-bound to an
        # operator-assigned gid (warm rebind, inherit, or fresh confirm)
        # re-acquires the flag, so a labelled person isn't demotable after a
        # chunk-boundary handoff drops the original binding.
        if self._operator_gids:
            for b in bindings_this_frame:
                if b.confirmed and b.global_id in self._operator_gids:
                    b.operator_assigned = True

        # Per-pass uniqueness: one physical person -> one gid. Two confirmed
        # tracks sharing a gid are a genuine duplicate ONLY when their
        # silhouettes are disjoint (two people); an overlapping co-hold is the
        # same person mid chunk-handoff and is left alone.
        self._enforce_gid_uniqueness(bindings_this_frame)

        if self._warm_writer is not None:
            frame_height = int(frame_rgb.shape[0])
            self._maybe_write_warm_face(
                sampled_this_frame, sam3_rows, frame_idx,
                prev_bbox_by_obj_id=prev_bbox_by_obj_id,
                frame_height=frame_height,
            )
            self._maybe_write_warm_osnet(
                sampled_osnet_this_frame, frame_idx,
                sam3_rows=sam3_rows,
                prev_bbox_by_obj_id=prev_bbox_by_obj_id,
                frame_height=frame_height,
            )

        self._frames_into_chunk += 1
        if self._frames_into_chunk >= 1:
            # Carryover only fires on chunk-frame 0; drop it so mid-chunk
            # tracks at the same position can't claim a snapshot.
            self._carryover = []

        # Record each gid's current location for the locality gate (used on the
        # next frame's rebinds — so it reflects positions BEFORE this frame).
        for b in self._tracks.values():
            if b.global_id >= 0 and b.last_bbox is not None:
                bx = b.last_bbox
                self._gid_last_seen[b.global_id] = (
                    frame_idx, ((bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0),
                )

        # Enrich AFTER confirmation so this frame reflects new gid assignments.
        sampled_obj_ids = {b.sam3_obj_id for b, _emb in sampled_this_frame}
        self._record_det_rows(sam3_rows, bindings_this_frame, sampled_obj_ids, frame_idx)
        # Face counts as "recently visible" if sampled within 3 cadence periods —
        # tolerates a missed sample without falsely suppressing a frontal face.
        face_window = 3 * self._K
        return [
            self._enrich(
                row, b,
                face_visible_recently=(
                    b.last_face_frame_idx >= 0
                    and frame_idx - b.last_face_frame_idx <= face_window
                ),
            )
            for row, b in zip(sam3_rows, bindings_this_frame)
        ]

    def apply_operator_overrides(
        self,
        mapping: dict[int, Optional[int]],
        sam3_rows: list[dict],
    ) -> list[dict]:
        for sam3_obj_id, gid in mapping.items():
            if sam3_obj_id not in self._tracks:
                raise KeyError(
                    f"operator override references unknown sam3_obj_id={sam3_obj_id}; "
                    f"live bindings: {sorted(self._tracks)}"
                )
            if gid is None:
                continue
            b = self._tracks[sam3_obj_id]
            b.global_id = int(gid)
            b.confirmed = True
            b.operator_assigned = True
            self._operator_gids.add(int(gid))
        return [
            self._enrich(row, self._tracks[int(row["sam3_obj_id"])])
            for row in sam3_rows
        ]

    def _revive_died_bindings(
        self,
        sam3_rows: list[dict],
        frame_idx: int,
    ) -> None:
        if self._intra_revival_iou_floor > 1.0:
            return
        current_obj_ids = {int(r["sam3_obj_id"]) for r in sam3_rows}
        # Bindings absent from this frame are revival candidates; sort by
        # recency so the freshest dropout wins ties.
        candidates = [
            b for oid, b in self._tracks.items()
            if oid not in current_obj_ids
            and b.last_bbox is not None
            and frame_idx - b.last_seen_frame_idx <= self._intra_revival_max_age
        ]
        if not candidates:
            return
        new_rows = [
            r for r in sam3_rows
            if int(r["sam3_obj_id"]) not in self._tracks
        ]
        if not new_rows:
            return
        pairs: list[tuple[float, int, int]] = []
        for ri, row in enumerate(new_rows):
            for ci, cand in enumerate(candidates):
                iou = iou_xyxy(tuple(row["bbox"]), cand.last_bbox)
                if iou >= self._intra_revival_iou_floor:
                    pairs.append((iou, ri, ci))
        pairs.sort(reverse=True)
        used_rows: set[int] = set()
        used_cands: set[int] = set()
        for _iou, ri, ci in pairs:
            if ri in used_rows or ci in used_cands:
                continue
            used_rows.add(ri)
            used_cands.add(ci)
            self._rekey_binding(
                candidates[ci], int(new_rows[ri]["sam3_obj_id"]),
            )

    def _rekey_binding(self, binding: TrackBinding, new_obj_id: int) -> None:
        old_obj_id = binding.sam3_obj_id
        old_tid = binding.track_id
        binding.sam3_obj_id = new_obj_id
        del self._tracks[old_obj_id]
        self._tracks[new_obj_id] = binding
        new_tid = binding.track_id
        if old_tid in self._first_frame_by_track:
            self._first_frame_by_track[new_tid] = (
                self._first_frame_by_track.pop(old_tid)
            )
        if old_tid in self._gid_prev_by_track:
            self._gid_prev_by_track[new_tid] = (
                self._gid_prev_by_track.pop(old_tid)
            )
        # WarmGalleryWriter is keyed by track_id and would otherwise drop the
        # face-confirm history on rekey, blocking OSNet writes for the rest
        # of the chunk (the trust-window gate sees `last_face is None`).
        if self._warm_writer is not None:
            self._warm_writer.rekey_track(old_tid, new_tid)

    def _compute_chunk_start_inherits(
        self, sam3_rows: list[dict], frame_rgb: np.ndarray,
        remove_masks: dict[int, Optional[np.ndarray]],
    ) -> dict[int, _ChunkBoundarySnapshot]:
        # One-to-one Hungarian between new sam3_obj_ids and end-of-prev-chunk
        # snapshots. bbox IoU below partial_carry_iou_floor is the cheap
        # feasibility pre-filter; among bbox-feasible pairs, mask IoU vetoes a
        # body whose silhouette is disjoint from the snapshot (two overlapping
        # bboxes, separate people — the duplicate-gid swap) and a face cosine
        # blend reorders the rest. Matched snapshots are popped from _carryover.
        frame_hw = frame_rgb.shape[:2]
        new_rows = [
            (int(r["sam3_obj_id"]), tuple(float(v) for v in r["bbox"]), r.get("mask"))
            for r in sam3_rows
            if int(r["sam3_obj_id"]) not in self._tracks
        ]
        if not new_rows or not self._carryover:
            return {}
        n_rows = len(new_rows)
        n_carry = len(self._carryover)
        # The new track has no accumulated embedding yet (this runs before Pass
        # 1), so sample its face/OSNet on demand — lazily, at most once per row,
        # only when a feasible snapshot is present to disambiguate.
        appearance_cache: dict[int, tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}

        def _new_appearance(i):
            if i not in appearance_cache:
                oid, bbox, _m = new_rows[i]
                appearance_cache[i] = self._sample_appearance(
                    frame_rgb, bbox, remove_masks.get(oid),
                )
            return appearance_cache[i]

        cost = np.full((n_rows, n_carry), _INHERIT_INFEASIBLE, dtype=np.float64)
        for i, (_oid, bbox, mask) in enumerate(new_rows):
            for j, carry in enumerate(self._carryover):
                iou = iou_xyxy(bbox, carry.last_bbox)
                # Mask refinement only when both silhouettes are full-frame
                # (real SAM3 masks). A disjoint mask vetoes a bbox-overlapping
                # stranger; a high mask IoU rescues a partial re-detection whose
                # bbox fell below the floor (the same person, smaller crop).
                miou = None
                if (mask is not None and carry.last_mask is not None
                        and mask.shape == frame_hw
                        and carry.last_mask.shape == frame_hw):
                    miou = _mask_iou(mask, carry.last_mask)
                bbox_ok = iou >= self._partial_carry_iou_floor
                # Only a face-confirmed snapshot may rescue a sub-bbox-floor
                # inherit on mask alone — a faceless (OSNet/operator-only) gid is
                # leak-prone and the inherit has no appearance veto, so mask
                # continuity would otherwise spread a wrong gid across chunks.
                mask_continues = (
                    miou is not None and miou >= _INHERIT_MASK_CONTINUITY_FLOOR
                    and carry.n_face_observations > 0
                )
                if not (bbox_ok or mask_continues):
                    continue
                if miou is not None and miou < _INHERIT_MASK_FLOOR:
                    continue
                geo = miou if miou is not None else iou
                new_face, new_osnet = _new_appearance(i)
                bonus = _inherit_appearance_bonus(carry, new_face, new_osnet)
                cost[i, j] = (1.0 - geo) - bonus
        row_idx, col_idx = linear_sum_assignment(cost)
        result: dict[int, _ChunkBoundarySnapshot] = {}
        matched_carry_indices: list[int] = []
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] >= _INHERIT_INFEASIBLE:
                continue
            obj_id = new_rows[r][0]
            result[obj_id] = self._carryover[c]
            matched_carry_indices.append(int(c))
        # Pop in reverse-index order so list indices stay valid.
        for c in sorted(matched_carry_indices, reverse=True):
            self._carryover.pop(c)
        return result

    def _sample_appearance(
        self, frame_rgb: np.ndarray, bbox: tuple, remove_mask: Optional[np.ndarray],
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Face (preferred) + OSNet embedding for a NEW box at inherit time —
        no binding accumulation, used only to disambiguate the inherit match."""
        face = None
        crop = _crop(frame_rgb, bbox, remove_mask=remove_mask)
        if crop.size > 0:
            result = self._face.extract_with_quality(crop, tuple(bbox))
            if result is not None and result[1] > 0:
                face = result[0]
        osnet = None
        if self._osnet is not None:
            embs = self._osnet.extract(frame_rgb, [bbox])
            if embs.shape[0] > 0 and np.any(embs[0]):
                osnet = embs[0]
        return face, osnet

    def _apply_carryover_snapshot(
        self,
        binding: TrackBinding,
        carry: _ChunkBoundarySnapshot,
    ) -> None:
        binding.face_emb_sum = carry.face_emb_sum
        binding.face_weight_total = carry.face_weight_total
        binding.n_face_observations = carry.n_face_observations
        binding.osnet_emb_sum = carry.osnet_emb_sum
        binding.osnet_weight_total = carry.osnet_weight_total
        binding.n_osnet_observations = carry.n_osnet_observations
        binding.confirmed = carry.confirmed
        binding.operator_assigned = carry.operator_assigned
        binding.global_id = carry.global_id
        binding.low_confidence_streak = 0
        binding.osnet_column_loss_streak = carry.osnet_column_loss_streak
        binding.blocked_gids = set(carry.blocked_gids)
        binding.long_gap = False

    def _consistency_reset_fires(
        self, binding: TrackBinding, new_emb: np.ndarray, det_score: float,
    ) -> bool:
        """True when the new face sample is inconsistent with the binding's
        running mean — a SAM3 mask hand-over signature. Forces accumulator
        reset so the obj_id rebinds fresh next chunk instead of contaminating
        the warm gallery."""
        if det_score < self._face_consistency_det_score_floor:
            return False
        if binding.n_face_observations < 2:
            return False
        running = binding.face_emb_running
        if running is None:
            return False
        cos = float(np.dot(new_emb, running))
        return cos < self._face_consistency_cos_floor

    def _neighbour_remove_masks(
        self, sam3_rows: list[dict]
    ) -> dict[int, Optional[np.ndarray]]:
        # Per obj_id: the pixels owned by any OTHER detection that intrudes this
        # bbox, minus this detection's own silhouette. Subtracting only these
        # from the face crop keeps a neighbour's face out without clipping the
        # person's own face (no-op for isolated detections → raw bbox).
        rows = [r for r in sam3_rows if r.get("mask") is not None]
        out: dict[int, Optional[np.ndarray]] = {}
        for r in rows:
            oid = int(r["sam3_obj_id"])
            own = np.asarray(r["mask"]).astype(bool)
            x1, y1, x2, y2 = (int(round(float(v))) for v in r["bbox"])
            union = None
            for other in rows:
                if other is r:
                    continue
                om = np.asarray(other["mask"]).astype(bool)
                if om.shape != own.shape:
                    continue  # masks must share the frame grid to be subtracted
                if not om[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].any():
                    continue  # neighbour does not intrude this bbox
                union = om.copy() if union is None else (union | om)
            out[oid] = (union & ~own) if union is not None else None
        return out

    def _maybe_accumulate_face(
        self, frame_rgb: np.ndarray, row: dict, binding: TrackBinding,
        remove_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Return the new face embedding sampled this call (or None)."""
        if binding.frames_seen % self._K != 0:
            return None
        bbox = row["bbox"]
        body_crop = _crop(frame_rgb, bbox, remove_mask=remove_mask)
        if body_crop.size == 0:
            return None
        result = self._face.extract_with_quality(body_crop, tuple(bbox))
        if result is None:
            return None
        emb, quality, det_score = result
        if quality <= 0:
            return None
        if self._consistency_reset_fires(binding, emb, det_score):
            log = _module_log()
            log.info(
                "face_consistency_reset gid=%d obj_id=%d det_score=%.2f cos=%.3f",
                binding.global_id, binding.sam3_obj_id, det_score,
                float(np.dot(emb, binding.face_emb_running))
                if binding.face_emb_running is not None else float("nan"),
            )
            binding.face_emb_sum = np.zeros(512, dtype=np.float32)
            binding.face_weight_total = 0.0
            binding.n_face_observations = 0
            # Sample still flows to streak detection — running mean was the
            # binding's own past, not the gallery centroid the streak gate
            # compares against.
            binding.last_face_det_score = float(det_score)
            return emb
        binding.face_emb_sum += emb * quality
        binding.face_weight_total += quality
        binding.n_face_observations += 1
        # Surface the per-detection InsightFace confidence so HungarianAssigner
        # can drop the face term for low-quality detections.
        binding.last_face_det_score = float(det_score)
        return emb

    def _maybe_accumulate_osnet(
        self, frame_rgb: np.ndarray, row: dict, binding: TrackBinding
    ) -> Optional[np.ndarray]:
        """OSNet fallback evidence — keeps back-view tracks identifiable when face fails."""
        if self._osnet is None:
            return None
        if binding.frames_seen % self._K != 0:
            return None
        bbox = row["bbox"]
        embs = self._osnet.extract(frame_rgb, [bbox])
        if embs.shape[0] == 0:
            return None
        emb = embs[0]
        if not np.any(emb):  # zero embedding → invalid crop, skip
            return None
        binding.osnet_emb_sum += emb
        binding.osnet_weight_total += 1.0
        binding.n_osnet_observations += 1
        return emb

    def _update_streaks(
        self, sampled: list[tuple[TrackBinding, np.ndarray]]
    ) -> None:
        # Reset streak on row_max OR available_row_max; increment if below threshold.
        # Demote at low_confidence_streak_max. Column-winner spans all bindings.
        if not sampled or len(self._gallery) == 0:
            return
        identities = list(self._gallery)

        # cos[obj_id][gid] = per-frame face emb cosine with identity centroid.
        cos_matrix: dict[int, dict[int, float]] = {
            b.sam3_obj_id: {
                ident.global_id: float(np.dot(new_emb, ident.face_centroid))
                for ident in identities
            }
            for b, new_emb in sampled
        }

        # Column winner per gid (strictly highest; tie → None).
        column_winner: dict[int, Optional[int]] = {}
        for ident in identities:
            gid = ident.global_id
            best_obj_id: Optional[int] = None
            best_cos = -float("inf")
            tied = False
            for b, _new_emb in sampled:
                c = cos_matrix[b.sam3_obj_id][gid]
                if c > best_cos:
                    best_cos = c
                    best_obj_id = b.sam3_obj_id
                    tied = False
                elif c == best_cos:
                    tied = True
            column_winner[gid] = None if tied else best_obj_id

        for b, _new_emb in sampled:
            if not b.confirmed:
                continue
            if b.operator_assigned:
                continue
            cosines = cos_matrix[b.sam3_obj_id]
            row_max_gid = max(cosines, key=cosines.get)
            won_gids = [g for g, w in column_winner.items() if w == b.sam3_obj_id]
            available_row_max_gid = (
                max(won_gids, key=lambda g: cosines[g]) if won_gids else None
            )
            cos_assigned = cosines.get(b.global_id, 0.0)
            if b.global_id == row_max_gid or b.global_id == available_row_max_gid:
                b.low_confidence_streak = 0
            elif cos_assigned < self._low_conf_cos and not self._running_mean_supports(
                b, identities
            ):
                # Only a per-frame dip that the accumulated identity ALSO
                # contradicts counts as a strike. The running mean is the
                # trusted identity; demoting on per-frame noise re-unknowns
                # distant/profile faces whose accumulated match is still solid.
                b.low_confidence_streak += 1
                if b.low_confidence_streak >= self._streak_max:
                    self._demote(b)
            else:
                b.low_confidence_streak = 0

    def _running_mean_supports(
        self, b: TrackBinding, identities: list
    ) -> bool:
        # The accumulated (running-mean) face still backs the assigned gid:
        # either it is the argmax identity, or it clears the low-confidence
        # floor. Absent a running mean, there is nothing to defend the bind.
        running = b.face_emb_running
        if running is None:
            return False
        cos = {
            ident.global_id: float(np.dot(running, ident.face_centroid))
            for ident in identities
        }
        argmax_gid = max(cos, key=cos.get)
        return b.global_id == argmax_gid or cos.get(b.global_id, 0.0) >= self._low_conf_cos

    def _enforce_gid_uniqueness(self, bindings: list[TrackBinding]) -> None:
        by_gid: dict[int, list[TrackBinding]] = {}
        for b in bindings:
            if b.confirmed and b.global_id >= 0:
                by_gid.setdefault(b.global_id, []).append(b)
        for group in by_gid.values():
            if len(group) < 2:
                continue
            anchor = max(group, key=self._uniqueness_strength)
            for b in group:
                if b is anchor or not self._spatially_disjoint(b, anchor):
                    continue
                # Wrong body holding the gid (a copied-evidence duplicate may
                # even carry operator_assigned) — clear it so the demotion sticks.
                b.operator_assigned = False
                self._demote(b)

    @staticmethod
    def _uniqueness_strength(b: TrackBinding) -> tuple:
        # Operator label first, then accumulated evidence, then tenure.
        return (b.operator_assigned, b.face_weight_total,
                b.osnet_weight_total, b.frames_seen)

    def _spatially_disjoint(self, a: TrackBinding, b: TrackBinding) -> bool:
        # Disjoint silhouettes = two people (a genuine duplicate). Overlapping =
        # one person mid-handoff (exempt). Mask IoU when both masks are present
        # and same-shape; else only a zero-bbox-overlap counts as disjoint, so a
        # mask-less handoff is never wrongly demoted.
        if (a.last_mask is not None and b.last_mask is not None
                and a.last_mask.shape == b.last_mask.shape):
            return _mask_iou(a.last_mask, b.last_mask) < _INHERIT_MASK_FLOOR
        if a.last_bbox is None or b.last_bbox is None:
            return False
        return iou_xyxy(tuple(a.last_bbox), tuple(b.last_bbox)) <= 0.0

    def _demote(self, b: TrackBinding) -> None:
        # Reset evidence + block re-binding to the same gid (sparse-KPL would repeat the error).
        if b.global_id >= 0:
            b.blocked_gids.add(b.global_id)
        b.global_id = -1
        b.confirmed = False
        b.low_confidence_streak = 0
        b.osnet_column_loss_streak = 0
        b.face_emb_sum = np.zeros(512, dtype=np.float32)
        b.face_weight_total = 0.0
        b.n_face_observations = 0
        b.osnet_emb_sum = np.zeros(512, dtype=np.float32)
        b.osnet_weight_total = 0.0
        b.n_osnet_observations = 0

    def _update_osnet_column_loss_streaks(
        self, sampled: list[tuple[TrackBinding, np.ndarray]]
    ) -> None:
        # Cross-track column-loss demotion (OSNet path). Only unconfirmed
        # tracks are challengers; face-confirmed defenders are exempt.
        if not sampled or len(self._gallery) == 0:
            return
        identities = list(self._gallery)
        cos_matrix: dict[int, dict[int, float]] = {
            b.sam3_obj_id: {
                ident.global_id: float(np.dot(new_emb, ident.appearance_centroid))
                for ident in identities
            }
            for b, new_emb in sampled
        }
        unconfirmed = [(b, e) for b, e in sampled if not b.confirmed]
        for b, _new_emb in sampled:
            if not b.confirmed:
                continue
            if b.operator_assigned:
                continue
            if b.n_face_observations > 0:
                continue
            my_cos = cos_matrix[b.sam3_obj_id][b.global_id]
            challenger_max = max(
                (cos_matrix[c.sam3_obj_id][b.global_id]
                 for c, _e in unconfirmed),
                default=-float("inf"),
            )
            # Require a meaningful margin — centroid noise ~0.05 produces
            # false challenges on stable tracks otherwise.
            if challenger_max - my_cos > self._osnet_loss_margin:
                b.osnet_column_loss_streak += 1
                if b.osnet_column_loss_streak >= self._osnet_loss_max:
                    self._demote(b)
            else:
                b.osnet_column_loss_streak = 0

    def _maybe_confirm(self) -> None:
        self._confirm_abstain = {}
        candidates_with_evidence = [
            b for b in self._tracks.values()
            if not b.confirmed and (
                b.n_face_observations >= self._M
                or b.n_osnet_observations >= self._M
            )
        ]
        if not candidates_with_evidence:
            return
        confirmed_gids = {
            b.global_id for b in self._tracks.values() if b.confirmed
        }
        trace = self._trace_log is not None
        free_identities = [
            ident for ident in self._gallery
            if ident.global_id not in confirmed_gids
        ]
        if not free_identities:
            if trace:
                for b in candidates_with_evidence:
                    self._note_confirm_abstain(b, "gid_taken", confirmed_gids)
            return

        detections = [
            Detection(
                box=tuple(float(v) for v in b.last_bbox or (0, 0, 0, 0)),
                face_emb=b.face_emb_running,
                osnet_emb=(b.osnet_emb_running
                           if b.osnet_emb_running is not None
                           else np.zeros(512, dtype=np.float32)),
                long_gap=b.long_gap,
                face_det_score=b.last_face_det_score,
                # Blocked pairs are masked in the cost matrix so Hungarian
                # can spend the assignment on the next-best free Identity.
                blocked_gids=set(b.blocked_gids),
            )
            for b in candidates_with_evidence
        ]
        candidates = [ident.to_candidate() for ident in free_identities]
        matched, _excess, infos = self._assigner.assign(
            detections, candidates, warm=self._warm_gallery,
        )
        # Filter through ConfidenceGate; OSNet-only matches must clear floor + margin.
        for (det_idx, gid), info in zip(matched, infos):
            b = candidates_with_evidence[det_idx]
            reason: str | None = None
            if not evaluate_gate(info, self._thresholds).confirmed:
                reason = "gate"
            # KPL-agreement: warm augmentation may rank a match, but the trusted
            # KPL centroid must independently clear the floor before a fresh
            # identity is committed. A contaminated crop can match a neighbour's
            # in-video warm faces, and the gate's max(KPL, warm) would otherwise
            # let warm single-handedly lock the wrong gid.
            elif info.cost_path == "face" and (
                info.kpl_sim is None
                or info.kpl_sim < self._thresholds.face_sim_floor
            ):
                reason = "kpl_disagree"
            # Per-box appearance gate: a body-only (OSNet) free-gid confirm must
            # WIN the KPL appearance by a clear margin. Warm-only scoring zeroes an
            # unseen identity, so a same-build body would otherwise be stolen by a
            # seen identity's warm pool; the live KPL centroids veto both a
            # different-identity argmax (f273 Klaus/Werner, Florian) AND a near-tie
            # second the body can't be told apart from (f1358 Luca/Florian) — both
            # are coin-flips, so abstain rather than guess.
            elif info.cost_path == "osnet" and self._osnet_match_ambiguous(b, gid):
                reason = "osnet_ambiguous"
            elif not self._locality_ok(b, gid, cost_path=info.cost_path):
                reason = "locality"
            if reason is not None:
                if trace:
                    self._note_confirm_abstain(b, reason, confirmed_gids)
                continue
            b.global_id = gid
            b.confirmed = True
            self._emit_rebind_event(b, info)

        if trace:
            for b in candidates_with_evidence:
                if b.confirmed or b.sam3_obj_id in self._confirm_abstain:
                    continue
                self._note_confirm_abstain(b, "no_free_match", confirmed_gids)

    def _note_confirm_abstain(
        self, b: TrackBinding, proximate: str, confirmed_gids: set[int]
    ) -> None:
        # Root cause beats proximate: when a track's best identity over the FULL
        # gallery is already held, the gid being unavailable is the real reason a
        # well-matched track can't bind — surface that over the fallback-gid miss.
        emb = b.face_emb_running
        attr = "face_centroid"
        if emb is None:
            emb = b.osnet_emb_running
            attr = "appearance_centroid"
        if emb is not None and len(self._gallery) > 0:
            cos = {
                ident.global_id: float(np.dot(emb, getattr(ident, attr)))
                for ident in self._gallery
            }
            if max(cos, key=cos.get) in confirmed_gids:
                self._confirm_abstain[b.sam3_obj_id] = "gid_taken"
                return
        self._confirm_abstain[b.sam3_obj_id] = proximate

    def _locality_ok(self, b: TrackBinding, gid: int, *, cost_path: str | None = None) -> bool:
        # Reject a (re)bind when the gid was seen <= max_stale_frames ago but the
        # target box is > max_jump_px from that last-seen centre — a gid can't
        # teleport across the frame in a few frames. Gids unseen for longer are
        # genuine re-entries and are not gated — EXCEPT a body-only (OSNet) rebind
        # whose implied speed is physically impossible: a same-build body just
        # past the stale window steals the gid (the seat->across-room teleport
        # that orphans a window). Face rebinds are trusted (identity matched), so
        # only the body path carries the speed cap.
        seen = self._gid_last_seen.get(gid)
        if seen is None or b.last_bbox is None:
            return True
        last_frame, (lx, ly) = seen
        bx = b.last_bbox
        cx, cy = (bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0
        dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
        gap = self._current_frame_idx - last_frame
        if gap > self._locality_max_stale_frames:
            if (
                cost_path == "osnet"
                and dist > self._locality_max_jump_px
                and dist / max(1, gap) > self._locality_max_speed_px
            ):
                return False
            return True
        return dist <= self._locality_max_jump_px

    def _osnet_match_ambiguous(self, b: TrackBinding, gid: int) -> bool:
        emb = b.osnet_emb_running
        if emb is None or len(self._gallery) == 0:
            return False
        cos = {
            ident.global_id: float(np.dot(emb, ident.appearance_centroid))
            for ident in self._gallery
        }
        my_cos = cos.get(gid, -float("inf"))
        best_other = max(
            (c for g, c in cos.items() if g != gid), default=-float("inf"),
        )
        return my_cos - best_other < _OSNET_TIE_MARGIN

    def _maybe_write_warm_face(
        self,
        sampled: list[tuple[TrackBinding, np.ndarray]],
        sam3_rows: list[dict],
        frame_idx: int,
        *,
        prev_bbox_by_obj_id: dict[int, tuple],
        frame_height: int,
    ) -> None:
        if not sampled:
            return
        score_by_obj_id = {
            int(r["sam3_obj_id"]): float(r.get("score", 0.0)) for r in sam3_rows
        }
        bbox_by_obj_id = {
            int(r["sam3_obj_id"]): tuple(float(v) for v in r["bbox"])
            for r in sam3_rows
        }
        for b, new_face in sampled:
            if not b.confirmed or b.global_id < 0:
                continue
            identity = self._gallery.get(b.global_id)
            if identity is None:
                continue
            matched_cosine = float(np.dot(new_face, identity.face_centroid))
            self._warm_writer.maybe_write_face(
                binding=b, gid=b.global_id, face_emb=new_face,
                det_score=score_by_obj_id.get(b.sam3_obj_id, 0.0),
                matched_cosine=matched_cosine, frame_idx=frame_idx,
                mask_sanity=self._mask_sanity_for(
                    b, bbox_by_obj_id, prev_bbox_by_obj_id, frame_height,
                ),
            )

    def _emit_pool_stats(self, *, closing_chunk_id: int) -> None:
        if self._confidence_log is None or self._warm_gallery is None:
            return
        if not hasattr(self._confidence_log, "log_pool_stats"):
            return
        last_seen = max(
            (b.last_seen_frame_idx for b in self._tracks.values()),
            default=0,
        )
        for gid, per_gid in self._warm_gallery.stats().items():
            if per_gid["face_pool_size"] == 0 and per_gid["osnet_pool_size"] == 0:
                continue
            self._confidence_log.log_pool_stats(
                gid=int(gid),
                chunk_id=int(closing_chunk_id),
                frame_idx=int(last_seen if last_seen >= 0 else 0),
                face_pool_size=int(per_gid["face_pool_size"]),
                osnet_pool_size=int(per_gid["osnet_pool_size"]),
                face_anchor_count=int(per_gid["face_anchor_count"]),
                osnet_anchor_count=int(per_gid["osnet_anchor_count"]),
                face_intra_pool_mean_cosine=per_gid["face_intra_pool_mean_cosine"],
                osnet_intra_pool_mean_cosine=per_gid["osnet_intra_pool_mean_cosine"],
            )

    def _emit_rebind_event(self, binding: TrackBinding, info: AssignmentInfo) -> None:
        if self._confidence_log is None or self._warm_gallery is None:
            return
        winning_source = (
            ("warm_" if info.warm_won else "kpl_") + info.cost_path
        )
        last_seen = binding.last_seen_frame_idx if binding.last_seen_frame_idx >= 0 else 0
        self._confidence_log.log_rebind(
            gid=int(binding.global_id),
            track_id=int(binding.track_id),
            frame_idx=int(last_seen),
            winning_source=winning_source,
            winning_cosine=float(info.assigned_sim),
            runner_up_cosine=info.runner_up_cosine,
            warm_pool_size=info.warm_pool_size,
            warm_within_second=info.warm_within_second,
        )

    def _maybe_write_warm_osnet(
        self,
        sampled: list[tuple[TrackBinding, np.ndarray]],
        frame_idx: int,
        *,
        sam3_rows: list[dict],
        prev_bbox_by_obj_id: dict[int, tuple],
        frame_height: int,
    ) -> None:
        if not sampled:
            return
        bbox_by_obj_id = {
            int(r["sam3_obj_id"]): tuple(float(v) for v in r["bbox"])
            for r in sam3_rows
        }
        for b, new_osnet in sampled:
            if not b.confirmed or b.global_id < 0:
                continue
            self._warm_writer.maybe_write_osnet(
                binding=b, gid=b.global_id, osnet_emb=new_osnet,
                frame_idx=frame_idx,
                mask_sanity=self._mask_sanity_for(
                    b, bbox_by_obj_id, prev_bbox_by_obj_id, frame_height,
                ),
            )

    def _mask_sanity_for(
        self,
        binding: TrackBinding,
        bbox_by_obj_id: dict[int, tuple],
        prev_bbox_by_obj_id: dict[int, tuple],
        frame_height: int,
    ) -> MaskSanityInputs | None:
        if not binding.operator_assigned:
            return None
        current = bbox_by_obj_id.get(binding.sam3_obj_id)
        if current is None:
            return None
        others = [
            bb for oid, bb in bbox_by_obj_id.items()
            if oid != binding.sam3_obj_id
        ]
        return MaskSanityInputs(
            current_bbox=current,
            prev_bbox=prev_bbox_by_obj_id.get(binding.sam3_obj_id),
            other_live_bboxes=others,
            frame_height=frame_height,
        )

    def _enrich(
        self, row: dict, binding: TrackBinding, face_visible_recently: bool = True
    ) -> dict:
        out = dict(row)
        out["track_id"] = binding.track_id
        out["sam3_obj_id"] = binding.sam3_obj_id
        out["global_id"] = binding.global_id
        ident = (
            self._gallery.get(binding.global_id)
            if binding.global_id >= 0 else None
        )
        out["prompt"] = ident.prompt if ident is not None else ""
        out["name"] = ident.name if ident is not None else ""
        out["mask_source"] = "detection"
        out["assignment_info"] = self._synthesise_assignment_info(binding)
        out["face_emb_present"] = binding.face_emb_running is not None
        # Debounced per-frame face visibility for the skeleton face-gate: a face
        # was sampled within the last few frames (orientation signal). NOT
        # face_emb_present (stays True for any track that ever saw a face) and
        # NOT raw this-frame sampling (K-cadence would dilute it to ~1/K).
        out["face_visible_recently"] = face_visible_recently
        out["operator_assigned"] = binding.operator_assigned
        return out

    def _synthesise_assignment_info(
        self, binding: TrackBinding
    ) -> Optional[AssignmentInfo]:
        # Per-frame info from the running mean. Demoted tracks emit None so
        # AnonymizationStage routes to Fallback without consulting the gate.
        if not binding.confirmed or binding.global_id < 0:
            return None
        identity = self._gallery.get(binding.global_id)
        if identity is None:
            return None
        n_identities = len(self._gallery)
        if binding.face_weight_total > 0:
            emb = binding.face_emb_running
            assert emb is not None
            assigned_sim = float(np.dot(emb, identity.face_centroid))
            # Face path: absolute-only gate, no margin.
            return AssignmentInfo(
                cost_path="face",
                assigned_sim=assigned_sim,
                second_best_sim=None,
                n_identities=n_identities,
            )
        if binding.osnet_emb_running is not None:
            emb = binding.osnet_emb_running
            assigned_sim = float(np.dot(emb, identity.appearance_centroid))
            others = [
                ident for ident in self._gallery
                if ident.global_id != binding.global_id
            ]
            second_best_sim = (
                max(float(np.dot(emb, o.appearance_centroid)) for o in others)
                if others else None
            )
            return AssignmentInfo(
                cost_path="osnet",
                assigned_sim=assigned_sim,
                second_best_sim=second_best_sim,
                n_identities=n_identities,
            )
        return None

    def enable_trace(self) -> None:
        self._trace_log = []
        self._gid_prev_by_track.clear()
        self._first_frame_by_track.clear()

    def dump_trace(self, jsonl_path, summary_path) -> None:
        if self._trace_log is None:
            raise RuntimeError("enable_trace() was never called")
        jsonl_path = Path(jsonl_path)
        summary_path = Path(summary_path)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w") as f:
            for rec in self._trace_log:
                f.write(json.dumps(rec) + "\n")

        det_rows = [r for r in self._trace_log if r["kind"] == "det"]
        n_chunks = sum(1 for r in self._trace_log if r["kind"] == "chunk_boundary")
        n_unmatched = sum(1 for r in det_rows if r["gid"] == -1)
        summary_path.write_text(
            f"resolver trace\n"
            f"  det rows         : {len(det_rows)}\n"
            f"  chunk_boundaries : {n_chunks}\n"
            f"  gid=-1 rows      : {n_unmatched}\n"
        )

    def _record_det_rows(
        self,
        sam3_rows: list[dict],
        bindings: list[TrackBinding],
        sampled_obj_ids: set[int],
        frame_idx: int,
    ) -> None:
        if self._trace_log is None:
            return
        identities = list(self._gallery)
        face_centroids = {ide.global_id: ide.face_centroid for ide in identities}
        appearance_centroids = {
            ide.global_id: ide.appearance_centroid for ide in identities
        }
        for row, b in zip(sam3_rows, bindings):
            tid = b.track_id
            self._first_frame_by_track.setdefault(tid, frame_idx)
            gid_prev = self._gid_prev_by_track.get(tid)
            gid = int(b.global_id)
            self._gid_prev_by_track[tid] = gid

            face_running = b.face_emb_running
            face_cos = (
                {gid_k: float(np.dot(face_running, c))
                 for gid_k, c in face_centroids.items()}
                if face_running is not None else None
            )
            osnet_running = b.osnet_emb_running
            osnet_cos = (
                {gid_k: float(np.dot(osnet_running, c))
                 for gid_k, c in appearance_centroids.items()}
                if osnet_running is not None else None
            )

            info = self._synthesise_assignment_info(b)
            assignment = (
                {
                    "cost_path": info.cost_path,
                    "assigned_sim": float(info.assigned_sim),
                    "second_best_sim": (
                        float(info.second_best_sim)
                        if info.second_best_sim is not None else None
                    ),
                    "n_identities": int(info.n_identities),
                }
                if info is not None else None
            )

            bbox = row.get("bbox")
            self._trace_log.append({
                "kind": "det",
                "frame_idx": int(frame_idx),
                "is_skip": None,
                "stage1_locked": None,
                "track_id": int(tid),
                "track_age": int(frame_idx - self._first_frame_by_track[tid]),
                "sam3_obj_id": int(b.sam3_obj_id),
                "chunk_id": int(b.chunk_id),
                "bbox": [float(v) for v in bbox] if bbox is not None else None,
                "det_score": float(row.get("score", 0.0)),
                "gid": gid,
                "gid_prev": int(gid_prev) if gid_prev is not None else None,
                "gid_changed": gid_prev is not None and gid_prev != gid,
                "had_face_this_frame": b.sam3_obj_id in sampled_obj_ids,
                "face_emb_present": face_running is not None,
                "face_weight_total": float(b.face_weight_total),
                "osnet_weight_total": float(b.osnet_weight_total),
                "face_cos_per_gid": face_cos,
                "osnet_cos_per_gid": osnet_cos,
                "assignment": assignment,
                "low_confidence_streak": int(b.low_confidence_streak),
                "confirm_abstain": self._confirm_abstain.get(int(b.sam3_obj_id)),
            })

    def _record_chunk_boundary(self, frame_idx: int) -> None:
        if self._trace_log is None:
            return
        self._trace_log.append({
            "kind": "chunk_boundary",
            "frame_idx": int(frame_idx),
            "chunk_id": int(self._chunk_id),
        })


def _crop(frame_rgb: np.ndarray, bbox, remove_mask: np.ndarray | None = None) -> np.ndarray:
    x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
    h, w = frame_rgb.shape[:2]
    x1 = max(0, min(x1, w)); x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h)); y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=frame_rgb.dtype)
    crop = frame_rgb[y1:y2, x1:x2]
    # Zero only pixels owned by an overlapping neighbour, never this person's
    # own silhouette or the surrounding background — keeps a neighbour's face
    # out of the crop while leaving the own (possibly outside-mask) face intact.
    if remove_mask is not None and remove_mask.shape[:2] == (h, w):
        crop = crop.copy()
        crop[remove_mask[y1:y2, x1:x2].astype(bool)] = 0
    return crop
