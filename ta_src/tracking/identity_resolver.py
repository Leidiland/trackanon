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


def _module_log() -> logging.Logger:
    return logging.getLogger(__name__)

import numpy as np

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
    last_seen_frame_idx: int = -1
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
class _PartialEvidenceCarry:
    """Chunk-end snapshot — matched into next chunk by bbox IoU for transparent boundaries."""
    last_bbox: tuple[float, float, float, float]
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

        self._chunk_id: int = -1
        self._tracks: dict[int, TrackBinding] = {}  # keyed by sam3_obj_id
        # Cross-chunk state snapshots; matched by bbox IoU on the next chunk's first frame.
        self._partial_carryover: list[_PartialEvidenceCarry] = []
        self._frames_into_chunk: int = 0

        self._trace_log: list[dict] | None = None
        self._gid_prev_by_track: dict[int, int] = {}
        self._first_frame_by_track: dict[int, int] = {}
        self._chunk_boundary_pending: bool = False

    def set_confidence_log(self, log) -> None:
        """Late-wire the shared ConfidenceLog (constructed by AnonymizationStage)."""
        self._confidence_log = log
        if self._warm_writer is not None:
            self._warm_writer.set_confidence_log(log)

    def start_chunk(self, chunk_id: int) -> None:
        if self._chunk_id >= 0:
            self._emit_pool_stats(closing_chunk_id=self._chunk_id)
        if self._chunk_id >= 0 and self._tracks:
            # Snapshot most-recent-frame bindings only — mid-chunk dropouts
            # shouldn't match against similarly-positioned new tracks.
            last_frame = max(
                b.last_seen_frame_idx for b in self._tracks.values()
            )
            self._partial_carryover = [
                _PartialEvidenceCarry(
                    last_bbox=tuple(b.last_bbox),
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
        if self._chunk_boundary_pending:
            self._record_chunk_boundary(frame_idx)
            self._chunk_boundary_pending = False

        # Pass 0: revive recently-died bindings whose last_bbox aligns with a
        # new sam3_obj_id this frame. SAM 3 occasionally drops an obj_id then
        # re-emits the same person under a fresh id (memory-attention quirk);
        # transferring the live binding to the new key keeps operator_assigned
        # and confirmed gid intact without waiting for the chunk boundary.
        self._revive_died_bindings(sam3_rows, frame_idx)

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
                if self._frames_into_chunk == 0 and self._partial_carryover:
                    self._maybe_inherit_partial_evidence(
                        binding, tuple(row["bbox"])
                    )
            else:
                binding = existing
            binding.last_bbox = tuple(row["bbox"])
            binding.last_seen_frame_idx = frame_idx
            new_face = self._maybe_accumulate_face(frame_rgb, row, binding)
            if new_face is not None:
                sampled_this_frame.append((binding, new_face))
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
            self._partial_carryover = []

        # Enrich AFTER confirmation so this frame reflects new gid assignments.
        sampled_obj_ids = {b.sam3_obj_id for b, _emb in sampled_this_frame}
        self._record_det_rows(sam3_rows, bindings_this_frame, sampled_obj_ids, frame_idx)
        return [self._enrich(row, b) for row, b in zip(sam3_rows, bindings_this_frame)]

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

    def _maybe_inherit_partial_evidence(
        self,
        binding: TrackBinding,
        bbox: tuple,
    ) -> None:
        # Match new binding to a prev-chunk snapshot by bbox IoU; one-to-one transfer.
        best_iou = 0.0
        best_idx = -1
        for i, carry in enumerate(self._partial_carryover):
            iou = iou_xyxy(bbox, carry.last_bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_idx < 0 or best_iou < self._partial_carry_iou_floor:
            return
        carry = self._partial_carryover.pop(best_idx)
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

    def _maybe_accumulate_face(
        self, frame_rgb: np.ndarray, row: dict, binding: TrackBinding
    ) -> Optional[np.ndarray]:
        """Return the new face embedding sampled this call (or None)."""
        if binding.frames_seen % self._K != 0:
            return None
        bbox = row["bbox"]
        body_crop = _crop(frame_rgb, bbox)
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
        # can drop the face term for low-quality detections (ADR-0018).
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
            elif cos_assigned < self._low_conf_cos:
                b.low_confidence_streak += 1
                if b.low_confidence_streak >= self._streak_max:
                    self._demote(b)
            else:
                b.low_confidence_streak = 0

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
        free_identities = [
            ident for ident in self._gallery
            if ident.global_id not in confirmed_gids
        ]
        if not free_identities:
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
            if gid in b.blocked_gids:
                # Previously demoted from this gid in this chunk — skip.
                continue
            if not evaluate_gate(info, self._thresholds).confirmed:
                continue
            b.global_id = gid
            b.confirmed = True
            self._emit_rebind_event(b, info)

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

    def _enrich(self, row: dict, binding: TrackBinding) -> dict:
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
            })

    def _record_chunk_boundary(self, frame_idx: int) -> None:
        if self._trace_log is None:
            return
        self._trace_log.append({
            "kind": "chunk_boundary",
            "frame_idx": int(frame_idx),
            "chunk_id": int(self._chunk_id),
        })


def _crop(frame_rgb: np.ndarray, bbox) -> np.ndarray:
    x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
    h, w = frame_rgb.shape[:2]
    x1 = max(0, min(x1, w)); x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h)); y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=frame_rgb.dtype)
    return frame_rgb[y1:y2, x1:x2]
