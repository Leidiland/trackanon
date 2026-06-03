from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    face_emb: np.ndarray | None
    osnet_emb: np.ndarray
    # No in-chunk prior + no snapshot inheritance: bbox carries no signal,
    # so no-face cost drops the spatial blend and decides on appearance alone.
    long_gap: bool = False
    # InsightFace's per-detection confidence in the face it produced for this
    # body crop. Rows with face_det_score < HungarianAssigner.face_quality_floor
    # drop the face term and decide on OSNet + spatial only. 0.0 is
    # the safe default: with floor=0.0 (today's behaviour) the gate never fires.
    face_det_score: float = 0.0
    # gids this binding was demoted from this chunk. Hungarian masks these
    # pairs to infeasible cost so a blocked gid never wins the assignment
    # only to be silently dropped by the resolver's post-filter.
    blocked_gids: set[int] = field(default_factory=set)


@dataclass
class Candidate:
    global_id: int
    face_centroid: np.ndarray
    appearance_centroid: np.ndarray
    kalman: Any  # KalmanBox or None


@dataclass(frozen=True)
class AssignmentInfo:
    """Per-row evidence for the ConfidenceGate. Similarities are raw cosine."""

    cost_path: Literal["face", "osnet"]
    assigned_sim: float
    second_best_sim: float | None
    n_identities: int
    # Telemetry surface: max similarity over the other N-1 candidates on the
    # same branch, and whether the warm pool (not KPL) contributed the
    # winning similarity. Set for both branches so the resolver can emit
    # uniform rebind events.
    runner_up_cosine: float | None = None
    warm_won: bool = False
    # Warm-pool diagnostics for the body branch: how many face-vouched views
    # backed the match, and the within-pool runner-up cosine (top1 >> this
    # means a single stored view carried a max-pooled match).
    warm_pool_size: int = 0
    warm_within_second: float | None = None
    # Face path only: the KPL-centroid cosine before warm augmentation. Lets a
    # caller require the trusted studio anchor — not just max(KPL, warm) — to
    # clear a floor, so warm can't single-handedly lock a fresh identity.
    kpl_sim: float | None = None


_INFEASIBLE_COST = 1e6


class HungarianAssigner:
    def __init__(
        self,
        emb_weight: float = 0.8,
        spatial_scale: float = 300.0,
        warm_match_floor: float = 0.5,
        face_quality_floor: float = 0.0,
        osnet_confirm_warm_only: bool = False,
        osnet_confirm_min_warm: int = 0,
    ):
        self.emb_weight = emb_weight
        self.spatial_scale = spatial_scale
        self.warm_match_floor = warm_match_floor
        self.face_quality_floor = float(face_quality_floor)
        # Body re-id off the face-vouched warm pool only — the KPL-seed
        # appearance centroid carries reference-photo clothing that mismatches
        # in-video appearance and drives look-alike swaps.
        self.osnet_confirm_warm_only = bool(osnet_confirm_warm_only)
        self.osnet_confirm_min_warm = int(osnet_confirm_min_warm)

    def assign(
        self,
        detections: list[Detection],
        candidates: list[Candidate],
        warm=None,
    ) -> tuple[list[tuple[int, int]], list[int], list[AssignmentInfo]]:
        n_det = len(detections)
        n_cand = len(candidates)
        if n_det == 0 or n_cand == 0:
            return [], list(range(n_det)), []

        cost = np.zeros((n_det, n_cand), dtype=np.float64)
        # mask[i, j] = True when warm strictly beats KPL but warm_cos is below
        # warm_match_floor — Hungarian must not commit to that pair.
        mask = np.zeros((n_det, n_cand), dtype=bool)
        # Raw cosine kept separately so ConfidenceGate doesn't see blended cost.
        # Warm augmentation upgrades these to max(kpl, warm) when warm wins —
        # the gate must see the same per-pair similarity the assigner saw.
        face_cos = np.full((n_det, n_cand), np.nan, dtype=np.float64)
        # KPL-only face cosine (pre-warm-augmentation), surfaced on the matched
        # face-path row so the resolver's KPL-agreement guard can read it.
        kpl_face_cos = np.full((n_det, n_cand), np.nan, dtype=np.float64)
        osnet_cos = np.zeros((n_det, n_cand), dtype=np.float64)
        warm_won_face = np.zeros((n_det, n_cand), dtype=bool)
        warm_won_osnet = np.zeros((n_det, n_cand), dtype=bool)
        for i, det in enumerate(detections):
            face_usable = (
                det.face_emb is not None
                and det.face_det_score >= self.face_quality_floor
            )
            for j, cand in enumerate(candidates):
                if cand.global_id in det.blocked_gids:
                    cost[i, j] = _INFEASIBLE_COST
                    mask[i, j] = True
                    continue
                pair_cost, pair_masked = self._pair_cost(det, cand, warm, face_usable)
                cost[i, j] = pair_cost
                mask[i, j] = pair_masked
                aug_osnet, _osnet_feasible, osnet_warm = self._osnet_sim(
                    det, cand, warm)
                osnet_cos[i, j] = aug_osnet
                warm_won_osnet[i, j] = osnet_warm
                if face_usable:
                    kpl_face = float(np.dot(det.face_emb, cand.face_centroid))
                    aug_face, face_warm = _augment_and_source(
                        kpl_face, warm, cand.global_id, "face", det.face_emb,
                    )
                    face_cos[i, j] = aug_face
                    kpl_face_cos[i, j] = kpl_face
                    warm_won_face[i, j] = face_warm

        row_idx, col_idx = linear_sum_assignment(cost)
        matched: list[tuple[int, int]] = []
        infos: list[AssignmentInfo] = []
        assigned_rows: set[int] = set()
        for r, c in zip(row_idx, col_idx):
            r, c = int(r), int(c)
            if mask[r, c]:
                continue
            matched.append((r, candidates[c].global_id))
            assigned_rows.add(r)
            if not np.isnan(face_cos[r, c]):
                # Face path: absolute-only gate. Runner-up is computed on the
                # same branch (max face_cos over the other N-1) for telemetry.
                if n_cand <= 1:
                    runner_up = None
                else:
                    row = face_cos[r].copy()
                    row[c] = -np.inf
                    runner_up = float(row.max())
                infos.append(AssignmentInfo(
                    cost_path="face",
                    assigned_sim=float(face_cos[r, c]),
                    second_best_sim=None,
                    n_identities=n_cand,
                    runner_up_cosine=runner_up,
                    warm_won=bool(warm_won_face[r, c]),
                    kpl_sim=float(kpl_face_cos[r, c]),
                ))
            else:
                # OSNet path: second_best_sim = max cosine over the other N-1 candidates.
                if n_cand <= 1:
                    second_best = None
                else:
                    row = osnet_cos[r].copy()
                    row[c] = -np.inf
                    second_best = float(row.max())
                pool_size, within_second = self._warm_osnet_diagnostics(
                    detections[r], candidates[c].global_id, warm)
                infos.append(AssignmentInfo(
                    cost_path="osnet",
                    assigned_sim=float(osnet_cos[r, c]),
                    second_best_sim=second_best,
                    n_identities=n_cand,
                    runner_up_cosine=second_best,
                    warm_won=bool(warm_won_osnet[r, c]),
                    warm_pool_size=pool_size,
                    warm_within_second=within_second,
                ))

        excess = [i for i in range(n_det) if i not in assigned_rows]
        return matched, excess, infos

    def _pair_cost(
        self, det: Detection, cand: Candidate, warm=None, face_usable: bool = True,
    ) -> tuple[float, bool]:
        """Returns (cost, masked). When warm strictly beats KPL but is below
        the absolute-cosine floor, masked=True signals the Hungarian must not
        commit to this pair — the downstream assignment dispatch drops the row
        rather than confirming on weak warm evidence.

        face_usable=False forces this pair onto the OSNet+spatial branch
        regardless of whether det.face_emb is populated — the face-quality
        gate drops the face term proactively for low-confidence
        InsightFace detections (profile views, motion blur)."""
        if face_usable and det.face_emb is not None:
            kpl_cos = float(np.dot(det.face_emb, cand.face_centroid))
            if warm is not None:
                warm_cos = warm.best_similarity(
                    cand.global_id, "face", det.face_emb,
                )
                if warm_cos is not None and warm_cos > kpl_cos:
                    if warm_cos < self.warm_match_floor:
                        return _INFEASIBLE_COST, True
                    return 1.0 - warm_cos, False
            return 1.0 - kpl_cos, False

        osnet_sim, feasible, _warm_won = self._osnet_sim(det, cand, warm)
        if not feasible:
            return _INFEASIBLE_COST, True
        osnet_cost = 1.0 - osnet_sim
        if cand.kalman is None or det.long_gap:
            return osnet_cost, False

        spatial_cost = self._spatial_cost(det.box, cand.kalman)
        return (
            self.emb_weight * osnet_cost + (1.0 - self.emb_weight) * spatial_cost,
            False,
        )

    def _osnet_sim(
        self, det: Detection, cand: Candidate, warm,
    ) -> tuple[float, bool, bool]:
        """Returns (cosine, feasible, warm_won) for the body branch.

        In warm-only mode the KPL-seed appearance centroid is ignored entirely:
        the body match scores against the warm pool, feasible only once it holds
        >= osnet_confirm_min_warm face-vouched views. Otherwise the legacy
        max(kpl, warm) blend stands; a warm win below warm_match_floor masks."""
        if self.osnet_confirm_warm_only:
            if warm is None or warm.size(
                    cand.global_id, "osnet") < self.osnet_confirm_min_warm:
                return 0.0, False, False
            warm_cos = warm.best_similarity(cand.global_id, "osnet", det.osnet_emb)
            if warm_cos is None:
                return 0.0, False, False
            return float(warm_cos), warm_cos >= self.warm_match_floor, True
        kpl = float(np.dot(det.osnet_emb, cand.appearance_centroid))
        if warm is not None:
            warm_cos = warm.best_similarity(cand.global_id, "osnet", det.osnet_emb)
            if warm_cos is not None and warm_cos > kpl:
                return float(warm_cos), warm_cos >= self.warm_match_floor, True
        return kpl, True, False

    @staticmethod
    def _warm_osnet_diagnostics(
        det: Detection, gid: int, warm,
    ) -> tuple[int, float | None]:
        """(pool_size, within_pool_runner_up) for the body warm pool; zeros when
        warm is absent or doesn't expose the accessors (minimal test fakes)."""
        if warm is None:
            return 0, None
        size_fn = getattr(warm, "size", None)
        pool_size = int(size_fn(gid, "osnet")) if size_fn is not None else 0
        top_fn = getattr(warm, "top_similarities", None)
        within_second = None
        if top_fn is not None:
            top = top_fn(gid, "osnet", det.osnet_emb, 2)
            within_second = float(top[1]) if len(top) >= 2 else None
        return pool_size, within_second

    def _spatial_cost(self, det_box, kalman) -> float:
        """1 − spatial_similarity, where similarity = max(0, 1 − dist/scale)."""
        x1, y1, x2, y2 = det_box
        det_cx, det_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        kx1, ky1, kx2, ky2 = kalman.box()
        cand_cx, cand_cy = (kx1 + kx2) / 2.0, (ky1 + ky2) / 2.0
        dist = float(np.hypot(det_cx - cand_cx, det_cy - cand_cy))
        spatial_sim = max(0.0, 1.0 - dist / self.spatial_scale)
        return 1.0 - spatial_sim


def _augment_and_source(
    kpl_sim: float, warm, gid: int, kind: str, query_emb: np.ndarray,
) -> tuple[float, bool]:
    """Returns (augmented_sim, warm_won) so callers can both score and
    attribute the winning branch in one pass."""
    if warm is None:
        return kpl_sim, False
    warm_sim = warm.best_similarity(gid, kind, query_emb)
    if warm_sim is None or float(warm_sim) <= kpl_sim:
        return kpl_sim, False
    return float(warm_sim), True
