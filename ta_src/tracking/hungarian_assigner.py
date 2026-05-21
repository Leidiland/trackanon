from __future__ import annotations

from dataclasses import dataclass
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
    # drop the face term and decide on OSNet + spatial only (ADR-0018). 0.0 is
    # the safe default: with floor=0.0 (today's behaviour) the gate never fires.
    face_det_score: float = 0.0


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


_INFEASIBLE_COST = 1e6


class HungarianAssigner:
    def __init__(
        self,
        emb_weight: float = 0.8,
        spatial_scale: float = 300.0,
        warm_match_floor: float = 0.5,
        face_quality_floor: float = 0.0,
    ):
        self.emb_weight = emb_weight
        self.spatial_scale = spatial_scale
        self.warm_match_floor = warm_match_floor
        self.face_quality_floor = float(face_quality_floor)

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
        osnet_cos = np.zeros((n_det, n_cand), dtype=np.float64)
        warm_won_face = np.zeros((n_det, n_cand), dtype=bool)
        warm_won_osnet = np.zeros((n_det, n_cand), dtype=bool)
        for i, det in enumerate(detections):
            face_usable = (
                det.face_emb is not None
                and det.face_det_score >= self.face_quality_floor
            )
            for j, cand in enumerate(candidates):
                pair_cost, pair_masked = self._pair_cost(det, cand, warm, face_usable)
                cost[i, j] = pair_cost
                mask[i, j] = pair_masked
                kpl_osnet = float(np.dot(det.osnet_emb, cand.appearance_centroid))
                aug_osnet, osnet_warm = _augment_and_source(
                    kpl_osnet, warm, cand.global_id, "osnet", det.osnet_emb,
                )
                osnet_cos[i, j] = aug_osnet
                warm_won_osnet[i, j] = osnet_warm
                if face_usable:
                    kpl_face = float(np.dot(det.face_emb, cand.face_centroid))
                    aug_face, face_warm = _augment_and_source(
                        kpl_face, warm, cand.global_id, "face", det.face_emb,
                    )
                    face_cos[i, j] = aug_face
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
                ))
            else:
                # OSNet path: second_best_sim = max cosine over the other N-1 candidates.
                if n_cand <= 1:
                    second_best = None
                else:
                    row = osnet_cos[r].copy()
                    row[c] = -np.inf
                    second_best = float(row.max())
                infos.append(AssignmentInfo(
                    cost_path="osnet",
                    assigned_sim=float(osnet_cos[r, c]),
                    second_best_sim=second_best,
                    n_identities=n_cand,
                    runner_up_cosine=second_best,
                    warm_won=bool(warm_won_osnet[r, c]),
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
        gate (ADR-0018) drops the face term proactively for low-confidence
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

        kpl_osnet_cos = float(np.dot(det.osnet_emb, cand.appearance_centroid))
        if warm is not None:
            warm_osnet_cos = warm.best_similarity(
                cand.global_id, "osnet", det.osnet_emb,
            )
            if warm_osnet_cos is not None and warm_osnet_cos > kpl_osnet_cos:
                if warm_osnet_cos < self.warm_match_floor:
                    return _INFEASIBLE_COST, True
                kpl_osnet_cos = warm_osnet_cos
        osnet_cost = 1.0 - kpl_osnet_cos
        if cand.kalman is None or det.long_gap:
            return osnet_cost, False

        spatial_cost = self._spatial_cost(det.box, cand.kalman)
        return (
            self.emb_weight * osnet_cost + (1.0 - self.emb_weight) * spatial_cost,
            False,
        )

    def _spatial_cost(self, det_box, kalman) -> float:
        """1 − spatial_similarity, where similarity = max(0, 1 − dist/scale)."""
        x1, y1, x2, y2 = det_box
        det_cx, det_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        kx1, ky1, kx2, ky2 = kalman.box()
        cand_cx, cand_cy = (kx1 + kx2) / 2.0, (ky1 + ky2) / 2.0
        dist = float(np.hypot(det_cx - cand_cx, det_cy - cand_cy))
        spatial_sim = max(0.0, 1.0 - dist / self.spatial_scale)
        return 1.0 - spatial_sim


def _augment_with_warm(
    kpl_sim: float, warm, gid: int, kind: str, query_emb: np.ndarray,
) -> float:
    return _augment_and_source(kpl_sim, warm, gid, kind, query_emb)[0]


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
