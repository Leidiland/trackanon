"""Pure-function gate: decides whether a Hungarian match is confident enough to confirm a gid."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ta_src.tracking.hungarian_assigner import AssignmentInfo


@dataclass(frozen=True)
class ConfidenceThresholds:
    face_sim_floor: float
    osnet_abs_floor: float
    osnet_margin_floor: float


@dataclass(frozen=True)
class GateDecision:
    confirmed: bool
    confirmed_via: Literal["face", "osnet", "none"]


def evaluate(info: AssignmentInfo, thr: ConfidenceThresholds) -> GateDecision:
    if info.cost_path == "face":
        if info.assigned_sim >= thr.face_sim_floor:
            return GateDecision(confirmed=True, confirmed_via="face")
        return GateDecision(confirmed=False, confirmed_via="none")

    # OSNet path: absolute floor first.
    if info.assigned_sim < thr.osnet_abs_floor:
        return GateDecision(confirmed=False, confirmed_via="none")

    # N=1 fallback: margin is undefined → absolute-only.
    if info.n_identities <= 1 or info.second_best_sim is None:
        return GateDecision(confirmed=True, confirmed_via="osnet")

    margin = info.assigned_sim - info.second_best_sim
    if margin >= thr.osnet_margin_floor:
        return GateDecision(confirmed=True, confirmed_via="osnet")
    return GateDecision(confirmed=False, confirmed_via="none")
