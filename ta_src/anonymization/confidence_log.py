"""Per-video JSONL writer for ConfidenceGate decisions. directory=None disables."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal


class ConfidenceLog:
    def __init__(self, directory: str | Path | None):
        self._dir: Path | None = Path(directory) if directory is not None else None
        if self._dir is not None:
            self._dir.mkdir(parents=True, exist_ok=True)
        self._fp = None

    def open(self, video_name: str) -> None:
        if self._dir is None:
            return
        if self._fp is not None:
            self.close()
        path = self._dir / f"{video_name}.jsonl"
        self._fp = path.open("w", encoding="utf-8")

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def log(
        self,
        *,
        gid: int,
        assigned_sim: float,
        second_best_sim: float | None,
        confirmed: bool,
        confirmed_via: Literal["face", "osnet", "none"],
        cost_path: Literal["face", "osnet"],
        frame_idx: int,
    ) -> None:
        if self._fp is None:
            return
        row = {
            "gid": int(gid),
            "frame_idx": int(frame_idx),
            "cost_path": cost_path,
            "assigned_sim": float(assigned_sim),
            "second_best_sim": (
                float(second_best_sim) if second_best_sim is not None else None
            ),
            "confirmed": bool(confirmed),
            "confirmed_via": confirmed_via,
        }
        self._fp.write(json.dumps(row) + "\n")
        self._fp.flush()

    def log_pool_stats(
        self,
        *,
        gid: int,
        chunk_id: int,
        frame_idx: int,
        face_pool_size: int,
        osnet_pool_size: int,
        face_anchor_count: int,
        osnet_anchor_count: int,
        face_intra_pool_mean_cosine: float | None,
        osnet_intra_pool_mean_cosine: float | None,
    ) -> None:
        if self._fp is None:
            return
        row = {
            "kind": "pool_stats",
            "gid": int(gid),
            "chunk_id": int(chunk_id),
            "frame_idx": int(frame_idx),
            "face_pool_size": int(face_pool_size),
            "osnet_pool_size": int(osnet_pool_size),
            "face_anchor_count": int(face_anchor_count),
            "osnet_anchor_count": int(osnet_anchor_count),
            "face_intra_pool_mean_cosine": (
                float(face_intra_pool_mean_cosine)
                if face_intra_pool_mean_cosine is not None else None
            ),
            "osnet_intra_pool_mean_cosine": (
                float(osnet_intra_pool_mean_cosine)
                if osnet_intra_pool_mean_cosine is not None else None
            ),
        }
        self._fp.write(json.dumps(row) + "\n")
        self._fp.flush()

    def log_warm_write_blocked(
        self,
        *,
        gid: int,
        kind: Literal["face", "osnet"],
        reason: Literal["iou_overlap", "bbox_jump"],
        frame_idx: int,
    ) -> None:
        if self._fp is None:
            return
        row = {
            "kind": "warm_write_blocked",
            "gid": int(gid),
            "pool_kind": kind,
            "reason": reason,
            "frame_idx": int(frame_idx),
        }
        self._fp.write(json.dumps(row) + "\n")
        self._fp.flush()

    def log_rebind(
        self,
        *,
        gid: int,
        track_id: int,
        frame_idx: int,
        winning_source: Literal[
            "kpl_face", "warm_face", "kpl_osnet", "warm_osnet"
        ],
        winning_cosine: float,
        runner_up_cosine: float | None,
    ) -> None:
        if self._fp is None:
            return
        row = {
            "kind": "rebind",
            "gid": int(gid),
            "track_id": int(track_id),
            "frame_idx": int(frame_idx),
            "winning_source": winning_source,
            "winning_cosine": float(winning_cosine),
            "runner_up_cosine": (
                float(runner_up_cosine) if runner_up_cosine is not None else None
            ),
        }
        self._fp.write(json.dumps(row) + "\n")
        self._fp.flush()
