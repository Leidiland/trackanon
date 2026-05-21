"""Per-gid runtime buffer of warm face/OSNet embeddings, additive to the
frozen KPL centroids on Identity.

Storage and eviction concerns only. Holds no knowledge of TrackBinding,
Candidate, or the InsightFace / OSNet wrappers — those live one layer up
in WarmGalleryWriter and the resolver.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _Entry:
    emb: np.ndarray
    frame_idx: int
    source_cosine: float
    is_anchor: bool


class WarmIdentityGallery:
    def __init__(
        self,
        face_size: int,
        osnet_size: int,
        dedup_cosine: float = 1.01,
    ) -> None:
        self._face_size = face_size
        self._osnet_size = osnet_size
        self._dedup_cosine = dedup_cosine
        self._pools: dict[tuple[int, str], list[_Entry]] = {}

    def write(
        self, gid: int, kind: str, emb: np.ndarray,
        source_cosine: float, frame_idx: int, is_anchor: bool,
    ) -> bool:
        pool = self._pools.setdefault((gid, kind), [])
        for existing in pool:
            if float(np.dot(emb, existing.emb)) > self._dedup_cosine:
                return False
        cap = self._face_size if kind == "face" else self._osnet_size
        if len(pool) >= cap:
            evict_idx = _most_redundant_non_anchor_idx(pool)
            if evict_idx is None:
                # Pool full of anchors: drop the candidate silently rather
                # than evict protected evidence.
                return False
            pool.pop(evict_idx)
        pool.append(_Entry(emb=emb, frame_idx=frame_idx,
                           source_cosine=source_cosine, is_anchor=is_anchor))
        return True

    def size(self, gid: int, kind: str) -> int:
        return len(self._pools.get((gid, kind), []))

    def best_similarity(
        self, gid: int, kind: str, query_emb: np.ndarray,
    ) -> float | None:
        pool = self._pools.get((gid, kind))
        if not pool:
            return None
        return max(float(np.dot(query_emb, e.emb)) for e in pool)

    def stats(self) -> dict[int, dict]:
        out: dict[int, dict] = {}
        gids = {gid for (gid, _kind) in self._pools.keys()}
        for gid in gids:
            face_pool = self._pools.get((gid, "face"), [])
            osnet_pool = self._pools.get((gid, "osnet"), [])
            out[gid] = {
                "face_pool_size": len(face_pool),
                "osnet_pool_size": len(osnet_pool),
                "face_anchor_count": sum(1 for e in face_pool if e.is_anchor),
                "osnet_anchor_count": sum(1 for e in osnet_pool if e.is_anchor),
                "face_intra_pool_mean_cosine": _intra_mean_cosine(face_pool),
                "osnet_intra_pool_mean_cosine": _intra_mean_cosine(osnet_pool),
            }
        return out


def _most_redundant_non_anchor_idx(pool: list[_Entry]) -> int | None:
    """Index of the non-anchor entry whose nearest-neighbour cosine within
    the pool is highest. None if every entry is anchored."""
    best_idx: int | None = None
    best_nn_cos = -float("inf")
    for i, entry in enumerate(pool):
        if entry.is_anchor:
            continue
        nn_cos = -float("inf")
        for j, other in enumerate(pool):
            if i == j:
                continue
            sim = float(np.dot(entry.emb, other.emb))
            if sim > nn_cos:
                nn_cos = sim
        if nn_cos > best_nn_cos:
            best_nn_cos = nn_cos
            best_idx = i
    return best_idx


def _intra_mean_cosine(pool: list[_Entry]) -> float | None:
    if len(pool) < 2:
        return None
    sims: list[float] = []
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            sims.append(float(np.dot(pool[i].emb, pool[j].emb)))
    return float(np.mean(sims))
