"""Per-frame tracking cache for the two-pass decouple.

Pass 1 (tracking) writes one file per emitted output frame; Pass 2 (diffusion)
reads it back to rebuild the FrameContext without re-running SAM3 / the resolver.
Masks dominate size — a full-frame 4K bool mask is ~8 MB — so they are RLE-encoded
(a silhouette is a few KB). Everything else is small and goes in a JSON blob.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from ta_src.tracking.hungarian_assigner import AssignmentInfo

# Tag a serialised AssignmentInfo so the reader can rebuild the dataclass — the
# confidence gate consumes it by attribute (info.cost_path, info.assigned_sim).
_AINFO_TAG = "__assignment_info__"


def _encode_value(v):
    if isinstance(v, AssignmentInfo):
        return {_AINFO_TAG: dataclasses.asdict(v)}
    return v


def _decode_value(v):
    if isinstance(v, dict) and _AINFO_TAG in v:
        return AssignmentInfo(**v[_AINFO_TAG])
    return v


def _encode_mask(mask: np.ndarray) -> tuple[list[int], np.ndarray]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return list(rle["size"]), np.frombuffer(rle["counts"], dtype=np.uint8)


def _decode_mask(size: list[int], counts: np.ndarray) -> np.ndarray:
    rle = {"size": [int(s) for s in size], "counts": counts.tobytes()}
    return mask_utils.decode(rle).astype(bool)


def _json_default(o):
    # Coerce numpy scalars that slip in from the resolver rows.
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"not JSON-serialisable: {type(o)}")


def _frame_path(cache_dir: Path, frame_idx: int) -> Path:
    return cache_dir / f"frame_{int(frame_idx):07d}.npz"


class TrackCacheWriter:
    def __init__(self, cache_dir):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def write(self, frame_idx: int, tracks: list[dict], keypoints: list[dict]) -> None:
        arrays: dict[str, np.ndarray] = {}
        meta_tracks = []
        for i, track in enumerate(tracks):
            t = {k: _encode_value(v) for k, v in track.items() if k != "mask"}
            mask = track.get("mask")
            if mask is not None:
                size, counts = _encode_mask(mask)
                arrays[f"t{i}_counts"] = counts
                t["_mask_size"] = size
            meta_tracks.append(t)

        meta_kps = []
        for i, kp in enumerate(keypoints):
            entry, arr_keys = {}, []
            for k, v in kp.items():
                if isinstance(v, np.ndarray):
                    arrays[f"kp{i}_{k}"] = v
                    arr_keys.append(k)
                else:
                    entry[k] = _encode_value(v)
            if arr_keys:
                entry["_arr"] = arr_keys
            meta_kps.append(entry)

        meta = {
            "frame_idx": int(frame_idx),
            "tracks": meta_tracks,
            "keypoints": meta_kps,
        }
        np.savez_compressed(
            _frame_path(self._dir, frame_idx),
            meta=json.dumps(meta, default=_json_default),
            **arrays,
        )


class TrackCacheReader:
    def __init__(self, cache_dir):
        self._dir = Path(cache_dir)

    def frame_indices(self) -> list[int]:
        # Sorted frame indices present on disk; the VACE window planner needs the
        # range Pass 1 cached without re-deriving the filename layout elsewhere.
        return sorted(
            int(p.stem.split("_")[1]) for p in self._dir.glob("frame_*.npz")
        )

    def read_meta(self, frame_idx: int) -> list[dict]:
        """Track metadata (global_id, bbox, assignment_info, `has_mask`) WITHOUT
        decoding the RLE masks — for window planning, which only needs presence
        and bbox. Skips the per-track decompress + pycocotools decode that makes
        a full `read` of every frame × gid the dominant cost on long clips."""
        with np.load(_frame_path(self._dir, frame_idx), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"].item()))
            tracks = []
            for raw in meta["tracks"]:
                t = {k: _decode_value(v) for k, v in raw.items() if k != "_mask_size"}
                t["has_mask"] = "_mask_size" in raw
                tracks.append(t)
        return tracks

    def read(self, frame_idx: int) -> tuple[list[dict], list[dict]]:
        with np.load(_frame_path(self._dir, frame_idx), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"].item()))
            tracks = []
            for i, raw in enumerate(meta["tracks"]):
                t = {k: _decode_value(v) for k, v in raw.items()}
                size = t.pop("_mask_size", None)
                if size is not None:
                    t["mask"] = _decode_mask(size, z[f"t{i}_counts"])
                tracks.append(t)

            keypoints = []
            for i, raw in enumerate(meta.get("keypoints", [])):
                entry = {k: _decode_value(v) for k, v in raw.items() if k != "_arr"}
                for k in raw.get("_arr", []):
                    entry[k] = z[f"kp{i}_{k}"]
                keypoints.append(entry)
        return tracks, keypoints
