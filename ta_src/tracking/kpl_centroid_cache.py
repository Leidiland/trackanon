"""On-disk cache for KPL-seeded face/OSNet centroids.

Each KPL sub-folder yields one cache entry keyed by a content hash over its
photos (filename + mtime + size). Hit → skip SAM 3 / InsightFace / OSNet
re-extraction; miss → recompute and overwrite. Operators editing or adding
photos invalidate the entry automatically.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np


def folder_content_key(folder: Path, extensions: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in extensions:
            continue
        stat = p.stat()
        h.update(p.name.encode("utf-8"))
        h.update(b"|")
        h.update(str(stat.st_mtime_ns).encode("ascii"))
        h.update(b"|")
        h.update(str(stat.st_size).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


class KPLCentroidCache:
    def __init__(self, cache_root: Path):
        self._root = Path(cache_root)

    def _npz_path(self, name: str) -> Path:
        return self._root / f"{name}.npz"

    def _key_path(self, name: str) -> Path:
        return self._root / f"{name}.key"

    def load(self, name: str, content_key: str) -> Optional[dict]:
        """Return {face_centroid, appearance_centroid, face_embeddings} or None
        on miss / key mismatch / unreadable cache."""
        kp = self._key_path(name)
        np_path = self._npz_path(name)
        if not kp.exists() or not np_path.exists():
            return None
        try:
            stored = kp.read_text().strip()
        except OSError:
            return None
        if stored != content_key:
            return None
        try:
            data = np.load(np_path, allow_pickle=False)
        except Exception:
            return None
        try:
            n = int(data["n_face_embeddings"])
            return {
                "face_centroid": data["face_centroid"].astype(np.float32),
                "appearance_centroid": data["appearance_centroid"].astype(np.float32),
                "face_embeddings": [
                    data[f"face_emb_{i}"].astype(np.float32) for i in range(n)
                ],
            }
        except KeyError:
            return None

    def save(
        self,
        name: str,
        content_key: str,
        face_centroid: np.ndarray,
        appearance_centroid: np.ndarray,
        face_embeddings: list[np.ndarray],
    ) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "face_centroid": np.asarray(face_centroid, dtype=np.float32),
            "appearance_centroid": np.asarray(appearance_centroid, dtype=np.float32),
            "n_face_embeddings": np.array(len(face_embeddings), dtype=np.int32),
        }
        for i, emb in enumerate(face_embeddings):
            payload[f"face_emb_{i}"] = np.asarray(emb, dtype=np.float32)
        np.savez(self._npz_path(name), **payload)
        self._key_path(name).write_text(content_key)
