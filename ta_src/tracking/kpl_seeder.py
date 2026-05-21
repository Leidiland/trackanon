from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass
class KPLSeed:
    global_id: int
    name: str  # KPL sub-folder name; used for human-readable vis labels
    face_centroid: np.ndarray
    appearance_centroid: np.ndarray
    prompt: str
    representative_path: Path
    # Per-photo face embeddings (one per face-detectable image in the
    # folder). Used by the img2img prewarm anti-leak gate, which rejects
    # candidates whose face resembles ANY single source photo, not just
    # the (potentially fictional) centroid.
    face_embeddings: list[np.ndarray] = None  # type: ignore[assignment]


def _l2_renormalise(vectors: list[np.ndarray]) -> np.ndarray:
    mean = np.mean(np.stack(vectors, axis=0), axis=0)
    return (mean / np.linalg.norm(mean)).astype(np.float32)


class KPLSeeder:
    def __init__(
        self,
        kpl_root: str | Path,
        face_extract: Callable[[np.ndarray], np.ndarray | None],
        osnet_extract: Callable[[np.ndarray], np.ndarray],
        captioner: Any,
        image_extensions: tuple[str, ...] = _IMAGE_EXTENSIONS,
    ):
        self.kpl_root = Path(kpl_root)
        self._face_extract = face_extract
        self._osnet_extract = osnet_extract
        self._captioner = captioner
        self._image_extensions = tuple(e.lower() for e in image_extensions)

    def seed(self) -> list[KPLSeed]:
        if not self.kpl_root.is_dir():
            raise RuntimeError(
                f"KPL root {self.kpl_root} is missing; pipeline aborts at startup (ADR-0003)"
            )
        sub_folders = sorted(p for p in self.kpl_root.iterdir() if p.is_dir())
        if not sub_folders:
            raise RuntimeError(
                f"KPL root {self.kpl_root} contains no person sub-folders; "
                f"pipeline aborts at startup (ADR-0003)"
            )
        records: list[KPLSeed] = []
        for gid, folder in enumerate(sub_folders):
            records.append(self._seed_one(gid, folder))
        return records

    def _seed_one(self, global_id: int, folder: Path) -> KPLSeed:
        image_paths = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in self._image_extensions
        )
        if not image_paths:
            raise RuntimeError(
                f"KPL sub-folder {folder} contains no images; pipeline aborts at startup"
            )

        face_embeddings: list[np.ndarray] = []
        appearance_embeddings: list[np.ndarray] = []
        first_face_image: np.ndarray | None = None
        first_face_path: Path | None = None

        for image_path in image_paths:
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            appearance_embeddings.append(self._osnet_extract(img))
            face_emb = self._face_extract(img)
            if face_emb is not None:
                face_embeddings.append(face_emb)
                if first_face_image is None:
                    first_face_image = img
                    first_face_path = image_path

        if not face_embeddings:
            raise RuntimeError(
                f"KPL sub-folder {folder} yielded zero face-detectable images; "
                f"every Person folder must have at least one face-detectable image"
            )
        face_centroid = _l2_renormalise(face_embeddings)
        appearance_centroid = _l2_renormalise(appearance_embeddings)
        h, w = first_face_image.shape[:2]
        whole_mask = np.ones((h, w), dtype=bool)
        first_face_rgb = cv2.cvtColor(first_face_image, cv2.COLOR_BGR2RGB)
        prompt = self._captioner.describe(first_face_rgb, whole_mask)

        return KPLSeed(
            global_id=global_id,
            name=folder.name,
            face_centroid=face_centroid,
            appearance_centroid=appearance_centroid,
            prompt=prompt,
            representative_path=first_face_path,
            face_embeddings=face_embeddings,
        )
