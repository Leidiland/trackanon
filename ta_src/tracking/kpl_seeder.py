from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from ta_src.tracking.kpl_centroid_cache import (
    KPLCentroidCache,
    folder_content_key,
)


log = logging.getLogger(__name__)


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
    # folder), retained alongside the averaged centroid for per-photo
    # identity checks.
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
        centroid_cache: Optional[KPLCentroidCache] = None,
    ):
        self.kpl_root = Path(kpl_root)
        self._face_extract = face_extract
        self._osnet_extract = osnet_extract
        self._captioner = captioner
        self._image_extensions = tuple(e.lower() for e in image_extensions)
        # Cache hit → skip SAM3 / InsightFace / OSNet re-extraction. Miss →
        # recompute and overwrite. Per-folder content hash keys invalidate the
        # entry automatically when the operator edits photos.
        self._centroid_cache = centroid_cache

    def seed(self) -> list[KPLSeed]:
        if not self.kpl_root.is_dir():
            raise RuntimeError(
                f"KPL root {self.kpl_root} is missing; pipeline aborts at startup"
            )
        sub_folders = sorted(p for p in self.kpl_root.iterdir() if p.is_dir())
        if not sub_folders:
            raise RuntimeError(
                f"KPL root {self.kpl_root} contains no person sub-folders; "
                f"pipeline aborts at startup"
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

        content_key = (
            folder_content_key(folder, self._image_extensions)
            if self._centroid_cache is not None else None
        )
        cached = (
            self._centroid_cache.load(folder.name, content_key)
            if self._centroid_cache is not None and content_key is not None
            else None
        )

        if cached is not None:
            # Cache hit: centroids + per-photo face embs land directly. We
            # still need a representative image and path for the captioner;
            # both can be recovered by re-reading just the first
            # face-detectable photo.
            first_face_image, first_face_path = self._first_face_image(image_paths)
            if first_face_image is None:
                # Photos changed since the cache was written but the content
                # hash matched anyway (extremely unlikely — fall through to
                # recompute path, which will raise the no-face error).
                cached = None
            else:
                prompt = self._caption(first_face_image)
                log.info("KPL cache hit: %s", folder.name)
                return KPLSeed(
                    global_id=global_id,
                    name=folder.name,
                    face_centroid=cached["face_centroid"],
                    appearance_centroid=cached["appearance_centroid"],
                    prompt=prompt,
                    representative_path=first_face_path,
                    face_embeddings=cached["face_embeddings"],
                )

        face_embeddings: list[np.ndarray] = []
        appearance_embeddings: list[np.ndarray] = []
        first_face_image: np.ndarray | None = None
        first_face_path: Path | None = None

        for image_path in image_paths:
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                continue
            # Normalise to RGB at the seeder seam so every downstream consumer
            # (FaceIDWrapper, OSNet, captioner) sees the same convention as the
            # runtime frame source — channel-swapped centroids will not align
            # with the runtime path's embeddings.
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
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
        prompt = self._caption(first_face_image)

        if self._centroid_cache is not None and content_key is not None:
            self._centroid_cache.save(
                name=folder.name,
                content_key=content_key,
                face_centroid=face_centroid,
                appearance_centroid=appearance_centroid,
                face_embeddings=face_embeddings,
            )

        return KPLSeed(
            global_id=global_id,
            name=folder.name,
            face_centroid=face_centroid,
            appearance_centroid=appearance_centroid,
            prompt=prompt,
            representative_path=first_face_path,
            face_embeddings=face_embeddings,
        )

    def _first_face_image(
        self, image_paths: list[Path],
    ) -> tuple[Optional[np.ndarray], Optional[Path]]:
        for path in image_paths:
            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if self._face_extract(img) is not None:
                return img, path
        return None, None

    def _caption(self, image_rgb: np.ndarray) -> str:
        h, w = image_rgb.shape[:2]
        whole_mask = np.ones((h, w), dtype=bool)
        return self._captioner.describe(image_rgb, whole_mask)
