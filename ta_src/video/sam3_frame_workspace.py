"""One SAM 3 chunk's temp JPEG directory.

A workspace owns a fresh subdir of `parent_dir`, materialises a chunk of RGB
frames into it as sequentially-named JPEGs (00000.jpg, 00001.jpg, ...), and
exposes that path for SAM 3's start_session(resource_path=...). Closing the
workspace deletes the subdir; the parent is untouched.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np


class Sam3FrameWorkspace:
    def __init__(self, parent_dir: Path, chunk_id: int) -> None:
        self._path = Path(parent_dir) / f"chunk_{chunk_id:05d}"
        self._path.mkdir(parents=True, exist_ok=False)

    @property
    def path(self) -> Path:
        return self._path

    def write_frames(self, frames: list[np.ndarray]) -> None:
        for i, frame_rgb in enumerate(frames):
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(self._path / f"{i:05d}.jpg"), bgr)

    def close(self) -> None:
        shutil.rmtree(self._path, ignore_errors=True)

    def __enter__(self) -> "Sam3FrameWorkspace":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
