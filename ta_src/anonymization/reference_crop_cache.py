from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


CACHE_VERSION = 2  # txt2img + InsightFace validation
_SENTINEL_NAME = ".cache_version"


class ReferenceCropCache:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._enforce_cache_version()

    def _enforce_cache_version(self) -> None:
        sentinel = self.directory / _SENTINEL_NAME
        if sentinel.exists():
            return
        # Legacy crops carry structural KPL leakage; refuse to start with them.
        legacy_pngs = list(self.directory.glob("*.png"))
        if legacy_pngs:
            raise RuntimeError(
                f"ReferenceCropCache: legacy crops found in {self.directory} "
                f"without .cache_version sentinel. Delete them before starting:\n"
                f"  rm -rf {self.directory}\n"
                f"PrewarmGenerator will re-create them synthetically at startup."
            )

    def _path(self, global_id: int) -> Path:
        return self.directory / f"{global_id}.png"

    def iter_global_ids(self) -> Iterator[int]:
        """Yield global_ids of all cached Reference Crops."""
        for p in self.directory.glob("*.png"):
            try:
                yield int(p.stem)
            except ValueError:
                continue

    def _write_sentinel(self) -> None:
        sentinel = self.directory / _SENTINEL_NAME
        if not sentinel.exists():
            sentinel.write_text(f"{CACHE_VERSION}\n")

    def load(self, global_id: int) -> np.ndarray | None:
        """Return the cached Reference Crop as an (H, W, 3) uint8 RGB array."""
        path = self._path(global_id)
        if not path.exists():
            return None
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def save_first_generation(self, global_id: int, crop_rgb: np.ndarray) -> bool:
        """Write the Reference Crop only if none exists; returns True if written."""
        final = self._path(global_id)
        if final.exists():
            return False
        bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        tmp = final.parent / f".{global_id}.tmp.png"
        ok = cv2.imwrite(str(tmp), bgr)
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed for {tmp}")
        os.replace(tmp, final)
        self._write_sentinel()
        return True

    def _prompt_path(self, global_id: int) -> Path:
        return self.directory / f"{global_id}.prompt.txt"

    def save_prompt(self, global_id: int, prompt: str) -> bool:
        """Persist the prompt used to generate this gid's Reference Crop.

        No-op when the file already exists — the on-disk prompt is treated as
        the operator's canonical version so manual edits survive re-prewarm.
        Returns True if a new file was written.
        """
        path = self._prompt_path(global_id)
        if path.exists():
            return False
        path.write_text(prompt.strip() + "\n")
        return True

    def load_prompt(self, global_id: int) -> str | None:
        """Return the persisted prompt for this gid (whitespace-trimmed) or None."""
        path = self._prompt_path(global_id)
        if not path.exists():
            return None
        return path.read_text().strip()
