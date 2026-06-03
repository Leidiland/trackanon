"""Disk-backed frame stores so Pass-2 VACE stitching stays O(crop) in RAM
rather than O(whole 4K clip × gids).

- DiskFrameProvider: source frames decoded once to disk and served by index on
  demand, so the full decoded clip never sits in RAM at once.
- StitchStore: one gid's stitched silhouette per frame, kept to the bounding box
  of the region that differs from source (mask + feather band). The cross-gid
  compositor reconstructs full frames on the fly and streams the output video
  frame by frame instead of holding every gid's full-frame render at once.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class DiskFrameProvider:
    """Random-access source frames backed by per-index .npy files. Callable so
    it drops in where the pipeline previously passed `lambda k: frames[k]`."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._n = 0

    def _path(self, k: int) -> Path:
        return self.root / f"{k:07d}.npy"

    def populate(self, frames_iter) -> int:
        n = 0
        for f in frames_iter:
            np.save(self._path(n), f)
            n += 1
        self._n = n
        (self.root / "_count").write_text(str(n))   # completion marker for reuse
        return n

    def existing_count(self) -> int:
        """Frame count of a fully-populated cache from a prior run, else 0.
        Lets a re-run reuse an already-decoded source instead of re-decoding."""
        p = self.root / "_count"
        if not p.exists():
            return 0
        try:
            n = int(p.read_text().strip())
        except ValueError:
            return 0
        self._n = n
        return n

    def __len__(self) -> int:
        return self._n

    def __call__(self, k: int) -> np.ndarray:
        # The encoder can drop the trailing frame, leaving the cache one short
        # of the nominal range; clamp so a tail window reuses the last frame
        # instead of crashing the whole render on a missing index.
        if self._n and k >= self._n:
            k = self._n - 1
        return np.load(self._path(k))


class StitchStore:
    """Per-frame stitched silhouette for one gid. Stores only the bounding box
    of the region that differs from source (so disk/RAM stay O(crop)); the full
    stitched frame is `source` with that box overwritten. `painted_frames` are
    the frames that actually carry persona pixels (non-empty box)."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.frames: set[int] = set()
        self.painted_frames: set[int] = set()

    def _path(self, k: int) -> Path:
        return self.root / f"{k:07d}.npz"

    def put(self, k: int, stitched_rgb, source_rgb, painted_mask) -> None:
        painted_mask = np.asarray(painted_mask).astype(bool)
        # Union the colour diff with the hard mask so the box never drops a
        # painted pixel that happens to equal source, and always spans the
        # feather band (where stitched differs from source past the mask).
        diff = np.any(stitched_rgb != source_rgb, axis=2) | painted_mask
        ys, xs = np.where(diff)
        if ys.size:
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            self.painted_frames.add(k)
        else:
            y0 = x0 = y1 = x1 = 0
        np.savez(
            self._path(k),
            bbox=np.array([y0, x0, y1, x1], np.int64),
            rgb=np.ascontiguousarray(stitched_rgb[y0:y1, x0:x1]),
            mask=np.ascontiguousarray(painted_mask[y0:y1, x0:x1]),
        )
        self.frames.add(k)

    def put_crop(self, k: int, bbox, rgb_sub, mask_sub) -> None:
        """Store an already-cropped stitch (crop-space path): bbox is the crop
        rect, rgb_sub/mask_sub are that rect's pixels/mask — no full-frame diff
        scan. Only painted frames are stored, so `frames`/`painted_frames` track
        together (the compositor reads only painted_frames)."""
        y0, x0, y1, x1 = (int(v) for v in bbox)
        mask_sub = np.asarray(mask_sub).astype(bool)
        np.savez(
            self._path(k),
            bbox=np.array([y0, x0, y1, x1], np.int64),
            rgb=np.ascontiguousarray(rgb_sub),
            mask=np.ascontiguousarray(mask_sub),
        )
        self.frames.add(k)
        if mask_sub.any():
            self.painted_frames.add(k)

    def load_existing(self) -> int:
        """Resume: repopulate frames/painted_frames from npz already on disk so
        a re-run can skip finished windows yet still read prior frames for
        preserve-overlap and the final composite. painted-ness is recovered
        from each stored mask. Frame npz are zero-padded indices; the anchor
        sidecar (non-numeric stem) is skipped."""
        for p in self.root.glob("*.npz"):
            try:
                k = int(p.stem)
            except ValueError:
                continue
            self.frames.add(k)
            try:
                if bool(np.load(p)["mask"].any()):
                    self.painted_frames.add(k)
            except (OSError, KeyError):
                continue
        return len(self.frames)

    def mark_window_done(self, wi: int) -> None:
        (self.root / f"win{wi:02d}.done").write_text("")

    def window_done(self, wi: int) -> bool:
        return (self.root / f"win{wi:02d}.done").exists()

    def save_anchor(self, mean, std) -> None:
        np.savez(self.root / "anchor.npz",
                 mean=np.asarray(mean), std=np.asarray(std))

    def load_anchor(self):
        """-> (mean, std) from a prior run's window 0, or None."""
        p = self.root / "anchor.npz"
        if not p.exists():
            return None
        d = np.load(p)
        return d["mean"], d["std"]

    def get(self, k: int):
        """-> ((y0, x0, y1, x1), rgb_sub, mask_sub) or None if not stored."""
        if k not in self.frames:
            return None
        d = np.load(self._path(k))
        return tuple(int(v) for v in d["bbox"]), d["rgb"], d["mask"]

    def stitched_crop(self, k: int, cb, source_full) -> np.ndarray:
        """Prior stitch within rect `cb=(y0,x0,y1,x1)`, crop-sized: source crop
        with any stored sub overlaid where it intersects cb. Avoids the
        full-frame copy `stitched_full` makes — the crop-space stitch base."""
        y0, x0, y1, x1 = cb
        out = source_full[y0:y1, x0:x1].copy()
        rec = self.get(k)
        if rec is not None:
            (py0, px0, py1, px1), sub, _ = rec
            iy0, ix0 = max(py0, y0), max(px0, x0)
            iy1, ix1 = min(py1, y1), min(px1, x1)
            if iy1 > iy0 and ix1 > ix0:
                out[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = \
                    sub[iy0 - py0:iy1 - py0, ix0 - px0:ix1 - px0]
        return out

    def stitched_full(self, k: int, source_rgb) -> np.ndarray:
        """Full-frame stitched output (== source outside the stored box)."""
        rec = self.get(k)
        if rec is None:
            return source_rgb
        (y0, x0, y1, x1), rgb_sub, _ = rec
        out = source_rgb.copy()
        if y1 > y0 and x1 > x0:
            out[y0:y1, x0:x1] = rgb_sub
        return out
