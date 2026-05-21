"""Identity consistency verification for trackanon output.

Reads crops saved by DiffusionPipeline (save_crops_dir) and computes two metrics:

  1. Replacement quality  — SSIM(orig, gen) per frame per identity.
     Low SSIM means the generated persona looks different from the original
     person (good: the original was successfully replaced).

  2. Temporal consistency — pairwise SSIM across all generated crops for the
     same global_id.  High SSIM means the same persona appears across frames.

Usage
-----
    python scripts/verify_identity_consistency.py --crops-dir outputs/crops

Set save_crops_dir in configs/anonymization/diffusion.yaml and run the
pipeline first to populate the directory.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Simplified single-window SSIM (no scikit-image dependency)
# ---------------------------------------------------------------------------

def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Return SSIM in [-1, 1] between two uint8 images (any spatial size)."""
    # Resize b to a's shape if needed.
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    sig_a = a.std()
    sig_b = b.std()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (sig_a ** 2 + sig_b ** 2 + c2)
    return float(num / den) if den != 0 else 0.0


def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 3:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr


# ---------------------------------------------------------------------------
# Crop discovery
# ---------------------------------------------------------------------------

_FNAME_RE = re.compile(r"^f(\d+)_gid(\d+)_(orig|gen)\.png$")


def _discover(crops_dir: Path) -> tuple[
    dict[int, list[tuple[int, np.ndarray]]],  # gid → [(frame, orig_gray), ...]
    dict[int, list[tuple[int, np.ndarray]]],  # gid → [(frame, gen_gray), ...]
]:
    orig: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    gen: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)

    for p in sorted(crops_dir.glob("*.png")):
        m = _FNAME_RE.match(p.name)
        if not m:
            continue
        frame_idx = int(m.group(1))
        gid = int(m.group(2))
        kind = m.group(3)
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = _to_gray(img)
        if kind == "orig":
            orig[gid].append((frame_idx, gray))
        else:
            gen[gid].append((frame_idx, gray))

    return orig, gen


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _replacement_ssims(
    orig: dict[int, list[tuple[int, np.ndarray]]],
    gen: dict[int, list[tuple[int, np.ndarray]]],
) -> dict[int, list[float]]:
    """SSIM(orig, gen) per gid per frame.  Lower = more replacement."""
    by_gid: dict[int, list[float]] = defaultdict(list)
    for gid in gen:
        orig_map = {f: img for f, img in orig.get(gid, [])}
        for frame_idx, g in gen[gid]:
            if frame_idx in orig_map:
                by_gid[gid].append(_ssim(orig_map[frame_idx], g))
    return by_gid


def _consistency_ssims(
    gen: dict[int, list[tuple[int, np.ndarray]]],
) -> dict[int, list[float]]:
    """Pairwise SSIM across gen crops for the same gid.  Higher = more stable."""
    by_gid: dict[int, list[float]] = {}
    for gid, frames in gen.items():
        crops = [img for _, img in frames]
        if len(crops) < 2:
            by_gid[gid] = []
            continue
        scores: list[float] = []
        for i in range(len(crops)):
            for j in range(i + 1, len(crops)):
                scores.append(_ssim(crops[i], crops[j]))
        by_gid[gid] = scores
    return by_gid


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(values: list[float]) -> str:
    if not values:
        return "n/a (need ≥2 crops)"
    return f"{np.mean(values):.3f}  (min {np.min(values):.3f}, max {np.max(values):.3f}, n={len(values)})"


def report(crops_dir: Path):
    orig, gen = _discover(crops_dir)

    if not gen:
        print(f"No generated crops found in {crops_dir}")
        print("Set save_crops_dir in configs/anonymization/diffusion.yaml and re-run the pipeline.")
        sys.exit(1)

    replace_ssims = _replacement_ssims(orig, gen)
    consist_ssims = _consistency_ssims(gen)

    gids = sorted(gen.keys())
    print(f"\nCrops directory : {crops_dir}")
    print(f"Identities found: {len(gids)}  (gids: {gids})\n")

    print("┌─────────────────────────────────────────────────────────────────────────────────┐")
    print("│ gid │  frames  │  replacement SSIM (lower=better)  │  consistency SSIM (higher=better) │")
    print("├─────┼──────────┼───────────────────────────────────┼───────────────────────────────────┤")
    for gid in gids:
        n_frames = len(gen[gid])
        r = _fmt(replace_ssims.get(gid, []))
        c = _fmt(consist_ssims.get(gid, []))
        print(f"│ {gid:3d} │  {n_frames:5d}   │ {r:33s} │ {c:33s} │")
    print("└─────────────────────────────────────────────────────────────────────────────────┘")

    all_replace = [s for v in replace_ssims.values() for s in v]
    all_consist = [s for v in consist_ssims.values() for s in v]
    print(f"\nOverall replacement SSIM : {_fmt(all_replace)}")
    print(f"Overall consistency SSIM : {_fmt(all_consist)}")

    print("\nInterpretation:")
    print("  Replacement SSIM < 0.5  → persons are clearly replaced (good)")
    print("  Consistency SSIM > 0.7  → same persona across frames (good)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crops-dir", required=True, type=Path, help="Directory containing saved crops")
    args = parser.parse_args()
    report(args.crops_dir)


if __name__ == "__main__":
    main()
