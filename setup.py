#!/usr/bin/env python3
"""Scaffold the runtime directory tree.

Creates the input/output/cache/model directories the pipeline expects but that
are gitignored (so they don't exist on a fresh clone). Idempotent — re-running
leaves existing directories and their contents untouched.

    python setup.py

Run alongside scripts/download_weights.py (model weights) and
scripts/setup_submodules.sh (vendored SAM 3 + ComfyUI) on first install.
"""
from __future__ import annotations

from pathlib import Path

# Runtime directories, gitignored and created on demand.
_DIRS = [
    "external",
    "models/osnet",
    "data/known_persons",
    "data/gallery/reference_crops",
    "data/gallery/kpl_centroids",
    "data/input_videos",
    "data/output_videos",
    "data/examples",
    "logs",
]


def main() -> int:
    root = Path(__file__).resolve().parent
    for rel in _DIRS:
        path = root / rel
        existed = path.is_dir()
        path.mkdir(parents=True, exist_ok=True)
        print(f"{'have' if existed else 'made'}: {rel}")
    print(
        "\nDirectories ready. Next:\n"
        "  bash scripts/setup_submodules.sh   # vendored SAM 3 + ComfyUI\n"
        "  python scripts/download_weights.py # model weights\n"
        "  drop one folder per person under data/known_persons/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
