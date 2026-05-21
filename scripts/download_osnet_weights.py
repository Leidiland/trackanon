"""Download an OSNet ReID checkpoint from the torchreid model zoo.

By default fetches osnet_x1_0 pre-trained on MSMT17 — the broadest training
set, generalises best across our domain. ImageNet-pretrained backbones (the
torchreid default) are not discriminative enough for person re-ID; see
ta_src/tracking/osnet_wrapper.py for why this matters.

    python scripts/download_osnet_weights.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gdown

# torchreid model zoo IDs (see torchreid.reid_model_factory.__trained_urls).
_MODEL_URLS = {
    "osnet_x1_0_msmt17": "https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M",
    "osnet_x1_0_market1501": "https://drive.google.com/uc?id=1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA",
    "osnet_x1_0_dukemtmcreid": "https://drive.google.com/uc?id=1QZO_4sNf4hdOKKKzKc-TZU9WW1v6zQbq",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name", default="osnet_x1_0_msmt17", choices=sorted(_MODEL_URLS),
    )
    parser.add_argument("--out-dir", default="models/osnet", type=Path)
    args = parser.parse_args()

    out_path = args.out_dir / f"{args.name}.pt"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"already present: {out_path}")
        return 0

    gdown.download(_MODEL_URLS[args.name], str(out_path), quiet=False)
    if not out_path.is_file() or out_path.stat().st_size < 1024:
        print(f"download failed: {out_path}", file=sys.stderr)
        return 1
    print(f"saved: {out_path} ({out_path.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
