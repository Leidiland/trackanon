"""Download the model weights the pipeline needs.

Fetches everything the Python side loads at runtime:

  osnet   OSNet ReID checkpoint (Google Drive, via gdown) -> models/osnet/
  sam3    SAM 3 segmentation weights (Hugging Face cache)
  face    InsightFace buffalo_l face embeddings (~/.insightface/models/)

Each target is independent — a failure in one is reported and the rest still
run. Pass one or more names to fetch a subset; default is all.

    python scripts/download_weights.py             # everything
    python scripts/download_weights.py osnet face   # subset

Not covered here (large, backend-specific, fetched elsewhere):
  - Wan-VACE UNet/CLIP/VAE/LoRA -> scripts/setup_comfyui_vace.sh
  - DWpose/RTMW -> rtmlib auto-fetches to ~/.cache/rtmlib on first use
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# OSNet x1_0 trained on MSMT17 — broadest training set, generalises best across
# our domain. ImageNet-pretrained backbones are not discriminative enough for
# person re-ID; see ta_src/tracking/osnet_wrapper.py.
_OSNET_URLS = {
    "osnet_x1_0_msmt17": "https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M",
    "osnet_x1_0_market1501": "https://drive.google.com/uc?id=1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA",
    "osnet_x1_0_dukemtmcreid": "https://drive.google.com/uc?id=1QZO_4sNf4hdOKKKzKc-TZU9WW1v6zQbq",
}
_SAM3_MODEL_ID = "facebook/sam3"


def fetch_osnet(name: str = "osnet_x1_0_msmt17", out_dir: Path = Path("models/osnet")) -> None:
    import gdown

    out_path = out_dir / f"{name}.pt"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"osnet: already present: {out_path}")
        return
    gdown.download(_OSNET_URLS[name], str(out_path), quiet=False)
    if not out_path.is_file() or out_path.stat().st_size < 1024:
        raise RuntimeError(f"download failed: {out_path}")
    print(f"osnet: saved {out_path} ({out_path.stat().st_size // 1024} KiB)")


def fetch_sam3() -> None:
    from huggingface_hub import snapshot_download

    path = snapshot_download(_SAM3_MODEL_ID)
    print(f"sam3: cached {_SAM3_MODEL_ID} at {path}")


def fetch_face() -> None:
    # Instantiating FaceAnalysis triggers the buffalo_l download into the
    # InsightFace cache; prepare() finalises the ONNX sessions.
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1)
    print("face: buffalo_l ready")


_TARGETS = {"osnet": fetch_osnet, "sam3": fetch_sam3, "face": fetch_face}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", choices=sorted(_TARGETS), default=[],
                        help="which weights to fetch (default: all)")
    parser.add_argument("--osnet-name", default="osnet_x1_0_msmt17", choices=sorted(_OSNET_URLS))
    args = parser.parse_args()

    targets = args.targets or sorted(_TARGETS)
    failures = []
    for name in targets:
        try:
            if name == "osnet":
                fetch_osnet(args.osnet_name)
            else:
                _TARGETS[name]()
        except Exception as e:  # keep going; report at the end
            print(f"{name}: FAILED — {e}", file=sys.stderr)
            failures.append(name)

    if failures:
        print(f"\nfailed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
