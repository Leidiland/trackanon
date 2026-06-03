<div align="center">

# TrackAnon: Towards Consistent Full-Body Anonymization

[[Video Examples]](https://www.youtube.com/playlist?list=PLOGoHFfQuUFbPs4Kd-pnbJl9JEeaJRdlb)

![Side-by-side: source video on the left, anonymized output on the right](data/examples/middle_slice_castle_vs_anon_14b_10sec.gif)

</div>

## What it does

Offline pipeline that anonymizes video by **replacing** every person or a smaller subset — with a blurred fallback. Each individual is detected, identified against a **Known Person Library** (KPL) of real photos, and re-rendered as a fictional but consistent diffusion-generated appearance. The same person keeps the same fabricated look across occlusions, chunks, and re-runs.

## How it works

```
  Pass 1 — track
  video ──► SAM 3 chunked segmentation (25-frame sessions)
              │
              ▼
        IdentityResolver  ──►  face (InsightFace) + body (OSNet) evidence
              │                  │
              │                  ▼
              │            Hungarian assigner over KPL identities
              │                  │
              ▼                  ▼
        per-frame track cache ◄── confirmed global_id

  Pass 2 — anonymize (per confirmed gid)
        plan windows ──► Wan-VACE render ──► stitch into frame
              │            (grey-control silhouette, reference-anchored)
              │
              └── fallback path: Gaussian-blur-over-mask (closed-world violations,
                    unconfirmed gids, mask-quality failures, unpainted silhouettes)
```

Closed-world by design: every identity is seeded from KPL at startup with a frozen face centroid, OSNet appearance centroid, captioned persona prompt, and a synthetic **Reference Crop**. The reference crop is a pre-generated synthetic image loaded from disk; VACE anchors each window's appearance to it, so the real KPL pixels never reach the generation stage.

## Setup

Requires Python ≥ 3.12, an NVIDIA GPU, and CUDA 12.6.

```bash
# 1. Runtime directory tree (input/output/cache/model dirs)
python setup.py

# 2. Vendored deps — clones SAM 3 + ComfyUI into external/
bash scripts/setup_submodules.sh

# 3. Python deps (CUDA 12.6 wheels; installs -e external/sam3)
pip install -r requirements.txt

# 4. Weights — OSNet ReID + SAM 3 + InsightFace buffalo_l
python scripts/download_weights.py
# VACE backend weights (Wan2.1-VACE UNet/CLIP/VAE) install into the portable.
```

## Usage

### Anonymize a video

```bash
python scripts/run_pipeline.py --input data/input_videos/example.mp4
```

The pipeline scans `data/known_persons/` at startup, seeds one identity per sub-folder (sorted, 1-indexed `global_id`), loads each identity's pre-generated reference crop from `data/gallery/reference_crops/`, then tracks the clip (Pass 1) and renders each confirmed gid with VACE (Pass 2).

### Using your own people

The repo ships a ready-to-run example KPL with pre-generated reference crops. The pipeline **consumes** reference crops as fixed synthetic images — it does not generate them at the moment. To anonymize your own footage, add one folder of face photos per person under `data/known_persons/<name>/`, and supply a matching synthetic `data/gallery/reference_crops/<global_id>.png` produced by any image generator (an optional `<global_id>.prompt.txt` overrides the VACE prompt for that identity).

## License

Released under the [MIT License](LICENSE).
