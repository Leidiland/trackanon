<div align="center">

# TrackAnon: Towards Consistent Full-Body Anonymization

[[Video Examples]](https://www.youtube.com/playlist?list=PLOGoHFfQuUFbPs4Kd-pnbJl9JEeaJRdlb)

![Side-by-side: source video on the left, anonymized output on the right](data/examples/middle_slice_castle_vs_anon.gif)

</div>

## What it does

Offline pipeline that anonymizes video by **replacing** every person — not blurring them. Each individual is detected, identified against a **Known Person Library** (KPL) of real photos, and re-rendered as a fictional but consistent diffusion-generated appearance. The same Person keeps the same fabricated look across occlusions, chunks, and re-runs.

## How it works

```
  video ──► SAM 3 chunked segmentation (60-frame sessions)
              │
              ▼
        IdentityResolver  ──►  face (InsightFace) + body (OSNet) evidence
              │                  │
              │                  ▼
              │            Hungarian assigner over KPL identities
              │                  │
              ▼                  ▼
        AnonymizationStage  ◄── confirmed global_id
              │
              ├── diffusion path: ComfyUI inpaint
              │     • body pass — ControlNet OpenPose + IP-Adapter (low face strength)
              │     • face polish pass (optional) — IP-Adapter at high face strength on
              │       a tight head-keypoint mask
              │
              └── fallback path: Gaussian-blur-over-mask (closed-world violations,
                    pre-confirmation frames, mask-quality failures)
```

Closed-world by design: every Identity is seeded from KPL at startup with a frozen face centroid, OSNet appearance centroid, captioned Persona prompt, and a synthetic **Reference Crop**. The Reference Crop — generated once by img2img with an anti-leak gate against the original photos — is the only thing fed to IP-Adapter at runtime, so the real KPL pixels never reach the diffusion stage.

## Setup

```bash
# 1. Submodules (ComfyUI, SAM 3, ViTPose, OSNet)
bash scripts/setup_submodules.sh

# 2. Weights
bash scripts/download_weights.sh
python scripts/download_osnet_weights.py

# 3. Python deps (CUDA 12.6 wheels)
pip install -r requirements.txt
```

## Usage

### Anonymize a video

```bash
python scripts/run_pipeline.py --input data/input_videos/example.mp4
```

The pipeline scans `data/known_persons/` at startup, seeds one Identity per sub-folder (sorted, 1-indexed `global_id`), runs **First Generation** for any Identity without a cached Reference Crop, then processes the video chunk-by-chunk.
