# vace_remote — rented-GPU VACE runner

Run the VACE backend on a rented vast.ai GPU box without changing the pipeline
code. Pass 1 (SAM3 + IdentityResolver) stays local; only bundles (crop, mask,
letterboxed reference) cross the SSH tunnel.

## One-time prerequisites

1. **vast.ai account** — create at <https://vast.ai>, add ~$30 credit to start.
2. **API key** — `Account → API Keys → Create New Key`; copy it.
3. **Local CLI**:
   ```bash
   pip install vastai
   vastai set api-key YOUR_KEY
   vastai show user                  # smoke check
   ```
4. **SSH key** — vast.ai uses your local `~/.ssh/id_*.pub` by default; add it
   under `Account → SSH Keys` if not already there.

## End-to-end (≈15 min on first run, ~$1 model-download time then cheap per clip)

Default is **1.3B with 6 workers** (sized for 96GB-class GPUs — RTX 6000 Ada
96GB, RTX PRO 6000 Blackwell, etc.). Pass a smaller `WORKERS` arg to
`deploy.sh` for smaller cards; see the cheat sheet. Switch to 14B by setting
`MODEL=14B` before `deploy.sh`.

```bash
# 1) Find a box. Lists top 5 cheapest matching offers.
bash scripts/vace_remote/provision.sh
# Pick an offer id and launch (100GB disk is enough for 1.3B; bump to 200 for 14B):
vastai create instance <OFFER_ID> --image pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel --disk 100

# 2) Deploy ComfyUI + 1.3B + workers. Idempotent — re-run after a reboot.
bash scripts/vace_remote/deploy.sh <INSTANCE_ID>
# Args: instance_id [workers] [quant]. Defaults: workers=6 for 1.3B / 2 for 14B.
# For 14B instead:
#   MODEL=14B bash scripts/vace_remote/deploy.sh <INSTANCE_ID> 2 Q8_0

# 3) Open the SSH tunnel in its OWN terminal. Leave this open.
#    Second arg = number of ports to forward (must match WORKERS above).
bash scripts/vace_remote/tunnel.sh <INSTANCE_ID> 6

# 4) In another terminal, run a clip.
bash scripts/vace_remote/run_clip.sh data/input_videos/5_sec_test_4k.mp4

# 5) Destroy the box when done (billing stops on destroy, not on shutdown).
bash scripts/vace_remote/teardown.sh <INSTANCE_ID>
```

## GPU + model cheat sheet

1.3B fits 4 workers in ~55GB at our default settings; 14B Q8 needs ~40GB per
worker so the math is different. Match `deploy.sh`'s `WORKERS` arg to the
table column, and adjust `pool.workers` in `configs/anonymization/vace_remote.yaml`
(or pass `++anonymization.vace.pool.workers=N` to `run_clip.sh`) to match.

### Default — 1.3B (`MODEL=1.3B`, no GGUF, no custom node)

Per-worker peak ~11–14GB at the default canvas/window size in no-offload mode.

| GPU                    | VRAM | Workers | ≈ $/hr | Notes |
|------------------------|-----:|--------:|-------:|-------|
| RTX 4090               | 24GB |       1 | $0.3–0.6 | Cheapest validation — same model your local box runs |
| RTX 5090               | 32GB |       2 | $0.5–0.9 | Tightest 2-worker option |
| A100 40GB PCIE         | 40GB |       2 | $0.7–1.0 | Comfortable |
| RTX 6000 Ada           | 48GB |       3 | $0.7–1.2 | Common; ~3× local speed |
| A100 80GB SXM          | 80GB |       5 | $1.2–1.8 | "Real" multi-worker tier |
| H100 80GB              | 80GB |       5 | $2.5–3.5 | Fastest per-window but priciest |
| RTX 6000 Ada 96GB      | 96GB |     **6** | $1.5–2.5 | Default target |
| RTX PRO 6000 Blackwell | 96GB |     **6** | $1.5–2.5 | Same VRAM tier, newer arch — best $/throughput |

**Recommendation for first 1.3B run**: A100 40GB PCIE or RTX 4090, `workers=1–2`,
~$1/hr. Anonymizing a 30-sec clip with 5 gids should land in 5–10 min wall
time and cost <$0.50.

### Opt-in — 14B Q8 GGUF (`MODEL=14B`, faithful colour + fine articulation)

| GPU                    | VRAM | Workers | ≈ $/hr | Notes |
|------------------------|-----:|--------:|-------:|-------|
| RTX 4090               | 24GB |       1 | $0.3–0.6 | Heavy offload, slow |
| A100 40GB PCIE         | 40GB |       1 | $0.7–1.0 | No offload, single-worker |
| A100 80GB SXM          | 80GB |       2 | $1.2–1.8 | Sweet spot for 14B |
| RTX 6000 Ada 96GB      | 96GB |       2 | $1.5–2.5 | 2 comfortable; 3 if Q6_K |
| RTX PRO 6000 Blackwell | 96GB |       2 | $1.5–2.5 | Same VRAM tier as Ada 96GB |
| H100 80GB              | 80GB |       2 | $2.5–3.5 | Fastest per-step on 14B |

Pick the quant from the table below; pass `QUANT=...` to `deploy.sh` after the
workers arg.

### 14B quant cheat sheet

| Quant   | Size  | Quality | Use when |
|---------|------:|---------|----------|
| Q8_0    | ~14GB | Best (~fp16 indistinguishable) | Default for 14B. Has VRAM. |
| Q6_K    | ~11GB | Very close to Q8 | Need headroom for workers=3 on 96GB |
| Q5_K_M  | ~9.5GB | Slight texture/colour loss | 4090 / tight memory |
| Q4_K_M  | ~8GB  | Visible quality loss | Last resort |

## Reusing Pass 1 across remote runs (skip SAM 3 on the rental)

Pass 1 (SAM 3 + IdentityResolver) is CPU+GPU heavy on the laptop but runs
cheaply there; Pass 2 (VACE) is where the rented GPU earns its keep. To avoid
paying for the rental's clock during Pass 1, run Pass 1 once locally and have
the remote VACE run reuse its cache.

**Save Pass 1 locally** (no anonymisation, ~10–15 min on a laptop):

```bash
.venv/bin/python scripts/run_pipeline.py \
  --input data/input_videos/CASTLE_day01_11.00-11.20_Meeting.mp4 \
  --output outputs/eval/runs/meeting_60s_cache \
  pipeline.track_cache_dir=outputs/eval/cache/meeting_60s \
  pipeline.run_anonymization=false \
  pipeline.run_pose=false \
  temporal.start_time=540 temporal.end_time=600 temporal.fps=25
```

That writes the per-frame npz files to `outputs/eval/cache/meeting_60s/` and
keeps them past the run (the cache lives outside `data/temp/`, so the
wipe-on-exit predicate leaves it alone).

**Reuse the saved cache for a remote VACE run** — point the same
`track_cache_dir` at the saved path; the pipeline detects the populated cache
and skips Pass 1 automatically:

```bash
bash scripts/vace_remote/run_clip.sh \
  data/input_videos/CASTLE_day01_11.00-11.20_Meeting.mp4 \
  outputs/vace_remote_meeting60 \
  temporal.start_time=540 temporal.end_time=600 \
  pipeline.track_cache_dir=outputs/eval/cache/meeting_60s
```

You'll see `VACE: reusing pre-populated cache at outputs/eval/cache/meeting_60s
(skipping Pass 1)` in the run log — that's the confirmation it took the
shortcut. Saves ~10 min of rental time and lets you A/B Pass 2 variants
(different `target_gids`, `window_len`, `preserve_overlap`, etc.) against a
fixed Pass 1 baseline without re-running tracking each time.

## How the tunnel actually works

The `VaceClient` uploads each bundle (`control.mp4`, `mask.mp4`,
`reference.png`) to ComfyUI's `/upload/image` endpoint and downloads the
rendered clip from `/view`. Both go through the SSH-tunneled HTTP port — no
filesystem assumption either side of the tunnel. The same code path is used
for the local `vace.yaml` profile, so the remote path is exercised every time
you run locally too (microsecond overhead over a `shutil.copy` on localhost).

## Privacy guard

Only the bundle files (`control.mp4`, `mask.mp4`, `reference.png`, `meta.json`)
travel over the SSH tunnel. The pipeline cuts these to the per-gid bounding box
crop, so even the bundles don't carry full-scene context. The original clip
never leaves the local box.

That said: uploading **faces** to a third-party GPU box is a privacy operation.
- The bundles persist in vast.ai-managed storage until you `destroy`.
- The host operator can in principle inspect the worker process' files.

Mitigations: use a reputable host (verified by vast.ai), destroy the instance
immediately when done, avoid uploading personally-identifying training data.

## After deploy: how to know things are healthy

On the remote box (`ssh $(vastai ssh-url <id>)`):
```bash
# All workers should answer:
for p in 8190 8191; do curl -sS http://127.0.0.1:$p/system_stats | head -c 80; echo; done
# Per-worker logs:
tail -f ~/ComfyUI/worker_0.log
nvidia-smi
```

From local, through the tunnel:
```bash
curl -sS http://127.0.0.1:8190/system_stats          # should match remote's :8190
```

## When something breaks

- **`deploy.sh` hangs at "Waiting for instance ... to reach running state"**
  vast.ai sometimes takes ~10 min for first boot of large disk allocations.
  Check the web console; if the instance says "loading", wait.
- **Workers don't come up**: `ssh` in, `tail worker_N.log` — usually a missing
  model file (HuggingFace 404 — quant name might have moved). The exact quant
  filenames live under `QuantStack/Wan2.1_14B_VACE-GGUF` on HF; confirm.
- **Tunnel drops**: `tunnel.sh` uses `ServerAliveInterval=30`; if it still
  exits, re-run it. Pipeline runs are checkpoint-safe per-window — re-running
  `run_clip.sh` against the same output dir will redo Pass 1 (cheap) and the
  remaining VACE windows.
- **Out-of-memory on the remote box**: drop workers (`4 → 2`) or — on 14B —
  drop the quant (`Q8_0 → Q6_K`). The setup script is idempotent; re-deploy
  with new args. Remember to drop `pool.workers` in `vace_remote.yaml` to
  match (or pass `++anonymization.vace.pool.workers=N` to `run_clip.sh`).

- **`workers` mismatch between deploy and pipeline**: if `deploy.sh` launches
  4 workers but `vace_remote.yaml` says `pool.workers: 2`, you waste 2 servers
  (no error). If deploy launches 2 and config says 4, the pipeline tries to
  reach :8192/:8193 and fails. Keep them in sync, or override at run time.
