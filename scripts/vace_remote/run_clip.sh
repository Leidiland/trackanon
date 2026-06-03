#!/usr/bin/env bash
# Run main_pipeline against the tunneled remote ComfyUI. Pass 1 (SAM3 +
# IdentityResolver) runs locally; Pass 2 (VACE) bundles go through the tunnel.
# Only bundles (crops/masks/reference) leave the box — never the full clip.
#
# Usage:
#   bash scripts/vace_remote/run_clip.sh <input_clip> [output_dir] [hydra_overrides...]
#
# Any args after output_dir are forwarded as Hydra overrides — use this for
# temporal windows, gid targeting, knob sweeps, etc. Examples:
#   bash .../run_clip.sh clip.mp4 out/ temporal.start_time=540 temporal.end_time=600
#   bash .../run_clip.sh clip.mp4 out/ '+anonymization.vace.target_gids=[3,8]'
set -euo pipefail

CLIP="${1:-}"
OUT="${2:-outputs/vace_remote_$(date +%Y%m%d_%H%M%S)}"
[ -z "$CLIP" ] && { echo "usage: run_clip.sh <input_clip> [output_dir] [hydra_overrides...]"; exit 1; }
[ -f "$CLIP" ] || { echo "Input clip not found: $CLIP"; exit 1; }
shift 2 2>/dev/null || shift "$#"           # consume positionals; "$@" = overrides

# Reap any stray pipeline from a prior crashed/OOM'd run before launching. A
# leftover python squatting on ~9GB co-resident with a fresh ~12GB run is what
# tipped the box into global OOM last time; the kernel killed the new run.
stray=$(pgrep -f "scripts/run_pipeline.py" || true)
if [ -n "$stray" ]; then
    echo "==> Reaping stray run_pipeline process(es): $stray"
    pkill -f "scripts/run_pipeline.py" || true
    sleep 2
fi

# Sanity: at least the first tunnel port must answer.
curl -sS --max-time 3 http://127.0.0.1:8190/system_stats >/dev/null || {
    echo "ComfyUI on :8190 not reachable — is tunnel.sh running?"; exit 1;
}

mkdir -p "$OUT"
echo "==> Running pipeline: input=$CLIP output=$OUT (anonymization=vace_remote)"
echo "==> Hydra overrides: $*"
{
    echo "==== remote VACE run $(date -Iseconds) ===="
    .venv/bin/python scripts/run_pipeline.py \
        --input "$CLIP" --output "$OUT" \
        anonymization=vace_remote \
        pipeline.run_anonymization=true \
        pipeline.run_tracking=true \
        pipeline.run_pose=false \
        temporal.fps=25 \
        "$@"
    echo "==== exit=$? $(date -Iseconds) ===="
} 2>&1 | tee "$OUT/run.log"
