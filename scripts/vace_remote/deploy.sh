#!/usr/bin/env bash
# Copy setup_comfy.sh to the rented box and run it. Idempotent.
#
# Usage:
#   bash scripts/vace_remote/deploy.sh <instance_id> [WORKERS] [QUANT]
#
# Env (optional):
#   MODEL       1.3B (default — light, fast, ~$0.5–1.5/hr usable) | 14B
#   COMFY_DIR   point at an existing ComfyUI install (e.g. /workspace/ComfyUI
#               on the vast.ai "ComfyUI" template) instead of cloning fresh.
set -euo pipefail

INSTANCE_ID="${1:-}"
# 14B GGUF Q8 is the rental default (the quality tier; matches vace_remote.yaml).
MODEL="${MODEL:-14B}"
# Default worker count: 14B Q8 fits 1 per 32GB card (5090) / 2 per 96GB; 1.3B fits 6.
# Override on a per-host basis (deploy.sh <ID> 2, etc.).
if [ "$MODEL" = "14B" ]; then DEFAULT_WORKERS=1; else DEFAULT_WORKERS=6; fi
WORKERS="${2:-$DEFAULT_WORKERS}"
QUANT="${3:-Q8_0}"
# SageAttention on by default — ~1.2-1.4x on the expensive 14B, quality ~neutral.
# setup_comfy falls back gracefully (launches without it) if the wheel won't build.
COMFY_SAGE="${COMFY_SAGE:-1}"
export COMFY_SAGE
COMFY_DIR_ENV="${COMFY_DIR:-}"
[ -z "$INSTANCE_ID" ] && { echo "usage: deploy.sh <instance_id> [workers] [quant]"; exit 1; }

# Wait for the instance to be in a runnable state (vast.ai's "running" status).
echo "==> Waiting for instance $INSTANCE_ID to reach running state"
for _ in $(seq 1 60); do
    STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -c \
        "import json,sys; print(json.load(sys.stdin).get('actual_status','unknown'))" 2>/dev/null || echo "unknown")
    if [ "$STATUS" = "running" ]; then break; fi
    sleep 5
done

SSH_URL=$(vastai ssh-url "$INSTANCE_ID")
echo "==> Deploying to $SSH_URL"

# Parse `ssh://user@host:port` into user@host + port for scp/ssh.
SSH_NOSCHEME="${SSH_URL#ssh://}"
SSH_HOSTPART="${SSH_NOSCHEME%:*}"            # user@host
SSH_PORT="${SSH_NOSCHEME##*:}"               # port

scp -P "$SSH_PORT" -o StrictHostKeyChecking=accept-new \
    "$(dirname "$0")/setup_comfy.sh" "$SSH_HOSTPART:setup_comfy.sh"

REMOTE_ENV="WORKERS=$WORKERS MODEL=$MODEL QUANT=$QUANT"
[ -n "$COMFY_DIR_ENV" ] && REMOTE_ENV="$REMOTE_ENV COMFY_DIR=$COMFY_DIR_ENV"
# Forward perf knobs so a deploy can request SageAttention / model residency /
# torch build (Blackwell needs cu128) without editing setup_comfy on the box.
[ -n "${COMFY_SAGE:-}" ]     && REMOTE_ENV="$REMOTE_ENV COMFY_SAGE=$COMFY_SAGE"
[ -n "${COMFY_HIGHVRAM:-}" ] && REMOTE_ENV="$REMOTE_ENV COMFY_HIGHVRAM=$COMFY_HIGHVRAM"
[ -n "${TORCH_INDEX:-}" ]    && REMOTE_ENV="$REMOTE_ENV TORCH_INDEX=$TORCH_INDEX"
ssh -p "$SSH_PORT" -o StrictHostKeyChecking=accept-new "$SSH_HOSTPART" \
    "$REMOTE_ENV bash setup_comfy.sh"

echo "==> Deployed. Open the tunnel with:"
echo "   bash scripts/vace_remote/tunnel.sh $INSTANCE_ID $WORKERS"
