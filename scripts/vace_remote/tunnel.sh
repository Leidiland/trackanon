#!/usr/bin/env bash
# SSH tunnel from the rented box's ComfyUI ports to local 127.0.0.1.
# Foreground (Ctrl-C to close) so it's obvious when it dies.
#
# Usage:
#   bash scripts/vace_remote/tunnel.sh <instance_id> [WORKERS=2] [BASE_PORT=8190]
set -euo pipefail

INSTANCE_ID="${1:-}"
WORKERS="${2:-2}"
BASE_PORT="${3:-8190}"
[ -z "$INSTANCE_ID" ] && { echo "usage: tunnel.sh <instance_id> [workers] [base_port]"; exit 1; }

SSH_URL=$(vastai ssh-url "$INSTANCE_ID")
SSH_NOSCHEME="${SSH_URL#ssh://}"
SSH_HOSTPART="${SSH_NOSCHEME%:*}"
SSH_PORT="${SSH_NOSCHEME##*:}"

PORT_FLAGS=()
for i in $(seq 0 $((WORKERS-1))); do
    P=$((BASE_PORT + i))
    PORT_FLAGS+=(-L "$P:127.0.0.1:$P")
done

echo "==> Tunneling $WORKERS port(s) starting at $BASE_PORT from $SSH_URL"
echo "==> Once 'ComfyUI alive on :PORT' prints, leave this terminal open and"
echo "    run scripts/vace_remote/run_clip.sh in another."

# Verify reachability from inside the tunnel before backgrounding.
exec ssh -N -p "$SSH_PORT" -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
    "${PORT_FLAGS[@]}" "$SSH_HOSTPART"
