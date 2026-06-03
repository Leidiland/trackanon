#!/usr/bin/env bash
# Self-healing SSH tunnel: restarts ssh whenever the flaky host drops it, so a
# long multi-clip render survives mid-run disconnects (exit 255). Same port
# forwarding as tunnel.sh; loops until killed.
set -uo pipefail

INSTANCE_ID="${1:-}"
WORKERS="${2:-2}"
BASE_PORT="${3:-8190}"
[ -z "$INSTANCE_ID" ] && { echo "usage: tunnel_keepalive.sh <instance_id> [workers] [base_port]"; exit 1; }

SSH_URL=$(vastai ssh-url "$INSTANCE_ID")
NS="${SSH_URL#ssh://}"; HOSTPART="${NS%:*}"; PORT="${NS##*:}"

PORT_FLAGS=()
for i in $(seq 0 $((WORKERS-1))); do P=$((BASE_PORT + i)); PORT_FLAGS+=(-L "$P:127.0.0.1:$P"); done

while true; do
    echo "[$(date '+%H:%M:%S')] connecting tunnel -> $SSH_URL"
    ssh -N -p "$PORT" \
        -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=accept-new \
        -o ConnectTimeout=15 \
        "${PORT_FLAGS[@]}" "$HOSTPART"
    echo "[$(date '+%H:%M:%S')] tunnel exited ($?), reconnecting in 3s"
    sleep 3
done
