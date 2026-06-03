#!/usr/bin/env bash
# Destroy the vast.ai instance (per-second billing — every minute idle costs).
# Tunnels are local and die on their own once the SSH session closes; this just
# nukes the remote box.
#
# Usage:
#   bash scripts/vace_remote/teardown.sh <instance_id>
set -euo pipefail

INSTANCE_ID="${1:-}"
[ -z "$INSTANCE_ID" ] && { echo "usage: teardown.sh <instance_id>"; exit 1; }

echo "==> Destroying instance $INSTANCE_ID"
vastai destroy instance "$INSTANCE_ID"
echo "==> Done. Tunnels (if any) close when you Ctrl-C the tunnel.sh terminal."
