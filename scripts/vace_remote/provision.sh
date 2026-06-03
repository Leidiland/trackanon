#!/usr/bin/env bash
# Lists vast.ai offers that match our GPU/disk requirements; prints the
# create-instance command for each. Deliberately not auto-creating — for a
# one-off rental, you want to eyeball the price + reliability before spending.
#
# Env knobs:
#   GPU_TYPE       e.g. RTX_6000_Ada, RTX_4090, A100_80GB_PCIE, H100_80GB
#   MIN_VRAM_GB    minimum GPU memory (default 40 for 14B Q8 @ workers=1)
#   DISK_GB        minimum disk space (default 200 — model files + cache)
#   MIN_RELIABILITY default 0.95
set -euo pipefail

GPU_TYPE="${GPU_TYPE:-RTX_6000_Ada}"
MIN_VRAM_GB="${MIN_VRAM_GB:-40}"
DISK_GB="${DISK_GB:-200}"
MIN_RELIABILITY="${MIN_RELIABILITY:-0.95}"

command -v vastai >/dev/null || { echo "Install: pip install vastai && vastai set api-key YOUR_KEY"; exit 1; }
vastai show user >/dev/null 2>&1 || { echo "Not authenticated. Run: vastai set api-key YOUR_KEY"; exit 1; }

echo "==> Searching: gpu_name=$GPU_TYPE gpu_ram>=${MIN_VRAM_GB}GB disk_space>=${DISK_GB}GB reliability>=${MIN_RELIABILITY}"
vastai search offers \
    "gpu_name=$GPU_TYPE gpu_ram>=$MIN_VRAM_GB disk_space>=$DISK_GB reliability>=$MIN_RELIABILITY rentable=true" \
    -o "dph_total" --raw 2>/dev/null | head -50 | \
    python3 -c "
import json, sys
offers = []
for line in sys.stdin:
    try: offers.append(json.loads(line))
    except: pass
offers = sorted(offers, key=lambda o: o['dph_total'])[:5]
print(f'{\"#\":<3} {\"id\":<10} {\"gpu\":<20} {\"vram\":<6} {\"\$/hr\":<8} {\"rel\":<5} {\"loc\":<15}')
for i, o in enumerate(offers, 1):
    print(f'{i:<3} {o[\"id\"]:<10} {o[\"gpu_name\"]:<20} {o[\"gpu_ram\"]/1024:<6.0f} {o[\"dph_total\"]:<8.3f} {o[\"reliability2\"]:<5.2f} {o.get(\"geolocation\",\"?\"):<15}')
print()
if offers:
    cheapest = offers[0]
    print('Create the cheapest:')
    print(f'  vastai create instance {cheapest[\"id\"]} --image pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel --disk {$DISK_GB}')
    print('Or pick any id and prefix the same: vastai create instance <id> ...')
"
