#!/usr/bin/env bash
# Clone the vendored dependencies the pipeline installs from source.
#   external/sam3          SAM 3 segmentation (editable install, see requirements.txt)
#   external/comfyui-vace  ComfyUI host for the Wan-VACE anonymisation backend
# Idempotent: re-running skips anything already cloned.
#
# OSNet and DWpose are pip packages (torchreid, rtmlib) — no clone needed; their
# weights come from scripts/download_weights.py and an on-first-use fetch.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL="$ROOT/external"
mkdir -p "$EXTERNAL"

# Pinned to the commits this project was built against. These upstreams move
# fast and newer tips can break the pipeline; bump deliberately, then re-test.
SAM3_COMMIT="4bf4a8aaff2ab34789731810eed45e6a61a73cfd"
COMFYUI_COMMIT="c37d2a0dacaa256a2fb1812ae026e09dd493661e"

clone() { # name repo-url commit
    local dest="$EXTERNAL/$1"
    if [ -d "$dest/.git" ]; then
        echo "have: external/$1"
        return 0
    fi
    echo "cloning external/$1 <- $2 @ ${3:0:10}"
    git init -q "$dest"
    git -C "$dest" remote add origin "$2"
    git -C "$dest" fetch -q --depth 1 origin "$3"
    git -C "$dest" checkout -q FETCH_HEAD
}

clone sam3          https://github.com/facebookresearch/sam3.git  "$SAM3_COMMIT"
clone comfyui-vace  https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_COMMIT"

echo
echo "Submodules cloned. Next:"
echo "  pip install -r requirements.txt           # installs -e external/sam3"
echo "  bash scripts/setup_comfyui_vace.sh        # fetch Wan-VACE backend weights"
