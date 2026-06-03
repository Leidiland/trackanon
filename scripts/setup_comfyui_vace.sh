#!/usr/bin/env bash
# Rebuild the VACE ComfyUI under external/comfyui-vace.
#   1.3B diffusion model + CausVid speed LoRA + umt5 text-encoder + VAE
#   14B: NOT fetched here — remote GPU box only (see scripts/vace_remote/)
# Idempotent: re-running skips anything already present.
#
# Overridable env:
#   COMFY_VACE_DIR     install location (default: external/comfyui-vace at the repo root)
#   COMFY_MODEL_STORE  keep the large encoder/VAE files on another disk and
#                      symlink them in (default: download straight into the model dirs)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL="${COMFY_VACE_DIR:-$ROOT/external/comfyui-vace}"
STORE="${COMFY_MODEL_STORE:-}"
# Pinned to the ComfyUI commit this project was built against; newer tips can
# break the VACE graph. Bump deliberately, then re-test.
COMFYUI_COMMIT="c37d2a0dacaa256a2fb1812ae026e09dd493661e"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# ---- ComfyUI clone (pinned) ----
if [ ! -d "$INSTALL/.git" ]; then
  log "cloning ComfyUI -> $INSTALL @ ${COMFYUI_COMMIT:0:10}"
  git init -q "$INSTALL"
  git -C "$INSTALL" remote add origin https://github.com/comfyanonymous/ComfyUI.git
  git -C "$INSTALL" fetch -q --depth 1 origin "$COMFYUI_COMMIT"
  git -C "$INSTALL" checkout -q FETCH_HEAD
fi
cd "$INSTALL"

# ---- isolated venv + torch cu126 + requirements ----
if [ ! -d .venv ]; then log "creating venv"; python3 -m venv .venv; fi
. .venv/bin/activate
python -m pip install -q --upgrade pip
log "installing torch cu126"
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu126
log "installing ComfyUI requirements"
pip install -q -r requirements.txt

# ---- model dirs ----
DM=models/diffusion_models; TE=models/text_encoders; VAE=models/vae; LORA=models/loras
mkdir -p "$DM" "$TE" "$VAE" "$LORA"
[ -n "$STORE" ] && mkdir -p "$STORE"

dl(){ # url dest  -> returns nonzero on failure
  if [ -s "$2" ]; then log "have $(basename "$2")"; return 0; fi
  log "downloading $(basename "$2")"
  wget -q --show-progress -O "$2.part" "$1" && mv "$2.part" "$2"
}

place(){ # url dest_dir filename  -> download into dest_dir, or into STORE + symlink
  local fname="$3"
  if [ -n "$STORE" ]; then
    dl "$1" "$STORE/$fname" && ln -sf "$STORE/$fname" "$2/$fname"
  else
    dl "$1" "$2/$fname"
  fi
}

base=https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files

# 1.3B diffusion model
dl "$base/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors" "$DM/wan2.1_vace_1.3B_fp16.safetensors"

# umt5 text encoder + VAE (large — honour COMFY_MODEL_STORE)
place "$base/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" "$TE" "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
place "$base/vae/wan_2.1_vae.safetensors" "$VAE" "wan_2.1_vae.safetensors"

# CausVid 1.3B speed LoRA
lora_url=https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors
if ! dl "$lora_url" "$LORA/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors"; then
  rm -f "$LORA/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors.part"
  log "WARN: LoRA download failed — verify the filename/URL and re-run"
fi

log "REBUILD COMPLETE"
ls -laL "$DM" "$TE" "$VAE" "$LORA" 2>/dev/null
