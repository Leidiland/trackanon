#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p models/rtdetr models/sam2

# RT-DETR
RTDETR_URL="https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_m_7x_coco_ema.pth"
RTDETR_DEST="models/rtdetr/rtdetrv2_r50vd_m_7x_coco_ema.pth"
if [ ! -f "$RTDETR_DEST" ]; then
    echo "Downloading RT-DETR weights..."
    wget -q --show-progress -O "$RTDETR_DEST" "$RTDETR_URL"
else
    echo "RT-DETR weights already present."
fi

# SAM2
SAM2_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
SAM2_DEST="models/sam2/sam2.1_hiera_base_plus.pt"
if [ ! -f "$SAM2_DEST" ]; then
    echo "Downloading SAM2 weights..."
    wget -q --show-progress -O "$SAM2_DEST" "$SAM2_URL"
else
    echo "SAM2 weights already present."
fi

# Realistic Vision V6 — non-inpainting variant. Loaded by the txt2img prewarm
# workflow (txt2img_default.json) used by PrewarmGenerator (ADR-0008). The
# matching inpainting checkpoint must already be present for the runtime
# inpaint workflows; SD inpainting checkpoints have a 9-channel UNet input and
# produce desaturated, cool-cast outputs when used in plain txt2img (which
# feeds a 4-channel latent).
mkdir -p external/comfyui/models/checkpoints
RV6_URL="https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE/resolve/main/Realistic_Vision_V6.0_NV_B1_fp16.safetensors?download=true"
RV6_DEST="external/comfyui/models/checkpoints/Realistic_Vision_V6.0_NV_B1_fp16.safetensors"
if [ ! -f "$RV6_DEST" ]; then
    echo "Downloading Realistic Vision V6 (non-inpainting) checkpoint..."
    wget -q --show-progress -O "$RV6_DEST" "$RV6_URL"
else
    echo "Realistic Vision V6 (non-inpainting) checkpoint already present."
fi

# IP-Adapter-FaceID-Plus-v2 (ADR-0004). Required for the structural face-
# anonymisation guarantee: the synthetic Reference Crop's InsightFace embedding
# is injected at runtime by ipadapter_injector.inject. DiffusionPipeline hard-
# aborts at startup if either of these two files is missing.
# Also requires the ComfyUI_IPAdapter_plus custom node pack:
#   git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus \
#       external/comfyui/custom_nodes/ComfyUI_IPAdapter_plus
mkdir -p external/comfyui/models/ipadapter external/comfyui/models/loras
IPA_FACEID_URL="https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin"
IPA_FACEID_DEST="external/comfyui/models/ipadapter/ip-adapter-faceid-plusv2_sd15.bin"
if [ ! -f "$IPA_FACEID_DEST" ]; then
    echo "Downloading IP-Adapter-FaceID-Plus-v2 model (~156 MB)..."
    wget -q --show-progress -O "$IPA_FACEID_DEST" "$IPA_FACEID_URL"
else
    echo "IP-Adapter-FaceID-Plus-v2 model already present."
fi

IPA_FACEID_LORA_URL="https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15_lora.safetensors"
IPA_FACEID_LORA_DEST="external/comfyui/models/loras/ip-adapter-faceid-plusv2_sd15_lora.safetensors"
if [ ! -f "$IPA_FACEID_LORA_DEST" ]; then
    echo "Downloading FaceID-Plus-v2 LoRA (~51 MB)..."
    wget -q --show-progress -O "$IPA_FACEID_LORA_DEST" "$IPA_FACEID_LORA_URL"
else
    echo "FaceID-Plus-v2 LoRA already present."
fi

echo "Done."
