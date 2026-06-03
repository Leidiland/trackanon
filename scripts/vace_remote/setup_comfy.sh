#!/usr/bin/env bash
# Runs ON the rented box. Clones ComfyUI, installs the GGUF custom node,
# downloads the Wan-VACE 14B + UMT5 XXL + Wan 2.1 VAE, and launches N ComfyUI
# workers on consecutive ports. Idempotent — safe to re-run.
set -euo pipefail

WORKERS="${WORKERS:-6}"                   # 1.3B on 96GB; drop on smaller cards
BASE_PORT="${BASE_PORT:-8190}"
MODEL="${MODEL:-1.3B}"                    # 1.3B (default) | 14B
QUANT="${QUANT:-Q8_0}"                    # only used when MODEL=14B: Q8_0|Q6_K|Q5_K_M|Q4_K_M
COMFY_DIR="${COMFY_DIR:-$HOME/ComfyUI}"
CACHE_LRU="${CACHE_LRU:-2}"               # bound node-output cache; unique-named inputs would otherwise pile decoded clips in RAM unbounded
# Keep the model GPU-resident across prompts (no per-prompt reload from CPU —
# the dominant non-sampling cost on a dedicated 1-worker-per-GPU box, helps
# distill and native alike). Auto-on for 1.3B (~4GB fits resident); off for 14B
# fp16 where it can crowd out activations on 40GB cards. Override with COMFY_HIGHVRAM=0/1.
COMFY_HIGHVRAM="${COMFY_HIGHVRAM:-}"
# Pinned upstream commits this project was built against; newer tips can break
# the VACE graph / GGUF node. Override to bump, then re-test. GGUF is unpinned
# by default (no known-good SHA recorded) — set COMFY_GGUF_COMMIT to pin it.
COMFYUI_COMMIT="${COMFYUI_COMMIT:-c37d2a0dacaa256a2fb1812ae026e09dd493661e}"
COMFY_GGUF_COMMIT="${COMFY_GGUF_COMMIT:-}"

echo "==> setup_comfy.sh: WORKERS=$WORKERS BASE_PORT=$BASE_PORT MODEL=$MODEL QUANT=$QUANT"

# ---- ComfyUI (pinned) ---------------------------------------------------------
if [ ! -d "$COMFY_DIR" ]; then
    echo "==> cloning ComfyUI @ ${COMFYUI_COMMIT:0:10}"
    git init -q "$COMFY_DIR"
    git -C "$COMFY_DIR" remote add origin https://github.com/comfyanonymous/ComfyUI.git
    git -C "$COMFY_DIR" fetch -q --depth 1 origin "$COMFYUI_COMMIT"
    git -C "$COMFY_DIR" checkout -q FETCH_HEAD
fi
cd "$COMFY_DIR"

# Torch index for the GPU arch. Blackwell (RTX 50xx, sm_120) needs a cu128+
# build; cu121 imports fine but has NO sm_120 kernels — a silent "no kernel
# image" at the first CUDA op (the likely cause of earlier dead 5090 hosts).
# cu128 also covers Ampere/Hopper, so it's a safe default for every box.
TORCH_INDEX="${TORCH_INDEX:-cu128}"
# Reuse the image's torch if present (most vast.ai PyTorch templates ship it);
# else make our own venv. Arch-correctness is verified below regardless.
if ! python3 -c "import torch" 2>/dev/null; then
    if [ ! -d venv ]; then
        python3 -m venv venv
    fi
    # shellcheck disable=SC1091
    source venv/bin/activate
    pip install --upgrade pip
fi
pip install -r requirements.txt
pip install huggingface_hub
# Verify torch can actually launch a kernel on THIS GPU; a wrong-arch wheel (a
# cu121/cu124 image torch on Blackwell) imports + reports cuda available but
# throws at the first op. If so, reinstall against TORCH_INDEX.
if ! python3 -c "import torch; assert torch.cuda.is_available(); torch.zeros(1, device='cuda')" 2>/dev/null; then
    echo "==> torch can't run a kernel on this GPU — reinstalling torch/$TORCH_INDEX"
    pip install --upgrade pip
    pip install --force-reinstall torch torchvision \
        --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
fi

# Optional: SageAttention (quantised attention, ~1.2-1.4x; helps distill +
# native, biggest at the 81-frame windows). Opt-in — the wheel build needs a
# matching CUDA toolchain and can be slow/fragile, so off by default. Enable
# with COMFY_SAGE=1; the launch loop adds --use-sage-attention when it imports.
if [ "${COMFY_SAGE:-0}" = "1" ]; then
    echo "==> Installing SageAttention (COMFY_SAGE=1)"
    pip install sageattention || echo "WARNING: sageattention install failed — will launch without it"
fi

# Newer huggingface_hub (>= ~1.17) ships `hf` and removed `huggingface-cli download`.
# Older images still ship `huggingface-cli`. Pick whichever resolves on this box.
if command -v hf >/dev/null 2>&1; then
    HF_DL=(hf download)
else
    HF_DL=(huggingface-cli download)
fi

# ---- ComfyUI-GGUF custom node (only needed for 14B GGUF) ----------------------
if [ "$MODEL" = "14B" ]; then
    if [ ! -d custom_nodes/ComfyUI-GGUF ]; then
        git clone --depth 1 https://github.com/city96/ComfyUI-GGUF custom_nodes/ComfyUI-GGUF
        [ -n "$COMFY_GGUF_COMMIT" ] && git -C custom_nodes/ComfyUI-GGUF fetch -q --depth 1 origin "$COMFY_GGUF_COMMIT" && git -C custom_nodes/ComfyUI-GGUF checkout -q FETCH_HEAD
    fi
    pip install -r custom_nodes/ComfyUI-GGUF/requirements.txt
fi

# ---- Models -------------------------------------------------------------------
mkdir -p models/unet models/clip models/vae models/diffusion_models

if [ "$MODEL" = "14B" ]; then
    UNET_FILE="Wan2.1_14B_VACE-${QUANT}.gguf"
    if [ ! -f "models/unet/$UNET_FILE" ]; then
        echo "==> Fetching $UNET_FILE (~14GB at Q8)"
        "${HF_DL[@]}" QuantStack/Wan2.1_14B_VACE-GGUF \
            "$UNET_FILE" --local-dir models/unet
    fi
else
    # 1.3B as a plain safetensors via Comfy-Org's repackaged repo. Goes into
    # diffusion_models/ (ComfyUI looks there before models/unet/ for Wan).
    UNET_FILE="wan2.1_vace_1.3B_fp16.safetensors"
    if [ ! -f "models/diffusion_models/$UNET_FILE" ]; then
        echo "==> Fetching $UNET_FILE (~2.6GB)"
        "${HF_DL[@]}" Comfy-Org/Wan_2.1_ComfyUI_repackaged \
            "split_files/diffusion_models/$UNET_FILE" \
            --local-dir models/diffusion_models
        mv "models/diffusion_models/split_files/diffusion_models/$UNET_FILE" \
            "models/diffusion_models/" 2>/dev/null || true
    fi
fi

CLIP_FILE="umt5_xxl_fp8_e4m3fn_scaled.safetensors"
if [ ! -f "models/clip/$CLIP_FILE" ]; then
    echo "==> Fetching $CLIP_FILE"
    "${HF_DL[@]}" Comfy-Org/Wan_2.1_ComfyUI_repackaged \
        "split_files/text_encoders/$CLIP_FILE" \
        --local-dir models/clip
    # The repackaged repo nests files under split_files/...; flatten.
    mv "models/clip/split_files/text_encoders/$CLIP_FILE" "models/clip/" 2>/dev/null || true
fi

VAE_FILE="wan_2.1_vae.safetensors"
if [ ! -f "models/vae/$VAE_FILE" ]; then
    echo "==> Fetching $VAE_FILE"
    "${HF_DL[@]}" Comfy-Org/Wan_2.1_ComfyUI_repackaged \
        "split_files/vae/$VAE_FILE" \
        --local-dir models/vae
    mv "models/vae/split_files/vae/$VAE_FILE" "models/vae/" 2>/dev/null || true
fi

# Distill LoRA — lets the pipeline render at ~4-6 steps / cfg=1. The 14B uses
# LightX2V cfg-step-distill (no 1.3B LightX2V exists upstream); 1.3B falls back
# to CausVid. Must match anonymization.vace.lora in the config.
mkdir -p models/loras
if [ "$MODEL" = "14B" ]; then
    LORA_FILE="Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"
else
    LORA_FILE="Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors"
fi
if [ ! -f "models/loras/$LORA_FILE" ]; then
    echo "==> Fetching $LORA_FILE"
    "${HF_DL[@]}" Kijai/WanVideo_comfy "$LORA_FILE" --local-dir models/loras
fi

# ---- Stop pre-existing ComfyUI on non-target ports ---------------------------
# vast.ai ComfyUI templates auto-launch ComfyUI on :8188 (the default), which
# holds VRAM we need for our own workers. Kill any ComfyUI process whose
# --port doesn't match our target ports. Idempotent re-deploys see workers
# already on target ports and the launch loop below skips them.
TARGET_PORTS=""
for i in $(seq 0 $((WORKERS-1))); do
    TARGET_PORTS="$TARGET_PORTS $((BASE_PORT + i))"
done
echo "==> Pruning stray ComfyUI processes (keeping target ports:$TARGET_PORTS)"
# Stop systemd units that might respawn ComfyUI before we get a chance.
if command -v systemctl >/dev/null 2>&1; then
    for unit in $(systemctl list-unit-files --no-legend 2>/dev/null | awk '{print $1}' | grep -iE "comfy" || true); do
        echo "==> Stopping systemd unit $unit"
        systemctl stop "$unit" 2>/dev/null || true
        systemctl disable "$unit" 2>/dev/null || true
    done
fi
# Kill any ComfyUI python processes whose port we're not about to claim.
while IFS= read -r line; do
    PID=$(echo "$line" | awk '{print $1}')
    [ -z "$PID" ] && continue
    PORT_USED=$(echo "$line" | grep -oE -- "--port[= ]+[0-9]+" | grep -oE "[0-9]+" | head -1)
    KEEP=0
    for tp in $TARGET_PORTS; do
        [ "$PORT_USED" = "$tp" ] && KEEP=1 && break
    done
    if [ "$KEEP" = "0" ]; then
        echo "==> Killing stray ComfyUI pid=$PID port=${PORT_USED:-?}"
        kill "$PID" 2>/dev/null || true
    fi
done < <(ps -eo pid,cmd | grep -E "python.*main\.py" | grep -v grep || true)
sleep 2                                     # let VRAM free before we relaunch
# Sanity: report whether anything still holds VRAM.
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | \
        head -1 | xargs -I{} echo "==> GPU memory after prune: {}"
fi

# ---- Launch workers -----------------------------------------------------------
# Each worker is a separate ComfyUI process holding its own model copy in VRAM.
# Per-worker input/output dirs match the VaceClientPool's `_{i}` staging suffix
# so concurrent bundle uploads from different gids don't trample.
#
# Pin each worker to its own GPU: without CUDA_VISIBLE_DEVICES every worker
# lands on cuda:0, so N workers just time-share one GPU (no speedup). Round-robin
# when WORKERS exceeds the GPU count. WORKERS=1 stays unpinned (use the whole box).
NGPU=1
if command -v nvidia-smi >/dev/null 2>&1; then
    NGPU=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU " || echo 1)
fi
[ "$NGPU" -lt 1 ] && NGPU=1
echo "==> Pinning workers across $NGPU GPU(s)"
if [ -z "$COMFY_HIGHVRAM" ]; then
    [ "$MODEL" = "1.3B" ] && COMFY_HIGHVRAM=1 || COMFY_HIGHVRAM=0
fi
HIGHVRAM_ARG=""
[ "$COMFY_HIGHVRAM" = "1" ] && HIGHVRAM_ARG="--highvram"
echo "==> Model residency: COMFY_HIGHVRAM=$COMFY_HIGHVRAM (${HIGHVRAM_ARG:-default mgmt})"
# Only pass --use-sage-attention if it actually imports, else ComfyUI won't start.
SAGE_ARG=""
if [ "${COMFY_SAGE:-0}" = "1" ] && python -c "import sageattention" 2>/dev/null; then
    SAGE_ARG="--use-sage-attention"
    echo "==> SageAttention enabled"
elif [ "${COMFY_SAGE:-0}" = "1" ]; then
    echo "==> SageAttention requested but not importable — launching without it"
fi
for i in $(seq 0 $((WORKERS-1))); do
    PORT=$((BASE_PORT + i))
    if curl -sS --max-time 2 "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1; then
        echo "==> Worker $i already running on :$PORT"
        continue
    fi
    if [ "$WORKERS" -gt 1 ]; then
        IN="input_$i"; OUT="output_$i"
        GPU=$((i % NGPU))
        PIN="CUDA_VISIBLE_DEVICES=$GPU"
    else
        IN="input"; OUT="output"                   # legacy single-server layout
        PIN=""                                      # unpinned: whole box for the one worker
    fi
    mkdir -p "$IN" "$OUT"
    env $PIN python main.py \
        --port "$PORT" --listen 127.0.0.1 \
        --cache-lru "$CACHE_LRU" $HIGHVRAM_ARG $SAGE_ARG \
        --input-directory "$PWD/$IN" --output-directory "$PWD/$OUT" \
        > "worker_${i}.log" 2>&1 &
    echo "==> Worker $i launched on :$PORT (pid $!)${PIN:+ ($PIN)}"
done

# ---- Wait for /system_stats ---------------------------------------------------
for i in $(seq 0 $((WORKERS-1))); do
    PORT=$((BASE_PORT + i))
    READY=0
    for _ in $(seq 1 120); do
        if curl -sS --max-time 2 "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1; then
            READY=1; break
        fi
        sleep 1
    done
    [ "$READY" = "1" ] && echo "==> Worker $i ready on :$PORT" || \
        { echo "!!! Worker $i did NOT come up on :$PORT — check worker_${i}.log"; exit 1; }
done

echo "==> All $WORKERS workers ready. Ports: $BASE_PORT..$((BASE_PORT + WORKERS - 1))"
