#!/bin/bash
#
# Model Download Script for mac-fairness
# Downloads models from HuggingFace for vLLM inference
#
# Prerequisites: HF_TOKEN environment variable must be set
#
# Usage: ./script/cluster/download_models.sh

set -euo pipefail

# Configuration
MODELS_DIR="$HF_HUB_CACHE"
LOG_DIR="$LFS_HOME/.log"
LOG_FILE="$LOG_DIR/hf_model_download_$(date +%Y%m%d_%H%M%S).log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Check HF_TOKEN
if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN environment variable not set"
    exit 1
fi

# Ensure huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli not found. Install with: uv pip install huggingface-hub"
    exit 1
fi

# Login using token
echo "Logging in to HuggingFace..."
hf auth login --token "$HF_TOKEN"  # skipping for now --add-to-git-credential

# Models to download (organized by model family)
# Check vLLM model support: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models
#
# Sampling parameter recommendations (from model cards):
# - Qwen3-4B: temp=0.7, top_p=0.8, top_k=20
# - Phi models (all): temp=0.0 (greedy decoding)
# - Others: No explicit recommendations in model cards

models=(
    # === Google Gemma (newest first) ===
    "google/gemma-2-27b-it"
    "google/gemma-2-9b-it"
    "google/gemma-2-2b-it"

    # === Meta Llama (newest first) ===
    # "meta-llama/Llama-3.3-70B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    # "meta-llama/Llama-3.1-70B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"

    # === Microsoft Phi (newest first) ===
    "microsoft/Phi-4-mini-instruct"
    "microsoft/Phi-3.5-mini-instruct"
    "microsoft/Phi-3-mini-128k-instruct"
    "microsoft/Phi-3-mini-4k-instruct"

    # === Mistral AI ===
    "mistralai/Mistral-7B-Instruct-v0.3"

    # === Alibaba Qwen (newest first) ===
    "Qwen/Qwen3-4B-Instruct-2507"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
)

echo "============================================================"
echo "Downloading ${#models[@]} models to: $MODELS_DIR"
echo "Log file: $LOG_FILE"
echo "============================================================"
echo ""

# Download each model
failed_models=()
for model in "${models[@]}"; do
    echo "[$(date +%H:%M:%S)] Downloading: $model"
    if ! hf download "$model" 2>&1 | tee -a "$LOG_FILE"; then
        echo "Warning: Failed to download $model"
        failed_models+=("$model")
    fi
    echo ""
done

echo "============================================================"
if [ ${#failed_models[@]} -eq 0 ]; then
    echo "All downloads complete!"
else
    echo "Downloads complete with ${#failed_models[@]} failure(s):"
    for model in "${failed_models[@]}"; do
        echo "  - $model"
    done
fi
echo "Models cached in: $MODELS_DIR"
echo "============================================================"
