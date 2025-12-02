# Node Setup Guide

Complete setup steps for running mac-fairness experiments on remote Ubuntu 22.04 nodes with NVIDIA GPUs.

---

## Prerequisites

- Ubuntu 22.04
- NVIDIA GPU with CUDA support
- Python ≥ 3.11
- Node.js ≥ 18
- Environment variables: `HF_TOKEN` (read-only), `LFS_HOME`

**Important Notes:**

- **CUDA:** No need to manually install CUDA toolkit. vLLM bundles CUDA runtime libraries. Just ensure the NVIDIA driver supports CUDA 11.8+ `nvidia-smi`.
- **Dependencies:** No need to manually install PyTorch or transformers. `uv pip install vllm>=0.11.2` automatically installs all required dependencies including PyTorch, transformers, etc.

---

## Step 1: Environment Configuration

Add to `~/.zshrc.user`:

```bash
# Project: mac-fairness configuration
export MAC_FAIRNESS_WORKSPACE="${LFS_HOME}/workspace/mac-fairness"
export MAC_FAIRNESS_EXPERIMENT_ROOT="${MAC_FAIRNESS_WORKSPACE}/experiment"

# HF_TOKEN should already be set (read-only token)
export HF_HOME="${LFS_HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"

# CUDA 12.4
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

---

## Step 2: Python Environment Setup

```bash
cd $MAC_FAIRNESS_WORKSPACE

# Create virtual environment with uv
uv venv --python 3.11

# Activate
source .venv/bin/activate

# Install dependencies from pyproject.toml
uv pip install -e .

# Verify
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python -c "from huggingface_hub import HfApi; print('HF Hub: OK')"
```

---

## Step 3: TypeScript Validation Setup

```bash
cd schema/2025-11-27

# Install dependencies
npm install

# Build validation scripts
npm run build

# Verify
ls dist/validate.js

cd ../..
```

---

## Step 4: Download Models

```bash
# Ensure HF_TOKEN is set
echo $HF_TOKEN

# Run download script (downloads 11 models, ~101 GB)
./script/download_models.sh
```

### Tiny/Small/Medium (all <10B params): 11 models, ~101 GB

- [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (3.1 GB)
- [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) (6.2 GB)
- [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (8.1 GB)
- [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) (2.0 GB)
- [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) (8.6 GB)
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (32.1 GB)
- [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) (5.0 GB)
- [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) (12.9 GB)
- [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) (7.6 GB)
- [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) (7.6 GB)
- [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) (7.7 GB)

**Important Notes:**

- **Llama models:** Accept license before downloading
- **Sampling params:** Use common defaults (see Model Configurations below)
- **vLLM support:** Verify at https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models
- **Disk space:** ~101 GB required

---

## Step 5: Verify Setup

```bash
# Check GPU
nvidia-smi

# Check cache
du -sh $HF_HOME/hub/

# List models
source .venv/bin/activate
hf cache scan [--verbose]

# Test imports
python << 'EOF'
import vllm
import yaml
from huggingface_hub import HfApi
print("✓ All imports successful")
print(f"✓ vLLM version: {vllm.__version__}")
EOF
```

---

## Model Configurations

### vLLM Common Settings

```yaml
vllm_config:
  tensor_parallel_size: 1 # GPUs for model parallelism
  gpu_memory_utilization: 0.85 # Conservative: 85% (safer across GPU types)
  max_num_seqs: 8 # Max parallel sequences/requests
  max_model_len: 4096 # Max context length
  enable_prefix_caching: true # Cache common prompts
```

**Parameter Explanations:**

- **`max_num_seqs`**: Maximum batch size (parallel requests)
  - vLLM does NOT auto-tune this - you must set it manually
  - Start with 4-8, increase gradually if no OOM
  - Higher = better throughput, but more GPU memory needed
- **`gpu_memory_utilization`**: Fraction of GPU RAM for model + KV cache
  - 0.85 (85%) is conservative and safe across different GPUs
  - 0.9 (90%) for well-tested GPU types
  - Leave headroom for framework overhead and temporary buffers
- **OOM prevention across multiple GPU types:**
  - Use conservative `gpu_memory_utilization: 0.85` initially
  - Start with lower `max_num_seqs: 4` on smaller GPUs (16GB)
  - Increase `max_num_seqs` on larger GPUs (40GB, 80GB)
  - Monitor with `nvidia-smi` together with job summaries
  - Can still OOM if `max_num_seqs` x `max_model_len` is too large

**Recommended settings by example GPU type (for 8B models like Llama-3.1-8B):**

| GPU Model       | VRAM  | gpu_memory_util | max_num_seqs | max_model_len | Notes                                      |
| --------------- | ----- | --------------- | ------------ | ------------- | ------------------------------------------ |
| V100 PCIe/SXM2  | 32GB  | 0.85            | 8-12         | 8192          | Comfortable for 8B                         |
| L40S            | 48GB  | 0.9             | 12-16        | 8192          | Excellent for 8B, can handle some batching |
| Quadro RTX 8000 | 48GB  | 0.9             | 12-16        | 8192          | Same tier as L40S                          |
| H100 SXM5       | 80GB  | 0.9             | 16-32        | 8192          | High throughput, great batching            |
| A100 80GB       | 80GB  | 0.9             | 16-32        | 8192          | Production-grade performance               |
| H200            | 144GB | 0.9             | 32-64        | 8192          | Massive batching capability                |
| B200            | 192GB | 0.9             | 64-128       | 8192          | Extreme batching, overkill for 8B          |

> **Omitted:** RTX 2080 Ti (11GB), RTX A4000 (16GB), P100 PCIe (16 GB), V100 SXM2 (16GB), P40 (24GB)

**Model size recommendations by GPU:**

- **11-16GB GPUs:** Best for 1B-4B models (Gemma-3-1b, Phi-3-mini, Qwen2.5-3B)
- **24-32GB GPUs:** Good for 4B-8B models (all models in download list)
- **48GB+ GPUs:** Comfortable for 8B, can handle larger batches
- **80GB+ GPUs:** Can explore larger models (14B+) or high `max_num_seqs`
- Check GPUs on Slurm system

  ```shell
  sinfo -o "%200f %G" -p gpu | grep -o 'GPU_SKU:[^ ,]*,GPU_MEM:[^ ,]*' | sort -u
  ```

### Gemma Family

```yaml
# All Gemma models use bfloat16
# No official sampling params - use common defaults
dtype: bfloat16
temperature: 0.7
top_p: 0.95
top_k: 40

# GPU memory requirements
gemma-3-1b-it:  max_model_len: 8192,  GPU: ~4 GB
gemma-3-4b-it:  max_model_len: 8192,  GPU: ~10 GB
```

### Llama Family

```yaml
# All Llama models use float16
# No official sampling params - use common defaults
dtype: float16
temperature: 0.7
top_p: 0.9

# GPU memory requirements
Llama-3.2-1B-Instruct:  max_model_len: 8192,  GPU: ~4 GB
Llama-3.2-3B-Instruct:  max_model_len: 8192,  GPU: ~8 GB
Llama-3.1-8B-Instruct:  max_model_len: 8192,  GPU: ~18 GB
```

### Phi Family

```yaml
# All Phi models use float16
# Official recommendation: greedy decoding (temp=0.0)
dtype: float16
temperature: 0.7

# GPU memory requirements
Phi-3-mini-4k-instruct:  max_model_len: 4096,  GPU: ~8 GB
Phi-3.5-mini-instruct:   max_model_len: 8192,  GPU: ~8 GB
Phi-4-mini-instruct:     max_model_len: 8192,  GPU: ~8 GB
```

### Qwen Family

```yaml
# Official params for Qwen3-4B: temp=0.7, top_p=0.8, top_k=20
# Qwen2.5 models: no official params (use Qwen3 as reference)
# Qwen3 uses float16, Qwen2.5 uses bfloat16

temperature: 0.7  # Official for Qwen3
top_p: 0.8        # Official for Qwen3
top_k: 20         # Official for Qwen3

# GPU memory requirements
Qwen2.5-1.5B-Instruct:  dtype: bfloat16, max_model_len: 8192,  GPU: ~4 GB
Qwen2.5-3B-Instruct:    dtype: bfloat16, max_model_len: 8192,  GPU: ~8 GB
Qwen3-4B-Instruct-2507: dtype: float16,  max_model_len: 8192,  GPU: ~10 GB
```

### Example Complete Config

```yaml
model_config:
  shared_model_backbone: qwen25_7b

  models:
    qwen25_7b:
      family: qwen
      backend: vllm
      model_path: Qwen/Qwen2.5-7B-Instruct

      vllm_config:
        tensor_parallel_size: 1
        gpu_memory_utilization: 0.9
        dtype: bfloat16
        max_model_len: 4096
        temperature: 0.7
        top_p: 0.8
        top_k: 20
        max_num_seqs: 8
        enable_prefix_caching: true
```

---

## Complete Setup Script (Copy-Paste)

```bash
#!/bin/bash
set -euo pipefail

# Navigate to workspace
cd $MAC_FAIRNESS_WORKSPACE

# Create and activate venv
uv venv --python 3.11
source .venv/bin/activate

# Install Python dependencies
uv pip install -e .

# Build TypeScript schemas
cd schema/2025-11-27
npm install
npm run build
cd ../..

# Download models (requires HF_TOKEN)
./script/download_models.sh

# Verify
echo "Setup complete! Verifying..."
python -c "import vllm; print(f'✓ vLLM {vllm.__version__}')"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "✓ Models: $(ls $HF_HOME/hub/ | grep models-- | wc -l) downloaded"
```

---

## Quick Reference

| Component    | Location                                 |
| ------------ | ---------------------------------------- |
| Workspace    | `$MAC_FAIRNESS_WORKSPACE`                |
| Python venv  | `.venv/`                                 |
| Models cache | `$LFS_HOME/.cache/huggingface/hub/`      |
| Download log | `$LFS_HOME/.log/hf_model_download_*.log` |
| Experiments  | `$MAC_FAIRNESS_EXPERIMENT_ROOT/`         |

---

## Next Steps

1. Implement vLLM agent (currently `NotImplementedError` in [src/agent/model_factory.py](../src/agent/model_factory.py#L171))
2. Create experiment configs (see [README.md](../README.md))
3. Test with `--range 0-10`
4. Submit to Slurm
