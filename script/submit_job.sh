#!/bin/bash

CONFIG_FILE=$1

# Copy config NOW (at submission time)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIG_BASENAME=$(basename "$CONFIG_FILE")
TMP_CONFIG="/tmp/${CONFIG_BASENAME}_${TIMESTAMP}"
cp "$CONFIG_FILE" "$TMP_CONFIG"

# Submit with the frozen (at the slurm submission time) config path
# Example usage: ./script/submit_job.sh config/prod_vllm/bbq-sampled-set_1agent_as-ai_v2025-12-10.yaml
sbatch --export=ALL,TMP_CONFIG="$TMP_CONFIG" run_job.slurm