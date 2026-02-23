#!/bin/bash

set -euo pipefail

CONFIG_FILE=${1:-}
if [[ -z "$CONFIG_FILE" ]]; then
  echo "Usage: $0 <config_file>" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Config file not found: $CONFIG_FILE" >&2
  exit 1
fi

# Copy config NOW (at submission time) to a shared filesystem
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIG_BASENAME=$(basename "$CONFIG_FILE")
SHARED_SNAPSHOT_DIR="${GROUP_HOME:-$HOME}/project-mac-fairness-exp-root/config_snapshots"
mkdir -p "$SHARED_SNAPSHOT_DIR"
TMP_CONFIG="${SHARED_SNAPSHOT_DIR}/${CONFIG_BASENAME}_${TIMESTAMP}"
cp "$CONFIG_FILE" "$TMP_CONFIG"

# Submit with the frozen (at the slurm submission time) config path
# Example usage: ./script/submit_job.sh config/prod_vllm/bbq-sampled-set_1agent_as-ai_v2025-12-10.yaml
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch --export=ALL,TMP_CONFIG="$TMP_CONFIG" "$SCRIPT_DIR/run_job.slurm"