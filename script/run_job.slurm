#!/bin/bash
#SBATCH --job-name=mac-fairness_PUT_STH_ELSE_HERE
#SBATCH -C GPU_SKU:H100_SXM5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/groups/sanmi/project-mac-fairness-exp-root/slurm_logs/%x_%j.out
#SBATCH --error=/home/groups/sanmi/project-mac-fairness-exp-root/slurm_logs/%x_%j.err

# Paths using environment variables
export SIF_IMAGE=$GROUP_HOME/$USER/simg/vllm-openai_v0.12.0.sif
export HF_HUB_CACHE=$GROUP_HOME/HF_HUB_CACHE
export MAC_FAIRNESS_EXPERIMENT_ROOT=$GROUP_HOME/project-mac-fairness-exp-root

# Environment variables
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
# To save storage space, can set live status to 0
export MAC_FAIRNESS_LIVE_STATUS=1

# Run the job (TMP_CONFIG passed from submit script)
singularity exec --nv \
    --env HF_HUB_CACHE=$HF_HUB_CACHE \
    --env TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM \
    --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
    --env MAC_FAIRNESS_LIVE_STATUS=$MAC_FAIRNESS_LIVE_STATUS \
    --env MAC_FAIRNESS_EXPERIMENT_ROOT=$MAC_FAIRNESS_EXPERIMENT_ROOT \
    $SIF_IMAGE python3 \
    script/run_job.py \
    "$TMP_CONFIG" \
    --grid \
    --rep 5

# (Optional) Cleanup
# rm -f "$TMP_CONFIG"