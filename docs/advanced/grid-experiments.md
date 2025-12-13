# Grid Experiments

This document covers running parameter sweep experiments using grid configurations.

## Overview

Grid experiments allow you to run multiple experiment configurations from a single config file. The system:

1. Expands parameter combinations into individual configurations
2. Runs each configuration sequentially
3. Tracks progress in a grid manifest for crash recovery
4. Supports resuming interrupted grid runs

## Grid Configuration

### Structure

A grid config file has a special `_grid` section that defines parameter sweeps:

```yaml
_grid:
  # Derivation rules: compute values from other fields
  derive:
    experiment_metadata.questions_file: "data/BBQ/{experiment_metadata.benchmark_subcategory}.jsonl"

  # Sweep parameters: all combinations will be generated
  sweep:
    experiment_metadata.experiment_name:
      - exp_variant
    prompt_template_config.for_participant.choice_display_format:
      - bullet
      - roman_dot
      - letter_colon
      - none

  # Broadcast: apply same value to multiple paths (linked sweep)
  broadcast:
    all_agent_temps:
      paths:
        - agent_definitions.0.temperature
        - agent_definitions.1.temperature
      values:
        - 0.0
        - 0.7

  # Zip: paired values that change together (not Cartesian product)
  zip:
    model_definitions:
      - model_definitions.llm_0.model_path: meta-llama/Llama-3.3-70B-Instruct
        model_definitions.llm_0.vllm_config.tensor_parallel_size: 2
      - model_definitions.llm_0.model_path: google/gemma-2-27b-it
        model_definitions.llm_0.vllm_config.tensor_parallel_size: 1

# Base configuration (same as regular config)
experiment_metadata:
  experiment_name: _ # Will be overridden by sweep
  # ...

model_definitions:
  llm_0:
    model_path: _ # Will be overridden by sweep
    # ...
```

### Sweep vs Broadcast vs Zip

- **sweep**: Each parameter sweeps independently, creating a Cartesian product of all values
- **broadcast**: Apply the same value to multiple config paths simultaneously (linked sweep)
- **zip**: Group multiple parameters that must change together as paired value sets

Use `broadcast` when you want to apply the same value to multiple locations, e.g., setting the same temperature for all agents.

Use `zip` when parameters are interdependent and must change together, e.g., model path and its corresponding tensor parallel size.

### Expansion

The grid expander creates all combinations of sweep, broadcast, and zip parameters.

Each expanded config is a complete, valid experiment configuration.

**Example**: If you have:

- 2 sweep values for `choice_display_format`
- 2 broadcast values for `temperature`
- 2 zip value sets for `model_config`

You get 2 x 2 x 2 = 8 configurations total.

## Running Grid Experiments

### Basic Usage

```bash
# Run all configurations in a grid
python script/run_experiment.py config/my_grid_config.yaml --grid

# Dry run to see expanded configurations
python script/run_experiment.py config/my_grid_config.yaml --grid --dry-run
```

### Environment Variables

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=32 \
MAC_FAIRNESS_LIVE_STATUS=1 \
python script/run_experiment.py config/my_grid_config.yaml --grid
```

### Output

Each grid configuration:

- Gets its own experiment directory
- Generates its own job manifest, transcripts, and job summary
- Uses a unique `job_task_id` format: `{pid}_{grid_index}` (e.g., `12345_0`, `12345_1`)

## Grid Manifests

### Purpose

Grid manifests track progress across all configurations in a grid run:

```text
bookkeeping/grid_manifest/{timestamp}_{pid}.json
```

The grid config is also snapshotted at:

```text
bookkeeping/grid_config_snapshot/{config_name}_{timestamp}.yaml
```

### Structure of Grid Manifest

```json
{
  "grid_config_path": "config/dev_vllm/my_grid_config.yaml",
  "grid_config_snapshot_path": "$MAC_FAIRNESS_WORKSPACE/bookkeeping/grid_config_snapshot/...",
  "pid": 12345,
  "submission_timestamp": "2025-12-13T10:00:00.000Z",
  "num_runs_planned": 8,
  "num_runs_processed": 2,
  "experiment_runs": [
    {
      "run_id": 0,
      "experiment_name": "exp_variant",
      "grid_sweep_specs": {
        // ...
      },
      "status": "processed"
    },
    {
      "run_id": 1,
      "experiment_name": "exp_variant",
      "grid_sweep_specs": {
        // ...
      },
      "status": "started"
    },
    {
      "run_id": 2,
      "experiment_name": "exp_variant",
      "grid_sweep_specs": {
        // ...
      },
      "status": null
    }
    // ...
  ]
}
```

Note: Each individual run's `job_task_id` is `{pid}_{run_id}` (e.g., `12345_0`, `12345_1`).

Status values:

- `null`: Not yet started
- `"started"`: Started but not completed (may have partial progress)
- `"processed"`: Completed job

### Automatic Cleanup

Grid manifests and their corresponding grid config snapshots are deleted when all configurations complete successfully. If any fail, both persist for resume.

## Resuming Grid Experiments

### When to Resume

Resume a grid run when:

- The process was interrupted (Ctrl+C, timeout, crash)
- Some configurations failed and you want to retry them
- The grid manifest still exists in `bookkeeping/grid_manifest/`

### Resume Command

Resume **requires** using the grid config snapshot path (not the original config):

```bash
# Find your snapshot
ls bookkeeping/grid_config_snapshot/

# Resume using the snapshot path
python script/run_experiment.py bookkeeping/grid_config_snapshot/{config_name}_{timestamp}.yaml --grid --resume
```

This ensures the same parameter combinations are used even if you've edited the original config file.

### What Happens During Resume

1. **Load grid manifest**: Finds the existing manifest for the config file
1. **Identify pending/started configurations**: Skips already-processed ones
1. **For started configurations**:
   - Finds the job manifest from the previous run
   - Identifies questions with `null` status (not succeeded)
   - Resumes only those questions using the original config snapshot
1. **For pending configurations**: Runs them fresh
1. **Update grid manifest**: Marks configurations as processed

### Resume Flow Diagram

```text
Grid Resume
    │
    ▼
┌─────────────────┐
│ Load grid       │
│ manifest        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Config status?  │────▶│ processed       │──▶ Skip
└────────┬────────┘     └─────────────────┘
         │
         │ pending
         ▼
┌─────────────────┐
│ Run fresh       │
└─────────────────┘
         │
         │ started
         ▼
┌─────────────────┐
│ Find job        │
│ manifest        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Null questions? │────▶│ None            │──▶ Skip (all succeeded)
└────────┬────────┘     └─────────────────┘
         │
         │ Some null
         ▼
┌─────────────────┐
│ Resume with     │
│ question_ids    │
└─────────────────┘
```

## Relationship to Job Recovery

Grid resume builds on the [job recovery](job-recovery.md) system but uses different input paths:

| Feature        | `repair_job.py`                               | `--grid --resume`                                     |
| -------------- | --------------------------------------------- | ----------------------------------------------------- |
| **Scope**      | Single job manifest                           | Grid manifest (multiple configs)                      |
| **Input path** | `experiment/.../job_manifest/{manifest}.json` | `bookkeeping/grid_config_snapshot/{config}_{ts}.yaml` |
| **Use case**   | Resume non-grid run                           | Resume grid run                                       |

### Input Path Summary

**For non-grid experiments** — use `repair_job.py` with the **job manifest path**:

```bash
python script/repair_job.py resume experiment/{benchmark}/{exp_name}/job_manifest/{manifest}.json
```

**For grid experiments** — use `--grid --resume` with the **grid config snapshot path**:

```bash
python script/run_experiment.py bookkeeping/grid_config_snapshot/{config}_{timestamp}.yaml --grid --resume
```

> **Important**: Do NOT use the original config path for grid resume. The snapshot path ensures identical parameter expansion even if you've edited the original config.

## Best Practices

### 1. Use Dry Run First

```bash
# See what configurations will run
python script/run_experiment.py config/my_grid_config.yaml --grid --dry-run
```

### 2. Start Small

Test with a subset before running the full grid:

```yaml
_grid:
  sweep:
    "experiment_metadata.experiment_name":
      - "test_only" # Just one value for testing
```

### 3. Monitor Progress

Grid runs print progress for each configuration:

```text
============================================================
Configuration 3/8 (index 2): exp_variant
============================================================
```

### 4. Check Grid Manifest on Failure

If a grid run fails, check the manifest:

```bash
ls bookkeeping/grid_manifest/
cat bookkeeping/grid_manifest/my_grid_config_*.json | python -m json.tool
```

### 5. Resume Promptly

Resume interrupted runs using the snapshot path:

```bash
# Find and use the snapshot path
ls bookkeeping/grid_config_snapshot/
python script/run_experiment.py bookkeeping/grid_config_snapshot/{config}_{timestamp}.yaml --grid --resume
```

## Troubleshooting

### "No existing manifest found"

The resume command couldn't find a grid manifest for this config. Either:

- The grid run completed successfully (manifest was deleted)
- The grid run never started
- You're using the wrong config file

### "All configurations already completed"

All configurations in the grid manifest are marked as processed. The grid run is complete.

### Configuration Skipped but Questions Failed

If a configuration is marked "processed" but has failed questions, use `repair_job.py` to repair that specific job:

```bash
# Find the job manifest
ls experiment/{benchmark_subcategory}/{experiment_name}/job_manifest/

# Analyze and resume
python script/repair_job.py analyze experiment/{benchmark_subcategory}/{experiment_name}/job_manifest/{manifest}.json
python script/repair_job.py resume experiment/{benchmark_subcategory}/{experiment_name}/job_manifest/{manifest}.json
```
