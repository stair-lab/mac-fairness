# Job Recovery

This document covers the job manifest system and how to resume interrupted experiment runs.

## Overview

Long-running experiments can be interrupted by various events (timeouts, crashes, manual stops). The job manifest system tracks question processing status to enable recovery without re-running already-completed questions.

## Job Manifests

### Purpose

Job manifests provide:

1. **Progress tracking**: Know which questions have been processed
2. **Recovery support**: Resume from where the job left off
3. **Atomic updates**: Each question status is updated independently

### Lifecycle

```text
   Job Start               Question Processing               Job End
       │                           │                            │
       ▼                           ▼                            ▼
┌─────────────┐            ┌─────────────────┐           ┌─────────────┐
│ Create      │            │ Mark question   │           │ All null?   │
│ manifest    │───────────▶│ as "succeeded"  │──────────▶│             │
│ (all null)  │            │ on completion   │           └──────┬──────┘
└─────────────┘            └─────────────────┘                  │
                                                         ┌──────┴──────┐
                                                         │             │
                                                    Yes  ▼         No  ▼
                                                  ┌──────────┐  ┌──────────┐
                                                  │ Delete   │  │ Keep for │
                                                  │ manifest │  │ recovery │
                                                  └──────────┘  └──────────┘
```

### Manifest Location and Naming

Manifests are stored at:

```text
experiment/{benchmark_subcategory}/{experiment_name}/job_manifest/{timestamp}_{job_task_id}.json
```

This follows the same naming convention as job summaries.

### Manifest Structure

```json
{
  "job_task_id": "local",
  "experiment_name": "llama33_70b_3agent_...",
  "benchmark_subcategory": "dev_vllm",
  "submission_timestamp": "2025-12-12T21:33:55.799Z",
  "config_snapshot_path": "$MAC_FAIRNESS_WORKSPACE/bookkeeping/config_snapshot/...",
  "num_questions_planned": 1024,
  "num_questions_processed": 317,
  "questions": [
    {"question_id": "bbq_race_0", "index": 0, "status": "succeeded"},
    {"question_id": "bbq_race_1", "index": 1, "status": "succeeded"},
    {"question_id": "bbq_race_2", "index": 2, "status": null},
    ...
  ],
  "created_at": "2025-12-12T21:33:55.810Z"
}
```

Key fields:

- `num_questions_planned`: Total questions in the job
- `num_questions_processed`: Questions that have completed processing (succeeded or failed)
- `questions[].status`: `"succeeded"` or `null` (not yet processed)
- `config_snapshot_path`: Path to the config used for this job (for resume)

## The repair.py Script

The `script/repair.py` utility provides two commands for job recovery.

### Analyze Command

Analyze a job manifest to see status without requiring GPU:

```bash
python script/repair.py analyze experiment/{benchmark}/{experiment}/job_manifest/{manifest}.json [--json]
```

Example output:

```text
======================================================================
JOB MANIFEST ANALYSIS
======================================================================
Manifest: $MAC_FAIRNESS_WORKSPACE/experiment/bbq_race/.../job_manifest/...json
Experiment: gemma2_27b_3agent_as-hybrid-demographics-persona_vanilla_v2025-12-10
Benchmark: bbq_race
Config snapshot: $MAC_FAIRNESS_WORKSPACE/bookkeeping/config_snapshot/...yaml
----------------------------------------------------------------------
Questions planned: 1024
Questions processed: 317
Succeeded: 317
Null (not succeeded): 707
----------------------------------------------------------------------

Null question IDs (first 20):
  - bbq_race_100
  - bbq_race_1000
  ...

----------------------------------------------------------------------
GPU REQUIREMENTS FOR RESUME
----------------------------------------------------------------------
Total GPUs needed: 1

Per-model breakdown:
gemma2_27b: 1 GPU(s) (tp=1, google/gemma-2-27b-it)

======================================================================
SUGGESTIONS
======================================================================
To resume 707 null questions:
  [ENV VAR SETTINGS HERE] python script/repair.py resume $MAC_FAIRNESS_WORKSPACE/...
```

Options:

- `--json`: Output analysis as JSON instead of formatted text

### Resume Command

Resume processing null questions from a manifest:

```bash
# Basic resume
python script/repair.py resume experiment/{benchmark}/{experiment}/job_manifest/{manifest}.json

# Dry run (show what would be processed without running)
python script/repair.py resume experiment/{benchmark}/{experiment}/job_manifest/{manifest}.json --dry-run
```

What happens during resume:

1. Reads the manifest to find questions with `null` status
1. Loads the config snapshot referenced in the manifest
1. Deletes the old manifest (a new one will be created)
1. Runs the orchestrator with only the null question IDs
1. A new manifest is created tracking the resumed questions

## Typical Recovery Workflow

### 1. Identify interrupted jobs

Look for job_manifest files in the experiment directory:

```bash
ls experiment/{benchmark}/{experiment}/job_manifest/
```

If a manifest exists, the job was interrupted (manifests are deleted on successful completion).

### 2. Analyze the manifest

```bash
python script/repair.py analyze experiment/{benchmark}/{experiment}/job_manifest/{manifest}.json
```

This shows:

- How many questions succeeded vs need to be resumed
- GPU requirements for the resume
- The config snapshot that will be used

### 3. Resume with appropriate resources

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 \
  python script/repair.py resume experiment/{benchmark}/{experiment}/job_manifest/{manifest}.json
```

### 4. Repeat if needed

If the resume is also interrupted, a new manifest will exist. Repeat the analyze/resume cycle until all questions complete.

## Implementation Details

### Question ID Filtering

The `ConversationOrchestrator.run_experiment()` method accepts an optional `question_ids` parameter:

```python
# Normal run (all questions)
await orchestrator.run_experiment()

# Resume run (specific questions)
await orchestrator.run_experiment(question_ids={"bbq_race_100", "bbq_race_101", ...})
```

When `question_ids` is provided:

- All questions are loaded from the questions file
- Only questions with matching IDs are processed
- A new manifest is created with only those questions

### Manifest Updates

Manifests are updated atomically after each question completes:

```python
# In conversation_orchestrator.py
self.bookkeeping.mark_question_processed(
    self.manifest_path, question_id, succeeded=(status == "succeeded")
)
```

This ensures progress is saved even if the job crashes mid-execution.

### Config Snapshot Reuse

The resume command uses the **original config snapshot** referenced in the manifest, ensuring:

- Same model configuration
- Same agent definitions
- Same retry settings
- Reproducible conditions

## Best Practices

1. **Always analyze before resuming**: Check GPU requirements and question counts
1. **Use dry-run first**: Verify the correct questions will be processed
1. **Monitor resumed jobs**: They can also be interrupted
1. **Clean up old manifests**: After successful completion, manifests are auto-deleted
