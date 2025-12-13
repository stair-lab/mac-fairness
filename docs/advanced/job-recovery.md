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
┌─────────────┐            ┌─────────────────┐           ┌───────────────────┐
│ Create      │            │ Mark question   │           │ All "succeeded"?  │
│ manifest    │───────────▶│ as "succeeded"  │──────────▶│                   │
│ (all null)  │            │ on completion   │           └──────┬────────────┘
└─────────────┘            └─────────────────┘                  │
                                                         ┌──────┴──────┐
                                                         │             │
                                                    Yes  ▼         No  ▼
                                                  ┌──────────┐    ┌──────────┐
                                                  │ Delete   │    │ Keep for │
                                                  │ manifest │    │ recovery │
                                                  └──────────┘    └──────────┘
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
  "job_task_id": "3983716_0",
  "experiment_name": "exp_variant_...",
  "benchmark_subcategory": "bbq_race",
  "submission_timestamp": "2025-12-12T21:33:55.799Z",
  "config_snapshot_path": "$MAC_FAIRNESS_WORKSPACE/bookkeeping/config_snapshot/...",
  "num_questions_planned": 6880,
  "num_questions_processed": 4030,
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
- `num_questions_processed`: Questions that have completed processing naturally ("succeeded" or "partial" or "failed")
- `questions[].status`: `"succeeded"` or `null` ("partial"/"failed", or not yet processed)
- `config_snapshot_path`: Path to the config used for this job (for resume)

## The repair_job.py Script

The `script/repair_job.py` utility provides two commands for job recovery.

### Analyze Command

Analyze a job manifest to see status without requiring GPU:

```bash
python script/repair_job.py analyze experiment/{benchmark_subcategory}/{experiment_name}/job_manifest/{manifest}.json [--json]
```

Example output:

```text
======================================================================
JOB MANIFEST ANALYSIS
======================================================================
Manifest: $MAC_FAIRNESS_WORKSPACE/experiment/bbq_race/.../job_manifest/..._3983716_0.json
Experiment: exp_variant_...
Benchmark: bbq_race
Config snapshot: $MAC_FAIRNESS_WORKSPACE/bookkeeping/config_snapshot/...yaml
----------------------------------------------------------------------
Questions planned: 6880
Questions processed: 6880
Succeeded: 4030
Null (not succeeded): 2850
----------------------------------------------------------------------

Null question IDs (first 20):
  - bbq_race_3128
  - bbq_race_3184
  - bbq_race_3220
  - bbq_race_3228
  - bbq_race_3260

----------------------------------------------------------------------
GPU REQUIREMENTS FOR RESUME
----------------------------------------------------------------------
Total GPUs needed: 2

Per-model breakdown:
llm_0: 2 GPU(s) (tp=2, meta-llama/Llama-3.3-70B-Instruct)

======================================================================
SUGGESTIONS
======================================================================
To resume 2850 null questions:
[ENV_VARS] python script/repair_job.py resume $MAC_FAIRNESS_WORKSPACE/experiment/bbq_race/.../job_manifest/...json [--dry-run]
```

Options:

- `--json`: Output analysis as JSON instead of formatted text

### Resume Command

Resume processing null questions from a job manifest:

```bash
# Basic resume
python script/repair_job.py resume experiment/{benchmark_subcategory}/{experiment_name}/job_manifest/{manifest}.json

# Dry run (show what would be processed without running)
python script/repair_job.py resume experiment/{benchmark_subcategory}/{experiment_name}/job_manifest/{manifest}.json --dry-run
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
ls experiment/{benchmark_subcategory}/{experiment_name}/job_manifest/
```

If a manifest exists, the job was interrupted (manifests are deleted on successful completion where ALL conversations have the `"succeeded"` status).

### 2. Analyze the manifest

```bash
python script/repair_job.py analyze experiment/{benchmark_subcategory}/{experiment_name}/job_manifest/{manifest}.json
```

This shows:

- How many questions succeeded vs need to be resumed
- GPU requirements for the resume
- The config snapshot that will be used

### 3. Resume with appropriate resources

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 \
  python script/repair_job.py resume experiment/{benchmark_subcategory}/{experiment_name}/job_manifest/{manifest}.json
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

### Resume Creates New Artifacts

Each resume run creates **new** job manifest and job summary files:

- **Job manifest**: Tracks only the questions being resumed (not the full original set)
- **Job summary**: Records statistics for only the resumed questions
- **Old manifest**: Deleted when resume starts (its null questions become the new manifest's planned questions)

This means:

- `num_questions_planned` in a resumed manifest equals the null count from the previous manifest
- Job summaries are per-run snapshots, not cumulative across resumes
- The **current job manifest** is always the source of truth for what still needs repair
- Multiple interrupted resumes create a chain of manifests, each smaller than the last

**Example**: Original run plans to process 6880 questions, 2850 not run or partial/failed. Resume #1 plans to process 2850 questions, 2 not run or partial/failed. Resume #2 processes 2 questions, both succeed. The manifests would show:

| Run       | `num_questions_planned` | Outcome                                                                                                                                               |
| --------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Original  | 6880                    | 2850 null → manifest kept, job summary frozen                                                                                                         |
| Resume #1 | 2850                    | 2 null → manifest replaced (since it's a different pid), original job summary untouched and new job summary created for Resume #1                     |
| Resume #2 | 2                       | 0 null → manifest replaced (yet another pid) and deleted on all-succeeded, previous job summaries untouched and new job summary created for Resume #2 |

### Atomic Question Completion

When a question completes, both the job manifest and `index.jsonl` are updated atomically using a file lock. This ensures consistency between the two data sources.

**Operation sequence per question:**

```text
1. save_transcript() # Can be lost on interrupt (recoverable)
2. streaming_summary.record_completion() # In-memory + disk update
3. ATOMIC (with file lock):
   a. Update job manifest status
   b. Append to index.jsonl
```

**Implementation:**

```python
# In conversation_orchestrator.py
self.bookkeeping.record_question_completion(
    manifest_path=self.manifest_path,
    question_id=question_id,
    succeeded=(status == "succeeded"),
    index_path=index_path,
    index_entry=index_entry,
)
```

**Guarantees:**

- Uses exclusive file lock (`bookkeeping/.completion.lock`)
- Blocks if another process holds the lock (no silent failures)
- Either both manifest and index are updated, or neither is
- Raises `ManifestParseError` if manifest JSON is corrupted
- Raises `ManifestWriteError` if manifest or index cannot be written

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
