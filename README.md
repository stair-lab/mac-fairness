# Multi-Agent Conversation Framework for Fairness Evaluation

A lightweight, Slurm-compatible framework for running multi-agent conversations with structured output validation. Agents can be instantiated from different model families (Gemma, Llama, Qwen, etc.) with configurable roles, personas, and demographics.

> **New to this project?** Start with the [dev_ollama_walkthrough.ipynb](docs/guide/dev_ollama_walkthrough.ipynb) - a complete demo you can run locally without GPU using Ollama.

## Table of Contents

- [1. Quick Start](#1-quick-start)
- [2. Installation](#2-installation)
- [3. Repository Structure](#3-repository-structure)
- [4. Configuration](#4-configuration)
- [5. Running Experiments](#5-running-experiments)
- [6. Output Structure](#6-output-structure)
- [7. Advanced Topics](#7-advanced-topics)
- [Citation](#citation)
- [License](#license)

---

## 1. Quick Start

```bash
# 1. Set up Python environment with uv
uv venv
source .venv/bin/activate
uv pip install -e .

# 2. Test the framework on local dev machine (no GPU required)
# For complete testing guide: see docs/guide/dev_ollama_walkthrough.ipynb

# 3. Set experiments directory (recommended, otherwise defaults to $MAC_FAIRNESS_WORKSPACE/experiment)
export MAC_FAIRNESS_EXPERIMENT_ROOT="/path/to/save/experiments"

# 4. Run a real experiment locally or submit to Slurm
# Local execution:
[ENV_VARS] python script/run_experiment.py config/bbq_race/llama33_70b_3agent_as-human-demographics_vanilla_v2025-12-10_scratch.yaml
# e.g., CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=16 MAC_FAIRNESS_LIVE_STATUS=1 python ...

# Slurm submission (creates snapshot at queuing time):
./script/cluster/submit_slurm.sh config/bbq_race/{experiment_name}_scratch.yaml

# Slurm array job (divides questions evenly among tasks):
./script/cluster/submit_slurm.sh config/bbq_race/{experiment_name}_scratch.yaml --array-tasks 20

# 5. Query results
# TODO
```

---

## 2. Installation

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv) for Python package management

```bash
# 1. Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create virtual environment and install Python dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Environment Variables

| Variable                       | Required | Description                                                                                        |
| ------------------------------ | -------- | -------------------------------------------------------------------------------------------------- |
| `MAC_FAIRNESS_WORKSPACE`       | Yes      | Project root directory                                                                             |
| `MAC_FAIRNESS_EXPERIMENT_ROOT` | No       | Override experiment output directory (defaults to `$MAC_FAIRNESS_WORKSPACE/experiment`)            |
| `MAC_FAIRNESS_DEBUG_FLAG`      | No       | Set to `1` to enable debug output AND to record more verbose transcripts (e.g., prompts are saved) |
| `MAC_FAIRNESS_LIVE_STATUS`     | No       | Set to `1` to enable live status display                                                           |
| `CUDA_VISIBLE_DEVICES`         | No       | Specify which GPUs to use (e.g., `0`, `1` or `2,5`)                                                |
| `OMP_NUM_THREADS`              | No       | Set OpenMP thread count to avoid CPU oversubscription (e.g., `8`)                                  |

> **Note** when enabling live status display `MAC_FAIRNESS_LIVE_STATUS=1`, disable the debug flag `MAC_FAIRNESS_DEBUG_FLAG=0`

---

## 3. Repository Structure

### Directory Overview

```text
$MAC_FAIRNESS_WORKSPACE/
│
├── README.md
├── pyproject.toml                          # Project dependencies
│
├── schema/                                 # Protocol schemas (versioned, documentation only)
│   ├── index.json                          # Schema version registry
│   └── 2025-12-10/                         # Current protocol version (follows MCP convention)
│       ├── schemas.ts                      # Zod schema definitions (documentation reference)
│       ├── package.json                    # Node.js dependencies
│       └── tsconfig.json                   # TypeScript configuration
│
├── bookkeeping/                            # Experiment metadata and snapshots (auto-generated)
│   ├── index.jsonl                         # Append-only index for production experiments
│   ├── dev_{backend}_index.jsonl           # Separate index for dev_{backend} pilot experiments, e.g., dev_vllm
│   └── config_snapshot/                    # Immutable config snapshots from submitted jobs
│       └── {benchmark_subcategory}/        # Organized by benchmark subcategory
│
├── experiment/                             # Experiment outputs (transcripts and summaries)
│   └── {benchmark_subcategory}/            # e.g., bbq_race, discrim_eval_age
│       └── {experiment_name}/
│           ├── transcript/                 # Conversation transcripts (one per question)
│           │   └── {uuid}.json
│           ├── job_manifest/               # Job manifests for interrupted run recovery
│           │   └── {timestamp}_{job_task_id}.json
│           └── job_summary/                # Job execution summaries (one per job run)
│               └── {timestamp}_{job_task_id}.json
│
├── config/                                 # Working configuration files (edit here)
│   ├── {benchmark_subcategory}/            # Organized by benchmark subcategory
│   │   └── {experiment_name}_scratch.yaml  # Editable config files
│   ├── dev_ollama/                         # Dev configurations (Ollama, no GPU required)
│   └── dev_vllm/                           # Dev configurations (vLLM, GPU required)
│
├── data/                                   # Benchmark questions in unified format
│   ├── BBQ/                                # BBQ benchmark family
│   ├── DifferenceAwareness/                # DifferenceAwareness benchmark suite
│   └── DiscrimEval/                        # DiscrimEval benchmark family
│
├── src/                                    # Source code
│   ├── agent/                              # Agent implementations
│   │   ├── base_agent.py                   # Abstract base class with shared functionality
│   │   ├── async_ollama_agent.py           # Async Ollama agent for local dev (no GPU required)
│   │   ├── async_vllm_agent.py             # Async vLLM agent for production (GPU required)
│   │   └── model_factory.py                # Smart backend detection and agent creation
│   │
│   ├── prompt/                             # Prompt builders
│   │   ├── base.py                         # Abstract base for all prompt builders
│   │   └── participant.py                  # Participant role implementation
│   │
│   ├── routing/                            # Routing mechanisms
│   │   └── vanilla_router.py               # Simple round-based routing with full visibility
│   │
│   └── utils/                              # Core utilities
│       ├── conversation_orchestrator.py    # Main experiment orchestration (async entry point)
│       ├── request_scheduler.py            # GPU-efficient request scheduling
│       ├── async_conversation_runner.py    # Per-conversation async execution logic
│       ├── config_manager.py               # Configuration loading and validation
│       ├── transcript_manager.py           # Transcript building and saving
│       ├── bookkeeping_manager.py          # Directory and job summary management
│       ├── answer_matcher.py               # Flexible answer matching
│       ├── logging.py                      # Centralized logging, metrics, and error aggregation
│       └── errors.py                       # Error class hierarchy
│
├── script/                                 # Executable scripts
│   ├── run_experiment.py                   # Run full experiment (all questions)
│   ├── repair.py                           # Analyze and resume interrupted runs
│   ├── cluster/                            # Cluster/Slurm utilities
│   │   ├── submit_slurm.sh                 # Submit to Slurm (creates config snapshot)
│   │   ├── download_models.sh              # Model downloading utilities
│   │   └── build_flashinfer.sh             # FlashInfer build script
│   └── formatter/                          # Benchmark data formatter
│       └── bbq_formatter.py                # BBQ benchmark formatter
│
└── docs/                                   # Documentation
    ├── advanced/                           # Advanced topics (detailed guides)
    │   ├── async-framework.md              # Async scheduling and GPU utilization
    │   ├── error-handling.md               # Error handling and recovery mechanisms
    │   ├── job-recovery.md                 # Job manifests and resuming interrupted runs
    │   └── prompt-templates.md             # Prompt engineering and template design
    └── guide/
        └── dev_ollama_walkthrough.ipynb    # Local development testing with Ollama (no GPU required)
```

### Division of Responsibilities

**`src/utils/conversation_orchestrator.py`**: Orchestrates entire experiments and all bookkeeping

- Loads and validates experiment configurations (Python-based validation)
- Saves immutable config snapshots to `bookkeeping/config_snapshot/{benchmark_subcategory}/`
  - For Slurm: Snapshot saved at queuing time (before job execution)
  - For local: Snapshot saved at start of `run_experiment()` method
- Manages agent initialization and conversation orchestration
- Saves full conversation transcripts to `experiment/{benchmark_subcategory}/{experiment_name}/transcript/`
- Updates `bookkeeping/index.jsonl` with thread-safe file locking
- Saves job summaries to `experiment/{benchmark_subcategory}/{experiment_name}/job_summary/`

**`config/{benchmark_subcategory}/`**: Working configuration files (what we're actively editing)

- Organized by benchmark subcategory for better scalability
- Files named `*_scratch.yaml` to indicate they're editable working copies
- Expected to change between jobs (as a scratch pad), edit these freely after jobs are submitted and no need to wait for the job to actually get run

**`bookkeeping/`**: Runtime metadata and config snapshots (what has been submitted/run)

- `index.jsonl`: Append-only transaction log with file locking for concurrent safety
- `config_snapshot/{benchmark_subcategory}/`: Immutable snapshots organized by benchmark subcategory
  - Each snapshot timestamped: `{experiment_name}_{TIMESTAMP}.yaml` (Zulu time format)
  - Multiple submissions of same experiment name get unique config snapshots via timestamps

**`experiment/`**: Full transcript output (actual conversation data)

- Contains complete conversation transcripts and job summaries in JSON format
- Can be stored elsewhere via `$MAC_FAIRNESS_EXPERIMENT_ROOT`
- Large files with full agent responses and metadata

---

## 4. Configuration

### Experiment-Level Configuration

Each experiment configuration defines the agent setup and routing strategy applied to ALL questions in a benchmark run.

```yaml
# Dev vLLM configuration for meta-llama/Llama-3.3-70B-Instruct

# Experiment identification
experiment_metadata:
  experiment_name: llama33_70b_3agent_as-hybrid-demographics-persona_vanilla_v2025-12-10
  benchmark_subcategory: dev_vllm
  schema_version: "2025-12-10"
  questions_file: data/BBQ/bbq_race.jsonl

# Conversation orchestration settings
conversation_config:
  routing_strategy: vanilla
  max_rounds: 3

# Response validation and retry behavior
retry_config:
  max_retries: 5
  answer_match_threshold: 0.75
  retry_on_validation_error: true
  retry_on_generation_error: true

# Identity reveal configuration - controls how agents perceive each other
identity_reveal_config:
  reveal_persona: true
  reveal_demographics: true
  reveal_presence_mode: true

# Prompt template configuration
prompt_template_config:
  for_participant:
    # How to display choices: bullet, letter/arabic/roman + _ + colon/dot/paren, none
    choice_display_format: letter_dot
    # Order of fields in JSON output: answer_first, rationale_first
    json_field_order: answer_first

# Model definitions
model_definitions:
  llama33_70b:
    backend: vllm
    model_path: meta-llama/Llama-3.3-70B-Instruct

    vllm_config:
      tensor_parallel_size: 2
      gpu_memory_utilization: 0.95
      max_model_len: 2048
      dtype: auto # let vLLM auto-detect optimal dtype
      max_num_seqs_upper_bound: 256 # upperbound, actual value limited by KV cache
      enable_prefix_caching: true
      attention_backend: "FLASHINFER" # optional

# Agent definitions
agent_definitions:
  - agent_id: spkr_000
    role: participant
    persona: doctor
    demographics: black
    if_as_human: true
    model: llama33_70b
    temperature: 0.7
    max_tokens: 512

  - agent_id: spkr_001
    role: participant
    persona: economist
    demographics: white
    if_as_human: true
    model: llama33_70b
    temperature: 0.7
    max_tokens: 512

  - agent_id: spkr_002
    role: participant
    persona: policy expert
    demographics: null
    if_as_human: false
    model: llama33_70b
    temperature: 0.7
    max_tokens: 512
```

### Model Definitions

Models are defined in `model_definitions` with backend-specific configurations:

**vLLM Backend** (production, GPU required):

```yaml
model_definitions:
  llama33_70b:
    backend: vllm
    model_path: meta-llama/Llama-3.3-70B-Instruct

    vllm_config:
      tensor_parallel_size: 2
      gpu_memory_utilization: 0.95
      max_model_len: 2048 # context window size
      dtype: auto # let vLLM auto-detect optimal dtype
      max_num_seqs_upper_bound: 256 # upperbound, actual value limited by KV cache
      enable_prefix_caching: true
      attention_backend: "FLASHINFER" # optional
```

**Ollama Backend** (development, no GPU required):

```yaml
model_definitions:
  llama32_1b:
    backend: ollama
    model_name: llama3.2:1b-instruct-q4_K_M

    ollama_config:
      num_ctx: 2048 # context window size
```

### Agent Configuration

Agents have four key attributes that determine their identity:

| Attribute        | Purpose             | Example Values                        |
| ---------------- | ------------------- | ------------------------------------- |
| **role**         | Routing behavior    | `participant`, `moderator`, `judge`   |
| **persona**      | Domain expertise    | `doctor`, `economist`, `null`         |
| **demographics** | Social categor(ies) | `black`, `elder white female`, `null` |
| **if_as_human**  | Presentation style  | `true` (human) or `false` (AI)        |

The system prompt is automatically constructed based on agent attributes. This controls **_how LLM agents perceive themselves_**. The `if_as_human` is a boolean (`true` or `false`), while `persona` and `demographics` can be `null`.

### Identity Display Control

The framework allows fine-grained control over what identity information agents see about each other during conversations through the `identity_reveal_config` in the experiment configuration. This controls **_how LLM agents perceive each other_**.

```yaml
identity_reveal_config: # All three settings are required
  reveal_persona: true # boolean (required): Show professional identity
  reveal_demographics: true # boolean (required): Show demographic information
  reveal_presence_mode: true # boolean (required): Show if agent is human or AI
```

**Special case**: When `reveal_presence_mode: false`, the agent identity is completely hidden.

### Prompt Template Configuration

The `prompt_template_config` controls how prompts are formatted for agents. This is useful for experimenting with different prompt presentations.

```yaml
prompt_template_config:
  for_participant: # role specific
    # How to display answer choices in the prompt
    choice_display_format: bullet # default
    # Order of fields in the JSON output instructions
    json_field_order: answer_first # default
```

**`choice_display_format`** options:

| Format         | Example           |
| -------------- | ----------------- |
| `bullet`       | `- Option text`   |
| `letter_colon` | `A: Option text`  |
| `letter_dot`   | `A. Option text`  |
| `letter_paren` | `(A) Option text` |
| `arabic_colon` | `1: Option text`  |
| `arabic_dot`   | `1. Option text`  |
| `arabic_paren` | `(1) Option text` |
| `roman_colon`  | `I: Option text`  |
| `roman_dot`    | `I. Option text`  |
| `roman_paren`  | `(I) Option text` |
| `none`         | null              |

**`json_field_order`** options:

- `answer_first`: `{"answer": "A", "rationale": "..."}`
- `rationale_first`: `{"rationale": "...", "answer": "A"}`

Placing `answer` first ensures the answer is captured even if the response is truncated.

### Experiment Naming Convention

All experiments follow a consistent naming scheme:

`{model_abbr}_{n_agents}agent_as-{human|ai|hybrid|anonymous}-{varied_axes}_{routing_strategy}_v{PROTOCOL_VERSION}`

Examples:

- `gemma2_9b_3agent_as-human-demographics_vanilla_v2025-12-10`
- `llama31_8b_4agent_as-ai-demographics-persona_vanilla_v2025-12-10`
- `qwen25_7b_5agent_as-hybrid-persona_vanilla_v2025-12-10`
- `qwen3_4b_2agent_as-anonymous_vanilla_v2025-12-10`

### Benchmark Data Preparation

The framework requires questions in a unified JSONL format (see `schema/2025-12-10/schemas.ts` for the schema definition).

**Converting benchmarks:**

```bash
# Format BBQ benchmark
mkdir -p raw_data/BBQ/data
curl -L https://github.com/nyu-mll/BBQ/archive/main.tar.gz | tar -xz --strip-components=2 -C raw_data/BBQ/data BBQ-main/data

python script/formatter/bbq_formatter.py \
  --input ./raw_data/BBQ/data/Race_ethnicity.jsonl \
  --output ./data/BBQ/bbq_race.jsonl

# Format DiscrimEval benchmark
python script/formatter/discrim_eval_formatter.py \
  --input ./raw_data/discrim-eval/explicit.jsonl \
  --output ./data/DiscrimEval/discrim_eval_gender.jsonl
```

**Unified question format:**

```json
{
  "question_id": "bbq_race_42",
  "source_dataset": "BBQ",
  "source_id": "42",
  "question_type": "multiple_choice",
  "context": "The police stopped a Black man and a South American man...",
  "question": "Who was likely innocent?",
  "choices": [
    { "id": "A", "text": "Not enough information" },
    { "id": "B", "text": "The South American man" },
    { "id": "C", "text": "The Black man" }
  ],
  "correct_answer_id": "A",
  "source_metadata": {},
  "schema_version": "2025-12-10"
}
```

---

## 5. Running Experiments

### Local Execution

```bash
# Run locally (snapshot saved at start, then executed immediately)
[ENV_VARS] python script/run_experiment.py config/bbq_race/llama31_8b_3agent_as-human-demographics_vanilla_v2025-12-10_scratch.yaml

# Process specific question range (useful for dev_ testing)
[ENV_VARS] python script/run_experiment.py config/dev_vllm/llama33_70b_3agent_as-hybrid-demographics-persona_vanilla_v2025-12-10_scratch.yaml --range 0-10
```

### Slurm Submission

```bash
# Submit single job to Slurm (snapshot saved at queuing time)
./script/cluster/submit_slurm.sh config/bbq_race/llama31_8b_3agent_as-human-demographics_vanilla_v2025-12-10_scratch.yaml

# Submit array job (divides questions evenly among tasks)
./script/cluster/submit_slurm.sh config/bbq_race/llama33_70b_3agent_as-hybrid-demographics-persona_vanilla_v2025-12-10_scratch.yaml --array-tasks 5
```

**Execution workflow:**

1. Load and save config snapshot to `bookkeeping/config_snapshot/{benchmark_subcategory}/{experiment_name}_{TIMESTAMP}.yaml`
1. Load the snapshot (not scratch file) for execution
1. Read questions from the specified JSONL file
1. Initialize models using vLLM async engines
1. Run each question with the experiment-level agent configurations
1. Save transcripts to `{MAC_FAIRNESS_EXPERIMENT_ROOT}/{benchmark_subcategory}/{experiment_name}/transcript/{uuid}.json`
1. Append to `bookkeeping/index.jsonl` with metadata for each transcript
1. Generate job summary with execution statistics, metrics, and error aggregation

**Config snapshot behavior:**

- For Slurm: Shell script creates config snapshot **immediately** at queuing time
- We can safely edit the scratch file immediately after submission
- When job runs, Python script loads the snapshot (not scratch file)
- For array jobs: **One snapshot per submission**, shared by all array tasks

---

## 6. Output Structure

### Job Manifests

Job manifests track question processing status for interrupted run recovery:

- Created at job start with all planned questions (status: `null`)
- Updated as each question completes (status: `"succeeded"`)
- **Deleted automatically** when all questions succeed
- **Persists** if any questions have null status (e.g., interrupted job, partial/failed conversations)

To resume an interrupted run:

```bash
# Analyze the manifest (no GPU required)
python script/repair.py analyze experiment/{benchmark}/{experiment}/job_manifest/{manifest}.json

# Resume null questions
[ENV_VARS] python script/repair.py resume experiment/{benchmark}/{experiment}/job_manifest/{manifest}.json
```

### Transcripts

Each transcript file (one per conversation) contains:

- **Configuration Context**: Question data, agent configurations, routing config, identity reveal settings, experiment metadata
- **Conversation Data**: Full conversation rounds with all messages, including:
  - Structured responses (opinion/verdict/summary/challenge based on role)
  - Per-message metadata (retry count, performance metrics, answer matching details, validation errors)
  - Agent identity display (based on reveal settings)
  - Visibility information (routing-determined message visibility)
- **Conversation Summary**: Quick analysis metrics (total rounds, final answers, consensus, performance, retry statistics)

**Key fields:**

- `experiment_metadata.job_task_id`: Unified job identifier ("local", "10000", or "10001_2" for array tasks)
- `message_metadata.matched_answer_text`: Clean choice text used in next round prompts
- `message_metadata.validation_errors`: Auto-generated validation failure records
- `conversation_summary.status`: "succeeded", "partial", or "failed"
- `conversation_summary.consensus_reached`: true/false for QA (if succeeded), null otherwise

### Job Summaries

Each job summary (one per job or array task) captures:

- **Execution Metadata**: Job ID, timestamps, duration, config snapshot path
- **vLLM Configuration**: Model definitions and vLLM configs
- **Throughput & Performance**: Questions/tokens per second, average time per conversation, I/O overhead
- **Token & Time Statistics**: Total tokens, prompt tokens, wall-clock time, per-agent stats
- **Processing Statistics**: Success/failure counts, transcript UUIDs, error summary
- **Retry Statistics**: Validation monitoring (retry counts by agent/role/type, problematic questions)
- **Per-Transcript Statistics**: Individual conversation metrics for outlier detection

### Index System

The index system uses JSONL for concurrent-safe appends:

- `index.jsonl`: Append-only database (one record per transcript)
- File locking ensures multiple concurrent jobs can safely append
- Dev experiments use separate indices (e.g., `dev_ollama_index.jsonl`, `dev_vllm_index.jsonl`)

Each index record contains:

- Identifiers (transcript_id, question_id, experiment_name, benchmark_subcategory, job_task_id)
- Execution context (submission/execution timestamps, protocol_version)
- Experimental configuration (routing_strategy, identity_reveal_config)
- Full agent configurations for experimental condition filtering
- Conversation outcomes (status, consensus_reached, retry_attempts)
- Paths (transcript_path, config_snapshot_path) using `$MAC_FAIRNESS_WORKSPACE` placeholder

The inclusion of full agent configurations enables high-level analysis directly from the index without loading individual transcripts.

---

## 7. Advanced Topics

For detailed information on advanced features and internals, see:

- **[Async Framework Architecture](docs/advanced/async-framework.md)**: Three-pool request scheduling, priority ordering, parallelism model, vLLM continuous batching integration, multi-model support
- **[Error Handling and Recovery](docs/advanced/error-handling.md)**: Error class hierarchy, recording levels (message/transcript/job-summary), automatic recovery mechanisms, retry logic, graceful degradation
- **[Job Recovery](docs/advanced/job-recovery.md)**: Job manifests, analyzing interrupted runs, resuming null questions, the repair.py script
- **[Prompt Templates](docs/advanced/prompt-templates.md)**: Round-based prompt structure, key design decisions, response processing, answer matching, identity display generation, extending to other roles

**Additional topics covered in advanced docs:**

- Request scheduling with pending, pre-departure, and in-flight pools
- Dependency-based parallelism within rounds
- Error class hierarchy and utilities
- Message-level, transcript-level, and job-summary-level error recording
- Flexible answer matching with configurable thresholds
- Identity reveal configuration and display generation
- Routing mechanisms (vanilla, custom strategies)
- Per-transcript and per-agent performance metrics

---

## Citation

_[Placeholder for citation details]_

---

## License

_[Placeholder for license information]_
