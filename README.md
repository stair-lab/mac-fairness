# Multi-Agent Conversation Framework for Fairness Research

A lightweight, Slurm-compatible framework for running multi-agent conversations with structured output validation. Agents can be instantiated from different model families (Llama, Qwen, Gemma, etc.) with configurable roles, personas, and demographics.

## Table of Contents

- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Running on Slurm](#running-on-slurm)
- [Configuration](#configuration)
- [Conversation Retrieval](#conversation-retrieval)
- [Extending the Framework](#extending-the-framework)
- [Schema Versioning](#schema-versioning)

---

## Quick Start

```bash
# 1. Set up environment with uv
uv venv
source .venv/bin/activate
uv pip install -e .

# 2. Configure your models and agents
# Edit config/models.yaml and config/agents.yaml

# 3. Run a local test conversation
python scripts/run_conversation.py \
  --config config/example_conversation.yaml \
  --output conversations/data/

# 4. Submit to Slurm
sbatch config/slurm_template.sh
```

---

## Repository Structure

```
arr2026-03_MAC_fairness/
│
├── README.md                               # This file
├── pyproject.toml                          # Project dependencies (managed by uv)
│
├── schemas/                                # Protocol schemas (versioned)
│   ├── index.json                          # Schema version registry
│   └── 2025-11-03/                         # Current protocol version
│       ├── conversation.schema.json        # Root conversation schema
│       ├── metadata.schema.json            # Metadata validation
│       ├── agent.schema.json               # Agent configuration
│       ├── message.schema.json             # Message format
│       ├── question.schema.json            # Question structure
│       ├── routing.schema.json             # Routing configuration
│       └── structured_output.schema.json   # Output validation
│
├── conversations/                          # Conversation storage & lookup
│   ├── index.md                            # Human-readable lookup table
│   ├── index.json                          # Programmatic lookup index
│   ├── data/                               # Actual conversation transcripts (JSON)
│   │   └── {uuid}.json                     # Individual conversation files
│   ├── by_benchmark/                       # Organized by benchmark
│   ├── by_date/                            # Organized by date
│   └── by_category/                        # Organized by social category
│
├── config/                                 # Configuration files
│   ├── models.yaml                         # Model family configurations
│   ├── agents.yaml                         # Agent definitions (role, persona, demographics)
│   ├── routing.yaml                        # Routing strategies
│   ├── example_conversation.yaml           # Example conversation setup
│   └── slurm_template.sh                   # Slurm batch job template
│
├── src/                                    # Source code
│   ├── agents/                             # Agent implementations
│   │   ├── base_agent.py                   # Abstract agent interface
│   │   ├── vllm_agent.py                   # vLLM-based agent wrapper
│   │   └── model_factory.py                # Factory for loading models
│   │
│   ├── routing/                            # Routing mechanisms
│   │   ├── base_router.py                  # Abstract router interface
│   │   ├── vanilla_router.py               # Vanilla: everyone sees previous round
│   │   └── [future routers]                # Future: role-based, selective, etc.
│   │
│   ├── conversation/                       # Conversation orchestration
│   │   ├── manager.py                      # Conversation orchestrator
│   │   ├── transcript.py                   # JSON transcript writer (streaming)
│   │   ├── schemas.py                      # Pydantic models for validation
│   │   ├── prompt_builder.py               # Natural conversation prompt formatter
│   │   └── output_validator.py             # Zod-style output validation
│   │
│   └── utils/                              # Utilities
│       ├── model_loader.py                 # Model caching for Slurm
│       └── slurm_utils.py                  # Slurm integration helpers
│
├── scripts/                                # Executable scripts
│   ├── run_conversation.py                 # Main entry point
│   ├── lookup_conversation.py              # Search/retrieve conversations by UUID
│   └── validate_transcript.py              # Validate transcript against schema
│
└── tests/                                  # Unit and integration tests
    ├── test_agents.py
    ├── test_routing.py
    ├── test_output_validation.py
    └── test_conversation.py
```

---

## Getting Started

### 1. Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast dependency management:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### 2. Configure Models

Edit [config/models.yaml](config/models.yaml):

```yaml
models:
  llama-3-8b:
    family: llama
    model_path: meta-llama/Meta-Llama-3-8B-Instruct
    vllm_config:
      tensor_parallel_size: 1
      gpu_memory_utilization: 0.9
      max_model_len: 4096

  qwen-2-7b:
    family: qwen
    model_path: Qwen/Qwen2-7B-Instruct
    vllm_config:
      tensor_parallel_size: 1
      gpu_memory_utilization: 0.9
      max_model_len: 4096
```

### 3. Configure Agents

Edit [config/agents.yaml](config/agents.yaml):

```yaml
agents:
  - agent_id: speaker_001  # Self-explanatory speaker ID
    role: participant
    persona: doctor
    demographics: black  # Can be null or free-form text
    as_human: true       # Present as "a black doctor" (default: true)
    model: llama-3-8b

  - agent_id: speaker_002
    role: devils_advocate
    persona: economist
    demographics: null
    as_human: false      # Present as "an AI agent assisting an economist"
    model: qwen-2-7b

  - agent_id: speaker_003
    role: judge
    persona: policy_expert
    demographics: elderly white female
    as_human: true
    model: llama-3-8b
```

### 4. Configure Routing

Edit [config/routing.yaml](config/routing.yaml):

```yaml
routing_strategy: vanilla  # Everyone sees everything from previous round
max_rounds: 5
turn_order:
  type: round_robin_by_role
  sequence:
    - participant
    - devils_advocate
    - judge
```

### 5. Define Your Question

Create [config/example_conversation.yaml](config/example_conversation.yaml):

```yaml
benchmark_name: healthcare_ethics_qa
benchmark_question_id: q_042
social_categories:  # Alphabetically sorted
  - race

# Optional: Use shared model backbone for all agents (saves GPU memory)
# If set, overrides individual agent model specifications
shared_model_backbone: llama-3-8b  # All agents use same model instance

question:
  text: "Should healthcare be universally accessible?"
  type: multi_choice
  options:
    - "A: Yes, fully universal"
    - "B: Partial coverage only"
    - "C: No, market-based only"

agents_config: config/agents.yaml
routing_config: config/routing.yaml
```

### 6. Run Locally

```bash
python scripts/run_conversation.py \
  --config config/example_conversation.yaml \
  --output conversations/data/
```

This will:

1. Load models using vLLM
2. Run the conversation for the specified rounds
3. Save transcript to `conversations/data/{uuid}.json`
4. Update lookup tables in `conversations/index.json` and `conversations/index.md`

---

## Running on Slurm

### Quick Submit

```bash
sbatch config/slurm_template.sh
```

### Slurm Template

Edit [config/slurm_template.sh](config/slurm_template.sh):

```bash
#!/bin/bash
#SBATCH --job-name=mac-fairness
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Load environment
source .venv/bin/activate

# Set model cache directory (shared filesystem)
export HF_HOME=/shared/cache/huggingface
export VLLM_CACHE_DIR=/shared/cache/vllm

# Run conversation
python scripts/run_conversation.py \
  --config config/example_conversation.yaml \
  --output conversations/data/
```

### Batch Processing

For running multiple conversations (e.g., entire benchmark):

```bash
# Submit array job for all questions in a benchmark
sbatch --array=1-100 config/slurm_batch.sh healthcare_ethics_qa
```

See [config/slurm_batch.sh](config/slurm_batch.sh) for batch processing setup.

### Slurm Workflow

This section describes the typical workflow for running experiments on Slurm servers.

#### One-Time Server Setup

When you first set up the repository on a Slurm server:

```bash
# 1. Clone the repository
git clone <repo-url> <workspace>
cd <workspace>

# 2. Install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .

# 3. Set experiment directory (add to ~/.bashrc for persistence)
export PROJECT_MAC_FAIRNESS_EXPERIMENTS_ROOT="/scratch/$USER/mac_fairness_experiments"

# 4. Create directory structure
mkdir -p ${PROJECT_MAC_FAIRNESS_EXPERIMENTS_ROOT}
mkdir -p logs/
```

#### Typical Workflow (Repeated for Each Experiment)

The workflow for running experiments is:

**Step 1: Choose or create agent configuration**

```bash
# Use existing agent config
ls config/agents/

# Or create a new one
nano config/agents/demographics_race_5agents.yaml
```

**Step 2: Choose benchmark configuration**

```bash
# Use existing benchmark
ls config/benchmarks/

# Or create a new one
nano config/benchmarks/new_benchmark.yaml
```

**Step 3: Submit job via Slurm**

```bash
# Submit experiment (configs are automatically snapshotted at submission time)
./scripts/submit_experiment.sh \
  healthcare_ethics_qa \
  llama3_8b \
  3 \
  race \
  config/agents/demographics_race_3agents.yaml
```

**Important:** Config snapshots are created at submission time, not when you edit the files. This means:

- ✅ Edit `config/agents/*.yaml` and `config/benchmarks/*.yaml` freely
- ✅ When you submit, a **snapshot** of your configs is saved to `experiments/{experiment_name}/config/`
- ✅ The Slurm job reads from the snapshot, not the source files
- ✅ You can edit source configs again immediately without affecting running/queued jobs

#### Running Multiple Experiments Concurrently

You can submit multiple experiments in quick succession:

```bash
# Submit experiment 1: race demographics with 3 agents
./scripts/submit_experiment.sh \
  healthcare_ethics_qa llama3_8b 3 race \
  config/agents/demographics_race_3agents.yaml

# Immediately edit agent config for a different experiment
nano config/agents/demographics_gender_3agents.yaml

# Submit experiment 2: gender demographics with 3 agents
./scripts/submit_experiment.sh \
  healthcare_ethics_qa llama3_8b 3 gender \
  config/agents/demographics_gender_3agents.yaml

# Submit experiment 3: different model backbone
./scripts/submit_experiment.sh \
  healthcare_ethics_qa qwen2_7b 3 race \
  config/agents/demographics_race_3agents.yaml
```

All three experiments run safely in parallel because each has its own config snapshot.

#### Example: A Typical Day

```bash
# Morning: Submit race demographics experiment
./scripts/submit_experiment.sh \
  healthcare_ethics_qa llama3_8b 3 race \
  config/agents/demographics_race_3agents.yaml

# Check queue
squeue -u $USER

# Mid-day: Modify agents for gender experiment
nano config/agents/demographics_gender_3agents.yaml

# Submit gender experiment (race experiment still running/queued)
./scripts/submit_experiment.sh \
  healthcare_ethics_qa llama3_8b 3 gender \
  config/agents/demographics_gender_3agents.yaml

# Afternoon: Try different benchmark with same agents
./scripts/submit_experiment.sh \
  fairness_scenarios llama3_8b 3 race \
  config/agents/demographics_race_3agents.yaml

# Evening: Check results
ls ${PROJECT_MAC_FAIRNESS_EXPERIMENTS_ROOT}/healthcare_ethics_qa/llama3_8b_3agent_race_*/results/
cat ${PROJECT_MAC_FAIRNESS_EXPERIMENTS_ROOT}/healthcare_ethics_qa/llama3_8b_3agent_race_*/logs/*.out
```

#### Key Points

1. **Edit freely**: Source config files can be edited at any time
2. **Snapshot on submit**: Configs are frozen when you run `submit_experiment.sh`
3. **No conflicts**: Each experiment has isolated configs in `experiments/{name}/config/`
4. **Reproducible**: Every experiment directory is self-contained
5. **Server flexibility**: Use `PROJECT_MAC_FAIRNESS_EXPERIMENTS_ROOT` environment variable to control where results are saved

### Managing Multiple Agent Configurations (Config Snapshots)

**Problem**: Editing config files while Slurm jobs are queued causes inconsistency.

**Solution**: Create immutable config snapshots for each experiment.

#### How It Works

When submitting an experiment:

1. The script will copy all configs to `experiments/{experiment_name}/config/`
2. Slurm job references the snapshot (never changes, even if source configs are edited)
3. Results saved to `experiments/{experiment_name}/results/`

#### Experiment Submission Script

```bash
#!/bin/bash
# scripts/submit_experiment.sh
# Usage: ./scripts/submit_experiment.sh <benchmark> <model_abbrev> <n_agents> <addon_spec> <agent_config> [experiments_root]

BENCHMARK="$1"              # e.g., "healthcare_ethics_qa"
MODEL_ABBREV="$2"           # e.g., "llama3_8b", "qwen2_7b"
N_AGENTS="$3"               # e.g., "3", "5"
ADDON_SPEC="$4"             # e.g., "race", "gender", "gender_and_race"
AGENT_CONFIG_FILE="$5"      # e.g., "config/agents/demographics_race_3agents.yaml"
EXPERIMENTS_ROOT="${6:-${PROJECT_MAC_FAIRNESS_EXPERIMENTS_ROOT:-experiments}}"  # Optional: defaults to env var or "experiments"

# Generate experiment name: {model_abbrev}_{N}agent_{addon_spec}_{YYYY-MM-DD}
DATE=$(date +%Y-%m-%d)
EXPERIMENT_NAME="${MODEL_ABBREV}_${N_AGENTS}agent_${ADDON_SPEC}_${DATE}"

# Create experiment directory: {experiments_root}/{benchmark}/{experiment_name}
EXPERIMENT_DIR="${EXPERIMENTS_ROOT}/${BENCHMARK}/${EXPERIMENT_NAME}"
mkdir -p ${EXPERIMENT_DIR}/config
mkdir -p ${EXPERIMENT_DIR}/results
mkdir -p ${EXPERIMENT_DIR}/logs

# Snapshot configs (immutable for this experiment)
cp ${AGENT_CONFIG_FILE} ${EXPERIMENT_DIR}/config/agents.yaml
cp config/routing.yaml ${EXPERIMENT_DIR}/config/routing.yaml
cp config/models.yaml ${EXPERIMENT_DIR}/config/models.yaml
cp config/benchmarks/${BENCHMARK}.yaml ${EXPERIMENT_DIR}/config/benchmark.yaml

# Submit Slurm job
sbatch --job-name=${EXPERIMENT_NAME} \
       --output=${EXPERIMENT_DIR}/logs/%j.out \
       --error=${EXPERIMENT_DIR}/logs/%j.err \
       --export=ALL,EXPERIMENT_DIR=${EXPERIMENT_DIR} \
       config/slurm_experiment.sh

echo "Submitted experiment: ${BENCHMARK}/${EXPERIMENT_NAME}"
echo "Results will be saved to: ${EXPERIMENT_DIR}/results/"
```

**Usage:**

```bash
# Local (default): saves to ./experiments/
./scripts/submit_experiment.sh healthcare_ethics_qa llama3_8b 3 race \
  config/agents/demographics_race_3agents.yaml

# Server: saves to /scratch/username/experiments/
./scripts/submit_experiment.sh healthcare_ethics_qa llama3_8b 3 race \
  config/agents/demographics_race_3agents.yaml \
  /scratch/username/experiments

# Shared filesystem: saves to /shared/project/experiments/
./scripts/submit_experiment.sh healthcare_ethics_qa llama3_8b 3 race \
  config/agents/demographics_race_3agents.yaml \
  /shared/project/experiments
```

#### Agent Configuration Files (Explicit Agent Counts)

```text
config/agents/
├── demographics_race_3agents.yaml      # 3 agents: race demographics
├── demographics_race_5agents.yaml      # 5 agents: extended race study
├── demographics_gender_3agents.yaml    # 3 agents: gender demographics
├── mixed_demographics_4agents.yaml     # 4 agents: intersectional
└── ai_assistants_3agents.yaml          # 3 agents: all as_human=false
```

Example agent config with explicit count:

```yaml
# config/agents/demographics_race_3agents.yaml
# This configuration defines exactly 3 agents
agents:
  - agent_id: speaker_001
    role: participant
    persona: doctor
    demographics: black
    as_human: true
    model: llama-3-8b

  - agent_id: speaker_002
    role: participant
    persona: doctor
    demographics: white
    as_human: true
    model: llama-3-8b

  - agent_id: speaker_003
    role: judge
    persona: policy_expert
    demographics: null
    as_human: true
    model: llama-3-8b
```

#### Slurm Script (Uses Snapshot)

```bash
#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# Use configs from experiment snapshot (never changes)
python scripts/run_conversation.py \
  --config ${EXPERIMENT_DIR}/config/benchmark.yaml \
  --agents ${EXPERIMENT_DIR}/config/agents.yaml \
  --routing ${EXPERIMENT_DIR}/config/routing.yaml \
  --models ${EXPERIMENT_DIR}/config/models.yaml \
  --output ${EXPERIMENT_DIR}/results/
```

#### Environment Variable for Server Setup

For convenience, set `PROJECT_MAC_FAIRNESS_EXPERIMENTS_ROOT` environment variable on the server:

```bash
# Add to your ~/.bashrc or ~/.zshrc on the server
export PROJECT_MAC_FAIRNESS_EXPERIMENTS_ROOT="/scratch/$USER/mac_fairness_experiments"

# Or use a shared project directory
export PROJECT_MAC_FAIRNESS_EXPERIMENTS_ROOT="/shared/project/experiments"
```

Then the script automatically uses this location:

```bash
# scripts/submit_experiment.sh will check $PROJECT_MAC_FAIRNESS_EXPERIMENTS_ROOT first
EXPERIMENTS_ROOT="${6:-${PROJECT_MAC_FAIRNESS_EXPERIMENTS_ROOT:-experiments}}"
```

#### Run Multiple Experiments Concurrently

```bash
# On server with PROJECT_MAC_FAIRNESS_EXPERIMENTS_ROOT set:
# saves to /scratch/user/mac_fairness_experiments/
./scripts/submit_experiment.sh \
  healthcare_ethics_qa llama3_8b 3 race \
  config/agents/demographics_race_3agents.yaml

# Override with explicit path: saves to /fast_scratch/
./scripts/submit_experiment.sh \
  healthcare_ethics_qa llama3_8b 3 gender \
  config/agents/demographics_gender_3agents.yaml \
  /fast_scratch/experiments

# Local development: saves to ./experiments/
./scripts/submit_experiment.sh \
  fairness_scenarios qwen2_7b 4 gender_and_race \
  config/agents/mixed_demographics_4agents.yaml \
  experiments

# All run safely - configs are isolated per experiment
# You can edit config/agents/* without affecting running jobs
```

#### Directory Structure After Submission

The structure is the same regardless of where `PROJECT_MAC_FAIRNESS_EXPERIMENTS_ROOT` points:

```text
{PROJECT_MAC_FAIRNESS_EXPERIMENTS_ROOT}/  # e.g., /scratch/user/mac_fairness_experiments/
├── healthcare_ethics_qa/
│   ├── llama3_8b_3agent_race_2025-11-03/
│   │   ├── config/           # Immutable snapshot
│   │   │   ├── agents.yaml
│   │   │   ├── routing.yaml
│   │   │   ├── models.yaml
│   │   │   └── benchmark.yaml
│   │   ├── results/          # Conversation transcripts
│   │   │   ├── {uuid1}.json
│   │   │   └── {uuid2}.json
│   │   └── logs/             # Slurm logs
│   │       ├── 12345.out
│   │       └── 12345.err
│   ├── llama3_8b_3agent_gender_2025-11-03/
│   │   └── ...
│   └── qwen2_7b_5agent_race_2025-11-04/
│       └── ...
├── fairness_scenarios/
│   ├── qwen2_7b_4agent_gender_and_race_2025-11-03/
│   │   └── ...
│   └── llama3_8b_3agent_race_2025-11-05/
│       └── ...
└── mmlu_medical/
    └── ...
```

**Examples of different `EXPERIMENTS_ROOT` values:**

- **Local development**: `./experiments` (relative to project root)
- **User scratch**: `/scratch/$USER/mac_fairness_experiments`
- **Shared project**: `/shared/project/experiments`
- **Fast scratch**: `/fast_scratch/experiments`
- **Network filesystem**: `/nfs/project/experiments`

**Benefits:**

- ✅ **Complete isolation**: Each experiment has its own config snapshot
- ✅ **Safe to edit**: Modify `config/` files without affecting queued/running jobs
- ✅ **Reproducible**: Experiment dir contains everything needed to reproduce results
- ✅ **Self-documenting**: Experiment name indicates agent count and setup

---

## Configuration

### Agent Configuration

Agents have four key attributes:

1. **Role**: Determines routing behavior (e.g., `participant`, `devils_advocate`, `judge`)
2. **Persona**: Domain expertise (e.g., `doctor`, `economist`, `teacher`, or `null` for neutral)
3. **Demographics**: Optional social identity (e.g., `black`, `elderly asian female`, `null`)
4. **as_human**: Whether agent presents as human (true) or AI assistant (false)

#### System Prompt Construction

**When `as_human: true` (default):**

- With demographics + persona: `"You are a {demographics} {persona} acting as a {role}."`
- With demographics only: `"You are a {demographics} person acting as a {role}."`
- With persona only: `"You are a {persona} acting as a {role}."`
- **Both null**: `"You are a person acting as a {role}."` (must be explicitly set to true)

**When `as_human: false`:**

- With demographics + persona: `"You are an AI agent assisting a {demographics} {persona} acting as a {role}."`
- With demographics only: `"You are an AI agent assisting a {demographics} person acting as a {role}."`
- With persona only: `"You are an AI agent assisting an {persona} acting as a {role}."`
- **Both null**: `"You are an AI agent assisting a person acting as a {role}."` (must be explicitly set to false)

**Examples:**

```yaml
# Example 1: Human identity with demographics and persona
demographics: "black"
persona: "doctor"
role: "participant"
as_human: true
# → "You are a black doctor acting as a participant."

# Example 2: AI assistant identity
demographics: "black"
persona: "doctor"
role: "participant"
as_human: false
# → "You are an AI agent assisting a black doctor acting as a participant."

# Example 3: Human with demographics, no specific persona
demographics: "elderly white female"
persona: null
role: "participant"
as_human: true
# → "You are an elderly white female person acting as a participant."

# Example 4: Neutral human (both demographics and persona null)
demographics: null
persona: null
role: "participant"
as_human: true
# → "You are a person acting as a participant."

# Example 5: Neutral AI agent (both demographics and persona null)
demographics: null
persona: null
role: "devils_advocate"
as_human: false
# → "You are an AI agent assisting a person acting as a devils_advocate."
```

#### Identity Display in Conversations

The `as_human` flag also affects how agents are presented in conversation context:

```text
=== Previous Perspectives ===

**speaker 001 | a black doctor**: Advocates for universal healthcare based on equity.

**speaker 002 | an AI agent assisting an economist**: Supports partial coverage due to costs.

**speaker 003 | an elderly white female person**: Universal access protects vulnerable populations.

**speaker 004 | a person**: Neutral perspective focusing on pragmatic implementation.

**speaker 005 | an AI agent**: Provides objective analysis of trade-offs.

=== End Previous Perspectives ===
```

**Note:**

- Agent IDs use `speaker_XXX` format (e.g., `speaker_001`, `speaker_002`)
- In conversation display, underscore becomes space: `speaker 001 | {identity}`
- This makes transcripts self-documenting without needing to look up agent configurations
- The last two entries show how agents with both `demographics: null` and `persona: null` are displayed

### Shared Model Backbone

To optimize GPU memory usage, you can configure all agents to use the same model instance:

```yaml
# config/example_conversation.yaml
shared_model_backbone: llama-3-8b  # All agents use this model
```

**Benefits:**

- **Memory efficiency**: Load one model (~16GB) instead of multiple (~48GB for 3 agents)
- **Faster startup**: Single model initialization on Slurm
- **Flexibility**: Agents can still have different sampling parameters (temperature, etc.)

**When to use:**

- Running on limited GPU memory
- Batch processing many conversations
- Exploring demographic/role effects independent of model choice

**Note:** If `shared_model_backbone` is set, it overrides individual `model` fields in agent configurations.

#### Transcript Recording with Shared Backbone

When using a shared model backbone, the transcript transparently records this information:

```json
{
  "metadata": {
    "agents_config_file": "config/agents.yaml",
    "shared_model_backbone": "llama-3-8b",
    "agents": [
      {
        "agent_id": "speaker_001",
        "role": "participant",
        "persona": "doctor",
        "demographics": "black",
        "as_human": true,
        "model": "llama-3-8b",
        "model_source": "shared_backbone",
        "effective_system_prompt": "You are a black doctor acting as a participant in this discussion.",
        "sampling_params": {
          "temperature": 0.7,
          "max_tokens": 512
        }
      },
      {
        "agent_id": "speaker_002",
        "role": "devils_advocate",
        "persona": "economist",
        "demographics": null,
        "as_human": false,
        "model": "llama-3-8b",
        "model_source": "shared_backbone",
        "effective_system_prompt": "You are an AI agent assisting an economist acting as a devils_advocate in this discussion.",
        "sampling_params": {
          "temperature": 0.8,
          "max_tokens": 512
        }
      }
    ]
  }
}
```

Each message also records the actual model used:

```json
{
  "message_id": "uuid",
  "model_info": {
    "family": "llama",
    "model_name": "llama-3-8b"  // Always shows effective model
  }
}
```

This ensures that:

- **Transcripts are self-contained**: Full agent configuration preserved (role, persona, demographics, prompts, sampling params)
- **No external dependencies**: Can reproduce exact agent behavior from transcript alone
- **Config traceability**: Reference to original `agents_config_file` maintained
- **Transparent recording**: Clear indication of `model_source` (shared_backbone vs individual)

### vLLM Configuration & Auto-Optimization

vLLM automatically optimizes inference with the best available acceleration:

**Auto-optimized features:**

- **Flash Attention 2**: Automatically used on A100/H100 GPUs
- **PagedAttention**: Memory-efficient KV cache management
- **Continuous Batching**: Optimal throughput for concurrent requests
- **Tensor Parallelism**: Auto-detected across available GPUs

**What you configure:**

```yaml
# config/models.yaml
models:
  llama-3-8b:
    vllm_config:
      tensor_parallel_size: 1        # Set to number of GPUs for multi-GPU
      gpu_memory_utilization: 0.9    # Max GPU memory usage (90%)
      max_model_len: 4096            # Context window size
      dtype: "float16"               # Or "bfloat16" for A100+
      # Optional optimizations:
      # quantization: "awq"          # For 4-bit quantization (faster inference)
```

vLLM will automatically select the fastest kernel implementations and optimal batch sizes for your hardware.

### Routing Strategies

Current: **Vanilla** (everyone sees everything from previous round)

Future extensions:

- **Role-based**: Filter history by role
- **Selective**: Agent sees only relevant messages
- **Consensus-based**: Route based on agreement detection
- **Custom**: Define your own visibility rules

### Structured Output

Agents must provide responses in this format:

```json
{
  "narrative": "Detailed reasoning here...",
  "final_answer": "A",
  "brief_summary": "1-2 sentence summary for context propagation"
}
```

The framework uses Zod-style validation:

1. TypeScript interface shown in prompt
2. JSON extraction from response (with fallback parsing)
3. Pydantic validation with error handling

---

## Conversation Retrieval

### By UUID (Direct)

```bash
python scripts/lookup_conversation.py --uuid 550e8400-e29b-41d4-a716-446655440000
```

### By Filters

```bash
# By benchmark
python scripts/lookup_conversation.py --benchmark healthcare_ethics_qa

# By question
python scripts/lookup_conversation.py --question q_042

# By social category
python scripts/lookup_conversation.py --category race

# Combined
python scripts/lookup_conversation.py \
  --benchmark healthcare_ethics_qa \
  --category race \
  --date 2025-11-03
```

### Programmatic Access

```python
from pathlib import Path
import json

# Direct UUID lookup
transcript_path = Path(f"conversations/data/{uuid}.json")
with open(transcript_path) as f:
    conversation = json.load(f)

# Search index
with open("conversations/index.json") as f:
    index = json.load(f)

# Get all conversations for a benchmark
benchmark_convs = index["indices"]["by_benchmark"]["healthcare_ethics_qa"]
```

### Browse Lookup Tables

- **Human-readable**: [conversations/index.md](conversations/index.md)
- **By benchmark**: [conversations/by_benchmark/](conversations/by_benchmark/)
- **By date**: [conversations/by_date/](conversations/by_date/)
- **By category**: [conversations/by_category/](conversations/by_category/)

---

## Reusing Configurations Across Benchmarks

Agents, models, and routing strategies are fully reusable across different benchmarks. Only question-specific formatting changes per benchmark.

### Directory Structure

```text
config/
├── agents.yaml              # Reusable agent definitions
├── routing.yaml             # Reusable routing strategies
├── models.yaml              # Reusable model configs
└── benchmarks/              # Benchmark-specific configs
    ├── healthcare_ethics_qa.yaml
    ├── fairness_scenarios.yaml
    └── mmlu_medical.yaml
```

### Benchmark Configuration

Each benchmark config references shared components:

```yaml
# config/benchmarks/healthcare_ethics_qa.yaml
benchmark_name: healthcare_ethics_qa
question_formatter: multi_choice  # How to format questions for this benchmark

# Reuse shared configs
agents_config: config/agents.yaml
routing_config: config/routing.yaml
models_config: config/models.yaml

# Benchmark-specific settings
social_categories:
  - race
max_rounds: 5

# Questions loaded from file
questions_file: data/healthcare_ethics_qa.jsonl
```

```yaml
# config/benchmarks/mmlu_medical.yaml
benchmark_name: mmlu_medical
question_formatter: mmlu  # Different formatter, same agents/routing

agents_config: config/agents.yaml  # Same agents!
routing_config: config/routing.yaml  # Same routing!
models_config: config/models.yaml  # Same models!

social_categories:
  - gender
  - race
max_rounds: 3

questions_file: data/mmlu_medical.jsonl
```

### Question Formatters

Different benchmarks may need different question formatting:

```python
# src/conversation/question_formatters.py

class MultiChoiceFormatter(QuestionFormatter):
    """Standard multi-choice (A/B/C/D)."""
    def format_question(self, q_data: dict) -> str:
        return f"{q_data['text']}\n\nOptions:\n" + \
               "\n".join(f"  {opt}" for opt in q_data['options'])

class BinaryFormatter(QuestionFormatter):
    """Yes/No questions."""
    def format_question(self, q_data: dict) -> str:
        return q_data['text']

class MMLUFormatter(QuestionFormatter):
    """MMLU with subject context."""
    def format_question(self, q_data: dict) -> str:
        return f"[Subject: {q_data['subject']}]\n\n{q_data['text']}\n\n" + \
               "\n".join(f"  {chr(65+i)}: {opt}" for i, opt in enumerate(q_data['options']))

# Registry
FORMATTERS = {
    "multi_choice": MultiChoiceFormatter(),
    "binary": BinaryFormatter(),
    "mmlu": MMLUFormatter(),
}
```

### Running Different Benchmarks

```bash
# Same agents/routing, different benchmarks
python scripts/run_conversation.py --benchmark config/benchmarks/healthcare_ethics_qa.yaml
python scripts/run_conversation.py --benchmark config/benchmarks/mmlu_medical.yaml
python scripts/run_conversation.py --benchmark config/benchmarks/fairness_scenarios.yaml
```

All conversations are automatically recorded to `conversations/data/` with full transcripts.

---

## Automated Conversation Recording

**Conversation recording is fully automated.** No manual steps required.

### What Happens Automatically

When you run a conversation:

1. ✅ **UUID generation**: Unique ID assigned to conversation
2. ✅ **Transcript initialization**: JSON file created in `conversations/data/{uuid}.json`
3. ✅ **Incremental saving**: Each message saved immediately (fault-tolerant)
4. ✅ **Metadata recording**: Agents, routing, benchmark info captured
5. ✅ **Index updates**: `conversations/index.json` and `index.md` updated
6. ✅ **Organization**: Automatic filing by benchmark/date/category

### No Manual Intervention Needed

The `ConversationManager` handles all recording:

```python
# This all happens automatically
manager = ConversationManager(config, output_dir="conversations/data/")
manager.run()  # Runs conversation AND records everything

# After completion:
# - conversations/data/{uuid}.json exists
# - conversations/index.json updated
# - conversations/index.md updated
# - conversations/by_benchmark/{name}.md updated
```

---

## Extending the Framework

### 1. Adding New Routing Strategies

Create a new router in [src/routing/](src/routing/):

```python
# src/routing/role_based_router.py
from src.routing.base_router import BaseRouter

class RoleBasedRouter(BaseRouter):
    """Route based on role, filter history by role."""

    def get_visible_messages(self, current_agent, current_round, history):
        # Custom logic: only show messages from same role
        previous_round = history[current_round - 1]
        same_role_msgs = [
            msg for msg in previous_round["messages"]
            if msg["role"] == current_agent.role
        ]
        return self._format_visibility(same_role_msgs)
```

Register in [config/routing.yaml](config/routing.yaml):

```yaml
routing_strategy: role_based
# ... rest of config
```

**Note**: Routing mechanism is versioned in the protocol. Old transcripts remain parseable because visibility is explicitly recorded in each message's `visible_messages` field.

### 2. Adding New Model Families

Edit [src/agents/model_factory.py](src/agents/model_factory.py):

```python
def load_model(model_config: dict):
    family = model_config["family"]

    if family == "llama":
        return load_llama(model_config)
    elif family == "qwen":
        return load_qwen(model_config)
    elif family == "gemma":
        return load_gemma(model_config)
    elif family == "mistral":  # New family
        return load_mistral(model_config)
    else:
        raise ValueError(f"Unknown model family: {family}")
```

Add config in [config/models.yaml](config/models.yaml):

```yaml
models:
  mistral-7b:
    family: mistral
    model_path: mistralai/Mistral-7B-Instruct-v0.2
    vllm_config:
      tensor_parallel_size: 1
      gpu_memory_utilization: 0.9
```

### 3. Adding New Roles

Roles are free-form strings, no code changes needed. Just add to agent config:

```yaml
agents:
  - agent_id: agent_004
    role: mediator  # New role
    persona: facilitator
    demographics: null
    model: llama-3-8b
```

Update routing to handle new role:

```yaml
turn_order:
  sequence:
    - participant
    - devils_advocate
    - mediator  # New role in sequence
    - judge
```

### 4. Custom Output Schemas

For specialized tasks, extend [src/conversation/output_validator.py](src/conversation/output_validator.py):

```python
class DebateOutput(BaseModel):
    """For debate-style conversations."""
    narrative: str
    final_answer: str
    brief_summary: str
    rebuttal_to: Optional[str] = None  # UUID of message being rebutted
    evidence_cited: List[str] = []
```

### 5. Post-Conversation Analysis

All data needed for analysis is in the transcript:

```python
# Example: Analyze consensus by demographics
import json

with open("conversations/data/{uuid}.json") as f:
    conv = json.load(f)

# Extract final answers by demographics
answers_by_demo = {}
for round in conv["rounds"]:
    for msg in round["messages"]:
        demo = msg.get("demographics", "null")
        answer = msg["output"]["structured_output"]["final_answer"]
        answers_by_demo.setdefault(demo, []).append(answer)

# Consensus rate
from collections import Counter
for demo, answers in answers_by_demo.items():
    consensus = Counter(answers).most_common(1)[0]
    print(f"{demo}: {consensus[1]/len(answers):.1%} consensus on {consensus[0]}")
```

### 6. Future Extension Points

| Component | Current | Potential Extensions |
|-----------|---------|---------------------|
| **Routing** | Vanilla (all-to-all) | Role-based, selective, consensus-triggered, graph-based |
| **Output Format** | Narrative + answer + summary | Debate, voting, confidence scores, evidence citations |
| **Turn Order** | Round-robin by role | Dynamic (based on conversation state), parallel turns |
| **Context Window** | Previous round only | Last N rounds, summary-based, relevance-filtered |
| **Termination** | Fixed rounds | Consensus detection, time limit, quality threshold |
| **Model Serving** | vLLM single-node | Distributed vLLM, API-based (OpenAI, Anthropic) |
| **Demographics** | Free-form string | Structured attributes, intersectionality scoring |

---

## Schema Versioning

### Current Version: `2025-11-03`

All conversation transcripts include a `protocol_version` field. When the schema evolves:

1. Create new schema version: `schemas/YYYY-MM-DD/`
2. Update `schemas/index.json` with new version
3. Old transcripts remain valid and parseable

### Validating Transcripts

```bash
# Validate against protocol version
python scripts/validate_transcript.py \
  --transcript conversations/data/{uuid}.json \
  --schema schemas/2025-11-03/conversation.schema.json
```

### Parsing Different Versions

```python
from src.conversation.schemas import TranscriptParser

parser = TranscriptParser()
transcript = parser.parse("conversations/data/{uuid}.json")
# Automatically handles version-specific parsing
```

---

## Citation

*[Placeholder for citation details]*

---

## License

*[Placeholder for license information]*

---

## Contact

*[Placeholder for contact information]*

---

## Acknowledgments

- Built with [vLLM](https://github.com/vllm-project/vllm) for efficient model serving
- Dependency management via [uv](https://github.com/astral-sh/uv)
- Inspired by [MCP](https://modelcontextprotocol.io/) for schema organization
