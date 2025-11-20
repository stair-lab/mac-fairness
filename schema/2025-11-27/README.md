# Protocol Version 2025-11-27 Schema Documentation

This directory contains the schema definitions for the Multi-Agent Conversation Framework protocol version `2025-11-27`. The schemas define the structure and validation rules for all data flowing through the framework, from question inputs to conversation transcripts.

## Architecture

The framework uses Zod (TypeScript runtime validation library) as the single source of truth for all schema definitions. This ensures type safety, runtime validation, and automatic generation of documentation.

```text
schemas.ts (Zod definitions)
    ├── TypeScript types (auto-inferred)
    ├── JSON schemas (auto-generated)
    └── Runtime validation (Zod direct)
```

## Core Schemas

### 1. Question Schema

Defines the unified format for benchmark questions across different fairness evaluation datasets.

**Required fields (enforced by Zod validation)**:

- `question_id` (string): Unified question ID across benchmarks
  - Format: `{benchmark_subcategory}_{source_id}` (all lowercase)
  - Examples: `"bbq_race_42"`, `"discrim_eval_age_15"`
- `source_dataset` (string): Name of the source benchmark
  - Examples: `"BBQ"`, `"DiscrimEval"`, `"DifferenceAwareness"`
- `source_id` (string): Original ID from the source dataset (if no explicit id, use line number in the original JSONL file)
- `question_type` (enum): Type of question format
  - Valid values: `"multiple_choice"`, `"binary"` (QA-oriented, can be extended to `"open_ended"`)
- `question` (string): The actual question text
- `choices` (array): Answer choices with IDs
  - `id` (string): Choice identifier
    - For `"binary"` and `"multiple_choice"`: Must be single capital letters (`"A"`, `"B"`, `"C"`, etc.)
    - For `"open_ended"`: Can use `"_"`
  - `text` (string): The choice text/content
- `correct_answer_id` (string): Ground truth answer ID
  - For `"binary"` and `"multiple_choice"`: Must be a single capital letter (`"A"`, `"B"`, `"C"`, etc.)
  - For `"open_ended"`: Can use `"_"`
- `schema_version` (string): Schema version in YYYY-MM-DD format
  - Current: `"2025-11-27"`

**Optional fields**:

- `context` (string): Context or narrative that precedes the question
- `source_metadata` (object): Original benchmark-specific metadata
  - This preserves benchmark-specific fields for downstream evaluation
  - Fields vary by benchmark (`"BBQ"` vs `"DiscrimEval"` vs `"DifferenceAwareness"`)
  - Examples of fields: `question_polarity`, `context_condition`, `category`, `target_group`, `stereotype_direction`

### 2. Agent Schema

Defines agent configuration and identity attributes.

**Required fields:**

- `agent_id` (string): Unique identifier for the agent
  - Pattern: `spkr_XXX` where XXX is a 3-digit number
  - Examples: `"spkr_001"`, `"spkr_002"`
- `role` (string): Agent's role in the conversation (determines routing behavior)
  - Examples: `"participant"`, `"judge"`, `"moderator"`, `"devils_advocate"`
- `as_human` (boolean): Presentation style
  - `true` = human actor
  - `false` = AI assistant
- `model` (string): Model identifier
  - Use `"shared"` to reference the shared_model_backbone from experiment config
  - Or specify a model name directly (e.g., `"llama31_8b"`, `"gemma2_9b"`)
- `temperature` (number): Sampling temperature for response generation
  - Range: 0.0 to 2.0
- `max_tokens` (integer): Maximum tokens to generate in responses
  - Minimum: 1

**Optional fields:**

- `persona` (string or null): Professional identity or domain expertise
  - Examples: `"doctor"`, `"economist"`, `"policy_expert"`, `"teacher"`
  - Set to `null` if not specified
- `demographics` (string or null): Social category or categories
  - Examples: `"black"`, `"white"`, `"white female"`, `"asian male"`
  - Set to `null` if not specified

### 3. Structured Output Schema

Discriminated union defining different response types based on agent roles. The `response_type` field determines which variant is expected.

**Response type: `"participant"`** (standard agent with opinion)

Required fields:

- `response_type` (string): Must be `"participant"`
- `opinion` (string): The agent's opinion/answer (e.g., `"A"`, `"B"`, `"C"` for multiple choice)
- `rationale` (string): The reasoning behind this response (minimum 1 character)

Optional fields:

- `confidence` (number): Confidence score, range 0.0 to 1.0
- `references` (array of strings): References to other agents' messages
  - Format: `msg_<round>_<speaker>` (e.g., `"msg_0_001"`, `"msg_1_002"`)

**Response type: `"judge"`** (evaluates other agents' arguments)

Required fields:

- `response_type` (string): Must be `"judge"`
- `verdict` (string): The judge's verdict or decision (minimum 1 character)
- `rationale` (string): The reasoning behind this response (minimum 1 character)

Optional fields:

- `confidence` (number): Confidence score, range 0.0 to 1.0
- `references` (array of strings): References to other agents' messages (format: `msg_<round>_<speaker>`)
- `evaluations` (object): Evaluations of specific agents' arguments
  - Keys: agent IDs matching pattern `spkr_XXX` (e.g., `"spkr_001"`)
  - Values: evaluation text (string)

**Response type: `"moderator"`** (summarizes discussion)

Required fields:

- `response_type` (string): Must be `"moderator"`
- `summary` (string): Summary of the discussion (minimum 1 character)
- `rationale` (string): The reasoning behind this response (minimum 1 character)

Optional fields:

- `confidence` (number): Confidence score, range 0.0 to 1.0
- `references` (array of strings): References to other agents' messages (format: `msg_<round>_<speaker>`)
- `consensus_level` (number): Level of consensus, range 0.0 (none) to 1.0 (complete)

**Response type: `"devils_advocate"`** (challenges positions)

Required fields:

- `response_type` (string): Must be `"devils_advocate"`
- `challenge` (string): The challenge or counter-argument (minimum 1 character)
- `target_position` (string): The position being challenged (minimum 1 character)
- `rationale` (string): The reasoning behind this response (minimum 1 character)

Optional fields:

- `confidence` (number): Confidence score, range 0.0 to 1.0
- `references` (array of strings): References to other agents' messages (format: `msg_<round>_<speaker>`)

### 4. Message Schema

Defines individual messages within conversations.

**Required fields:**

- `message_id` (string): Unique identifier for the message
  - Format: `msg_<round>_<speaker>` where round is 0-based and seq is 3-digit sequence
  - Examples: `"msg_0_001"`, `"msg_0_002"`, `"msg_1_001"`
- `agent_id` (string): ID of the agent who generated this message
  - Pattern: `spkr_XXX` (e.g., `"spkr_001"`)
- `agent_role` (string): Role of the agent (for routing purposes)
- `round_number` (integer): Conversation round when this message was sent
  - 0-indexed (starts at 0)
- `structured_response` (object): The agent's structured response
  - Must be a valid Structured Output (see Structured Output Schema)
  - Type determined by `response_type` field
- `visible_to` (array of strings): List of agent IDs that can see this message
  - Determined by routing strategy
  - Each element matches pattern `spkr_XXX`
- `message_metadata` (object): Performance and validation metrics
  - `tokens_generated` (integer, required): Number of tokens in the generated response (minimum 0)
  - `generation_time_ms` (number, required): Wall-clock time in milliseconds including all retries (minimum 0)
  - `temperature_used` (number, required): Temperature parameter used for generation
  - `exceeded_max_tokens` (boolean, required): Flag indicating if response was clamped at max_tokens limit
  - `retry_count` (integer, required): Number of retry attempts needed for valid structured output (minimum 0)
  - `validation_errors` (array, optional): Validation errors encountered during retries
    - Each error has:
      - `attempt` (integer): Attempt number when this error occurred (minimum 1)
      - `error` (string): Error message from Zod validation
      - `zod_path` (array): Path in schema where validation failed

**Optional fields:**

- `agent_identity_display` (string): Auto-generated display text based on agent config and reveal settings
  - Examples: `"a black doctor"`, `"an AI agent assisting an economist"`, `"a person"`
  - Omit to show only agent_id

### 5. Identity Reveal Settings Schema

Controls what identity information agents see about each other.

**Required fields (all booleans):**

- `reveal_persona` (boolean): Show professional identity/persona if specified
- `reveal_demographics` (boolean): Show demographic information if specified
- `reveal_as_human` (boolean): Show whether agent is human or AI assistant

### 6. Routing Schema

Defines conversation flow control.

**Required fields:**

- `strategy` (string): Name of the routing strategy
  - Examples: `"vanilla"`, `"role_based"`
- `max_rounds` (integer): Maximum number of conversation rounds
  - Minimum: 1

**Optional fields:**

- `parameters` (object): Strategy-specific configuration parameters
  - Contents vary by strategy

### 7. Conversation Schema

Complete transcript structure containing all conversation data.

**Required fields:**

- `transcript_id` (string): UUID for this transcript
- `protocol_version` (string): Schema version used
  - Pattern: YYYY-MM-DD format (e.g., `"2025-11-27"`)
- `experiment_metadata` (object): Experiment execution metadata (all sub-fields required)
  - `experiment_name` (string): Name of the experiment configuration
  - `benchmark_subcategory` (string): Benchmark subcategory being run
    - Examples: `"bbq_race"`, `"discrim_eval_age"`, `"mocktest"`
  - `config_snapshot_path` (string): Path to the configuration snapshot file
  - `submission_timestamp` (string): When the job was submitted (ISO 8601 with Z)
  - `execution_timestamp` (string): When the conversation was executed (ISO 8601 with Z)
  - `job_task_id` (string): Unified job identifier
    - Examples: `"local"`, `"10001"`, `"10001_2"` (for Slurm array tasks)
- `question` (object): The question being discussed (see Question Schema)
- `routing_config` (object): Routing configuration used (see Routing Schema)
- `identity_reveal_settings` (object): Controls what identity information agents see (see Identity Reveal Settings Schema)
- `agents` (array): Agent configurations (see Agent Schema)
- `conversation_rounds` (array): The actual conversation
  - Each round has:
    - `round_number` (integer): Round index (0-based, minimum 0)
    - `messages` (array): Messages sent in this round (see Message Schema)
- `conversation_summary` (object): Summary statistics for quick analysis (all sub-fields required)
  - `total_rounds` (integer): Number of conversation rounds completed
  - `total_messages` (integer): Total messages sent across all rounds
  - `status` (enum): Conversation completion status
    - Valid values: `"success"`, `"partial"`, `"failed"`
  - `consensus_reached` (boolean or null): Consensus outcome
    - `true`/`false` for QA if success, `null` for open-ended or non-success
  - `final_answers` (object): Final answer from each agent
    - Keys: agent IDs (pattern `spkr_XXX`)
    - Values: answers in terms of `id`'s in question `choices`
  - `performance_metrics` (object): Performance metrics (all sub-fields required)
    - `total_tokens` (number): Total tokens generated in this conversation
    - `total_prompt_tokens` (number): Total prompt tokens processed
    - `total_time_seconds` (number): Total time for this conversation
    - `average_response_time_ms` (number): Average time per message in milliseconds
  - `retry_statistics` (object): Retry statistics for structured output validation (all sub-fields required)
    - `total_retry_attempts` (integer): Total retry attempts across all messages
    - `messages_requiring_retries` (integer): Number of messages that needed retries
    - `validation_errors_summary` (array): Summary of validation errors encountered
      - Each entry has:
        - `error` (string): The validation error message
        - `count` (integer): How many times this error occurred
- `created_at` (string): When the transcript was created (ISO 8601 with Z)

### 8. Metadata Schema

Index record structure for querying experiments. Used in `bookkeeping/index.jsonl` files.

**Required fields:**

- `transcript_id` (string): UUID of the transcript file
- `experiment_name` (string): Name of the experiment configuration
- `benchmark_subcategory` (string): Benchmark subcategory being run
  - Examples: `"bbq_race"`, `"discrim_eval_age"`, `"mocktest"`
- `question_id` (string): ID of the question being answered
  - Examples: `"bbq_race_400"`, `"discrim_eval_age_015"`
- `job_task_id` (string): Unified job identifier
  - Examples: `"local"`, `"10001"`, `"10001_2"` (Slurm job ID or array task ID)
- `agent_config_axes` (array of strings): Configuration axes varied in experiment
  - Examples: `["as_human", "demographics"]`, `["persona"]`
- `submission_timestamp` (string): When the job was submitted (ISO 8601 with Z)
- `execution_timestamp` (string): When the conversation was executed (ISO 8601 with Z)
- `transcript_path` (string): Path to the full transcript file
- `config_snapshot_path` (string): Path to the configuration snapshot
- `protocol_version` (string): Protocol version
  - Pattern: YYYY-MM-DD format (e.g., `"2025-11-27"`)
- `routing_strategy` (string): Routing strategy used
  - Examples: `"vanilla"`, `"role_based"`
- `identity_reveal_settings` (object): Settings controlling identity visibility (see Identity Reveal Settings Schema)
- `n_agents` (integer): Number of agents in conversation (minimum 1)
- `agents` (array): Full agent configurations for experimental condition filtering (see Agent Schema)
- `status` (enum): Conversation completion status
  - Valid values: `"success"`, `"partial"`, `"failed"`
- `consensus_reached` (boolean or null): Whether consensus was reached (at the index level, not necessary to include `final_answers`)
- `total_rounds_completed` (integer): Number of rounds completed
- `retry_attempts` (integer): Total retry attempts for structured output validation

**Optional fields:**

- `shared_model_backbone` (string): Shared model backbone if used
  - Examples: `"llama31_8b"`, `"gpt-4-0613"`

## File Organization

### Source Files (Edit These)

- `schemas.ts` - Zod schema definitions (single source of truth)
- `generate-json-schemas.ts` - JSON schema generator
- `validate.ts` - Runtime validation script
- `test-validation.ts` - Validation test suite
- `package.json` - Node.js dependencies
- `tsconfig.json` - TypeScript configuration

### Generated Files (Auto-Generated)

Do not edit these files directly. They are generated from schemas.ts:

- `*.schema.json` - JSON schema files for documentation/tooling
- `dist/` - Compiled JavaScript

## Development Workflow

### Initial Setup

```bash
cd schema/2025-11-27
npm install        # Install dependencies
npm run build      # Compile TypeScript
```

### Modifying Schemas

1. Edit `schemas.ts` with your changes
2. Regenerate all artifacts:

   ```bash
   npm run regenerate  # build + generate
   ```

3. Test validation:

   ```bash
   npm run test
   ```

4. Commit both source and generated files

### Available Commands

```bash
npm install      # Install dependencies
npm run build    # Compile TypeScript
npm run generate # Generate JSON schemas from Zod
npm run regenerate # Build + generate
npm run test     # Run validation tests
npm run clean    # Remove build artifacts
```

## Validation Usage

### TypeScript/Node.js

```typescript
import { schemas } from './schemas.js';

// Validate data
const result = schemas.agent.safeParse(agentData);
if (result.success) {
  console.log('Valid:', result.data);
} else {
  console.error('Errors:', result.error);
}
```

### Command Line

```bash
# Validate JSON from stdin
echo '{"response_type": "participant", "opinion": "A", "rationale": "Because..."}' | \
  node dist/validate.js structured_output

# Validate file contents
cat transcript.json | node dist/validate.js conversation
```

### Python Integration

The framework's Python code validates data by calling the Node.js validator:

```python
from src.conversation.manager import ZodValidator

validator = ZodValidator()
success, data, errors = validator.validate('agent', agent_data)
```

This ensures data integrity and enables systematic analysis of agent behavior across different experimental conditions.

## Schema Evolution

The protocol version (2025-11-27) follows the YYYY-MM-DD convention. When schemas need to evolve:

1. Create a new version directory (e.g., schema/2025-12-24/)
2. Update schema/index.json with the new version
3. Maintain backward compatibility for existing transcripts
4. Update SCHEMA_VERSION constant in schemas.ts

All data files include their schema version to ensure proper validation and migration paths.
