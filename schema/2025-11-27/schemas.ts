/**
 * Zod Schema Definitions - Version 2025-11-27
 *
 * PURPOSE:
 * This file is the single source of truth for all data schemas in the multi-agent conversation framework.
 * All TypeScript types and JSON schemas are automatically derived from these Zod definitions.
 *
 * SCHEMA OVERVIEW:
 * - QuestionSchema: Unified question format across all fairness benchmarks (BBQ, DiscrimEval, etc.)
 * - AgentSchema: Agent configuration (role, persona, demographics, model parameters)
 * - StructuredOutputSchema: Discriminated union for agent responses (participant, judge, moderator, devils_advocate)
 * - MessageSchema: Individual message structure in multi-agent conversations
 * - IdentityRevealSettingsSchema: Controls what identity information agents see about each other
 * - RoutingSchema: Conversation routing strategy configuration
 * - RetryConfigSchema: Retry behavior for structured output validation
 * - ConversationSchema: Complete conversation transcript with all metadata
 * - MetadataSchema: Index entry structure for efficient querying
 *
 * VALIDATION PIPELINE:
 * 1. schemas.ts (this file) - Define Zod schemas with validation rules
 * 2. generate-json-schemas.ts - Generate JSON Schema files from Zod (optional, for documentation)
 * 3. validate.ts - Runtime validation CLI tool used by Python code
 * 4. test-validation.ts - Comprehensive test suite for all schemas
 *
 * USAGE IN PYTHON:
 * Python code calls validate.ts via subprocess to validate all data structures:
 * - Question data loaded from JSONL files
 * - Agent responses (structured output)
 * - Complete conversation transcripts
 * - Index metadata entries
 *
 * EXTENDING SCHEMAS:
 * When adding new fields or validation rules:
 * 1. Update the Zod schema in this file
 * 2. Run `npm run build` to compile TypeScript
 * 3. Run `npm run generate` to regenerate JSON schemas (optional)
 * 4. Run `npm run test` to verify all test cases pass
 * 5. Update test-validation.ts with new test cases
 *
 * SCHEMA VERSIONING:
 * This version (2025-11-27) follows the YYYY-MM-DD convention from Model Context Protocol (MCP).
 * Breaking changes require a new schema version directory (e.g., schema/2026-01-15/).
 */

import { z } from 'zod';

// ========================================
// Question Schema
// ========================================

const ChoiceSchema = z.object({
  id: z.string().describe("Choice identifier (e.g., 'A', 'B', 'C', or '_' for open-ended questions)"),
  text: z.string().describe("The choice text/content")
});

export const QuestionSchema = z.object({
  // Core fields shared across all benchmarks (minimal and unified)
  question_id: z.string().describe("Unified question ID in format {benchmark_subcategory}_{source_id} (e.g., 'bbq_race_42', 'discrim_eval_age_15')"),
  source_dataset: z.string().describe("Name of the source benchmark (e.g., 'BBQ', 'DiscrimEval', 'DifferenceAwareness')"),
  source_id: z.string().describe("Original ID from the source dataset (if no explicit id, use line number in the original JSONL file)"),
  question_type: z.enum(["multiple_choice", "binary"]).describe("Type of question format (focusing on QA for now, can be extended to 'open_ended' questions)"),
  context: z.string().optional().describe("Optional context or narrative that precedes the question"),
  question: z.string().describe("The actual question text"),
  choices: z.array(ChoiceSchema).describe("Available answer choices with capital letter IDs or '_' for open-ended questions"),
  correct_answer_id: z.string().describe("Ground truth answer ID (e.g., 'C' for QA or '_' for open-ended)"),

  // Source-specific metadata (varies by benchmark, not unified)
  source_metadata: z.record(z.unknown()).optional().describe("Original benchmark metadata (not unified across benchmarks)"),

  // Schema version
  schema_version: z.string().regex(/^\d{4}-\d{2}-\d{2}$/).describe("Schema version in YYYY-MM-DD format")
}).strict().refine(
  (data) => {
    // For binary and multiple_choice questions, choice IDs must be capital letters (A, B, C, etc.), not "_"
    if (data.question_type === "binary" || data.question_type === "multiple_choice") {
      return data.choices.every(choice => /^[A-Z]$/.test(choice.id));
    }
    return true;
  },
  {
    message: "For 'binary' and 'multiple_choice' questions, all choice IDs must be single capital letters (A, B, C, etc.); for 'open_ended' questions the choice ID must be '_'",
    path: ["choices"]
  }
).refine(
  (data) => {
    // For binary and multiple_choice questions, correct_answer_id must also be a capital letter
    if (data.question_type === "binary" || data.question_type === "multiple_choice") {
      return /^[A-Z]$/.test(data.correct_answer_id);
    }
    return true;
  },
  {
    message: "For 'binary' and 'multiple_choice' questions, correct_answer_id must be a single capital letter (A, B, C, etc.); for 'open_ended' questions the correct_answer_id must be '_'",
    path: ["correct_answer_id"]
  }
); // additionalProperties: false

// ========================================
// Agent Schema
// ========================================

export const AgentSchema = z.object({
  agent_id: z.string().regex(/^spkr_\d{3}$/).describe("Unique identifier for the agent (e.g., 'spkr_001', 'spkr_002')"),
  role: z.string().describe("Agent's role in the conversation (determines routing behavior)"), // e.g., "participant", "judge", "moderator", "devils_advocate"
  persona: z.union([z.string(), z.null()]).describe("Domain expertise or professional identity (null if not specified)"),
  demographics: z.union([z.string(), z.null()]).describe("Social category/categories (null if not specified)"),
  if_as_human: z.boolean().describe("Presentation style: true = human actor, false = AI assistant"),
  model: z.string().describe("Model identifier ('shared' to use shared_model_backbone, or specific model name)"),
  temperature: z.number().min(0.0).max(2.0).describe("Sampling temperature for response generation"),
  max_tokens: z.number().int().min(1).describe("Maximum tokens to generate in responses")
}).strict(); // additionalProperties: false

// ========================================
// Structured Output Schema (Discriminated Union)
// ========================================

// Base schema shared by all response types
const BaseResponseSchema = z.object({
  rationale: z.string().min(1).describe("The reasoning behind this response"),
  confidence: z.number().min(0.0).max(1.0).optional().describe("Optional confidence score (0.0 to 1.0)"),
  references: z.array(
    z.string().regex(/^msg_\d+_\d{3}$/)
  ).optional().describe("Optional references to other agents' messages (format: msg_<round>_<speaker>)")
});

// Discriminated union for different agent roles
export const StructuredOutputSchema = z.discriminatedUnion('response_type', [
  // Standard participant response
  BaseResponseSchema.extend({
    response_type: z.literal('participant').describe("Standard participant response with opinion"),
    opinion: z.string().min(1).describe("The agent's opinion/answer (e.g., 'A', 'B', 'C' for multiple choice)")
  }).strict(), // additionalProperties: false

  // Judge/evaluator response
  BaseResponseSchema.extend({
    response_type: z.literal('judge').describe("Judge evaluating other agents' arguments"),
    verdict: z.string().min(1).describe("The judge's verdict or decision"),
    evaluations: z.record(
      z.string().regex(/^spkr_\d{3}$/),
      z.string()
    ).optional().describe("Optional evaluations of specific agents' arguments")
  }).strict(), // additionalProperties: false

  // Moderator response
  BaseResponseSchema.extend({
    response_type: z.literal('moderator').describe("Moderator summarizing discussion"),
    summary: z.string().min(1).describe("Summary of the discussion"),
    consensus_level: z.number().min(0).max(1).optional().describe("Level of consensus (0=none, 1=complete)")
  }).strict(), // additionalProperties: false

  // Devil's advocate response
  BaseResponseSchema.extend({
    response_type: z.literal('devils_advocate').describe("Devil's advocate challenging positions"),
    challenge: z.string().min(1).describe("The challenge or counter-argument"),
    target_position: z.string().min(1).describe("The position being challenged")
  }).strict() // additionalProperties: false
]);

// ========================================
// Message Schema
// ========================================

// Answer match info schema for flexible answer matching
// Note: No 'matched' boolean field - consumers check match_details[0].match_score >= threshold
// This avoids potential inconsistency bugs between redundant fields
const AnswerMatchInfoSchema = z.object({
  original_answer: z.string().describe("The original answer text from the agent"),
  match_details: z.array(z.object({
    text: z.string().describe("Choice text"),
    id: z.string().describe("Choice ID (e.g., 'A', 'B', 'C')"),
    match_score: z.number().min(0).max(1).describe("Similarity score (0.0-1.0)"),
    match_type: z.enum(["exact", "case_insensitive", "fuzzy", "below_threshold"]).describe("Match type for this choice")
  })).describe("All choices ranked by match score descending; check [0].match_score >= threshold to determine match")
}).strict();

// Performance metrics within message metadata
const PerformanceMetricsSchema = z.object({
  generation_time_ms: z.number().min(0).describe("Wall-clock time for generating this message in milliseconds"),
  prompt_tokens: z.number().int().min(0).describe("Number of prompt tokens processed"),
  tokens_generated: z.number().int().min(0).describe("Number of tokens in the generated response")
}).strict();

const MessageMetadataSchema = z.object({
  retry_count: z.number().int().min(0).describe("Number of retry attempts needed for valid structured output"),
  performance: PerformanceMetricsSchema.describe("Performance metrics for this message"),
  answer_match_info: AnswerMatchInfoSchema.optional().describe("Answer matching information (for participant responses)"),
  matched_answer_text: z.string().optional().describe("The matched choice text (for quick reference)"),
  validation_errors: z.array(z.object({
    attempt: z.number().int().min(1).describe("Attempt number when this error occurred"),
    error: z.string().describe("Error message from validation"),
    zod_path: z.array(z.union([z.string(), z.number()])).optional().describe("Path in schema where validation failed")
  })).optional().describe("Validation errors encountered during retries")
}).strict(); // additionalProperties: false

export const MessageSchema = z.object({
  message_id: z.string().regex(/^msg_\d+_\d{3}$/).describe("Unique identifier (format: msg_<round>_<speaker>, e.g., msg_0_001)"),
  agent_id: z.string().regex(/^spkr_\d{3}$/).describe("ID of the agent who generated this message"),
  agent_role: z.string().describe("Role of the agent (for routing purposes)"),
  agent_identity_display: z.string().optional().describe("Auto-generated based on agent config and experiment reveal settings. Examples: 'a black doctor', 'an AI agent assisting an economist', 'a person'. Omit to show only agent_id"),
  round_id: z.number().int().min(0).describe("Conversation round when this message was sent (0-indexed)"),
  structured_response: StructuredOutputSchema.describe("The agent's structured response containing all content (type determined by response_type field)"),
  visible_to: z.array(
    z.string().regex(/^spkr_\d{3}$/)
  ).describe("List of agent IDs that can see this message (determined by routing strategy)"),
  message_metadata: MessageMetadataSchema.describe("Optional metadata about message generation")
}).strict(); // additionalProperties: false

// ========================================
// Identity Reveal Settings Schema
// ========================================

export const IdentityRevealSettingsSchema = z.object({
  reveal_persona: z.boolean().describe("Show professional identity/persona if specified"),
  reveal_demographics: z.boolean().describe("Show demographic information if specified"),
  reveal_presence_mode: z.boolean().describe("Show whether agent is human or AI assistant")
}).strict().describe("Controls what identity information agents see about each other");

// ========================================
// Routing Schema
// ========================================

export const RoutingSchema = z.object({
  strategy: z.string().describe("Name of the routing strategy (e.g., 'vanilla', 'role_based')"),
  max_rounds: z.number().int().min(1).describe("Maximum number of conversation rounds"),
  parameters: z.record(z.unknown()).optional().describe("Strategy-specific configuration parameters")
}).strict(); // additionalProperties: false

// ========================================
// Retry Configuration Schema
// ========================================

export const RetryConfigSchema = z.object({
  max_retries: z.number().int().min(0).describe("Maximum retry attempts per agent response"),
  answer_match_threshold: z.number().min(0).max(1).describe("Similarity threshold for accepting answers (0.0-1.0)"),
  retry_on_validation_error: z.boolean().describe("Whether to retry when response format is invalid"),
  retry_on_generation_error: z.boolean().describe("Whether to retry when generation fails")
}).strict(); // additionalProperties: false

// ========================================
// Conversation Schema (full transcript)
// ========================================

const ConversationRoundSchema = z.object({
  round_id: z.number().int().min(0).describe("Round index (0-based)"),
  messages: z.array(MessageSchema).describe("Messages sent in this round")
});

export const ConversationSchema = z.object({
  transcript_id: z.string().uuid().describe("UUID for this transcript"),
  protocol_version: z.string().regex(/^\d{4}-\d{2}-\d{2}$/).describe("Schema version used"),
  experiment_metadata: z.object({
    experiment_name: z.string().describe("Name of the experiment configuration"),
    benchmark_subcategory: z.string().describe("Benchmark subcategory being run (e.g., 'bbq_race', 'discrim_eval_age')"),
    config_snapshot_path: z.string().describe("Path to the configuration snapshot file"),
    submission_timestamp: z.string().datetime().describe("When the job was submitted (ISO 8601 with Z)"),
    execution_timestamp: z.string().datetime().describe("When the conversation was executed (ISO 8601 with Z)"),
    job_task_id: z.string().describe("Unified job identifier: 'local', Slurm job ID, or array task ID (e.g., '10001_2')")
  }).strict().describe("Experiment execution metadata"),
  question: QuestionSchema.describe("The question being discussed"),
  routing_config: RoutingSchema.describe("Routing configuration used"),
  retry_config: RetryConfigSchema.describe("Retry configuration for structured output validation"),
  identity_reveal_config: IdentityRevealSettingsSchema.describe("Controls what identity information agents see about each other"),
  agents: z.array(AgentSchema).describe("Agent configurations"),
  conversation_rounds: z.array(ConversationRoundSchema).describe("The actual conversation"),
  conversation_summary: z.object({
    total_rounds: z.number().int().describe("Number of conversation rounds completed"),
    total_messages: z.number().int().describe("Total messages sent across all rounds"),
    status: z.enum(["succeeded", "partial", "failed"]).describe("Conversation completion status"),
    final_answers: z.record(
      z.string().regex(/^spkr_\d{3}$/),
      z.string()
    ).describe("Final answer from each agent (keyed by agent_id)"),
    consensus_reached: z.union([z.boolean(), z.null()]).describe("true/false for QA if success, null for open-ended or non-success"),
    performance_metrics: z.object({
      total_tokens: z.number().describe("Total tokens generated in this conversation"),
      total_prompt_tokens: z.number().describe("Total prompt tokens processed"),
      total_time_seconds: z.number().describe("Total time for this conversation"),
      average_response_time_ms: z.number().describe("Average time per message in milliseconds")
    }).strict().describe("Performance metrics for the conversation"),
    retry_statistics: z.object({
      total_retry_attempts: z.number().int().describe("Total retry attempts across all messages"),
      messages_requiring_retries: z.number().int().describe("Number of messages that needed retries"),
      validation_errors_summary: z.array(z.object({
        error: z.string().describe("The validation error message"),
        count: z.number().int().describe("How many times this error occurred")
      })).describe("Summary of validation errors encountered")
    }).strict().describe("Retry statistics for structured output validation")
  }).strict().describe("Summary statistics for quick analysis"),
  created_at: z.string().datetime().describe("When the transcript was created (ISO 8601 with Z)")
}).strict(); // additionalProperties: false

// ========================================
// Metadata Schema (for index entries)
// ========================================

export const MetadataSchema = z.object({
  transcript_id: z.string().uuid().describe("UUID of the transcript file"),
  experiment_name: z.string().describe("Name of the experiment configuration"),
  benchmark_subcategory: z.string().describe("Benchmark subcategory being run (e.g., 'bbq_race', 'discrim_eval_age')"),
  question_id: z.string().describe("ID of the question being answered (e.g., 'bbq_race_400')"),
  job_task_id: z.string().describe("Unified job identifier: 'local', Slurm job ID, or array task ID (e.g., '10001_2')"),
  submission_timestamp: z.string().datetime().describe("When the job was submitted (ISO 8601 with Z)"),
  execution_timestamp: z.string().datetime().describe("When the conversation was executed (ISO 8601 with Z)"),
  transcript_path: z.string().describe("Path to the full transcript file"),
  config_snapshot_path: z.string().describe("Path to the configuration snapshot"),
  protocol_version: z.string().regex(/^\d{4}-\d{2}-\d{2}$/).describe("Protocol version (e.g., '2025-11-27')"),
  routing_strategy: z.string().describe("Routing strategy used (e.g., 'vanilla')"),
  identity_reveal_config: IdentityRevealSettingsSchema.describe("Settings controlling identity visibility"),
  n_agents: z.number().int().min(1).describe("Number of agents in conversation"),
  shared_model_backbone: z.string().optional().describe("Shared model backbone if used (e.g., 'llama31_8b')"),
  agents: z.array(
    z.object({
      agent_id: z.string().regex(/^spkr_\d{3}$/),
      role: z.string(),
      persona: z.union([z.string(), z.null()]),
      demographics: z.union([z.string(), z.null()]),
      if_as_human: z.boolean(),
      model: z.string(),
      temperature: z.number(),
      max_tokens: z.number().int()
    }).strict()
  ).describe("Full agent configurations for experimental condition filtering"),
  status: z.enum(["succeeded", "partial", "failed"]).describe("Conversation completion status"),
  consensus_reached: z.union([z.boolean(), z.null()]).describe("Whether consensus was reached"),
  total_rounds_completed: z.number().int().describe("Number of rounds completed"),
  retry_attempts: z.number().int().describe("Total retry attempts for structured output validation")
}).strict(); // additionalProperties: false

// ========================================
// Type Exports
// ========================================

export type Question = z.infer<typeof QuestionSchema>;
export type Agent = z.infer<typeof AgentSchema>;
export type StructuredOutput = z.infer<typeof StructuredOutputSchema>;
export type Message = z.infer<typeof MessageSchema>;
export type IdentityRevealSettings = z.infer<typeof IdentityRevealSettingsSchema>;
export type Routing = z.infer<typeof RoutingSchema>;
export type RetryConfig = z.infer<typeof RetryConfigSchema>;
export type Conversation = z.infer<typeof ConversationSchema>;
export type Metadata = z.infer<typeof MetadataSchema>;

// ========================================
// Schema Version
// ========================================

export const SCHEMA_VERSION = '2025-11-27';

// Export all schemas as a collection
export const schemas = {
  question: QuestionSchema,
  agent: AgentSchema,
  structuredOutput: StructuredOutputSchema,
  message: MessageSchema,
  identityRevealSettings: IdentityRevealSettingsSchema,
  routing: RoutingSchema,
  retryConfig: RetryConfigSchema,
  conversation: ConversationSchema,
  metadata: MetadataSchema,
} as const;