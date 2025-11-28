/**
 * JSON Schema Generator - Version 2025-11-27
 *
 * PURPOSE:
 * Generates JSON Schema (.schema.json) files from Zod schema definitions.
 * These JSON schemas are OPTIONAL and primarily for documentation purposes.
 * The framework uses Zod schemas directly for runtime validation via validate.ts.
 *
 * WHEN TO RUN:
 * Run this script after updating schemas.ts to regenerate JSON Schema documentation:
 *   npm run generate
 *
 * OUTPUT FILES:
 * All generated files are written to schema/2025-11-27/:
 * - question.schema.json - Question format schema
 * - agent.schema.json - Agent configuration schema
 * - structured_output.schema.json - Discriminated union for agent responses
 * - message.schema.json - Message structure schema
 * - routing.schema.json - Routing configuration schema
 * - conversation.schema.json - Full transcript schema
 * - metadata.schema.json - Index entry schema
 *
 * VALIDATION PIPELINE:
 * schemas.ts (Zod) → [generate-json-schemas.ts] → *.schema.json (optional docs)
 *                              ↓
 *                     validate.ts (runtime validation used by Python)
 *
 * TECHNICAL DETAILS:
 * - Uses zod-to-json-schema library for conversion
 * - Preserves exact field descriptions and validation rules
 * - Handles discriminated unions correctly (for StructuredOutputSchema)
 * - Sets additionalProperties: false for strict validation
 * - Inlines all definitions ($refStrategy: 'none') for standalone schemas
 *
 * NOTE:
 * JSON schemas are auto-generated and should NOT be manually edited.
 * Always modify schemas.ts and regenerate using this script.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { zodToJsonSchema } from 'zod-to-json-schema';
import {
  schemas,
  SCHEMA_VERSION
} from './schemas.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration for each schema with exact metadata
const schemaConfigs = [
  {
    zodSchema: schemas.question,
    fileName: 'question.schema.json',
    title: 'Question Format',
    description: 'Unified question format across all fairness benchmarks',
    required: ['question_id', 'source_dataset', 'source_id', 'question_type', 'question', 'choices', 'correct_answer_id', 'schema_version']
  },
  {
    zodSchema: schemas.agent,
    fileName: 'agent.schema.json',
    title: 'Agent Configuration',
    description: 'Configuration for a single agent in the multi-agent conversation',
    required: ['agent_id', 'role', 'if_as_human', 'model', 'temperature', 'max_tokens']
  },
  {
    zodSchema: schemas.structuredOutput,
    fileName: 'structured_output.schema.json',
    title: 'Structured Agent Output',
    description: 'Discriminated union for different agent response types (participant, judge, moderator, devils_advocate)',
    required: ['response_type', 'rationale'] // Note: actual required fields vary by type
  },
  {
    zodSchema: schemas.message,
    fileName: 'message.schema.json',
    title: 'Conversation Message',
    description: 'Structure for a single message in a multi-agent conversation',
    required: ['message_id', 'agent_id', 'agent_role', 'round_id', 'structured_response', 'visible_to', 'message_metadata']
  },
  {
    zodSchema: schemas.routing,
    fileName: 'routing.schema.json',
    title: 'Routing Configuration',
    description: 'Configuration for conversation routing strategy',
    required: ['strategy', 'max_rounds']
  },
  {
    zodSchema: schemas.conversation,
    fileName: 'conversation.schema.json',
    title: 'Conversation Transcript',
    description: 'Complete multi-agent conversation transcript',
    required: ['transcript_id', 'protocol_version', 'experiment_metadata', 'question', 'routing_config', 'agents', 'conversation_rounds', 'conversation_summary', 'created_at']
  },
  {
    zodSchema: schemas.metadata,
    fileName: 'metadata.schema.json',
    title: 'Transcript Metadata',
    description: 'Metadata for a conversation transcript (used in index)',
    required: ['transcript_id', 'experiment_name', 'benchmark_subcategory', 'question_id', 'job_task_id', 'submission_timestamp', 'execution_timestamp', 'transcript_path', 'config_snapshot_path', 'protocol_version', 'routing_strategy', 'identity_reveal_config', 'n_agents', 'agents', 'status', 'consensus_reached', 'total_rounds_completed', 'retry_attempts']
  },
];

// Generate each JSON schema
schemaConfigs.forEach(config => {
  console.log(`Generating ${config.fileName}...`);

  // Generate base JSON schema from Zod
  const jsonSchema = zodToJsonSchema(config.zodSchema, {
    $refStrategy: 'none', // Don't use refs, inline everything
    errorMessages: false,
    markdownDescription: false,
    target: 'jsonSchema7',
    strictUnions: true
  }) as any;

  // Build the complete schema with exact structure
  const finalSchema: any = {
    '$schema': 'http://json-schema.org/draft-07/schema#',
    '$id': config.fileName,
    'title': config.title,
    'description': config.description,
    ...jsonSchema, // Include all generated properties (handles discriminated unions)
  };

  // Only override type and required if not a discriminated union
  if (!jsonSchema.discriminator && !jsonSchema.anyOf && !jsonSchema.oneOf) {
    finalSchema.type = 'object';
    finalSchema.required = config.required;
    if (jsonSchema.additionalProperties === undefined) {
      finalSchema.additionalProperties = false;
    }
  }

  // Add examples for specific schemas
  if (config.fileName === 'agent.schema.json' && finalSchema.properties) {
    if (finalSchema.properties.role) {
      finalSchema.properties.role.examples = ['participant', 'devils_advocate', 'judge', 'mediator'];
    }
    if (finalSchema.properties.persona) {
      finalSchema.properties.persona.examples = ['doctor', 'economist', 'policy_expert', 'teacher', null];
    }
    if (finalSchema.properties.demographics) {
      finalSchema.properties.demographics.examples = ['black', 'white', 'white female', 'asian male', null];
    }
  }

  // Handle special cases and references
  if (config.fileName === 'message.schema.json' && finalSchema.properties.structured_response) {
    // Replace inline schema with $ref
    finalSchema.properties.structured_response = {
      '$ref': 'structured_output.schema.json',
      'description': 'Parsed structured output if agent used structured response format'
    };
  }

  // Write the schema file
  const outputPath = path.join(__dirname, config.fileName);
  fs.writeFileSync(outputPath, JSON.stringify(finalSchema, null, 2));
  console.log(`✓ Generated ${config.fileName}`);
});

console.log(`\nAll schemas generated for version ${SCHEMA_VERSION}`);
console.log(`Output directory: ${__dirname}`);