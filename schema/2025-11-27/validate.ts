/**
 * Runtime Validation CLI - Version 2025-11-27
 *
 * PURPOSE:
 * Command-line interface for validating JSON data against Zod schemas.
 * This is the primary interface between Python code and the Zod validation system.
 *
 * USAGE FROM PYTHON:
 * Python code calls this script via subprocess to validate data structures:
 *
 *   import subprocess
 *   import json
 *
 *   # Validate structured output from an agent
 *   data = {"response_type": "participant", "opinion": "B", "rationale": "..."}
 *   result = subprocess.run(
 *       ["node", "dist/validate.js", "structured_output"],
 *       input=json.dumps(data),
 *       capture_output=True,
 *       text=True
 *   )
 *   validation_result = json.loads(result.stdout)
 *
 * VALIDATION PIPELINE:
 * schemas.ts (Zod definitions) → validate.ts (this file, CLI) → Python (subprocess caller)
 *
 * CLI INTERFACE:
 * Input:  JSON data via stdin
 * Args:   schema_type (question|agent|structured_output|message|routing|conversation|metadata)
 * Output: JSON result via stdout with structure:
 *   Success: {"success": true, "data": <validated_data>, "schemaType": "..."}
 *   Failure: {"success": false, "errors": [...], "schemaType": "..."}
 *
 * AVAILABLE SCHEMA TYPES:
 * - question: Validate question format from JSONL files
 * - agent: Validate agent configuration
 * - structured_output: Validate agent response (most common)
 * - message: Validate individual message in conversation
 * - routing: Validate routing configuration
 * - conversation: Validate complete transcript
 * - metadata: Validate index entry
 *
 * ERROR HANDLING:
 * The script catches three types of errors:
 * 1. Invalid schema type (unknown schema name)
 * 2. JSON parsing errors (malformed input)
 * 3. Zod validation errors (data doesn't match schema)
 *
 * All errors are returned as JSON with success: false
 *
 * TYPESCRIPT USAGE:
 * This file can also be imported in TypeScript:
 *   import { validateData } from './validate.js';
 *   const result = validateData('structured_output', data);
 */

import { schemas } from './schemas.js';
import * as readline from 'readline';

// Map of schema names to validators (order matches generate-json-schemas.ts)
const validators = {
  'question': schemas.question,
  'agent': schemas.agent,
  'structured_output': schemas.structuredOutput,
  'message': schemas.message,
  'routing': schemas.routing,
  'conversation': schemas.conversation,
  'metadata': schemas.metadata
} as const;

type SchemaType = keyof typeof validators;

// Only run CLI mode if this is the main module
if (import.meta.url === `file://${process.argv[1]}`) {
  // Handle command line arguments
  const args = process.argv.slice(2);
  const schemaType = (args[0] || 'structured_output') as SchemaType;

  if (!(schemaType in validators)) {
    console.error(JSON.stringify({
      success: false,
      error: `Unknown schema type: ${schemaType}. Valid types: ${Object.keys(validators).join(', ')}`
    }));
    process.exit(1);
  }

  // Create interface for reading from stdin
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
  });

  let inputBuffer = '';

  rl.on('line', (line) => {
    inputBuffer += line;
  });

  rl.on('close', () => {
    try {
      // Parse input JSON
      const data = JSON.parse(inputBuffer);

      // Get the appropriate validator
      const validator = validators[schemaType];

      // Validate using Zod
      const result = validator.safeParse(data);

      if (result.success) {
        // Output success with validated data
        console.log(JSON.stringify({
          success: true,
          data: result.data,
          schemaType: schemaType
        }));
      } else {
        // Output validation errors
        console.log(JSON.stringify({
          success: false,
          errors: result.error.errors.map((err: any) => ({
            path: err.path.join('.'),
            message: err.message,
            code: err.code
          })),
          schemaType: schemaType
        }));
      }
    } catch (error) {
      // Handle JSON parsing errors
      console.log(JSON.stringify({
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        raw: inputBuffer,
        schemaType: schemaType
      }));
    }
  });
}

// Export for use in other TypeScript files
export function validateData(schemaType: SchemaType, data: unknown) {
  const validator = validators[schemaType];
  return validator.safeParse(data);
}