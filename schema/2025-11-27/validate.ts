/**
 * Validation Script
 *
 * Validates data against Zod schemas
 * Can be called from Python via subprocess or used directly
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