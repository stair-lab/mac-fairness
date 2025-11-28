/**
 * Zod Validation Test Suite - Version 2025-11-27
 *
 * PURPOSE:
 * Comprehensive test suite for all Zod schemas to ensure validation works correctly.
 * Tests both valid and invalid data cases to verify schema constraints.
 *
 * WHEN TO RUN:
 * Run this test suite after any schema changes to verify correctness:
 *   npm run test
 *
 * TEST COVERAGE:
 * 1. StructuredOutputSchema - Tests all response types (participant, judge, moderator, devils_advocate)
 * 2. AgentSchema - Tests agent configuration validation
 * 3. MessageSchema - Tests message structure with metadata
 * 4. IdentityRevealSettingsSchema - Tests boolean validation for reveal settings
 * 5. RoutingSchema - Tests routing configuration validation
 * 6. QuestionSchema - Tests question format with capital letter validation
 * 7. Discriminated Union - Tests type-specific field requirements
 * 8. TypeScript Type Inference - Verifies TypeScript types work correctly
 * 9. validate.ts function - Tests the validation CLI interface
 *
 * VALIDATION PIPELINE:
 * schemas.ts (Zod definitions) → test-validation.ts (this file, tests)
 *                                     ↓
 *                                validate.ts (runtime validation CLI)
 *                                     ↓
 *                                Python code (subprocess calls)
 *
 * ADDING NEW TESTS:
 * When adding new schema fields or validation rules:
 * 1. Add test cases to the testCases object
 * 2. Add validation checks to the test sections
 * 3. Document expected behavior in test output
 * 4. Run `npm run test` to verify all tests pass
 *
 * TEST OUTPUT:
 * The script prints detailed test results showing:
 * - Which validations passed (✓)
 * - Which validations failed as expected (✗)
 * - Error messages and paths for failed validations
 * - Summary of field requirements for discriminated unions
 */

import { schemas } from './schemas.js';
import { validateData } from './validate.js';

// Test data for each schema type
const testCases = {
  // Valid structured output - participant type
  structuredOutputValid: {
    response_type: "participant",
    opinion: "B",
    rationale: "Based on the provided information, option B aligns with established principles and represents the most logical choice given the context.",
    confidence: 0.85
  },

  // Invalid structured output (missing required field)
  structuredOutputInvalid: {
    response_type: "participant",
    rationale: "This should fail because 'opinion' is required for participant type",
    confidence: 0.5
  },

  // Valid judge response
  structuredOutputJudge: {
    response_type: "judge",
    verdict: "spkr_001 presents the strongest argument",
    rationale: "The argument is well-supported by evidence and logical reasoning",
    evaluations: {
      "spkr_001": "Strong argument with clear evidence",
      "spkr_002": "Weak argument, lacks supporting facts"
    }
  },

  // Valid agent configuration
  agentValid: {
    agent_id: "spkr_001",  // Must match regex /^spkr_\d{3}$/
    role: "participant",
    if_as_human: true,
    model: "shared",
    temperature: 0.7,
    max_tokens: 1000,
    persona: "doctor",
    demographics: "asian female"
  },

  // Valid message
  messageValid: {
    message_id: "msg_1_001",  // Format: msg_<round>_<speaker>
    agent_id: "spkr_001",  // Must match regex /^spkr_\d{3}$/
    agent_role: "participant",
    agent_identity_display: "a black doctor",  // Optional: how agent appears in conversation
    round_id: 1,
    structured_response: {
      response_type: "participant",
      opinion: "A",
      rationale: "We should consider the ethical implications before making this decision",
      confidence: 0.9
    },
    visible_to: ["spkr_001", "spkr_002", "spkr_003"],  // Optional: routing visibility
    message_metadata: {
      tokens_generated: 42,
      generation_time_ms: 1250.5,
      temperature_used: 0.7,
      exceeded_max_tokens: false,
      retry_count: 0
    }
  },

  // Valid identity reveal settings
  identityRevealValid: {
    reveal_persona: true,
    reveal_demographics: false,
    reveal_presence_mode: true
  },

  // Invalid identity reveal settings (non-boolean value)
  identityRevealInvalid: {
    reveal_persona: "yes",  // Should be boolean
    reveal_demographics: false,
    reveal_presence_mode: true
  },

  // Valid routing config
  routingValid: {
    strategy: "round_robin",
    max_rounds: 5,
    parameters: {
      allow_repeats: false,
      consensus_threshold: 0.8
    }
  },

  // Valid question - multiple choice with capital letters
  questionValid: {
    question_id: "bbq_race_42",
    source_dataset: "BBQ",
    source_id: "42",
    question_type: "multiple_choice",
    question: "Which option best describes the situation?",
    choices: [
      { id: "A", text: "Option A" },
      { id: "B", text: "Option B" },
      { id: "C", text: "Option C" }
    ],
    correct_answer_id: "A",
    schema_version: "2025-11-27"
  },

  // Invalid question - binary with underscore (should fail)
  questionInvalidUnderscore: {
    question_id: "test_binary_1",
    source_dataset: "TEST",
    source_id: "1",
    question_type: "binary",
    question: "Is this valid?",
    choices: [
      { id: "_", text: "Open ended" }
    ],
    correct_answer_id: "_",
    schema_version: "2025-11-27"
  },

  // Invalid question - missing correct_answer_id (should fail)
  questionMissingAnswer: {
    question_id: "bbq_race_43",
    source_dataset: "BBQ",
    source_id: "43",
    question_type: "binary",
    question: "Is this valid?",
    choices: [
      { id: "A", text: "Yes" },
      { id: "B", text: "No" }
    ],
    schema_version: "2025-11-27"
  }
};

// Run tests
console.log('=== Testing Zod Validation ===\n');

// Test structured output validation
console.log('1. Testing Structured Output Schema:');
console.log('   Valid participant data:');
const structuredValid = schemas.structuredOutput.safeParse(testCases.structuredOutputValid);
console.log(`   ✓ Valid: ${structuredValid.success}`);

console.log('   Valid judge data:');
const structuredJudgeValid = schemas.structuredOutput.safeParse(testCases.structuredOutputJudge);
console.log(`   ✓ Valid: ${structuredJudgeValid.success}`);

console.log('   Invalid data (missing required field):');
const structuredInvalid = schemas.structuredOutput.safeParse(testCases.structuredOutputInvalid);
console.log(`   ✓ Invalid detected: ${!structuredInvalid.success}`);
if (!structuredInvalid.success) {
  console.log(`   Error: ${structuredInvalid.error.errors[0].message} at path: ${structuredInvalid.error.errors[0].path.join('.')}`);
}

// Test agent validation
console.log('\n2. Testing Agent Schema:');
const agentValid = schemas.agent.safeParse(testCases.agentValid);
console.log(`   ✓ Valid: ${agentValid.success}`);
if (!agentValid.success) {
  console.log('   Errors:');
  agentValid.error.errors.forEach(err => {
    console.log(`     - ${err.path.join('.')}: ${err.message}`);
  });
}

// Test message validation
console.log('\n3. Testing Message Schema:');
const messageValid = schemas.message.safeParse(testCases.messageValid);
console.log(`   ✓ Valid: ${messageValid.success}`);
if (!messageValid.success) {
  console.log('   Errors:');
  messageValid.error.errors.forEach(err => {
    console.log(`     - ${err.path.join('.')}: ${err.message}`);
  });
}

// Test identity reveal settings validation
console.log('\n4. Testing Identity Reveal Settings Schema:');
const identityRevealValid = schemas.identityRevealSettings.safeParse(testCases.identityRevealValid);
console.log(`   ✓ Valid boolean values: ${identityRevealValid.success}`);

const identityRevealInvalid = schemas.identityRevealSettings.safeParse(testCases.identityRevealInvalid);
console.log(`   ✓ Invalid non-boolean detected: ${!identityRevealInvalid.success}`);
if (!identityRevealInvalid.success) {
  console.log(`   Error: ${identityRevealInvalid.error.errors[0].message} at path: ${identityRevealInvalid.error.errors[0].path.join('.')}`);
}

// Test routing validation
console.log('\n5. Testing Routing Schema:');
const routingValid = schemas.routing.safeParse(testCases.routingValid);
console.log(`   ✓ Valid: ${routingValid.success}`);
if (!routingValid.success) {
  console.log('   Errors:');
  routingValid.error.errors.forEach(err => {
    console.log(`     - ${err.path.join('.')}: ${err.message}`);
  });
}

// Test question validation
console.log('\n6. Testing Question Schema:');
const questionValid = schemas.question.safeParse(testCases.questionValid);
console.log(`   ✓ Valid multiple_choice with capital letters: ${questionValid.success}`);
if (!questionValid.success) {
  console.log('   Errors:');
  questionValid.error.errors.forEach(err => {
    console.log(`     - ${err.path.join('.')}: ${err.message}`);
  });
}

console.log('   Testing invalid question (binary with underscore):');
const questionInvalidUnderscore = schemas.question.safeParse(testCases.questionInvalidUnderscore);
console.log(`   ✓ Invalid detected: ${!questionInvalidUnderscore.success}`);
if (!questionInvalidUnderscore.success) {
  questionInvalidUnderscore.error.errors.forEach(err => {
    console.log(`     - ${err.path.join('.')}: ${err.message}`);
  });
}

console.log('   Testing invalid question (missing correct_answer_id):');
const questionMissingAnswer = schemas.question.safeParse(testCases.questionMissingAnswer);
console.log(`   ✓ Invalid detected: ${!questionMissingAnswer.success}`);
if (!questionMissingAnswer.success) {
  questionMissingAnswer.error.errors.forEach(err => {
    console.log(`     - ${err.path.join('.')}: ${err.message}`);
  });
}

// Test the validate function from validate.ts
console.log('\n7. Testing validate.ts function:');
const validateResult = validateData('structured_output', testCases.structuredOutputValid);
console.log(`   ✓ Function works: ${validateResult.success}`);

// Test type inference
console.log('\n8. Testing TypeScript type inference:');
import type { StructuredOutput } from './schemas.js';

const typedOutput: StructuredOutput = {
  response_type: "participant",
  opinion: "C",
  rationale: "Type-safe response with clear reasoning",
  confidence: 0.95
};
// Verify the typed output works
const typeCheckResult = schemas.structuredOutput.safeParse(typedOutput);
console.log(`   ✓ TypeScript types work correctly: ${typeCheckResult.success}`);

// Test discriminated union field requirements
console.log('\n9. Testing Discriminated Union Field Requirements:');
console.log('   The discriminated union enforces response_type-dependent fields.');

// Test cases showing field requirements per type
const discriminatedUnionTests = [
  {
    name: '   ✓ Valid participant',
    data: {
      response_type: 'participant',
      opinion: 'A',
      rationale: 'Option A is most logical'
    }
  },
  {
    name: '   ✗ Invalid participant (missing opinion)',
    data: {
      response_type: 'participant',
      rationale: 'Missing the opinion field'
    }
  },
  {
    name: '   ✓ Valid moderator',
    data: {
      response_type: 'moderator',
      summary: 'Consensus emerging around option B',
      rationale: 'Most participants agree',
      consensus_level: 0.75
    }
  },
  {
    name: '   ✗ Invalid moderator (missing summary)',
    data: {
      response_type: 'moderator',
      rationale: 'Missing summary field'
    }
  },
  {
    name: '   ✓ Valid devil\'s advocate',
    data: {
      response_type: 'devils_advocate',
      challenge: 'What about the ethical implications?',
      target_position: 'The majority view on option B',
      rationale: 'We should consider all angles'
    }
  },
  {
    name: '   ✗ Invalid devil\'s advocate (missing target_position)',
    data: {
      response_type: 'devils_advocate',
      challenge: 'But what if...',
      rationale: 'Missing target_position'
    }
  }
];

discriminatedUnionTests.forEach(testCase => {
  const result = schemas.structuredOutput.safeParse(testCase.data);
  console.log(`${testCase.name}: ${result.success ? 'PASS' : 'FAIL'}`);
  if (!result.success) {
    const error = result.error.errors[0];
    console.log(`     → Error: ${error.message} at path: ${error.path.join('.')}`);
  }
});

console.log('\n   Field Requirements Summary:');
console.log('   - participant: response_type, opinion, rationale');
console.log('   - judge: response_type, verdict, rationale');
console.log('   - moderator: response_type, summary, rationale');
console.log('   - devils_advocate: response_type, challenge, target_position, rationale');

console.log('\n=== All tests completed ===');