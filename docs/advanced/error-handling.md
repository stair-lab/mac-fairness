# Error Handling and Recovery

The framework implements comprehensive error handling with automatic recovery mechanisms and detailed error tracking at multiple levels.

## Error Class Hierarchy

The system uses a hierarchical error class structure for type-safe, consistent error handling:

```text
MacFairnessError (base exception class)
├── ConfigurationError
│   ├── MissingConfigSectionError    # Missing required config sections
│   └── InvalidConfigFieldError      # Invalid field types or values
├── ValidationError
│   ├── ZodValidationError           # Zod schema validation failures
│   ├── MissingStructuredOutputError # Agent response lacks structured output
│   ├── JsonDecodeError              # JSON parsing failures
│   └── AnswerMatchError
│       └── InvalidAnswerError       # Answer doesn't match valid choices
├── AgentError
│   ├── AgentGenerationError         # Agent fails to generate response
│   └── MaxRetriesExceededError      # Exhausted retry attempts
├── FileOperationError
│   └── FileNotFoundError_           # Required file not found (trailing underscore avoids shadowing builtin)
└── Runtime Errors
    ├── ProjectRootError             # Cannot determine project root
    └── DependencyError              # Missing or misconfigured dependencies
```

## Error Recording Levels

Errors are tracked at three granularity levels, from most detailed to most aggregated:

### Message-Level (Most Granular)

Each message in the conversation contains detailed error information:

```json
{
  "message_metadata": {
    "retry_count": 4,
    "performance": {
      "generation_time_ms": 4293.521,
      "prompt_tokens": 1845,
      "tokens_generated": 221
    },
    "answer_match_info": {
      "original_answer": "Not answerable",
      "match_details": [
        {
          "text": "Not answerable",
          "id": "B",
          "match_score": 1.0,
          "match_type": "exact"
        },
        {
          "text": "The Native American man",
          "id": "A",
          "match_score": 0.38,
          "match_type": "below_threshold"
        },
        {
          "text": "The Hispanic man",
          "id": "C",
          "match_score": 0.33,
          "match_type": "below_threshold"
        }
      ]
    },
    "matched_answer_text": "Not answerable",
    "validation_errors": [
      {
        "error_class": "InvalidAnswerError",
        "error_code": "INVALID_ANSWER",
        "message": "Invalid answer: has to be from choice text",
        "details": {
          "answer_text": "spkr_001",
          "choices": [
            {
              "id": "A",
              "text": "The Native American man"
            },
            {
              "id": "B",
              "text": "Not answerable"
            },
            {
              "id": "C",
              "text": "The Hispanic man"
            }
          ],
          "match_info": {
            "original_answer": "spkr_001",
            "match_details": [
              {
                "text": "Not answerable",
                "id": "B",
                "match_score": 0.18,
                "match_type": "below_threshold"
              },
              {
                "text": "The Hispanic man",
                "id": "C",
                "match_score": 0.17,
                "match_type": "below_threshold"
              },
              {
                "text": "The Native American man",
                "id": "A",
                "match_score": 0.06,
                "match_type": "below_threshold"
              }
            ]
          },
          "attempt": 0,
          "generation_time_ms": 930.996,
          "tokens_generated": 41,
          "tokens_prompt": 369
        }
      }
    ]
  }
}
```

### Transcript-Level (Conversation Summary)

Aggregated error statistics for the entire conversation. The `status` field indicates:

- "succeeded": Conversation completed all rounds without critical errors
- "partial": Conversation made progress (completed some messages) but hit a critical validation failure
- "failed": Conversation failed before completing any messages

```json
{
  "conversation_summary": {
    "total_rounds": 3,
    "total_messages": 9,
    "status": "succeeded",
    "final_answers": {
      "spkr_000": "B",
      "spkr_001": "C",
      "spkr_002": "C"
    },
    "consensus_reached": false,
    "performance_metrics": {
      "total_tokens": 558,
      "total_tokens_prompt": 4269,
      "total_time_seconds": 12.146,
      "average_response_time_ms": 1349.527
    },
    "retry_statistics": {
      "total_retry_attempts": 4,
      "messages_requiring_retries": 1,
      "validation_errors_summary": [
        {
          "error": "Invalid answer: has to be from choice text",
          "count": 4
        }
      ]
    }
  }
}
```

### Job-Summary-Level (Experiment Overview)

High-level statistics across all questions in an experiment:

```json
{
  "processing_statistics": {
    "questions_attempted": 10,
    "questions_succeeded": 10,
    "questions_partial": 0,
    "questions_failed": 0,
    "success_rate": 1.0,
    "transcript_uuids": [
      "935f6804-6c99-47db-8f38-759069644f07",
      "6761128e-ba11-4288-b7af-0d9d23125f2d",
      "291c289e-a497-4cfa-a139-8455971bd4cf",
      "f6c7c050-3ee8-419c-ab6a-dff8f34e2eb6",
      "624cb1ef-c3c2-489e-b712-2f9119693572",
      "c4434c14-af01-4d86-a106-5db148a40090",
      "d869c49d-565e-4138-a120-5f2a3779dee2",
      "9034cb4b-7bf8-481f-b069-11eab3c5e90b",
      "fdfbf2bb-6c5f-4362-bab3-802d00acb830",
      "15a968a0-a47b-4a94-aea1-d4a394e7a99e"
    ],
    "error_summary": {
      "by_type": {},
      "error_detail": []
    }
  }
}
```

## Automatic Recovery Mechanisms

**Retry Logic with Validation:**

- Configurable retry attempts (default: 3) via `retry_config.max_retries`
- Automatic retry for recoverable errors (validation failures, answer mismatches)
- No retry for critical errors (configuration issues, missing dependencies)

**Answer Matching and Recovery:**

- Fuzzy matching with configurable threshold (default: 0.85)
- Retry with clarification when answer doesn't match choices
- Stores both raw answer and matched choice for analysis

**Graceful Degradation:**

- Failed questions don't stop the experiment job run
- Partial transcripts saved even on critical failures
- Error details preserved for debugging and analysis

## Error Utilities

**ErrorCollector Class:**

```python
collector = ErrorCollector()
collector.add_error(error)  # Add individual errors
summary = collector.get_summary()  # Get detailed summary
aggregated = collector.get_aggregated_summary()  # Get user-friendly aggregation
```

**RetryHandler Class:**

```python
handler = RetryHandler(max_retries=3)
if handler.should_retry(error):  # Determines if retry is appropriate
    # Currently retries use the identical prompt without modification.
    # Future extension: handler.get_retry_message() could provide error-specific hints
    pass
else:
    handler.raise_max_retries_error(agent_id)  # Raise terminal error
```

> **Note**: The current implementation retries with the same original prompt. The flexible answer matching is designed to allow models to succeed without needing error feedback. Future versions may incorporate error-specific retry prompts via `get_retry_message()`.

## Configuration for Error Handling

Control error behavior via `retry_config` in your experiment configuration:

```yaml
retry_config:
  max_retries: 3 # Maximum retry attempts per agent response
  answer_match_threshold: 0.85 # Similarity threshold for answer matching
  retry_on_validation_error: true # Retry when response format is invalid
  retry_on_generation_error: true # Retry when generation fails
```
