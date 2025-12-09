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
│   ├── MissingStructuredOutputError # Agent response lacks structured output
│   ├── MaxLengthExceededError       # Response exceeds maximum token limit
│   └── AnswerMatchError
│       └── InvalidAnswerError       # Answer doesn't match valid choices
├── AgentError
│   └── MaxRetriesExceededError      # Exhausted retry attempts
├── FileOperationError
│   └── FileNotFoundError_           # Required file not found (trailing underscore avoids shadowing builtin)
├── VLLMEngineError                  # Base class for vLLM backend errors
│   ├── VLLMEngineNotStartedError    # Engine not started before use
│   ├── VLLMEngineNotInitializedError # Engine not initialized
│   ├── VLLMBatchError               # Batch processing failure
│   ├── VLLMInferenceError           # Single request inference failure
│   └── VLLMOOMError                 # GPU out of memory
├── OllamaError                      # Base class for Ollama backend errors
│   ├── OllamaAPIError               # Ollama API request failure
│   ├── OllamaConnectionError        # Connection to Ollama failed
│   └── OllamaNotAvailableError      # Ollama not installed or running
└── Runtime Errors
    ├── ProjectRootError             # Cannot determine project root
    ├── DependencyError              # Missing or misconfigured dependencies
    └── UnexpectedError              # Wrapper for unexpected exceptions
```

## Error Recording Levels

Errors are tracked at three granularity levels, from most detailed to most aggregated:

### Message-Level (Most Granular)

Each message in the conversation contains detailed error information:

```json
{
  "message_metadata": {
    "retry_count": 1,
    "prompt_tokens": 351,
    "response_tokens": 43,
    "matched_answer_text": "The Native American man",
    "answer_match_info": {
      "original_answer": "The Native American man",
      "match_details": [
        {
          "text": "The Native American man",
          "id": "A",
          "match_score": 1.0,
          "match_type": "exact"
        },
        {
          "text": "The Hispanic man",
          "id": "C",
          "match_score": 0.56,
          "match_type": "below_threshold"
        },
        {
          "text": "Unknown",
          "id": "B",
          "match_score": 0.2,
          "match_type": "below_threshold"
        }
      ]
    },
    "validation_errors": [
      {
        "error_class": "InvalidAnswerError",
        "error_code": "INVALID_ANSWER",
        "message": "Invalid answer: has to be from choice text",
        "details": {
          "answer_text": "Both",
          "choices": [
            {
              "id": "A",
              "text": "The Native American man"
            },
            {
              "id": "B",
              "text": "Unknown"
            },
            {
              "id": "C",
              "text": "The Hispanic man"
            }
          ],
          "match_info": {
            "original_answer": "Both",
            "match_details": [
              {
                "text": "The Hispanic man",
                "id": "C",
                "match_score": 0.2,
                "match_type": "below_threshold"
              },
              {
                "text": "Unknown",
                "id": "B",
                "match_score": 0.18,
                "match_type": "below_threshold"
              },
              {
                "text": "The Native American man",
                "id": "A",
                "match_score": 0.15,
                "match_type": "below_threshold"
              }
            ]
          },
          "attempt": 0
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
      "spkr_001": "A",
      "spkr_000": "A",
      "spkr_002": "A"
    },
    "consensus_reached": true,
    "token_metrics": {
      "total_messages": 9,
      "prompt_tokens": {
        "total": 2757,
        "avg": 306.3,
        "max": 373
      },
      "response_tokens": {
        "total": 395,
        "avg": 43.9,
        "max": 60
      },
      "combined_tokens": {
        "total": 3152,
        "avg": 350.2,
        "max": 415
      }
    },
    "retry_statistics": {
      "total_retry_attempts": 2,
      "messages_requiring_retries": 2,
      "validation_errors_summary": [
        {
          "error": "Invalid answer: has to be from choice text",
          "count": 2
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
    "questions_attempted": 1024,
    "questions_succeeded": 1024,
    "questions_partial": 0,
    "questions_failed": 0,
    "success_rate": 1.0,
    "transcript_uuids": ["...", "..."],
    "error_summary": {
      "by_type": {},
      "error_detail": []
    }
  }
}
```

## Automatic Recovery Mechanisms

**Retry Logic with Validation:**

- Configurable retry attempts (default: 5) via `retry_config.max_retries`
- Automatic retry for recoverable errors (validation failures, answer mismatches)
- No retry for critical errors (configuration issues, missing dependencies)

**Answer Matching and Recovery:**

- Fuzzy matching with configurable threshold (default: 0.75)
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
collector.add_error(error)  # add individual errors
summary = collector.get_summary()  # get detailed summary
aggregated = collector.get_aggregated_summary()  # get user-friendly aggregation
```

**RetryHandler Class:**

```python
handler = RetryHandler(max_retries=5)
if handler.should_retry(error):  # determines if retry is appropriate
    # Currently retries use the identical prompt without modification.
    # Future extension: handler.get_retry_message() could provide error-specific hints to LLMs
    pass
else:
    handler.raise_max_retries_error(agent_id)  # raise terminal error
```

> **Note**: The current implementation retries with the same original prompt. The flexible answer matching is designed to allow models to succeed without needing error feedback. Future versions may incorporate error-specific retry prompts via `get_retry_message()`.

## Configuration for Error Handling

Control error behavior via `retry_config` in your experiment configuration:

```yaml
retry_config:
  max_retries: 5 # Maximum retry attempts per agent response
  answer_match_threshold: 0.75 # Similarity threshold for answer matching
  retry_on_validation_error: true # Retry when response format is invalid
  retry_on_generation_error: true # Retry when generation fails
```
