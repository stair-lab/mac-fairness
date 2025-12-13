"""Standardized error classes for the MAC-fairness project."""

from typing import Any, Dict, List, Optional


# Import shared error code mapping (import at runtime to avoid circular imports)
def _get_error_message(error_code: str) -> str:
    """Get user-friendly message for an error code."""
    # Lazy import to avoid circular dependency
    from src.utils.logging import ERROR_CODE_MESSAGES

    return ERROR_CODE_MESSAGES.get(error_code, error_code.replace("_", " ").title())


class MacFairnessError(Exception):
    """Base exception class for all MAC-fairness errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the error.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code for categorization
            details: Additional error details/context
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_class": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


# Configuration Errors
class ConfigurationError(MacFairnessError):
    """Base class for configuration-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CONFIG_ERROR",
            details=details,
        )


class MissingConfigSectionError(ConfigurationError):
    """Raised when a required configuration section is missing."""

    def __init__(self, section: str):
        super().__init__(
            message=f"Missing required configuration section: {section}",
            details={"missing_section": section},
        )
        self.error_code = "MISSING_CONFIG_SECTION"


class InvalidConfigFieldError(ConfigurationError):
    """Raised when a configuration field has an invalid value or type."""

    def __init__(self, field: str, expected_type: str, actual_type: str):
        super().__init__(
            message=f"Invalid type for {field}: expected {expected_type}, got {actual_type}",
            details={
                "field": field,
                "expected_type": expected_type,
                "actual_type": actual_type,
            },
        )
        self.error_code = "INVALID_CONFIG_FIELD"


# Validation Errors
class ValidationError(MacFairnessError):
    """Base class for validation-related errors."""

    def __init__(
        self,
        message: str,
        attempt: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if attempt is not None:
            details["attempt"] = attempt
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details,
        )


class MissingStructuredOutputError(ValidationError):
    """Raised when structured output is missing from agent response."""

    def __init__(self, agent_id: str, attempt: Optional[int] = None):
        super().__init__(
            message=f"No structured output found in response from {agent_id}",
            attempt=attempt,
            details={"agent_id": agent_id},
        )
        self.error_code = "MISSING_STRUCTURED_OUTPUT"


class JSONParseError(ValidationError):
    """Raised when JSON parsing fails or requires repair."""

    def __init__(
        self,
        response_text: str,
        repaired: bool = False,
        attempt: Optional[int] = None,
    ):
        status = "repaired" if repaired else "failed"
        super().__init__(
            message=f"JSON parsing {status}",
            attempt=attempt,
            details={
                "response_text": response_text[:500],  # Truncate long responses
                "repaired": repaired,
            },
        )
        self.error_code = "JSON_PARSE_REPAIRED" if repaired else "JSON_PARSE_FAILED"


# Answer Matching Errors
class AnswerMatchError(ValidationError):
    """Base class for answer matching errors."""

    def __init__(
        self,
        message: str,
        answer_text: str,
        choices: List[Dict[str, str]],
        match_info: Optional[Dict[str, Any]] = None,
        attempt: Optional[int] = None,
    ):
        super().__init__(
            message=message,
            attempt=attempt,
            details={
                "answer_text": answer_text,
                "choices": choices,
                "match_info": match_info,
            },
        )


class InvalidAnswerError(AnswerMatchError):
    """Raised when an answer doesn't match any valid choice."""

    def __init__(
        self,
        answer_text: str,
        choices: List[Dict[str, str]],
        match_info: Dict[str, Any],
        attempt: Optional[int] = None,
    ):
        super().__init__(
            message="Invalid answer: has to be from choice text",
            answer_text=answer_text,
            choices=choices,
            match_info=match_info,
            attempt=attempt,
        )
        self.error_code = "INVALID_ANSWER"


class MaxLengthExceededError(ValidationError):
    """Raised when response exceeds maximum token limit."""

    def __init__(
        self,
        agent_id: str,
        max_tokens: int,
        tokens_generated: int,
        truncated: bool = True,
        attempt: Optional[int] = None,
    ):
        super().__init__(
            message=f"Response exceeded max tokens ({tokens_generated}/{max_tokens})",
            attempt=attempt,
            details={
                "agent_id": agent_id,
                "max_tokens": max_tokens,
                "tokens_generated": tokens_generated,
                "truncated": truncated,
            },
        )
        self.error_code = "MAX_LENGTH_EXCEEDED"


# Agent/Model Errors
class AgentError(MacFairnessError):
    """Base class for agent-related errors."""

    def __init__(
        self,
        message: str,
        agent_id: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        details["agent_id"] = agent_id
        super().__init__(
            message=message,
            error_code="AGENT_ERROR",
            details=details,
        )


class MaxRetriesExceededError(AgentError):
    """Raised when maximum retry attempts are exceeded."""

    def __init__(
        self,
        agent_id: str,
        max_retries: int,
        validation_errors: List[Dict[str, Any]],
    ):
        super().__init__(
            message=f"Agent {agent_id} failed after {max_retries} retries",
            agent_id=agent_id,
            details={
                "max_retries": max_retries,
                "validation_errors": validation_errors,
            },
        )
        self.error_code = "MAX_RETRIES_EXCEEDED"


# File/IO Errors
class FileOperationError(MacFairnessError):
    """Base class for file operation errors."""

    def __init__(
        self,
        message: str,
        file_path: str,
        operation: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        details.update({"file_path": file_path, "operation": operation})
        super().__init__(
            message=message,
            error_code="FILE_OPERATION_ERROR",
            details=details,
        )


class FileNotFoundError_(FileOperationError):
    """Raised when a required file is not found.

    Note: Named with trailing underscore to avoid shadowing Python's builtin FileNotFoundError.
    """

    def __init__(self, file_path: str, operation: str = "read"):
        super().__init__(
            message=f"File not found: {file_path}",
            file_path=file_path,
            operation=operation,
        )
        self.error_code = "FILE_NOT_FOUND"


# Runtime Errors
class ProjectRootError(MacFairnessError):
    """Raised when project root cannot be determined."""

    def __init__(self):
        super().__init__(
            message="Could not find project root (no pyproject.toml found)",
            error_code="PROJECT_ROOT_NOT_FOUND",
        )


class UnexpectedError(MacFairnessError):
    """Wrapper for unexpected exceptions during execution."""

    def __init__(
        self,
        original_error: Exception,
        context: str = "execution",
        question_id: Optional[str] = None,
    ):
        details = {
            "original_error_type": type(original_error).__name__,
            "original_message": str(original_error),
            "context": context,
        }
        if question_id:
            details["question_id"] = question_id

        super().__init__(
            message=f"Unexpected error during {context}: {type(original_error).__name__}: {original_error}",
            error_code="UNEXPECTED_ERROR",
            details=details,
        )


class DependencyError(MacFairnessError):
    """Raised when a required dependency is missing or misconfigured."""

    def __init__(self, dependency: str, message: str):
        super().__init__(
            message=f"Dependency error for {dependency}: {message}",
            error_code="DEPENDENCY_ERROR",
            details={"dependency": dependency},
        )


# vLLM Engine Errors
class VLLMEngineError(MacFairnessError):
    """Base class for vLLM engine errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "VLLM_ENGINE_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message=message, error_code=error_code, details=details)


class VLLMEngineNotStartedError(VLLMEngineError):
    """Raised when attempting to use engine before starting."""

    def __init__(self):
        super().__init__(
            message="BatchedVLLMEngine not started. Call start() first.",
            error_code="VLLM_ENGINE_NOT_STARTED",
        )


class VLLMEngineNotInitializedError(VLLMEngineError):
    """Raised when engine is not initialized."""

    def __init__(self):
        super().__init__(
            message="BatchedVLLMEngine not initialized. Create an AsyncVLLMAgent first.",
            error_code="VLLM_ENGINE_NOT_INITIALIZED",
        )


class VLLMBatchError(VLLMEngineError):
    """Raised when a batch processing fails."""

    def __init__(
        self,
        message: str,
        batch_size: int,
        request_ids: List[str],
        original_error: Optional[Exception] = None,
    ):
        details = {
            "batch_size": batch_size,
            "request_ids": request_ids,
        }
        if original_error:
            details["original_error"] = str(original_error)
            details["original_error_type"] = type(original_error).__name__
        super().__init__(
            message=message,
            error_code="VLLM_BATCH_ERROR",
            details=details,
        )


class VLLMInferenceError(VLLMEngineError):
    """Raised when vLLM inference fails for a single request."""

    def __init__(
        self,
        message: str,
        request_id: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        details = {}
        if request_id:
            details["request_id"] = request_id
        if original_error:
            details["original_error"] = str(original_error)
            details["original_error_type"] = type(original_error).__name__
        super().__init__(
            message=message,
            error_code="VLLM_INFERENCE_ERROR",
            details=details,
        )


class VLLMOOMError(VLLMEngineError):
    """Raised when vLLM runs out of GPU memory."""

    def __init__(self, max_tokens: int, batch_size: int):
        super().__init__(
            message=f"GPU OOM during inference. max_tokens={max_tokens}, batch_size={batch_size}",
            error_code="VLLM_OOM_ERROR",
            details={"max_tokens": max_tokens, "batch_size": batch_size},
        )


# Ollama Errors
class OllamaError(MacFairnessError):
    """Base class for Ollama-related errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "OLLAMA_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message=message, error_code=error_code, details=details)


class OllamaAPIError(OllamaError):
    """Raised when Ollama API request fails."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
    ):
        details = {}
        if status_code is not None:
            details["status_code"] = status_code
        if response_text is not None:
            details["response_text"] = response_text[:500]  # Truncate long responses
        super().__init__(
            message=message,
            error_code="OLLAMA_API_ERROR",
            details=details if details else None,
        )


class OllamaConnectionError(OllamaError):
    """Raised when connection to Ollama fails."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        details = {}
        if original_error is not None:
            details["original_error"] = str(original_error)
        super().__init__(
            message=message,
            error_code="OLLAMA_CONNECTION_ERROR",
            details=details if details else None,
        )


class OllamaNotAvailableError(OllamaError):
    """Raised when Ollama is not installed or not running."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            error_code="OLLAMA_NOT_AVAILABLE",
        )


# Manifest Errors
class ManifestError(MacFairnessError):
    """Base class for manifest-related errors."""

    def __init__(
        self,
        message: str,
        manifest_path: Optional[str] = None,
        error_code: str = "MANIFEST_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if manifest_path:
            details["manifest_path"] = manifest_path
        super().__init__(message=message, error_code=error_code, details=details)


class ManifestParseError(ManifestError):
    """Raised when manifest JSON parsing fails."""

    def __init__(self, manifest_path: str, original_error: Optional[Exception] = None):
        details = {}
        if original_error:
            details["original_error"] = str(original_error)
            details["original_error_type"] = type(original_error).__name__
        super().__init__(
            message=f"Failed to parse manifest: {manifest_path}",
            manifest_path=manifest_path,
            error_code="MANIFEST_PARSE_ERROR",
            details=details,
        )


class ManifestWriteError(ManifestError):
    """Raised when manifest file write fails."""

    def __init__(self, manifest_path: str, original_error: Optional[Exception] = None):
        details = {}
        if original_error:
            details["original_error"] = str(original_error)
            details["original_error_type"] = type(original_error).__name__
        super().__init__(
            message=f"Failed to write manifest: {manifest_path}",
            manifest_path=manifest_path,
            error_code="MANIFEST_WRITE_ERROR",
            details=details,
        )


# Error Aggregation Utilities
class ErrorCollector:
    """Collects and aggregates errors during processing."""

    def __init__(self):
        self.errors: List[MacFairnessError] = []
        self.error_counts: Dict[str, int] = {}

    def add_error(self, error: MacFairnessError):
        """Add an error to the collection."""
        self.errors.append(error)
        error_code = error.error_code
        self.error_counts[error_code] = self.error_counts.get(error_code, 0) + 1

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of collected errors."""
        return {
            "total_errors": len(self.errors),
            "error_counts": self.error_counts,
            "errors": [e.to_dict() for e in self.errors],
        }

    def get_aggregated_summary(self) -> List[Dict[str, Any]]:
        """Get aggregated error summary for reporting.

        Returns list with error code included for debugging:
        [{"error": "user message", "count": n, "code": "ERROR_CODE"}]
        """
        aggregated = []
        for error_code, count in sorted(self.error_counts.items(), key=lambda x: -x[1]):
            message = _get_error_message(error_code)
            aggregated.append({"error": message, "count": count, "code": error_code})

        return aggregated

    def clear(self):
        """Clear all collected errors."""
        self.errors = []
        self.error_counts = {}

    def has_errors(self) -> bool:
        """Check if any errors have been collected."""
        return len(self.errors) > 0


# Retry handler (placeholder for now)
class RetryHandler:
    """Handles retry logic with error tracking."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.attempt = 0
        self.errors: List[MacFairnessError] = []

    def should_retry(self, error: MacFairnessError) -> bool:
        """Determine if operation should be retried."""
        self.errors.append(error)
        self.attempt += 1

        # Don't retry certain critical errors
        non_retryable = [
            "CONFIG_ERROR",
            "MISSING_CONFIG_SECTION",
            "PROJECT_ROOT_NOT_FOUND",
            "DEPENDENCY_ERROR",
        ]

        if error.error_code in non_retryable:
            return False

        return self.attempt < self.max_retries

    def get_retry_message(self) -> str:
        """Get a message for retry prompt modification."""
        if self.attempt == 1:
            return "Please ensure your response includes valid JSON in ```json blocks."
        elif self.attempt == 2:
            return "Your previous responses were invalid. Please provide a properly formatted JSON response."
        else:
            return "This is your final attempt. Please carefully format your response as valid JSON."

    def raise_max_retries_error(self, agent_id: str):
        """Raise error when max retries exceeded."""
        raise MaxRetriesExceededError(
            agent_id=agent_id,
            max_retries=self.max_retries,
            validation_errors=[e.to_dict() for e in self.errors],
        )
