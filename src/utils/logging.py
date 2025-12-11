"""Centralized logging, formatting, and metrics utilities for MAC-fairness.

This module provides:
- Debug and info printing functions with runtime environment variable checks
- Path display formatting (replacing absolute paths with env var names)
- Timestamp formatting for ISO 8601
- Error message aggregation utilities
- Performance metrics collection for conversations and experiments

Environment variables:
- MAC_FAIRNESS_DEBUG_FLAG: Enable debug output
- MAC_FAIRNESS_LIVE_STATUS: Enable live status display (suppresses some output)
- MAC_FAIRNESS_WORKSPACE: Project root directory
- MAC_FAIRNESS_EXPERIMENT_ROOT: External experiments directory
"""

import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Environment variable names
DEBUG_ENV = "MAC_FAIRNESS_DEBUG_FLAG"
LIVE_STATUS_ENV = "MAC_FAIRNESS_LIVE_STATUS"
WORKSPACE_ENV = "MAC_FAIRNESS_WORKSPACE"
EXPERIMENT_ROOT_ENV = "MAC_FAIRNESS_EXPERIMENT_ROOT"

# Display names for path substitution
_WORKSPACE_VAR = "$MAC_FAIRNESS_WORKSPACE"
_EXPERIMENT_ROOT_VAR = "$MAC_FAIRNESS_EXPERIMENT_ROOT"

# Shared error code to user-friendly message mapping
ERROR_CODE_MESSAGES = {
    "INVALID_ANSWER": "Invalid answer: has to be from choice text",
    "MISSING_STRUCTURED_OUTPUT": "Missing structured output in response",
    "MAX_LENGTH_EXCEEDED": "Response exceeded maximum token limit",
    "MAX_RETRIES_EXCEEDED": "Maximum retry attempts exceeded",
    "JSON_PARSE_REPAIRED": "JSON parsing required repair",
    "JSON_PARSE_FAILED": "JSON parsing failed",
}


# =============================================================================
# Environment Variable Checks
# =============================================================================


def is_debug_enabled() -> bool:
    """Check if debug flag is enabled (checked at runtime)."""
    return os.environ.get(DEBUG_ENV) == "1"


def is_live_status_enabled() -> bool:
    """Check if live status display is enabled (checked at runtime)."""
    return os.environ.get(LIVE_STATUS_ENV) == "1"


# =============================================================================
# Print Functions
# =============================================================================


def debug_print(msg: str) -> None:
    """Print debug message if MAC_FAIRNESS_DEBUG_FLAG is set.

    Args:
        msg: Message to print (will be prefixed with [DEBUG])
    """
    if is_debug_enabled():
        print(f"[DEBUG] {msg}")


def info_print(msg: str, prefix: bool = True) -> None:
    """Print info message.

    Args:
        msg: Message to print
        prefix: If True, prefix with [INFO] (default True)
    """
    if prefix:
        print(f"[INFO] {msg}")
    else:
        print(msg)


def status_print(msg: str) -> None:
    """Print status message only if live status display is NOT enabled.

    Use this for progress messages that would interfere with live display.

    Args:
        msg: Message to print
    """
    if not is_live_status_enabled():
        print(msg)


# =============================================================================
# Formatting Functions
# =============================================================================


def format_timestamp(dt: datetime) -> str:
    """Format a datetime to ISO 8601 with microseconds and Z suffix.

    Args:
        dt: Datetime object (timezone-aware or naive, assumed UTC if naive)

    Returns:
        ISO 8601 formatted string like "2025-12-04T12:00:00.000000Z"
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def display_path(
    path: Union[str, Path], project_root: Union[str, Path, None] = None
) -> str:
    """Format a path for display, replacing known roots with environment variable names.

    Handles two environment variables:
    - $MAC_FAIRNESS_WORKSPACE: Project root (always replaced)
    - $MAC_FAIRNESS_EXPERIMENT_ROOT: External experiments directory (if set)

    Args:
        path: The path to format (absolute or relative)
        project_root: Project root to replace (auto-detected if None)

    Returns:
        Path string with roots replaced by environment variable names
    """
    path_str = str(path)

    exp_root = os.environ.get(EXPERIMENT_ROOT_ENV)
    if exp_root:
        exp_root_str = str(Path(exp_root).resolve())
        if path_str.startswith(exp_root_str):
            return path_str.replace(exp_root_str, _EXPERIMENT_ROOT_VAR, 1)

    if project_root is None:
        current = Path(__file__).resolve()
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                project_root = current
                break
            current = current.parent
        else:
            return path_str

    root_str = str(project_root)

    if path_str.startswith(root_str):
        return path_str.replace(root_str, _WORKSPACE_VAR, 1)

    return path_str


# =============================================================================
# Error Aggregation
# =============================================================================


def aggregate_validation_errors(
    all_validation_errors: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate validation errors by error code into user-friendly summaries.

    Args:
        all_validation_errors: List of error dicts with 'error_code' or 'error_class'

    Returns:
        List of {"error": message, "count": n} sorted by frequency descending
    """
    if not all_validation_errors:
        return []

    error_code_counts: Dict[str, int] = defaultdict(int)

    for error in all_validation_errors:
        error_code = error.get("error_code")
        if not error_code:
            error_class = error.get("error_class")
            if not error_class:
                continue
            error_code = error_class.replace("Error", "").upper()

        generic_msg = ERROR_CODE_MESSAGES.get(
            error_code, error_code.replace("_", " ").title()
        )
        error_code_counts[generic_msg] += 1

    return [
        {"error": msg, "count": count}
        for msg, count in sorted(error_code_counts.items(), key=lambda x: -x[1])
    ]


# =============================================================================
# Metrics Collection
# =============================================================================


class MetricsCollector:
    """Collects and aggregates performance metrics for conversations."""

    def calculate_conversation_metrics(
        self, conversation_rounds: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate metrics for a single conversation.

        Args:
            conversation_rounds: List of conversation round data

        Returns:
            Dictionary of calculated metrics
        """
        total_messages = sum(len(r["messages"]) for r in conversation_rounds)
        total_rounds = len(conversation_rounds)

        total_tokens_generated = 0
        total_tokens_prompt = 0

        for round_data in conversation_rounds:
            for msg in round_data["messages"]:
                metadata = msg.get("message_metadata", {})
                perf = metadata.get("performance", {})

                total_tokens_generated += perf.get("tokens_generated", 0)
                total_tokens_prompt += perf.get("prompt_tokens", 0)

        return {
            "total_messages": total_messages,
            "total_rounds": total_rounds,
            "total_tokens_generated": total_tokens_generated,
            "total_tokens_prompt": total_tokens_prompt,
        }

    def extract_final_answers(
        self, conversation_rounds: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract final answers from the last round of conversation.

        Args:
            conversation_rounds: List of conversation round data

        Returns:
            Dictionary mapping agent_id to their final answer
        """
        final_answers = {}

        if conversation_rounds and conversation_rounds[-1]["messages"]:
            for msg in conversation_rounds[-1]["messages"]:
                agent_id = msg["agent_id"]
                response = msg.get("structured_response", {})

                if response.get("response_type") == "participant":
                    final_answers[agent_id] = response.get("opinion")
                elif response.get("response_type") == "judge":
                    final_answers[agent_id] = response.get("verdict")

        return final_answers

    def check_consensus(self, final_answers: Dict[str, Any]) -> Optional[bool]:
        """Check if consensus was reached among agents.

        Args:
            final_answers: Dictionary of agent final answers

        Returns:
            True if consensus reached, False if not, None if no answers
        """
        if not final_answers:
            return None

        unique_answers = set(final_answers.values())
        return len(unique_answers) == 1

    def calculate_retry_statistics(
        self, conversation_rounds: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate retry statistics from conversation rounds.

        Args:
            conversation_rounds: List of conversation round data

        Returns:
            Dictionary of retry statistics
        """
        total_retry_attempts = 0
        messages_requiring_retries = 0
        max_retries_per_message = 0

        for round_data in conversation_rounds:
            for msg in round_data["messages"]:
                retry_count = msg.get("message_metadata", {}).get("retry_count", 0)
                total_retry_attempts += retry_count
                if retry_count > 0:
                    messages_requiring_retries += 1
                    max_retries_per_message = max(max_retries_per_message, retry_count)

        return {
            "total_retry_attempts": total_retry_attempts,
            "messages_requiring_retries": messages_requiring_retries,
            "max_retries_per_message": max_retries_per_message,
        }

    def aggregate_validation_errors(
        self, all_validation_errors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Aggregate validation errors using error codes.

        Args:
            all_validation_errors: List of all validation errors

        Returns:
            List of aggregated error summaries with counts
        """
        return aggregate_validation_errors(all_validation_errors)

    def calculate_experiment_summary(
        self, transcripts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate summary statistics for an entire experiment.

        Args:
            transcripts: List of conversation transcripts

        Returns:
            Dictionary of experiment-level statistics
        """
        total_conversations = len(transcripts)
        successful_conversations = sum(
            1
            for t in transcripts
            if t.get("conversation_summary", {}).get("status") == "succeeded"
        )
        failed_conversations = total_conversations - successful_conversations

        total_tokens_all = sum(
            t.get("conversation_summary", {})
            .get("performance_metrics", {})
            .get("total_tokens", 0)
            for t in transcripts
        )

        consensus_count = sum(
            1
            for t in transcripts
            if t.get("conversation_summary", {}).get("consensus_reached") is True
        )

        all_errors = []
        for t in transcripts:
            errors = (
                t.get("conversation_summary", {})
                .get("retry_statistics", {})
                .get("validation_errors_summary", [])
            )
            all_errors.extend(errors)

        aggregated_errors = self.aggregate_validation_errors(all_errors)

        return {
            "total_conversations": total_conversations,
            "successful_conversations": successful_conversations,
            "failed_conversations": failed_conversations,
            "success_rate": successful_conversations / total_conversations
            if total_conversations > 0
            else 0,
            "consensus_rate": consensus_count / successful_conversations
            if successful_conversations > 0
            else 0,
            "total_tokens_used": total_tokens_all,
            "average_tokens_per_conversation": total_tokens_all / total_conversations
            if total_conversations > 0
            else 0,
            "error_summary": aggregated_errors,
        }
