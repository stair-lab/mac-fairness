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
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Environment variable names (exported for use by other modules)
DEBUG_ENV = "MAC_FAIRNESS_DEBUG_FLAG"
LIVE_STATUS_ENV = "MAC_FAIRNESS_LIVE_STATUS"
WORKSPACE_ENV = "MAC_FAIRNESS_WORKSPACE"
EXPERIMENT_ROOT_ENV = "MAC_FAIRNESS_EXPERIMENT_ROOT"

# Display names for path substitution
WORKSPACE_VAR = "$MAC_FAIRNESS_WORKSPACE"
EXPERIMENT_ROOT_VAR = "$MAC_FAIRNESS_EXPERIMENT_ROOT"

# Keep private aliases for backwards compatibility within this module
_WORKSPACE_VAR = WORKSPACE_VAR
_EXPERIMENT_ROOT_VAR = EXPERIMENT_ROOT_VAR

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
# Hardware Info
# =============================================================================


def get_gpu_info() -> Optional[Dict[str, Any]]:
    """Detect GPU information using nvidia-smi and CUDA_VISIBLE_DEVICES.

    Returns:
        Dictionary with GPU info or None if detection fails:
        - name: GPU model name (e.g., "NVIDIA H100 80GB HBM3")
        - memory_gb: Memory per GPU in GB
        - count: Number of GPUs (respects CUDA_VISIBLE_DEVICES)
    """
    try:
        # Get GPU IDs from CUDA_VISIBLE_DEVICES if set
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            # CUDA_VISIBLE_DEVICES can be "0,1,2" or "0" or "" (empty = no GPUs)
            if cuda_visible.strip() == "":
                return None
            gpu_ids = [id.strip() for id in cuda_visible.split(",")]
            gpu_count = len(gpu_ids)
            # Query specific GPU by ID
            first_gpu_id = gpu_ids[0]
            gpu_id_flag = f"--id={first_gpu_id}"
        else:
            # No CUDA_VISIBLE_DEVICES set, count all GPUs via nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            )
            gpu_count = int(result.stdout.strip().split("\n")[0].strip())
            if gpu_count == 0:
                return None
            gpu_id_flag = None  # Query first GPU by default

        # Build nvidia-smi command with optional GPU ID filter
        name_cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        memory_cmd = [
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ]
        if gpu_id_flag:
            name_cmd.insert(1, gpu_id_flag)
            memory_cmd.insert(1, gpu_id_flag)

        # Get GPU name (from first visible GPU)
        result = subprocess.run(
            name_cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        gpu_name = result.stdout.strip().split("\n")[0].strip()

        # Get memory per GPU (from first visible GPU)
        result = subprocess.run(
            memory_cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        memory_mb = int(result.stdout.strip().split("\n")[0].strip())
        memory_gb = memory_mb / 1024

        return {
            "name": gpu_name,
            "memory_gb": memory_gb,
            "count": gpu_count,
        }
    except Exception:
        return None


def format_gpu_info(gpu_info: Optional[Dict[str, Any]]) -> str:
    """Format GPU info for display.

    Args:
        gpu_info: Dictionary from get_gpu_info() or None

    Returns:
        Formatted string like "2x NVIDIA H100 (80GB)" or "Unknown"
    """
    if not gpu_info:
        return "Unknown"

    count = gpu_info.get("count", 1)
    name = gpu_info.get("name", "Unknown GPU")
    memory_gb = gpu_info.get("memory_gb", 0)

    # Simplify GPU name (remove "NVIDIA " prefix if present)
    if name.startswith("NVIDIA "):
        name = name[7:]

    if count > 1:
        return f"{count}x {name} ({memory_gb:.0f}GB)"
    else:
        return f"{name} ({memory_gb:.0f}GB)"


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


def format_filename_timestamp(dt: datetime) -> str:
    """Format a datetime for use in filenames with millisecond precision.

    Uses a filesystem-safe format without colons, with millisecond precision
    to avoid collisions when multiple jobs start within the same second.

    Args:
        dt: Datetime object (timezone-aware or naive, assumed UTC if naive)

    Returns:
        Filename-safe timestamp like "20251204T120000.123Z"
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    # Format: YYYYMMDDTHHMMss.mmmZ (millisecond precision)
    return dt.strftime("%Y%m%dT%H%M%S") + f".{dt.microsecond // 1000:03d}Z"


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
            "success_rate": float(f"{successful_conversations / total_conversations:.5f}")
            if total_conversations > 0
            else 0.0,
            "consensus_rate": float(f"{consensus_count / successful_conversations:.5f}")
            if successful_conversations > 0
            else 0.0,
            "total_tokens_used": total_tokens_all,
            "average_tokens_per_conversation": total_tokens_all / total_conversations
            if total_conversations > 0
            else 0,
            "error_summary": aggregated_errors,
        }
