"""Utilities for recording and persisting experiment data."""

import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Union

# Environment variable names for path display
_WORKSPACE_VAR = "$MAC_FAIRNESS_WORKSPACE"
_EXPERIMENT_ROOT_VAR = "$MAC_FAIRNESS_EXPERIMENT_ROOT"

# Shared error code to user-friendly message mapping
# Used by ErrorCollector, TranscriptManager, and MetricsCollector
ERROR_CODE_MESSAGES = {
    "INVALID_ANSWER": "Invalid answer: has to be from choice text",
    "MISSING_STRUCTURED_OUTPUT": "Missing structured output in response",
    "ZOD_VALIDATION_FAILED": "Response format validation failed",
    "JSON_DECODE_FAILED": "Failed to parse JSON from response",
    "MAX_LENGTH_EXCEEDED": "Response exceeded maximum token limit",
    "MAX_RETRIES_EXCEEDED": "Maximum retry attempts exceeded",
    "UNEXPECTED_ERROR": "Unexpected error during execution",
}


def format_timestamp(dt: datetime) -> str:
    """Format a datetime to ISO 8601 with microseconds and Z suffix.

    Args:
        dt: Datetime object (timezone-aware or naive, assumed UTC if naive)

    Returns:
        ISO 8601 formatted string like "2025-12-04T12:00:00.000000Z"
    """
    # Ensure UTC timezone
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    # Format without timezone offset, then add Z
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

    # Check for MAC_FAIRNESS_EXPERIMENT_ROOT first (more specific)
    exp_root = os.environ.get("MAC_FAIRNESS_EXPERIMENT_ROOT")
    if exp_root:
        exp_root_str = str(Path(exp_root).resolve())
        if path_str.startswith(exp_root_str):
            return path_str.replace(exp_root_str, _EXPERIMENT_ROOT_VAR, 1)

    # Auto-detect project root if not provided
    if project_root is None:
        current = Path(__file__).resolve()
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                project_root = current
                break
            current = current.parent
        else:
            # Can't find project root, return path as-is
            return path_str

    root_str = str(project_root)

    # Replace project root with workspace variable
    if path_str.startswith(root_str):
        return path_str.replace(root_str, _WORKSPACE_VAR, 1)

    return path_str


def aggregate_validation_errors(
    all_validation_errors: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate validation errors by error code into user-friendly summaries.

    This shared utility handles error aggregation at any granularity level:
    - Per-generation: errors from a single _generate_with_strict_validation call
    - Per-transcript: errors from one conversation
    - Per-experiment: errors across multiple transcripts

    Args:
        all_validation_errors: List of error dicts with 'error_code' or 'error_class'

    Returns:
        List of {"error": message, "count": n} sorted by frequency descending
    """
    if not all_validation_errors:
        return []

    error_code_counts: Dict[str, int] = defaultdict(int)

    for error in all_validation_errors:
        # Use error_code for aggregation, or error_class as fallback
        error_code = error.get("error_code")
        if not error_code:
            error_class = error.get("error_class")
            if not error_class:
                continue
            error_code = error_class.replace("Error", "").upper()

        # Map to user-friendly message
        generic_msg = ERROR_CODE_MESSAGES.get(
            error_code, error_code.replace("_", " ").title()
        )
        error_code_counts[generic_msg] += 1

    return [
        {"error": msg, "count": count}
        for msg, count in sorted(error_code_counts.items(), key=lambda x: -x[1])
    ]
