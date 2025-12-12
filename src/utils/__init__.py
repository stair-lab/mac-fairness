"""Utility modules for the MAC-fairness project."""

from .answer_matcher import FlexibleAnswerMatcher
from .bookkeeping_manager import BookkeepingManager
from .config_manager import ConfigManager
from .errors import (
    ConfigurationError,
    ErrorCollector,
    MacFairnessError,
    ProjectRootError,
    RetryHandler,
    ValidationError,
)
from .logging import (
    DEBUG_ENV,
    LIVE_STATUS_ENV,
    ERROR_CODE_MESSAGES,
    is_debug_enabled,
    is_live_status_enabled,
    debug_print,
    info_print,
    status_print,
    format_timestamp,
    display_path,
    aggregate_validation_errors,
    MetricsCollector,
    get_gpu_info,
    format_gpu_info,
)
from .transcript_manager import TranscriptManager

# Note: ConversationOrchestrator is not exported here to avoid circular imports
# Import directly: from src.utils.conversation_orchestrator import ConversationOrchestrator

__all__ = [
    "aggregate_validation_errors",
    "BookkeepingManager",
    "ConfigManager",
    "ConfigurationError",
    "DEBUG_ENV",
    "debug_print",
    "display_path",
    "ErrorCollector",
    "ERROR_CODE_MESSAGES",
    "FlexibleAnswerMatcher",
    "format_gpu_info",
    "format_timestamp",
    "get_gpu_info",
    "info_print",
    "is_debug_enabled",
    "is_live_status_enabled",
    "LIVE_STATUS_ENV",
    "MacFairnessError",
    "MetricsCollector",
    "ProjectRootError",
    "RetryHandler",
    "status_print",
    "TranscriptManager",
    "ValidationError",
]
