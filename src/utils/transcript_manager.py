"""Transcript building and persistence utilities."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.utils.errors import ProjectRootError
from src.utils.bookkeeping_manager import get_current_job_task_id
from src.utils.logging import (
    EXPERIMENT_ROOT_ENV,
    aggregate_validation_errors,
    display_path,
    format_timestamp,
    info_print,
    is_debug_enabled,
    is_live_status_enabled,
)


class TranscriptManager:
    """Manages conversation transcript building and persistence."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the transcript manager.

        Args:
            project_root: Project root directory (auto-detected if None)
        """
        if project_root is None:
            current = Path(__file__).resolve()
            while current != current.parent:
                if (current / "pyproject.toml").exists():
                    self.project_root = current
                    break
                current = current.parent
            else:
                raise ProjectRootError()
        else:
            self.project_root = project_root

    def build_transcript(
        self,
        transcript_id: str,
        question: Dict[str, Any],
        conversation_rounds: List[Dict[str, Any]],
        config: Dict[str, Any],
        snapshot_path: str,
        submission_timestamp: datetime,
        execution_timestamp: datetime,
        token_stats: Dict[str, Any],
        all_validation_errors: List[Dict[str, Any]],
        status: str = "succeeded",
        error_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a complete conversation transcript.

        Args:
            transcript_id: Unique ID for the transcript
            question: Question dictionary
            conversation_rounds: List of conversation round data
            config: Full configuration dictionary
            snapshot_path: Path to config snapshot
            submission_timestamp: When the experiment was submitted
            execution_timestamp: When this conversation was executed
            token_stats: Token statistics dictionary with:
                - total_messages: Number of messages
                - prompt_tokens: {total, avg, max}
                - response_tokens: {total, avg, max}
                - combined_tokens: {total, avg, max}
            all_validation_errors: All validation errors encountered
            status: Conversation status (success/failed/error)
            error_info: Optional error information for failed conversations

        Returns:
            Complete transcript dictionary
        """
        exp_meta = config["experiment_metadata"]
        conversation_config = config["conversation_config"]
        retry_config = config.get("retry_config", {})
        identity_config = config["identity_reveal_config"]
        agent_defs = config["agent_definitions"]
        schema_version = exp_meta["schema_version"]

        # Summary metrics
        total_rounds = len(conversation_rounds)

        # Extract final answers if available
        final_answers = {}
        if conversation_rounds and conversation_rounds[-1]["messages"]:
            for msg in conversation_rounds[-1]["messages"]:
                agent_id = msg["agent_id"]
                response = msg["structured_response"]
                # Extract answer based on response type
                if response.get("response_type") == "participant":
                    final_answers[agent_id] = response.get("opinion")
                elif response.get("response_type") == "judge":
                    final_answers[agent_id] = response.get("verdict")

        # Determine consensus (whenever there are final answers)
        consensus_reached = None
        if final_answers:
            unique_answers = set(final_answers.values())
            consensus_reached = len(unique_answers) == 1

        # Aggregate validation errors (using shared utility)
        validation_errors_summary = aggregate_validation_errors(all_validation_errors)

        # Calculate retry statistics
        # Logic: Each validation error represents a failed attempt that was retried, EXCEPT
        # the final error of a failed request (which exhausted retries without triggering another).
        #
        # For successful messages: each error triggered a retry that eventually succeeded
        # For failed messages: last error didn't trigger a retry (retries exhausted)
        #
        # Using all_validation_errors ensures we count retries from ALL messages, including
        # those in incomplete rounds (e.g., when one agent fails mid-round, other successful
        # agents' retries in that round are still counted).
        has_failed_request = error_info and error_info.get("details", {}).get(
            "validation_errors"
        )
        total_retry_attempts = len(all_validation_errors) - (1 if has_failed_request else 0)
        total_retry_attempts = max(0, total_retry_attempts)  # Guard against edge cases

        # Count messages that required retries
        # From completed rounds: messages with retry_count > 0
        messages_requiring_retries = sum(
            1
            for r in conversation_rounds
            for msg in r["messages"]
            if msg["message_metadata"]["retry_count"] > 0
        )

        # From incomplete rounds: we can't know exact count, but if there are errors
        # in all_validation_errors that aren't from the failed request, some messages
        # in incomplete rounds had retries. Count based on errors from successful messages.
        failed_error_count = (
            len(error_info.get("details", {}).get("validation_errors", []))
            if error_info
            else 0
        )
        successful_errors_count = len(all_validation_errors) - failed_error_count
        completed_round_errors = sum(
            msg["message_metadata"]["retry_count"]
            for r in conversation_rounds
            for msg in r["messages"]
        )
        incomplete_round_retries = successful_errors_count - completed_round_errors
        if incomplete_round_retries > 0:
            # There were successful messages with retries in incomplete rounds
            # We conservatively estimate at least 1 message required retries
            # (could be more, but we don't have per-message breakdown)
            messages_requiring_retries += 1

        # If there was a fatal error with validation errors, that's one more message requiring retries
        if has_failed_request:
            messages_requiring_retries += 1

        # Build job_task_id
        job_task_id = get_current_job_task_id()

        # Build experiment_metadata (always includes question_id)
        experiment_metadata = {
            "experiment_name": exp_meta["experiment_name"],
            "benchmark_subcategory": exp_meta["benchmark_subcategory"],
            "config_snapshot_path": snapshot_path,
            "submission_timestamp": format_timestamp(submission_timestamp),
            "execution_timestamp": format_timestamp(execution_timestamp),
            "job_task_id": job_task_id,
            "question_id": question["question_id"],
        }

        # Build complete transcript
        transcript = {
            "transcript_id": transcript_id,
            "protocol_version": schema_version,
            "experiment_metadata": experiment_metadata,
            "conversation_rounds": conversation_rounds,
            "conversation_summary": {
                "total_rounds": total_rounds,
                "total_messages": token_stats.get("total_messages", 0),
                "status": status,
                "final_answers": final_answers,
                "consensus_reached": consensus_reached,
                "token_metrics": token_stats,
                "retry_statistics": {
                    "total_retry_attempts": total_retry_attempts,
                    "messages_requiring_retries": messages_requiring_retries,
                    "validation_errors_summary": validation_errors_summary,
                },
            },
            "created_at": format_timestamp(datetime.now(timezone.utc)),
        }

        # Debug-only fields: question, conversation_config, retry_config,
        # identity_reveal_config, agent_definitions
        if is_debug_enabled():
            transcript["question"] = question
            transcript["conversation_config"] = {
                "routing_strategy": conversation_config["routing_strategy"],
                "max_rounds": conversation_config["max_rounds"],
            }
            transcript["retry_config"] = {
                "max_retries": retry_config["max_retries"],
                "answer_match_threshold": retry_config["answer_match_threshold"],
                "retry_on_validation_error": retry_config["retry_on_validation_error"],
                "retry_on_generation_error": retry_config["retry_on_generation_error"],
            }
            transcript["identity_reveal_config"] = identity_config
            transcript["agent_definitions"] = agent_defs

        # Add error info if conversation failed
        if error_info:
            transcript["conversation_summary"]["error_info"] = error_info

        return transcript

    def save_transcript(self, transcript: Dict[str, Any]) -> str:
        """Save transcript to file.

        Args:
            transcript: Complete transcript dictionary

        Returns:
            Path to the saved transcript file
        """
        exp_root = os.environ.get(EXPERIMENT_ROOT_ENV, "experiment")
        exp_root_path = Path(exp_root)
        if not exp_root_path.is_absolute():
            exp_root_path = self.project_root / exp_root

        benchmark = transcript["experiment_metadata"]["benchmark_subcategory"]
        experiment = transcript["experiment_metadata"]["experiment_name"]

        transcript_path = (
            exp_root_path
            / benchmark
            / experiment
            / "transcript"
            / f"{transcript['transcript_id']}.json"
        )

        # Create directory if it doesn't exist
        transcript_path.parent.mkdir(parents=True, exist_ok=True)

        with open(transcript_path, "w") as f:
            json.dump(transcript, f, indent=2)

        # Only print if live status display is not enabled
        if not is_live_status_enabled():
            info_print(
                f"Transcript saved: {display_path(transcript_path, self.project_root)}"
            )
        return str(transcript_path)

    def get_index_path(self, benchmark: str) -> Path:
        """Get the index file path for a benchmark.

        Args:
            benchmark: Benchmark subcategory name

        Returns:
            Path to the appropriate index.jsonl file
        """
        # Use separate index for dev benchmarks, production uses main index
        # All are in bookkeeping/ directory
        if benchmark == "dev_ollama":
            return self.project_root / "bookkeeping" / "dev_ollama_index.jsonl"
        elif benchmark == "dev_vllm":
            return self.project_root / "bookkeeping" / "dev_vllm_index.jsonl"
        else:
            return self.project_root / "bookkeeping" / "index.jsonl"

    def build_index_entry(
        self,
        transcript: Dict[str, Any],
        question: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build an index entry for a completed transcript.

        Args:
            transcript: Complete transcript
            question: Original question
            config: Full configuration dictionary

        Returns:
            Index entry dictionary ready to be appended to index.jsonl
        """
        benchmark = config["experiment_metadata"]["benchmark_subcategory"]
        exp_meta = config["experiment_metadata"]
        conversation_config = config["conversation_config"]
        identity_config = config["identity_reveal_config"]
        prompt_template_config = config.get("prompt_template_config", {})
        agent_defs = config["agent_definitions"]
        summary = transcript["conversation_summary"]

        # Build transcript_path using display_path format
        # This handles EXPERIMENT_ROOT_ENV if set
        exp_root = os.environ.get(EXPERIMENT_ROOT_ENV, "experiment")
        exp_root_path = Path(exp_root)
        if not exp_root_path.is_absolute():
            exp_root_path = self.project_root / exp_root

        transcript_abs_path = (
            exp_root_path
            / benchmark
            / exp_meta["experiment_name"]
            / "transcript"
            / f"{transcript['transcript_id']}.json"
        )
        transcript_display = display_path(transcript_abs_path, self.project_root)

        # config_snapshot_path is already in display_path format from save_snapshot
        config_snapshot_display = transcript["experiment_metadata"][
            "config_snapshot_path"
        ]

        return {
            "transcript_id": transcript["transcript_id"],
            "experiment_name": exp_meta["experiment_name"],
            "benchmark_subcategory": benchmark,
            "question_id": question["question_id"],
            "job_task_id": transcript["experiment_metadata"]["job_task_id"],
            "submission_timestamp": transcript["experiment_metadata"][
                "submission_timestamp"
            ],
            "execution_timestamp": transcript["experiment_metadata"][
                "execution_timestamp"
            ],
            "transcript_path": transcript_display,
            "config_snapshot_path": config_snapshot_display,
            "protocol_version": exp_meta["schema_version"],
            "routing_strategy": conversation_config["routing_strategy"],
            "identity_reveal_config": identity_config,
            "prompt_template_config": prompt_template_config,
            "n_agents": len(agent_defs),
            "agent_definitions": agent_defs,
            "status": summary["status"],
            "consensus_reached": summary["consensus_reached"],
            "total_rounds_completed": summary["total_rounds"],
            "retry_attempts": summary["retry_statistics"]["total_retry_attempts"],
            "fatal_error": self._strip_error_details(summary.get("error_info")),
        }

    def _strip_error_details(
        self, error_info: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Strip the 'details' field from error_info for index entries.

        Args:
            error_info: Error information dictionary or None

        Returns:
            Error info without 'details' field, or None if input is None
        """
        if error_info is None:
            return None
        # Return a copy without the 'details' key
        return {k: v for k, v in error_info.items() if k != "details"}
