"""Transcript building and persistence utilities."""

import json
import fcntl
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.utils.errors import ProjectRootError
from src.utils.recording import (
    aggregate_validation_errors,
    display_path,
    format_timestamp,
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
        total_tokens_generated: int,
        total_prompt_tokens: int,
        total_time_ms: float,
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
            total_tokens_generated: Total tokens generated
            total_prompt_tokens: Total prompt tokens used
            total_time_ms: Total generation time in milliseconds
            all_validation_errors: All validation errors encountered
            status: Conversation status (success/failed/error)
            error_info: Optional error information for failed conversations

        Returns:
            Complete transcript dictionary
        """
        exp_meta = config["experiment_metadata"]
        conv_config = config["conversation_config"]
        retry_config = config.get("retry_config", {})
        identity_config = config["identity_reveal_config"]
        agent_defs = config["agent_definitions"]
        schema_version = exp_meta.get("schema_version", "2025-11-27")

        # Calculate summary metrics
        total_messages = sum(len(r["messages"]) for r in conversation_rounds)
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

        # Determine consensus (only for success status)
        consensus_reached = None
        if status == "succeeded" and final_answers:
            unique_answers = set(final_answers.values())
            consensus_reached = len(unique_answers) == 1

        # Aggregate validation errors (using shared utility)
        validation_errors_summary = aggregate_validation_errors(all_validation_errors)

        # Calculate retry statistics
        total_retry_attempts = sum(
            msg["message_metadata"]["retry_count"]
            for r in conversation_rounds
            for msg in r["messages"]
        )
        messages_requiring_retries = sum(
            1
            for r in conversation_rounds
            for msg in r["messages"]
            if msg["message_metadata"]["retry_count"] > 0
        )

        # Build job_task_id
        slurm_job = os.environ.get("SLURM_JOB_ID")
        slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_job and slurm_task:
            job_task_id = f"{slurm_job}_{slurm_task}"
        elif slurm_job:
            job_task_id = slurm_job
        else:
            job_task_id = "local"

        # Build complete transcript
        transcript = {
            "transcript_id": transcript_id,
            "protocol_version": schema_version,
            "experiment_metadata": {
                "experiment_name": exp_meta["experiment_name"],
                "benchmark_subcategory": exp_meta["benchmark_subcategory"],
                "config_snapshot_path": snapshot_path,
                "submission_timestamp": format_timestamp(submission_timestamp),
                "execution_timestamp": format_timestamp(execution_timestamp),
                "job_task_id": job_task_id,
            },
            "question": question,
            "routing_config": {
                "strategy": conv_config["routing_strategy"],
                "max_rounds": conv_config["max_rounds"],
            },
            "retry_config": {
                "max_retries": retry_config.get("max_retries", 3),
                "answer_match_threshold": retry_config.get(
                    "answer_match_threshold", 0.85
                ),
                "retry_on_validation_error": retry_config.get(
                    "retry_on_validation_error", True
                ),
                "retry_on_generation_error": retry_config.get(
                    "retry_on_generation_error", True
                ),
            },
            "identity_reveal_config": identity_config,
            "agents": agent_defs,
            "conversation_rounds": conversation_rounds,
            "conversation_summary": {
                "total_rounds": total_rounds,
                "total_messages": total_messages,
                "status": status,
                "final_answers": final_answers,
                "consensus_reached": consensus_reached,
                "performance_metrics": {
                    "total_tokens": total_tokens_generated,
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_time_seconds": round(total_time_ms / 1000, 3)
                    if total_time_ms
                    else 0,
                    "average_response_time_ms": round(total_time_ms / total_messages, 3)
                    if total_messages
                    else 0,
                },
                "retry_statistics": {
                    "total_retry_attempts": total_retry_attempts,
                    "messages_requiring_retries": messages_requiring_retries,
                    "validation_errors_summary": validation_errors_summary,
                },
            },
            "created_at": format_timestamp(datetime.now(timezone.utc)),
        }

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
        exp_root = os.environ.get("MAC_FAIRNESS_EXPERIMENT_ROOT", "experiment")
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

        print(f"✓ Transcript saved: {display_path(transcript_path, self.project_root)}")
        return str(transcript_path)

    def append_to_index(
        self,
        transcript: Dict[str, Any],
        question: Dict[str, Any],
        config: Dict[str, Any],
    ):
        """Append transcript information to JSONL index.

        Args:
            transcript: Complete transcript
            question: Original question
            config: Full configuration dictionary
        """
        benchmark = config["experiment_metadata"]["benchmark_subcategory"]

        # Use separate index for dev benchmarks, production uses main index
        # All are in bookkeeping/ directory
        if benchmark == "dev_ollama":
            index_path = self.project_root / "bookkeeping" / "dev_ollama_index.jsonl"
        elif benchmark == "dev_snap":
            index_path = self.project_root / "bookkeeping" / "dev_snap_index.jsonl"
        elif benchmark == "dev_sherlock":
            index_path = self.project_root / "bookkeeping" / "dev_sherlock_index.jsonl"
        else:
            index_path = self.project_root / "bookkeeping" / "index.jsonl"

        # Create if doesn't exist
        if not index_path.exists():
            index_path.touch()

        # Build index entry
        exp_meta = config["experiment_metadata"]
        conv_config = config["conversation_config"]
        identity_config = config["identity_reveal_config"]
        agent_defs = config["agent_definitions"]
        summary = transcript["conversation_summary"]

        # Build transcript_path using display_path format
        # This handles MAC_FAIRNESS_EXPERIMENT_ROOT if set
        exp_root = os.environ.get("MAC_FAIRNESS_EXPERIMENT_ROOT", "experiment")
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

        index_entry = {
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
            "protocol_version": "2025-11-27",
            "routing_strategy": conv_config["routing_strategy"],
            "identity_reveal_config": identity_config,
            "n_agents": len(agent_defs),
            "shared_model_backbone": config["model_config"].get(
                "shared_model_backbone"
            ),
            "agents": agent_defs,
            "status": summary["status"],
            "consensus_reached": summary["consensus_reached"],
            "total_rounds_completed": summary["total_rounds"],
            "retry_attempts": summary["retry_statistics"]["total_retry_attempts"],
            "fatal_error": self._strip_error_details(summary.get("error_info")),
        }

        # Write with file locking
        with open(index_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(index_entry) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

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
