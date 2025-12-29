"""Bookkeeping and record management utilities."""

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, TextIO, Tuple
from collections import Counter

from src.utils.errors import ManifestParseError, ManifestWriteError, ProjectRootError
from src.utils.logging import (
    MAC_FAIRNESS_EXPERIMENT_ROOT,
    MAC_FAIRNESS_WORKSPACE,
    display_path,
    format_filename_timestamp,
    format_timestamp,
    info_print,
    is_debug_enabled,
    resolve_path,
)


def get_job_task_id(grid_index: Optional[int] = None) -> str:
    """Get job-task identifier for per-task tracking.

    Uses the process ID as the base identifier. For grid runs,
    appends the grid index.

    Args:
        grid_index: Optional 0-indexed grid configuration index

    Returns:
        Job-task identifier string: "{pid}" or "{pid}_{grid_index}"
    """
    pid = os.getpid()
    if grid_index is not None:
        return f"{pid}_{grid_index}"
    return str(pid)


# Module-level grid index for the current task (set by run_job.py for grid jobs)
_current_grid_index: Optional[int] = None


def set_grid_index(index: Optional[int]) -> None:
    """Set the current grid index for this process.

    Called by run_job.py at the start of each grid task.

    Args:
        index: 0-indexed grid configuration index, or None for non-grid runs
    """
    global _current_grid_index
    _current_grid_index = index


def get_current_job_task_id() -> str:
    """Get job-task identifier using the module-level grid index.

    Returns:
        Job-task identifier string: "{pid}" or "{pid}_{grid_index}"
    """
    return get_job_task_id(_current_grid_index)


class BookkeepingManager:
    """Manages bookkeeping operations including directories, job summaries, and other records."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the bookkeeping manager.

        Args:
            project_root: Project root directory (from MAC_FAIRNESS_WORKSPACE if None)
        """
        if project_root is None:
            workspace = os.environ.get(MAC_FAIRNESS_WORKSPACE)
            if workspace:
                self.project_root = Path(workspace)
            else:
                raise ProjectRootError()
        else:
            self.project_root = project_root

    def ensure_directories(self, config: Dict[str, Any]) -> Dict[str, Path]:
        """Create necessary directories for bookkeeping and experiment outputs.

        Args:
            config: Full configuration dictionary

        Returns:
            Dictionary of created directory paths
        """
        exp_root = os.environ.get(MAC_FAIRNESS_EXPERIMENT_ROOT, "experiment")
        # Make exp_root absolute if not already
        exp_root_path = Path(exp_root)
        if not exp_root_path.is_absolute():
            exp_root_path = self.project_root / exp_root

        exp_meta = config["experiment_metadata"]
        benchmark = exp_meta["benchmark_subcategory"]
        experiment = exp_meta["experiment_name"]

        # Bookkeeping directories (always in project root)
        bookkeeping_dir = self.project_root / "bookkeeping"
        bookkeeping_dir.mkdir(exist_ok=True)

        config_snapshot_dir = bookkeeping_dir / "config_snapshot" / benchmark
        config_snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Experiment directories (can be elsewhere)
        experiment_dir = exp_root_path / benchmark / experiment
        transcript_dir = experiment_dir / "transcript"
        transcript_dir.mkdir(parents=True, exist_ok=True)

        task_summary_dir = experiment_dir / "task_summary"
        task_summary_dir.mkdir(parents=True, exist_ok=True)

        info_print(
            f"Directories ensured for: {display_path(experiment_dir, self.project_root)}"
        )

        return {
            "bookkeeping": bookkeeping_dir,
            "config_snapshot": config_snapshot_dir,
            "experiment": experiment_dir,
            "transcript": transcript_dir,
            "task_summary": task_summary_dir,
        }

    def save_task_summary(
        self,
        config: Dict[str, Any],
        questions_total: int,
        questions_succeeded: int,
        questions_partial: int,
        questions_failed: int,
        start_time: datetime,
        end_time: datetime,
        error_summary: Optional[List[Dict[str, Any]]] = None,
        per_transcript_stats: Optional[List[Dict[str, Any]]] = None,
        config_snapshot_path: Optional[str] = None,
        effective_backend_config: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        """Save a comprehensive job summary for the experiment run.

        Args:
            config: Full configuration dictionary
            questions_total: Total number of questions attempted
            questions_succeeded: Number of successful conversations
            questions_partial: Number of partially completed conversations
            questions_failed: Number of failed conversations
            start_time: When the job started
            end_time: When the job ended
            error_summary: Optional list of error information
            per_transcript_stats: Optional list of per-transcript statistics
            effective_backend_config: Optional dict of effective backend config per model
                keyed by model_path (includes auto-calculated values like max_num_seqs)

        Returns:
            Path to the saved job summary file
        """
        exp_root = os.environ.get(MAC_FAIRNESS_EXPERIMENT_ROOT, "experiment")
        exp_root_path = Path(exp_root)
        if not exp_root_path.is_absolute():
            exp_root_path = self.project_root / exp_root

        exp_meta = config["experiment_metadata"]
        conversation_config = config["conversation_config"]
        retry_config = config.get("retry_config", {})
        identity_reveal_config = config.get("identity_reveal_config", {})
        prompt_template_config = config.get("prompt_template_config", {})
        model_definitions = config.get("model_definitions", {})
        agent_defs = config["agent_definitions"]
        benchmark = exp_meta["benchmark_subcategory"]
        experiment = exp_meta["experiment_name"]

        # Build job ID: "{pid}" or "{pid}_{grid_index}"
        job_task_id = get_current_job_task_id()

        # Filename: {timestamp}_{job_task_id}.json
        timestamp_str = format_filename_timestamp(start_time)
        filename_id = f"{timestamp_str}_{job_task_id}"

        # Calculate duration
        duration_seconds = (end_time - start_time).total_seconds()

        # Aggregate per-transcript statistics
        aggregated_stats = self._aggregate_transcript_stats(
            per_transcript_stats or [],
            agent_defs,
            duration_seconds,
        )

        # Build retry statistics from per-transcript data
        retry_statistics = self._build_retry_statistics(
            per_transcript_stats or [], agent_defs
        )

        # Build error summary structure
        error_summary_structured = self._build_error_summary(error_summary or [])

        # Build comprehensive job summary matching README specification
        task_summary = {
            "job_task_id": job_task_id,
            "experiment_name": experiment,
            "benchmark_subcategory": benchmark,
            "start_time": format_timestamp(start_time),
            "end_time": format_timestamp(end_time),
            "duration_seconds": round(duration_seconds, 3),
            "config_snapshot_path": config_snapshot_path or "",
            # Throughput and performance metrics
            "throughput_performance": aggregated_stats.get(
                "throughput_performance", {}
            ),
            # Processing statistics
            "processing_statistics": {
                "questions_total": questions_total,
                "questions_attempted": questions_succeeded + questions_partial + questions_failed,
                "questions_succeeded": questions_succeeded,
                "questions_partial": questions_partial,
                "questions_failed": questions_failed,
                "success_rate": float(f"{questions_succeeded / questions_total:.5f}")
                if questions_total > 0
                else 0.0,
                "error_summary": error_summary_structured,
            },
            # Retry statistics
            "retry_statistics": retry_statistics,
            # Per-transcript statistics for outlier detection
            "per_transcript_statistics": per_transcript_stats or [],
            # Metadata
            "created_at": format_timestamp(datetime.now(timezone.utc)),
        }

        # Add verbose config sections only in debug mode (lighter in production)
        if is_debug_enabled():
            task_summary["conversation_config"] = {
                "routing_strategy": conversation_config["routing_strategy"],
                "max_rounds": conversation_config["max_rounds"],
                "agent_count": len(agent_defs),
            }
            task_summary["retry_config"] = {
                "max_retries": retry_config["max_retries"],
                "answer_match_threshold": retry_config["answer_match_threshold"],
                "retry_on_validation_error": retry_config["retry_on_validation_error"],
                "retry_on_generation_error": retry_config["retry_on_generation_error"],
            }
            task_summary["identity_reveal_config"] = identity_reveal_config
            task_summary["prompt_template_config"] = prompt_template_config
            task_summary["model_definitions"] = self._build_backend_config(
                model_definitions, effective_backend_config
            )
            task_summary["processing_statistics"]["transcript_uuids"] = [
                stat["transcript_id"] for stat in (per_transcript_stats or [])
            ]

        # Save to file: {timestamp}_{job_task_id}.json
        summary_path = (
            exp_root_path
            / benchmark
            / experiment
            / "task_summary"
            / f"{filename_id}.json"
        )

        # Create directory if it doesn't exist
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, "w") as f:
            json.dump(task_summary, f, indent=2)

        info_print(
            f"Task summary saved: {display_path(summary_path, self.project_root)}"
        )
        return str(summary_path)

    def _build_backend_config(
        self,
        model_definitions: Dict[str, Any],
        effective_backend_config: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build backend configuration section for all models.

        Args:
            model_definitions: Model definitions (model_name as key)
            effective_backend_config: Optional dict of effective config per model
                keyed by model_path (includes auto-calculated values like max_num_seqs)

        Returns:
            Backend configuration dictionary with per-model configs
        """
        if not model_definitions:
            return {}

        result: Dict[str, Any] = {}

        for model_id, model_def in model_definitions.items():
            backend = model_def["backend"]

            if backend == "vllm":
                model_path = model_def.get("model_path")
                vllm_config = model_def.get("vllm_config", {})

                # Get max_num_seqs: prefer effective (auto-calculated) value over upper bound
                max_num_seqs_upper_bound = vllm_config.get("max_num_seqs_upper_bound")
                max_num_seqs_effective = max_num_seqs_upper_bound
                if effective_backend_config and model_path in effective_backend_config:
                    max_num_seqs_effective = effective_backend_config[model_path].get(
                        "max_num_seqs", max_num_seqs_upper_bound
                    )

                result[model_id] = {
                    "backend": backend,
                    "model_path": model_path,
                    "tensor_parallel_size": vllm_config.get("tensor_parallel_size"),
                    "gpu_memory_utilization": vllm_config.get("gpu_memory_utilization"),
                    "max_model_len": vllm_config.get("max_model_len"),
                    "max_num_seqs_upper_bound": max_num_seqs_upper_bound,
                    "max_num_seqs_effective": max_num_seqs_effective,
                    "dtype": vllm_config.get("dtype"),
                    "enable_prefix_caching": vllm_config.get(
                        "enable_prefix_caching", False
                    ),
                }
            else:  # ollama
                model_name = model_def.get("model_name")
                result[model_id] = {
                    "backend": backend,
                    "model_name": model_name,
                }

        return result

    def _aggregate_transcript_stats(
        self,
        per_transcript_stats: List[Dict[str, Any]],
        agent_defs: List[Dict[str, Any]],
        total_duration: float,
    ) -> Dict[str, Any]:
        """Aggregate statistics across all transcripts.

        Args:
            per_transcript_stats: List of per-transcript statistics
            agent_defs: Agent definitions for per-agent stats
            total_duration: Total job duration in seconds

        Returns:
            Aggregated statistics dictionary
        """
        if not per_transcript_stats:
            return {"throughput_performance": {}}

        # Aggregate totals
        total_tokens_generated = sum(
            stat.get("tokens_generated", 0) for stat in per_transcript_stats
        )
        total_tokens_prompt = sum(
            stat.get("tokens_prompt", 0) for stat in per_transcript_stats
        )

        # Find max tokens across all transcripts (for context length optimization)
        max_tokens_prompt = max(
            (stat.get("max_tokens_prompt", 0) for stat in per_transcript_stats),
            default=0,
        )
        max_tokens_combined = max(
            (stat.get("max_tokens_combined", 0) for stat in per_transcript_stats),
            default=0,
        )

        # Per-agent statistics
        per_agent_stats = self._calculate_per_agent_stats(
            per_transcript_stats, agent_defs
        )

        # Throughput metrics - only wall-clock based metrics are meaningful for async
        num_conversations = len(per_transcript_stats)
        throughput_performance = {
            "wall_clock_seconds": round(total_duration, 3),
            "questions_per_second": round(
                num_conversations / total_duration if total_duration > 0 else 0, 3
            ),
            "tokens_per_second": round(
                total_tokens_generated / total_duration if total_duration > 0 else 0, 3
            ),
            "total_tokens_generated": total_tokens_generated,
            "total_tokens_prompt": total_tokens_prompt,
            "max_tokens_prompt": max_tokens_prompt,
            "max_tokens_combined": max_tokens_combined,
            "per_agent_stats": per_agent_stats,
        }

        return {"throughput_performance": throughput_performance}

    def _calculate_per_agent_stats(
        self,
        per_transcript_stats: List[Dict[str, Any]],
        agent_defs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Calculate per-agent token statistics.

        Args:
            per_transcript_stats: List of per-transcript statistics
            agent_defs: Agent definitions

        Returns:
            List of per-agent statistics
        """
        # Initialize per-agent accumulators
        agent_tokens_generated = {agent["agent_id"]: 0 for agent in agent_defs}
        agent_messages = {agent["agent_id"]: 0 for agent in agent_defs}

        # Aggregate from per-transcript stats
        for stat in per_transcript_stats:
            agent_stats = stat.get("per_agent", {})
            for agent_id, data in agent_stats.items():
                if agent_id in agent_tokens_generated:
                    agent_tokens_generated[agent_id] += data.get("tokens_generated", 0)
                    agent_messages[agent_id] += data.get("message_count", 0)

        # Build result
        result = []
        for agent in agent_defs:
            agent_id = agent["agent_id"]
            token_count = agent_tokens_generated.get(agent_id, 0)
            message_count = agent_messages.get(agent_id, 0)
            result.append(
                {
                    "agent_id": agent_id,
                    "role": agent["role"],
                    "token_count": token_count,
                    "message_count": message_count,
                    "avg_tokens_per_message": round(
                        token_count / message_count if message_count > 0 else 0, 1
                    ),
                }
            )

        return result

    def _build_retry_statistics(
        self,
        per_transcript_stats: List[Dict[str, Any]],
        agent_defs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build retry statistics from per-transcript data.

        Args:
            per_transcript_stats: List of per-transcript statistics
            agent_defs: Agent definitions

        Returns:
            Retry statistics dictionary
        """
        total_retries = 0
        total_messages = 0
        by_agent = {agent["agent_id"]: 0 for agent in agent_defs}
        agent_messages = {agent["agent_id"]: 0 for agent in agent_defs}
        by_role = Counter()
        role_messages = Counter()
        validation_errors = Counter()

        for stat in per_transcript_stats:
            retries = stat.get("retry_attempts", 0)
            total_retries += retries

            # Per-agent and per-role retries
            agent_stats = stat.get("per_agent", {})
            for agent_id, data in agent_stats.items():
                msg_count = data.get("message_count", 0)
                retry_count = data.get("retry_count", 0)

                total_messages += msg_count

                if agent_id in by_agent:
                    by_agent[agent_id] += retry_count
                    agent_messages[agent_id] += msg_count

                # Find role for this agent
                for agent in agent_defs:
                    if agent["agent_id"] == agent_id:
                        role = agent["role"]
                        by_role[role] += retry_count
                        role_messages[role] += msg_count
                        break

            # Validation error types (use error_code for aggregation)
            for error in stat.get("validation_errors", []):
                error_code = error.get("error_code", "UNKNOWN")
                validation_errors[error_code] += 1

        # Build by_agent list
        num_conversations = len(per_transcript_stats)
        by_agent_list = [
            {
                "agent_id": agent_id,
                "retries": count,
                "messages": agent_messages[agent_id],
            }
            for agent_id, count in by_agent.items()
        ]

        # Build by_role list (aggregate retries by role)
        by_role_list = [
            {
                "role": role,
                "retries": count,
                "messages": role_messages[role],
            }
            for role, count in by_role.items()
        ]

        # Validation errors by type - counts ALL validation errors
        # including those from successful conversations that recovered via retry
        validation_errors_by_type = [
            {"error_code": error_code, "count": count}
            for error_code, count in validation_errors.most_common(10)
        ]

        return {
            "total_retry_attempts": total_retries,
            "average_retries_per_conversation": round(
                total_retries / num_conversations if num_conversations > 0 else 0, 3
            ),
            "by_agent": by_agent_list,
            "by_role": by_role_list,
            "validation_errors_by_type": validation_errors_by_type,
        }

    def _build_error_summary(self, error_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build structured error summary.

        Args:
            error_list: List of error dictionaries

        Returns:
            Structured error summary with both top-level and underlying error breakdowns
        """
        by_type = Counter()
        by_underlying_type = Counter()  # Track underlying validation errors
        error_details = []

        for error_entry in error_list:
            error_info = error_entry.get("error", {})
            if isinstance(error_info, dict):
                error_type = error_info.get("error_code", "unknown")
                error_msg = error_info.get("message", str(error_info))

                # Extract underlying validation errors for MAX_RETRIES_EXCEEDED
                details = error_info.get("details", {})
                validation_errors = details.get("validation_errors", [])
                underlying_types = []
                for val_err in validation_errors:
                    underlying_code = val_err.get("error_code", "unknown")
                    by_underlying_type[underlying_code] += 1
                    underlying_types.append(underlying_code)
            else:
                error_type = "unknown"
                error_msg = str(error_info)
                underlying_types = []

            by_type[error_type] += 1
            error_detail_entry = {
                "question_id": error_entry.get("question_id"),
                "error_type": error_type,
                "error": error_msg,
            }
            # Add underlying error breakdown if available
            if underlying_types:
                error_detail_entry["underlying_errors"] = underlying_types

            error_details.append(error_detail_entry)

        # Build result with fields in order: by_type, by_underlying_type, error_detail
        result = {
            "by_type": dict(by_type),
        }

        # Add underlying error breakdown if there are any (before error_detail)
        if by_underlying_type:
            result["by_underlying_type"] = dict(by_underlying_type)

        result["error_detail"] = error_details

        return result

    def get_experiment_root(self) -> Path:
        """Get the experiment root directory.

        Returns:
            Path to experiment root directory
        """
        exp_root = os.environ.get(MAC_FAIRNESS_EXPERIMENT_ROOT, "experiment")
        exp_root_path = Path(exp_root)
        if not exp_root_path.is_absolute():
            exp_root_path = self.project_root / exp_root
        return exp_root_path

    def save_task_manifest(
        self,
        config: Dict[str, Any],
        questions: List[Dict[str, Any]],
        submission_timestamp: datetime,
        config_snapshot_path: str,
        previous_questions: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[Path, Dict[str, str]]:
        """Save a task manifest recording all planned questions for recovery.

        The manifest uses the same timestamp as the config_snapshot.
        One manifest per task run.

        Questions are stored as a dict indexed by question_id for O(1) lookup.
        Each question entry has:
        - status: "succeeded" or null
        - transcript_id: pre-assigned UUID for deterministic transcript paths

        Args:
            config: Full configuration dictionary
            questions: List of questions to process (already sliced by range)
            submission_timestamp: Same timestamp used for config_snapshot
            config_snapshot_path: Path to the config snapshot (for reference)
            previous_questions: Optional dict of ALL question entries from previous manifest
                (for resume). Contains both succeeded and null questions. Reuses transcript_id
                for all questions to ensure orphan transcripts are overwritten on retry.

        Returns:
            Tuple of (manifest_path, question_to_transcript_id_map)
            question_to_transcript_id_map maps question_id -> transcript_id
        """
        import uuid

        exp_root_path = self.get_experiment_root()
        exp_meta = config["experiment_metadata"]
        benchmark = exp_meta["benchmark_subcategory"]
        experiment = exp_meta["experiment_name"]

        # Use same timestamp format as config_snapshot (millisecond precision)
        timestamp_str = format_filename_timestamp(submission_timestamp)

        manifest_dir = exp_root_path / benchmark / experiment / "task_manifest"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        # Same naming format as task_summary: {timestamp}_{job_task_id}.json
        job_task_id = get_current_job_task_id()
        manifest_path = manifest_dir / f"{timestamp_str}_{job_task_id}.json"

        if previous_questions is None:
            previous_questions = {}

        # Build question entries as dict indexed by question_id
        # Reuse transcript_id from previous manifest if available (for all questions)
        questions_dict: Dict[str, Dict[str, Any]] = {}
        question_to_transcript_id: Dict[str, str] = {}
        num_succeeded = 0

        for q in questions:
            qid = q.get("question_id", f"q_{questions.index(q)}")
            if qid in previous_questions:
                # Reuse entry from previous manifest (preserves transcript_id)
                # This works for both succeeded (carry over) and null (retry with same path)
                questions_dict[qid] = previous_questions[qid]
                question_to_transcript_id[qid] = previous_questions[qid]["transcript_id"]
                if previous_questions[qid].get("status") == "succeeded":
                    num_succeeded += 1
            else:
                # New question with null status and pre-assigned transcript_id
                transcript_id = str(uuid.uuid4())
                questions_dict[qid] = {
                    "status": None,
                    "transcript_id": transcript_id,
                }
                question_to_transcript_id[qid] = transcript_id

        # Build manifest content
        manifest = {
            "job_task_id": job_task_id,
            "config_snapshot_path": config_snapshot_path,
            "num_questions_planned": len(questions),
            "num_questions_processed": num_succeeded,
            "questions": questions_dict,
            "created_at": format_timestamp(datetime.now(timezone.utc)),
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        if num_succeeded > 0:
            info_print(
                f"Task manifest saved: {display_path(manifest_path, self.project_root)} "
                f"({num_succeeded} succeeded carried over)"
            )
        else:
            info_print(
                f"Task manifest saved: {display_path(manifest_path, self.project_root)}"
            )

        return manifest_path, question_to_transcript_id

    def record_question_completion(
        self,
        manifest_path: Path,
        question_id: str,
        succeeded: bool,
        index_path: Path,
        index_entry: Dict[str, Any],
        transcript_path: Path,
        transcript: Dict[str, Any],
    ) -> None:
        """Record question completion: manifest and index atomically, transcript separately.

        Design rationale:
        - Manifest + index are the source of truth, must stay in sync
        - Transcript is large and slow to write, kept outside lock for performance
        - Orphan transcripts (transcript exists but manifest says null) are harmless:
          on resume, question is re-run and transcript is overwritten

        Operation order:
        1. Save transcript (outside lock - fast parallel writes)
        2. Under lock: update manifest + append to index (atomic)

        If interrupted:
        - Before step 1: no transcript, manifest null -> retry
        - After step 1, before step 2: orphan transcript, manifest null -> retry (overwrites)
        - After step 2: all consistent

        Raises:
            ManifestParseError: If manifest cannot be parsed
            ManifestWriteError: If any write operation fails

        Args:
            manifest_path: Absolute path to the task manifest file
            question_id: The question_id to mark
            succeeded: True if the question succeeded, False otherwise
            index_path: Absolute path to the index.jsonl file
            index_entry: The index entry dictionary to append
            transcript_path: Absolute path to save the transcript
            transcript: The transcript dictionary to save
        """
        # Step 1: Save transcript OUTSIDE lock (performance: large file, parallel writes OK)
        # Orphan transcripts are harmless - overwritten on retry
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_temp = transcript_path.with_suffix(".json.tmp")
        try:
            with open(transcript_temp, "w") as f:
                json.dump(transcript, f, indent=2)
            transcript_temp.rename(transcript_path)
        except OSError as e:
            if transcript_temp.exists():
                transcript_temp.unlink(missing_ok=True)
            raise ManifestWriteError(
                f"Failed to write transcript {transcript_path}: {e}"
            ) from e

        # Step 2: Update manifest + index UNDER lock (must stay in sync)
        experiment_name = index_path.stem.replace("_index", "")
        lock_path = self.project_root / "bookkeeping" / f".{experiment_name}.completion.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        if not lock_path.exists():
            lock_path.touch()

        if not index_path.exists():
            index_path.touch()

        with open(lock_path, "r") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                # Update manifest (questions stored as dict indexed by question_id)
                if manifest_path.exists():
                    try:
                        with open(manifest_path, "r") as f:
                            manifest = json.load(f)
                    except json.JSONDecodeError as e:
                        raise ManifestParseError(
                            f"Failed to parse manifest {manifest_path}: {e}"
                        ) from e

                    # Update question status (O(1) lookup with dict)
                    questions = manifest.get("questions", {})
                    if question_id in questions:
                        questions[question_id]["status"] = "succeeded" if succeeded else None

                    manifest["num_questions_processed"] = (
                        manifest.get("num_questions_processed", 0) + 1
                    )

                    temp_path = manifest_path.with_suffix(".json.tmp")
                    try:
                        with open(temp_path, "w") as f:
                            json.dump(manifest, f, indent=2)
                        temp_path.rename(manifest_path)
                    except OSError as e:
                        if temp_path.exists():
                            temp_path.unlink(missing_ok=True)
                        raise ManifestWriteError(
                            f"Failed to write manifest {manifest_path}: {e}"
                        ) from e

                # Append to index
                try:
                    with open(index_path, "a") as f:
                        f.write(json.dumps(index_entry) + "\n")
                except OSError as e:
                    raise ManifestWriteError(
                        f"Failed to append to index {index_path}: {e}"
                    ) from e

            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def check_manifest_complete(self, manifest_path: Path) -> bool:
        """Check if all questions in manifest are succeeded.

        Args:
            manifest_path: Absolute path to the manifest file

        Returns:
            True if all questions have status="succeeded", False otherwise
        """
        if not manifest_path.exists():
            return True  # No manifest = nothing to do

        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            # Questions stored as dict indexed by question_id
            questions = manifest.get("questions", {})
            return all(q.get("status") == "succeeded" for q in questions.values())
        except (json.JSONDecodeError, OSError):
            return False

    def delete_manifest(self, manifest_path: Path) -> bool:
        """Delete a task manifest file.

        Called after all questions succeeded. The caller is responsible for
        checking completion via check_manifest_complete() before calling this.

        Args:
            manifest_path: Absolute path to the manifest file

        Returns:
            True if manifest was deleted, False on error
        """
        try:
            if manifest_path.exists():
                manifest_path.unlink()
                info_print(
                    f"Task manifest deleted (all succeeded): {display_path(manifest_path, self.project_root)}"
                )
                return True
        except OSError:
            pass
        return False

    def cleanup_lock_file(self, experiment_name: str) -> None:
        """Remove the per-experiment lock file after task completion.

        Called when a task finishes processing (regardless of question success/failure).
        Lock files are only needed during active writes; cleaning them up keeps
        the bookkeeping directory tidy.

        Args:
            experiment_name: Name of the experiment (used to derive lock file path)
        """
        lock_path = self.project_root / "bookkeeping" / f".{experiment_name}.completion.lock"
        try:
            if lock_path.exists():
                lock_path.unlink()
        except OSError:
            pass  # Non-fatal: lock file cleanup is best-effort

    def find_all_task_manifests_by_grid_snapshot(
        self,
        grid_config_snapshot_path: str,
    ) -> List[Tuple[Path, List[str], Dict[str, Dict[str, Any]]]]:
        """Find ALL task manifests whose config_snapshot references the grid config snapshot.

        Used for rep run resume: finds all task manifests across all experiment directories
        that were created from runs using this grid config snapshot.

        The config_snapshot for each task is saved in:
            bookkeeping/config_snapshot/{benchmark}/{experiment_name}_{timestamp}.yaml

        This config_snapshot file contains the full expanded config. We can't directly
        match it to the grid config snapshot. Instead, we rely on the grid manifest
        to tell us which experiment_name/benchmark combinations to look for.

        Args:
            grid_config_snapshot_path: Path to the grid config snapshot file

        Returns:
            List of (manifest_path, null_question_ids, all_questions) tuples.
            Each tuple represents a task manifest with incomplete questions.
        """
        # This method is designed to be called after load_all_manifests_for_resume
        # which provides the experiment_name/benchmark pairs to search for.
        # For now, we scan experiment directories and check task manifests.

        exp_root_path = self.get_experiment_root()
        results: List[Tuple[Path, List[str], Dict[str, Dict[str, Any]]]] = []

        # Resolve the grid config snapshot path for comparison
        resolved_grid_snapshot = str(Path(resolve_path(
            grid_config_snapshot_path, self.project_root
        )).resolve())

        # Scan all benchmark directories
        if not exp_root_path.exists():
            return results

        for benchmark_dir in exp_root_path.iterdir():
            if not benchmark_dir.is_dir():
                continue

            # Scan all experiment directories in this benchmark
            for exp_dir in benchmark_dir.iterdir():
                if not exp_dir.is_dir():
                    continue

                manifest_dir = exp_dir / "task_manifest"
                if not manifest_dir.exists():
                    continue

                # Check each task manifest
                for manifest_file in manifest_dir.glob("*.json"):
                    try:
                        with open(manifest_file, "r") as f:
                            manifest = json.load(f)
                    except (json.JSONDecodeError, OSError):
                        continue

                    # Check if this task manifest's config_snapshot references
                    # a config that was derived from the grid config snapshot
                    config_snapshot_path = manifest.get("config_snapshot_path", "")
                    if not config_snapshot_path:
                        continue

                    # Load the config snapshot to check if it contains grid metadata
                    resolved_config_snapshot = resolve_path(
                        config_snapshot_path, self.project_root
                    )
                    if not Path(resolved_config_snapshot).exists():
                        continue

                    try:
                        import yaml
                        with open(resolved_config_snapshot, "r") as f:
                            config = yaml.safe_load(f)

                        # Check if the config has _grid_config_snapshot_path that matches
                        grid_snapshot_in_config = config.get(
                            "experiment_metadata", {}
                        ).get("_grid_config_snapshot_path", "")

                        if grid_snapshot_in_config:
                            resolved_from_config = str(Path(resolve_path(
                                grid_snapshot_in_config, self.project_root
                            )).resolve())
                            if resolved_from_config != resolved_grid_snapshot:
                                continue
                        else:
                            # No grid snapshot reference in config, skip
                            continue

                    except (yaml.YAMLError, OSError):
                        continue

                    # Found a matching task manifest - extract null questions
                    questions = manifest.get("questions", {})
                    null_question_ids = [
                        qid for qid, q_entry in questions.items()
                        if q_entry.get("status") != "succeeded"
                    ]

                    if null_question_ids:
                        results.append((manifest_file, null_question_ids, questions))

        return results

    def find_task_manifest_and_get_null_questions(
        self,
        experiment_name: str,
        benchmark_subcategory: str,
        expected_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[Path, List[str], str, Dict[str, Dict[str, Any]]]]:
        """Find a task manifest for an experiment and get null question IDs.

        Used for resuming interrupted runs: finds questions that didn't succeed
        so they can be re-run while skipping succeeded ones.

        Args:
            experiment_name: Name of the experiment
            benchmark_subcategory: Benchmark category
            expected_config: Optional expected configuration to validate against.
                When provided, validates each candidate manifest's config_snapshot
                against this expected config to ensure we resume with the correct
                configuration (important for grid runs where multiple configurations
                may share the same experiment_name but have different parameters).

        Returns:
            Tuple of (manifest_path, null_question_ids, config_snapshot_path, all_questions)
            or None if no manifest found.
            null_question_ids are questions with status != "succeeded".
            config_snapshot_path is the path to the config snapshot used for this job.
            all_questions is a dict mapping question_id -> question entry for ALL questions
            (both succeeded and null). This preserves transcript_id for null questions so
            retries overwrite orphan transcripts instead of creating new files.
        """
        exp_root_path = self.get_experiment_root()
        manifest_dir = exp_root_path / benchmark_subcategory / experiment_name / "task_manifest"

        if not manifest_dir.exists():
            return None

        # Clean up any stale .json.tmp files from interrupted writes
        for tmp_file in manifest_dir.glob("*.json.tmp"):
            try:
                tmp_file.unlink()
            except OSError:
                pass  # Best-effort cleanup

        # Find all manifests for this experiment (*.json excludes .json.tmp)
        manifests = list(manifest_dir.glob("*.json"))
        if not manifests:
            return None

        # Sort by modification time (most recent first) for deterministic fallback
        manifests = sorted(manifests, key=lambda p: p.stat().st_mtime, reverse=True)

        # Try each manifest, validating config_snapshot if expected_config is provided
        for manifest_path in manifests:
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            config_snapshot_path = manifest.get("config_snapshot_path", "")

            # Validate config_snapshot matches expected_config if provided
            if expected_config is not None and config_snapshot_path:
                if not self._validate_config_snapshot_match(
                    config_snapshot_path, expected_config
                ):
                    continue  # Try next manifest

            # Found a valid manifest - extract null question IDs and all question entries
            # Questions are stored as dict indexed by question_id
            # Return ALL entries (both succeeded and null) to preserve transcript_id
            # for null questions, so retries overwrite orphan transcripts
            null_question_ids: List[str] = []
            questions = manifest.get("questions", {})
            for qid, q_entry in questions.items():
                if q_entry.get("status") != "succeeded":
                    null_question_ids.append(qid)

            # Return entire questions dict to preserve transcript_id for all questions
            return manifest_path, null_question_ids, config_snapshot_path, questions

        # No matching manifest found
        return None

    def _validate_config_snapshot_match(
        self,
        config_snapshot_path: str,
        expected_config: Dict[str, Any],
    ) -> bool:
        """Validate that a config snapshot exactly matches the expected configuration.

        Performs exact comparison of the entire configuration to ensure the snapshot
        is for the exact same experiment configuration (important for grid runs where
        multiple configurations may share the same experiment_name).

        Args:
            config_snapshot_path: Path to the config snapshot file
            expected_config: Expected configuration to compare against

        Returns:
            True if configurations match exactly, False otherwise
        """
        import yaml

        # Resolve env var placeholders to absolute path
        resolved_path = resolve_path(config_snapshot_path, self.project_root)
        snapshot_path = Path(resolved_path)
        if not snapshot_path.exists():
            return False

        try:
            with open(snapshot_path, "r") as f:
                snapshot_config = yaml.safe_load(f)
        except (yaml.YAMLError, OSError):
            return False

        # Exact match: same number of keys and same values for all keys
        return snapshot_config == expected_config


class StreamingJobSummary:
    """Manages a streaming job summary that updates in real-time.

    Keeps the file handle open for efficient incremental updates.
    Writes completed transcript stats as they finish, enabling recovery
    even if the process is killed.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        questions_total: int,
        start_time: datetime,
        config_snapshot_path: str,
        project_root: Path,
    ):
        """Initialize streaming job summary.

        Args:
            config: Full configuration dictionary
            questions_total: Total number of questions to process
            start_time: When the job started
            config_snapshot_path: Path to config snapshot
            project_root: Project root directory
        """
        self.config = config
        self.questions_total = questions_total
        self.start_time = start_time
        self.config_snapshot_path = config_snapshot_path
        self.project_root = project_root

        # Counters
        self.questions_succeeded = 0
        self.questions_partial = 0
        self.questions_failed = 0
        self.per_transcript_stats: List[Dict[str, Any]] = []
        self.error_summary: List[Dict[str, Any]] = []

        # File handle
        self._file_handle: Optional[TextIO] = None
        self._summary_path: Optional[Path] = None

        # Thread lock for atomic file writes (progress_callback runs in thread pool)
        import threading
        self._write_lock = threading.Lock()

        self._initialize_file()

    def _initialize_file(self) -> None:
        """Create and initialize the summary file."""
        exp_root = os.environ.get(MAC_FAIRNESS_EXPERIMENT_ROOT, "experiment")
        exp_root_path = Path(exp_root)
        if not exp_root_path.is_absolute():
            exp_root_path = self.project_root / exp_root

        exp_meta = self.config["experiment_metadata"]
        benchmark = exp_meta["benchmark_subcategory"]
        experiment = exp_meta["experiment_name"]

        timestamp_str = format_filename_timestamp(self.start_time)
        job_task_id = get_current_job_task_id()
        filename_id = f"{timestamp_str}_{job_task_id}"

        self._summary_path = (
            exp_root_path / benchmark / experiment / "task_summary" / f"{filename_id}.json"
        )
        self._summary_path.parent.mkdir(parents=True, exist_ok=True)

        # Write initial structure
        self._write_current_state()

        info_print(
            f"Streaming job summary: {display_path(self._summary_path, self.project_root)}"
        )

    def _write_current_state(self) -> None:
        """Write current state to file (atomic write via temp file).

        Thread-safe: uses lock to prevent race conditions when multiple
        progress_callback threads call this concurrently.
        """
        if self._summary_path is None:
            return

        with self._write_lock:
            job_task_id = get_current_job_task_id()
            exp_meta = self.config["experiment_metadata"]

            # Calculate duration so far
            current_time = datetime.now(timezone.utc)
            duration_seconds = (current_time - self.start_time).total_seconds()

            summary = {
                "job_task_id": job_task_id,
                "experiment_name": exp_meta["experiment_name"],
                "benchmark_subcategory": exp_meta["benchmark_subcategory"],
                "start_time": format_timestamp(self.start_time),
                "last_update": format_timestamp(current_time),
                "duration_seconds": round(duration_seconds, 3),
                "config_snapshot_path": self.config_snapshot_path,
                "status": "in_progress",
                "processing_statistics": {
                    "questions_total": self.questions_total,
                    "questions_attempted": self.questions_succeeded + self.questions_partial + self.questions_failed,
                    "questions_succeeded": self.questions_succeeded,
                    "questions_partial": self.questions_partial,
                    "questions_failed": self.questions_failed,
                },
                "completed_transcripts": [
                    {
                        "transcript_id": stat["transcript_id"],
                        "question_id": stat["question_id"],
                        "status": stat["status"],
                    }
                    for stat in self.per_transcript_stats
                ],
                "errors": self.error_summary,
            }

            # Atomic write: write to temp file, then rename
            temp_path = self._summary_path.with_suffix(".json.tmp")
            with open(temp_path, "w") as f:
                json.dump(summary, f, indent=2)
            temp_path.rename(self._summary_path)

    def record_completion(
        self,
        transcript_stat: Dict[str, Any],
        error_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a completed transcript and update the file.

        Args:
            transcript_stat: Statistics from the completed transcript
            error_info: Optional error information for failed/partial
        """
        self.per_transcript_stats.append(transcript_stat)

        status = transcript_stat.get("status", "failed")
        if status == "succeeded":
            self.questions_succeeded += 1
        elif status == "partial":
            self.questions_partial += 1
            if error_info:
                self.error_summary.append({
                    "question_id": transcript_stat.get("question_id"),
                    "error": error_info,
                })
        else:
            self.questions_failed += 1
            if error_info:
                self.error_summary.append({
                    "question_id": transcript_stat.get("question_id"),
                    "error": error_info,
                })

        # Write updated state
        self._write_current_state()

    def finalize(
        self,
        end_time: datetime,
        effective_backend_config: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        """Finalize the job summary with complete statistics.

        Args:
            end_time: When the job ended
            effective_backend_config: Optional backend config info

        Returns:
            Path to the saved summary file
        """
        if self._summary_path is None:
            return ""

        # Use the existing save_task_summary for final comprehensive output
        manager = BookkeepingManager(self.project_root)
        return manager.save_task_summary(
            config=self.config,
            questions_total=self.questions_total,
            questions_succeeded=self.questions_succeeded,
            questions_partial=self.questions_partial,
            questions_failed=self.questions_failed,
            start_time=self.start_time,
            end_time=end_time,
            error_summary=self.error_summary if self.error_summary else None,
            per_transcript_stats=self.per_transcript_stats if self.per_transcript_stats else None,
            config_snapshot_path=self.config_snapshot_path,
            effective_backend_config=effective_backend_config,
        )

    def get_path(self) -> Optional[Path]:
        """Get the path to the summary file."""
        return self._summary_path


class GridManifestManager:
    """Manages grid manifest for tracking multi-configuration grid runs.

    A grid manifest tracks the progress of a grid configuration run,
    recording which configurations have been started, completed, or failed.
    This enables resuming interrupted grid runs.
    """

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the grid manifest manager.

        Args:
            project_root: Project root directory (from MAC_FAIRNESS_WORKSPACE if None)
        """
        if project_root is None:
            workspace = os.environ.get(MAC_FAIRNESS_WORKSPACE)
            if workspace:
                self.project_root = Path(workspace)
            else:
                raise ProjectRootError()
        else:
            self.project_root = project_root

    def _get_manifest_dir(self) -> Path:
        """Get directory for grid manifests."""
        return self.project_root / "bookkeeping" / "grid_manifest"

    def _save_grid_config_snapshot(
        self,
        grid_config_path: str,
        submission_timestamp: datetime,
    ) -> str:
        """Save a snapshot of the grid config file.

        Args:
            grid_config_path: Path to the original grid config file
            submission_timestamp: When the grid run started

        Returns:
            Path to the saved snapshot (with $MAC_FAIRNESS_WORKSPACE prefix)
        """
        snapshot_dir = self.project_root / "bookkeeping" / "_grid_config_snapshot"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Use same naming convention as config_snapshot: {config_name}_{timestamp}.yaml
        timestamp_str = format_filename_timestamp(submission_timestamp)
        original_name = Path(grid_config_path).stem
        snapshot_filename = f"{original_name}_{timestamp_str}.yaml"
        snapshot_path = snapshot_dir / snapshot_filename

        # Copy the grid config file
        import shutil
        shutil.copy2(grid_config_path, snapshot_path)

        # Return path with $MAC_FAIRNESS_WORKSPACE prefix for portability
        return f"$MAC_FAIRNESS_WORKSPACE/bookkeeping/_grid_config_snapshot/{snapshot_filename}"

    def _get_model_signature(self, config: Dict[str, Any]) -> Optional[str]:
        """Extract a signature representing the GPU resources used by a config.

        Used to determine if consecutive tasks can reuse the same loaded model.
        Returns a string that uniquely identifies the model configuration, or None
        if no vLLM models are used.

        Args:
            config: Full experiment configuration

        Returns:
            Signature string (sorted model paths joined), or None if no vLLM backend
        """
        model_defs = config.get("model_definitions", {})
        vllm_paths = []
        for model_def in model_defs.values():
            if model_def.get("backend") == "vllm":
                model_path = model_def.get("model_path")
                if model_path:
                    vllm_paths.append(model_path)
        if not vllm_paths:
            return None
        # Sort for consistent comparison
        return "|".join(sorted(vllm_paths))

    def save_grid_manifest(
        self,
        grid_config_path: str,
        expanded_configs: list,
        submission_timestamp: datetime,
        is_resume: bool = False,
        succeeded_task_info: Optional[Dict[int, Dict[str, Any]]] = None,
        existing_grid_snapshot_path: Optional[str] = None,
    ) -> Tuple[Path, str]:
        """Save a grid manifest at the start of a grid run.

        Args:
            grid_config_path: Path to the grid config file (or snapshot on resume)
            expanded_configs: List of (config, grid_sweep_specs) tuples from GridConfigExpander
            submission_timestamp: When the grid run started
            is_resume: If True, grid_config_path is already a snapshot; reuse it
            succeeded_task_info: Dict mapping task_id to the full run entry dict for runs
                that were already processed (for resume). The run entries are used directly.
            existing_grid_snapshot_path: Optional path to an existing grid config snapshot.
                When provided (e.g., for rep runs), this snapshot is reused instead of
                creating a new one. This ensures all repetitions share the same snapshot.

        Returns:
            Tuple of (manifest_path, _grid_config_snapshot_path)
        """
        manifest_dir = self._get_manifest_dir()
        manifest_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = format_filename_timestamp(submission_timestamp)
        pid = os.getpid()
        manifest_path = manifest_dir / f"{timestamp_str}_{pid}.json"

        # Save or reuse grid config snapshot
        if existing_grid_snapshot_path is not None:
            # Reuse existing snapshot (for rep runs - all repetitions share one snapshot)
            _grid_config_snapshot_path = existing_grid_snapshot_path
        elif is_resume:
            # On resume, grid_config_path is already the snapshot path
            snapshot_filename = Path(grid_config_path).name
            _grid_config_snapshot_path = f"$MAC_FAIRNESS_WORKSPACE/bookkeeping/_grid_config_snapshot/{snapshot_filename}"
        else:
            _grid_config_snapshot_path = self._save_grid_config_snapshot(
                grid_config_path, submission_timestamp
            )

        if succeeded_task_info is None:
            succeeded_task_info = {}

        # Pre-compute model signatures for skip_cleanup optimization
        # skip_cleanup=True means the next task uses the same GPU model, so don't unload
        model_signatures = [self._get_model_signature(config) for config, _ in expanded_configs]

        # Build experiment run entries
        tasks = []
        for i, (config, grid_sweep_specs) in enumerate(expanded_configs):
            if i in succeeded_task_info:
                # Reuse the full run entry from the old manifest (preserves all info)
                tasks.append(succeeded_task_info[i])
            else:
                # Determine if we can skip GPU cleanup for this task
                # Skip cleanup if next task uses the same model (avoids unload/reload cycle)
                current_sig = model_signatures[i]
                next_sig = model_signatures[i + 1] if i + 1 < len(model_signatures) else None
                skip_cleanup = (
                    current_sig is not None
                    and next_sig is not None
                    and current_sig == next_sig
                )

                # New run entry
                exp_meta = config.get("experiment_metadata", {})
                tasks.append({
                    "task_id": i,
                    "experiment_name": exp_meta.get("experiment_name", "unknown"),
                    "benchmark_subcategory": exp_meta.get("benchmark_subcategory", "unknown"),
                    "grid_sweep_specs": grid_sweep_specs,
                    "skip_cleanup": skip_cleanup,
                    "status": None,
                    "started_at": None,
                    "completed_at": None,
                })

        manifest = {
            "grid_config_snapshot_path": _grid_config_snapshot_path,
            "pid": pid,
            "submission_timestamp": format_timestamp(submission_timestamp),
            "num_tasks_planned": len(tasks),
            "num_tasks_succeeded": len(succeeded_task_info),
            "tasks": tasks,
            "created_at": format_timestamp(datetime.now(timezone.utc)),
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        info_print(f"Grid manifest saved: {display_path(manifest_path, self.project_root)}")
        return manifest_path, _grid_config_snapshot_path

    def mark_task_started(self, manifest_path: Path, task_id: int) -> None:
        """Mark a task as started.

        Status values:
        - null: not yet run, or stopped abruptly (needs re-run)
        - "started": task started, may be interrupted or some questions failed (needs re-run)
        - "succeeded": all questions in the task succeeded (don't re-run)

        Args:
            manifest_path: Path to the manifest file
            task_id: ID of the task being started
        """
        if not manifest_path.exists():
            return

        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except json.JSONDecodeError as e:
            info_print(f"Warning: Could not parse grid manifest: {e}")
            return
        except OSError as e:
            info_print(f"Warning: Could not read grid manifest: {e}")
            return

        tasks = manifest.get("tasks", [])
        if 0 <= task_id < len(tasks):
            tasks[task_id]["status"] = "started"
            tasks[task_id]["started_at"] = format_timestamp(
                datetime.now(timezone.utc)
            )

        # Atomic write: write to temp file, then rename
        # This ensures manifest is never left in a corrupted state on Ctrl+C
        temp_path = manifest_path.with_suffix(".json.tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(manifest, f, indent=2)
            temp_path.rename(manifest_path)
        except OSError as e:
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            info_print(f"Warning: Could not write grid manifest: {e}")

    def mark_task_succeeded(self, manifest_path: Path, task_id: int) -> None:
        """Mark a task as succeeded (all questions completed successfully).

        Status values:
        - null: not yet run, or stopped abruptly (needs re-run)
        - "started": task started, may be interrupted or some questions failed (needs re-run)
        - "succeeded": all questions in the task succeeded (don't re-run)

        Args:
            manifest_path: Path to the manifest file
            task_id: ID of the task that succeeded
        """
        if not manifest_path.exists():
            return

        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except json.JSONDecodeError as e:
            info_print(f"Warning: Could not parse grid manifest: {e}")
            return
        except OSError as e:
            info_print(f"Warning: Could not read grid manifest: {e}")
            return

        tasks = manifest.get("tasks", [])
        if 0 <= task_id < len(tasks):
            tasks[task_id]["status"] = "succeeded"
            tasks[task_id]["completed_at"] = format_timestamp(
                datetime.now(timezone.utc)
            )

        # Update succeeded count
        manifest["num_tasks_succeeded"] = manifest.get("num_tasks_succeeded", 0) + 1

        # Atomic write: write to temp file, then rename
        # This ensures manifest is never left in a corrupted state on Ctrl+C
        temp_path = manifest_path.with_suffix(".json.tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(manifest, f, indent=2)
            temp_path.rename(manifest_path)
        except OSError as e:
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            info_print(f"Warning: Could not write grid manifest: {e}")

    def get_pending_indices(self, manifest_path: Path) -> list:
        """Get indices of tasks that haven't succeeded.

        Returns indices where status is null or "started" (not "succeeded").

        Args:
            manifest_path: Path to the manifest file

        Returns:
            List of task indices that need to be run
        """
        if not manifest_path.exists():
            return []

        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except json.JSONDecodeError as e:
            info_print(f"Warning: Could not parse grid manifest: {e}")
            return []
        except OSError as e:
            info_print(f"Warning: Could not read grid manifest: {e}")
            return []

        pending = []
        for task in manifest.get("tasks", []):
            if task.get("status") != "succeeded":
                pending.append(task["task_id"])
        return pending

    def is_complete(self, manifest_path: Path) -> bool:
        """Check if all tasks have succeeded.

        Args:
            manifest_path: Path to the manifest file

        Returns:
            True if all tasks have status="succeeded"
        """
        if not manifest_path.exists():
            return True

        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except json.JSONDecodeError as e:
            info_print(f"Warning: Could not parse grid manifest: {e}")
            return False
        except OSError as e:
            info_print(f"Warning: Could not read grid manifest: {e}")
            return False

        tasks = manifest.get("tasks", [])
        return all(t.get("status") == "succeeded" for t in tasks)

    def delete_if_complete(
        self, manifest_path: Path, delete_snapshot: bool = True
    ) -> bool:
        """Delete manifest and optionally grid config snapshot if all tasks succeeded.

        Args:
            manifest_path: Path to the manifest file
            delete_snapshot: Whether to also delete the grid config snapshot.

        Returns:
            True if manifest was deleted
        """
        if self.is_complete(manifest_path):
            # Read manifest to get grid config snapshot path before deleting
            _grid_config_snapshot_path = None
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                    _grid_config_snapshot_path = manifest.get("grid_config_snapshot_path")
            except (json.JSONDecodeError, OSError):
                pass

            try:
                manifest_path.unlink(missing_ok=True)
                info_print(
                    f"Grid manifest deleted (all succeeded): "
                    f"{display_path(manifest_path, self.project_root)}"
                )

                # Delete the grid config snapshot only if requested
                if delete_snapshot and _grid_config_snapshot_path:
                    resolved_path = resolve_path(_grid_config_snapshot_path, self.project_root)
                    snapshot_path = Path(resolved_path)
                    if snapshot_path.exists():
                        snapshot_path.unlink(missing_ok=True)
                        info_print(
                            f"Grid config snapshot deleted: "
                            f"{display_path(snapshot_path, self.project_root)}"
                        )

                return True
            except OSError as e:
                info_print(f"Warning: Could not delete grid manifest: {e}")
        return False

    def _find_all_manifests(
        self, grid_config_path: str
    ) -> List[Tuple[Path, dict]]:
        """Find ALL manifests matching a grid config snapshot path.

        Used for rep runs where multiple grid manifests share the same snapshot.

        Args:
            grid_config_path: Path to the grid config file (or snapshot)

        Returns:
            List of (manifest_path, manifest_dict) tuples, sorted by modification time
            (most recent first). Empty list if none found.
        """
        manifest_dir = self._get_manifest_dir()
        if not manifest_dir.exists():
            return []

        # Clean up any stale .json.tmp files from interrupted writes
        for tmp_file in manifest_dir.glob("*.json.tmp"):
            try:
                tmp_file.unlink()
            except OSError:
                pass  # Best-effort cleanup

        # Find all manifests that match this grid config (by snapshot path)
        matching_manifests = []
        # Resolve input path to absolute for comparison
        config_path_resolved = str(Path(grid_config_path).resolve())
        for manifest_file in manifest_dir.glob("*.json"):
            try:
                with open(manifest_file, "r") as f:
                    manifest = json.load(f)
                # Match by snapshot path (resume requires using snapshot path)
                snapshot_path = manifest.get("grid_config_snapshot_path", "")
                # Resolve env var placeholders and normalize to absolute path for comparison
                resolved_snapshot = str(Path(resolve_path(snapshot_path, self.project_root)).resolve())
                if config_path_resolved == resolved_snapshot:
                    matching_manifests.append((manifest_file, manifest))
            except json.JSONDecodeError as e:
                info_print(f"Warning: Could not parse manifest {manifest_file.name}: {e}")
                continue
            except OSError as e:
                info_print(f"Warning: Could not read manifest {manifest_file.name}: {e}")
                continue

        # Sort by modification time (most recent first)
        return sorted(matching_manifests, key=lambda x: x[0].stat().st_mtime, reverse=True)

    def _find_manifest(
        self, grid_config_path: str
    ) -> Optional[Tuple[Path, dict]]:
        """Find the most recent manifest for a grid config.

        Args:
            grid_config_path: Path to the grid config file (or snapshot)

        Returns:
            Tuple of (manifest_path, manifest_dict) or None if not found.
        """
        all_manifests = self._find_all_manifests(grid_config_path)
        if not all_manifests:
            return None
        # Return most recent (first in sorted list)
        return all_manifests[0]

    def _extract_pending_from_manifest(
        self, manifest: dict
    ) -> Tuple[List[int], Dict[int, Dict[str, str]], Dict[int, Dict[str, Any]]]:
        """Extract pending task IDs, started task info, and succeeded task info from a manifest.

        Args:
            manifest: The manifest dictionary

        Returns:
            Tuple of (pending_indices, started_task_info, succeeded_task_info).
            started_task_info maps task_id to {"experiment_name", "benchmark_subcategory"}
            for tasks with status="started" (interrupted tasks that may have partial progress).
            succeeded_task_info maps task_id to the full task entry dict
            for tasks with status="succeeded" (to preserve all info on resume).
        """
        pending: List[int] = []
        started_task_info: Dict[int, Dict[str, Any]] = {}
        succeeded_task_info: Dict[int, Dict[str, Any]] = {}
        for task in manifest.get("tasks", []):
            status = task.get("status")
            task_id = task["task_id"]
            if status == "succeeded":
                # Preserve the full task entry for succeeded tasks
                succeeded_task_info[task_id] = task
            else:
                pending.append(task_id)
                # For "started" tasks, save info needed to find task manifest
                if status == "started":
                    started_task_info[task_id] = {
                        "experiment_name": task.get("experiment_name", ""),
                        "benchmark_subcategory": task.get("benchmark_subcategory", ""),
                        "skip_cleanup": task.get("skip_cleanup", False),
                    }
        return pending, started_task_info, succeeded_task_info

    def get_task_skip_cleanup(self, manifest_path: Path, task_id: int) -> bool:
        """Get the skip_cleanup flag for a task from the manifest.

        Args:
            manifest_path: Path to the grid manifest file
            task_id: ID of the task

        Returns:
            True if cleanup should be skipped (next task uses same model), False otherwise
        """
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            tasks = manifest.get("tasks", [])
            if 0 <= task_id < len(tasks):
                return tasks[task_id].get("skip_cleanup", False)
        except (json.JSONDecodeError, OSError):
            pass
        return False

    def _has_pending_tasks(self, manifest_path: Path) -> bool:
        """Check if a manifest has any pending (non-succeeded) tasks.

        Args:
            manifest_path: Path to the grid manifest file

        Returns:
            True if manifest has pending tasks, False otherwise
        """
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            pending, _, _ = self._extract_pending_from_manifest(manifest)
            return len(pending) > 0
        except (json.JSONDecodeError, OSError):
            return False

    def load_all_manifests_for_resume(
        self, grid_config_path: str
    ) -> Optional[Tuple[List[Dict[str, Any]], List[Path]]]:
        """Load ALL manifests for resume (for rep runs with multiple manifests).

        For rep runs, multiple grid manifests share the same grid config snapshot.
        This method finds all of them and extracts started task info from each,
        so we can find task manifests with null questions across all repetitions.

        Args:
            grid_config_path: Path to the grid config snapshot

        Returns:
            Tuple of (all_started_tasks, old_manifest_paths) or None if no manifests found.
            - all_started_tasks: List of dicts with keys:
                - experiment_name: The experiment name (includes rep timestamp)
                - benchmark_subcategory: The benchmark subcategory
                - manifest_path: Path to the grid manifest this task came from
            - old_manifest_paths: List of manifest paths to delete after new manifest created
        """
        all_manifests = self._find_all_manifests(grid_config_path)
        if not all_manifests:
            return None

        all_started_tasks: List[Dict[str, Any]] = []
        old_manifest_paths: List[Path] = []

        for manifest_path, manifest in all_manifests:
            old_manifest_paths.append(manifest_path)
            # Extract started tasks from this manifest
            for task in manifest.get("tasks", []):
                status = task.get("status")
                # Include both "started" and null status tasks
                # (null means task was never started, started means it may have partial progress)
                if status != "succeeded":
                    all_started_tasks.append({
                        "experiment_name": task.get("experiment_name", ""),
                        "benchmark_subcategory": task.get("benchmark_subcategory", ""),
                        "manifest_path": manifest_path,
                    })

        if not all_started_tasks:
            return None

        info_print(f"Found {len(all_manifests)} grid manifest(s) with {len(all_started_tasks)} incomplete task(s)")
        return all_started_tasks, old_manifest_paths

    def load_manifest_for_resume(
        self, grid_config_path: str
    ) -> Optional[Tuple[List[int], Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]], Path]]:
        """Load manifest for resume WITHOUT deleting it.

        Returns the manifest path so the caller can delete it AFTER creating
        the new manifest. This ensures atomic create-then-delete ordering.

        Args:
            grid_config_path: Path to the grid config file

        Returns:
            Tuple of (pending_indices, started_task_info, succeeded_task_info, old_manifest_path)
            or None if no manifest found.
            - pending_indices: Task IDs that need to be run (status != "succeeded")
            - started_task_info: Maps task_id to {"experiment_name", "benchmark_subcategory"}
              for tasks with status="started" (may have partial progress)
            - succeeded_task_info: Maps task_id to full task entry dict for tasks
              with status="succeeded" (to preserve all info on resume)
            - old_manifest_path: Path to delete AFTER new manifest is created
        """
        result = self._find_manifest(grid_config_path)
        if result is None:
            return None

        manifest_path, manifest = result
        pending, started_task_info, succeeded_task_info = self._extract_pending_from_manifest(manifest)

        return pending, started_task_info, succeeded_task_info, manifest_path
