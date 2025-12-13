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
    EXPERIMENT_ROOT_ENV,
    display_path,
    format_filename_timestamp,
    format_timestamp,
    info_print,
    is_debug_enabled,
)


def get_array_job_id() -> str:
    """Get unique job ID for SLURM array jobs.

    For array jobs, uses SLURM_ARRAY_JOB_ID (shared across all tasks).
    For non-array jobs, uses SLURM_JOB_ID.
    For local runs, returns "local".

    Returns:
        Unique job identifier for the array job (not per-task)
    """
    # SLURM_ARRAY_JOB_ID is the parent job ID shared by all array tasks
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    if array_job_id:
        return array_job_id

    # Non-array SLURM job
    slurm_job = os.environ.get("SLURM_JOB_ID")
    if slurm_job:
        return slurm_job

    return "local"


def get_job_task_id() -> str:
    """Get job-task identifier for per-task tracking.

    For array jobs: "{SLURM_JOB_ID}_{SLURM_ARRAY_TASK_ID}"
    For non-array jobs: "{SLURM_JOB_ID}"
    For local runs: "local"

    Returns:
        Job-task identifier string
    """
    slurm_job = os.environ.get("SLURM_JOB_ID")
    slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_job and slurm_task:
        return f"{slurm_job}_{slurm_task}"
    elif slurm_job:
        return slurm_job
    return "local"


class BookkeepingManager:
    """Manages bookkeeping operations including directories, job summaries, and other records."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the bookkeeping manager.

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

    def ensure_directories(self, config: Dict[str, Any]) -> Dict[str, Path]:
        """Create necessary directories for bookkeeping and experiment outputs.

        Args:
            config: Full configuration dictionary

        Returns:
            Dictionary of created directory paths
        """
        exp_root = os.environ.get(EXPERIMENT_ROOT_ENV, "experiment")
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

        job_summary_dir = experiment_dir / "job_summary"
        job_summary_dir.mkdir(parents=True, exist_ok=True)

        info_print(
            f"Directories ensured for: {display_path(experiment_dir, self.project_root)}"
        )

        return {
            "bookkeeping": bookkeeping_dir,
            "config_snapshot": config_snapshot_dir,
            "experiment": experiment_dir,
            "transcript": transcript_dir,
            "job_summary": job_summary_dir,
        }

    def save_job_summary(
        self,
        config: Dict[str, Any],
        questions_total: int,
        questions_succeeded: int,
        questions_partial: int,
        questions_failed: int,
        start_time: datetime,
        end_time: datetime,
        question_range: Optional[tuple] = None,
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
            question_range: Optional question range processed
            error_summary: Optional list of error information
            per_transcript_stats: Optional list of per-transcript statistics
            effective_backend_config: Optional dict of effective backend config per model
                keyed by model_path (includes auto-calculated values like max_num_seqs)

        Returns:
            Path to the saved job summary file
        """
        exp_root = os.environ.get(EXPERIMENT_ROOT_ENV, "experiment")
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

        # Build job ID: "local", "10000", or "10001_2"
        slurm_job = os.environ.get("SLURM_JOB_ID")
        slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_job and slurm_task:
            job_task_id = f"{slurm_job}_{slurm_task}"
        elif slurm_job:
            job_task_id = slurm_job
        else:
            job_task_id = "local"

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
        job_summary = {
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
                "questions_attempted": questions_total,
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
            job_summary["conversation_config"] = {
                "routing_strategy": conversation_config["routing_strategy"],
                "max_rounds": conversation_config["max_rounds"],
                "agent_count": len(agent_defs),
            }
            job_summary["retry_config"] = {
                "max_retries": retry_config["max_retries"],
                "answer_match_threshold": retry_config["answer_match_threshold"],
                "retry_on_validation_error": retry_config["retry_on_validation_error"],
                "retry_on_generation_error": retry_config["retry_on_generation_error"],
            }
            job_summary["identity_reveal_config"] = identity_reveal_config
            job_summary["prompt_template_config"] = prompt_template_config
            job_summary["model_definitions"] = self._build_backend_config(
                model_definitions, effective_backend_config
            )
            job_summary["processing_statistics"]["transcript_uuids"] = [
                stat["transcript_id"] for stat in (per_transcript_stats or [])
            ]

        # Save to file: {timestamp}_{job_task_id}.json
        summary_path = (
            exp_root_path
            / benchmark
            / experiment
            / "job_summary"
            / f"{filename_id}.json"
        )

        # Create directory if it doesn't exist
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, "w") as f:
            json.dump(job_summary, f, indent=2)

        info_print(
            f"Job summary saved: {display_path(summary_path, self.project_root)}"
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
        exp_root = os.environ.get(EXPERIMENT_ROOT_ENV, "experiment")
        exp_root_path = Path(exp_root)
        if not exp_root_path.is_absolute():
            exp_root_path = self.project_root / exp_root
        return exp_root_path

    def save_job_manifest(
        self,
        config: Dict[str, Any],
        questions: List[Dict[str, Any]],
        submission_timestamp: datetime,
        config_snapshot_path: str,
    ) -> Path:
        """Save a job manifest recording all planned questions for recovery.

        The manifest uses the same timestamp as the config_snapshot.
        One manifest per job run (SLURM or local).

        Each question has a status field:
        - "succeeded": completed successfully
        - null: not yet succeeded (not started, failed, partial, or interrupted)

        Args:
            config: Full configuration dictionary
            questions: List of questions to process (already sliced by range)
            submission_timestamp: Same timestamp used for config_snapshot
            config_snapshot_path: Path to the config snapshot (for reference)

        Returns:
            Absolute path to the saved manifest file
        """
        exp_root_path = self.get_experiment_root()
        exp_meta = config["experiment_metadata"]
        benchmark = exp_meta["benchmark_subcategory"]
        experiment = exp_meta["experiment_name"]

        # Use same timestamp format as config_snapshot (millisecond precision)
        timestamp_str = format_filename_timestamp(submission_timestamp)

        manifest_dir = exp_root_path / benchmark / experiment / "job_manifest"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        # Same naming format as job_summary: {timestamp}_{job_task_id}.json
        job_task_id = get_job_task_id()
        manifest_path = manifest_dir / f"{timestamp_str}_{job_task_id}.json"

        # Build manifest content with per-question status (all start as null)
        manifest = {
            "job_task_id": get_job_task_id(),
            "experiment_name": experiment,
            "benchmark_subcategory": benchmark,
            "submission_timestamp": format_timestamp(submission_timestamp),
            "config_snapshot_path": config_snapshot_path,
            "num_questions_planned": len(questions),
            "num_questions_processed": 0,
            "questions": [
                {
                    "index": i,
                    "question_id": q.get("question_id", f"q_{i}"),
                    "status": None,  # null = not yet succeeded
                }
                for i, q in enumerate(questions)
            ],
            "created_at": format_timestamp(datetime.now(timezone.utc)),
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        info_print(
            f"Job manifest saved: {display_path(manifest_path, self.project_root)}"
        )

        return manifest_path

    def mark_question_processed(
        self, manifest_path: Path, question_id: str, succeeded: bool
    ) -> None:
        """Mark a question as processed in the manifest.

        Updates the question status and increments num_questions_processed.
        Called after transcript and bookkeeping are finished for a question.

        Args:
            manifest_path: Absolute path to the manifest file
            question_id: The question_id to mark
            succeeded: True if the question succeeded, False otherwise
        """
        if not manifest_path.exists():
            return

        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            # Find and update the question status
            for q in manifest.get("questions", []):
                if q.get("question_id") == question_id:
                    q["status"] = "succeeded" if succeeded else None
                    break

            # Increment processed count
            manifest["num_questions_processed"] = (
                manifest.get("num_questions_processed", 0) + 1
            )

            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
        except (json.JSONDecodeError, OSError) as e:
            info_print(f"Warning: Could not update manifest: {e}")

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

            questions = manifest.get("questions", [])
            return all(q.get("status") == "succeeded" for q in questions)
        except (json.JSONDecodeError, OSError):
            return False

    def delete_manifest_if_complete(self, manifest_path: Path) -> bool:
        """Delete manifest if all questions succeeded.

        Args:
            manifest_path: Absolute path to the manifest file

        Returns:
            True if manifest was deleted (all complete), False otherwise
        """
        if self.check_manifest_complete(manifest_path):
            try:
                manifest_path.unlink(missing_ok=True)
                info_print(
                    f"Job manifest deleted (all succeeded): {display_path(manifest_path, self.project_root)}"
                )
                return True
            except OSError:
                pass
        return False

    def find_job_manifest_and_get_null_questions(
        self,
        experiment_name: str,
        benchmark_subcategory: str,
    ) -> Optional[Tuple[Path, List[str], str]]:
        """Find the most recent job manifest for an experiment and get null question IDs.

        Used for resuming interrupted runs: finds questions that didn't succeed
        so they can be re-run while skipping succeeded ones.

        Args:
            experiment_name: Name of the experiment
            benchmark_subcategory: Benchmark category

        Returns:
            Tuple of (manifest_path, null_question_ids, config_snapshot_path) or None if no manifest found.
            null_question_ids are questions with status != "succeeded".
            config_snapshot_path is the path to the config snapshot used for this job.
        """
        exp_root_path = self.get_experiment_root()
        manifest_dir = exp_root_path / benchmark_subcategory / experiment_name / "job_manifest"

        if not manifest_dir.exists():
            return None

        # Find all manifests for this experiment
        manifests = list(manifest_dir.glob("*.json"))
        if not manifests:
            return None

        # Get most recently modified
        latest_manifest = max(manifests, key=lambda p: p.stat().st_mtime)

        try:
            with open(latest_manifest, "r") as f:
                manifest = json.load(f)
        except json.JSONDecodeError as e:
            info_print(f"Warning: Could not parse job manifest {latest_manifest.name}: {e}")
            return None
        except OSError as e:
            info_print(f"Warning: Could not read job manifest {latest_manifest.name}: {e}")
            return None

        # Extract null question IDs (not succeeded)
        null_question_ids = []
        for q in manifest.get("questions", []):
            if q.get("status") != "succeeded":
                qid = q.get("question_id", f"q_{q.get('index', '?')}")
                null_question_ids.append(qid)

        config_snapshot_path = manifest.get("config_snapshot_path", "")

        return latest_manifest, null_question_ids, config_snapshot_path


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

        self._initialize_file()

    def _initialize_file(self) -> None:
        """Create and initialize the summary file."""
        exp_root = os.environ.get(EXPERIMENT_ROOT_ENV, "experiment")
        exp_root_path = Path(exp_root)
        if not exp_root_path.is_absolute():
            exp_root_path = self.project_root / exp_root

        exp_meta = self.config["experiment_metadata"]
        benchmark = exp_meta["benchmark_subcategory"]
        experiment = exp_meta["experiment_name"]

        timestamp_str = format_filename_timestamp(self.start_time)
        job_task_id = get_job_task_id()
        filename_id = f"{timestamp_str}_{job_task_id}"

        self._summary_path = (
            exp_root_path / benchmark / experiment / "job_summary" / f"{filename_id}.json"
        )
        self._summary_path.parent.mkdir(parents=True, exist_ok=True)

        # Write initial structure
        self._write_current_state()

        info_print(
            f"Streaming job summary: {display_path(self._summary_path, self.project_root)}"
        )

    def _write_current_state(self) -> None:
        """Write current state to file (atomic write via temp file)."""
        if self._summary_path is None:
            return

        job_task_id = get_job_task_id()
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
                "questions_completed": len(self.per_transcript_stats),
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

        # Use the existing save_job_summary for final comprehensive output
        manager = BookkeepingManager(self.project_root)
        return manager.save_job_summary(
            config=self.config,
            questions_total=self.questions_total,
            questions_succeeded=self.questions_succeeded,
            questions_partial=self.questions_partial,
            questions_failed=self.questions_failed,
            start_time=self.start_time,
            end_time=end_time,
            question_range=None,  # Already applied
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

    def _get_manifest_dir(self) -> Path:
        """Get directory for grid manifests."""
        return self.project_root / "bookkeeping" / "grid_manifest"

    def save_grid_manifest(
        self,
        grid_config_path: str,
        expanded_configs: list,
        submission_timestamp: datetime,
    ) -> Path:
        """Save a grid manifest at the start of a grid run.

        Args:
            grid_config_path: Path to the original grid config file
            expanded_configs: List of (config, grid_sweep_specs) tuples from GridConfigExpander
            submission_timestamp: When the grid run started

        Returns:
            Path to the saved manifest file
        """
        manifest_dir = self._get_manifest_dir()
        manifest_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = format_filename_timestamp(submission_timestamp)
        job_task_id = get_job_task_id()
        manifest_path = manifest_dir / f"{timestamp_str}_{job_task_id}.json"

        # Build experiment run entries
        experiment_runs = []
        for i, (config, grid_sweep_specs) in enumerate(expanded_configs):
            exp_meta = config.get("experiment_metadata", {})
            experiment_runs.append({
                "run_id": i,
                "experiment_name": exp_meta.get("experiment_name", "unknown"),
                "benchmark_subcategory": exp_meta.get("benchmark_subcategory", "unknown"),
                "grid_sweep_specs": grid_sweep_specs,
                "status": None,  # null = not started
                "started_at": None,
                "completed_at": None,
            })

        manifest = {
            "grid_config_path": str(grid_config_path),
            "job_task_id": job_task_id,
            "submission_timestamp": format_timestamp(submission_timestamp),
            "num_runs_planned": len(experiment_runs),
            "num_runs_processed": 0,
            "experiment_runs": experiment_runs,
            "created_at": format_timestamp(datetime.now(timezone.utc)),
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        info_print(f"Grid manifest saved: {display_path(manifest_path, self.project_root)}")
        return manifest_path

    def mark_run_started(self, manifest_path: Path, run_id: int) -> None:
        """Mark an experiment run as started.

        Status values:
        - null: not yet run, or stopped abruptly (needs re-run)
        - "started": job started, may be interrupted (check job manifest for repair)
        - "processed": job finished naturally (don't re-run)

        Args:
            manifest_path: Path to the manifest file
            run_id: ID of the experiment run being started
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

        runs = manifest.get("experiment_runs", [])
        if 0 <= run_id < len(runs):
            runs[run_id]["status"] = "started"
            runs[run_id]["started_at"] = format_timestamp(
                datetime.now(timezone.utc)
            )

        try:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
        except OSError as e:
            info_print(f"Warning: Could not write grid manifest: {e}")

    def mark_run_processed(self, manifest_path: Path, run_id: int) -> None:
        """Mark an experiment run as processed (job finished naturally).

        Status values:
        - null: not yet run, or stopped abruptly (needs re-run)
        - "started": job started, may be interrupted (check job manifest for repair)
        - "processed": job finished naturally (don't re-run)

        Args:
            manifest_path: Path to the manifest file
            run_id: ID of the experiment run that finished
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

        runs = manifest.get("experiment_runs", [])
        if 0 <= run_id < len(runs):
            runs[run_id]["status"] = "processed"
            runs[run_id]["completed_at"] = format_timestamp(
                datetime.now(timezone.utc)
            )

        # Update processed count
        manifest["num_runs_processed"] = manifest.get("num_runs_processed", 0) + 1

        try:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
        except OSError as e:
            info_print(f"Warning: Could not write grid manifest: {e}")

    def get_pending_indices(self, manifest_path: Path) -> list:
        """Get indices of configurations that haven't been processed.

        Returns indices where status is null or "started" (not "processed").

        Args:
            manifest_path: Path to the manifest file

        Returns:
            List of configuration indices that need to be run
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
        for run in manifest.get("experiment_runs", []):
            if run.get("status") != "processed":
                pending.append(run["run_id"])
        return pending

    def is_complete(self, manifest_path: Path) -> bool:
        """Check if all configurations have been processed.

        Args:
            manifest_path: Path to the manifest file

        Returns:
            True if all configurations have status="processed"
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

        configs = manifest.get("experiment_runs", [])
        return all(c.get("status") == "processed" for c in configs)

    def delete_if_complete(self, manifest_path: Path) -> bool:
        """Delete manifest if all configurations have been processed.

        Args:
            manifest_path: Path to the manifest file

        Returns:
            True if manifest was deleted
        """
        if self.is_complete(manifest_path):
            try:
                manifest_path.unlink(missing_ok=True)
                info_print(
                    f"Grid manifest deleted (all succeeded): "
                    f"{display_path(manifest_path, self.project_root)}"
                )
                return True
            except OSError as e:
                info_print(f"Warning: Could not delete grid manifest: {e}")
        return False

    def load_and_delete_manifest(
        self, grid_config_path: str
    ) -> Optional[Tuple[List[int], Dict[int, Dict[str, str]]]]:
        """Load pending configurations from existing manifest and delete it.

        Finds the most recent manifest for this grid config, extracts pending
        configuration indices (those without status="processed"), deletes the
        manifest, and returns the pending indices along with info for "started"
        runs (needed to find their job manifests for resume).

        Args:
            grid_config_path: Path to the grid config file

        Returns:
            Tuple of (pending_indices, started_run_info) or None if no manifest found.
            started_run_info maps run_id to {"experiment_name", "benchmark_subcategory"}
            for runs with status="started" (interrupted runs that may have partial progress).
        """
        manifest_dir = self._get_manifest_dir()
        if not manifest_dir.exists():
            return None

        # Find all manifests that match this grid config
        matching_manifests = []
        for manifest_file in manifest_dir.glob("*.json"):
            try:
                with open(manifest_file, "r") as f:
                    manifest = json.load(f)
                if manifest.get("grid_config_path") == str(grid_config_path):
                    matching_manifests.append((manifest_file, manifest))
            except json.JSONDecodeError as e:
                info_print(f"Warning: Could not parse manifest {manifest_file.name}: {e}")
                continue
            except OSError as e:
                info_print(f"Warning: Could not read manifest {manifest_file.name}: {e}")
                continue

        if not matching_manifests:
            return None

        # Get most recently modified
        latest_path, latest_manifest = max(
            matching_manifests, key=lambda x: x[0].stat().st_mtime
        )

        # Extract pending run IDs and info for "started" runs
        pending: List[int] = []
        started_run_info: Dict[int, Dict[str, str]] = {}
        for run in latest_manifest.get("experiment_runs", []):
            status = run.get("status")
            if status != "processed":
                run_id = run["run_id"]
                pending.append(run_id)
                # For "started" runs, save info needed to find job manifest
                if status == "started":
                    started_run_info[run_id] = {
                        "experiment_name": run.get("experiment_name", ""),
                        "benchmark_subcategory": run.get("benchmark_subcategory", ""),
                    }

        # Delete the old manifest
        try:
            latest_path.unlink()
            info_print(f"Old manifest deleted: {latest_path.name}")
        except OSError as e:
            info_print(f"Warning: Could not delete old manifest: {e}")

        return pending, started_run_info
