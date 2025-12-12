"""Bookkeeping and record management utilities."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter

from src.utils.errors import ProjectRootError
from src.utils.logging import display_path, format_timestamp, info_print, is_debug_enabled


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
        exp_root = os.environ.get("MAC_FAIRNESS_EXPERIMENT_ROOT", "experiment")
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
        exp_root = os.environ.get("MAC_FAIRNESS_EXPERIMENT_ROOT", "experiment")
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
        timestamp_str = start_time.strftime("%Y%m%dT%H%M%SZ")
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
        exp_root = os.environ.get("MAC_FAIRNESS_EXPERIMENT_ROOT", "experiment")
        exp_root_path = Path(exp_root)
        if not exp_root_path.is_absolute():
            exp_root_path = self.project_root / exp_root
        return exp_root_path
