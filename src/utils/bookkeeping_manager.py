"""Bookkeeping and record management utilities."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter

from src.utils.errors import ProjectRootError
from src.utils.recording import display_path, format_timestamp


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

        print(
            f"✓ Directories ensured for: {display_path(experiment_dir, self.project_root)}"
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

        Returns:
            Path to the saved job summary file
        """
        exp_root = os.environ.get("MAC_FAIRNESS_EXPERIMENT_ROOT", "experiment")
        exp_root_path = Path(exp_root)
        if not exp_root_path.is_absolute():
            exp_root_path = self.project_root / exp_root

        exp_meta = config["experiment_metadata"]
        conv_config = config["conversation_config"]
        retry_config = config.get("retry_config", {})
        model_config = config.get("model_config", {})
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

        # Determine backend
        backend = "ollama" if benchmark == "dev_ollama" else "vllm"

        # Build comprehensive job summary matching README specification
        job_summary = {
            "job_task_id": job_task_id,
            "experiment_name": experiment,
            "benchmark_subcategory": benchmark,
            "start_time": format_timestamp(start_time),
            "end_time": format_timestamp(end_time),
            "duration_seconds": round(duration_seconds, 3),
            "config_snapshot": config_snapshot_path or "",
            "hostname": os.uname().nodename,
            # Experiment configuration
            "experiment_configuration": {
                "routing_strategy": conv_config.get("routing_strategy"),
                "max_rounds": conv_config.get("max_rounds"),
                "agent_count": len(agent_defs),
                "shared_model_backbone": model_config.get("shared_model_backbone"),
            },
            # Retry configuration
            "retry_configuration": {
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
            # vLLM/Ollama configuration
            f"{backend}_configuration": self._build_backend_config(
                model_config, backend
            ),
            # Hardware utilization (placeholder - requires runtime monitoring)
            "hardware_utilization": aggregated_stats.get(
                "hardware_utilization",
                {
                    "gpu_info": [],
                    "peak_gpu_memory_gb": None,
                    "average_gpu_memory_gb": None,
                    "kv_cache_stats": None,
                },
            ),
            # Throughput and performance metrics
            "throughput_performance": aggregated_stats.get(
                "throughput_performance", {}
            ),
            # Token and time statistics
            "token_time_statistics": aggregated_stats.get("token_time_statistics", {}),
            # Processing statistics
            "processing_statistics": {
                "questions_attempted": questions_total,
                "questions_succeeded": questions_succeeded,
                "questions_partial": questions_partial,
                "questions_failed": questions_failed,
                "success_rate": questions_succeeded / questions_total
                if questions_total > 0
                else 0,
                "transcript_uuids": [
                    stat["transcript_id"] for stat in (per_transcript_stats or [])
                ],
                "error_summary": error_summary_structured,
            },
            # Retry statistics
            "retry_statistics": retry_statistics,
            # Per-transcript statistics for outlier detection
            "per_transcript_statistics": per_transcript_stats or [],
            # Metadata
            "created_at": format_timestamp(datetime.now(timezone.utc)),
        }

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

        print(
            f"\n\n✓ Job summary saved: {display_path(summary_path, self.project_root)}"
        )
        return str(summary_path)

    def _build_backend_config(
        self, model_config: Dict[str, Any], backend: str
    ) -> Dict[str, Any]:
        """Build backend-specific configuration section.

        Args:
            model_config: Model configuration from experiment config
            backend: Backend type ('vllm' or 'ollama')

        Returns:
            Backend configuration dictionary
        """
        shared_backbone = model_config.get("shared_model_backbone")
        models = model_config.get("models", {})

        if shared_backbone and shared_backbone in models:
            model_def = models[shared_backbone]
            if backend == "vllm":
                vllm_config = model_def.get("vllm_config", {})
                return {
                    "model_path": model_def.get("model_path"),
                    "model_family": model_def.get("family"),
                    "tensor_parallel_size": vllm_config.get("tensor_parallel_size"),
                    "gpu_memory_utilization": vllm_config.get("gpu_memory_utilization"),
                    "max_model_len": vllm_config.get("max_model_len"),
                    "dtype": vllm_config.get("dtype"),
                    "gpu_device_ids": vllm_config.get("gpu_device_ids", []),
                    "enable_prefix_caching": vllm_config.get(
                        "enable_prefix_caching", False
                    ),
                }
            else:  # ollama
                return {
                    "model_path": model_def.get("model_path"),
                    "model_family": model_def.get("family"),
                }
        return {}

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
            return {
                "throughput_performance": {},
                "token_time_statistics": {},
            }

        # Aggregate totals
        total_tokens_generated = sum(
            stat.get("tokens_generated", 0) for stat in per_transcript_stats
        )
        total_prompt_tokens = sum(
            stat.get("tokens_prompt", 0) for stat in per_transcript_stats
        )
        total_conversation_time = sum(
            stat.get("time_seconds", 0) for stat in per_transcript_stats
        )

        # Calculate overhead
        inference_time = total_conversation_time
        overhead_time = total_duration - inference_time

        # Per-agent statistics
        per_agent_stats = self._calculate_per_agent_stats(
            per_transcript_stats, agent_defs
        )

        # Throughput metrics (3-digit precision for time values)
        num_conversations = len(per_transcript_stats)
        throughput_performance = {
            "questions_per_second": round(
                num_conversations / total_duration if total_duration > 0 else 0, 3
            ),
            "tokens_per_second": round(
                total_tokens_generated / total_duration if total_duration > 0 else 0, 3
            ),
            "average_time_per_conversation_seconds": round(
                total_conversation_time / num_conversations
                if num_conversations > 0
                else 0,
                3,
            ),
            "io_overhead_seconds": round(max(0, overhead_time), 3),
        }

        # Token and time statistics (3-digit precision for time values)
        token_time_statistics = {
            "total_tokens_generated": total_tokens_generated,
            "total_prompt_tokens": total_prompt_tokens,
            "total_wall_clock_seconds": round(total_duration, 3),
            "inference_time_seconds": round(inference_time, 3),
            "overhead_time_seconds": round(max(0, overhead_time), 3),
            "per_agent_stats": per_agent_stats,
        }

        return {
            "throughput_performance": throughput_performance,
            "token_time_statistics": token_time_statistics,
        }

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
        agent_tokens = {agent["agent_id"]: 0 for agent in agent_defs}
        agent_messages = {agent["agent_id"]: 0 for agent in agent_defs}
        agent_exceeded_max = {agent["agent_id"]: 0 for agent in agent_defs}

        # Aggregate from per-transcript stats
        for stat in per_transcript_stats:
            agent_stats = stat.get("per_agent", {})
            for agent_id, data in agent_stats.items():
                if agent_id in agent_tokens:
                    agent_tokens[agent_id] += data.get("tokens_generated", 0)
                    agent_messages[agent_id] += data.get("message_count", 0)
                    agent_exceeded_max[agent_id] += data.get(
                        "exceeded_max_tokens_count", 0
                    )

        # Build result
        result = []
        for agent in agent_defs:
            agent_id = agent["agent_id"]
            total_tokens = agent_tokens.get(agent_id, 0)
            total_messages = agent_messages.get(agent_id, 0)
            result.append(
                {
                    "agent_id": agent_id,
                    "role": agent.get("role"),
                    "temperature": agent.get("temperature"),
                    "max_tokens": agent.get("max_tokens"),
                    "total_tokens": total_tokens,
                    "average_tokens_per_message": round(
                        total_tokens / total_messages if total_messages > 0 else 0, 3
                    ),
                    "messages_exceeding_max_tokens": agent_exceeded_max.get(
                        agent_id, 0
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
        questions_with_retries = []
        messages_exceeded_limit = 0

        for stat in per_transcript_stats:
            retries = stat.get("retry_attempts", 0)
            total_retries += retries

            if retries > 0:
                questions_with_retries.append(
                    {
                        "question_id": stat.get("question_id"),
                        "retries": retries,
                    }
                )

            # Per-agent and per-role retries
            agent_stats = stat.get("per_agent", {})
            for agent_id, data in agent_stats.items():
                msg_count = data.get("message_count", 0)
                retry_count = data.get("retry_count", 0)
                exceeded = data.get("exceeded_retry_limit", False)

                total_messages += msg_count

                if agent_id in by_agent:
                    by_agent[agent_id] += retry_count
                    agent_messages[agent_id] += msg_count

                # Find role for this agent
                for agent in agent_defs:
                    if agent["agent_id"] == agent_id:
                        role = agent.get("role", "unknown")
                        by_role[role] += retry_count
                        role_messages[role] += msg_count
                        break

                if exceeded:
                    messages_exceeded_limit += 1

            # Validation error types (use error_code for aggregation)
            for error in stat.get("validation_errors", []):
                error_code = error.get("error_code", "UNKNOWN")
                validation_errors[error_code] += 1

        # Sort questions by retry count
        questions_with_retries.sort(key=lambda x: x["retries"], reverse=True)

        # Build by_agent list (retry rate = retries per conversation)
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

        # Most common validation errors (by error_code) - counts ALL validation errors
        # including those from successful conversations that recovered via retry
        total_validation_errors = sum(validation_errors.values())
        most_common_errors = [
            {"error_code": error_code, "count": count}
            for error_code, count in validation_errors.most_common(10)
        ]

        # Calculate retry rate per conversation (more meaningful than per message)
        num_conversations = len(per_transcript_stats)
        conversations_with_retries = len(questions_with_retries)

        return {
            "total_retry_attempts": total_retries,
            "total_validation_errors": total_validation_errors,
            "total_conversations": num_conversations,
            "conversations_with_retries": conversations_with_retries,
            "average_retries_per_conversation": round(
                total_retries / num_conversations if num_conversations > 0 else 0, 3
            ),
            "by_agent": by_agent_list,
            "by_role": by_role_list,
            "validation_errors_by_type": most_common_errors,
            "questions_with_most_retries": questions_with_retries[:10],
            "messages_exceeded_retry_limit": messages_exceeded_limit,
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
