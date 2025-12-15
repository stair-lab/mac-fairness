"""Orchestration for multi-agent conversations."""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

from src.agent import ModelFactory
from src.routing import VanillaRouter
from src.prompt.participant import ParticipantPromptBuilder
from src.utils import (
    BookkeepingManager,
    ConfigManager,
    MetricsCollector,
    ProjectRootError,
    TranscriptManager,
    info_print,
    is_live_status_enabled,
    get_gpu_info,
    format_gpu_info,
)
from src.utils.bookkeeping_manager import StreamingJobSummary


class ConversationOrchestrator:
    """Orchestrates multi-agent conversations with async execution.

    All conversations run in parallel. Backend handles request queuing:
    - vLLM: Batches via max_num_seqs_upper_bound (effective value limited by KV cache)
    - Ollama: Internal request queue (no true batching)
    """

    def __init__(self, config_path: str):
        """Initialize the conversation orchestrator.

        Args:
            config_path: Path to the configuration YAML file
        """
        # Find project root
        current = Path(__file__).resolve()
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                self.project_root = current
                break
            current = current.parent
        else:
            raise ProjectRootError()

        # Initialize utility managers
        self.config_manager = ConfigManager(config_path, self.project_root)
        self.bookkeeping = BookkeepingManager(self.project_root)
        self.transcript_manager = TranscriptManager(self.project_root)
        self.metrics = MetricsCollector()

        # Load and validate configuration
        self.config = self.config_manager.load_and_validate()

        # Ensure necessary directories
        self.paths = self.bookkeeping.ensure_directories(self.config)

        # Initialize components (will be set up later)
        self.agents = {}
        self.snapshot_path = None
        self.model_factory = None
        self.router = None
        # Initialize prompt builder with template config (if provided)
        participant_template_config = self.config.get(
            "prompt_template_config", {}
        ).get("for_participant")
        self.prompt_builder = ParticipantPromptBuilder(participant_template_config)
        self.submission_timestamp = None
        self.manifest_path: Optional[Path] = None

    def save_config_snapshot(self) -> str:
        """Save a snapshot of the configuration.

        Returns:
            Path to the saved snapshot
        """
        self.submission_timestamp = datetime.now(timezone.utc)
        self.snapshot_path = self.config_manager.save_snapshot(
            self.config, self.submission_timestamp
        )
        return self.snapshot_path

    def initialize_agents(self):
        """Initialize async agents using model factory."""
        agent_defs = self.config["agent_definitions"]

        self.model_factory = ModelFactory(self.config)
        info_print("Model factory initialized")

        for agent_config in agent_defs:
            agent_id = agent_config["agent_id"]
            agent = self.model_factory.create_agent(agent_config)
            self.agents[agent_id] = agent
            agent_type = type(agent).__name__
            info_print(f"Created agent: {agent_id} ({agent_type})")

    def initialize_router(self):
        """Initialize routing strategy."""
        conversation_config = self.config["conversation_config"]
        routing_strategy = conversation_config["routing_strategy"]

        if routing_strategy == "vanilla":
            self.router = VanillaRouter(conversation_config)
        else:
            raise ValueError(f"Unknown routing strategy: {routing_strategy}")

        info_print(f"Router initialized: {routing_strategy}")

    async def _cleanup_agents(self) -> None:
        """Cleanup all agent resources (sessions, engines, GPU memory)."""
        # Get unique agent classes to call class-level cleanup
        agent_classes = set(type(agent) for agent in self.agents.values())
        for agent_class in agent_classes:
            if hasattr(agent_class, "cleanup_all_async"):
                await agent_class.cleanup_all_async()

    def _extract_transcript_stats(
        self, transcript: Dict[str, Any], question_id: str
    ) -> Dict[str, Any]:
        """Extract statistics from a completed transcript for task summary aggregation.

        Args:
            transcript: Completed transcript dictionary
            question_id: Question identifier

        Returns:
            Dictionary of transcript statistics for task summary
        """
        summary = transcript.get("conversation_summary", {})
        retry_stats = summary.get("retry_statistics", {})
        rounds = transcript.get("conversation_rounds", [])
        error_info = summary.get("error_info")

        # Aggregate token counts from messages
        total_tokens_generated = 0
        total_tokens_prompt = 0
        max_tokens_prompt = 0
        max_tokens_combined = 0
        total_retry_attempts = retry_stats.get("total_retry_attempts", 0)
        validation_errors = []

        # Per-agent statistics
        per_agent = {}
        agent_defs = self.config.get("agent_definitions", [])
        for agent in agent_defs:
            per_agent[agent["agent_id"]] = {
                "tokens_generated": 0,
                "message_count": 0,
                "retry_count": 0,
                "exceeded_retry_limit": False,
            }

        # Aggregate from rounds
        for round_data in rounds:
            for msg in round_data.get("messages", []):
                agent_id = msg.get("agent_id")
                metadata = msg.get("message_metadata", {})

                tokens_gen = metadata.get("response_tokens", 0)
                tokens_prompt = metadata.get("prompt_tokens", 0)
                retry_count = metadata.get("retry_count", 0)

                total_tokens_generated += tokens_gen
                total_tokens_prompt += tokens_prompt

                max_tokens_prompt = max(max_tokens_prompt, tokens_prompt)
                combined = tokens_prompt + tokens_gen
                max_tokens_combined = max(max_tokens_combined, combined)

                if agent_id in per_agent:
                    per_agent[agent_id]["tokens_generated"] += tokens_gen
                    per_agent[agent_id]["message_count"] += 1
                    per_agent[agent_id]["retry_count"] += retry_count

                for err in metadata.get("validation_errors", []):
                    validation_errors.append(err)

        # For failed/partial conversations, extract from error_info
        if error_info:
            details = error_info.get("details", {})
            failed_agent_id = error_info.get("agent_id")

            if "cumulative_tokens_generated" in details:
                total_tokens_generated += details["cumulative_tokens_generated"]
            if "cumulative_tokens_prompt" in details:
                total_tokens_prompt += details["cumulative_tokens_prompt"]

            for err in details.get("validation_errors", []):
                validation_errors.append(err)

            if failed_agent_id and failed_agent_id in per_agent:
                per_agent[failed_agent_id]["exceeded_retry_limit"] = True
                num_errors = len(details.get("validation_errors", []))
                if num_errors > 0:
                    per_agent[failed_agent_id]["retry_count"] += num_errors - 1
                    total_retry_attempts += num_errors - 1
                if "cumulative_tokens_generated" in details:
                    per_agent[failed_agent_id]["tokens_generated"] += details[
                        "cumulative_tokens_generated"
                    ]

        return {
            "transcript_id": transcript.get("transcript_id"),
            "question_id": question_id,
            "status": summary.get("status"),
            "rounds_completed": len(rounds),
            "tokens_generated": total_tokens_generated,
            "tokens_prompt": total_tokens_prompt,
            "max_tokens_prompt": max_tokens_prompt,
            "max_tokens_combined": max_tokens_combined,
            "retry_attempts": total_retry_attempts,
            "consensus_reached": summary.get("consensus_reached"),
            "validation_errors": validation_errors,
            "per_agent": per_agent,
        }

    def _extract_timestamp_from_snapshot(self, snapshot_path: str) -> datetime:
        """Extract submission timestamp from snapshot filename.

        Snapshot filename format: {experiment_name}_{timestamp}.yaml
        Timestamp format: 20251204T120000.123Z

        Args:
            snapshot_path: Path to the config snapshot file

        Returns:
            Parsed datetime object (UTC)
        """
        # Get filename without extension: {exp_name}_{timestamp}
        filename = Path(snapshot_path).stem
        # Timestamp is after last underscore: 20251204T120000.123Z
        timestamp_str = filename.split("_")[-1]
        # Parse: 20251204T120000.123Z
        dt_str = timestamp_str.rstrip("Z")
        main_part, ms_part = dt_str.split(".")
        dt = datetime.strptime(main_part, "%Y%m%dT%H%M%S")
        dt = dt.replace(microsecond=int(ms_part) * 1000, tzinfo=timezone.utc)
        return dt

    async def run_job(
        self,
        question_ids: Optional[Set[str]] = None,
        succeeded_questions: Optional[List[Dict[str, Any]]] = None,
        existing_snapshot_path: Optional[str] = None,
        old_manifest_path: Optional[Path] = None,
    ):
        """Run job with async parallel conversation processing.

        Runs all conversations in parallel using AsyncConversationRunner.
        Backend handles request queuing and batching (vLLM's max_num_seqs).

        Args:
            question_ids: Optional set of question IDs to process (for resume)
            succeeded_questions: Optional list of question entries that already succeeded
                (carried over from previous manifest on resume). These will be preserved
                in the new manifest to ensure correct progress tracking.
            existing_snapshot_path: Optional path to existing config snapshot (for resume).
                If provided, reuses this snapshot instead of creating a new one.
            old_manifest_path: Optional path to old task manifest to delete after new
                manifest is created (for atomic resume). Ensures create-then-delete order.
        """
        start_time = datetime.now(timezone.utc)

        # Reuse existing snapshot or create new one
        if existing_snapshot_path:
            self.snapshot_path = existing_snapshot_path
            self.submission_timestamp = self._extract_timestamp_from_snapshot(existing_snapshot_path)
            info_print(f"Reusing existing config snapshot: {Path(existing_snapshot_path).name}")
        else:
            self.save_config_snapshot()

        # Initialize components
        self.initialize_agents()
        self.initialize_router()

        try:
            await self._run_job_inner(
                question_ids=question_ids,
                succeeded_questions=succeeded_questions,
                start_time=start_time,
                old_manifest_path=old_manifest_path,
            )
        finally:
            await self._cleanup_agents()

    async def _run_job_inner(
        self,
        question_ids: Optional[Set[str]],
        succeeded_questions: Optional[List[Dict[str, Any]]],
        start_time: datetime,
        old_manifest_path: Optional[Path] = None,
    ):
        """Inner job logic wrapped for cleanup."""
        from src.utils.request_scheduler import RequestScheduler
        from src.agent.async_vllm_agent import AsyncVLLMAgent

        # Load questions
        questions_file = Path(self.config["experiment_metadata"]["questions_file"])
        if not questions_file.is_absolute():
            questions_file = self.project_root / questions_file

        with open(questions_file, "r") as f:
            all_questions = [json.loads(line) for line in f if line.strip()]

        info_print(f"Loaded {len(all_questions)} questions from {questions_file.name}")

        # For resume, filter to only process pending questions
        if question_ids:
            questions_to_process = [q for q in all_questions if q.get("question_id") in question_ids]
            info_print(f"Processing {len(questions_to_process)} questions by ID filter")
        else:
            questions_to_process = all_questions

        # Save task manifest (pre-registration of planned questions for recovery)
        # Each question starts with status=null, marked "succeeded" when complete
        # For resume, succeeded_questions carries over questions that already succeeded
        self.manifest_path = self.bookkeeping.save_task_manifest(
            config=self.config,
            questions=all_questions,  # Always save ALL questions for correct total count
            submission_timestamp=self.submission_timestamp,
            config_snapshot_path=self.snapshot_path,
            succeeded_questions=succeeded_questions,
        )

        # Delete old manifest AFTER new one is created (atomic create-then-delete)
        if old_manifest_path and old_manifest_path.exists():
            old_manifest_path.unlink()
            info_print(f"Deleted old task manifest: {old_manifest_path.name}")

        # Calculate total questions (for correct live status display)
        # On resume: total = carried-over succeeded + questions being processed now
        num_succeeded_carry_over = len(succeeded_questions) if succeeded_questions else 0
        questions_total = num_succeeded_carry_over + len(questions_to_process)

        # Initialize streaming task summary (updates in real-time as transcripts complete)
        streaming_summary = StreamingJobSummary(
            config=self.config,
            questions_total=questions_total,
            start_time=start_time,
            config_snapshot_path=self.snapshot_path,
            project_root=self.project_root,
        )
        # Pre-populate succeeded count from carry-over
        streaming_summary.questions_succeeded = num_succeeded_carry_over

        # Get effective backend config from vLLM agent (if available)
        # This includes actual max_num_seqs computed from KV cache availability
        effective_backend_config = None
        for agent in self.agents.values():
            if hasattr(agent, "get_effective_backend_config"):
                effective_backend_config = agent.get_effective_backend_config()
                break

        # Create request scheduler (reads max_num_seqs per-model from config)
        scheduler = RequestScheduler(
            agents=self.agents,
            router=self.router,
            prompt_builder=self.prompt_builder,
            config=self.config,
            transcript_manager=self.transcript_manager,
            snapshot_path=self.snapshot_path,
            submission_timestamp=self.submission_timestamp,
            effective_backend_config=effective_backend_config,
        )
        # Print per-model scheduling info
        model_info = ", ".join(
            f"{m}:{scheduler.model_max_num_seqs[m]}"
            for m in sorted(scheduler.model_max_num_seqs.keys())
        )
        info_print(f"Request scheduler initialized (per-model effective max_num_seqs: {model_info})")

        # Process questions with progress tracking
        # Save transcripts immediately as they complete (crash-safe)
        # Streaming summary updates in real-time
        per_transcript_stats: List[Dict[str, Any]] = []
        error_summary: List[Dict[str, Any]] = []

        def progress_callback(
            completed: int, total: int, question_idx: int, transcript: Dict[str, Any]
        ):
            # Get question_id from transcript (source of truth)
            question_id = transcript["experiment_metadata"]["question_id"]
            question = questions_to_process[question_idx]

            # Save transcript immediately (can be lost on interrupt, recoverable)
            self.transcript_manager.save_transcript(transcript)

            # Collect statistics
            transcript_stat = self._extract_transcript_stats(transcript, question_id)
            per_transcript_stats.append(transcript_stat)

            # Get error info if present
            status = transcript.get("conversation_summary", {}).get("status", "failed")
            error_info = None
            if status in ("partial", "failed"):
                error_info = transcript.get("conversation_summary", {}).get("error_info")
                if error_info:
                    error_summary.append({"question_id": question_id, "error": error_info})

            # Update streaming summary (writes to disk)
            streaming_summary.record_completion(transcript_stat, error_info)

            # Atomically record completion in both manifest and index
            # This ensures consistency: either both are updated or neither
            benchmark = self.config["experiment_metadata"]["benchmark_subcategory"]
            index_path = self.transcript_manager.get_index_path(benchmark)
            index_entry = self.transcript_manager.build_index_entry(
                transcript, question, self.config
            )
            self.bookkeeping.record_question_completion(
                manifest_path=self.manifest_path,
                question_id=question_id,
                succeeded=(status == "succeeded"),
                index_path=index_path,
                index_entry=index_entry,
            )

            # Only print progress if live status display is not enabled
            if not is_live_status_enabled():
                info_print(f"[{completed}/{total}] Question {question_id}: {status}", prefix=False)

        info_print(f"Processing {len(questions_to_process)} questions...", prefix=False)

        # Run all questions with request-level scheduling
        await scheduler.run_questions(
            questions=questions_to_process,
            progress_callback=progress_callback,
        )

        # Get batching metrics if available (AsyncVLLMAgent imported at top of method)
        batching_metrics = AsyncVLLMAgent.get_engine_metrics()

        # Get effective backend config (includes auto-calculated max_num_seqs)
        effective_backend_config = AsyncVLLMAgent.get_effective_config()

        # Finalize task summary
        end_time = datetime.now(timezone.utc)
        self.bookkeeping.save_task_summary(
            config=self.config,
            questions_total=questions_total,
            questions_succeeded=streaming_summary.questions_succeeded,
            questions_partial=streaming_summary.questions_partial,
            questions_failed=streaming_summary.questions_failed,
            start_time=start_time,
            end_time=end_time,
            error_summary=error_summary if error_summary else None,
            per_transcript_stats=per_transcript_stats if per_transcript_stats else None,
            config_snapshot_path=self.snapshot_path,
            effective_backend_config=effective_backend_config
            if effective_backend_config
            else None,
        )

        # Delete manifest if all questions succeeded, otherwise keep for resume
        manifest_deleted = self.bookkeeping.delete_manifest_if_complete(self.manifest_path)

        # Print summary
        print(f"\n{'=' * 60}")
        print("EXPERIMENT COMPLETE")
        print(f"{'=' * 60}")

        # Model and backend config info
        model_defs = self.config.get("model_definitions", {})

        # Check if any backend is vLLM (GPU info only relevant for vLLM)
        has_vllm = any(
            model_def.get("backend") == "vllm"
            for model_def in model_defs.values()
        )
        if has_vllm:
            gpu_info = get_gpu_info()
            print(f"GPU: {format_gpu_info(gpu_info)}")

        for model_name, model_def in model_defs.items():
            # vLLM uses model_path, Ollama uses model_name
            model_id = model_def.get("model_path") or model_def.get("model_name", "unknown")
            backend = model_def.get("backend", "unknown")
            print(f"Model: {model_id}")

            if backend == "vllm":
                vllm_config = model_def.get("vllm_config", {})
                # Get effective max_num_seqs from backend config if available
                effective_max_seqs = None
                if effective_backend_config:
                    for path, cfg in effective_backend_config.items():
                        if model_id in path or path in model_id:
                            effective_max_seqs = cfg.get("max_num_seqs")
                            break

                config_parts = []
                if "max_model_len" in vllm_config:
                    config_parts.append(f"ctx={vllm_config['max_model_len']}")
                if effective_max_seqs:
                    config_parts.append(f"max_num_seqs={effective_max_seqs}")
                elif "max_num_seqs_upper_bound" in vllm_config:
                    config_parts.append(
                        f"max_num_seqs<={vllm_config['max_num_seqs_upper_bound']}"
                    )
                if "tensor_parallel_size" in vllm_config:
                    tp = vllm_config["tensor_parallel_size"]
                    if tp > 1:
                        config_parts.append(f"tp={tp}")
                if config_parts:
                    print(f"vLLM: {', '.join(config_parts)}")

            elif backend == "ollama":
                ollama_config = model_def.get("ollama_config", {})
                config_parts = []
                if "num_ctx" in ollama_config:
                    config_parts.append(f"ctx={ollama_config['num_ctx']}")
                if config_parts:
                    print(f"Ollama: {', '.join(config_parts)}")

        print(f"{'=' * 60}")
        print(f"Total questions: {questions_total}")
        print(f"Succeeded: {streaming_summary.questions_succeeded}")
        print(f"Partial: {streaming_summary.questions_partial}")
        print(f"Failed: {streaming_summary.questions_failed}")
        total_completed = streaming_summary.questions_succeeded + streaming_summary.questions_partial + streaming_summary.questions_failed
        if total_completed > 0:
            print(f"Success rate: {streaming_summary.questions_succeeded / total_completed * 100:.1f}%")
        else:
            print("Success rate: N/A (no questions completed)")
        print(f"Duration: {(end_time - start_time).total_seconds():.1f}s")
        if batching_metrics:
            timing = batching_metrics.get("timing", {})
            concurrency = timing.get("concurrency", {})
            print(
                f"Requests: {batching_metrics.get('total_requests', 0)}, "
                f"peak concurrent: {concurrency.get('peak_concurrent_requests', 0)}, "
                f"avg latency: {timing.get('avg_latency_seconds', 0):.3f}s"
            )
        if not manifest_deleted:
            info_print("Some questions failed. To resume via grid:")
            info_print("[ENV_VARS] python script/run_job.py <grid_config_snapshot> --grid --resume")
        print()  # Trailing newline


def main():
    """Main entry point for running experiments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run MAC-fairness conversation experiments"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to configuration YAML file",
    )

    args = parser.parse_args()

    # Run job
    orchestrator = ConversationOrchestrator(args.config)
    asyncio.run(orchestrator.run_job())


if __name__ == "__main__":
    main()
