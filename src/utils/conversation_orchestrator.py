"""Orchestration for multi-agent conversations."""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

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
        """Extract statistics from a completed transcript for job summary aggregation.

        Args:
            transcript: Completed transcript dictionary
            question_id: Question identifier

        Returns:
            Dictionary of transcript statistics for job summary
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

    async def run_experiment(
        self,
        question_range: Optional[Tuple[int, int]] = None,
    ):
        """Run experiment with async parallel conversation processing.

        Runs all conversations in parallel using AsyncConversationRunner.
        Backend handles request queuing and batching (vLLM's max_num_seqs).

        Args:
            question_range: Optional tuple of (start_idx, end_idx) for question subset
        """
        start_time = datetime.now(timezone.utc)

        # Save config snapshot
        self.save_config_snapshot()

        # Initialize components
        self.initialize_agents()
        self.initialize_router()

        try:
            await self._run_experiment_inner(
                question_range=question_range,
                start_time=start_time,
            )
        finally:
            await self._cleanup_agents()

    async def _run_experiment_inner(
        self,
        question_range: Optional[Tuple[int, int]],
        start_time: datetime,
    ):
        """Inner experiment logic wrapped for cleanup."""
        from src.utils.request_scheduler import RequestScheduler
        from src.agent.async_vllm_agent import AsyncVLLMAgent

        # Load questions
        questions_file = Path(self.config["experiment_metadata"]["questions_file"])
        if not questions_file.is_absolute():
            questions_file = self.project_root / questions_file

        with open(questions_file, "r") as f:
            all_questions = [json.loads(line) for line in f if line.strip()]

        info_print(f"Loaded {len(all_questions)} questions from {questions_file.name}")

        if question_range:
            start_idx, end_idx = question_range
            questions = all_questions[start_idx:end_idx]
            info_print(
                f"Processing range {start_idx}-{end_idx} ({len(questions)} questions)"
            )
        else:
            questions = all_questions

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
        questions_succeeded = 0
        questions_partial = 0
        questions_failed = 0
        error_summary = []
        per_transcript_stats = []

        def progress_callback(
            completed: int, total: int, question_idx: int, transcript: Dict[str, Any]
        ):
            nonlocal questions_succeeded, questions_partial, questions_failed

            question = questions[question_idx]
            question_id = question.get("question_id", f"q_{question_idx}")

            # Save transcript immediately
            self.transcript_manager.save_transcript(transcript)
            self.transcript_manager.append_to_index(transcript, question, self.config)

            # Collect statistics
            transcript_stat = self._extract_transcript_stats(transcript, question_id)
            per_transcript_stats.append(transcript_stat)

            # Update counters
            status = transcript.get("conversation_summary", {}).get("status", "failed")
            if status == "succeeded":
                questions_succeeded += 1
            elif status == "partial":
                questions_partial += 1
                error_info = transcript.get("conversation_summary", {}).get(
                    "error_info"
                )
                if error_info:
                    error_summary.append(
                        {"question_id": question_id, "error": error_info}
                    )
            else:
                questions_failed += 1
                error_info = transcript.get("conversation_summary", {}).get(
                    "error_info"
                )
                if error_info:
                    error_summary.append(
                        {"question_id": question_id, "error": error_info}
                    )

            # Only print progress if live status display is not enabled
            if not is_live_status_enabled():
                info_print(f"[{completed}/{total}] Question {question_id}: {status}", prefix=False)

        info_print(f"Processing {len(questions)} questions...", prefix=False)

        # Run all questions with request-level scheduling
        await scheduler.run_questions(
            questions=questions,
            progress_callback=progress_callback,
        )

        # Get batching metrics if available (AsyncVLLMAgent imported at top of method)
        batching_metrics = AsyncVLLMAgent.get_engine_metrics()

        # Get effective backend config (includes auto-calculated max_num_seqs)
        effective_backend_config = AsyncVLLMAgent.get_effective_config()

        # Save job summary
        end_time = datetime.now(timezone.utc)
        self.bookkeeping.save_job_summary(
            config=self.config,
            questions_total=len(questions),
            questions_succeeded=questions_succeeded,
            questions_partial=questions_partial,
            questions_failed=questions_failed,
            start_time=start_time,
            end_time=end_time,
            question_range=question_range,
            error_summary=error_summary if error_summary else None,
            per_transcript_stats=per_transcript_stats if per_transcript_stats else None,
            config_snapshot_path=self.snapshot_path,
            effective_backend_config=effective_backend_config
            if effective_backend_config
            else None,
        )

        # Print summary
        print(f"\n{'=' * 60}")
        print("EXPERIMENT COMPLETE")
        print(f"{'=' * 60}")

        # Hardware & Model info
        gpu_info = get_gpu_info()
        print(f"GPU: {format_gpu_info(gpu_info)}")

        # Model and backend config info
        model_defs = self.config.get("model_definitions", {})
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
        print(f"Total questions: {len(questions)}")
        print(f"Succeeded: {questions_succeeded}")
        print(f"Partial: {questions_partial}")
        print(f"Failed: {questions_failed}")
        if len(questions) > 0:
            print(f"Success rate: {questions_succeeded / len(questions) * 100:.1f}%")
        else:
            print("Success rate: N/A (no questions processed)")
        print(f"Duration: {(end_time - start_time).total_seconds():.1f}s")
        if batching_metrics:
            timing = batching_metrics.get("timing", {})
            concurrency = timing.get("concurrency", {})
            print(
                f"Requests: {batching_metrics.get('total_requests', 0)}, "
                f"peak concurrent: {concurrency.get('peak_concurrent_requests', 0)}, "
                f"avg latency: {timing.get('avg_latency_seconds', 0):.3f}s"
            )
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
    parser.add_argument(
        "--range",
        type=str,
        help="Question range in format 'start:end' (e.g., '0:10')",
    )

    args = parser.parse_args()

    # Parse range if provided
    question_range = None
    if args.range:
        parts = args.range.split(":")
        if len(parts) == 2:
            try:
                start = int(parts[0])
                end = int(parts[1])
                question_range = (start, end)
            except ValueError:
                info_print(f"Invalid range format: {args.range}")
                return

    # Run experiment
    orchestrator = ConversationOrchestrator(args.config)
    asyncio.run(orchestrator.run_experiment(question_range=question_range))


if __name__ == "__main__":
    main()
