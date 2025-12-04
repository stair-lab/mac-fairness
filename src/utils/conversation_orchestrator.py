"""Orchestration for multi-agent conversations."""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from src.agent import ModelFactory
from src.routing import VanillaRouter
from src.prompt.participant import ParticipantPromptBuilder
from src.utils.answer_matcher import FlexibleAnswerMatcher
from src.utils.bookkeeping_manager import BookkeepingManager
from src.utils.config_manager import ConfigManager
from src.utils.metrics import MetricsCollector
from src.utils.transcript_manager import TranscriptManager
from src.utils.zod_validator import ZodValidator
from src.utils.errors import (
    MacFairnessError,
    ValidationError,
    InvalidAnswerError,
    MissingStructuredOutputError,
    JsonDecodeError,
    ZodValidationError,
    MaxRetriesExceededError,
    MaxLengthExceededError,
    UnexpectedError,
    ErrorCollector,
    ProjectRootError,
)
from src.utils.vllm_metrics import (
    VLLMMetricsCollector,
    get_metrics_collector,
    reset_metrics_collector,
)


class ConversationOrchestrator:
    """Orchestrates multi-agent conversations with strict validation."""

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

        # Initialize validator with schema version
        schema_version = self.config["experiment_metadata"]["schema_version"]
        self.validator = ZodValidator(schema_version)

        # Ensure necessary directories
        self.paths = self.bookkeeping.ensure_directories(self.config)

        # Initialize components (will be set up later)
        self.agents = {}
        self.snapshot_path = None
        self.model_factory = None
        self.router = None
        self.prompt_builder = ParticipantPromptBuilder()
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
        """Initialize agents using model factory."""
        model_config = self.config["model_config"]
        agent_defs = self.config["agent_definitions"]

        self.model_factory = ModelFactory(model_config)
        print("✓ Model factory initialized")

        for agent_config in agent_defs:
            agent_id = agent_config["agent_id"]
            agent = self.model_factory.create_agent(agent_config)
            self.agents[agent_id] = agent
            print(f"  ✓ Created agent: {agent_id}")

    def initialize_router(self):
        """Initialize routing strategy."""
        conv_config = self.config["conversation_config"]
        routing_strategy = conv_config["routing_strategy"]
        routing_config = {
            "strategy": routing_strategy,
            "max_rounds": conv_config["max_rounds"],
        }

        if routing_strategy == "vanilla":
            self.router = VanillaRouter(routing_config)
        else:
            raise ValueError(f"Unknown routing strategy: {routing_strategy}")

        print(f"✓ Router initialized: {routing_strategy}")

    def _transform_llm_response(
        self,
        raw_response: Dict[str, Any],
        agent_config: Dict[str, Any],
        question: Dict[str, Any],
        answer_match_threshold: float,
    ) -> Dict[str, Any]:
        """Transform LLM response format to schema format.

        LLM returns: {"rationale": "...", "answer": "choice text"}
        Schema expects: {"response_type": "...", "opinion": "letter code", "rationale": "..."}

        For participant role with choice questions, uses FlexibleAnswerMatcher
        to match answer text to choices and convert to letter code.
        """
        role = agent_config.get("role", "participant")
        question_type = question.get("question_type", "multiple_choice")
        is_choice_question = question_type in ["binary", "multiple_choice"]

        transformed = {
            "response_type": role,
            "rationale": raw_response.get("rationale", ""),
        }

        if role == "participant" and is_choice_question:
            choices = question.get("choices", [])
            answer_text = raw_response.get("answer", "")

            if choices and answer_text:
                matcher = FlexibleAnswerMatcher()
                match_result = matcher.match_with_feedback(
                    answer_text, choices, threshold=answer_match_threshold
                )

                match_details = match_result.get("match_details", [])
                # Defensive check: ensure both values are not None before comparison
                if (
                    match_details
                    and match_details[0].get("match_score") is not None
                    and answer_match_threshold is not None
                    and match_details[0]["match_score"] >= answer_match_threshold
                ):
                    # Match found - use letter code as opinion
                    transformed["opinion"] = match_details[0]["id"]
                    transformed["_matched_answer_text"] = match_details[0]["text"]
                    transformed["_answer_match_info"] = match_result
                else:
                    # No valid match - keep original for retry/error handling
                    transformed["opinion"] = answer_text
                    transformed["_answer_match_info"] = match_result
            else:
                transformed["opinion"] = answer_text
        else:
            transformed["opinion"] = raw_response.get(
                "opinion", raw_response.get("answer", "")
            )

        return transformed

    def _generate_with_strict_validation(
        self,
        agent,
        agent_config: Dict[str, Any],
        prompt: str,
        question: Dict[str, Any],
        round_id: int,
        max_retries: int = 3,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[Any]]:
        """Generate response with strict validation and retry logic.

        Returns:
            Tuple of (structured_response, metadata, validation_errors)
        """
        retry_config = self.config.get("retry_config", {})
        max_retries = retry_config.get("max_retries", max_retries)
        answer_match_threshold = retry_config.get("answer_match_threshold", 0.85)

        error_collector = ErrorCollector()
        retry_count = 0
        agent_id = agent_config.get("agent_id", "unknown")

        # Track cumulative metrics across all attempts
        cumulative_time_ms = 0.0
        cumulative_tokens_generated = 0
        cumulative_tokens_prompt = 0

        for attempt in range(max_retries + 1):
            try:
                # Generate response
                start_time = time.time()
                response_data = agent.generate(prompt)
                generation_time_ms = round((time.time() - start_time) * 1000, 3)

                # Extract response text and metrics from agent's dict response
                response_text = response_data.get("text", "")
                exceeded_max_tokens = response_data.get("exceeded_max_tokens", False)
                tokens_generated = response_data.get("tokens_generated", 0)
                tokens_prompt = response_data.get("tokens_prompt", 0)

                # Accumulate metrics for this attempt
                cumulative_time_ms += generation_time_ms
                cumulative_tokens_generated += tokens_generated
                cumulative_tokens_prompt += tokens_prompt

                # Check if max tokens was exceeded and retry
                if exceeded_max_tokens:
                    error = MaxLengthExceededError(
                        agent_id=agent_id,
                        max_tokens=agent_config.get("max_tokens", 512),
                        tokens_generated=tokens_generated,
                        truncated=True,
                        attempt=attempt,
                    )
                    # Add per-attempt metrics to error details
                    error.details["generation_time_ms"] = generation_time_ms
                    error.details["tokens_prompt"] = tokens_prompt
                    error_collector.add_error(error)

                    if attempt < max_retries:
                        retry_count += 1
                        continue  # Retry with same prompt (agent may generate shorter response)
                    else:
                        # Final attempt failed due to max length - raise error
                        error = MaxRetriesExceededError(
                            agent_id=agent_id,
                            max_retries=max_retries,
                            validation_errors=[
                                e.to_dict() for e in error_collector.errors
                            ],
                        )
                        # Include cumulative metrics for transcript recording
                        error.details["cumulative_time_ms"] = round(
                            cumulative_time_ms, 3
                        )
                        error.details["cumulative_tokens_generated"] = (
                            cumulative_tokens_generated
                        )
                        error.details["cumulative_tokens_prompt"] = (
                            cumulative_tokens_prompt
                        )
                        raise error

                # Extract structured output
                if "```json" in response_text:
                    json_start = response_text.index("```json") + 7
                    json_end = response_text.index("```", json_start)
                    json_str = response_text[json_start:json_end].strip()
                else:
                    # Try to find JSON object directly
                    import re

                    json_match = re.search(r"\{[^{}]*\}", response_text)
                    if json_match:
                        json_str = json_match.group()
                    else:
                        raise MissingStructuredOutputError(
                            agent_id=agent_id, attempt=attempt
                        )

                try:
                    structured_response = json.loads(json_str)
                except json.JSONDecodeError as e:
                    raise JsonDecodeError(
                        message=str(e), raw_text=json_str, attempt=attempt
                    )

                # Transform LLM response to schema format
                structured_response = self._transform_llm_response(
                    structured_response, agent_config, question, answer_match_threshold
                )

                # Validate against Zod schema
                response_type = structured_response.get("response_type")
                if not response_type:
                    raise MissingStructuredOutputError(
                        agent_id=agent_id, attempt=attempt
                    )

                # Extract internal fields from transformation (before Zod strips them)
                answer_match_info = structured_response.pop("_answer_match_info", None)
                matched_answer_text = structured_response.pop(
                    "_matched_answer_text", None
                )

                # Check if answer matching failed (for participant + choice questions)
                if answer_match_info and not matched_answer_text:
                    # No valid match found - trigger retry
                    choices = question.get("choices", [])
                    original_answer = answer_match_info.get("original_answer", "")
                    error = InvalidAnswerError(
                        answer_text=original_answer,
                        choices=choices,
                        match_info=answer_match_info,
                        attempt=attempt,
                    )
                    # Add per-attempt metrics to error details
                    error.details["generation_time_ms"] = generation_time_ms
                    error.details["tokens_generated"] = tokens_generated
                    error.details["tokens_prompt"] = tokens_prompt
                    error_collector.add_error(error)

                    if attempt < max_retries:
                        retry_count += 1
                        continue  # Retry
                    else:
                        # Final attempt failed - raise error
                        error = MaxRetriesExceededError(
                            agent_id=agent_id,
                            max_retries=max_retries,
                            validation_errors=[
                                e.to_dict() for e in error_collector.errors
                            ],
                        )
                        # Include cumulative metrics for transcript recording
                        error.details["cumulative_time_ms"] = round(
                            cumulative_time_ms, 3
                        )
                        error.details["cumulative_tokens_generated"] = (
                            cumulative_tokens_generated
                        )
                        error.details["cumulative_tokens_prompt"] = (
                            cumulative_tokens_prompt
                        )
                        raise error

                validated_response = self.validator.validate(
                    "structured_output", structured_response
                )

                # Build metadata with cumulative values across all attempts
                metadata = {
                    "retry_count": retry_count,
                    "performance": {
                        "generation_time_ms": round(cumulative_time_ms, 3),
                        "prompt_tokens": cumulative_tokens_prompt,
                        "tokens_generated": cumulative_tokens_generated,
                    },
                }

                # Add answer match info to metadata if applicable
                if answer_match_info:
                    metadata["answer_match_info"] = answer_match_info
                if matched_answer_text:
                    metadata["matched_answer_text"] = matched_answer_text

                if error_collector.has_errors():
                    metadata["validation_errors"] = [
                        e.to_dict() for e in error_collector.errors
                    ]

                return validated_response, metadata, error_collector.errors

            except (
                JsonDecodeError,
                MissingStructuredOutputError,
                ZodValidationError,
            ) as e:
                # Ensure attempt is recorded in error details
                if e.details.get("attempt") is None:
                    e.details["attempt"] = attempt
                # Add per-attempt metrics to error details
                e.details["generation_time_ms"] = generation_time_ms
                e.details["tokens_generated"] = tokens_generated
                e.details["tokens_prompt"] = tokens_prompt
                error_collector.add_error(e)

                if attempt < max_retries:
                    retry_count += 1
                    # Future extension: Could use RetryHandler.get_retry_message()
                    # to provide error-specific hints in retry prompts
                else:
                    # Final attempt failed
                    error = MaxRetriesExceededError(
                        agent_id=agent_id,
                        max_retries=max_retries,
                        validation_errors=[e.to_dict() for e in error_collector.errors],
                    )
                    # Include cumulative metrics for transcript recording
                    error.details["cumulative_time_ms"] = round(cumulative_time_ms, 3)
                    error.details["cumulative_tokens_generated"] = (
                        cumulative_tokens_generated
                    )
                    error.details["cumulative_tokens_prompt"] = cumulative_tokens_prompt
                    raise error

    def run_conversation(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single conversation for a question.

        Args:
            question: Question dictionary with choices and metadata

        Returns:
            Complete conversation transcript
        """
        transcript_id = str(uuid.uuid4())
        execution_timestamp = datetime.now(timezone.utc)
        conversation_rounds = []
        identity_reveal_config = self.config["identity_reveal_config"]
        agent_defs = self.config["agent_definitions"]

        # Build agent configs lookup
        agent_configs_map = {ac["agent_id"]: ac for ac in agent_defs}

        # Track performance metrics
        total_tokens_generated = 0
        total_prompt_tokens = 0
        total_time_ms = 0
        all_validation_errors = []

        # Run conversation rounds
        for round_id in range(self.router.max_rounds):
            if not self.router.should_continue(round_id):
                break

            round_messages = []
            agent_ids = [ac["agent_id"] for ac in agent_defs]
            speaking_order = self.router.get_speaking_order(agent_ids, round_id)

            for agent_id in speaking_order:
                # Get visible messages
                visible_messages = self.router.get_visible_messages(
                    round_id, conversation_rounds, agent_id
                )

                agent_config = agent_configs_map[agent_id]
                agent = self.agents[agent_id]

                prompt = self.prompt_builder.build_full_prompt(
                    agent_config,
                    question,
                    identity_reveal_config,
                    visible_messages,
                    agent_configs_map,
                )

                # Debug prompts if requested
                if os.environ.get("MAC_FAIRNESS_DEBUG_FLAG"):
                    print(f"\n{'=' * 60}")
                    print(f"PROMPT for {agent_id} (Round {round_id}):")
                    print(prompt)
                    print(f"{'=' * 60}\n")

                try:
                    # Generate response with validation
                    (
                        structured_response,
                        metadata,
                        errors,
                    ) = self._generate_with_strict_validation(
                        agent, agent_config, prompt, question, round_id
                    )

                    # Track metrics
                    perf = metadata["performance"]
                    total_tokens_generated += perf["tokens_generated"]
                    total_prompt_tokens += perf["prompt_tokens"]
                    total_time_ms += perf["generation_time_ms"]

                    if "validation_errors" in metadata:
                        all_validation_errors.extend(metadata["validation_errors"])

                    # Build identity display
                    agent_identity_display = (
                        self.prompt_builder.build_agent_identity_display(
                            agent_config, identity_reveal_config
                        )
                    )

                    # Get visibility list
                    visible_to = self.router.get_visibility_list(
                        agent_id, agent_config["role"], agent_ids, round_id
                    )

                    # Generate message_id: msg_{round}_{agent_index}
                    agent_index = speaking_order.index(agent_id)
                    message_id = f"msg_{round_id}_{agent_index:03d}"

                    # Create message record with all required fields
                    message = {
                        "message_id": message_id,
                        "agent_id": agent_id,
                        "agent_role": agent_config["role"],
                        "round_id": round_id,
                        "agent_identity_display": agent_identity_display,
                        "structured_response": structured_response,
                        "visible_to": visible_to,
                        "message_metadata": metadata,
                    }

                    round_messages.append(message)

                except MacFairnessError as e:
                    # Known error (validation failure, max retries exceeded, etc.)
                    details = e.details if hasattr(e, "details") else {}

                    # Extract cumulative metrics from error (if present)
                    # and add to totals for accurate transcript recording
                    if "cumulative_time_ms" in details:
                        total_time_ms += details["cumulative_time_ms"]
                    if "cumulative_tokens_generated" in details:
                        total_tokens_generated += details["cumulative_tokens_generated"]
                    if "cumulative_tokens_prompt" in details:
                        total_prompt_tokens += details["cumulative_tokens_prompt"]

                    # Extract validation errors from MaxRetriesExceededError
                    if "validation_errors" in details:
                        all_validation_errors.extend(details["validation_errors"])

                    error_info = {
                        "error_class": e.__class__.__name__,
                        "error_code": e.error_code
                        if hasattr(e, "error_code")
                        else "MAC_FAIRNESS_ERROR",
                        "agent_id": agent_id,
                        "round_id": round_id,
                        "message": str(e),
                        "details": details,
                    }

                    # Determine status: "partial" if made progress, "failed" otherwise
                    total_messages_completed = sum(
                        len(r["messages"]) for r in conversation_rounds
                    )
                    status = "partial" if total_messages_completed > 0 else "failed"

                    # Build transcript with error
                    transcript = self.transcript_manager.build_transcript(
                        transcript_id=transcript_id,
                        question=question,
                        conversation_rounds=conversation_rounds,
                        config=self.config,
                        snapshot_path=self.snapshot_path,
                        submission_timestamp=self.submission_timestamp,
                        execution_timestamp=execution_timestamp,
                        total_tokens_generated=total_tokens_generated,
                        total_prompt_tokens=total_prompt_tokens,
                        total_time_ms=total_time_ms,
                        all_validation_errors=all_validation_errors,
                        status=status,
                        error_info=error_info,
                    )
                    return transcript

                except Exception as e:
                    # Unexpected error (network, I/O, etc.) - still save transcript
                    details = getattr(e, "details", {}) or {}

                    # Extract cumulative metrics if available
                    if "cumulative_time_ms" in details:
                        total_time_ms += details["cumulative_time_ms"]
                    if "cumulative_tokens_generated" in details:
                        total_tokens_generated += details["cumulative_tokens_generated"]
                    if "cumulative_tokens_prompt" in details:
                        total_prompt_tokens += details["cumulative_tokens_prompt"]

                    # Extract validation errors if available
                    if "validation_errors" in details:
                        all_validation_errors.extend(details["validation_errors"])

                    error_info = {
                        "error_class": e.__class__.__name__,
                        "error_code": "UNEXPECTED_ERROR",
                        "agent_id": agent_id,
                        "round_id": round_id,
                        "message": str(e),
                        "details": details,
                    }

                    total_messages_completed = sum(
                        len(r["messages"]) for r in conversation_rounds
                    )
                    status = "partial" if total_messages_completed > 0 else "failed"

                    transcript = self.transcript_manager.build_transcript(
                        transcript_id=transcript_id,
                        question=question,
                        conversation_rounds=conversation_rounds,
                        config=self.config,
                        snapshot_path=self.snapshot_path,
                        submission_timestamp=self.submission_timestamp,
                        execution_timestamp=execution_timestamp,
                        total_tokens_generated=total_tokens_generated,
                        total_prompt_tokens=total_prompt_tokens,
                        total_time_ms=total_time_ms,
                        all_validation_errors=all_validation_errors,
                        status=status,
                        error_info=error_info,
                    )
                    return transcript

            # Add round to conversation
            if round_messages:
                conversation_rounds.append(
                    {
                        "round_id": round_id,
                        "messages": round_messages,
                    }
                )

        # Build final transcript
        transcript = self.transcript_manager.build_transcript(
            transcript_id=transcript_id,
            question=question,
            conversation_rounds=conversation_rounds,
            config=self.config,
            snapshot_path=self.snapshot_path,
            submission_timestamp=self.submission_timestamp,
            execution_timestamp=execution_timestamp,
            total_tokens_generated=total_tokens_generated,
            total_prompt_tokens=total_prompt_tokens,
            total_time_ms=total_time_ms,
            all_validation_errors=all_validation_errors,
            status="succeeded",
        )

        return transcript

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

        # Calculate total time from messages
        total_time_ms = 0
        total_tokens_generated = 0
        total_tokens_prompt = 0
        max_prompt_tokens = 0  # Max prompt tokens in any single generation
        max_combined_tokens = (
            0  # Max (prompt + generated) for context length optimization
        )
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
                "exceeded_max_tokens_count": 0,
                "exceeded_retry_limit": False,
            }

        # Aggregate from rounds (for successful/partial conversations)
        for round_data in rounds:
            for msg in round_data.get("messages", []):
                agent_id = msg.get("agent_id")
                metadata = msg.get("message_metadata", {})
                perf = metadata.get("performance", {})

                gen_time = perf.get("generation_time_ms", 0)
                tokens_gen = perf.get("tokens_generated", 0)
                tokens_prompt = perf.get("prompt_tokens", 0)
                retry_count = metadata.get("retry_count", 0)
                exceeded_max = perf.get("exceeded_max_tokens", False)

                total_time_ms += gen_time
                total_tokens_generated += tokens_gen
                total_tokens_prompt += tokens_prompt

                # Track max tokens for context length optimization
                max_prompt_tokens = max(max_prompt_tokens, tokens_prompt)
                combined = tokens_prompt + tokens_gen
                max_combined_tokens = max(max_combined_tokens, combined)

                # Per-agent
                if agent_id in per_agent:
                    per_agent[agent_id]["tokens_generated"] += tokens_gen
                    per_agent[agent_id]["message_count"] += 1
                    per_agent[agent_id]["retry_count"] += retry_count
                    if exceeded_max:
                        per_agent[agent_id]["exceeded_max_tokens_count"] += 1

                # Collect validation errors
                for err in metadata.get("validation_errors", []):
                    validation_errors.append(err)

        # For failed/partial conversations, extract from error_info
        if error_info:
            details = error_info.get("details", {})
            failed_agent_id = error_info.get("agent_id")

            # Extract cumulative metrics from error details
            if "cumulative_time_ms" in details:
                total_time_ms += details["cumulative_time_ms"]
            if "cumulative_tokens_generated" in details:
                total_tokens_generated += details["cumulative_tokens_generated"]
            if "cumulative_tokens_prompt" in details:
                total_tokens_prompt += details["cumulative_tokens_prompt"]

            # Extract validation errors from error details
            for err in details.get("validation_errors", []):
                validation_errors.append(err)

            # Update per-agent stats for the failed agent
            if failed_agent_id and failed_agent_id in per_agent:
                per_agent[failed_agent_id]["exceeded_retry_limit"] = True
                # Count retries from validation errors (each error = 1 attempt)
                num_errors = len(details.get("validation_errors", []))
                if num_errors > 0:
                    # Retry count = attempts - 1 (first attempt isn't a retry)
                    per_agent[failed_agent_id]["retry_count"] += num_errors - 1
                    total_retry_attempts += num_errors - 1
                # Add cumulative tokens to per-agent
                if "cumulative_tokens_generated" in details:
                    per_agent[failed_agent_id]["tokens_generated"] += details[
                        "cumulative_tokens_generated"
                    ]

        return {
            "transcript_id": transcript.get("transcript_id"),
            "question_id": question_id,
            "status": summary.get("status"),
            "rounds_completed": len(rounds),
            "time_seconds": round(total_time_ms / 1000.0, 3),
            "tokens_generated": total_tokens_generated,
            "tokens_prompt": total_tokens_prompt,
            "max_prompt_tokens": max_prompt_tokens,
            "max_combined_tokens": max_combined_tokens,
            "retry_attempts": total_retry_attempts,
            "consensus_reached": summary.get("consensus_reached"),
            "validation_errors": validation_errors,
            "per_agent": per_agent,
        }

    def _should_collect_vllm_metrics(self) -> bool:
        """Check if vLLM metrics collection is enabled in config.

        Configure via: model_config.models.<model>.vllm_config.collect_metrics: true

        Returns:
            True if metrics collection is enabled, False otherwise (default)
        """
        model_config = self.config.get("model_config", {})
        models = model_config.get("models", {})

        for model_def in models.values():
            vllm_config = model_def.get("vllm_config", {})
            if vllm_config.get("collect_metrics", False):
                return True

        return False

    def _collect_vllm_metrics_snapshot(
        self, collector: Optional[VLLMMetricsCollector]
    ) -> None:
        """Collect a vLLM metrics snapshot from the shared model.

        Args:
            collector: The VLLMMetricsCollector to record the snapshot (can be None)
        """
        if collector is None:
            return

        try:
            from src.agent.vllm_agent import VLLMAgent

            # Get the first agent to access the shared model
            for agent in self.agents.values():
                if isinstance(agent, VLLMAgent) and agent.llm is not None:
                    snapshot = collector.collect_snapshot(agent.llm)
                    collector.record_snapshot(snapshot)
                    break
        except Exception:
            # Silently skip if metrics collection fails
            pass

    def run_experiment(self, question_range: Optional[Tuple[int, int]] = None):
        """Run the full experiment for a set of questions.

        Args:
            question_range: Optional tuple of (start_idx, end_idx) for question subset
        """
        start_time = datetime.now(timezone.utc)

        # Save config snapshot
        self.save_config_snapshot()

        # Initialize components
        self.initialize_agents()
        self.initialize_router()

        # Check if vLLM metrics collection is enabled (default: False)
        # Configure via model_config.models.<model>.vllm_config.collect_metrics: true
        collect_vllm_metrics = self._should_collect_vllm_metrics()
        vllm_metrics_collector = None
        if collect_vllm_metrics:
            reset_metrics_collector()
            vllm_metrics_collector = get_metrics_collector()
            self._collect_vllm_metrics_snapshot(vllm_metrics_collector)

        # Load and validate questions
        questions = []
        questions_file = Path(self.config["experiment_metadata"]["questions_file"])
        if not questions_file.is_absolute():
            questions_file = self.project_root / questions_file

        # Load questions from JSONL format (unified question format)
        with open(questions_file, "r") as f:
            all_questions = [json.loads(line) for line in f if line.strip()]

        print(f"✓ Loaded {len(all_questions)} questions from {questions_file.name}")

        # Apply range FIRST to avoid validating all questions during testing
        if question_range:
            start_idx, end_idx = question_range
            questions_to_validate = all_questions[start_idx:end_idx]
            print(
                f"  Processing range {start_idx}-{end_idx} ({len(questions_to_validate)} questions)"
            )
        else:
            questions_to_validate = all_questions

        # Validate questions
        invalid_questions = []
        for q_idx, q in enumerate(questions_to_validate):
            try:
                validated_q = self.validator.validate("question", q)
                questions.append(validated_q)
            except ValidationError as e:
                # Store absolute index if range was applied
                abs_idx = (start_idx + q_idx) if question_range else q_idx
                invalid_questions.append({"index": abs_idx, "error": str(e)})

        print(f"✓ Validated {len(questions)} questions")

        # Fail if any questions are invalid - indicates data corruption
        if invalid_questions:
            error_msg = f"Found {len(invalid_questions)} invalid question(s):\n"
            for inv in invalid_questions[:5]:  # Show first 5 errors
                error_msg += f"  - Question {inv['index']}: {inv['error']}\n"
            if len(invalid_questions) > 5:
                error_msg += f"  ... and {len(invalid_questions) - 5} more\n"
            raise ValidationError(
                message=error_msg,
                details={"invalid_questions": invalid_questions},
            )

        # Process questions
        questions_succeeded = 0
        questions_partial = 0
        questions_failed = 0
        error_summary = []
        per_transcript_stats = []

        print(f"\nProcessing {len(questions)} questions...")

        for idx, question in enumerate(questions):
            question_id = question.get("question_id", f"q_{idx}")
            print(f"\n{'=' * 60}")
            print(f"Processing question {idx + 1}/{len(questions)}: {question_id}")
            print(f"{'=' * 60}")

            try:
                # Run conversation
                transcript = self.run_conversation(question)

                # Save transcript
                self.transcript_manager.save_transcript(transcript)

                # Append to index
                self.transcript_manager.append_to_index(
                    transcript, question, self.config
                )

                # Collect per-transcript statistics for job summary
                transcript_stat = self._extract_transcript_stats(
                    transcript, question_id
                )
                per_transcript_stats.append(transcript_stat)

                status = transcript["conversation_summary"]["status"]
                if status == "succeeded":
                    questions_succeeded += 1
                    print(f"✓ Question {question_id} completed successfully")
                elif status == "partial":
                    questions_partial += 1
                    error_info = transcript["conversation_summary"].get("error_info")
                    error_summary.append(
                        {
                            "question_id": question_id,
                            "error": error_info,
                        }
                    )
                    error_class = error_info.get("error_class", "Unknown")
                    error_msg = error_info.get("message", "No message")
                    print(
                        f"⚠ Question {question_id} partially completed: {error_class} - {error_msg}"
                    )
                else:  # status == "failed"
                    questions_failed += 1
                    error_info = transcript["conversation_summary"].get("error_info")
                    error_summary.append(
                        {
                            "question_id": question_id,
                            "error": error_info,
                        }
                    )
                    error_class = error_info.get("error_class", "Unknown")
                    error_msg = error_info.get("message", "No message")
                    print(
                        f"✗ Question {question_id} failed: {error_class} - {error_msg}"
                    )

            except Exception as e:
                questions_failed += 1
                # Wrap unexpected exceptions in UnexpectedError
                unexpected_error = UnexpectedError(
                    original_error=e,
                    context="question_processing",
                    question_id=question_id,
                )
                error_summary.append(
                    {
                        "question_id": question_id,
                        "error": unexpected_error.to_dict(),
                    }
                )
                print(f"✗ Question {question_id} error: {unexpected_error.message}")

        # Collect final vLLM metrics snapshot and get aggregated metrics
        aggregated_vllm_metrics = None
        if vllm_metrics_collector is not None:
            self._collect_vllm_metrics_snapshot(vllm_metrics_collector)
            aggregated_vllm_metrics = vllm_metrics_collector.get_aggregated().to_dict()

        # Save job summary with comprehensive statistics
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
            vllm_metrics=aggregated_vllm_metrics,
        )

        # Print summary
        print(f"\n{'=' * 60}")
        print("EXPERIMENT COMPLETE")
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
                print(f"Invalid range format: {args.range}")
                return

    # Run experiment
    orchestrator = ConversationOrchestrator(args.config)
    orchestrator.run_experiment(question_range)


if __name__ == "__main__":
    main()
