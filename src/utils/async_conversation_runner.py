"""Async conversation runner for parallel conversation processing.

This module provides async orchestration that runs multiple conversations
in parallel while maintaining dependency-based speaking order within each round.

Key features:
- Runs multiple conversations concurrently via asyncio
- Dependency-based agent speaking order within rounds
- Participants can run in parallel, observers wait for dependencies
- Proper error propagation - no graceful handling
- Backend-agnostic: works with AsyncOllamaAgent and AsyncVLLMAgent
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.utils.errors import (
    MacFairnessError,
    MaxRetriesExceededError,
    MaxLengthExceededError,
    MissingStructuredOutputError,
    InvalidAnswerError,
    ErrorCollector,
    MissingConfigSectionError,
)
from src.utils.logging import debug_print


@dataclass
class ConversationTokenStats:
    """Token statistics for a single conversation."""

    prompt_tokens: List[int] = field(default_factory=list)
    response_tokens: List[int] = field(default_factory=list)

    def record(self, prompt_tokens: int, response_tokens: int) -> None:
        """Record token counts for a single message."""
        self.prompt_tokens.append(prompt_tokens)
        self.response_tokens.append(response_tokens)

    def to_dict(self) -> Dict[str, Any]:
        """Return conversation-level token statistics."""
        if not self.prompt_tokens:
            return {
                "total_messages": 0,
                "prompt_tokens": {"total": 0, "avg": 0, "max": 0},
                "response_tokens": {"total": 0, "avg": 0, "max": 0},
                "combined_tokens": {"total": 0, "avg": 0, "max": 0},
            }

        combined = [p + r for p, r in zip(self.prompt_tokens, self.response_tokens)]
        n = len(self.prompt_tokens)

        return {
            "total_messages": n,
            "prompt_tokens": {
                "total": sum(self.prompt_tokens),
                "avg": round(sum(self.prompt_tokens) / n, 1),
                "max": max(self.prompt_tokens),
            },
            "response_tokens": {
                "total": sum(self.response_tokens),
                "avg": round(sum(self.response_tokens) / n, 1),
                "max": max(self.response_tokens),
            },
            "combined_tokens": {
                "total": sum(combined),
                "avg": round(sum(combined) / n, 1),
                "max": max(combined),
            },
        }


class AsyncConversationRunner:
    """Runs multiple conversations in parallel with async agents.

    This class wraps the conversation running logic to enable parallelism
    across conversations while respecting dependency-based ordering within
    each conversation round.

    Speaking Order Model:
    - Participants (role="participant"): No dependencies, can run in parallel
    - Observers (moderator, devils_advocate, etc.): Have speak_after_within_round
      dependencies specifying which agent_ids must complete first

    Example agent_config with dependencies:
        {
            "agent_id": "moderator_001",
            "role": "moderator",
            "role_specific_config": {
                "speak_after_within_round": ["spkr_000", "spkr_001", "spkr_002"]
            }
        }

    Usage:
        runner = AsyncConversationRunner(
            agents=agents_dict,
            router=router,
            prompt_builder=prompt_builder,
            config=config,
            ...
        )

        # Run all questions in parallel
        results = await runner.run_questions_async(questions)
    """

    def __init__(
        self,
        agents: Dict[str, Any],
        router: Any,
        prompt_builder: Any,
        config: Dict[str, Any],
        transcript_manager: Any,
        snapshot_path: str,
        submission_timestamp: datetime,
    ):
        """Initialize the async conversation runner.

        Args:
            agents: Dictionary of agent_id -> async agent instance
            router: Router instance for conversation flow
            prompt_builder: Prompt builder instance
            config: Full experiment configuration
            transcript_manager: Transcript manager for building/saving
            snapshot_path: Path to config snapshot
            submission_timestamp: When experiment was submitted

        Raises:
            MissingConfigSectionError: If required config sections are missing
        """
        self.agents = agents
        self.router = router
        self.prompt_builder = prompt_builder
        self.config = config
        self.transcript_manager = transcript_manager
        self.snapshot_path = snapshot_path
        self.submission_timestamp = submission_timestamp

        # Extract commonly used config
        self.identity_reveal_config = config["identity_reveal_config"]
        self.agent_defs = config["agent_definitions"]
        self.agent_configs_map = {ac["agent_id"]: ac for ac in self.agent_defs}

        # Validate required config sections
        if "retry_config" not in config:
            raise MissingConfigSectionError("retry_config")
        self.retry_config = config["retry_config"]

        if "max_retries" not in self.retry_config:
            raise MissingConfigSectionError("retry_config.max_retries")

        # Build dependency graph for speaking order
        self._build_dependency_graph()

    def _build_dependency_graph(self) -> None:
        """Build dependency graph from agent configurations.

        Creates a mapping of agent_id -> set of agent_ids that must complete first.
        """
        self.dependencies: Dict[str, Set[str]] = {}

        for agent_config in self.agent_defs:
            agent_id = agent_config["agent_id"]
            role_config = agent_config.get("role_specific_config", {})
            speak_after = role_config.get("speak_after_within_round", [])

            self.dependencies[agent_id] = set(speak_after)

    def _get_ready_agents(
        self,
        pending_agents: Set[str],
        completed_agents: Set[str],
    ) -> List[str]:
        """Get agents whose dependencies are satisfied.

        Args:
            pending_agents: Set of agent_ids not yet processed
            completed_agents: Set of agent_ids already completed

        Returns:
            List of agent_ids ready to speak (dependencies satisfied)
        """
        ready = []
        for agent_id in pending_agents:
            deps = self.dependencies.get(agent_id, set())
            if deps.issubset(completed_agents):
                ready.append(agent_id)
        return ready

    async def run_questions_async(
        self,
        questions: List[Dict[str, Any]],
        progress_callback: Optional[
            Callable[[int, int, int, Dict[str, Any]], None]
        ] = None,
    ) -> List[Dict[str, Any]]:
        """Run all questions/conversations in parallel.

        All conversations run concurrently. Backend handles request queuing:
        - vLLM: Batches via max_num_seqs
        - Ollama: Internal queue (no true batching)

        Args:
            questions: List of question dictionaries
            progress_callback: Optional callback(completed, total, question_idx, result)
                - completed: Number of conversations finished so far
                - total: Total number of conversations
                - question_idx: Original 0-based index of this question
                - result: The transcript dictionary

        Returns:
            List of transcript dictionaries (in original question order)
        """
        await self._start_engine_if_needed()

        try:
            completed_count = 0
            count_lock = asyncio.Lock()

            async def run_question(idx: int, question: Dict[str, Any]):
                nonlocal completed_count
                result = await self.run_conversation_async(question)
                if progress_callback:
                    async with count_lock:
                        completed_count += 1
                        progress_callback(completed_count, len(questions), idx, result)
                return result

            tasks = [run_question(idx, q) for idx, q in enumerate(questions)]
            results = await asyncio.gather(*tasks)

            return results

        finally:
            await self._stop_engine_if_needed()

    async def run_conversation_async(
        self,
        question: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a single conversation asynchronously.

        Args:
            question: Question dictionary with choices and metadata

        Returns:
            Complete conversation transcript

        Raises:
            MacFairnessError: On validation/generation failures
        """
        transcript_id = str(uuid.uuid4())
        execution_timestamp = datetime.now(timezone.utc)
        conversation_rounds: List[Dict[str, Any]] = []

        # Track token statistics at conversation level
        token_stats = ConversationTokenStats()
        all_validation_errors: List[Dict[str, Any]] = []

        # Track completed messages for dependency resolution
        # Format: "msg_{round_id}_{agent_id}"
        completed_messages: Set[str] = set()

        try:
            # Run conversation rounds
            for round_id in range(self.router.max_rounds):
                if not self.router.should_continue(round_id):
                    break

                round_messages, round_errors = await self._run_round_async(
                    question=question,
                    round_id=round_id,
                    conversation_rounds=conversation_rounds,
                    completed_messages=completed_messages,
                    token_stats=token_stats,
                )

                all_validation_errors.extend(round_errors)

                # Add completed round
                conversation_rounds.append(
                    {
                        "round_id": round_id,
                        "messages": round_messages,
                    }
                )

            # Build successful transcript
            return self.transcript_manager.build_transcript(
                transcript_id=transcript_id,
                question=question,
                conversation_rounds=conversation_rounds,
                config=self.config,
                snapshot_path=self.snapshot_path,
                submission_timestamp=self.submission_timestamp,
                execution_timestamp=execution_timestamp,
                token_stats=token_stats.to_dict(),
                all_validation_errors=all_validation_errors,
                status="succeeded",
            )

        except MacFairnessError as e:
            # Build error transcript with partial data
            return self._build_error_transcript_from_exception(
                transcript_id=transcript_id,
                question=question,
                conversation_rounds=conversation_rounds,
                execution_timestamp=execution_timestamp,
                error=e,
                token_stats=token_stats.to_dict(),
                all_validation_errors=all_validation_errors,
            )

    async def _run_round_async(
        self,
        question: Dict[str, Any],
        round_id: int,
        conversation_rounds: List[Dict[str, Any]],
        completed_messages: Set[str],
        token_stats: ConversationTokenStats,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run a single round with dependency-based speaking order.

        Agents with no dependencies (participants) are batched together.
        Agents with dependencies wait for their prerequisites.

        Args:
            question: Question dictionary
            round_id: Current round number
            conversation_rounds: Previous rounds' messages
            completed_messages: Set of completed message IDs
            token_stats: Token statistics collector

        Returns:
            Tuple of (round_messages, validation_errors)

        Raises:
            MacFairnessError: On generation/validation failures
        """
        round_messages: List[Dict[str, Any]] = []
        round_errors: List[Dict[str, Any]] = []

        # Get all agents for this round
        agent_ids = [ac["agent_id"] for ac in self.agent_defs]
        speaking_order = self.router.get_speaking_order(agent_ids, round_id)

        pending_agents = set(speaking_order)
        round_completed_agents: Set[str] = set()

        # Message index for ordering
        message_index = 0

        while pending_agents:
            # Get agents ready to speak (dependencies satisfied)
            ready_agents = self._get_ready_agents(
                pending_agents, round_completed_agents
            )

            if not ready_agents:
                # This shouldn't happen with valid config
                remaining = list(pending_agents)
                raise MacFairnessError(
                    message=f"Dependency deadlock: agents {remaining} have unsatisfied dependencies",
                    error_code="DEPENDENCY_DEADLOCK",
                    details={
                        "pending_agents": remaining,
                        "completed_agents": list(round_completed_agents),
                        "round_id": round_id,
                    },
                )

            # Run all ready agents in parallel
            tasks = []
            for agent_id in ready_agents:
                task = self._generate_agent_response_async(
                    agent_id=agent_id,
                    question=question,
                    round_id=round_id,
                    conversation_rounds=conversation_rounds,
                    round_messages=round_messages,
                    message_index=message_index,
                )
                tasks.append((agent_id, task))
                message_index += 1

            # Execute all ready agents - errors propagate immediately
            results = await asyncio.gather(*[t[1] for t in tasks])

            # Process results
            for (agent_id, _), (message, prompt_tokens, response_tokens, errors) in zip(
                tasks, results
            ):
                round_messages.append(message)
                token_stats.record(prompt_tokens, response_tokens)
                round_errors.extend(errors)

                # Mark agent as completed
                pending_agents.remove(agent_id)
                round_completed_agents.add(agent_id)

                # Register completed message
                message_id = f"msg_{round_id}_{agent_id}"
                completed_messages.add(message_id)

        return round_messages, round_errors

    async def _generate_agent_response_async(
        self,
        agent_id: str,
        question: Dict[str, Any],
        round_id: int,
        conversation_rounds: List[Dict[str, Any]],
        round_messages: List[Dict[str, Any]],
        message_index: int,
    ) -> Tuple[Dict[str, Any], int, int, List[Dict[str, Any]]]:
        """Generate a single agent's response asynchronously.

        Args:
            agent_id: ID of the agent to generate response for
            question: Question dictionary
            round_id: Current round number
            conversation_rounds: Previous rounds' messages
            round_messages: Messages from current round so far
            message_index: Index for message_id

        Returns:
            Tuple of (message_dict, prompt_tokens, response_tokens, validation_errors)

        Raises:
            MacFairnessError: On generation/validation failures
        """
        agent = self.agents[agent_id]
        agent_config = self.agent_configs_map[agent_id]

        # Get visible messages (includes previous rounds + current round so far)
        visible_messages = self.router.get_visible_messages(
            round_id, conversation_rounds, agent_id
        )

        # Add messages from current round that are visible
        for msg in round_messages:
            if agent_id in msg.get("visible_to", []):
                visible_messages.append(msg)

        prompt = self.prompt_builder.build_full_prompt(
            agent_config,
            question,
            self.identity_reveal_config,
            visible_messages,
            self.agent_configs_map,
        )

        question_id = question.get("question_id", "?")
        debug_print(f"[{question_id}] REQUEST round {round_id} | {agent_id}")

        structured_response, metadata = await self._generate_with_retry_async(
            agent=agent,
            agent_config=agent_config,
            prompt=prompt,
            question=question,
        )

        debug_print(f"[{question_id}] DONE    round {round_id} | {agent_id}")

        # Build identity display
        agent_identity_display = self.prompt_builder.build_agent_identity_display(
            agent_config, self.identity_reveal_config
        )

        # Get visibility list
        agent_ids = [ac["agent_id"] for ac in self.agent_defs]
        visible_to = self.router.get_visibility_list(
            agent_id, agent_config["role"], agent_ids, round_id
        )

        # Generate message_id
        message_id = f"msg_{round_id}_{message_index:03d}"

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

        prompt_tokens = metadata.get("prompt_tokens", 0)
        response_tokens = metadata.get("response_tokens", 0)
        validation_errors = metadata.get("validation_errors", [])

        return message, prompt_tokens, response_tokens, validation_errors

    async def _generate_with_retry_async(
        self,
        agent: Any,
        agent_config: Dict[str, Any],
        prompt: str,
        question: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate response with async retry logic.

        Args:
            agent: Async agent instance
            agent_config: Agent configuration
            prompt: Full prompt string
            question: Question dictionary

        Returns:
            Tuple of (structured_response, metadata)

        Raises:
            MaxRetriesExceededError: On validation failure after max retries
            MacFairnessError: On other errors
        """
        max_retries = self.retry_config["max_retries"]
        answer_match_threshold = self.retry_config["answer_match_threshold"]

        error_collector = ErrorCollector()
        retry_count = 0
        agent_id = agent_config["agent_id"]

        # Track tokens across retries (use last successful values)
        last_prompt_tokens = 0
        last_response_tokens = 0

        for attempt in range(max_retries + 1):
            response_data = await agent.generate(prompt)

            # Extract response data
            exceeded_max_tokens = response_data.get("exceeded_max_tokens", False)
            tokens_generated = response_data.get("tokens_generated", 0)
            tokens_prompt = response_data.get("tokens_prompt", 0)

            # Track last token counts
            last_prompt_tokens = tokens_prompt
            last_response_tokens = tokens_generated

            # Check max tokens exceeded
            if exceeded_max_tokens:
                error = MaxLengthExceededError(
                    agent_id=agent_id,
                    max_tokens=agent_config["max_tokens"],
                    tokens_generated=tokens_generated,
                    truncated=True,
                    attempt=attempt,
                )
                error_collector.add_error(error)
                if attempt < max_retries:
                    retry_count += 1
                    continue
                else:
                    raise MaxRetriesExceededError(
                        agent_id=agent_id,
                        max_retries=max_retries,
                        validation_errors=error_collector.get_summary()["errors"],
                    )

            # Get structured output
            structured_output = response_data.get("structured_output")
            if not structured_output:
                error = MissingStructuredOutputError(
                    agent_id=agent_id,
                    attempt=attempt,
                )
                error_collector.add_error(error)
                if attempt < max_retries:
                    retry_count += 1
                    continue
                else:
                    raise MaxRetriesExceededError(
                        agent_id=agent_id,
                        max_retries=max_retries,
                        validation_errors=error_collector.get_summary()["errors"],
                    )

            # Transform response
            transformed = self._transform_llm_response(
                structured_output, agent_config, question, answer_match_threshold
            )

            # Check answer validity
            answer_match_info = transformed.pop("_answer_match_info", None)
            matched_answer_text = transformed.pop("_matched_answer_text", None)

            if answer_match_info and not matched_answer_text:
                # Answer didn't match any choice
                error = InvalidAnswerError(
                    answer_text=structured_output.get("answer", ""),
                    choices=[c for c in question.get("choices", [])],
                    match_info=answer_match_info,
                    attempt=attempt,
                )
                error_collector.add_error(error)
                if attempt < max_retries:
                    retry_count += 1
                    continue
                else:
                    raise MaxRetriesExceededError(
                        agent_id=agent_id,
                        max_retries=max_retries,
                        validation_errors=error_collector.get_summary()["errors"],
                    )

            # Build metadata
            metadata: Dict[str, Any] = {
                "retry_count": retry_count,
                "prompt_tokens": last_prompt_tokens,
                "response_tokens": last_response_tokens,
            }

            if matched_answer_text:
                metadata["matched_answer_text"] = matched_answer_text

            if answer_match_info:
                metadata["answer_match_info"] = answer_match_info

            if error_collector.has_errors():
                metadata["validation_errors"] = error_collector.get_summary()["errors"]

            # Add batching metrics if available (vLLM)
            if "batch_size" in response_data:
                metadata["batch_size"] = response_data["batch_size"]

            return transformed, metadata

        # Should not reach here
        raise RuntimeError("Unexpected end of retry loop")

    def _transform_llm_response(
        self,
        raw_response: Dict[str, Any],
        agent_config: Dict[str, Any],
        question: Dict[str, Any],
        answer_match_threshold: float,
    ) -> Dict[str, Any]:
        """Transform LLM response format to schema format."""
        from src.utils.answer_matcher import FlexibleAnswerMatcher

        role = agent_config["role"]
        question_type = question.get("question_type", "multiple_choice")
        is_choice_question = question_type in ["binary", "multiple_choice"]

        transformed: Dict[str, Any] = {
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
                if (
                    match_details
                    and match_details[0].get("match_score") is not None
                    and answer_match_threshold is not None
                    and match_details[0]["match_score"] >= answer_match_threshold
                ):
                    transformed["opinion"] = match_details[0]["id"]
                    transformed["_matched_answer_text"] = match_details[0]["text"]
                    transformed["_answer_match_info"] = match_result
                else:
                    transformed["opinion"] = answer_text
                    transformed["_answer_match_info"] = match_result
            else:
                transformed["opinion"] = answer_text
        else:
            transformed["opinion"] = raw_response.get(
                "opinion", raw_response.get("answer", "")
            )

        return transformed

    def _build_error_transcript_from_exception(
        self,
        transcript_id: str,
        question: Dict[str, Any],
        conversation_rounds: List[Dict[str, Any]],
        execution_timestamp: datetime,
        error: MacFairnessError,
        token_stats: Dict[str, Any],
        all_validation_errors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build transcript with error information from MacFairnessError."""
        details = error.details if hasattr(error, "details") else {}

        if "validation_errors" in details:
            all_validation_errors.extend(details["validation_errors"])

        # Determine status
        total_messages = sum(len(r["messages"]) for r in conversation_rounds)
        status = "partial" if total_messages > 0 else "failed"

        error_info = {
            "error_class": error.__class__.__name__,
            "error_code": error.error_code
            if hasattr(error, "error_code")
            else "MAC_FAIRNESS_ERROR",
            "message": str(error),
            "details": details,
        }

        return self.transcript_manager.build_transcript(
            transcript_id=transcript_id,
            question=question,
            conversation_rounds=conversation_rounds,
            config=self.config,
            snapshot_path=self.snapshot_path,
            submission_timestamp=self.submission_timestamp,
            execution_timestamp=execution_timestamp,
            token_stats=token_stats,
            all_validation_errors=all_validation_errors,
            status=status,
            error_info=error_info,
        )

    async def _start_engine_if_needed(self) -> None:
        """Start backend engine if needed (e.g., vLLM batched engine)."""
        for agent in self.agents.values():
            if hasattr(agent, "start_engine"):
                await agent.start_engine()
                return

    async def _stop_engine_if_needed(self) -> None:
        """Stop backend engine if it was started."""
        for agent in self.agents.values():
            if hasattr(agent, "stop_engine"):
                await agent.stop_engine()
                return
