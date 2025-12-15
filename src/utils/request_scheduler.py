"""Request-level scheduler for GPU-efficient async conversation processing."""

import heapq
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import asyncio

from src.utils.errors import (
    MacFairnessError,
    MaxRetriesExceededError,
    MaxLengthExceededError,
    MissingStructuredOutputError,
    MissingConfigSectionError,
    InvalidAnswerError,
    ErrorCollector,
    UnexpectedError,
)
from src.utils.logging import is_debug_enabled, is_live_status_enabled

# Global reference to live display for debug print coordination
_live_display_instance: "LiveStatusDisplay | None" = None


def _debug_print(msg: str) -> None:
    """Print debug message with live display coordination.

    Wraps the centralized debug check but adds coordination with
    the live status display to avoid corrupting the terminal output.
    """
    if is_debug_enabled():
        if _live_display_instance and _live_display_instance.initialized:
            _live_display_instance.pause_for_print()
        print(f"[DEBUG] {msg}")
        if _live_display_instance and _live_display_instance.initialized:
            _live_display_instance.resume_after_print()


class LiveStatusDisplay:
    """htop-like live status display for scheduler pools.

    Uses ANSI escape codes to update fixed lines in terminal.
    Enable with MAC_FAIRNESS_LIVE_STATUS=1 environment variable.

    Displays Pre-Departure and Pending pools side by side in 2 columns.
    """

    # ANSI escape codes
    CLEAR_LINE = "\033[2K"
    MOVE_UP = "\033[{}A"
    MOVE_DOWN = "\033[{}B"
    HIDE_CURSOR = "\033[?25l"
    SHOW_CURSOR = "\033[?25h"
    SAVE_CURSOR = "\033[s"
    RESTORE_CURSOR = "\033[u"

    # Display configuration
    DISPLAY_LINES = 40  # Total terminal lines to reserve
    QUEUE_ROWS = 35  # Rows for queue items (after header)
    COLUMN_WIDTH = 50  # Width of each column
    TOTAL_WIDTH = COLUMN_WIDTH * 2 + 3  # +3 for separator

    def __init__(self):
        """Initialize live status display."""
        self.display_lines = self.DISPLAY_LINES
        self.initialized = False
        # Check at runtime so env var can be set after import
        self.enabled = is_live_status_enabled()
        # agent_id -> short model name mapping (set by scheduler)
        self.agent_model_names: Dict[str, str] = {}

    def initialize(self) -> None:
        """Reserve lines in terminal for live display."""
        global _live_display_instance
        if not self.enabled or self.initialized:
            return
        # Print empty lines to reserve space
        print("\n" * self.display_lines, end="")
        print(self.HIDE_CURSOR, end="", flush=True)
        self.initialized = True
        _live_display_instance = self

    def cleanup(self) -> None:
        """Restore terminal state."""
        global _live_display_instance
        if not self.enabled or not self.initialized:
            return
        print(self.SHOW_CURSOR, end="", flush=True)
        self.initialized = False
        _live_display_instance = None

    def pause_for_print(self) -> None:
        """Pause display before a debug print (move cursor below display area)."""
        if not self.enabled or not self.initialized:
            return
        # Move cursor to bottom of display area and show cursor
        print(self.SHOW_CURSOR, end="", flush=True)

    def resume_after_print(self) -> None:
        """Resume display after a debug print."""
        if not self.enabled or not self.initialized:
            return
        # Hide cursor again, display will update on next cycle
        print(self.HIDE_CURSOR, end="", flush=True)
        # Add newlines to push the display area down to accommodate the debug line
        print("\n" * self.display_lines, end="", flush=True)

    def _format_request_short(self, idx: int, req: "Request") -> str:
        """Format a single request for display (short format)."""
        q_id = (
            req.question.get("question_id", f"q_{req.conversation_id}")
            if req.question
            else f"q_{req.conversation_id}"
        )
        # Get short model name if available
        model_name = self.agent_model_names.get(req.agent_id, "")
        model_suffix = f" ({model_name})" if model_name else ""
        # Format: "  1. [q_0] round 0 agent_id (model)"
        return f" {idx:3d}. [{q_id}] r{req.round_id} {req.agent_id}{model_suffix}"

    def _format_in_flight_status(
        self,
        model_in_flight: Dict[str, int],
        model_max_num_seqs: Dict[str, int],
    ) -> str:
        """Format in-flight status for display.

        For single model: "In-Flight: 5/32"
        For multi model: "In-Flight: model_a 5/32, model_b 3/16"
        """
        if len(model_in_flight) == 1:
            model_name = list(model_in_flight.keys())[0]
            in_flight = model_in_flight[model_name]
            max_seqs = model_max_num_seqs[model_name]
            return f"In-Flight: {in_flight}/{max_seqs}"
        else:
            # Multi-model: show each model's status
            parts = []
            for model_name in sorted(model_in_flight.keys()):
                in_flight = model_in_flight[model_name]
                max_seqs = model_max_num_seqs[model_name]
                # Shorten model name for display (take last part after /)
                short_name = (
                    model_name.split("/")[-1] if "/" in model_name else model_name
                )
                # Truncate if still too long
                if len(short_name) > 15:
                    short_name = short_name[:12] + "..."
                parts.append(f"{short_name}:{in_flight}/{max_seqs}")
            return "In-Flight: " + ", ".join(parts)

    def update(
        self,
        model_in_flight: Dict[str, int],
        model_max_num_seqs: Dict[str, int],
        pre_departure_pool: list,
        pending_pool: dict,
        completed: int,
        total: int,
    ) -> None:
        """Update the live display with current pool status.

        Shows Pre-Departure (left) and Pending (right) pools side by side.
        Supports multi-model in-flight tracking.

        Args:
            model_in_flight: Dict of model_name -> current in-flight count
            model_max_num_seqs: Dict of model_name -> max_num_seqs
            pre_departure_pool: List of PrioritizedRequest in pre-departure
            pending_pool: Dict of request_key -> Request in pending
            completed: Number of completed conversations
            total: Total number of conversations
        """
        if not self.enabled:
            return

        if not self.initialized:
            self.initialize()

        lines = []

        # Format in-flight status (handles single and multi-model)
        in_flight_status = self._format_in_flight_status(
            model_in_flight, model_max_num_seqs
        )

        # Header (5 lines - removed GPU line)
        lines.append(f"{'═' * self.TOTAL_WIDTH}")
        lines.append(
            f" Progress: {completed}/{total} conversations | "
            f"{in_flight_status} | "
            f"Pre-Dep: {len(pre_departure_pool)} | "
            f"Pending: {len(pending_pool)}"
        )
        lines.append(f"{'─' * self.TOTAL_WIDTH}")
        lines.append(
            f" {'PRE-DEPARTURE (ready to fly)':<{self.COLUMN_WIDTH - 1}}"
            f" │ {'PENDING (blocked on deps)':<{self.COLUMN_WIDTH - 1}}"
        )
        lines.append(f"{'─' * self.TOTAL_WIDTH}")

        # Get sorted pre-departure items
        sorted_pre_dep = sorted(pre_departure_pool)[: self.QUEUE_ROWS]

        # Get pending items (convert dict values to list)
        pending_items = list(pending_pool.values())[: self.QUEUE_ROWS]

        # Build rows with 2 columns
        for row_idx in range(self.QUEUE_ROWS):
            # Column 1: Pre-Departure
            if row_idx < len(sorted_pre_dep):
                pr = sorted_pre_dep[row_idx]
                col1 = self._format_request_short(row_idx + 1, pr.request)
            else:
                col1 = ""

            # Column 2: Pending
            if row_idx < len(pending_items):
                req = pending_items[row_idx]
                col2 = self._format_request_short(row_idx + 1, req)
            else:
                col2 = ""

            # Combine columns
            line = f"{col1:<{self.COLUMN_WIDTH}} │ {col2:<{self.COLUMN_WIDTH}}"
            lines.append(line)

        # Footer
        lines.append(f"{'═' * self.TOTAL_WIDTH}")

        # Pad to fixed height
        while len(lines) < self.display_lines:
            lines.append("")

        # Move cursor up and overwrite
        # Use save/restore cursor position for more reliable updates
        output = self.MOVE_UP.format(self.display_lines)
        for line in lines[: self.display_lines]:
            # Pad line to full width to overwrite any previous content
            padded_line = line[: self.TOTAL_WIDTH].ljust(self.TOTAL_WIDTH)
            output += self.CLEAR_LINE + padded_line + "\n"

        print(output, end="", flush=True)


@dataclass
class ConversationState:
    """Tracks the state of a single conversation."""

    conversation_id: int
    question: Dict[str, Any]
    rounds_completed: int = 0
    current_round_completed_agents: Set[str] = field(default_factory=set)
    conversation_rounds: List[Dict[str, Any]] = field(default_factory=list)
    current_round_messages: List[Dict[str, Any]] = field(default_factory=list)
    completed_messages: Set[str] = field(default_factory=set)
    token_stats: Any = None  # ConversationTokenStats instance
    validation_errors: List[Dict[str, Any]] = field(default_factory=list)
    is_complete: bool = False
    is_finalized: bool = False  # Guard against double finalization
    error: Optional[MacFairnessError] = None
    transcript_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    execution_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass(order=True)
class PrioritizedRequest:
    """A request with priority for the pre-departure pool.

    Priority ordering (lower = higher priority):
    1. is_reprompt: 0 if reprompt, 1 otherwise (reprompts first)
    2. -rounds_completed: More progress = higher priority (finish started conversations)
    3. conversation_id: Lower ID = earlier conversations (FIFO tiebreaker)
    4. round_id: Lower round first within same conversation
    """

    priority: Tuple[int, int, int, int] = field(compare=True)
    request: "Request" = field(compare=False)

    @classmethod
    def create(cls, request: "Request") -> "PrioritizedRequest":
        """Create a prioritized request with correct priority tuple."""
        priority = (
            0 if request.is_reprompt else 1,  # Re-prompts first
            -request.rounds_completed,  # More progress = higher priority (finish started convos)
            request.conversation_id,  # Earlier conversations (FIFO tiebreaker)
            request.round_id,  # Lower round first within same conversation
        )
        return cls(priority=priority, request=request)


@dataclass
class Request:
    """Represents a single agent generation request."""

    conversation_id: int
    round_id: int
    agent_id: str
    is_reprompt: bool = False
    reprompt_attempt: int = 0
    rounds_completed: int = 0  # How many rounds this conversation has completed

    # These are set when the request is ready to execute
    prompt: Optional[str] = None
    agent: Optional[Any] = None
    agent_config: Optional[Dict[str, Any]] = None
    question: Optional[Dict[str, Any]] = None
    message_index: int = 0


@dataclass
class RequestResult:
    """Result of a completed request."""

    request: Request
    success: bool
    message: Optional[Dict[str, Any]] = None
    prompt_tokens: int = 0
    response_tokens: int = 0
    validation_errors: List[Dict[str, Any]] = field(default_factory=list)
    needs_reprompt: bool = False
    error: Optional[Exception] = None
    structured_response: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class ConversationTokenStats:
    """Token statistics for a single conversation."""

    def __init__(self):
        self.prompt_tokens: List[int] = []
        self.response_tokens: List[int] = []

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


class RequestScheduler:
    """Request-level scheduler for GPU-efficient conversation processing.

    Maintains three pools:
    - Pending Pool: Requests blocked on dependencies
    - Pre-Departure Pool: Ready requests, prioritized (unified across models)
    - In-Flight: Per-model semaphore control (each model has its own max_num_seqs)

    Priority in Pre-Departure Pool:
    1. Re-prompts (highest)
    2. More rounds completed (finish conversations with progress first)
    3. Earlier conversations (FIFO tiebreaker)
    4. Lower round_id within same conversation

    Multi-Model Support:
    - Each model gets its own semaphore based on its max_num_seqs
    - Per-model in-flight tracking for GPU utilization monitoring
    - Unified queuing pools (readiness check is model-agnostic)
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
        effective_backend_config: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Initialize the request scheduler.

        Args:
            agents: Dictionary of agent_id -> async agent instance
            router: Router instance for conversation flow
            prompt_builder: Prompt builder instance
            config: Full experiment configuration
            transcript_manager: Transcript manager for building/saving
            snapshot_path: Path to config snapshot
            submission_timestamp: When experiment was submitted
            effective_backend_config: Optional dict of effective config per model
                keyed by model_path (includes auto-calculated values like max_num_seqs
                from vLLM based on actual KV cache availability)
        """
        self.effective_backend_config = effective_backend_config or {}
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
        self.retry_config = config["retry_config"]
        self.model_definitions = config.get("model_definitions", {})

        # Build agent_id -> model_id mapping
        self.agent_to_model: Dict[str, str] = {}
        for agent_config in self.agent_defs:
            agent_id = agent_config["agent_id"]
            model_name = agent_config["model"]  # References model_definitions key
            self.agent_to_model[agent_id] = model_name

        # Build per-model semaphores and in-flight tracking
        self.model_semaphores: Dict[str, asyncio.Semaphore] = {}
        self.model_max_num_seqs: Dict[str, int] = {}
        self.model_in_flight: Dict[str, int] = {}
        self._init_model_semaphores()

        # Build speak-after dependency graph
        self.dependencies: Dict[str, Set[str]] = {}
        for agent_config in self.agent_defs:
            agent_id = agent_config["agent_id"]
            role_config = agent_config.get("role_specific_config", {})
            speak_after = role_config.get("speak_after_within_round", [])
            self.dependencies[agent_id] = set(speak_after)

        # Pools (unified across models - readiness is model-agnostic)
        self.pending_pool: Dict[
            str, Request
        ] = {}  # key: f"{conv_id}_{round_id}_{agent_id}"
        self.pre_departure_pool: List[PrioritizedRequest] = []  # heapq

        # Conversation states
        self.conversation_states: Dict[int, ConversationState] = {}

        # Results tracking
        self.completed_transcripts: List[Dict[str, Any]] = []
        self.progress_callback: Optional[Callable] = None
        self.total_questions: int = 0

        # Live status display
        self.live_display = LiveStatusDisplay()
        self._init_display_model_names()

    def _init_display_model_names(self) -> None:
        """Initialize short model names for display.

        Creates a mapping from agent_id to a short model name for the live display.
        Uses display_name from model_definitions.
        """
        for agent_id, model_key in self.agent_to_model.items():
            display_name = model_key

            self.live_display.agent_model_names[agent_id] = display_name

    def _init_model_semaphores(self) -> None:
        """Initialize per-model semaphores based on configuration.

        Each model gets its own semaphore sized to its max_num_seqs.
        For vLLM, max_num_seqs is mandatory in config, but may be overridden
        by effective_backend_config (actual value from vLLM engine based on KV cache).
        For Ollama, max_num_seqs is optional (defaults to 32, no true batching anyway).

        Raises:
            MissingConfigSectionError: If max_num_seqs not specified for vLLM backend
        """
        # Get unique models used by agents
        used_models = set(self.agent_to_model.values())

        for model_name in used_models:
            model_config = self.model_definitions.get(model_name, {})
            backend = model_config.get("backend")
            model_path = model_config.get("model_path", model_name)

            if backend == "vllm":
                vllm_config = model_config.get("vllm_config")
                config_max_num_seqs = vllm_config.get("max_num_seqs_upper_bound")
                if config_max_num_seqs is None:
                    raise MissingConfigSectionError(
                        f"model_definitions.{model_name}.vllm_config.max_num_seqs_upper_bound "
                        "(required for vLLM request scheduling)"
                    )

                # Check for effective value from vLLM engine (may be lower due to KV cache)
                effective_config = self.effective_backend_config.get(model_path, {})
                effective_max_num_seqs = effective_config.get("max_num_seqs")

                if effective_max_num_seqs and effective_max_num_seqs < config_max_num_seqs:
                    max_num_seqs = effective_max_num_seqs
                    _debug_print(
                        f"Model '{model_name}': using effective max_num_seqs={max_num_seqs} "
                        f"(config was {config_max_num_seqs}, limited by KV cache)"
                    )
                else:
                    max_num_seqs = config_max_num_seqs

            elif backend == "ollama":
                # Ollama or other backends - optional, default to 32
                max_num_seqs = model_config.get("max_num_seqs", 32)
            else:
                raise MissingConfigSectionError(
                    f"model_definitions.{model_name}.backend missing/unsupported"
                )

            self.model_semaphores[model_name] = asyncio.Semaphore(max_num_seqs)
            self.model_max_num_seqs[model_name] = max_num_seqs
            self.model_in_flight[model_name] = 0

            _debug_print(
                f"Model '{model_name}' ({backend}): max_num_seqs={max_num_seqs}"
            )

    def _get_model_for_agent(self, agent_id: str) -> str:
        """Get the model name for a given agent."""
        return self.agent_to_model.get(agent_id, "unknown")

    def get_total_in_flight(self) -> int:
        """Get total in-flight count across all models."""
        return sum(self.model_in_flight.values())

    def get_total_max_num_seqs(self) -> int:
        """Get total max_num_seqs across all models."""
        return sum(self.model_max_num_seqs.values())

    def _get_request_key(self, request: Request) -> str:
        """Get unique key for a request."""
        return f"{request.conversation_id}_{request.round_id}_{request.agent_id}"

    def _is_agent_ready(self, agent_id: str, round_completed_agents: Set[str]) -> bool:
        """Check if agent's dependencies are satisfied."""
        deps = self.dependencies.get(agent_id, set())
        return deps.issubset(round_completed_agents)

    def _check_pending_for_readiness(self) -> None:
        """Move requests from Pending to Pre-Departure if dependencies satisfied."""
        to_remove = []

        for key, request in self.pending_pool.items():
            conv_state = self.conversation_states[request.conversation_id]

            # Check if this request's dependencies are satisfied
            if self._is_agent_ready(
                request.agent_id, conv_state.current_round_completed_agents
            ):
                # Prepare the request with prompt and context
                self._prepare_request(request, conv_state)

                # Add to pre-departure pool with priority
                prioritized = PrioritizedRequest.create(request)
                heapq.heappush(self.pre_departure_pool, prioritized)
                to_remove.append(key)

        for key in to_remove:
            del self.pending_pool[key]

    def _prepare_request(self, request: Request, conv_state: ConversationState) -> None:
        """Prepare a request with prompt and context before submission."""
        agent_id = request.agent_id
        agent = self.agents[agent_id]
        agent_config = self.agent_configs_map[agent_id]
        question = conv_state.question

        # Get visible messages
        visible_messages = self.router.get_visible_messages(
            request.round_id, conv_state.conversation_rounds, agent_id
        )

        # Add messages from current round that are visible
        for msg in conv_state.current_round_messages:
            if agent_id in msg.get("visible_to", []):
                visible_messages.append(msg)

        prompt = self.prompt_builder.build_full_prompt(
            agent_config,
            question,
            self.identity_reveal_config,
            visible_messages,
            self.agent_configs_map,
        )

        request.prompt = prompt
        request.agent = agent
        request.agent_config = agent_config
        request.question = question
        request.rounds_completed = conv_state.rounds_completed

    def _initialize_conversation(
        self, conversation_id: int, question: Dict[str, Any]
    ) -> None:
        """Initialize a new conversation and create its round 0 requests."""
        conv_state = ConversationState(
            conversation_id=conversation_id,
            question=question,
            token_stats=ConversationTokenStats(),
        )
        self.conversation_states[conversation_id] = conv_state

        # Create requests for round 0
        self._create_round_requests(conv_state, round_id=0)

    def _create_round_requests(
        self, conv_state: ConversationState, round_id: int
    ) -> None:
        """Create requests for all agents in a round."""
        agent_ids = [ac["agent_id"] for ac in self.agent_defs]
        speaking_order = self.router.get_speaking_order(agent_ids, round_id)

        for idx, agent_id in enumerate(speaking_order):
            request = Request(
                conversation_id=conv_state.conversation_id,
                round_id=round_id,
                agent_id=agent_id,
                rounds_completed=conv_state.rounds_completed,
                message_index=idx,
            )

            # Check if immediately ready (no dependencies)
            if self._is_agent_ready(
                agent_id, conv_state.current_round_completed_agents
            ):
                self._prepare_request(request, conv_state)
                prioritized = PrioritizedRequest.create(request)
                heapq.heappush(self.pre_departure_pool, prioritized)
            else:
                # Add to pending pool
                key = self._get_request_key(request)
                self.pending_pool[key] = request

    async def _execute_request(self, request: Request) -> RequestResult:
        """Execute a single request (with retry logic)."""
        max_retries = self.retry_config["max_retries"]
        answer_match_threshold = self.retry_config["answer_match_threshold"]

        error_collector = ErrorCollector()
        retry_count = 0
        agent = request.agent
        agent_config = request.agent_config
        prompt = request.prompt
        question = request.question
        agent_id = request.agent_id

        last_prompt_tokens = 0
        last_response_tokens = 0

        for attempt in range(max_retries + 1):
            try:
                response_data = await agent.generate(prompt)
            except Exception as e:
                return RequestResult(
                    request=request,
                    success=False,
                    error=e,
                )

            exceeded_max_tokens = response_data.get("exceeded_max_tokens", False)
            tokens_generated = response_data.get("tokens_generated", 0)
            tokens_prompt = response_data.get("tokens_prompt", 0)

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
                    return RequestResult(
                        request=request,
                        success=False,
                        needs_reprompt=False,
                        error=MaxRetriesExceededError(
                            agent_id=agent_id,
                            max_retries=max_retries,
                            validation_errors=error_collector.get_summary()["errors"],
                        ),
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
                    return RequestResult(
                        request=request,
                        success=False,
                        needs_reprompt=False,
                        error=MaxRetriesExceededError(
                            agent_id=agent_id,
                            max_retries=max_retries,
                            validation_errors=error_collector.get_summary()["errors"],
                        ),
                        validation_errors=error_collector.get_summary()["errors"],
                    )

            # Transform response
            transformed = self._transform_llm_response(
                structured_output, agent_config, question, answer_match_threshold
            )

            answer_match_info = transformed.pop("_answer_match_info", None)
            matched_answer_text = transformed.pop("_matched_answer_text", None)

            if answer_match_info and not matched_answer_text:
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
                    return RequestResult(
                        request=request,
                        success=False,
                        needs_reprompt=False,
                        error=MaxRetriesExceededError(
                            agent_id=agent_id,
                            max_retries=max_retries,
                            validation_errors=error_collector.get_summary()["errors"],
                        ),
                        validation_errors=error_collector.get_summary()["errors"],
                    )

            # Success - build metadata
            metadata: Dict[str, Any] = {
                "retry_count": retry_count,
                "prompt_tokens": last_prompt_tokens,
                "response_tokens": last_response_tokens,
            }

            # Debug-only fields: prompt and answer_match_info
            if is_debug_enabled():
                metadata["prompt"] = prompt
                if answer_match_info:
                    metadata["answer_match_info"] = answer_match_info

            if matched_answer_text:
                metadata["matched_answer_text"] = matched_answer_text
            if error_collector.has_errors():
                metadata["validation_errors"] = error_collector.get_summary()["errors"]

            return RequestResult(
                request=request,
                success=True,
                structured_response=transformed,
                metadata=metadata,
                prompt_tokens=last_prompt_tokens,
                response_tokens=last_response_tokens,
                validation_errors=error_collector.get_summary()["errors"]
                if error_collector.has_errors()
                else [],
            )

        # Should not reach here
        return RequestResult(
            request=request,
            success=False,
            error=RuntimeError("Unexpected end of retry loop"),
        )

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

    def _remove_conversation_requests(self, conversation_id: int) -> None:
        """Remove all pending and pre-departure requests for a failed conversation.

        Called when a conversation fails to clean up queued requests.
        """
        # Remove from pending pool
        keys_to_remove = [
            key
            for key, req in self.pending_pool.items()
            if req.conversation_id == conversation_id
        ]
        for key in keys_to_remove:
            del self.pending_pool[key]

        # Remove from pre-departure pool (rebuild without this conversation's requests)
        self.pre_departure_pool = [
            pr
            for pr in self.pre_departure_pool
            if pr.request.conversation_id != conversation_id
        ]
        heapq.heapify(self.pre_departure_pool)

    async def _on_request_complete(self, result: RequestResult) -> None:
        """Handle request completion - update state and check readiness.

        This method performs all state updates and dependency resolution
        synchronously (before any await), then awaits the async finalization
        if the conversation is complete. This ensures that:
        1. Dependent requests are unblocked immediately (no waiting for writes)
        2. File I/O happens in thread pool without blocking event loop
        """
        request = result.request
        conv_state = self.conversation_states[request.conversation_id]

        if not result.success:
            # Handle error - mark conversation as failed
            conv_state.error = result.error
            conv_state.is_complete = True
            # Add validation errors from the failed request
            conv_state.validation_errors.extend(result.validation_errors)
            # Remove any queued requests for this failed conversation
            self._remove_conversation_requests(request.conversation_id)
            # Check pending pool before async finalization
            self._check_pending_for_readiness()
            # Finalize with async write (other coroutines can run during write)
            await self._finalize_conversation(conv_state, error=result.error)
            return

        # Build message
        agent_config = request.agent_config
        agent_id = request.agent_id

        agent_identity_display = self.prompt_builder.build_agent_identity_display(
            agent_config, self.identity_reveal_config
        )

        agent_ids = [ac["agent_id"] for ac in self.agent_defs]
        visible_to = self.router.get_visibility_list(
            agent_id, agent_config["role"], agent_ids, request.round_id
        )

        message_id = f"msg_{request.round_id}_{request.message_index:03d}"

        message = {
            "message_id": message_id,
            "agent_id": agent_id,
            "agent_role": agent_config["role"],
            "round_id": request.round_id,
            "agent_identity_display": agent_identity_display,
            "structured_response": result.structured_response,
            "visible_to": visible_to,
            "message_metadata": result.metadata,
        }

        # Update conversation state
        conv_state.current_round_messages.append(message)
        conv_state.current_round_completed_agents.add(agent_id)
        conv_state.completed_messages.add(f"msg_{request.round_id}_{agent_id}")
        conv_state.token_stats.record(result.prompt_tokens, result.response_tokens)
        conv_state.validation_errors.extend(result.validation_errors)

        # Check if round is complete
        expected_agents = set(
            self.router.get_speaking_order(
                [ac["agent_id"] for ac in self.agent_defs], request.round_id
            )
        )

        # Track if we need to finalize after all sync work is done
        should_finalize = False

        if conv_state.current_round_completed_agents == expected_agents:
            # Round complete - finalize it
            conv_state.conversation_rounds.append(
                {
                    "round_id": request.round_id,
                    "messages": conv_state.current_round_messages,
                }
            )
            conv_state.rounds_completed += 1
            conv_state.current_round_messages = []
            conv_state.current_round_completed_agents = set()

            # Check if conversation is complete
            next_round_id = request.round_id + 1
            if (
                next_round_id >= self.router.max_rounds
                or not self.router.should_continue(next_round_id)
            ):
                conv_state.is_complete = True
                should_finalize = True
            else:
                # Start next round
                self._create_round_requests(conv_state, next_round_id)

        # Check pending pool for newly ready requests (sync, before any await)
        # This ensures dependent requests are unblocked immediately
        self._check_pending_for_readiness()

        # Now do async finalization if needed (file I/O in thread pool)
        if should_finalize:
            await self._finalize_conversation(conv_state)

    async def _finalize_conversation(
        self,
        conv_state: ConversationState,
        error: Optional[Exception] = None,
    ) -> None:
        """Finalize a conversation and save transcript.

        This method is idempotent - calling it multiple times for the same
        conversation has no effect after the first call.

        The progress callback (which performs blocking file I/O) is executed
        in a thread pool via asyncio.to_thread() to avoid blocking the event
        loop. This ensures GPU dispatch continues while file writes complete.
        """
        # Guard against double finalization
        if conv_state.is_finalized:
            return
        conv_state.is_finalized = True

        if error:
            # Build error transcript
            error_info = {
                "error_class": error.__class__.__name__,
                "error_code": getattr(error, "error_code", "UNKNOWN_ERROR"),
                "message": str(error),
                "details": getattr(error, "details", {}),
            }

            total_messages = sum(
                len(r["messages"]) for r in conv_state.conversation_rounds
            )
            status = "partial" if total_messages > 0 else "failed"

            transcript = self.transcript_manager.build_transcript(
                transcript_id=conv_state.transcript_id,
                question=conv_state.question,
                conversation_rounds=conv_state.conversation_rounds,
                config=self.config,
                snapshot_path=self.snapshot_path,
                submission_timestamp=self.submission_timestamp,
                execution_timestamp=conv_state.execution_timestamp,
                token_stats=conv_state.token_stats.to_dict(),
                all_validation_errors=conv_state.validation_errors,
                status=status,
                error_info=error_info,
            )
        else:
            # Build success transcript
            transcript = self.transcript_manager.build_transcript(
                transcript_id=conv_state.transcript_id,
                question=conv_state.question,
                conversation_rounds=conv_state.conversation_rounds,
                config=self.config,
                snapshot_path=self.snapshot_path,
                submission_timestamp=self.submission_timestamp,
                execution_timestamp=conv_state.execution_timestamp,
                token_stats=conv_state.token_stats.to_dict(),
                all_validation_errors=conv_state.validation_errors,
                status="succeeded",
            )

        self.completed_transcripts.append(transcript)

        # Call progress callback in thread pool to avoid blocking event loop.
        # The callback performs blocking file I/O (transcript save, manifest
        # update with fcntl.flock, index.jsonl append). Running in a thread
        # allows other coroutines (especially GPU dispatch) to continue.
        if self.progress_callback:
            await asyncio.to_thread(
                self.progress_callback,
                len(self.completed_transcripts),
                self.total_questions,
                conv_state.conversation_id,
                transcript,
            )

    async def _execute_with_semaphore(self, request: Request) -> None:
        """Execute a single request with model-specific semaphore control.

        The semaphore is released immediately after GPU work completes, before
        calling _on_request_complete. This decouples GPU scheduling from file I/O:
        - Semaphore slot becomes available for next GPU request immediately
        - File writes (in _on_request_complete) happen in thread pool
        - Event loop stays responsive, GPU stays saturated

        Note: model_in_flight is managed by the caller (_scheduler_loop) to ensure
        accurate capacity tracking before task creation.
        """
        # Get model-specific semaphore
        model_name = self._get_model_for_agent(request.agent_id)
        semaphore = self.model_semaphores[model_name]

        result = None
        try:
            # GPU work under semaphore
            async with semaphore:
                # Check if this conversation already failed
                conv_state = self.conversation_states[request.conversation_id]
                if conv_state.is_complete:
                    return

                result = await self._execute_request(request)
            # Semaphore RELEASED here - slot available for next GPU request

            # State updates and file I/O happen outside semaphore
            # _on_request_complete is async; file writes run in thread pool
            if result is not None:
                await self._on_request_complete(result)

        except (SystemExit, KeyboardInterrupt):
            # Suppress shutdown exceptions to avoid noisy "Task exception was never
            # retrieved" warnings on Ctrl+C. Resources (GPU, locks, memory) are
            # automatically released by the OS when the process exits.
            pass
        except Exception as e:
            # Handle unexpected exceptions to prevent conversation from getting stuck.
            # Without this, unhandled exceptions cause the task to complete without
            # marking the conversation as complete, leaving it orphaned with no requests
            # in any queue, causing the scheduler loop to hang indefinitely.
            conv_state = self.conversation_states[request.conversation_id]
            if not conv_state.is_complete:
                # Wrap in UnexpectedError for consistent error handling
                question_id = (
                    conv_state.question.get("question_id")
                    if conv_state.question
                    else None
                )
                wrapped_error = UnexpectedError(
                    original_error=e,
                    context="request execution",
                    question_id=question_id,
                )
                conv_state.is_complete = True
                self._remove_conversation_requests(request.conversation_id)
                # Async finalization - file writes in thread pool
                await self._finalize_conversation(conv_state, error=wrapped_error)
                _debug_print(
                    f"Unexpected error in conversation {request.conversation_id}: "
                    f"{type(e).__name__}: {e}"
                )
        finally:
            # Decrement in-flight count when done (increment happens in scheduler loop)
            self.model_in_flight[model_name] -= 1

    def _has_semaphore_capacity(self, model_name: str) -> bool:
        """Check if a model's semaphore has capacity without acquiring it."""
        return self.model_in_flight[model_name] < self.model_max_num_seqs[model_name]

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop - dispatches requests as they become ready.

        Only dispatches requests when the target model has semaphore capacity.
        This keeps requests in pre-departure pool until they can actually execute.

        Note: Shutdown is handled via immediate sys.exit in signal handler.
        In-flight requests will be aborted, manifests preserved for resume.
        """
        active_tasks: Set[asyncio.Task] = set()

        while True:
            # Update live display
            self.live_display.update(
                model_in_flight=self.model_in_flight,
                model_max_num_seqs=self.model_max_num_seqs,
                pre_departure_pool=self.pre_departure_pool,
                pending_pool=self.pending_pool,
                completed=len(self.completed_transcripts),
                total=self.total_questions,
            )

            # Check termination condition
            all_complete = self.conversation_states and all(
                cs.is_complete for cs in self.conversation_states.values()
            )
            if all_complete:
                break

            # Dispatch requests only when their model has capacity
            # This keeps requests visible in pre-departure until they can execute
            dispatched_any = True
            while dispatched_any and self.pre_departure_pool:
                dispatched_any = False

                # Peek at highest priority request
                prioritized = self.pre_departure_pool[0]
                request = prioritized.request

                # Skip if conversation already complete
                conv_state = self.conversation_states[request.conversation_id]
                if conv_state.is_complete:
                    heapq.heappop(self.pre_departure_pool)
                    dispatched_any = True
                    continue

                # Check if model has capacity
                model_name = self._get_model_for_agent(request.agent_id)
                if self._has_semaphore_capacity(model_name):
                    heapq.heappop(self.pre_departure_pool)
                    # Increment in-flight BEFORE creating task to ensure accurate capacity check
                    self.model_in_flight[model_name] += 1
                    task = asyncio.create_task(self._execute_with_semaphore(request))
                    active_tasks.add(task)
                    task.add_done_callback(active_tasks.discard)
                    dispatched_any = True
                else:
                    # No capacity for highest priority request's model
                    # Could check lower priority requests for other models, but
                    # that would violate priority ordering - so we wait
                    break

            # Wait for at least one task to complete (or a short timeout for live display updates)
            if active_tasks:
                done, _ = await asyncio.wait(
                    active_tasks,
                    timeout=0.1,  # Brief timeout to update live display
                    return_when=asyncio.FIRST_COMPLETED,
                )
            else:
                # No active tasks - brief sleep to avoid busy-waiting
                await asyncio.sleep(0.01)

        # Wait for remaining tasks to complete
        if active_tasks:
            await asyncio.gather(*active_tasks)

    async def run_questions(
        self,
        questions: List[Dict[str, Any]],
        progress_callback: Optional[
            Callable[[int, int, int, Dict[str, Any]], None]
        ] = None,
    ) -> List[Dict[str, Any]]:
        """Run all questions using request-level scheduling.

        Args:
            questions: List of question dictionaries
            progress_callback: Optional callback(completed, total, question_idx, transcript)

        Returns:
            List of transcript dictionaries (in original question order)
        """
        self.progress_callback = progress_callback
        self.total_questions = len(questions)

        # Start engine if needed
        await self._start_engine_if_needed()

        try:
            # Initialize all conversations (creates round 0 requests)
            for idx, question in enumerate(questions):
                self._initialize_conversation(idx, question)

            # Initial readiness check (moves ready requests to pre-departure)
            self._check_pending_for_readiness()

            # Initialize live display AFTER all startup messages are printed
            # This ensures the display area is reserved at the correct position
            self.live_display.initialize()

            # Run the scheduler loop
            await self._scheduler_loop()

            # Sort results by conversation_id to maintain original order
            results = [None] * len(questions)
            for transcript in self.completed_transcripts:
                # Find the conversation_id from the transcript
                for conv_id, conv_state in self.conversation_states.items():
                    if conv_state.transcript_id == transcript["transcript_id"]:
                        results[conv_id] = transcript
                        break

            return results

        finally:
            self.live_display.cleanup()
            await self._stop_engine_if_needed()

    async def _start_engine_if_needed(self) -> None:
        """Start backend engine if needed."""
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
