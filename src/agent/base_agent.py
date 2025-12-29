"""Base agent class with shared functionality for all agent implementations."""

import json
import re
from abc import ABC
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

from json_repair import repair_json

from src.utils import debug_print
from src.utils.errors import JSONParseError


@runtime_checkable
class AsyncAgentProtocol(Protocol):
    """Protocol defining the required interface for async agents.

    Async agents support concurrent generation calls, enabling
    cross-conversation parallelism (multiple conversations at once).

    The generate method takes a session parameter for connection reuse.
    """

    agent_id: str
    role: str
    temperature: float
    max_tokens: int
    top_p: Optional[float]
    top_k: Optional[int]
    min_p: Optional[float]
    presence_penalty: Optional[float]
    enable_thinking: Optional[bool]

    async def generate(
        self,
        session: Any,  # aiohttp.ClientSession or similar
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: str = "json",
    ) -> Dict[str, Any]:
        """Generate a response from the agent asynchronously."""
        ...


class BaseAgent(ABC):
    """Abstract base class for all agent implementations.

    Provides shared functionality for prompt building, JSON parsing,
    and configuration validation.
    """

    # Required fields that must be present in agent_config
    REQUIRED_AGENT_FIELDS: List[str] = [
        "agent_id",
        "role",
        "if_as_human",
        "temperature",
        "max_tokens",
    ]

    def __init__(self, agent_config: Dict[str, Any], model_config: Dict[str, Any]):
        """Initialize base agent with configuration validation.

        Args:
            agent_config: Agent configuration dictionary
            model_config: Model configuration dictionary

        Raises:
            ValueError: If required fields are missing or invalid
        """
        self._validate_agent_config(agent_config)

        self.config = agent_config
        self.model_config = model_config

        # Agent attributes
        self.agent_id: str = agent_config["agent_id"]
        self.role: str = agent_config["role"]
        self.persona: Optional[str] = agent_config.get("persona")
        self.demographics: Optional[str] = agent_config.get("demographics")
        self.if_as_human: bool = agent_config["if_as_human"]
        self.temperature: float = agent_config["temperature"]
        self.max_tokens: int = agent_config["max_tokens"]
        self.top_p: Optional[float] = agent_config.get("top_p")
        self.top_k: Optional[int] = agent_config.get("top_k")
        self.min_p: Optional[float] = agent_config.get("min_p")
        self.presence_penalty: Optional[float] = agent_config.get("presence_penalty")
        self.enable_thinking: Optional[bool] = agent_config.get("enable_thinking")

        # Validate parameter ranges
        self._validate_generation_params(
            self.temperature,
            self.max_tokens,
            self.top_p,
            self.top_k,
            self.min_p,
            self.presence_penalty,
        )

    def _validate_agent_config(self, agent_config: Dict[str, Any]) -> None:
        """Validate that all required fields are present.

        Args:
            agent_config: Agent configuration to validate

        Raises:
            ValueError: If required field is missing
        """
        missing = [f for f in self.REQUIRED_AGENT_FIELDS if f not in agent_config]
        if missing:
            raise ValueError(
                f"Missing required agent fields: {missing}. "
                f"Required: {self.REQUIRED_AGENT_FIELDS}"
            )

    def _validate_generation_params(
        self,
        temperature: float,
        max_tokens: int,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> None:
        """Validate generation parameters are in valid ranges.

        Args:
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling probability (optional)
            top_k: Top-k sampling (optional)
            min_p: Minimum probability threshold (optional)
            presence_penalty: Presence penalty for repetition control (optional)

        Raises:
            ValueError: If parameters are out of valid range
        """
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(
                f"temperature must be between 0.0 and 2.0, got {temperature}"
            )
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if top_p is not None and not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be between 0.0 and 1.0, got {top_p}")
        if top_k is not None and top_k < -1:
            raise ValueError(f"top_k must be -1 (disabled) or >= 0, got {top_k}")
        if min_p is not None and not 0.0 <= min_p <= 1.0:
            raise ValueError(f"min_p must be between 0.0 and 1.0, got {min_p}")
        if presence_penalty is not None and not -2.0 <= presence_penalty <= 2.0:
            raise ValueError(
                f"presence_penalty must be between -2.0 and 2.0, got {presence_penalty}"
            )

    def _build_system_prompt(self) -> str:
        """Build system prompt based on agent configuration.

        Returns:
            System prompt string describing the agent's identity
        """
        # Build identity components
        identity_parts = []
        if self.demographics:
            identity_parts.append(self.demographics)
        if self.persona:
            identity_parts.append(self.persona)
        elif self.demographics:
            identity_parts.append("person")

        # Construct identity
        identity = " ".join(identity_parts) if identity_parts else "person"

        # Build prompt based on if_as_human flag
        if self.if_as_human:
            return f"You are a {identity} acting as a {self.role}."
        else:
            return (
                f"You are an AI agent assisting a {identity} acting as a {self.role}."
            )

    def _parse_json_response(
        self, response_text: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[JSONParseError]]:
        """Parse JSON from response text with multiple fallback strategies.

        Tries:
        1. Direct JSON parsing
        2. Extract from ```json ... ``` code block
        3. Find any JSON object in text
        4. Repair malformed JSON using json-repair library

        Args:
            response_text: Raw response text from model

        Returns:
            Tuple of (parsed JSON object or None, JSONParseError if repair was needed or failed)
        """
        # Strategy 1: Direct parsing
        try:
            return json.loads(response_text), None
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code block
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1)), None
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find any JSON object
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0)), None
            except json.JSONDecodeError:
                pass

        # Strategy 4: Repair malformed JSON using json-repair library
        try:
            repaired = repair_json(response_text, return_objects=True)
            if isinstance(repaired, dict):
                debug_print(f"Repaired malformed JSON from response:\n{response_text}")
                return repaired, JSONParseError(response_text, repaired=True)
        except Exception:
            pass

        debug_print(f"Failed to parse JSON from response:\n{response_text}")
        return None, JSONParseError(response_text, repaired=False)

    def __repr__(self) -> str:
        """String representation of agent."""
        return f"{self.__class__.__name__}(id={self.agent_id}, role={self.role})"
