"""Base agent class with shared functionality for all agent implementations."""

import json
import re
from abc import ABC
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from src.utils import debug_print


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

        # Validate parameter ranges
        self._validate_generation_params(self.temperature, self.max_tokens)

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
    ) -> None:
        """Validate generation parameters are in valid ranges.

        Args:
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Raises:
            ValueError: If parameters are out of valid range
        """
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(
                f"temperature must be between 0.0 and 2.0, got {temperature}"
            )
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

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

    def _parse_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from response text with multiple fallback strategies.

        Tries:
        1. Direct JSON parsing
        2. Extract from ```json ... ``` code block
        3. Find any JSON object in text

        Args:
            response_text: Raw response text from model

        Returns:
            Parsed JSON object or None if parsing fails
        """
        # Strategy 1: Direct parsing
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code block
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find any JSON object
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        debug_print(f"Failed to parse JSON from response: {response_text[:200]}...")
        return None

    def __repr__(self) -> str:
        """String representation of agent."""
        return f"{self.__class__.__name__}(id={self.agent_id}, role={self.role})"
