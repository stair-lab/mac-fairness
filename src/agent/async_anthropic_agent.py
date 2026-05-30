"""Async Anthropic (Claude) agent for hosted LLM inference via the Messages API.

This agent talks to Anthropic's hosted Claude models (e.g. Claude Sonnet 4.6)
using the async Anthropic SDK. Like the OpenAI agent, it manages no local engine;
concurrency is bounded by the request scheduler's per-model semaphore.

Design mirrors AsyncOpenAIAgent / AsyncVLLMAgent / AsyncOllamaAgent so the agent
is a drop-in backend behind ModelFactory:
- ``async generate(prompt, ...)`` returns the same response dict shape.
- Class-level lifecycle hooks (start_engine / stop_engine / cleanup) are no-ops
  or client teardown, for interface compatibility with the schedulers.

Caching note: no prompt caching is configured. The system prompt here is a single
short sentence (well below Sonnet 4.6's 2048-token cache minimum) and every BBQ
question is a unique single-turn request, so there is no reusable cacheable prefix.

Configuration (model_definitions.<name>):
    backend: anthropic
    model_name: claude-sonnet-4-6      # provider model id (required)
    max_num_seqs: 128                  # max concurrent in-flight requests (optional)
    api_config:
      api_key_env: ANTHROPIC_API_KEY   # env var holding the API key
      base_url: null                   # override for compatible endpoints
      timeout_seconds: 120
      max_retries: 5                   # SDK-level retries on 429/5xx
      thinking: disabled               # disabled | adaptive
      effort: null                     # null | low | medium | high (Sonnet: no "max")
      send_temperature: true           # Sonnet 4.6 accepts temperature (thinking off)
      send_top_p: false
"""

import os
import time
from typing import Any, ClassVar, Dict, Optional

from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
    RateLimitError,
)

from .base_agent import BaseAgent
from src.utils import debug_print, info_print
from src.utils.errors import APIError, APIRequestError


class AsyncAnthropicAgent(BaseAgent):
    """Async agent backed by Anthropic's hosted Claude Messages API.

    Uses a shared ``AsyncAnthropic`` client per (base_url, api_key_env) so all
    agents pointing at the same endpoint reuse one connection pool.
    """

    # Shared async clients keyed by endpoint identity (base_url::api_key_env)
    _clients: ClassVar[Dict[str, AsyncAnthropic]] = {}

    VALID_THINKING: ClassVar[set] = {"disabled", "adaptive"}
    # Sonnet 4.6 supports low/medium/high; "max" is Opus-tier only.
    VALID_EFFORTS: ClassVar[set] = {"low", "medium", "high", "max"}

    def __init__(self, agent_config: Dict[str, Any], model_config: Dict[str, Any]):
        """Initialize async Anthropic agent.

        Args:
            agent_config: Agent configuration dictionary
            model_config: Model configuration dictionary (see module docstring)

        Raises:
            ValueError: If required base agent fields are missing
            APIError: If API configuration is invalid or the API key is missing
        """
        super().__init__(agent_config, model_config)

        # Provider model id (e.g. "claude-sonnet-4-6"). Accept model_path as alias.
        self.model_name = model_config.get("model_name") or model_config.get("model_path")
        if not self.model_name:
            raise APIError(
                "Anthropic backend requires 'model_name' (provider model id) in "
                "model_config. Example: 'claude-sonnet-4-6'"
            )

        api_config = model_config.get("api_config", {}) or {}
        self.api_config = api_config

        self.api_key_env: str = api_config.get("api_key_env", "ANTHROPIC_API_KEY")
        self.base_url: Optional[str] = api_config.get("base_url")
        self.timeout_seconds: float = api_config.get("timeout_seconds", 120)
        self.client_max_retries: int = api_config.get("max_retries", 5)

        # Thinking mode (gpt-5.x analog: reasoning_effort none == thinking disabled).
        self.thinking: str = api_config.get("thinking", "disabled")
        if self.thinking not in self.VALID_THINKING:
            raise APIError(f"Invalid thinking '{self.thinking}'. Must be one of {sorted(self.VALID_THINKING)}")

        # Effort (optional; goes inside output_config). None == omit.
        self.effort: Optional[str] = api_config.get("effort")
        if self.effort is not None and self.effort not in self.VALID_EFFORTS:
            raise APIError(f"Invalid effort '{self.effort}'. Must be one of {sorted(self.VALID_EFFORTS)}")

        # Sampling params. Claude rejects temperature when thinking is enabled,
        # so only forward it when thinking is disabled and the flag is set.
        self.send_temperature: bool = api_config.get("send_temperature", True)
        self.send_top_p: bool = api_config.get("send_top_p", False)

        if not self.base_url and not os.environ.get(self.api_key_env):
            raise APIError(
                f"Environment variable '{self.api_key_env}' is not set. "
                f"Export your API key, e.g. export {self.api_key_env}=sk-ant-..."
            )

        self._ensure_client()

    @property
    def _client_key(self) -> str:
        return f"{self.base_url or 'anthropic-default'}::{self.api_key_env}"

    def _ensure_client(self) -> None:
        """Create or reuse a shared AsyncAnthropic client for this endpoint."""
        key = self._client_key
        client = self._clients.get(key)
        if client is None:
            kwargs: Dict[str, Any] = {
                "timeout": self.timeout_seconds,
                "max_retries": self.client_max_retries,
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            api_key = os.environ.get(self.api_key_env)
            if api_key:
                kwargs["api_key"] = api_key
            client = AsyncAnthropic(**kwargs)
            self._clients[key] = client
            info_print(
                f"Initialized AsyncAnthropic client for {self.model_name} "
                f"(endpoint={self.base_url or 'anthropic-default'})"
            )
        self.client = client

    # ------------------------------------------------------------------
    # Lifecycle hooks (interface compatibility with vLLM/Ollama agents)
    # ------------------------------------------------------------------
    @classmethod
    async def start_engine(cls) -> None:
        """No-op: the AsyncAnthropic client is created lazily at agent init."""
        return None

    @classmethod
    async def stop_engine(cls) -> None:
        """No-op: clients are kept open for reuse until cleanup."""
        return None

    @classmethod
    async def cleanup_all_async(cls) -> None:
        """Close all shared API clients and clear the cache."""
        for key, client in list(cls._clients.items()):
            try:
                await client.close()
            except Exception as e:  # best-effort teardown
                debug_print(f"Error closing Anthropic client {key}: {e}")
        cls._clients.clear()

    @classmethod
    def cleanup_all(cls) -> None:
        """Sync cleanup: drop client references (async close not possible here)."""
        cls._clients.clear()

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: str = "json",
    ) -> Dict[str, Any]:
        """Generate a response via the hosted Claude Messages API.

        Args:
            prompt: Input prompt
            temperature: Override temperature (only sent when thinking disabled
                and send_temperature=True)
            max_tokens: Override max output tokens (optional)
            response_format: Expected format ("json" or "text")

        Returns:
            Dictionary containing:
                - text: Raw text response (concatenated text blocks)
                - structured_output: Parsed JSON (if json format)
                - json_parse_error: JSONParseError if parsing needed repair/failed
                - tokens_generated: Output tokens
                - tokens_prompt: Input tokens (incl. any cache read/creation)
                - exceeded_max_tokens: Whether the response was truncated by length
                - finish_reason: Provider stop reason
                - latency_ms: Request latency in milliseconds

        Raises:
            APIRequestError: If the API request fails
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        if temperature is not None:
            self._validate_generation_params(temp, max_tok)

        system_prompt = self._build_system_prompt()
        user_message = prompt
        if response_format == "json":
            user_message += (
                "\n\nYou must respond with valid JSON only. Do not include any text outside the JSON object."
            )

        request_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_tok,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
            "thinking": {"type": self.thinking},
        }
        if self.effort is not None:
            request_kwargs["output_config"] = {"effort": self.effort}
        # Claude rejects sampling params while thinking is enabled.
        if self.thinking == "disabled":
            if self.send_temperature:
                request_kwargs["temperature"] = temp
            if self.send_top_p and self.top_p is not None:
                request_kwargs["top_p"] = self.top_p

        start_time = time.perf_counter()
        try:
            response = await self.client.messages.create(**request_kwargs)
        except APIStatusError as e:
            raise APIRequestError(
                message=f"API request failed ({e.status_code}): {e}",
                model=self.model_name,
                status_code=e.status_code,
                original_error=e,
            )
        except (RateLimitError, APITimeoutError, AnthropicAPIConnectionError) as e:
            raise APIRequestError(
                message=f"API request failed: {e}",
                model=self.model_name,
                original_error=e,
            )
        except Exception as e:
            raise APIRequestError(
                message=f"API request failed: {e}",
                model=self.model_name,
                original_error=e,
            )
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Concatenate text blocks (ignore any thinking blocks).
        response_text = "".join(block.text for block in response.content if block.type == "text")
        finish_reason = response.stop_reason
        exceeded_max_tokens = finish_reason == "max_tokens"

        usage = response.usage
        tokens_generated = getattr(usage, "output_tokens", 0) or 0
        tokens_prompt = (
            (getattr(usage, "input_tokens", 0) or 0)
            + (getattr(usage, "cache_read_input_tokens", 0) or 0)
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
        )

        structured_output = None
        json_parse_error = None
        if response_format == "json":
            structured_output, json_parse_error = self._parse_json_response(response_text)

        return {
            "text": response_text,
            "structured_output": structured_output,
            "json_parse_error": json_parse_error,
            "tokens_generated": tokens_generated,
            "tokens_prompt": tokens_prompt,
            "exceeded_max_tokens": exceeded_max_tokens,
            "finish_reason": finish_reason,
            "latency_ms": latency_ms,
        }

    def __repr__(self) -> str:
        """String representation of agent."""
        return f"AsyncAnthropicAgent(id={self.agent_id}, role={self.role}, model={self.model_name})"
