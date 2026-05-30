"""Async OpenAI agent for hosted LLM inference via OpenAI Chat Completions.

This agent talks to OpenAI's hosted models (e.g. GPT-5.5) — or any
OpenAI-compatible endpoint via ``base_url`` — using the async OpenAI SDK.
Unlike the vLLM/Ollama agents it does not manage any local engine; concurrency
is bounded by the request scheduler's per-model semaphore.

Design mirrors AsyncVLLMAgent / AsyncOllamaAgent so the agent is a drop-in
backend behind ModelFactory:
- ``async generate(prompt, ...)`` returns the same response dict shape.
- Class-level lifecycle hooks (start_engine / stop_engine / cleanup) are no-ops
  or client teardown, for interface compatibility with the schedulers.

Configuration (model_definitions.<name>):
    backend: openai
    model_name: gpt-5.5                # provider model id (required)
    max_num_seqs: 16                   # max concurrent in-flight requests (optional)
    api_config:
      base_url: null                   # override for OpenAI-compatible endpoints
      api_key_env: OPENAI_API_KEY      # env var holding the API key
      timeout_seconds: 120
      max_retries: 5                   # SDK-level retries on 429/5xx
      reasoning_effort: none           # none|minimal|low|medium|high|xhigh
      verbosity: low                   # low|medium|high
      send_temperature: false          # reasoning models reject non-default temp
      send_top_p: false
"""

import os
import time
from typing import Any, ClassVar, Dict, Optional

from openai import (
    APIConnectionError as OpenAIAPIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from .base_agent import BaseAgent
from src.utils import debug_print, info_print
from src.utils.errors import APIError, APIRequestError


class AsyncOpenAIAgent(BaseAgent):
    """Async agent backed by OpenAI's hosted Chat Completions API.

    Uses a shared ``AsyncOpenAI`` client per (base_url, api_key_env) so all
    agents pointing at the same endpoint reuse one connection pool.
    """

    # Shared async clients keyed by endpoint identity (base_url::api_key_env)
    _clients: ClassVar[Dict[str, AsyncOpenAI]] = {}

    # Valid control values for the gpt-5.x family
    VALID_REASONING_EFFORTS: ClassVar[set] = {
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
    }
    VALID_VERBOSITY: ClassVar[set] = {"low", "medium", "high"}

    def __init__(self, agent_config: Dict[str, Any], model_config: Dict[str, Any]):
        """Initialize async API agent.

        Args:
            agent_config: Agent configuration dictionary
            model_config: Model configuration dictionary (see module docstring)

        Raises:
            ValueError: If required base agent fields are missing
            APIError: If API configuration is invalid or the API key is missing
        """
        super().__init__(agent_config, model_config)

        # Provider model id (e.g. "gpt-5.5"). Accept model_path as an alias.
        self.model_name = model_config.get("model_name") or model_config.get("model_path")
        if not self.model_name:
            raise APIError("API backend requires 'model_name' (provider model id) in model_config. Example: 'gpt-5.5'")

        api_config = model_config.get("api_config", {}) or {}
        self.api_config = api_config

        self.api_key_env: str = api_config.get("api_key_env", "OPENAI_API_KEY")
        self.base_url: Optional[str] = api_config.get("base_url")
        self.timeout_seconds: float = api_config.get("timeout_seconds", 120)
        self.client_max_retries: int = api_config.get("max_retries", 5)

        # Reasoning / verbosity controls (gpt-5.x family)
        self.reasoning_effort: Optional[str] = api_config.get("reasoning_effort")
        if self.reasoning_effort is not None and self.reasoning_effort not in self.VALID_REASONING_EFFORTS:
            raise APIError(
                f"Invalid reasoning_effort '{self.reasoning_effort}'. "
                f"Must be one of {sorted(self.VALID_REASONING_EFFORTS)}"
            )

        self.verbosity: Optional[str] = api_config.get("verbosity")
        if self.verbosity is not None and self.verbosity not in self.VALID_VERBOSITY:
            raise APIError(f"Invalid verbosity '{self.verbosity}'. Must be one of {sorted(self.VALID_VERBOSITY)}")

        # Reasoning models reject non-default sampling params; gate behind flags.
        self.send_temperature: bool = api_config.get("send_temperature", False)
        self.send_top_p: bool = api_config.get("send_top_p", False)

        # Fail fast if no credentials are available for the default OpenAI endpoint.
        if not self.base_url and not os.environ.get(self.api_key_env):
            raise APIError(
                f"Environment variable '{self.api_key_env}' is not set. "
                f"Export your API key, e.g. export {self.api_key_env}=sk-..."
            )

        self._ensure_client()

    @property
    def _client_key(self) -> str:
        return f"{self.base_url or 'openai-default'}::{self.api_key_env}"

    def _ensure_client(self) -> None:
        """Create or reuse a shared AsyncOpenAI client for this endpoint."""
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
            client = AsyncOpenAI(**kwargs)
            self._clients[key] = client
            info_print(
                f"Initialized AsyncOpenAI client for {self.model_name} (endpoint={self.base_url or 'openai-default'})"
            )
        self.client = client

    # ------------------------------------------------------------------
    # Lifecycle hooks (interface compatibility with vLLM/Ollama agents)
    # ------------------------------------------------------------------
    @classmethod
    async def start_engine(cls) -> None:
        """No-op: the AsyncOpenAI client is created lazily at agent init."""
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
                debug_print(f"Error closing API client {key}: {e}")
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
        """Generate a response via the hosted Chat Completions API.

        Args:
            prompt: Input prompt
            temperature: Override temperature (only sent if send_temperature=True)
            max_tokens: Override max output tokens (optional)
            response_format: Expected format ("json" or "text")

        Returns:
            Dictionary containing:
                - text: Raw text response
                - structured_output: Parsed JSON (if json format)
                - json_parse_error: JSONParseError if parsing needed repair/failed
                - tokens_generated: Completion tokens (includes reasoning tokens)
                - tokens_prompt: Prompt tokens
                - reasoning_tokens: Reasoning tokens (0 if not a reasoning model)
                - exceeded_max_tokens: Whether the response was truncated by length
                - finish_reason: Provider finish reason
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
            # response_format=json_object requires the word "json" in the prompt.
            user_message += (
                "\n\nYou must respond with valid JSON only. Do not include any text outside the JSON object."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        request_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": max_tok,
        }
        if response_format == "json":
            request_kwargs["response_format"] = {"type": "json_object"}
        if self.reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = self.reasoning_effort
        if self.verbosity is not None:
            request_kwargs["verbosity"] = self.verbosity
        if self.send_temperature:
            request_kwargs["temperature"] = temp
        if self.send_top_p and self.top_p is not None:
            request_kwargs["top_p"] = self.top_p

        start_time = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(**request_kwargs)
        except APIStatusError as e:
            raise APIRequestError(
                message=f"API request failed ({e.status_code}): {e}",
                model=self.model_name,
                status_code=e.status_code,
                original_error=e,
            )
        except (RateLimitError, APITimeoutError, OpenAIAPIConnectionError) as e:
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

        choice = response.choices[0]
        response_text = choice.message.content or ""
        finish_reason = choice.finish_reason
        exceeded_max_tokens = finish_reason == "length"

        usage = response.usage
        tokens_prompt = usage.prompt_tokens if usage else 0
        tokens_generated = usage.completion_tokens if usage else 0
        reasoning_tokens = 0
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        if details is not None:
            reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

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
            "reasoning_tokens": reasoning_tokens,
            "exceeded_max_tokens": exceeded_max_tokens,
            "finish_reason": finish_reason,
            "latency_ms": latency_ms,
        }

    def __repr__(self) -> str:
        """String representation of agent."""
        return f"AsyncOpenAIAgent(id={self.agent_id}, role={self.role}, model={self.model_name})"
