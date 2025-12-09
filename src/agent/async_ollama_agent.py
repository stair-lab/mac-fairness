"""Async Ollama agent for local model inference with aiohttp.

This module provides an async Ollama agent that supports concurrent HTTP requests,
enabling cross-conversation parallelism.

Note: Ollama itself does not perform continuous batching like vLLM.
Multiple concurrent requests are queued internally by Ollama.
The async interface here enables parallel conversations.
"""

import subprocess
import time
from typing import Any, ClassVar, Dict, Optional

import aiohttp

from .base_agent import BaseAgent
from src.utils.errors import (
    OllamaAPIError,
    OllamaConnectionError,
    OllamaNotAvailableError,
)


class AsyncOllamaAgent(BaseAgent):
    """Async agent using Ollama for local inference.

    Provides async HTTP interface to Ollama's /api/generate endpoint.
    Uses a class-level aiohttp.ClientSession for connection reuse across
    all agent instances.

    Interface:
        async generate(prompt, ...) -> Dict[str, Any]

    Usage:
        # Create agent
        agent = AsyncOllamaAgent(agent_config, model_config)

        # Ensure session is created before first use
        await AsyncOllamaAgent.ensure_session()

        # Generate response
        result = await agent.generate(prompt)

        # Cleanup when done
        await AsyncOllamaAgent.close_session()
    """

    # Class-level session for connection reuse
    _session: ClassVar[Optional[aiohttp.ClientSession]] = None

    def __init__(self, agent_config: Dict[str, Any], model_config: Dict[str, Any]):
        """Initialize async Ollama agent.

        Args:
            agent_config: Agent configuration dictionary
            model_config: Model configuration dictionary

        Raises:
            ValueError: If required fields are missing
            RuntimeError: If Ollama is not available
        """
        super().__init__(agent_config, model_config)

        # Model name
        self.model_name = model_config.get("model_name")
        if not self.model_name:
            raise ValueError("model_name is required in model_config")

        # Ollama API configuration
        self.api_base_url = model_config.get("ollama_api_url", "http://localhost:11434")
        self.timeout_seconds = model_config.get("timeout_seconds", 120)

        # Validate Ollama availability (sync check at init time)
        self._validate_ollama()

    def _validate_ollama(self) -> None:
        """Ensure Ollama is installed and running.

        Raises:
            OllamaNotAvailableError: If Ollama is not available
        """
        try:
            result = subprocess.run(
                ["zsh", "-c", "ollama list"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            if self.model_name not in result.stdout:
                raise OllamaNotAvailableError(
                    f"Model {self.model_name} not found in Ollama. "
                    f"Run: ollama pull {self.model_name}"
                )
        except subprocess.CalledProcessError as e:
            raise OllamaNotAvailableError(f"Ollama command failed: {e.stderr}")
        except FileNotFoundError:
            raise OllamaNotAvailableError(
                "Ollama is not installed. Please install from https://ollama.ai"
            )
        except subprocess.TimeoutExpired:
            raise OllamaNotAvailableError("Ollama list command timed out")

    @classmethod
    async def ensure_session(cls) -> None:
        """Ensure aiohttp session is created.

        Should be called before first generate() call.
        """
        if cls._session is None:
            cls._session = aiohttp.ClientSession()

    @classmethod
    async def close_session(cls) -> None:
        """Close the aiohttp session.

        Should be called at cleanup time.
        """
        if cls._session is not None:
            await cls._session.close()
            cls._session = None

    @classmethod
    async def start_engine(cls) -> None:
        """Start the async engine (creates session).

        For interface compatibility with AsyncVLLMAgent.
        """
        await cls.ensure_session()

    @classmethod
    async def stop_engine(cls) -> None:
        """Stop the async engine (closes session).

        For interface compatibility with AsyncVLLMAgent.
        """
        await cls.close_session()

    @classmethod
    async def cleanup_all_async(cls) -> None:
        """Async cleanup: close session.

        For interface compatibility with AsyncVLLMAgent.
        """
        await cls.close_session()

    @classmethod
    def cleanup_all(cls) -> None:
        """Sync cleanup - limited for Ollama (session requires async close).

        For interface compatibility with AsyncVLLMAgent.
        Note: Use cleanup_all_async() to properly close the session.
        """
        # Cannot close aiohttp session synchronously
        cls._session = None

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: str = "json",
    ) -> Dict[str, Any]:
        """Generate response using Ollama asynchronously.

        Args:
            prompt: Input prompt
            temperature: Override temperature (optional)
            max_tokens: Override max tokens (optional)
            response_format: Expected format ("json" or "text")

        Returns:
            Dictionary containing:
                - text: Raw text response
                - structured_output: Parsed JSON (if json format)
                - tokens_generated: Actual tokens in response
                - tokens_prompt: Tokens in prompt
                - generation_time_ms: Generation time
                - exceeded_max_tokens: Whether max_tokens was hit

        Raises:
            OllamaConnectionError: If session not initialized or connection fails
            OllamaAPIError: If Ollama API returns an error
        """
        if self._session is None:
            raise OllamaConnectionError(
                "aiohttp session not initialized. Call await AsyncOllamaAgent.ensure_session() first."
            )

        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        # Build full prompt with system message
        system_prompt = self._build_system_prompt()
        full_prompt = f"{system_prompt}\n\n{prompt}"

        # Build request payload
        request_data = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": max_tok,
            },
        }
        if response_format == "json":
            request_data["format"] = "json"

        api_url = f"{self.api_base_url}/api/generate"
        start_time = time.time()

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            async with self._session.post(
                api_url,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise OllamaAPIError(
                        f"Ollama API returned status {response.status}",
                        status_code=response.status,
                        response_text=error_text,
                    )
                result = await response.json()

        except aiohttp.ClientError as e:
            raise OllamaConnectionError("Ollama API call failed", original_error=e)

        generation_time_ms = round((time.time() - start_time) * 1000, 3)

        # Parse response
        response_text = result.get("response", "")
        tokens_generated = result.get("eval_count", 0)
        tokens_prompt = result.get("prompt_eval_count", 0)
        exceeded_max_tokens = tokens_generated >= max_tok

        structured_output = None
        if response_format == "json":
            structured_output = self._parse_json_response(response_text)

        return {
            "text": response_text,
            "structured_output": structured_output,
            "tokens_generated": tokens_generated,
            "tokens_prompt": tokens_prompt,
            "generation_time_ms": generation_time_ms,
            "exceeded_max_tokens": exceeded_max_tokens,
        }

    def __repr__(self) -> str:
        """String representation of agent."""
        return (
            f"AsyncOllamaAgent(id={self.agent_id}, role={self.role}, "
            f"model={self.model_name})"
        )
