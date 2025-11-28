"""Ollama agent for local model inference."""

import json
import subprocess
import time
from typing import Dict, Any, Optional
import re


class OllamaAgent:
    """Agent using Ollama for local inference with strict validation.

    This agent provides realistic proof-of-concept for development testing
    without requiring GPU servers or vLLM infrastructure.
    """

    def __init__(self, agent_config: Dict[str, Any], model_config: Dict[str, Any]):
        """Initialize Ollama agent.

        Args:
            agent_config: Agent configuration dictionary
            model_config: Model configuration dictionary

        Raises:
            ValueError: If required fields are missing
            RuntimeError: If Ollama is not available
        """
        # Validate required fields
        required_agent_fields = [
            "agent_id",
            "role",
            "if_as_human",
            "temperature",
            "max_tokens",
        ]
        for field in required_agent_fields:
            if field not in agent_config:
                raise ValueError(f"Missing required agent field: {field}")

        self.config = agent_config
        self.model_config = model_config

        # Agent attributes
        self.agent_id = agent_config["agent_id"]
        self.role = agent_config["role"]
        self.persona = agent_config.get("persona")
        self.demographics = agent_config.get("demographics")
        self.if_as_human = agent_config["if_as_human"]
        self.temperature = agent_config["temperature"]
        self.max_tokens = agent_config["max_tokens"]

        # Model name
        self.model_name = model_config.get("model_name")
        if not self.model_name:
            raise ValueError("model_name is required in model_config")

        # Validate Ollama availability
        self._validate_ollama()

    def _validate_ollama(self):
        """Ensure Ollama is installed and running.

        Raises:
            RuntimeError: If Ollama is not available
        """
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            # Check if our model is available
            if self.model_name not in result.stdout:
                print(f"⚠ Warning: Model {self.model_name} not found in Ollama.")
                print(f"  Please run: ollama pull {self.model_name}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Ollama command failed: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                "Ollama is not installed. Please install from https://ollama.ai"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Ollama list command timed out")

    def _build_system_prompt(self) -> str:
        """Build system prompt based on agent configuration.

        Returns:
            System prompt string
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
        if identity_parts:
            identity = " ".join(identity_parts)
        else:
            identity = "person"

        # Build prompt based on if_as_human flag
        if self.if_as_human:
            return f"You are a {identity} acting as a {self.role}."
        else:
            return (
                f"You are an AI agent assisting a {identity} acting as a {self.role}."
            )

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: str = "json",
    ) -> Dict[str, Any]:
        """Generate response using Ollama with proper metrics.

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
                - tokens_prompt: Estimated prompt tokens
                - generation_time_ms: Generation time
                - exceeded_max_tokens: Whether max_tokens was hit

        Raises:
            RuntimeError: If Ollama inference fails
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        # Build full prompt with system message
        system_prompt = self._build_system_prompt()
        full_prompt = f"{system_prompt}\n\n{prompt}"

        # Prepare Ollama API call using the HTTP API for better control
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

        start_time = time.time()

        try:
            # Use Ollama HTTP API for better response data
            import urllib.request
            import urllib.error

            api_url = "http://localhost:11434/api/generate"
            req = urllib.request.Request(
                api_url,
                data=json.dumps(request_data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )

            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    result = json.loads(response.read().decode("utf-8"))
            except urllib.error.URLError:
                # Fallback to CLI if API not available
                return self._generate_via_cli(
                    full_prompt, temp, max_tok, response_format, start_time
                )

        except Exception as e:
            raise RuntimeError(f"Ollama API call failed: {e}")

        generation_time_ms = round((time.time() - start_time) * 1000, 3)

        # Extract response and metrics
        response_text = result.get("response", "")

        # Get actual token counts from Ollama response
        tokens_generated = result.get("eval_count", 0)  # Tokens in response
        tokens_prompt = result.get("prompt_eval_count", 0)  # Tokens in prompt

        # Check if we hit max tokens
        exceeded_max_tokens = tokens_generated >= max_tok

        # Parse JSON if requested
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

    def _generate_via_cli(
        self,
        full_prompt: str,
        temp: float,
        max_tok: int,
        response_format: str,
        start_time: float,
    ) -> Dict[str, Any]:
        """Fallback generation via CLI when API is not available.

        Args:
            full_prompt: Complete prompt with system message
            temp: Temperature
            max_tok: Max tokens
            response_format: Response format
            start_time: Start timestamp

        Returns:
            Generation result dictionary
        """
        cmd = ["ollama", "run"]
        if response_format == "json":
            cmd.extend(["--format", "json"])
        cmd.append(self.model_name)

        try:
            result = subprocess.run(
                cmd,
                input=full_prompt,
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
            response_text = result.stdout.strip()
        except subprocess.TimeoutExpired:
            raise RuntimeError("Ollama CLI inference timed out after 120s")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Ollama CLI inference failed: {e.stderr}")

        generation_time_ms = round((time.time() - start_time) * 1000, 3)

        # Estimate tokens for CLI fallback (less accurate)
        # Using tiktoken or a better estimation would be ideal
        tokens_generated = self._estimate_tokens(response_text)
        tokens_prompt = self._estimate_tokens(full_prompt)

        # Conservative estimate for max tokens
        exceeded_max_tokens = tokens_generated >= max_tok * 0.95

        # Parse JSON if needed
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

    def _parse_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from response text.

        Args:
            response_text: Raw response text

        Returns:
            Parsed JSON object or None if parsing fails
        """
        try:
            # First try direct parsing
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try extracting from markdown code block
            json_match = re.search(
                r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL
            )
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            # Try finding any JSON object
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass

        return None

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        This is a rough approximation. For production, use proper tokenizer.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Rough estimation based on common patterns
        # Average English word is ~4-5 characters, ~1.3 tokens per word
        words = len(text.split())
        chars = len(text)

        # Use combination of word and character count
        # This is model-specific and should be calibrated
        estimated_tokens = int(words * 1.3 + chars / 6) // 2

        return max(1, estimated_tokens)

    def __repr__(self) -> str:
        """String representation of agent."""
        return (
            f"OllamaAgent(id={self.agent_id}, role={self.role}, "
            f"model={self.model_name})"
        )
