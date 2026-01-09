"""Model factory for creating agent instances."""

from __future__ import annotations

from typing import Any, Dict, Union, TYPE_CHECKING
from .async_ollama_agent import AsyncOllamaAgent

if TYPE_CHECKING:
    from .async_vllm_agent import AsyncVLLMAgent


class ModelFactory:
    """Factory for creating model instances.

    Agents specify model names which are looked up in the models config.
    Backend must be explicitly specified in model_definitions.
    """

    def __init__(self, experiment_config: Dict[str, Any]):
        """Initialize factory with experiment configuration.

        Args:
            experiment_config: Experiment configuration dictionary

        The model_definitions section contains models (model_name as key):
            model_definitions:
              llama31_8b:
                backend: vllm
                model_path: meta-llama/Llama-3.1-8B-Instruct
                vllm_config: ...
        """
        self.experiment_config = experiment_config
        # Models are defined under model_definitions (model_name as key)
        self.model_definitions = experiment_config.get("model_definitions", {})

        # Cache for shared model instances (not used for Ollama, prepared for vLLM)
        self._shared_instances = {}

    def _resolve_model_config(
        self, agent_config: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any], str]:
        """Resolve model name, config, and backend from agent config.

        Args:
            agent_config: Agent configuration dictionary

        Returns:
            Tuple of (model_name, model_config, backend)

        Raises:
            ValueError: If configuration is invalid
        """
        model_name = agent_config["model"]

        if model_name not in self.model_definitions:
            raise ValueError(
                f"Model '{model_name}' not found in model definitions. "
                f"Available: {list(self.model_definitions.keys())}"
            )

        model_config = self.model_definitions[model_name]
        backend = self._get_backend(model_name, model_config)

        return model_name, model_config, backend

    def create_agent(
        self, agent_config: Dict[str, Any]
    ) -> Union[AsyncOllamaAgent, AsyncVLLMAgent]:
        """Create an async agent instance based on configuration.

        Args:
            agent_config: Agent configuration dictionary

        Returns:
            Agent instance (AsyncOllamaAgent or AsyncVLLMAgent)

        Raises:
            ValueError: If configuration is invalid
        """
        _, model_config, backend = self._resolve_model_config(agent_config)

        if backend == "ollama":
            return self._create_async_ollama_agent(agent_config, model_config)
        elif backend == "vllm":
            return self._create_async_vllm_agent(agent_config, model_config)
        else:
            raise ValueError(
                f"Unsupported backend: {backend}. Supported backends: ollama, vllm"
            )

    def _get_backend(self, model_name: str, model_config: Dict[str, Any]) -> str:
        """Get backend from model configuration.

        Args:
            model_name: Name of the model
            model_config: Model configuration dictionary

        Returns:
            Backend name ("ollama" or "vllm")

        Raises:
            ValueError: If backend not specified
        """
        backend = model_config.get("backend")
        if not backend:
            raise ValueError(
                f"Model '{model_name}' missing required 'backend' field. "
                "Must be 'vllm' or 'ollama'."
            )
        return backend

    def _create_async_ollama_agent(
        self, agent_config: Dict[str, Any], model_config: Dict[str, Any]
    ) -> AsyncOllamaAgent:
        """Create an async Ollama agent instance.

        Args:
            agent_config: Agent configuration
            model_config: Model configuration

        Returns:
            AsyncOllamaAgent instance
        """
        # Ensure model_name is present for Ollama
        if "model_name" not in model_config:
            raise ValueError(
                "Ollama backend requires 'model_name' in model configuration. "
                "Example: 'llama3.2:1b-instruct-q4_K_M'"
            )

        return AsyncOllamaAgent(agent_config, model_config)

    def _create_async_vllm_agent(
        self, agent_config: Dict[str, Any], model_config: Dict[str, Any]
    ) -> "AsyncVLLMAgent":
        """Create an async vLLM agent instance with batching support.

        Args:
            agent_config: Agent configuration
            model_config: Model configuration

        Returns:
            AsyncVLLMAgent instance

        Raises:
            ValueError: If configuration is invalid
        """
        # Check for required vLLM configuration
        if "model_path" not in model_config and "model_name" not in model_config:
            raise ValueError(
                "vLLM backend requires 'model_path' or 'model_name' in model configuration. "
                "Example: 'meta-llama/Llama-3.1-8B-Instruct'"
            )

        # Lazy import: only load vLLM dependencies when actually creating a vLLM agent
        from .async_vllm_agent import AsyncVLLMAgent

        # Create and return async vLLM agent
        # AsyncVLLMAgent handles shared model instance and batched engine internally
        return AsyncVLLMAgent(agent_config, model_config)

    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about configured backends and models.

        Returns:
            Dictionary keyed by model name with backend and config info
        """
        info: Dict[str, Any] = {}

        for model_name, model_config in self.model_definitions.items():
            backend = self._get_backend(model_name, model_config)
            info[model_name] = {
                "backend": backend,
                "config": model_config,
            }

        return info

    def __repr__(self) -> str:
        """String representation of factory."""
        model_list = list(self.model_definitions.keys())
        return f"ModelFactory(models={model_list})"
