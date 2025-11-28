"""Model factory for creating agent instances with automatic backend detection."""

from typing import Dict, Any
from .ollama_agent import OllamaAgent


class ModelFactory:
    """Factory for creating model instances with shared backbone support.

    Automatically detects backend based on model naming patterns and configuration.
    Supports shared model backbone for memory efficiency.
    """

    def __init__(self, experiment_config: Dict[str, Any]):
        """Initialize factory with experiment configuration.

        Args:
            experiment_config: Experiment configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        self.experiment_config = experiment_config
        self.shared_backbone = experiment_config.get("shared_model_backbone")
        self.model_definitions = experiment_config.get("models", {})

        # Cache for shared model instances (not used for Ollama, prepared for vLLM)
        self._shared_instances = {}

        # Validate shared backbone if specified
        if self.shared_backbone and self.shared_backbone not in self.model_definitions:
            raise ValueError(
                f"shared_model_backbone '{self.shared_backbone}' not found in models. "
                f"Available models: {list(self.model_definitions.keys())}"
            )

    def create_agent(self, agent_config: Dict[str, Any]) -> Any:
        """Create an agent instance based on configuration.

        Args:
            agent_config: Agent configuration dictionary

        Returns:
            Agent instance (OllamaAgent, VLLMAgent, etc.)

        Raises:
            ValueError: If configuration is invalid
            NotImplementedError: If backend is not yet implemented
        """
        # Determine which model to use
        model_spec = agent_config.get("model")
        if not model_spec:
            raise ValueError(
                f"Agent {agent_config.get('agent_id')} missing 'model' field"
            )

        if model_spec == "shared":
            if not self.shared_backbone:
                raise ValueError(
                    f"Agent {agent_config.get('agent_id')} specifies 'model: shared' "
                    "but no shared_model_backbone is defined in experiment config"
                )
            model_name = self.shared_backbone
        else:
            model_name = model_spec

        # Get model configuration
        if model_name not in self.model_definitions:
            raise ValueError(
                f"Model '{model_name}' not found in model definitions. "
                f"Available: {list(self.model_definitions.keys())}"
            )

        model_config = self.model_definitions[model_name]

        # Detect backend
        backend = self._detect_backend(model_name, model_config)

        # Create agent based on backend
        if backend == "ollama":
            return self._create_ollama_agent(agent_config, model_config)
        elif backend == "vllm":
            return self._create_vllm_agent(agent_config, model_config)
        else:
            raise ValueError(
                f"Unsupported backend: {backend}. " "Supported backends: ollama, vllm"
            )

    def _detect_backend(self, model_name: str, model_config: Dict[str, Any]) -> str:
        """Detect backend based on model configuration and naming patterns.

        Args:
            model_name: Name of the model
            model_config: Model configuration dictionary

        Returns:
            Backend name ("ollama" or "vllm")
        """
        # Explicit backend specification takes precedence
        if "backend" in model_config:
            return model_config["backend"]

        # Auto-detect based on model configuration patterns

        # Ollama-specific patterns
        if "model_name" in model_config:
            model_str = model_config["model_name"]
            # Ollama models typically have version/quantization suffixes
            ollama_patterns = [
                "-q4_K_M",
                "-q8_0",
                "-q5_K_S",  # Quantization patterns
                ":1b",
                ":3b",
                ":7b",
                ":8b",  # Size patterns
                "-instruct",
                "-it",  # Instruction-tuned patterns
            ]
            if any(pattern in model_str for pattern in ollama_patterns):
                return "ollama"

        # vLLM-specific patterns
        if "model_path" in model_config:
            # HuggingFace model paths or local paths typically use vLLM
            model_path = model_config["model_path"]
            if (
                "/" in model_path
                or model_path.startswith("meta-")
                or model_path.startswith("mistralai")
            ):
                return "vllm"

        # Check for vLLM-specific configuration
        if "vllm_config" in model_config:
            return "vllm"

        # Check for Ollama-specific configuration
        if "ollama_config" in model_config:
            return "ollama"

        # Default based on environment
        # If in dev_ollama benchmark, default to ollama
        if self.experiment_config.get("benchmark_subcategory") == "dev_ollama":
            return "ollama"

        # Otherwise default to vllm for production
        return "vllm"

    def _create_ollama_agent(
        self, agent_config: Dict[str, Any], model_config: Dict[str, Any]
    ) -> OllamaAgent:
        """Create an Ollama agent instance.

        Args:
            agent_config: Agent configuration
            model_config: Model configuration

        Returns:
            OllamaAgent instance
        """
        # Ensure model_name is present for Ollama
        if "model_name" not in model_config:
            raise ValueError(
                "Ollama backend requires 'model_name' in model configuration. "
                "Example: 'llama3.2:1b-instruct-q4_K_M'"
            )

        return OllamaAgent(agent_config, model_config)

    def _create_vllm_agent(
        self, agent_config: Dict[str, Any], model_config: Dict[str, Any]
    ) -> Any:
        """Create a vLLM agent instance.

        Args:
            agent_config: Agent configuration
            model_config: Model configuration

        Returns:
            VLLMAgent instance

        Raises:
            NotImplementedError: vLLM backend not yet implemented
        """
        # Check for required vLLM configuration
        if "model_path" not in model_config and "model_name" not in model_config:
            raise ValueError(
                "vLLM backend requires 'model_path' or 'model_name' in model configuration. "
                "Example: 'meta-llama/Llama-3.1-8B-Instruct'"
            )

        # Would implement shared model instance caching here
        # For now, not implemented
        raise NotImplementedError(
            "vLLM backend not yet implemented. "
            "Please use 'backend: ollama' in your model configuration for development testing."
        )

    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about configured backends and models.

        Returns:
            Dictionary with backend and model information
        """
        info = {"shared_backbone": self.shared_backbone, "models": {}}

        for model_name, model_config in self.model_definitions.items():
            backend = self._detect_backend(model_name, model_config)
            info["models"][model_name] = {
                "backend": backend,
                "family": model_config.get("family", "unknown"),
                "config": model_config,
            }

        return info

    def __repr__(self) -> str:
        """String representation of factory."""
        model_list = list(self.model_definitions.keys())
        return (
            f"ModelFactory(shared_backbone={self.shared_backbone}, "
            f"models={model_list})"
        )
