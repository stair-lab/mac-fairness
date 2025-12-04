"""vLLM agent for GPU-accelerated inference with batching support."""

import gc
import os
import time
from typing import Any, ClassVar, Dict, Optional

from .base_agent import BaseAgent, _debug_print, _info_print


class VLLMConfigError(ValueError):
    """Raised when vLLM configuration is invalid."""

    pass


class VLLMInferenceError(RuntimeError):
    """Raised when vLLM inference fails."""

    pass


class VLLMAgent(BaseAgent):
    """Agent using vLLM for high-performance GPU inference.

    This agent provides production-grade inference with automatic batching,
    KV cache optimization, and hardware utilization tracking.

    The model is loaded once and cached at the class level for sharing
    across multiple agent instances (shared backbone pattern).
    """

    # Class-level cache for shared model instances
    # Using ClassVar to make the intent clear
    _shared_models: ClassVar[Dict[str, Any]] = {}

    # Supported vLLM config parameters with their types
    VLLM_CONFIG_PARAMS = {
        "tensor_parallel_size": int,
        "gpu_memory_utilization": float,
        "max_model_len": int,
        "dtype": str,
        "max_num_seqs": int,
        "enable_prefix_caching": bool,
        "swap_space": int,
        "enforce_eager": bool,
        "quantization": str,
        "gpu_device_ids": list,
        "top_p": float,
        "top_k": int,
    }

    def __init__(self, agent_config: Dict[str, Any], model_config: Dict[str, Any]):
        """Initialize vLLM agent.

        Args:
            agent_config: Agent configuration dictionary
            model_config: Model configuration dictionary containing:
                - model_path: HuggingFace model ID or local path (required)
                - vllm_config: Optional dict of vLLM-specific settings

        Raises:
            ValueError: If required fields are missing
            VLLMConfigError: If vLLM configuration is invalid
            ImportError: If vLLM is not installed
        """
        # Initialize base agent (validates common fields)
        super().__init__(agent_config, model_config)

        # Model path (HuggingFace model ID or local path)
        self.model_path = model_config.get("model_path")
        if not self.model_path:
            raise VLLMConfigError(
                "model_path is required in model_config for vLLM. "
                "Example: 'meta-llama/Llama-3.1-8B-Instruct' or '/path/to/model'"
            )

        # Get and validate vLLM configuration
        self.vllm_config = model_config.get("vllm_config", {})
        self._validate_vllm_config()

        # Initialize vLLM engine (may reuse shared instance)
        self._initialize_vllm_engine()

    def _validate_vllm_config(self) -> None:
        """Validate vLLM configuration parameters.

        Raises:
            VLLMConfigError: If configuration is invalid
        """
        # Validate gpu_memory_utilization range
        gpu_mem = self.vllm_config.get("gpu_memory_utilization")
        if gpu_mem is not None and not 0.0 < gpu_mem <= 1.0:
            raise VLLMConfigError(
                f"gpu_memory_utilization must be between 0.0 and 1.0, got {gpu_mem}"
            )

        # Validate max_model_len is positive
        max_len = self.vllm_config.get("max_model_len")
        if max_len is not None and max_len <= 0:
            raise VLLMConfigError(f"max_model_len must be positive, got {max_len}")

        # Validate max_num_seqs is positive
        max_seqs = self.vllm_config.get("max_num_seqs")
        if max_seqs is not None and max_seqs <= 0:
            raise VLLMConfigError(f"max_num_seqs must be positive, got {max_seqs}")

        # Validate tensor_parallel_size is positive
        tp_size = self.vllm_config.get("tensor_parallel_size")
        if tp_size is not None and tp_size <= 0:
            raise VLLMConfigError(
                f"tensor_parallel_size must be positive, got {tp_size}"
            )

        # Warn about unknown parameters
        unknown = set(self.vllm_config.keys()) - set(self.VLLM_CONFIG_PARAMS.keys())
        if unknown:
            _debug_print(f"Unknown vllm_config parameters (will be ignored): {unknown}")

    def _initialize_vllm_engine(self) -> None:
        """Initialize or retrieve vLLM engine instance.

        Uses class-level cache to share model instances across agents
        (for shared backbone configuration).

        Raises:
            ImportError: If vLLM is not installed
            VLLMConfigError: If engine initialization fails
        """
        # Ensure CUDA paths are set correctly before importing vLLM
        self._setup_cuda_env()

        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM is not installed. Install with: pip install vllm>=0.11.2"
            ) from e

        # Store SamplingParams class for later use
        self.SamplingParams = SamplingParams

        # Check if we can reuse a shared model instance
        cache_key = self.model_path

        if cache_key in self._shared_models:
            self.llm = self._shared_models[cache_key]
            _info_print(f"  ✓ Reusing shared vLLM engine for {self.agent_id}")
            return

        # Create new model instance
        _info_print(f"  ✓ Initializing vLLM engine for {self.model_path}")

        # Build vLLM initialization arguments
        llm_kwargs = self._build_llm_kwargs()

        _debug_print(f"LLM() kwargs: {llm_kwargs}")

        try:
            self.llm = LLM(**llm_kwargs)
        except Exception as e:
            # Provide more specific error messages for common failures
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "oom" in error_msg:
                raise VLLMConfigError(
                    f"GPU out of memory loading {self.model_path}. "
                    f"Try reducing gpu_memory_utilization (current: {llm_kwargs.get('gpu_memory_utilization')}) "
                    f"or max_num_seqs (current: {llm_kwargs.get('max_num_seqs', 'default')})"
                ) from e
            elif "not found" in error_msg or "does not exist" in error_msg:
                raise VLLMConfigError(
                    f"Model {self.model_path} not found. "
                    f"Ensure it's downloaded: huggingface-cli download {self.model_path}"
                ) from e
            else:
                raise VLLMConfigError(f"Failed to initialize vLLM engine: {e}") from e

        # Cache for shared use
        self._shared_models[cache_key] = self.llm
        _info_print("  ✓ vLLM engine initialized successfully")

    def _setup_cuda_env(self) -> None:
        """Setup CUDA environment variables if CUDA_HOME is set."""
        if "CUDA_HOME" in os.environ:
            cuda_home = os.environ["CUDA_HOME"]
            os.environ["CUDACXX"] = f"{cuda_home}/bin/nvcc"
            if f"{cuda_home}/bin" not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{cuda_home}/bin:" + os.environ.get("PATH", "")

    def _build_llm_kwargs(self) -> Dict[str, Any]:
        """Build keyword arguments for vLLM LLM constructor.

        Returns:
            Dictionary of validated vLLM initialization parameters
        """
        # Start with required model path
        llm_kwargs: Dict[str, Any] = {"model": self.model_path}

        # Add parameters from vllm_config, using explicit defaults only when not specified
        # This ensures users see what defaults are being applied

        # Parameters with explicit defaults (always set)
        llm_kwargs["tensor_parallel_size"] = self.vllm_config.get(
            "tensor_parallel_size", 1
        )
        llm_kwargs["gpu_memory_utilization"] = self.vllm_config.get(
            "gpu_memory_utilization", 0.9
        )
        llm_kwargs["max_model_len"] = self.vllm_config.get("max_model_len", 4096)
        llm_kwargs["dtype"] = self.vllm_config.get("dtype", "auto")

        # Optional parameters (only add if explicitly set)
        optional_params = [
            "max_num_seqs",
            "enable_prefix_caching",
            "swap_space",
            "enforce_eager",
            "quantization",
        ]
        for param in optional_params:
            if param in self.vllm_config:
                llm_kwargs[param] = self.vllm_config[param]

        # Handle gpu_device_ids -> tensor_parallel_size conversion
        if "gpu_device_ids" in self.vllm_config:
            llm_kwargs["tensor_parallel_size"] = len(self.vllm_config["gpu_device_ids"])

        return llm_kwargs

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: str = "json",
    ) -> Dict[str, Any]:
        """Generate response using vLLM with proper metrics.

        Args:
            prompt: Input prompt
            temperature: Override temperature (optional, uses agent default if None)
            max_tokens: Override max tokens (optional, uses agent default if None)
            response_format: Expected format ("json" or "text")

        Returns:
            Dictionary containing:
                - text: Raw text response
                - structured_output: Parsed JSON (if json format), None otherwise
                - tokens_generated: Actual tokens in response
                - tokens_prompt: Prompt tokens
                - generation_time_ms: Generation time in milliseconds
                - exceeded_max_tokens: Whether max_tokens was hit

        Raises:
            VLLMInferenceError: If vLLM inference fails
        """
        # Use provided values or fall back to agent defaults
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        # Validate overridden parameters
        if temperature is not None:
            self._validate_generation_params(temp, max_tok)

        # Build the full prompt with chat template
        full_prompt = self._build_full_prompt(prompt, response_format)

        # Debug output
        _debug_print(
            f"Full prompt ({len(full_prompt)} chars):\n{full_prompt[:1000]}..."
        )

        # Build sampling parameters
        sampling_params = self._build_sampling_params(temp, max_tok)

        # Generate
        start_time = time.time()
        try:
            outputs = self.llm.generate([full_prompt], sampling_params)
            generation_time_ms = round((time.time() - start_time) * 1000, 3)

            output = outputs[0]
            response_text = output.outputs[0].text

            # Get token counts from vLLM metrics
            tokens_prompt = len(output.prompt_token_ids)
            tokens_generated = len(output.outputs[0].token_ids)

            # Check if we hit max tokens (vLLM sets finish_reason to "length")
            exceeded_max_tokens = output.outputs[0].finish_reason == "length"

        except Exception as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "oom" in error_msg:
                raise VLLMInferenceError(
                    f"GPU OOM during inference. Try reducing max_tokens (current: {max_tok}) "
                    f"or max_num_seqs in vllm_config."
                ) from e
            raise VLLMInferenceError(f"vLLM inference failed: {e}") from e

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

    def _build_full_prompt(self, prompt: str, response_format: str) -> str:
        """Build full prompt with system message and chat template.

        Args:
            prompt: User prompt
            response_format: Expected format ("json" or "text")

        Returns:
            Full prompt string with chat template applied
        """
        system_prompt = self._build_system_prompt()

        # Build user message with JSON instruction if needed
        user_message = prompt
        if response_format == "json":
            user_message += (
                "\n\nYou must respond with valid JSON only. "
                "Do not include any text outside the JSON object."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Use vLLM's tokenizer to apply model-specific chat template
        return self.llm.get_tokenizer().apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _build_sampling_params(self, temperature: float, max_tokens: int) -> Any:
        """Build vLLM SamplingParams object.

        Args:
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            SamplingParams instance
        """
        sampling_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Only add top_p and top_k if explicitly set in config
        if self.vllm_config.get("top_p") is not None:
            sampling_kwargs["top_p"] = self.vllm_config["top_p"]
        if self.vllm_config.get("top_k") is not None:
            sampling_kwargs["top_k"] = self.vllm_config["top_k"]

        return self.SamplingParams(**sampling_kwargs)

    def __repr__(self) -> str:
        """String representation of agent."""
        return (
            f"VLLMAgent(id={self.agent_id}, role={self.role}, model={self.model_path})"
        )

    @classmethod
    def get_model_cache_info(cls) -> Dict[str, Any]:
        """Get information about cached model instances.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_models": list(cls._shared_models.keys()),
            "num_cached": len(cls._shared_models),
        }

    @classmethod
    def collect_metrics_snapshot(cls) -> Optional[Dict[str, Any]]:
        """Collect a metrics snapshot from the shared vLLM engine.

        This method collects vLLM metrics useful for parameter tuning:
        - KV cache usage
        - Prefix cache hit rate
        - Request queue state
        - Preemption count

        Returns:
            Dictionary with metrics, or None if no model is loaded
        """
        if not cls._shared_models:
            return None

        # Get the first (typically only) shared model
        llm = next(iter(cls._shared_models.values()))

        try:
            from src.utils.vllm_metrics import VLLMMetricsCollector

            collector = VLLMMetricsCollector()
            snapshot = collector.collect_snapshot(llm)
            return {
                "kv_cache_usage_perc": snapshot.kv_cache_usage_perc,
                "prefix_cache_queries": snapshot.prefix_cache_queries,
                "prefix_cache_hits": snapshot.prefix_cache_hits,
                "prefix_cache_hit_rate": snapshot.prefix_cache_hit_rate,
                "num_requests_running": snapshot.num_requests_running,
                "num_requests_waiting": snapshot.num_requests_waiting,
                "num_preemptions_total": snapshot.num_preemptions_total,
            }
        except Exception as e:
            _debug_print(f"Failed to collect metrics: {e}")
            return None

    @classmethod
    def cleanup_all_models(cls) -> None:
        """Release all cached vLLM models and free GPU memory.

        This should be called at the end of the experiment to ensure
        GPU resources are properly released.
        """
        if not cls._shared_models:
            return

        _info_print(f"  Releasing {len(cls._shared_models)} cached model(s)...")

        # Delete all cached models
        for model_path in list(cls._shared_models.keys()):
            del cls._shared_models[model_path]

        cls._shared_models.clear()

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache
        try:
            import torch

            torch.cuda.empty_cache()
            _info_print("  ✓ GPU memory released")
        except ImportError:
            pass  # torch not available
