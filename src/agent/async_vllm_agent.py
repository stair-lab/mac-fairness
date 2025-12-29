"""Async vLLM agent using AsyncLLMEngine.

This agent provides native async interface to vLLM's AsyncLLMEngine.
vLLM handles continuous batching internally.
"""

import gc
import os
import pprint
import re
import select
import sys
import threading
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Dict, List, Optional
from uuid import uuid4

# Suppress FutureWarning from mistral_common tokenizer (deprecated special token policy)
warnings.filterwarnings("ignore", category=FutureWarning, module="mistral_common")

from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from jinja2.exceptions import TemplateError

from .base_agent import BaseAgent
from src.utils import debug_print, info_print, is_debug_enabled
from src.utils.errors import (
    VLLMBatchError,
    VLLMEngineNotInitializedError,
    VLLMEngineNotStartedError,
    VLLMOOMError,
)


class VLLMConfigError(ValueError):
    """Raised when vLLM configuration is invalid."""

    pass


class VLLMOutputCapture:
    """Capture vLLM's Maximum concurrency from stdout/stderr during engine initialization.

    vLLM v1 runs EngineCore in a subprocess which writes directly to file descriptors,
    bypassing Python's sys.stdout/sys.stderr. We use os.dup2() to redirect at the
    file descriptor level, with a pipe to capture while still forwarding output.

    Output is only forwarded to stdout when debug mode is enabled (MAC_FAIRNESS_DEBUG_FLAG=1).
    """

    # Pattern matches: "Maximum concurrency for 2,048 tokens per request: 110.31x"
    _pattern = re.compile(
        r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x"
    )

    def __init__(self) -> None:
        self.max_concurrency: Optional[float] = None
        self._captured_output: str = ""
        self._original_stdout_fd: Optional[int] = None
        self._original_stderr_fd: Optional[int] = None
        self._saved_stdout_fd: Optional[int] = None
        self._saved_stderr_fd: Optional[int] = None
        self._pipe_read_fd: Optional[int] = None
        self._pipe_write_fd: Optional[int] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._forward_output: bool = is_debug_enabled()

    def __enter__(self) -> "VLLMOutputCapture":
        """Start capturing stdout/stderr at the file descriptor level."""
        # Create a pipe for capturing output
        self._pipe_read_fd, self._pipe_write_fd = os.pipe()

        # Save original file descriptors
        self._original_stdout_fd = sys.stdout.fileno()
        self._original_stderr_fd = sys.stderr.fileno()
        self._saved_stdout_fd = os.dup(self._original_stdout_fd)
        self._saved_stderr_fd = os.dup(self._original_stderr_fd)

        # Redirect stdout and stderr to the pipe
        os.dup2(self._pipe_write_fd, self._original_stdout_fd)
        os.dup2(self._pipe_write_fd, self._original_stderr_fd)

        # Start a thread to read from the pipe and forward to original stdout
        self._stop_event = threading.Event()
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True
        )
        self._reader_thread.start()

        return self

    def _reader_loop(self) -> None:
        """Read from pipe, capture, and optionally forward to original stdout."""
        buffer = []
        while not self._stop_event.is_set():
            try:
                # Non-blocking read with small timeout
                readable, _, _ = select.select([self._pipe_read_fd], [], [], 0.1)
                if readable:
                    data = os.read(self._pipe_read_fd, 4096)
                    if data:
                        # Forward to original stdout only in debug mode
                        if self._forward_output:
                            os.write(self._saved_stdout_fd, data)
                        # Capture for parsing
                        buffer.append(data.decode("utf-8", errors="replace"))
            except Exception:
                break

        # Read any remaining data
        try:
            while True:
                readable, _, _ = select.select([self._pipe_read_fd], [], [], 0)
                if not readable:
                    break
                data = os.read(self._pipe_read_fd, 4096)
                if not data:
                    break
                if self._forward_output:
                    os.write(self._saved_stdout_fd, data)
                buffer.append(data.decode("utf-8", errors="replace"))
        except Exception:
            pass

        self._captured_output = "".join(buffer)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop capturing and parse the output."""
        # Signal reader thread to stop
        if self._stop_event:
            self._stop_event.set()

        # Restore original file descriptors
        if self._saved_stdout_fd is not None and self._original_stdout_fd is not None:
            os.dup2(self._saved_stdout_fd, self._original_stdout_fd)
        if self._saved_stderr_fd is not None and self._original_stderr_fd is not None:
            os.dup2(self._saved_stderr_fd, self._original_stderr_fd)

        # Close pipe write end to signal EOF to reader
        if self._pipe_write_fd is not None:
            os.close(self._pipe_write_fd)

        # Wait for reader thread to finish
        if self._reader_thread:
            self._reader_thread.join(timeout=2.0)

        # Close remaining file descriptors
        if self._pipe_read_fd is not None:
            os.close(self._pipe_read_fd)
        if self._saved_stdout_fd is not None:
            os.close(self._saved_stdout_fd)
        if self._saved_stderr_fd is not None:
            os.close(self._saved_stderr_fd)

        # Parse captured output for max concurrency
        self._parse_output(self._captured_output)

    def _parse_output(self, text: str) -> None:
        """Extract max concurrency from captured text."""
        match = self._pattern.search(text)
        if match:
            self.max_concurrency = float(match.group(1))


@dataclass
class RequestTiming:
    """Timing data for a single vLLM request."""

    request_id: str
    model_path: str
    start_time: float
    end_time: float
    tokens_prompt: int
    tokens_generated: int
    agent_id: str

    @property
    def latency_seconds(self) -> float:
        """Total request latency in seconds."""
        return self.end_time - self.start_time

    @property
    def tokens_per_second(self) -> float:
        """Generation throughput (output tokens per second)."""
        if self.latency_seconds > 0:
            return self.tokens_generated / self.latency_seconds
        return 0.0


@dataclass
class RequestMetricsCollector:
    """Collects and aggregates request-level timing metrics.

    Also tracks concurrent request patterns to understand batching behavior.
    """

    timings: List[RequestTiming] = field(default_factory=list)
    _active_requests: int = field(default=0, repr=False)
    _peak_concurrent: int = field(default=0, repr=False)
    _concurrency_samples: List[int] = field(default_factory=list, repr=False)

    def record_start(self) -> None:
        """Record that a request has started (for concurrency tracking)."""
        self._active_requests += 1
        if self._active_requests > self._peak_concurrent:
            self._peak_concurrent = self._active_requests
        # Cap samples to avoid unbounded memory growth (keep last 10K for avg calc)
        if len(self._concurrency_samples) < 10000:
            self._concurrency_samples.append(self._active_requests)

    def record_end(self) -> None:
        """Record that a request has ended."""
        self._active_requests = max(0, self._active_requests - 1)

    def record(self, timing: RequestTiming) -> None:
        """Record a request timing."""
        self.timings.append(timing)

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated metrics summary."""
        if not self.timings:
            return {
                "total_requests": 0,
                "total_latency_seconds": 0,
                "avg_latency_seconds": 0,
                "min_latency_seconds": 0,
                "max_latency_seconds": 0,
                "total_tokens_generated": 0,
                "total_tokens_prompt": 0,
                "avg_tokens_per_second": 0,
                "per_model": {},
            }

        latencies = [t.latency_seconds for t in self.timings]
        tps_values = [t.tokens_per_second for t in self.timings]

        # Per-model breakdown
        per_model: Dict[str, Dict[str, Any]] = {}
        for t in self.timings:
            if t.model_path not in per_model:
                per_model[t.model_path] = {
                    "requests": 0,
                    "total_latency": 0.0,
                    "tokens_generated": 0,
                    "tokens_prompt": 0,
                }
            per_model[t.model_path]["requests"] += 1
            per_model[t.model_path]["total_latency"] += t.latency_seconds
            per_model[t.model_path]["tokens_generated"] += t.tokens_generated
            per_model[t.model_path]["tokens_prompt"] += t.tokens_prompt

        # Calculate per-model averages
        for model_path, stats in per_model.items():
            if stats["requests"] > 0:
                stats["avg_latency_seconds"] = round(
                    stats["total_latency"] / stats["requests"], 4
                )
                if stats["total_latency"] > 0:
                    stats["tokens_per_second"] = round(
                        stats["tokens_generated"] / stats["total_latency"], 2
                    )
                else:
                    stats["tokens_per_second"] = 0.0
            stats["total_latency"] = round(stats["total_latency"], 4)

        # Concurrency metrics
        avg_concurrency = (
            round(sum(self._concurrency_samples) / len(self._concurrency_samples), 2)
            if self._concurrency_samples
            else 0
        )

        return {
            "total_requests": len(self.timings),
            "total_latency_seconds": round(sum(latencies), 4),
            "avg_latency_seconds": round(sum(latencies) / len(latencies), 4),
            "min_latency_seconds": round(min(latencies), 4),
            "max_latency_seconds": round(max(latencies), 4),
            "total_tokens_generated": sum(t.tokens_generated for t in self.timings),
            "total_tokens_prompt": sum(t.tokens_prompt for t in self.timings),
            "avg_tokens_per_second": round(sum(tps_values) / len(tps_values), 2)
            if tps_values
            else 0,
            "concurrency": {
                "peak_concurrent_requests": self._peak_concurrent,
                "avg_concurrent_requests": avg_concurrency,
            },
            "per_model": per_model,
        }

    def clear(self) -> None:
        """Clear all recorded timings and concurrency data."""
        self.timings.clear()
        self._active_requests = 0
        self._peak_concurrent = 0
        self._concurrency_samples.clear()


class AsyncVLLMAgent(BaseAgent):
    """Async vLLM agent using AsyncLLMEngine.

    Uses vLLM's native AsyncLLMEngine for true async generation.
    vLLM handles continuous batching internally.

    Multi-Model Support:
    - Each model_path gets its own AsyncLLMEngine
    - Agents with the same model_path share the engine
    """

    # Class-level cache for AsyncLLMEngine instances (keyed by model_path)
    _engines: ClassVar[Dict[str, Any]] = {}  # AsyncLLMEngine
    _engines_started: ClassVar[Dict[str, bool]] = {}
    _tokenizers: ClassVar[Dict[str, Any]] = {}
    _sampling_params_class: ClassVar[Any] = None
    _no_system_role_models: ClassVar[set] = set()  # Models that don't support system role

    # Metrics
    _total_requests: ClassVar[Dict[str, int]] = {}
    _metrics_collector: ClassVar[RequestMetricsCollector] = RequestMetricsCollector()

    # Captured max concurrency from vLLM logs (handles hybrid attention models correctly)
    _max_concurrency_from_log: ClassVar[Dict[str, Optional[float]]] = {}

    # Supported vLLM config parameters
    VLLM_CONFIG_PARAMS = {
        "tensor_parallel_size": int,
        "gpu_memory_utilization": float,
        "max_model_len": int,
        "dtype": str,
        "max_num_seqs_upper_bound": int,  # user-facing config key (actual value limited by KV cache)
        "enable_prefix_caching": bool,
        "swap_space": int,
        "enforce_eager": bool,
        "quantization": str,
        "gpu_device_ids": list,
        "attention_backend": str,
    }

    def __init__(self, agent_config: Dict[str, Any], model_config: Dict[str, Any]):
        """Initialize async vLLM agent.

        Args:
            agent_config: Agent configuration dictionary
            model_config: Model configuration dictionary containing:
                - model_path: HuggingFace model ID or local path (required)
                - vllm_config: Dict of vLLM-specific settings (optional)

        Raises:
            ValueError: If required fields are missing
            VLLMConfigError: If vLLM configuration is invalid
        """
        super().__init__(agent_config, model_config)

        self.model_path = model_config.get("model_path")
        if not self.model_path:
            raise VLLMConfigError(
                "model_path is required in model_config for vLLM. "
                "Example: 'meta-llama/Llama-3.1-8B-Instruct'"
            )

        self.vllm_config = model_config.get("vllm_config", {})
        self._validate_vllm_config()
        self._ensure_engine()

    def _validate_vllm_config(self) -> None:
        """Validate vLLM configuration parameters."""
        gpu_mem = self.vllm_config.get("gpu_memory_utilization")
        if gpu_mem is not None and not 0.0 < gpu_mem <= 1.0:
            raise VLLMConfigError(
                f"gpu_memory_utilization must be between 0.0 and 1.0, got {gpu_mem}"
            )

        max_len = self.vllm_config.get("max_model_len")
        if max_len is not None and max_len <= 0:
            raise VLLMConfigError(f"max_model_len must be positive, got {max_len}")

        max_seqs = self.vllm_config.get("max_num_seqs_upper_bound")
        if max_seqs is not None and max_seqs <= 0:
            raise VLLMConfigError(
                f"max_num_seqs_upper_bound must be positive, got {max_seqs}"
            )

        tp_size = self.vllm_config.get("tensor_parallel_size")
        if tp_size is not None and tp_size <= 0:
            raise VLLMConfigError(
                f"tensor_parallel_size must be positive, got {tp_size}"
            )

        unknown = set(self.vllm_config.keys()) - set(self.VLLM_CONFIG_PARAMS.keys())
        if unknown:
            debug_print(f"Unknown vllm_config parameters (ignored): {unknown}")

    def _ensure_engine(self) -> None:
        """Ensure AsyncLLMEngine is initialized for this model."""
        if self.model_path in self._engines:
            info_print(f"Reusing shared AsyncLLMEngine for {self.agent_id}")
            return

        self._setup_cuda_env()

        AsyncVLLMAgent._sampling_params_class = SamplingParams

        info_print(f"Initializing AsyncLLMEngine for {self.model_path}")

        engine_args = self._build_engine_args()
        if is_debug_enabled():
            print("[DEBUG] AsyncEngineArgs:")
            pprint.pprint(asdict(engine_args), width=100, sort_dicts=True)

        # Capture vLLM's "Maximum concurrency" from stdout/stderr during initialization
        # vLLM v1 runs EngineCore in a subprocess, so logs go to stdout/stderr
        # This value correctly handles hybrid attention models (Gemma2, Ministral, etc.)
        output_capture = VLLMOutputCapture()

        with output_capture:
            try:
                async_engine = AsyncLLMEngine.from_engine_args(engine_args)
            except Exception as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg or "oom" in error_msg:
                    raise VLLMConfigError(
                        f"GPU out of memory loading {self.model_path}. "
                        f"Try reducing gpu_memory_utilization or max_num_seqs"
                    ) from e
                elif "not found" in error_msg or "does not exist" in error_msg:
                    raise VLLMConfigError(
                        f"Model {self.model_path} not found. "
                        f"Ensure it's downloaded: huggingface-cli download {self.model_path}"
                    ) from e
                raise VLLMConfigError(f"Failed to initialize AsyncLLMEngine: {e}") from e

        # Store captured max concurrency for later use
        self._max_concurrency_from_log[self.model_path] = output_capture.max_concurrency

        # Load tokenizer with fallbacks for custom tokenizers (e.g., Mistral's tekken)
        tokenizer = self._load_tokenizer()
        self._tokenizers[self.model_path] = tokenizer

        self._engines[self.model_path] = async_engine
        self._engines_started[self.model_path] = False
        self._total_requests[self.model_path] = 0

        # Extract effective max_num_seqs from vLLM (may be lower than config due to KV cache)
        effective_max_num_seqs = self._get_effective_max_num_seqs(
            async_engine, output_capture.max_concurrency
        )
        config_max_num_seqs = self.vllm_config.get("max_num_seqs_upper_bound")
        if effective_max_num_seqs and config_max_num_seqs:
            if effective_max_num_seqs < config_max_num_seqs:
                info_print(
                    f"max_num_seqs: {config_max_num_seqs} (upper bound) → {effective_max_num_seqs} "
                    f"(effective, limited by KV cache)"
                )

        info_print("AsyncLLMEngine initialized successfully")

    def _load_tokenizer(self) -> Any:
        """Load tokenizer with fallbacks for custom tokenizers.

        Tries multiple approaches to handle custom tokenizers like Mistral's tekken:
        1. MistralCommonBackend (transformers v5+)
        2. mistral_common.tokens.tokenizers directly (for tekken tokenizer)
        3. AutoTokenizer with trust_remote_code=True
        4. AutoTokenizer without trust_remote_code

        Returns:
            Loaded tokenizer instance

        Raises:
            VLLMConfigError: If all tokenizer loading methods fail
        """
        # Try MistralCommonBackend first (transformers v5+ only)
        try:
            from transformers import MistralCommonBackend

            tokenizer = MistralCommonBackend.from_pretrained(self.model_path)
            info_print("Loaded tokenizer using MistralCommonBackend")
            return tokenizer
        except ImportError:
            pass  # MistralCommonBackend not available (requires transformers v5+)
        except Exception:
            pass  # Model doesn't use MistralCommonBackend

        # Try mistral_common directly (for Mistral models with tekken tokenizer)
        try:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

            tokenizer = MistralTokenizer.from_hf_hub(self.model_path)
            info_print("Loaded tokenizer using mistral_common.MistralTokenizer")
            return tokenizer
        except ImportError:
            pass  # mistral_common not installed
        except Exception:
            pass  # Model doesn't use MistralTokenizer

        # Try AutoTokenizer with trust_remote_code
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            return tokenizer
        except Exception:
            pass

        # Try AutoTokenizer without trust_remote_code
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            return tokenizer
        except Exception as e:
            raise VLLMConfigError(
                f"Failed to load tokenizer for {self.model_path}. "
                f"For Mistral models, try: uv pip install 'transformers>=5.0.0.rc1' "
                f"or ensure mistral-common>=1.8.6 is installed. "
                f"Error: {e}"
            ) from e

    @staticmethod
    def _get_effective_max_num_seqs(
        engine: Any, max_concurrency_from_log: Optional[float] = None
    ) -> Optional[int]:
        """Extract effective max_num_seqs from vLLM engine.

        Uses vLLM's logged "Maximum concurrency" value when available, which correctly
        handles hybrid attention models (Gemma2, Ministral, etc.) that have different
        KV cache requirements per layer.

        Falls back to manual calculation if log capture failed.

        Note: vLLM v1 runs EngineCore in a subprocess, so log capture may not work.
        The fallback calculation may overestimate capacity for hybrid attention models.
        Check vLLM's "Maximum concurrency" log output for the accurate value.

        Args:
            engine: AsyncLLMEngine instance
            max_concurrency_from_log: Value captured from vLLM's log output during
                engine initialization. This is the most accurate value as it accounts
                for hybrid attention architectures.

        Returns:
            Effective max_num_seqs (limited by KV cache) or None if not accessible
        """
        try:
            vllm_config = engine.vllm_config
            scheduler_config = vllm_config.scheduler_config
            config_max_num_seqs = scheduler_config.max_num_seqs

            # Prefer the captured log value (handles hybrid attention correctly)
            if max_concurrency_from_log is not None:
                kv_cache_capacity = int(max_concurrency_from_log)
                return min(config_max_num_seqs, kv_cache_capacity)

            # Fallback: manual calculation
            # Note: This may overestimate for hybrid attention models (Gemma2, Ministral)
            # which interleave sliding window and full attention layers
            cache_config = vllm_config.cache_config
            model_config = vllm_config.model_config

            num_gpu_blocks = cache_config.num_gpu_blocks
            block_size = cache_config.block_size
            max_model_len = model_config.max_model_len

            if num_gpu_blocks and block_size and max_model_len:
                kv_cache_capacity = int(num_gpu_blocks * block_size / max_model_len)
                return min(config_max_num_seqs, kv_cache_capacity)

            return config_max_num_seqs
        except AttributeError:
            return None

    @classmethod
    def get_effective_backend_config(cls) -> Dict[str, Dict[str, Any]]:
        """Get effective backend configuration for all initialized models.

        Returns actual values being used by vLLM (which may differ from config
        due to KV cache constraints, auto-detection, etc).

        Returns:
            Dict keyed by model_path with effective config values
        """
        result = {}
        for model_path, engine in cls._engines.items():
            config: Dict[str, Any] = {"model_path": model_path}

            # Get effective max_num_seqs (use captured log value if available)
            max_concurrency_from_log = cls._max_concurrency_from_log.get(model_path)
            effective_max_num_seqs = cls._get_effective_max_num_seqs(
                engine, max_concurrency_from_log
            )
            if effective_max_num_seqs is not None:
                config["max_num_seqs"] = effective_max_num_seqs

            result[model_path] = config

        return result

    def _setup_cuda_env(self) -> None:
        """Setup CUDA environment variables if CUDA_HOME is set."""
        if "CUDA_HOME" in os.environ:
            cuda_home = os.environ["CUDA_HOME"]
            os.environ["CUDACXX"] = f"{cuda_home}/bin/nvcc"
            if f"{cuda_home}/bin" not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{cuda_home}/bin:" + os.environ.get("PATH", "")

    def _build_engine_args(self) -> AsyncEngineArgs:
        """Build AsyncEngineArgs for vLLM."""
        kwargs: Dict[str, Any] = {"model": self.model_path}

        kwargs["tensor_parallel_size"] = self.vllm_config.get("tensor_parallel_size", 1)
        kwargs["gpu_memory_utilization"] = self.vllm_config.get(
            "gpu_memory_utilization", 0.9
        )
        kwargs["max_model_len"] = self.vllm_config.get("max_model_len", 4096)
        kwargs["dtype"] = self.vllm_config.get("dtype", "auto")

        # Map max_num_seqs_upper_bound to vLLM's max_num_seqs
        if "max_num_seqs_upper_bound" in self.vllm_config:
            kwargs["max_num_seqs"] = self.vllm_config["max_num_seqs_upper_bound"]

        optional_params = [
            "enable_prefix_caching",
            "swap_space",
            "enforce_eager",
            "quantization",
        ]
        for param in optional_params:
            if param in self.vllm_config:
                kwargs[param] = self.vllm_config[param]

        if "gpu_device_ids" in self.vllm_config:
            kwargs["tensor_parallel_size"] = len(self.vllm_config["gpu_device_ids"])

        # Set attention backend via environment variable (vLLM uses env var, not engine arg)
        if "attention_backend" in self.vllm_config:
            os.environ["VLLM_ATTENTION_BACKEND"] = self.vllm_config["attention_backend"]

        return AsyncEngineArgs(**kwargs)

    def _build_full_prompt(self, prompt: str, response_format: str) -> str:
        """Build full prompt with system message and chat template.

        Handles models that don't support system roles (e.g., Gemma 2) by
        prepending the system message to the user message.

        Also handles MistralTokenizer from mistral_common which uses a different API.
        """
        system_prompt = self._build_system_prompt()

        user_message = prompt
        if response_format == "json":
            user_message += (
                "\n\nYou must respond with valid JSON only. "
                "Do not include any text outside the JSON object."
            )

        tokenizer = self._tokenizers.get(self.model_path)
        if tokenizer is None:
            raise VLLMEngineNotInitializedError()

        # Check if this is a MistralTokenizer from mistral_common
        if self._is_mistral_tokenizer(tokenizer):
            return self._build_prompt_mistral(tokenizer, system_prompt, user_message)

        # Build kwargs for apply_chat_template (HuggingFace tokenizers)
        chat_template_kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = self.enable_thinking

        # Check if we already know this model doesn't support system role
        if self.model_path in self._no_system_role_models:
            combined_user_message = f"{system_prompt}\n\n{user_message}"
            messages = [{"role": "user", "content": combined_user_message}]
            return tokenizer.apply_chat_template(messages, **chat_template_kwargs)

        # Try with system role first
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            return tokenizer.apply_chat_template(messages, **chat_template_kwargs)
        except TemplateError as e:
            # Some models (e.g., Gemma 2) don't support system roles
            # Fall back to prepending system message to user message
            if "system role not supported" in str(e).lower():
                # Cache this model as not supporting system role (log once)
                self._no_system_role_models.add(self.model_path)
                info_print(
                    f"Model {self.model_path} doesn't support system role, "
                    "prepending to user message"
                )
                combined_user_message = f"{system_prompt}\n\n{user_message}"
                messages_no_system = [
                    {"role": "user", "content": combined_user_message},
                ]
                return tokenizer.apply_chat_template(
                    messages_no_system, **chat_template_kwargs
                )
            raise

    def _is_mistral_tokenizer(self, tokenizer: Any) -> bool:
        """Check if tokenizer is a MistralTokenizer from mistral_common."""
        try:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

            return isinstance(tokenizer, MistralTokenizer)
        except ImportError:
            return False

    def _build_prompt_mistral(
        self, tokenizer: Any, system_prompt: str, user_message: str
    ) -> str:
        """Build prompt using MistralTokenizer from mistral_common.

        MistralTokenizer uses encode_chat_completion instead of apply_chat_template.
        """
        from mistral_common.protocol.instruct.messages import (
            SystemMessage,
            UserMessage,
        )
        from mistral_common.protocol.instruct.request import ChatCompletionRequest

        messages = [
            SystemMessage(content=system_prompt),
            UserMessage(content=user_message),
        ]

        request = ChatCompletionRequest(messages=messages)
        tokenized = tokenizer.encode_chat_completion(request)

        # Return the text representation for vLLM
        # MistralTokenizer returns tokens, we need to decode them
        return tokenizer.decode(tokenized.tokens)

    @classmethod
    async def start_engine(cls) -> None:
        """Mark engines as started."""
        if not cls._engines:
            raise VLLMEngineNotInitializedError()

        for model_path in cls._engines:
            if not cls._engines_started.get(model_path, False):
                cls._engines_started[model_path] = True
                info_print(f"AsyncLLMEngine started for {model_path}")

    @classmethod
    async def stop_engine(cls) -> None:
        """Mark engines as stopped."""
        for model_path in cls._engines:
            if cls._engines_started.get(model_path, False):
                cls._engines_started[model_path] = False
                info_print(f"AsyncLLMEngine stopped for {model_path}")

    @classmethod
    def get_engine_metrics(cls) -> Optional[Dict[str, Any]]:
        """Get metrics from all engines including detailed timing.

        Returns:
            Dictionary with:
                - total_requests: Total request count
                - per_model: Per-model request counts
                - timing: Detailed timing metrics from RequestMetricsCollector
        """
        if not cls._engines:
            return None

        total = sum(cls._total_requests.values())
        per_model = {k: {"total_requests": v} for k, v in cls._total_requests.items()}

        return {
            "total_requests": total,
            "per_model": per_model,
            "timing": cls._metrics_collector.get_summary(),
        }

    @classmethod
    def get_timing_metrics(cls) -> Dict[str, Any]:
        """Get detailed timing metrics only.

        Returns:
            Aggregated timing metrics from all requests.
        """
        return cls._metrics_collector.get_summary()

    @classmethod
    def clear_timing_metrics(cls) -> None:
        """Clear accumulated timing metrics."""
        cls._metrics_collector.clear()

    @classmethod
    def get_effective_config(cls) -> Dict[str, Dict[str, Any]]:
        """Get effective configuration for all initialized models.

        This is an alias for get_effective_backend_config() for backward compatibility.
        """
        return cls.get_effective_backend_config()

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: str = "json",
    ) -> Dict[str, Any]:
        """Generate response using vLLM asynchronously.

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
                - tokens_prompt: Prompt tokens
                - exceeded_max_tokens: Whether max_tokens was hit

        Raises:
            VLLMEngineNotStartedError: If engine not started
            VLLMEngineNotInitializedError: If engine not initialized
            VLLMBatchError: If generation fails
            VLLMOOMError: If GPU runs out of memory
        """
        engine = self._engines.get(self.model_path)
        if engine is None:
            raise VLLMEngineNotInitializedError()
        if not self._engines_started.get(self.model_path, False):
            raise VLLMEngineNotStartedError()

        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        if temperature is not None:
            self._validate_generation_params(temp, max_tok)

        full_prompt = self._build_full_prompt(prompt, response_format)

        # Build sampling params
        sampling_kwargs: Dict[str, Any] = {
            "temperature": temp,
            "max_tokens": max_tok,
        }
        if self.top_p is not None:
            sampling_kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            sampling_kwargs["top_k"] = self.top_k
        if self.min_p is not None:
            sampling_kwargs["min_p"] = self.min_p
        if self.presence_penalty is not None:
            sampling_kwargs["presence_penalty"] = self.presence_penalty

        sampling_params = self._sampling_params_class(**sampling_kwargs)
        request_id = str(uuid4())

        self._metrics_collector.record_start()
        start_time = time.perf_counter()
        try:
            final_output = None
            async for output in engine.generate(
                full_prompt, sampling_params, request_id
            ):
                final_output = output
            end_time = time.perf_counter()

            if final_output is None:
                raise VLLMBatchError(
                    message="No output received from engine",
                    batch_size=1,
                    request_ids=[request_id],
                )

            self._total_requests[self.model_path] += 1

            response_text = final_output.outputs[0].text
            tokens_generated = len(final_output.outputs[0].token_ids)
            tokens_prompt = len(final_output.prompt_token_ids)
            finish_reason = final_output.outputs[0].finish_reason
            exceeded_max_tokens = finish_reason == "length"

            # Record timing metrics (only for successful requests)
            timing = RequestTiming(
                request_id=request_id,
                model_path=self.model_path,
                start_time=start_time,
                end_time=end_time,
                tokens_prompt=tokens_prompt,
                tokens_generated=tokens_generated,
                agent_id=self.agent_id,
            )
            self._metrics_collector.record(timing)

            structured_output = None
            json_parse_error = None
            if response_format == "json":
                structured_output, json_parse_error = self._parse_json_response(
                    response_text
                )

            latency_ms = round((end_time - start_time) * 1000, 2)

            return {
                "text": response_text,
                "structured_output": structured_output,
                "json_parse_error": json_parse_error,
                "tokens_generated": tokens_generated,
                "tokens_prompt": tokens_prompt,
                "exceeded_max_tokens": exceeded_max_tokens,
                "latency_ms": latency_ms,
            }

        except (VLLMBatchError, VLLMOOMError):
            raise
        except FileNotFoundError as e:
            # FileNotFoundError typically indicates vLLM's IPC socket/pipe was lost
            # (e.g., worker process crashed, tensor_parallel communication failure)
            raise VLLMBatchError(
                message=f"vLLM IPC error (worker may have crashed): {e}",
                batch_size=1,
                request_ids=[request_id],
                original_error=e,
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "oom" in error_msg:
                raise VLLMOOMError(max_tokens=max_tok, batch_size=1)
            raise VLLMBatchError(
                message=f"Generation failed: {e}",
                batch_size=1,
                request_ids=[request_id],
                original_error=e,
            )
        finally:
            # Always decrement active requests, regardless of success/failure
            self._metrics_collector.record_end()

    @classmethod
    def get_model_cache_info(cls) -> Dict[str, Any]:
        """Get information about cached engine instances."""
        return {
            "cached_models": list(cls._engines.keys()),
            "num_cached": len(cls._engines),
        }

    @classmethod
    def collect_metrics_snapshot(cls) -> Optional[Dict[str, Any]]:
        """Collect a metrics snapshot from the vLLM engine."""
        return cls.get_engine_metrics()

    @classmethod
    async def cleanup_all_async(cls) -> None:
        """Async cleanup: stop all engines, free GPU memory."""
        await cls.stop_engine()
        cls._cleanup_engines()

    @classmethod
    def cleanup_all(cls) -> None:
        """Sync cleanup: release engines, free GPU memory."""
        cls._cleanup_engines()

    @classmethod
    def _cleanup_engines(cls) -> None:
        """Internal: clear engine references and free memory.

        For tensor_parallel > 1, vLLM spawns worker processes that need explicit
        shutdown. We call engine.shutdown() to gracefully terminate these processes
        before clearing references.
        """
        if cls._engines:
            info_print(f"Releasing {len(cls._engines)} engine(s)...")

            # Shutdown each engine gracefully (important for tensor_parallel > 1)
            for model_path, engine in cls._engines.items():
                try:
                    if hasattr(engine, "shutdown"):
                        engine.shutdown()
                except Exception as e:
                    # Ignore shutdown errors - process may already be dead from Ctrl+C
                    debug_print(f"Engine shutdown for {model_path}: {e}")

            cls._engines.clear()
            cls._engines_started.clear()
            cls._tokenizers.clear()
            cls._total_requests.clear()
            cls._metrics_collector.clear()
            cls._no_system_role_models.clear()

        gc.collect()

        try:
            import torch

            torch.cuda.empty_cache()
            info_print("GPU memory released")
        except ImportError:
            pass

    def __repr__(self) -> str:
        """String representation of agent."""
        return (
            f"AsyncVLLMAgent(id={self.agent_id}, role={self.role}, "
            f"model={self.model_path})"
        )
