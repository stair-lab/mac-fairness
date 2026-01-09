"""Agent implementations for multi-agent conversation framework."""

from .base_agent import BaseAgent, AsyncAgentProtocol
from .async_ollama_agent import AsyncOllamaAgent
from .model_factory import ModelFactory


def __getattr__(name: str):
    """Lazy import for vLLM components (requires transformers/vllm)."""
    if name in ("AsyncVLLMAgent", "RequestTiming", "RequestMetricsCollector"):
        from .async_vllm_agent import (
            AsyncVLLMAgent,
            RequestTiming,
            RequestMetricsCollector,
        )

        return {"AsyncVLLMAgent": AsyncVLLMAgent, "RequestTiming": RequestTiming, "RequestMetricsCollector": RequestMetricsCollector}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseAgent",
    "AsyncAgentProtocol",
    "AsyncOllamaAgent",
    "AsyncVLLMAgent",
    "RequestTiming",
    "RequestMetricsCollector",
    "ModelFactory",
]
