"""vLLM metrics collection utilities for parameter tuning.

This module provides utilities to collect and aggregate vLLM metrics
that are useful for tuning parameters like max_num_seqs, gpu_memory_utilization,
and enable_prefix_caching.

Key metrics collected (from vLLM v0.12.0 Prometheus metrics):
- KV cache usage percentage
- Prefix cache hit rate (queries and hits)
- Request queue state (running, waiting)
- Preemption count (indicates memory pressure)
- Token throughput (prompt and generation)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VLLMMetricsSnapshot:
    """A snapshot of vLLM metrics at a point in time.

    Metrics are aligned with vLLM v0.12.0 Prometheus metric names.
    """

    # KV Cache metrics (Gauges)
    kv_cache_usage_perc: float = 0.0  # vllm:kv_cache_usage_perc (0-1)

    # Prefix cache metrics (Counters - cumulative)
    prefix_cache_queries: int = 0  # vllm:prefix_cache_queries
    prefix_cache_hits: int = 0  # vllm:prefix_cache_hits

    # Request state metrics (Gauges)
    num_requests_running: int = 0  # vllm:num_requests_running
    num_requests_waiting: int = 0  # vllm:num_requests_waiting

    # Throughput metrics (Counters - cumulative)
    prompt_tokens_total: int = 0  # vllm:prompt_tokens
    generation_tokens_total: int = 0  # vllm:generation_tokens

    # Preemption metrics (Counter - cumulative, indicates OOM pressure)
    num_preemptions_total: int = 0  # vllm:num_preemptions

    # Success metrics (Counter - cumulative)
    request_success_total: int = 0  # vllm:request_success

    @property
    def prefix_cache_hit_rate(self) -> float:
        """Calculate prefix cache hit rate from queries and hits."""
        if self.prefix_cache_queries == 0:
            return 0.0
        return self.prefix_cache_hits / self.prefix_cache_queries


@dataclass
class VLLMMetricsAggregated:
    """Aggregated vLLM metrics over a job run.

    These are the metrics that get saved in job summaries for parameter tuning.
    """

    # KV Cache - aggregate stats
    peak_kv_cache_usage_perc: float = 0.0
    avg_kv_cache_usage_perc: float = 0.0

    # Prefix cache - computed from deltas
    prefix_cache_queries: int = 0
    prefix_cache_hits: int = 0

    # Preemptions - total count (indicates memory pressure)
    num_preemptions: int = 0

    # Token throughput
    prompt_tokens: int = 0
    generation_tokens: int = 0

    # Request counts
    requests_completed: int = 0

    # Peak concurrent requests
    peak_requests_running: int = 0
    peak_requests_waiting: int = 0

    @property
    def prefix_cache_hit_rate(self) -> float:
        """Calculate prefix cache hit rate."""
        if self.prefix_cache_queries == 0:
            return 0.0
        return self.prefix_cache_hits / self.prefix_cache_queries

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Uses vLLM-standard naming conventions.
        """
        return {
            "kv_cache": {
                "peak_usage_perc": round(self.peak_kv_cache_usage_perc, 4),
                "avg_usage_perc": round(self.avg_kv_cache_usage_perc, 4),
            },
            "prefix_cache": {
                "queries": self.prefix_cache_queries,
                "hits": self.prefix_cache_hits,
                "hit_rate": round(self.prefix_cache_hit_rate, 4),
            },
            "preemptions": self.num_preemptions,
            "tokens": {
                "prompt": self.prompt_tokens,
                "generation": self.generation_tokens,
            },
            "requests": {
                "completed": self.requests_completed,
                "peak_running": self.peak_requests_running,
                "peak_waiting": self.peak_requests_waiting,
            },
        }


class VLLMMetricsCollector:
    """Collects and aggregates vLLM metrics over a job run.

    Usage:
        collector = VLLMMetricsCollector()

        # After each generation, collect a snapshot
        snapshot = collector.collect_snapshot(llm)
        collector.record_snapshot(snapshot)

        # At end of job, get aggregated metrics
        aggregated = collector.get_aggregated()
    """

    def __init__(self):
        """Initialize the metrics collector."""
        self._snapshots: List[VLLMMetricsSnapshot] = []
        self._initial_snapshot: Optional[VLLMMetricsSnapshot] = None

    def collect_snapshot(self, llm: Any) -> VLLMMetricsSnapshot:
        """Collect a metrics snapshot from a vLLM LLM instance.

        Args:
            llm: vLLM LLM instance

        Returns:
            VLLMMetricsSnapshot with current metrics
        """
        snapshot = VLLMMetricsSnapshot()

        try:
            # Use get_metrics() which returns Prometheus metrics
            metrics = llm.get_metrics()

            if metrics:
                if isinstance(metrics, str):
                    # Parse Prometheus text format
                    snapshot = self._parse_prometheus_metrics(metrics)
                elif isinstance(metrics, dict):
                    # Parse dict format (some vLLM versions return dict)
                    snapshot = self._parse_metrics_dict(metrics)

        except AttributeError:
            # get_metrics() not available - try legacy approach
            try:
                snapshot = self._collect_from_engine(llm)
            except Exception:
                pass  # Return empty snapshot

        except Exception:
            pass  # Return empty snapshot

        return snapshot

    def _parse_prometheus_metrics(self, metrics: str) -> VLLMMetricsSnapshot:
        """Parse Prometheus-format metrics string.

        Args:
            metrics: Prometheus metrics string from get_metrics()

        Returns:
            Parsed VLLMMetricsSnapshot
        """
        snapshot = VLLMMetricsSnapshot()

        if not metrics or not isinstance(metrics, str):
            return snapshot

        # Parse line by line
        for line in metrics.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Parse metric line: name{labels} value
            try:
                # Simple parsing - split on space, take last part as value
                parts = line.rsplit(" ", 1)
                if len(parts) != 2:
                    continue

                metric_part, value_str = parts
                value = float(value_str)

                # Extract metric name (before any labels)
                metric_name = metric_part.split("{")[0]

                # Map to snapshot fields
                if metric_name == "vllm:kv_cache_usage_perc":
                    snapshot.kv_cache_usage_perc = value
                elif metric_name == "vllm:prefix_cache_queries":
                    snapshot.prefix_cache_queries = int(value)
                elif metric_name == "vllm:prefix_cache_hits":
                    snapshot.prefix_cache_hits = int(value)
                elif metric_name == "vllm:num_requests_running":
                    snapshot.num_requests_running = int(value)
                elif metric_name == "vllm:num_requests_waiting":
                    snapshot.num_requests_waiting = int(value)
                elif metric_name == "vllm:prompt_tokens":
                    snapshot.prompt_tokens_total = int(value)
                elif metric_name == "vllm:generation_tokens":
                    snapshot.generation_tokens_total = int(value)
                elif metric_name == "vllm:num_preemptions":
                    snapshot.num_preemptions_total = int(value)
                elif metric_name == "vllm:request_success":
                    snapshot.request_success_total = int(value)

            except (ValueError, IndexError):
                continue

        return snapshot

    def _parse_metrics_dict(self, metrics: Dict[str, Any]) -> VLLMMetricsSnapshot:
        """Parse metrics from dict format.

        Args:
            metrics: Metrics dictionary from get_metrics()

        Returns:
            Parsed VLLMMetricsSnapshot
        """
        snapshot = VLLMMetricsSnapshot()

        # Try common dict key patterns used by vLLM
        # Pattern 1: Direct keys
        if "kv_cache_usage_perc" in metrics:
            snapshot.kv_cache_usage_perc = float(metrics.get("kv_cache_usage_perc", 0))
        elif "gpu_cache_usage_perc" in metrics:
            snapshot.kv_cache_usage_perc = float(metrics.get("gpu_cache_usage_perc", 0))

        if "prefix_cache_hit_rate" in metrics:
            snapshot.prefix_cache_hits = 1  # Mark as having data
            # Store as percentage
            hit_rate = float(metrics.get("prefix_cache_hit_rate", 0))
            snapshot.prefix_cache_queries = 100
            snapshot.prefix_cache_hits = int(hit_rate * 100)

        if "num_requests_running" in metrics:
            snapshot.num_requests_running = int(metrics.get("num_requests_running", 0))

        if "num_requests_waiting" in metrics:
            snapshot.num_requests_waiting = int(metrics.get("num_requests_waiting", 0))

        if "num_preemptions" in metrics:
            snapshot.num_preemptions_total = int(metrics.get("num_preemptions", 0))

        # Pattern 2: Nested under "gauges" or similar
        if "gauges" in metrics:
            gauges = metrics["gauges"]
            if isinstance(gauges, dict):
                for key, value in gauges.items():
                    if "kv_cache_usage" in key:
                        snapshot.kv_cache_usage_perc = float(value)
                    elif "prefix_cache" in key and "hit" in key:
                        snapshot.prefix_cache_hits = int(value) if value else 0

        return snapshot

    def _collect_from_engine(self, llm: Any) -> VLLMMetricsSnapshot:
        """Collect metrics directly from vLLM engine internals.

        For vLLM v0.12 offline mode, metrics are accessed via:
        1. CacheEngine - KV cache utilization
        2. Scheduler - request queue state, preemptions
        3. Prefix caching stats - if enabled

        Note: These are internal APIs and may change between versions.

        Args:
            llm: vLLM LLM instance

        Returns:
            VLLMMetricsSnapshot with available metrics
        """
        snapshot = VLLMMetricsSnapshot()

        if not hasattr(llm, "llm_engine"):
            return snapshot

        engine = llm.llm_engine

        # Method 1: Access KV cache usage from cache_config and scheduler
        self._collect_kv_cache_metrics(engine, snapshot)

        # Method 2: Access scheduler for queue state
        self._collect_scheduler_metrics(engine, snapshot)

        # Method 3: Access prefix caching stats (if enabled)
        self._collect_prefix_cache_metrics(engine, snapshot)

        # Method 4: Try stat_logger for additional stats
        self._collect_stat_logger_metrics(engine, snapshot)

        return snapshot

    def _collect_kv_cache_metrics(self, engine: Any, snapshot: VLLMMetricsSnapshot) -> None:
        """Collect KV cache utilization metrics.

        Args:
            engine: vLLM LLMEngine instance
            snapshot: Snapshot to populate
        """
        try:
            # vLLM v0.12: Access via scheduler[0].block_manager
            if hasattr(engine, "scheduler"):
                schedulers = engine.scheduler
                # Handle both list and single scheduler
                scheduler = schedulers[0] if isinstance(schedulers, list) else schedulers

                if hasattr(scheduler, "block_manager"):
                    block_manager = scheduler.block_manager
                    # Try get_num_free_gpu_blocks / get_num_total_gpu_blocks
                    if hasattr(block_manager, "get_num_free_gpu_blocks") and \
                       hasattr(block_manager, "get_num_total_gpu_blocks"):
                        free_blocks = block_manager.get_num_free_gpu_blocks()
                        total_blocks = block_manager.get_num_total_gpu_blocks()
                        if total_blocks > 0:
                            used_blocks = total_blocks - free_blocks
                            snapshot.kv_cache_usage_perc = used_blocks / total_blocks

            # Alternative: Access via model_executor.driver_worker
            if snapshot.kv_cache_usage_perc == 0.0 and hasattr(engine, "model_executor"):
                executor = engine.model_executor
                if hasattr(executor, "driver_worker"):
                    worker = executor.driver_worker
                    if hasattr(worker, "cache_engine"):
                        cache_engine = worker.cache_engine
                        if hasattr(cache_engine, "gpu_cache"):
                            # Some versions expose cache stats directly
                            gpu_cache = cache_engine.gpu_cache
                            if hasattr(gpu_cache, "get_usage"):
                                snapshot.kv_cache_usage_perc = gpu_cache.get_usage()

        except Exception:
            pass

    def _collect_scheduler_metrics(self, engine: Any, snapshot: VLLMMetricsSnapshot) -> None:
        """Collect scheduler queue state metrics.

        Args:
            engine: vLLM LLMEngine instance
            snapshot: Snapshot to populate
        """
        try:
            if hasattr(engine, "scheduler"):
                schedulers = engine.scheduler
                scheduler = schedulers[0] if isinstance(schedulers, list) else schedulers

                # Get running requests count
                if hasattr(scheduler, "running"):
                    snapshot.num_requests_running = len(scheduler.running)
                elif hasattr(scheduler, "get_num_unfinished_requests"):
                    snapshot.num_requests_running = scheduler.get_num_unfinished_requests()

                # Get waiting requests count
                if hasattr(scheduler, "waiting"):
                    snapshot.num_requests_waiting = len(scheduler.waiting)

                # Get preemption count (v0.12 tracks this in scheduler)
                if hasattr(scheduler, "num_cumulative_preemption"):
                    snapshot.num_preemptions_total = scheduler.num_cumulative_preemption

        except Exception:
            pass

    def _collect_prefix_cache_metrics(self, engine: Any, snapshot: VLLMMetricsSnapshot) -> None:
        """Collect prefix caching hit rate metrics.

        Args:
            engine: vLLM LLMEngine instance
            snapshot: Snapshot to populate
        """
        try:
            # Check if prefix caching is enabled
            if hasattr(engine, "cache_config"):
                cache_config = engine.cache_config
                if not getattr(cache_config, "enable_prefix_caching", False):
                    return

            # Access prefix cache stats via scheduler's block_manager
            if hasattr(engine, "scheduler"):
                schedulers = engine.scheduler
                scheduler = schedulers[0] if isinstance(schedulers, list) else schedulers

                if hasattr(scheduler, "block_manager"):
                    block_manager = scheduler.block_manager

                    # v0.12: block_manager may have prefix_caching stats
                    if hasattr(block_manager, "prefix_pool"):
                        prefix_pool = block_manager.prefix_pool
                        if hasattr(prefix_pool, "get_stats"):
                            stats = prefix_pool.get_stats()
                            if isinstance(stats, dict):
                                snapshot.prefix_cache_queries = stats.get("queries", 0)
                                snapshot.prefix_cache_hits = stats.get("hits", 0)
                            elif hasattr(stats, "queries"):
                                snapshot.prefix_cache_queries = stats.queries
                                snapshot.prefix_cache_hits = stats.hits

                    # Alternative: check for prefix_caching attribute
                    if hasattr(block_manager, "prefix_caching"):
                        prefix_cache = block_manager.prefix_caching
                        if hasattr(prefix_cache, "num_queries"):
                            snapshot.prefix_cache_queries = prefix_cache.num_queries
                        if hasattr(prefix_cache, "num_hits"):
                            snapshot.prefix_cache_hits = prefix_cache.num_hits

        except Exception:
            pass

    def _collect_stat_logger_metrics(self, engine: Any, snapshot: VLLMMetricsSnapshot) -> None:
        """Collect metrics from stat_logger (if available).

        Args:
            engine: vLLM LLMEngine instance
            snapshot: Snapshot to populate
        """
        try:
            if not hasattr(engine, "stat_loggers") and not hasattr(engine, "stat_logger"):
                return

            # v0.12: stat_loggers is a dict
            stat_loggers = getattr(engine, "stat_loggers", None)
            if stat_loggers and isinstance(stat_loggers, dict):
                for logger in stat_loggers.values():
                    if hasattr(logger, "stats"):
                        stats = logger.stats
                        # Extract available metrics
                        if hasattr(stats, "gpu_cache_usage") and snapshot.kv_cache_usage_perc == 0.0:
                            snapshot.kv_cache_usage_perc = float(stats.gpu_cache_usage)
                        if hasattr(stats, "num_preemption") and snapshot.num_preemptions_total == 0:
                            snapshot.num_preemptions_total = int(stats.num_preemption)
                        break

            # Legacy: single stat_logger
            stat_logger = getattr(engine, "stat_logger", None)
            if stat_logger and hasattr(stat_logger, "stats"):
                stats = stat_logger.stats
                if hasattr(stats, "gpu_cache_usage") and snapshot.kv_cache_usage_perc == 0.0:
                    snapshot.kv_cache_usage_perc = float(stats.gpu_cache_usage)

        except Exception:
            pass

    def record_snapshot(self, snapshot: VLLMMetricsSnapshot) -> None:
        """Record a metrics snapshot.

        Args:
            snapshot: VLLMMetricsSnapshot to record
        """
        if self._initial_snapshot is None:
            self._initial_snapshot = snapshot

        self._snapshots.append(snapshot)

    def get_aggregated(self) -> VLLMMetricsAggregated:
        """Get aggregated metrics from all recorded snapshots.

        Returns:
            VLLMMetricsAggregated with computed statistics
        """
        if not self._snapshots:
            return VLLMMetricsAggregated()

        # Calculate aggregates
        kv_usages = [s.kv_cache_usage_perc for s in self._snapshots]
        peak_kv = max(kv_usages) if kv_usages else 0.0
        avg_kv = sum(kv_usages) / len(kv_usages) if kv_usages else 0.0

        # Get final snapshot for counter values
        final = self._snapshots[-1]
        initial = self._initial_snapshot or VLLMMetricsSnapshot()

        # Calculate deltas for counters
        prefix_queries = final.prefix_cache_queries - initial.prefix_cache_queries
        prefix_hits = final.prefix_cache_hits - initial.prefix_cache_hits
        preemptions = final.num_preemptions_total - initial.num_preemptions_total
        prompt_tokens = final.prompt_tokens_total - initial.prompt_tokens_total
        gen_tokens = final.generation_tokens_total - initial.generation_tokens_total
        requests = final.request_success_total - initial.request_success_total

        # Peak concurrent requests
        peak_running = max(s.num_requests_running for s in self._snapshots)
        peak_waiting = max(s.num_requests_waiting for s in self._snapshots)

        return VLLMMetricsAggregated(
            peak_kv_cache_usage_perc=peak_kv,
            avg_kv_cache_usage_perc=avg_kv,
            prefix_cache_queries=max(0, prefix_queries),
            prefix_cache_hits=max(0, prefix_hits),
            num_preemptions=max(0, preemptions),
            prompt_tokens=max(0, prompt_tokens),
            generation_tokens=max(0, gen_tokens),
            requests_completed=max(0, requests),
            peak_requests_running=peak_running,
            peak_requests_waiting=peak_waiting,
        )

    def reset(self) -> None:
        """Reset the collector for a new job run."""
        self._snapshots.clear()
        self._initial_snapshot = None


# Global collector instance for use across the application
_global_collector: Optional[VLLMMetricsCollector] = None


def get_metrics_collector() -> VLLMMetricsCollector:
    """Get or create the global metrics collector.

    Returns:
        The global VLLMMetricsCollector instance
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = VLLMMetricsCollector()
    return _global_collector


def reset_metrics_collector() -> None:
    """Reset the global metrics collector."""
    global _global_collector
    if _global_collector is not None:
        _global_collector.reset()
