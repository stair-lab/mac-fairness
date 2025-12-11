#!/usr/bin/env python3
"""Run a multi-agent conversation experiment from configuration file.

This is the main entry point for running experiments locally or on compute clusters.

Usage:
    # Run full experiment
    python script/run_experiment.py config/dev_ollama/llama32_1b_3agent_..._scratch.yaml

    # Run subset of questions
    python script/run_experiment.py config/dev_ollama/llama32_1b_3agent_..._scratch.yaml --range 0-10

    # Add env var setting
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 MAC_FAIRNESS_LIVE_STATUS=1 uv run python ...

    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=16 MAC_FAIRNESS_DEBUG_FLAG=1 uv run python ...

Environment Variables:
    MAC_FAIRNESS_WORKSPACE - Project root directory (required)
    MAC_FAIRNESS_EXPERIMENT_ROOT - Override experiment output directory (default: ./experiment)
"""

import argparse
import os
import signal
import sys
from pathlib import Path
from typing import Tuple

import yaml

# Get project root from environment variable (required)
project_root = Path(os.environ["MAC_FAIRNESS_WORKSPACE"])
sys.path.insert(0, str(project_root))

from src.utils import debug_print, is_debug_enabled
from src.utils.conversation_orchestrator import ConversationOrchestrator

# Track if we've been asked to stop
_shutdown_requested = False


def parse_range(range_str: str) -> Tuple[int, int]:
    """Parse range string like '0-10' into tuple (0, 10).

    Args:
        range_str: Range in format 'start-end'

    Returns:
        Tuple of (start, end) indices

    Raises:
        ValueError: If range format is invalid
    """
    try:
        parts = range_str.split("-")
        if len(parts) != 2:
            raise ValueError("Expected exactly one '-' separator")
        start, end = int(parts[0]), int(parts[1])
        if start < 0:
            raise ValueError("Start index cannot be negative")
        if end <= start:
            raise ValueError("End index must be greater than start index")
        return start, end
    except ValueError as e:
        raise ValueError(
            f"Invalid range format: '{range_str}'. "
            f"Expected format: 'start-end' (e.g., '0-10'). Error: {e}"
        ) from e


def validate_config(config_path: Path) -> dict:
    """Validate config file exists and is valid YAML.

    Args:
        config_path: Path to config file

    Returns:
        Parsed config dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid YAML or missing required fields
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}") from e

    if not config:
        raise ValueError("Config file is empty")

    # Validate required top-level keys
    required_keys = ["experiment_metadata", "model_definitions", "agent_definitions"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    return config


def get_schema_version(config: dict) -> str:
    """Extract schema version from config.

    Args:
        config: Parsed config dictionary

    Returns:
        Schema version string
    """
    return config.get("experiment_metadata", {}).get("schema_version", "unknown")


def get_backend_type(config: dict) -> str:
    """Get backend type from config.

    Args:
        config: Parsed config dictionary

    Returns:
        "vllm" or "ollama"

    Raises:
        ValueError: If backend not specified in any model definition
    """
    model_definitions = config.get("model_definitions", {})
    for model_def in model_definitions.values():
        backend = model_def.get("backend")
        if backend:
            return backend
    raise ValueError(
        "backend must be specified in model_definitions (either 'vllm' or 'ollama')"
    )


def cleanup_resources(backend: str) -> None:
    """Clean up GPU/model resources based on backend type.

    Args:
        backend: Backend type ("vllm" or "ollama")
    """
    if backend == "vllm":
        try:
            from src.agent.async_vllm_agent import AsyncVLLMAgent

            AsyncVLLMAgent.cleanup_all()
        except Exception as e:
            debug_print(f"vLLM cleanup error (non-fatal): {e}")
    # Ollama doesn't need cleanup (external process)


def signal_handler(signum: int, _frame) -> None:
    """Handle shutdown signals gracefully.

    Args:
        signum: Signal number
        _frame: Current stack frame (unused)
    """
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    if _shutdown_requested:
        print(f"\n\n✗ Forced shutdown ({sig_name})")
        sys.exit(128 + signum)
    else:
        _shutdown_requested = True
        print(f"\n\n✗ Received {sig_name}, finishing current question...")
        print("  (Press Ctrl+C again to force quit)")


def main() -> int:
    """Main entry point for experiment runner.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(
        description="Run multi-agent conversation experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full experiment
  python script/run_experiment.py config/dev_snap/base_sweep.yaml

  # Run first 10 questions
  python script/run_experiment.py config/dev_snap/base_sweep.yaml --range 0-10

  # Run specific range
  python script/run_experiment.py config/bbq_race/llama31_8b_..._scratch.yaml --range 100-200

Environment Variables:
  MAC_FAIRNESS_DEBUG_FLAG - Enable debug output
  MAC_FAIRNESS_EXPERIMENT_ROOT - Override experiment output directory (default: ./experiment)
        """,
    )

    parser.add_argument(
        "config", type=str, help="Path to experiment configuration YAML file"
    )

    parser.add_argument(
        "--range",
        type=str,
        metavar="START-END",
        help="Process only questions in range START-END (e.g., '0-10' for first 10 questions)",
    )

    args = parser.parse_args()

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    config_path = Path(args.config)
    backend = None

    try:
        # Validate config before proceeding
        config = validate_config(config_path)
        schema_version = get_schema_version(config)
        backend = get_backend_type(config)
        exp_name = config.get("experiment_metadata", {}).get(
            "experiment_name", "unknown"
        )

        # Parse question range if specified
        question_range = None
        if args.range:
            question_range = parse_range(args.range)
            print(
                f"✓ Processing question range: {question_range[0]}-{question_range[1]}"
            )

        # Print header
        print("=" * 60)
        print("Multi-Agent Conversation Framework")
        print(f"Protocol Version: {schema_version}")
        print("=" * 60)
        print(f"\n✓ Config: {args.config}")
        print(f"✓ Experiment: {exp_name}")
        print(f"✓ Backend: {backend}\n")

        debug_print("Debug mode enabled")
        debug_print(
            f"Experiment root: {os.environ.get('MAC_FAIRNESS_EXPERIMENT_ROOT', 'experiment')}"
        )

        # Run experiment (async)
        import asyncio

        orchestrator = ConversationOrchestrator(str(config_path))
        asyncio.run(orchestrator.run_experiment(question_range=question_range))

        print("\n" + "=" * 60)
        print("✓ Experiment completed successfully!")
        print("=" * 60)
        return 0

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return 1

    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        return 1

    except KeyboardInterrupt:
        print("\n\n✗ Experiment interrupted by user")
        return 130

    except Exception as e:
        print("\n\n✗ Experiment failed with error:")
        print(f"  {type(e).__name__}: {e}")
        if is_debug_enabled():
            import traceback

            traceback.print_exc()
        return 1

    finally:
        # Always cleanup resources
        print("\n✓ Cleaning up resources...")
        cleanup_resources(backend)


if __name__ == "__main__":
    sys.exit(main())
