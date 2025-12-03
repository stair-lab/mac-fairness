#!/usr/bin/env python3
"""Run a multi-agent conversation experiment from configuration file.

This is the main entry point for running experiments locally or on compute clusters.

Usage:
    # Run full experiment
    python script/run_experiment.py config/dev_ollama/llama32_1b_3agent_..._scratch.yaml

    # Run subset of questions
    python script/run_experiment.py config/dev_ollama/llama32_1b_3agent_..._scratch.yaml --range 0-10

Environment Variables:
    MAC_FAIRNESS_WORKSPACE - Project root directory (required)
    MAC_FAIRNESS_DEBUG_FLAG - Enable debug output for prompts
    MAC_FAIRNESS_EXPERIMENT_ROOT - Override experiment output directory (default: ./experiment)
"""

import argparse
import os
import signal
import sys
from pathlib import Path
from typing import Optional, Tuple

import yaml

# Get project root from environment variable (required)
project_root = Path(os.environ["MAC_FAIRNESS_WORKSPACE"])
sys.path.insert(0, str(project_root))

from src.utils.conversation_orchestrator import ConversationOrchestrator

# Debug flag
DEBUG = os.environ.get("MAC_FAIRNESS_DEBUG_FLAG")

# Track if we've been asked to stop
_shutdown_requested = False


def _debug_print(msg: str) -> None:
    """Print debug message if debug flag is set."""
    if DEBUG:
        print(f"[DEBUG] {msg}")


def _info_print(msg: str) -> None:
    """Print info message (always shown)."""
    print(msg)


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
    required_keys = ["experiment_metadata", "model_config", "agent_definitions"]
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


def get_backend_type(config: dict) -> Optional[str]:
    """Detect which backend the config uses.

    Args:
        config: Parsed config dictionary

    Returns:
        "vllm", "ollama", or None if unknown
    """
    models = config.get("model_config", {}).get("models", {})
    for model_def in models.values():
        backend = model_def.get("backend")
        if backend:
            return backend
        # Auto-detect from config patterns
        if "vllm_config" in model_def or "model_path" in model_def:
            return "vllm"
        if "ollama_config" in model_def or "model_name" in model_def:
            return "ollama"
    return None


def cleanup_resources(backend: Optional[str]) -> None:
    """Clean up GPU/model resources based on backend type.

    Args:
        backend: Backend type ("vllm", "ollama", or None)
    """
    if backend == "vllm":
        try:
            from src.agent.vllm_agent import VLLMAgent

            VLLMAgent.cleanup_all_models()
        except Exception as e:
            _debug_print(f"vLLM cleanup error (non-fatal): {e}")
    # Ollama doesn't need cleanup (external process)


def signal_handler(signum: int, frame) -> None:
    """Handle shutdown signals gracefully.

    Args:
        signum: Signal number
        frame: Current stack frame
    """
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    if _shutdown_requested:
        _info_print(f"\n\n✗ Forced shutdown ({sig_name})")
        sys.exit(128 + signum)
    else:
        _shutdown_requested = True
        _info_print(f"\n\n✗ Received {sig_name}, finishing current question...")
        _info_print("  (Press Ctrl+C again to force quit)")


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
            _info_print(
                f"✓ Processing question range: {question_range[0]}-{question_range[1]}"
            )

        # Print header
        _info_print("=" * 60)
        _info_print("Multi-Agent Conversation Framework")
        _info_print(f"Protocol Version: {schema_version}")
        _info_print("=" * 60)
        _info_print(f"\n✓ Config: {args.config}")
        _info_print(f"✓ Experiment: {exp_name}")
        _info_print(f"✓ Backend: {backend or 'auto-detect'}\n")

        _debug_print("Debug mode enabled")
        _debug_print(
            f"Experiment root: {os.environ.get('MAC_FAIRNESS_EXPERIMENT_ROOT', 'experiment')}"
        )

        # Run experiment
        orchestrator = ConversationOrchestrator(str(config_path))
        orchestrator.run_experiment(question_range=question_range)

        _info_print("\n" + "=" * 60)
        _info_print("✓ Experiment completed successfully!")
        _info_print("=" * 60)
        return 0

    except FileNotFoundError as e:
        _info_print(f"\n✗ Error: {e}")
        return 1

    except ValueError as e:
        _info_print(f"\n✗ Configuration error: {e}")
        return 1

    except KeyboardInterrupt:
        _info_print("\n\n✗ Experiment interrupted by user")
        return 130

    except Exception as e:
        _info_print("\n\n✗ Experiment failed with error:")
        _info_print(f"  {type(e).__name__}: {e}")
        if DEBUG:
            import traceback

            traceback.print_exc()
        return 1

    finally:
        # Always cleanup resources
        _info_print("\n✓ Cleaning up resources...")
        cleanup_resources(backend)


if __name__ == "__main__":
    sys.exit(main())
