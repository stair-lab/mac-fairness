#!/usr/bin/env python3
"""Run a multi-agent conversation experiment from configuration file.

This is the main entry point for running experiments locally or on compute clusters.

Usage:
    # Run full experiment
    python script/run_experiment.py config/dev_ollama/llama32_1b_3agent_..._scratch.yaml

    # Run subset of questions
    python script/run_experiment.py config/dev_ollama/llama32_1b_3agent_..._scratch.yaml --range 0-10
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.conversation_orchestrator import ConversationOrchestrator


def parse_range(range_str: str) -> tuple[int, int]:
    """Parse range string like '0-10' into tuple (0, 10).

    Args:
        range_str: Range in format 'start-end'

    Returns:
        Tuple of (start, end) indices

    Raises:
        ValueError: If range format is invalid
    """
    try:
        start, end = range_str.split("-")
        return int(start), int(end)
    except Exception as e:
        raise ValueError(
            f"Invalid range format: {range_str}. Expected format: 'start-end' (e.g., '0-10')"
        ) from e


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run multi-agent conversation experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full experiment
  python script/run_experiment.py config/dev_ollama/llama32_1b_3agent_as-human-demographics_vanilla_v2025-11-27_scratch.yaml

  # Run first 10 questions
  python script/run_experiment.py config/dev_ollama/llama32_1b_3agent_as-human-demographics_vanilla_v2025-11-27_scratch.yaml --range 0-10

  # Run specific range
  python script/run_experiment.py config/bbq_race/llama31_8b_3agent_..._scratch.yaml --range 100-200

Environment Variables:
  MAC_FAIRNESS_EXPERIMENT_ROOT - Override default experiment output directory (default: ./experiment)
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

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"✗ Error: Config file not found: {args.config}")
        sys.exit(1)

    # Parse question range if specified
    question_range = None
    if args.range:
        try:
            question_range = parse_range(args.range)
            print(
                f"✓ Processing question range: {question_range[0]}-{question_range[1]}"
            )
        except ValueError as e:
            print(f"✗ Error: {e}")
            sys.exit(1)

    # Create conversation orchestrator and run experiment
    try:
        print("=" * 60)
        print("Multi-Agent Conversation Framework")
        print("Protocol Version: 2025-11-27")
        print("=" * 60)
        print(f"\n✓ Config file: {args.config}\n")

        orchestrator = ConversationOrchestrator(str(config_path))
        orchestrator.run_experiment(question_range=question_range)

        print("\n" + "=" * 60)
        print("✓ Experiment completed successfully!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n✗ Experiment interrupted by user")
        sys.exit(130)
    except Exception as e:
        print("\n\n✗ Experiment failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
