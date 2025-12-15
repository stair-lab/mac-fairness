#!/usr/bin/env python3
"""Run a multi-agent conversation job from configuration file.

This is the main entry point for running experiments locally or on compute clusters.

Usage:
    # Run single task
    uv run python script/run_job.py config/dev_ollama/llama32_1b_3agent_..._scratch.yaml

    # Run grid job (parameter sweep with multiple tasks)
    uv run python script/run_job.py config/my_exp/my_grid_config.yaml --grid

    # Dry run (validate config and show what would run without executing)
    uv run python script/run_job.py config/my_exp/my_config.yaml --dry-run

    # Dry run grid job (show all expanded configurations)
    uv run python script/run_job.py config/my_exp/my_grid_config.yaml --grid --dry-run

    # Add env var setting
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 MAC_FAIRNESS_LIVE_STATUS=1 uv run python ...

    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=16 MAC_FAIRNESS_DEBUG_FLAG=1 uv run python ...

Environment Variables:
    MAC_FAIRNESS_WORKSPACE - Project root directory (required)
    MAC_FAIRNESS_EXPERIMENT_ROOT - Override experiment output directory (default: ./experiment)
"""

import argparse
import asyncio
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

# Get project root from environment variable (required)
project_root = Path(os.environ["MAC_FAIRNESS_WORKSPACE"])
sys.path.insert(0, str(project_root))

from src.utils import debug_print, is_debug_enabled
from src.utils.logging import info_print
from src.utils.bookkeeping_manager import BookkeepingManager, GridManifestManager, set_grid_index
from src.utils.conversation_orchestrator import ConversationOrchestrator
from src.utils.grid_config import GridConfigExpander

# Module-level variable for grid resume info (used by signal handler)
_grid_resume_path: Optional[str] = None


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
    """Handle shutdown signals - immediate exit on first signal.

    Treats shutdown as a disaster scenario: exit immediately, rely on
    persistent storage (manifest, transcripts already saved) for recovery.

    Args:
        signum: Signal number
        _frame: Current stack frame (unused)
    """
    sig_name = signal.Signals(signum).name
    info_print(f"\n✗ Received {sig_name}, exiting immediately...", prefix=False)
    info_print("  (Completed transcripts saved, manifest preserved for resume)", prefix=False)

    # Print resume command for grid jobs
    if _grid_resume_path:
        info_print(f"\nTo resume this grid job, run:", prefix=False)
        info_print(f"[ENV_VARS] python script/run_job.py {_grid_resume_path} --grid --resume", prefix=False)

    # Exit immediately - no graceful shutdown, just stop
    sys.exit(128 + signum)


def run_grid_experiments(args: argparse.Namespace) -> int:
    """Run experiments from a grid configuration file.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    import tempfile

    config_path = Path(args.config)

    # Load and expand grid config
    expander = GridConfigExpander(str(config_path))

    if not expander.is_grid_config():
        info_print(f"Error: --grid flag specified but config has no _grid.sweep section")
        return 1

    # Print summary
    expander.print_summary()

    # Get expanded configs with sweep specs
    expanded_configs = expander.expand_with_sweep_specs()

    # Handle dry-run for non-resume mode
    if args.dry_run and not args.resume:
        expander.print_expanded_configs(verbose=False)
        info_print(f"Dry run complete. {len(expanded_configs)} configurations would be executed.")
        return 0

    # Initialize grid manifest manager and bookkeeping manager
    grid_manifest_manager = GridManifestManager()
    bookkeeping_manager = BookkeepingManager()
    pending_indices: Optional[list] = None
    started_task_info: Dict[int, Dict[str, str]] = {}

    # Check for resume mode
    if args.resume:
        # Resume requires using the grid config snapshot path
        config_parent = config_path.parent.name
        config_grandparent = config_path.parent.parent.name if config_path.parent.parent else ""
        is_snapshot_path = (config_grandparent == "bookkeeping" and config_parent == "_grid_config_snapshot")
        if not is_snapshot_path:
            info_print("Error: --resume requires using the grid config snapshot path")
            info_print("  Expected: bookkeeping/_grid_config_snapshot/{config}_{timestamp}.yaml")
            info_print(f"  Got: {config_path}")
            info_print("To find your snapshot, check: ls bookkeeping/_grid_config_snapshot/")
            print()
            return 1

        # For dry-run, peek at manifest without deleting; otherwise load and delete
        if args.dry_run:
            result = grid_manifest_manager.peek_manifest(str(config_path))
        else:
            result = grid_manifest_manager.load_and_delete_manifest(str(config_path))

        if result is not None:
            pending_indices, started_task_info, succeeded_task_info = result
            if not pending_indices:
                info_print(f"All configurations already completed. Nothing to resume.")
                return 0
            started_count = len(started_task_info)
            info_print(f"Resuming: {len(pending_indices)} pending of {len(expanded_configs)} configurations")
            if started_count > 0:
                info_print(f"  ({started_count} were previously started, will check job manifests for partial progress)")
        else:
            info_print(f"No existing manifest found for this snapshot.")
            info_print("Check bookkeeping/grid_manifest/ for available manifests.")
            return 1

    # Handle dry-run for resume mode
    if args.dry_run:
        info_print("-" * 60, prefix=False)
        info_print("Configurations to resume:", prefix=False)
        for i in pending_indices:
            config, _ = expanded_configs[i]
            exp_name = config["experiment_metadata"]["experiment_name"]
            status = "started" if i in started_task_info else "pending"
            info_print(f"  [{i}] {exp_name} ({status})", prefix=False)
        info_print(f"Dry run complete. {len(pending_indices)} configurations would be resumed.")
        return 0

    # Create new manifest (always - fresh or with carried-over pending status)
    if pending_indices is None:
        pending_indices = list(range(len(expanded_configs)))
        succeeded_task_info = {}  # Fresh run, no processed runs to carry over

    submission_timestamp = datetime.now(timezone.utc)
    manifest_path, _grid_config_snapshot_path = grid_manifest_manager.save_grid_manifest(
        str(config_path), expanded_configs, submission_timestamp,
        is_resume=args.resume, succeeded_task_info=succeeded_task_info
    )

    # Set module-level variable for signal handler to print resume command
    # (Process-local: only affects this Python process, not others)
    global _grid_resume_path
    _grid_resume_path = _grid_config_snapshot_path

    # Run each configuration sequentially
    info_print(f"Running {len(pending_indices)} experiment configurations...")

    successful = 0
    failed = 0
    skipped = 0
    failed_configs: List[Tuple[int, str, str]] = []  # (index, exp_name, error_msg)
    backend = None

    for i, (config, _sweep_specs) in enumerate(expanded_configs):
        # Skip if not in pending list (already completed)
        if i not in pending_indices:
            skipped += 1
            continue

        exp_name = config["experiment_metadata"]["experiment_name"]
        config_num = pending_indices.index(i) + 1
        info_print("=" * 60, prefix=False)
        info_print(f"Configuration {config_num}/{len(pending_indices)} (index {i}): {exp_name}")
        info_print("=" * 60, prefix=False)

        # Mark as started in manifest
        grid_manifest_manager.mark_task_started(manifest_path, i)

        # Set grid index for job_task_id generation
        set_grid_index(i)

        # For "started" runs, find task manifest and get null question IDs
        # Do this before creating temp file to allow early skip
        question_ids: Optional[Set[str]] = None
        old_task_manifest_path: Optional[Path] = None
        resume_config_path: Optional[str] = None  # Use existing config snapshot for resume
        succeeded_questions: Optional[List[Dict[str, Any]]] = None  # Carry over for resume
        if i in started_task_info:
            run_info = started_task_info[i]
            started_exp_name = run_info["experiment_name"]
            started_benchmark = run_info["benchmark_subcategory"]
            # Pass expected config to validate that the task manifest's config_snapshot
            # matches this configuration (important when multiple configs share experiment_name)
            result = bookkeeping_manager.find_task_manifest_and_get_null_questions(
                experiment_name=started_exp_name,
                benchmark_subcategory=started_benchmark,
                expected_config=config,
            )
            if result is not None:
                old_task_manifest_path, null_question_ids, task_config_snapshot, succeeded_questions = result
                question_ids = set(null_question_ids)
                if not question_ids:
                    info_print(f"  All questions already succeeded in previous run, skipping")
                    grid_manifest_manager.mark_task_succeeded(manifest_path, i)
                    successful += 1
                    continue
                # Validate config snapshot exists and use it for resume
                if task_config_snapshot:
                    # Resolve $MAC_FAIRNESS_WORKSPACE if present
                    resolved_snapshot = task_config_snapshot
                    if resolved_snapshot.startswith("$MAC_FAIRNESS_WORKSPACE"):
                        resolved_snapshot = resolved_snapshot.replace(
                            "$MAC_FAIRNESS_WORKSPACE", str(project_root)
                        )
                    if Path(resolved_snapshot).exists():
                        resume_config_path = resolved_snapshot
                        info_print(f"  Resuming {len(question_ids)} null questions using existing config snapshot")
                    else:
                        info_print(f"  Warning: Config snapshot not found: {task_config_snapshot}")
                        info_print(f"  Resuming {len(question_ids)} null questions with current config")
                else:
                    info_print(f"  Resuming {len(question_ids)} null questions from previous task manifest")

        tmp_path = None
        try:
            # For resume, use existing config snapshot; otherwise write new temp file
            if resume_config_path:
                config_to_use = resume_config_path
            else:
                # Write config to a temporary file
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".yaml", delete=False
                ) as tmp:
                    yaml.dump(config, tmp, default_flow_style=False, sort_keys=False)
                    tmp_path = tmp.name
                config_to_use = tmp_path

            # Get backend for cleanup
            backend = get_backend_type(config)

            # Run experiment
            orchestrator = ConversationOrchestrator(config_to_use)

            # Delete old task manifest now that a new one will be created
            if old_task_manifest_path and old_task_manifest_path.exists():
                old_task_manifest_path.unlink()
                info_print(f"  Deleted old task manifest: {old_task_manifest_path.name}")

            asyncio.run(orchestrator.run_job(
                question_ids=question_ids,
                succeeded_questions=succeeded_questions,
            ))

            successful += 1
            grid_manifest_manager.mark_task_succeeded(manifest_path, i)
            info_print(f"Task {config_num} completed successfully")

        except Exception as e:
            # Task failed - leave status as "started" so it can be resumed
            failed += 1
            error_msg = f"{type(e).__name__}: {e}"
            failed_configs.append((i, exp_name, error_msg))
            info_print(f"Task {config_num} failed: {error_msg}")
            if is_debug_enabled():
                import traceback
                traceback.print_exc()

        finally:
            # Clean up temp file
            if tmp_path:
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass

    # Print summary
    info_print("=" * 60, prefix=False)
    info_print(f"Grid job complete: {successful} succeeded, {failed} failed, {skipped} skipped")
    if failed_configs:
        info_print("Failed configurations:", prefix=False)
        for idx, name, err in failed_configs:
            info_print(f"  [{idx}] {name}: {err}", prefix=False)
    info_print("=" * 60, prefix=False)

    # Delete manifest and grid config snapshot only if all tasks succeeded
    if manifest_path:
        grid_manifest_manager.delete_if_complete(manifest_path)

    # Cleanup resources once at the end
    if backend:
        cleanup_resources(backend)

    return 0 if failed == 0 else 1


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
  CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=16 MAC_FAIRNESS_LIVE_STATUS=1 python script/run_job.py config/bbq_race/llama33_70b_3agent_as-human-demographics_vanilla_v2025-12-10_scratch.yaml

Environment Variables:
  MAC_FAIRNESS_DEBUG_FLAG - Enable debug output
  MAC_FAIRNESS_EXPERIMENT_ROOT - Override experiment output directory (default: ./experiment)
        """,
    )

    parser.add_argument(
        "config", type=str, help="Path to experiment configuration YAML file"
    )

    parser.add_argument(
        "--grid",
        action="store_true",
        help="Treat config as grid config and expand parameter combinations",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and show what would run without executing",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a grid run from the most recent manifest (requires --grid)",
    )

    args = parser.parse_args()

    # Validate --resume requires --grid
    if args.resume and not args.grid:
        info_print("Error: --resume requires --grid flag")
        return 1

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Handle grid config mode
    if args.grid:
        return run_grid_experiments(args)

    # Regular single-config mode
    config_path = Path(args.config)
    backend = None

    try:
        # Validate config before proceeding
        config = validate_config(config_path)

        # Check if config has _grid section but --grid not specified
        expander = GridConfigExpander(str(config_path))
        if expander.is_grid_config():
            info_print("Error: Config has _grid section but --grid flag not specified.")
            info_print("Use --grid to expand and run parameter combinations.")
            return 1

        schema_version = get_schema_version(config)
        backend = get_backend_type(config)
        exp_name = config.get("experiment_metadata", {}).get(
            "experiment_name", "unknown"
        )
        questions_file = config.get("experiment_metadata", {}).get(
            "questions_file", "unknown"
        )
        benchmark_subcategory = config.get("experiment_metadata", {}).get(
            "benchmark_subcategory", "unknown"
        )

        # Print header
        info_print("=" * 60, prefix=False)
        info_print("Multi-Agent Conversation Framework", prefix=False)
        info_print(f"Protocol Version: {schema_version}", prefix=False)
        info_print("=" * 60, prefix=False)
        info_print(f"✓ Config: {args.config}", prefix=False)
        info_print(f"✓ Experiment: {exp_name}", prefix=False)
        info_print(f"✓ Backend: {backend}", prefix=False)
        info_print(f"✓ Benchmark: {benchmark_subcategory}", prefix=False)
        info_print(f"✓ Questions: {questions_file}", prefix=False)

        # Handle dry-run mode for regular configs
        if args.dry_run:
            info_print("Dry run complete. Configuration is valid.")
            return 0

        debug_print("Debug mode enabled")
        debug_print(
            f"Experiment root: {os.environ.get('MAC_FAIRNESS_EXPERIMENT_ROOT', 'experiment')}"
        )

        # Run job (async)
        orchestrator = ConversationOrchestrator(str(config_path))
        asyncio.run(orchestrator.run_job())

        info_print("=" * 60, prefix=False)
        info_print("✓ Experiment completed successfully!", prefix=False)
        info_print("=" * 60, prefix=False)
        return 0

    except FileNotFoundError as e:
        info_print(f"✗ Error: {e}", prefix=False)
        return 1

    except ValueError as e:
        info_print(f"✗ Configuration error: {e}", prefix=False)
        return 1

    except KeyboardInterrupt:
        info_print("✗ Experiment interrupted by user", prefix=False)
        return 130

    except Exception as e:
        info_print("✗ Experiment failed with error:", prefix=False)
        info_print(f"  {type(e).__name__}: {e}", prefix=False)
        if is_debug_enabled():
            import traceback

            traceback.print_exc()
        return 1

    finally:
        # Always cleanup resources
        info_print("✓ Cleaning up resources...", prefix=False)
        cleanup_resources(backend)


if __name__ == "__main__":
    sys.exit(main())
