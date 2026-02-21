#!/usr/bin/env python3
"""Run a multi-agent conversation job from configuration file.

This is the main entry point for running experiments locally or on compute clusters.

Usage:
    # Run grid job (parameter sweep with multiple tasks)
    uv run python script/run_job.py config/my_exp/my_grid_config.yaml --grid

    # Run with repetitions (same grid config, different timestamps)
    uv run python script/run_job.py config/my_exp/my_grid_config.yaml --grid --rep 5

    # Dry run (validate config and show what would run without executing)
    uv run python script/run_job.py config/my_exp/my_grid_config.yaml --grid --dry-run

    # Resume: continue the most recently modified incomplete grid manifest
    # (processes one rep at a time, run repeatedly to complete all reps)
    uv run python script/run_job.py bookkeeping/_grid_config_snapshot/{snapshot}.yaml --grid --resume

    # Repair: retry all failed questions across ALL task manifests for a snapshot
    # (scans all reps and retries null questions in each)
    uv run python script/run_job.py bookkeeping/_grid_config_snapshot/{snapshot}.yaml --grid --repair

Environment Variables:
    MAC_FAIRNESS_WORKSPACE - Project root directory (required)
    MAC_FAIRNESS_EXPERIMENT_ROOT - Override experiment output directory (default: ./experiment)
    MAC_FAIRNESS_DEBUG_FLAG - Enable debug output
    MAC_FAIRNESS_LIVE_STATUS - Enable live status display
"""

import argparse
import asyncio
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

# Get project root from environment variable (required)
project_root = Path(os.environ["MAC_FAIRNESS_WORKSPACE"])
sys.path.insert(0, str(project_root))

from src.utils import debug_print, is_debug_enabled
from src.utils.logging import info_print, resolve_path
from src.utils.bookkeeping_manager import (
    BookkeepingManager,
    GridManifestManager,
    set_grid_index,
)
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

    Uses os._exit() for truly immediate termination, bypassing Python's
    cleanup (atexit, finally blocks, asyncio shutdown). This ensures a single
    Ctrl+C is sufficient even when asyncio or vLLM cleanup is blocked.

    Resource cleanup notes:
    - GPU memory: Reclaimed by Linux kernel when process dies
    - vLLM workers: Child processes receive SIGTERM when parent exits
    - File handles: Closed automatically by OS
    - Lock files: Left as empty files (harmless, reused on next run)

    Args:
        signum: Signal number
        _frame: Current stack frame (unused)
    """
    sig_name = signal.Signals(signum).name
    info_print(f"\n✗ Received {sig_name}, exiting immediately...", prefix=False)
    info_print(
        "  (Completed transcripts saved, manifest preserved for resume)", prefix=False
    )
    info_print(
        "\nTo restore the cursor to its default visible state, run:", prefix=False
    )
    info_print("tput cnorm", prefix=False)

    # Print resume command for grid jobs
    if _grid_resume_path:
        info_print("\nTo resume this grid job, run:", prefix=False)
        info_print(
            f"[ENV_VARS] python script/run_job.py {_grid_resume_path} --grid --resume",
            prefix=False,
        )

    # Force immediate exit - bypass all Python cleanup (atexit, finally, asyncio)
    # This ensures single Ctrl+C works even when cleanup code is blocked
    # Resources (GPU memory, child processes) are reclaimed by OS/kernel
    os._exit(128 + signum)


def run_repair(args: argparse.Namespace) -> int:
    """Repair failed questions across all task manifests for a grid config snapshot.

    Unlike --resume which processes one grid manifest at a time, --repair scans
    ALL task manifests that reference the grid config snapshot and retries any
    questions that didn't succeed. This is useful for fixing up partially failed
    runs across multiple repetitions.

    Args:
        args: Parsed command-line arguments (config should be grid config snapshot path)

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    config_path = Path(args.config)

    # Validate that we're using a grid config snapshot path
    config_parent = config_path.parent.name
    config_grandparent = (
        config_path.parent.parent.name if config_path.parent.parent else ""
    )
    is_snapshot_path = (
        config_grandparent == "bookkeeping" and config_parent == "_grid_config_snapshot"
    )
    if not is_snapshot_path:
        info_print("Error: --repair requires using the grid config snapshot path")
        info_print(
            "  Expected: bookkeeping/_grid_config_snapshot/{config}_{timestamp}.yaml"
        )
        info_print(f"  Got: {config_path}")
        info_print(
            "To find your snapshot, check: ls bookkeeping/_grid_config_snapshot/"
        )
        return 1

    # Find all task manifests with incomplete questions
    bookkeeping_manager = BookkeepingManager()
    incomplete_task_manifests = (
        bookkeeping_manager.find_all_task_manifests_by_grid_snapshot(str(config_path))
    )

    if not incomplete_task_manifests:
        info_print("No task manifests with incomplete questions found.")
        info_print("All questions have succeeded, or no tasks have been run yet.")
        return 0

    # Count total null questions
    total_null = sum(len(null_ids) for _, null_ids, _ in incomplete_task_manifests)
    info_print(
        f"Found {len(incomplete_task_manifests)} task manifest(s) with {total_null} incomplete question(s)"
    )

    # Handle dry-run
    if args.dry_run:
        info_print("-" * 60, prefix=False)
        info_print("Task manifests to repair:", prefix=False)
        for (
            manifest_path,
            null_question_ids,
            all_questions,
        ) in incomplete_task_manifests:
            exp_dir = manifest_path.parent.parent
            exp_name = exp_dir.name
            benchmark = exp_dir.parent.name
            succeeded_count = sum(
                1 for q in all_questions.values() if q.get("status") == "succeeded"
            )
            total = len(all_questions)
            info_print(f"  {benchmark}/{exp_name}", prefix=False)
            info_print(
                f"    ├── Progress: {succeeded_count}/{total} succeeded", prefix=False
            )
            info_print(
                f"    └── Null questions: {len(null_question_ids)}", prefix=False
            )
        info_print(f"Dry run complete. {total_null} questions would be retried.")
        return 0

    # Set up signal handler
    global _grid_resume_path
    _grid_resume_path = str(config_path)

    info_print(f"Repairing {len(incomplete_task_manifests)} task manifest(s)...")

    successful = 0
    failed = 0
    backend = None

    for idx, (manifest_path, null_question_ids, all_questions) in enumerate(
        incomplete_task_manifests
    ):
        # Extract experiment info from manifest path
        exp_dir = manifest_path.parent.parent
        exp_name = exp_dir.name
        benchmark = exp_dir.parent.name

        info_print("=" * 60, prefix=False)
        info_print(
            f"Task {idx + 1}/{len(incomplete_task_manifests)}: {benchmark}/{exp_name}"
        )
        info_print(f"  Null questions: {len(null_question_ids)}")
        info_print("=" * 60, prefix=False)

        # Load manifest to get config_snapshot_path
        try:
            import json

            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            config_snapshot_path = manifest.get("config_snapshot_path", "")
            if not config_snapshot_path:
                info_print("  Error: No config_snapshot_path in manifest, skipping")
                failed += 1
                continue

            # Resolve the config snapshot path
            resolved_config_path = resolve_path(config_snapshot_path, project_root)
            if not Path(resolved_config_path).exists():
                info_print(
                    f"  Error: Config snapshot not found: {config_snapshot_path}"
                )
                failed += 1
                continue

        except (json.JSONDecodeError, OSError) as e:
            info_print(f"  Error reading manifest: {e}")
            failed += 1
            continue

        try:
            # Get backend from config for cleanup
            config = validate_config(Path(resolved_config_path))
            backend = get_backend_type(config)

            # Run orchestrator with the specific question IDs
            orchestrator = ConversationOrchestrator(resolved_config_path)

            # Pass the existing question data for transcript_id preservation
            all_succeeded = asyncio.run(
                orchestrator.run_job(
                    question_ids=set(null_question_ids),
                    succeeded_questions=all_questions,
                    existing_snapshot_path=config_snapshot_path,
                    old_manifest_path=manifest_path,
                    skip_cleanup=False,
                )
            )

            if all_succeeded:
                successful += 1
                info_print(f"Task {idx + 1} completed successfully")
            else:
                failed += 1
                info_print(
                    f"Task {idx + 1} completed with some questions not succeeded"
                )

        except Exception as e:
            failed += 1
            error_msg = f"{type(e).__name__}: {e}"
            info_print(f"Task {idx + 1} failed: {error_msg}")
            if is_debug_enabled():
                import traceback

                traceback.print_exc()

    # Print summary
    info_print("=" * 60, prefix=False)
    info_print(
        f"Repair complete: {successful} task(s) fully repaired, {failed} still have issues"
    )
    info_print("=" * 60, prefix=False)

    if failed > 0:
        info_print("\nTo retry remaining failures, run:", prefix=False)
        info_print(
            f"[ENV_VARS] python script/run_job.py {config_path} --grid --repair",
            prefix=False,
        )

    # Cleanup resources
    if backend:
        cleanup_resources(backend)

    return 0 if failed == 0 else 1


def run_grid_experiments(args: argparse.Namespace) -> int:
    """Run experiments from a grid configuration file.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    config_path = Path(args.config)
    rep_count = getattr(args, "rep", 1) or 1

    # For repetitions > 1, we run the entire grid multiple times with different timestamps
    if rep_count > 1:
        # For dry-run, don't save snapshot or print repetition info
        # Just run _run_single_grid which handles dry-run display
        if args.dry_run:
            submission_timestamp = datetime.now(timezone.utc)
            return _run_single_grid(args, config_path, submission_timestamp)

        info_print(f"Running {rep_count} repetitions of the grid configuration")

        # Save grid config snapshot ONCE from the original scratch config
        # All repetitions will reuse this same snapshot
        grid_manifest_manager = GridManifestManager()
        initial_timestamp = datetime.now(timezone.utc)
        grid_config_snapshot_path = grid_manifest_manager._save_grid_config_snapshot(
            str(config_path), initial_timestamp
        )
        info_print(f"Grid config snapshot saved: {grid_config_snapshot_path}")

        # Resolve the snapshot path - all repetitions load config from the snapshot
        resolved_snapshot_path = Path(
            resolve_path(grid_config_snapshot_path, project_root)
        )

        total_success = 0
        total_fail = 0
        for rep_idx in range(rep_count):
            info_print("=" * 60, prefix=False)
            info_print(f"REPETITION {rep_idx + 1}/{rep_count}", prefix=False)
            info_print("=" * 60, prefix=False)
            # Create a new timestamp for each repetition
            rep_timestamp = datetime.now(timezone.utc)
            # Pass resolved_snapshot_path so GridConfigExpander loads from the frozen snapshot
            result = _run_single_grid(
                args, resolved_snapshot_path, rep_timestamp, grid_config_snapshot_path
            )
            if result == 0:
                total_success += 1
            else:
                total_fail += 1
        info_print("=" * 60, prefix=False)
        info_print(
            f"All repetitions complete: {total_success} succeeded, {total_fail} not fully succeeded"
        )
        info_print("=" * 60, prefix=False)

        # Delete the grid config snapshot after all repetitions complete (if all succeeded)
        if total_fail == 0 and resolved_snapshot_path.exists():
            resolved_snapshot_path.unlink(missing_ok=True)
            info_print(f"Grid config snapshot deleted: {grid_config_snapshot_path}")
        else:
            # Print resume command for failed rep runs
            info_print("\nTo resume this grid job, run:", prefix=False)
            info_print(
                f"[ENV_VARS] python script/run_job.py {grid_config_snapshot_path} --grid --resume",
                prefix=False,
            )

        return 0 if total_fail == 0 else 1

    # Single run (no repetitions)
    submission_timestamp = datetime.now(timezone.utc)
    return _run_single_grid(args, config_path, submission_timestamp)


def _run_single_grid(
    args: argparse.Namespace,
    config_path: Path,
    submission_timestamp: datetime,
    existing_grid_snapshot_path: Optional[str] = None,
) -> int:
    """Run a single grid experiment with a specific timestamp.

    Args:
        args: Parsed command-line arguments
        config_path: Path to the grid configuration file
        submission_timestamp: Timestamp for this grid run (used for {runtime.timestamp})
        existing_grid_snapshot_path: Optional path to an existing grid config snapshot.
            When provided (e.g., for rep runs), this snapshot is reused instead of
            creating a new one. This ensures all repetitions share the same snapshot.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    import tempfile

    # Load and expand grid config with the submission timestamp
    expander = GridConfigExpander(
        str(config_path), runtime_timestamp=submission_timestamp
    )

    if not expander.is_grid_config():
        info_print("Error: --grid flag specified but config has no _grid.sweep section")
        return 1

    # Print summary
    expander.print_summary()

    # Get expanded configs with sweep specs
    expanded_configs = expander.expand_with_sweep_specs()

    # Handle dry-run for non-resume mode
    if args.dry_run and not args.resume:
        expander.print_expanded_configs(verbose=False)
        info_print(
            f"Dry run complete. {len(expanded_configs)} configurations would be executed."
        )
        return 0

    # Initialize grid manifest manager and bookkeeping manager
    grid_manifest_manager = GridManifestManager()
    bookkeeping_manager = BookkeepingManager()
    pending_indices: Optional[list] = None
    started_task_info: Dict[int, Dict[str, str]] = {}
    succeeded_task_info: Dict[int, Dict[str, Any]] = {}
    resume_manifest_path: Optional[Path] = None

    # Check for resume mode
    old_grid_manifest_paths: List[Path] = []
    if args.resume:
        # Resume requires using the grid config snapshot path
        config_parent = config_path.parent.name
        config_grandparent = (
            config_path.parent.parent.name if config_path.parent.parent else ""
        )
        is_snapshot_path = (
            config_grandparent == "bookkeeping"
            and config_parent == "_grid_config_snapshot"
        )
        if not is_snapshot_path:
            info_print("Error: --resume requires using the grid config snapshot path")
            info_print(
                "  Expected: bookkeeping/_grid_config_snapshot/{config}_{timestamp}.yaml"
            )
            info_print(f"  Got: {config_path}")
            info_print(
                "To find your snapshot, check: ls bookkeeping/_grid_config_snapshot/"
            )
            print()
            return 1

        # Find all grid manifests matching this snapshot
        # Grid manifest is authoritative - task manifests are only used for tasks
        # that the grid manifest tracks as "started" (to find null questions)
        all_manifests_result = grid_manifest_manager.load_all_manifests_for_resume(
            str(config_path)
        )

        if all_manifests_result is None:
            info_print("No existing manifest found for this snapshot.")
            info_print("Check bookkeeping/grid_manifest/ for available manifests.")
            return 1

        all_started_tasks, old_grid_manifest_paths = all_manifests_result

        # Find the LATEST incomplete grid manifest to resume (most likely the one interrupted)
        # Each rep run has its own grid manifest - resume one at a time
        # Sort by modification time (most recent first)
        sorted_manifests = sorted(
            old_grid_manifest_paths,
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        for manifest_path in sorted_manifests:
            try:
                import json

                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                p, s, succ = grid_manifest_manager._extract_pending_from_manifest(
                    manifest
                )
                if p:  # This manifest has pending work
                    # Use the latest incomplete manifest for resume
                    resume_manifest_path = manifest_path
                    pending_indices = p
                    started_task_info = s
                    succeeded_task_info = succ
                    break  # Found the latest incomplete one
            except (json.JSONDecodeError, OSError):
                continue

        # Count how many manifests have pending work (for user info)
        incomplete_manifest_count = sum(
            1
            for mp in old_grid_manifest_paths
            if mp != resume_manifest_path
            and grid_manifest_manager._has_pending_tasks(mp)
        )

        if pending_indices:
            pending_indices.sort()

        if not pending_indices:
            info_print("All configurations already completed. Nothing to resume.")
            return 0

        started_count = len(started_task_info)
        info_print(
            f"Resuming: {len(pending_indices)} pending of {len(expanded_configs)} configurations"
        )
        if started_count > 0:
            info_print(
                f"  ({started_count} were previously started, will check task manifests for partial progress)"
            )
        if incomplete_manifest_count > 0:
            info_print(
                f"  ({incomplete_manifest_count} other grid manifest(s) also have pending work)"
            )

        # For resume, we continue using the existing manifest instead of creating a new one
        # Set old_grid_manifest_paths to only delete completed manifests, not the one we're resuming
        old_grid_manifest_paths = [
            mp for mp in old_grid_manifest_paths if mp != resume_manifest_path
        ]

    # Handle dry-run for resume mode with tree output
    if args.dry_run:
        info_print("-" * 60, prefix=False)
        info_print("Configurations to resume:", prefix=False)
        for i in pending_indices:
            if i in started_task_info:
                # Started task - use experiment_name from grid manifest (has correct timestamp)
                run_info = started_task_info[i]
                exp_name = run_info["experiment_name"]
                benchmark = run_info["benchmark_subcategory"]
                # Note: don't pass expected_config - the grid manifest already has the
                # correct experiment_name/benchmark, and the current expanded config may
                # have a different timestamp from when the task was originally started
                result = bookkeeping_manager.find_task_manifest_and_get_null_questions(
                    experiment_name=exp_name,
                    benchmark_subcategory=benchmark,
                )

                if result is not None:
                    _, null_question_ids, _, all_questions = result
                    null_count = len(null_question_ids)
                    # all_questions contains both succeeded and null - count succeeded only
                    succeeded_count = (
                        sum(
                            1
                            for q in all_questions.values()
                            if q.get("status") == "succeeded"
                        )
                        if all_questions
                        else 0
                    )
                    total = len(all_questions) if all_questions else 0

                    info_print(
                        f"  [{i}] {benchmark}/{exp_name} (started)", prefix=False
                    )
                    info_print(
                        f"      ├── Progress: {succeeded_count}/{total} succeeded",
                        prefix=False,
                    )
                    info_print(f"      ├── Null questions: {null_count}", prefix=False)

                    # Show first few null question IDs
                    if null_question_ids:
                        preview = null_question_ids[:5]
                        for j, qid in enumerate(preview):
                            is_last = (j == len(preview) - 1) and (
                                len(null_question_ids) <= 5
                            )
                            prefix_char = "└" if is_last else "├"
                            info_print(f"      │   {prefix_char}── {qid}", prefix=False)
                        if len(null_question_ids) > 5:
                            info_print(
                                f"      │   └── ... and {len(null_question_ids) - 5} more",
                                prefix=False,
                            )
                else:
                    # No task manifest found
                    info_print(
                        f"  [{i}] {benchmark}/{exp_name} (started, no manifest found)",
                        prefix=False,
                    )
            else:
                # Pending task - use experiment_name from current expanded config
                config, _ = expanded_configs[i]
                exp_name = config["experiment_metadata"]["experiment_name"]
                benchmark = config["experiment_metadata"]["benchmark_subcategory"]
                info_print(f"  [{i}] {benchmark}/{exp_name} (pending)", prefix=False)

        info_print(
            f"Dry run complete. {len(pending_indices)} configurations would be resumed."
        )
        return 0

    # Create new manifest or reuse existing one for resume
    if args.resume and resume_manifest_path is not None:
        # Resume: reuse the existing manifest, don't create a new one
        manifest_path = resume_manifest_path
        # Get the grid config snapshot path from the existing manifest
        try:
            import json

            with open(manifest_path, "r") as f:
                manifest_data = json.load(f)
            _grid_config_snapshot_path = manifest_data.get(
                "grid_config_snapshot_path", str(config_path)
            )
        except (json.JSONDecodeError, OSError):
            _grid_config_snapshot_path = str(config_path)

        # Delete other completed manifests (not the one we're resuming)
        for old_path in old_grid_manifest_paths:
            try:
                if old_path.exists() and not grid_manifest_manager._has_pending_tasks(
                    old_path
                ):
                    old_path.unlink()
                    info_print(f"Deleted completed grid manifest: {old_path.name}")
            except OSError:
                pass
    else:
        # Fresh run: create new manifest
        if pending_indices is None:
            pending_indices = list(range(len(expanded_configs)))
            succeeded_task_info = {}  # Fresh run, no processed runs to carry over

        # Use the submission_timestamp passed in (already set for runtime placeholder substitution)
        # For rep runs, reuse the existing_grid_snapshot_path instead of creating a new one
        manifest_path, _grid_config_snapshot_path = (
            grid_manifest_manager.save_grid_manifest(
                str(config_path),
                expanded_configs,
                submission_timestamp,
                is_resume=False,
                succeeded_task_info=succeeded_task_info,
                existing_grid_snapshot_path=existing_grid_snapshot_path,
            )
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
        info_print(
            f"Configuration {config_num}/{len(pending_indices)} (index {i}): {exp_name}"
        )
        info_print("=" * 60, prefix=False)

        # Mark as started in manifest
        grid_manifest_manager.mark_task_started(manifest_path, i)

        # Set grid index for job_task_id generation
        set_grid_index(i)

        # For "started" runs, find task manifest and get null question IDs
        # Do this before creating temp file to allow early skip
        question_ids: Optional[Set[str]] = None
        old_task_manifest_path: Optional[Path] = None
        resume_config_path: Optional[str] = (
            None  # Use existing config snapshot for resume
        )
        succeeded_questions: Optional[Dict[str, Dict[str, Any]]] = (
            None  # Carry over for resume
        )
        if i in started_task_info:
            run_info = started_task_info[i]
            started_exp_name = run_info["experiment_name"]
            started_benchmark = run_info["benchmark_subcategory"]
            # Note: don't pass expected_config - the grid manifest already has the
            # correct experiment_name/benchmark, and the current expanded config may
            # have a different timestamp from when the task was originally started
            result = bookkeeping_manager.find_task_manifest_and_get_null_questions(
                experiment_name=started_exp_name,
                benchmark_subcategory=started_benchmark,
            )
            if result is None:
                # Task was marked started but no manifest found - run fresh
                info_print(
                    "  Warning: Task was started but no manifest found, running all questions"
                )
            else:
                (
                    old_task_manifest_path,
                    null_question_ids,
                    task_config_snapshot,
                    succeeded_questions,
                ) = result
                question_ids = set(null_question_ids)
                if not question_ids:
                    info_print(
                        "  All questions already succeeded in previous run, skipping"
                    )
                    grid_manifest_manager.mark_task_succeeded(manifest_path, i)
                    successful += 1
                    continue
                # Validate config snapshot exists and use it for resume
                if task_config_snapshot:
                    # Resolve env var for existence check only
                    resolved_snapshot = resolve_path(task_config_snapshot, project_root)
                    if Path(resolved_snapshot).exists():
                        # Keep original path with env var for portability
                        resume_config_path = task_config_snapshot
                        info_print(
                            f"  Resuming {len(question_ids)} null questions using existing config snapshot"
                        )
                    else:
                        info_print(
                            f"  Warning: Config snapshot not found: {task_config_snapshot}"
                        )
                        info_print(
                            f"  Resuming {len(question_ids)} null questions with current config"
                        )
                else:
                    info_print(
                        f"  Resuming {len(question_ids)} null questions from previous task manifest"
                    )

        tmp_path = None
        try:
            # For resume, use existing config snapshot; otherwise write new temp file
            if resume_config_path:
                # Resolve env var for file access (resume_config_path may have $MAC_FAIRNESS_WORKSPACE)
                config_to_use = resolve_path(resume_config_path, project_root)
            else:
                # Add grid config snapshot path to experiment_metadata for resume tracking
                # This allows find_all_task_manifests_by_grid_snapshot to identify tasks
                # that were created from this grid config (important for rep run resume)
                config["experiment_metadata"]["_grid_config_snapshot_path"] = (
                    _grid_config_snapshot_path
                )

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

            # skip_cleanup optimization disabled for now - vLLM engine reuse across tasks
            # can propagate corrupted state (EngineCore crashes, OOM) to subsequent tasks.
            # TODO: Implement engine health check before reuse, or recreate engine while
            # keeping model loaded in GPU memory.
            # skip_cleanup = grid_manifest_manager.get_task_skip_cleanup(manifest_path, i)
            skip_cleanup = False

            # Pass existing_snapshot_path for resume (avoids creating duplicate snapshots)
            # Pass old_manifest_path for atomic create-then-delete
            all_succeeded = asyncio.run(
                orchestrator.run_job(
                    question_ids=question_ids,
                    succeeded_questions=succeeded_questions,
                    existing_snapshot_path=resume_config_path,
                    old_manifest_path=old_task_manifest_path,
                    skip_cleanup=skip_cleanup,
                )
            )

            # Only mark grid task as succeeded if ALL questions succeeded
            if all_succeeded:
                successful += 1
                grid_manifest_manager.mark_task_succeeded(manifest_path, i)
                info_print(f"Task {config_num} completed successfully")
            else:
                # Some questions failed - leave status as "started" for resume
                failed += 1
                failed_configs.append((i, exp_name, "Some questions not succeeded"))
                info_print(
                    f"Task {config_num} completed with some questions not succeeded"
                )

        except Exception as e:
            # Task crashed - leave status as "started" so it can be resumed
            failed += 1
            error_msg = f"{type(e).__name__}: {e}"
            failed_configs.append((i, exp_name, error_msg))
            info_print(f"Task {config_num} not fully succeeded: {error_msg}")
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
    info_print(
        f"Grid job complete: {successful} succeeded, {failed} not fully succeeded, {skipped} skipped"
    )
    if failed_configs:
        info_print("Some questions not succeeded for configurations:", prefix=False)
        for idx, name, err in failed_configs:
            info_print(f"  [{idx}] {name}: {err}", prefix=False)
    info_print("=" * 60, prefix=False)

    # Delete manifest only if all tasks succeeded
    # For snapshot: only delete if this was NOT a rep run (existing_grid_snapshot_path is None)
    # AND there are no other incomplete manifests sharing this snapshot
    if manifest_path:
        # Check if there are other manifests with pending work
        other_incomplete = False
        if args.resume:
            # Re-scan to see if any other manifests have pending work
            all_manifests_result = grid_manifest_manager.load_all_manifests_for_resume(
                str(config_path)
            )
            if all_manifests_result:
                _, all_paths = all_manifests_result
                for mp in all_paths:
                    if mp != manifest_path and grid_manifest_manager._has_pending_tasks(
                        mp
                    ):
                        other_incomplete = True
                        break

        # For rep runs (existing_grid_snapshot_path is set), never delete snapshot here
        # For resume, only delete snapshot if no other incomplete manifests
        delete_snapshot = (existing_grid_snapshot_path is None) and not other_incomplete
        grid_manifest_manager.delete_if_complete(
            manifest_path, delete_snapshot=delete_snapshot
        )

    # Cleanup resources once at the end
    if backend:
        cleanup_resources(backend)

    return 0 if failed == 0 else 1


def main() -> int:
    """Main entry point for experiment runner.

    Grid configuration is the ONLY entry point. Even single-task experiments
    should use a grid config with one configuration for consistent behavior.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(
        description="Run multi-agent conversation experiments (grid config required)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a grid experiment
  python script/run_job.py config/my_exp/my_grid_config.yaml --grid

  # Run with 5 repetitions (different timestamps each rep)
  python script/run_job.py config/my_exp/my_grid_config.yaml --grid --rep 5

  # Dry run to see expanded configurations
  python script/run_job.py config/my_exp/my_grid_config.yaml --grid --dry-run

  # Resume: continue the most recently modified incomplete grid manifest
  # Picks the latest incomplete rep and finishes it. Run again for next rep.
  python script/run_job.py bookkeeping/_grid_config_snapshot/{snapshot}.yaml --grid --resume

  # Repair: retry failed questions across ALL reps at once
  # Scans all task manifests and retries any null questions.
  python script/run_job.py bookkeeping/_grid_config_snapshot/{snapshot}.yaml --grid --repair

Resume vs Repair:
  --resume: Processes ONE rep at a time (the most recently modified incomplete one).
            Run repeatedly to complete all reps. Grid manifest is authoritative.
  --repair: Processes ALL reps at once. Scans all task manifests matching the
            snapshot and retries any questions that didn't succeed.

Environment Variables:
  MAC_FAIRNESS_DEBUG_FLAG - Enable debug output
  MAC_FAIRNESS_EXPERIMENT_ROOT - Override experiment output directory
        """,
    )

    parser.add_argument(
        "config",
        type=str,
        help="Path to grid configuration YAML file (or snapshot for resume)",
    )

    parser.add_argument(
        "--grid",
        action="store_true",
        required=True,
        help="Required flag to run grid experiments",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and show what would run without executing",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the most recently modified incomplete rep (run repeatedly to finish all reps)",
    )

    parser.add_argument(
        "--repair",
        action="store_true",
        help="Retry all failed questions across ALL reps at once",
    )

    parser.add_argument(
        "--rep",
        type=int,
        default=1,
        metavar="N",
        help="Run the grid N times with different timestamps (for experiment repetitions)",
    )

    args = parser.parse_args()

    # Validate argument combinations
    if args.resume and args.repair:
        info_print("Error: --resume and --repair cannot be used together")
        info_print("  --resume: continue interrupted grid run (one manifest at a time)")
        info_print("  --repair: retry failed questions across all task manifests")
        return 1

    if args.resume and args.rep > 1:
        info_print("Error: --resume and --rep cannot be used together")
        info_print(
            "  --resume continues an interrupted run, --rep starts fresh repetitions"
        )
        return 1

    if args.repair and args.rep > 1:
        info_print("Error: --repair and --rep cannot be used together")
        info_print(
            "  --repair retries failed questions, --rep starts fresh repetitions"
        )
        return 1

    if args.rep < 1:
        info_print("Error: --rep must be at least 1")
        return 1

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Dispatch to appropriate handler
    if args.repair:
        return run_repair(args)
    else:
        return run_grid_experiments(args)


if __name__ == "__main__":
    sys.exit(main())
