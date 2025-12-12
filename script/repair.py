#!/usr/bin/env python3
"""Repair and resume utilities for interrupted experiment runs.

This script provides tools for:
1. Analyzing job manifests to find questions with null status (not succeeded)
2. Resuming/re-running only the null questions

The analyze command does NOT require GPU allocation - it only reads files.
It outputs GPU requirements so you know what to allocate for resume.

Usage:
    # Analyze a job manifest (find null statuses)
    python script/repair.py analyze \\
        experiment/bbq/my_exp/job_manifest/20251212T100000.123Z_local.json

    # Resume null questions from manifest
    python script/repair.py resume \\
        experiment/bbq/my_exp/job_manifest/20251212T100000.123Z_local.json

Environment Variables:
    MAC_FAIRNESS_WORKSPACE - Project root directory (required)
    MAC_FAIRNESS_DEBUG_FLAG - Enable debug output AND verbose transcript (recommended for debugging)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Get project root from environment variable (required)
project_root = Path(os.environ["MAC_FAIRNESS_WORKSPACE"])
sys.path.insert(0, str(project_root))

from src.utils.logging import info_print


def format_path_with_env_var(path: Path) -> str:
    """Format a path to use $MAC_FAIRNESS_WORKSPACE prefix if applicable."""
    path_str = str(path.resolve())
    project_root_str = str(project_root.resolve())
    if path_str.startswith(project_root_str):
        return "$MAC_FAIRNESS_WORKSPACE" + path_str[len(project_root_str):]
    return path_str


def load_json_file(file_path: Path) -> Dict:
    """Load and return a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def load_config_from_snapshot(config_snapshot_path: str) -> Optional[Dict]:
    """Load config from snapshot path, resolving environment variables."""
    if not config_snapshot_path:
        return None

    path = config_snapshot_path
    if path.startswith("$MAC_FAIRNESS_WORKSPACE"):
        path = path.replace("$MAC_FAIRNESS_WORKSPACE", str(project_root))

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = project_root / path

    if not config_path.exists():
        return None

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_gpu_requirements(config: Optional[Dict]) -> Dict[str, Any]:
    """Extract GPU requirements from config.

    Returns:
        Dictionary with:
        - total_gpus: Total GPUs needed
        - per_model: Dict of model_name -> gpu_count
        - details: Human-readable details
    """
    if not config:
        return {"total_gpus": 0, "per_model": {}, "details": "Config not available"}

    model_defs = config.get("model_definitions", {})
    if not model_defs:
        return {"total_gpus": 0, "per_model": {}, "details": "No model definitions"}

    per_model = {}
    details = []

    for model_name, model_def in model_defs.items():
        backend = model_def.get("backend", "unknown")
        if backend == "vllm":
            vllm_config = model_def.get("vllm_config", {})
            tp_size = vllm_config.get("tensor_parallel_size", 1)
            per_model[model_name] = tp_size
            model_path = model_def.get("model_path", "unknown")
            details.append(f"{model_name}: {tp_size} GPU(s) (tp={tp_size}, {model_path})")
        elif backend == "ollama":
            per_model[model_name] = 1
            details.append(f"{model_name}: 1 GPU (ollama)")
        else:
            per_model[model_name] = 0
            details.append(f"{model_name}: 0 GPU ({backend})")

    # Total is max of all models (they share GPUs, not additive)
    total_gpus = max(per_model.values()) if per_model else 0

    return {
        "total_gpus": total_gpus,
        "per_model": per_model,
        "details": "\n".join(details) if details else "No GPU requirements",
    }


def analyze_manifest(manifest_path: Path) -> Dict:
    """Analyze a job manifest to find questions with null status (not succeeded).

    Job manifest is created at job start and tracks which questions succeeded.
    A null status means the question was planned but never succeeded (interrupted,
    failed, or not started).

    Args:
        manifest_path: Path to the job manifest file

    Returns:
        Analysis results dictionary
    """
    manifest = load_json_file(manifest_path)

    # Job manifest uses "questions" array with status field
    questions = manifest["questions"]

    null_questions = []
    succeeded_questions = []

    for q in questions:
        qid = q.get("question_id", f"index_{q.get('index', '?')}")
        status = q.get("status")
        if status is None:
            null_questions.append(qid)
        elif status == "succeeded":
            succeeded_questions.append(qid)
        else:
            # Any other status treated as not succeeded
            null_questions.append(qid)

    config_snapshot_path = manifest.get("config_snapshot_path", "")
    config = load_config_from_snapshot(config_snapshot_path)
    gpu_req = get_gpu_requirements(config)

    return {
        "path": format_path_with_env_var(manifest_path),
        "experiment": manifest.get("experiment_name", "unknown"),
        "benchmark": manifest.get("benchmark_subcategory", "unknown"),
        "num_questions_planned": manifest.get("num_questions_planned", len(questions)),
        "num_questions_processed": manifest.get("num_questions_processed", 0),
        "succeeded": len(succeeded_questions),
        "null_count": len(null_questions),
        "null_question_ids": sorted(null_questions),
        "config_snapshot_path": config_snapshot_path,
        "gpu_requirements": gpu_req,
    }


def print_analysis(analysis: Dict) -> None:
    """Print job manifest analysis results."""
    info_print("=" * 70, prefix=False)
    info_print("JOB MANIFEST ANALYSIS", prefix=False)
    info_print("=" * 70, prefix=False)
    info_print(f"Manifest: {analysis['path']}", prefix=False)
    info_print(f"Experiment: {analysis['experiment']}", prefix=False)
    info_print(f"Benchmark: {analysis['benchmark']}", prefix=False)
    info_print(f"Config snapshot: {analysis['config_snapshot_path']}", prefix=False)
    info_print("-" * 70, prefix=False)
    info_print(f"Questions planned: {analysis['num_questions_planned']}", prefix=False)
    info_print(f"Questions processed: {analysis['num_questions_processed']}", prefix=False)
    info_print(f"Succeeded: {analysis['succeeded']}", prefix=False)
    info_print(f"Null (not succeeded): {analysis['null_count']}", prefix=False)
    info_print("-" * 70, prefix=False)

    if analysis["null_question_ids"]:
        info_print("\nNull question IDs (first 20):", prefix=False)
        for qid in analysis["null_question_ids"][:20]:
            info_print(f"  - {qid}", prefix=False)
        if len(analysis["null_question_ids"]) > 20:
            info_print(f"  ... and {len(analysis['null_question_ids']) - 20} more", prefix=False)

    # GPU requirements
    gpu_req = analysis["gpu_requirements"]
    info_print("\n" + "-" * 70, prefix=False)
    info_print("GPU REQUIREMENTS FOR RESUME", prefix=False)
    info_print("-" * 70, prefix=False)
    info_print(f"Total GPUs needed: {gpu_req['total_gpus']}", prefix=False)
    if gpu_req["per_model"]:
        info_print("\nPer-model breakdown:", prefix=False)
        info_print(gpu_req["details"], prefix=False)

    # Suggestions
    info_print("\n" + "=" * 70, prefix=False)
    info_print("SUGGESTIONS", prefix=False)
    info_print("=" * 70, prefix=False)

    if analysis["null_count"] == 0:
        info_print("All questions succeeded! Nothing to resume.", prefix=False)
    else:
        info_print(f"To resume {analysis['null_count']} null questions:", prefix=False)
        info_print(f"  [ENV VAR SETTINGS HERE] python script/repair.py resume {analysis['path']} [--dry-run]", prefix=False)


def cmd_analyze(args: argparse.Namespace) -> int:
    """Handle the 'analyze' command."""
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        info_print(f"Error: Manifest not found: {manifest_path}")
        return 1

    analysis = analyze_manifest(manifest_path)

    if args.json:
        info_print(json.dumps(analysis, indent=2, default=str), prefix=False)
    else:
        print_analysis(analysis)

    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """Handle the 'resume' command - run null questions from manifest."""
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        info_print(f"Error: Manifest not found: {manifest_path}")
        return 1

    analysis = analyze_manifest(manifest_path)
    question_ids = analysis["null_question_ids"]

    if not question_ids:
        info_print("No null questions to resume!")
        return 0

    info_print(f"Found {len(question_ids)} null questions to resume")
    info_print(f"Config snapshot: {analysis['config_snapshot_path']}")

    if args.dry_run:
        info_print("Dry run - would run these question IDs:")
        for qid in question_ids[:20]:
            info_print(f"  {qid}", prefix=False)
        if len(question_ids) > 20:
            info_print(f"  ... and {len(question_ids) - 20} more", prefix=False)
        return 0

    # Run the experiment with the specified questions
    return _run_questions(
        config_path=analysis["config_snapshot_path"],
        question_ids=question_ids,
        old_manifest_path=manifest_path,
    )


def _run_questions(
    config_path: str,
    question_ids: List[str],
    old_manifest_path: Optional[Path] = None,
) -> int:
    """Run experiment for specific question IDs.

    Args:
        config_path: Path to config snapshot
        question_ids: List of question IDs to run
        old_manifest_path: Path to old manifest to delete after new one is created

    Returns:
        Exit code
    """
    import asyncio

    from src.utils.conversation_orchestrator import ConversationOrchestrator

    # Resolve config path
    if config_path.startswith("$MAC_FAIRNESS_WORKSPACE"):
        config_path = config_path.replace("$MAC_FAIRNESS_WORKSPACE", str(project_root))

    config_full_path = Path(config_path)
    if not config_full_path.is_absolute():
        config_full_path = project_root / config_path

    if not config_full_path.exists():
        info_print(f"Error: Config not found: {config_full_path}")
        return 1

    try:
        # Create orchestrator with existing config snapshot
        orchestrator = ConversationOrchestrator(str(config_full_path))

        # Delete old manifest now that a new one will be created
        if old_manifest_path and old_manifest_path.exists():
            old_manifest_path.unlink()
            info_print(f"Deleted old manifest: {old_manifest_path}")

        # Run experiment with question_ids filter
        asyncio.run(orchestrator.run_experiment(question_ids=set(question_ids)))

        info_print("Resume completed!")
        return 0

    except KeyboardInterrupt:
        info_print("Resume interrupted by user")
        return 130

    except Exception as e:
        info_print(f"Resume failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Repair and resume interrupted experiment runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a job manifest for null statuses",
    )
    analyze_parser.add_argument(
        "manifest", type=str, help="Path to job manifest file"
    )
    analyze_parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )

    # resume command
    resume_parser = subparsers.add_parser(
        "resume", help="Resume null questions from a job manifest"
    )
    resume_parser.add_argument("manifest", type=str, help="Path to job manifest file")
    resume_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be run without running"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "resume":
        return cmd_resume(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
