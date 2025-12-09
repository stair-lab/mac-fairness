#!/usr/bin/env python3
"""vLLM parameter sweep tool for benchmarking configuration options.

Sweeps vLLM parameters (max_num_seqs, gpu_memory_utilization, max_model_len) to
find optimal settings. Questions per run = 2x max(max_num_seqs), capped by dataset size.

Usage:

    # Sweep with  max_num_seqs and gpu_memory_utilization
    python script/cluster/vllm_param_sweep.py --config config/dev_vllm/param_sweep.yaml \
        --max-num-seqs 256 512 --gpu-memory-util 0.85 0.95

    # Clean previous data before sweep
    python script/cluster/vllm_param_sweep.py --clean --config config/dev_vllm/param_sweep.yaml \
        --max-num-seqs 256 512 --gpu-memory-util 0.9

    # Clean only (no sweep)
    python script/cluster/vllm_param_sweep.py --clean --benchmark dev_vllm

    # Dry run to see what would be tested
    python script/cluster/vllm_param_sweep.py --config config/dev_vllm/param_sweep.yaml \
        --max-num-seqs 256 512 1024 --dry-run

    # Add live status or debug flag
    MAC_FAIRNESS_LIVE_STATUS=1 python ...

Environment Variables:
    MAC_FAIRNESS_WORKSPACE - Project root directory (required)
    MAC_FAIRNESS_DEBUG_FLAG - Enable debug output
    MAC_FAIRNESS_LIVE_STATUS - Enable live status display
    MAC_FAIRNESS_EXPERIMENT_ROOT - Override experiment output directory
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Get project root from environment variable (required)
project_root = Path(os.environ["MAC_FAIRNESS_WORKSPACE"])
sys.path.insert(0, str(project_root))

from src.utils import debug_print, is_debug_enabled, is_live_status_enabled


def clean_benchmark_data(benchmark: str) -> None:
    """Clear previous experiment data for the specified benchmark.

    Clears:
    - bookkeeping/config_snapshot/<benchmark>/*
    - $MAC_FAIRNESS_EXPERIMENT_ROOT/<benchmark>/*

    Args:
        benchmark: Benchmark subcategory (e.g., 'dev_vllm')
    """
    # Clear config snapshots
    config_snapshot_dir = project_root / "bookkeeping" / "config_snapshot" / benchmark
    if config_snapshot_dir.exists():
        print(f"Clearing config snapshots: {config_snapshot_dir}")
        shutil.rmtree(config_snapshot_dir)
        config_snapshot_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Cleared {config_snapshot_dir.name}/")

    # Clear experiment results
    exp_root = os.environ.get("MAC_FAIRNESS_EXPERIMENT_ROOT", "experiment")
    exp_root_path = Path(exp_root)
    if not exp_root_path.is_absolute():
        exp_root_path = project_root / exp_root
    experiment_dir = exp_root_path / benchmark

    if experiment_dir.exists():
        print(f"Clearing experiment data: {experiment_dir}")
        shutil.rmtree(experiment_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Cleared {experiment_dir.name}/")


def get_gpu_info() -> Optional[Dict[str, Any]]:
    """Detect GPU information using nvidia-smi.

    Returns:
        Dictionary with GPU info or None if detection fails
    """
    try:
        # Get GPU name
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu_name = result.stdout.strip().split("\n")[0].strip()

        # Get memory
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        memory_mb = int(result.stdout.strip().split("\n")[0].strip())
        memory_gb = memory_mb / 1024

        # Get GPU count
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu_count = int(result.stdout.strip().split("\n")[0].strip())

        return {
            "name": gpu_name,
            "memory_gb": memory_gb,
            "count": gpu_count,
        }
    except Exception as e:
        debug_print(f"GPU detection failed: {e}")
        return None


def load_base_config(config_path: Path) -> Dict[str, Any]:
    """Load and return base configuration.

    Args:
        config_path: Path to base config file

    Returns:
        Parsed configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config is invalid YAML
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    debug_print(f"Loaded base config: {config_path.name}")
    return config


def create_sweep_config(
    base_config: Dict[str, Any],
    params: Dict[str, Any],
    sweep_id: str,
) -> Dict[str, Any]:
    """Create a new config with swept parameters.

    Args:
        base_config: Base configuration dictionary
        params: Parameters to override in vllm_config
        sweep_id: Unique identifier for this sweep run

    Returns:
        New configuration with updated parameters
    """
    import copy

    config = copy.deepcopy(base_config)

    # Update experiment name to include sweep info
    base_name = config.get("experiment_metadata", {}).get(
        "experiment_name", "sweep_experiment"
    )
    config["experiment_metadata"]["experiment_name"] = f"{base_name}_sweep_{sweep_id}"

    # Find the model definitions and update vllm_config
    model_definitions = config.get("model_definitions", {})

    for model_name, model_def in model_definitions.items():
        if "vllm_config" in model_def:
            for key, value in params.items():
                model_def["vllm_config"][key] = value
                debug_print(f"Set {model_name}.vllm_config.{key} = {value}")

    return config


def run_experiment(
    config: Dict[str, Any],
    questions: int,
) -> Tuple[bool, Optional[str]]:
    """Run experiment with given config.

    Args:
        config: Configuration dictionary
        questions: Number of questions to run

    Returns:
        Tuple of (success, experiment_name)
    """
    exp_name = config["experiment_metadata"]["experiment_name"]

    # Use a temp file for the config - run_experiment.py will create the
    # authoritative snapshot via orchestrator.save_config_snapshot()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp_file:
        yaml.dump(config, tmp_file, default_flow_style=False)
        tmp_path = tmp_file.name

    try:
        # Build command
        run_script = project_root / "script" / "run_experiment.py"
        cmd = [
            sys.executable,
            str(run_script),
            tmp_path,
            "--range",
            f"0-{questions}",
        ]

        print(f"Running: {exp_name} ({questions} questions)")
        debug_print(f"Command: {' '.join(cmd)}")

        # Show output in real-time if debug or live status mode is enabled
        show_output = is_debug_enabled() or is_live_status_enabled()

        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=not show_output,
            text=True,
        )
        success = result.returncode == 0
        if not success and not show_output:
            debug_print(f"stdout: {result.stdout}")
            debug_print(f"stderr: {result.stderr}")
        return success, exp_name
    except Exception as e:
        print(f"Error: {exp_name} - {e}")
        return False, exp_name
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def find_job_summary(exp_name: str, benchmark: str) -> Optional[Dict[str, Any]]:
    """Find and load the latest job summary for an experiment.

    Args:
        exp_name: Experiment name
        benchmark: Benchmark subcategory (e.g., dev_vllm)

    Returns:
        Job summary dictionary or None if not found
    """
    exp_root = os.environ.get("MAC_FAIRNESS_EXPERIMENT_ROOT", "experiment")
    summary_dir = project_root / exp_root / benchmark / exp_name / "job_summary"

    if not summary_dir.exists():
        debug_print(f"Summary dir not found: {summary_dir}")
        return None

    summaries = sorted(summary_dir.glob("*.json"))
    if not summaries:
        debug_print(f"No summaries in: {summary_dir}")
        return None

    latest = summaries[-1]
    debug_print(f"Loading summary: {latest}")

    with open(latest) as f:
        return json.load(f)


def collect_results(
    experiments: List[str],
    benchmark: str,
) -> List[Dict[str, Any]]:
    """Collect results from completed experiments.

    Args:
        experiments: List of experiment names
        benchmark: Benchmark subcategory

    Returns:
        List of result dictionaries
    """
    results = []

    for exp_name in experiments:
        summary = find_job_summary(exp_name, benchmark)
        if not summary:
            continue

        tp = summary.get("throughput_performance", {})
        proc = summary.get("processing_statistics", {})

        # Get vLLM config from first model in model_definitions
        model_configs = summary.get("model_definitions", {})
        vllm_cfg = {}
        for _, cfg in model_configs.items():
            if cfg.get("backend") == "vllm":
                vllm_cfg = cfg
                break

        # Extract params from experiment name or config
        results.append(
            {
                "experiment": exp_name,
                "max_num_seqs": vllm_cfg.get("max_num_seqs", "N/A"),
                "gpu_memory_util": vllm_cfg.get("gpu_memory_utilization", "N/A"),
                "max_model_len": vllm_cfg.get("max_model_len", "N/A"),
                "questions": proc.get("questions_attempted", 0),
                "succeeded": proc.get("questions_succeeded", 0),
                "success_rate": proc.get("questions_succeeded", 0)
                / max(proc.get("questions_attempted", 1), 1)
                * 100,
                # Throughput metrics
                "tokens_per_sec": tp.get("tokens_per_second", 0),
                "questions_per_sec": tp.get("questions_per_second", 0),
                "duration_sec": summary.get("duration_seconds", 0),
                # Context length optimization stats
                "max_tokens_prompt": tp.get("max_tokens_prompt", 0),
                "max_tokens_combined": tp.get("max_tokens_combined", 0),
            }
        )

    return results


def print_report(
    results: List[Dict[str, Any]], gpu_info: Optional[Dict[str, Any]]
) -> None:
    """Print formatted comparison report.

    Args:
        results: List of result dictionaries
        gpu_info: GPU information for context
    """
    if not results:
        print("\nNo results to report.")
        return

    print("\n" + "=" * 90)
    print("PARAMETER SWEEP REPORT")
    print("=" * 90)

    if gpu_info:
        print(f"\nGPU: {gpu_info['name']} ({gpu_info['memory_gb']:.0f} GB)")

    # Table 1: Configuration and Success
    print("\n" + "-" * 90)
    print(
        f"{'max_num_seqs':<14} {'gpu_mem':<10} {'Questions':<12} {'Success':<10} {'Duration':<12}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r['max_num_seqs']:<14} {r['gpu_memory_util']:<10} "
            f"{r['questions']:<12} {r['success_rate']:.1f}%{'':<6} "
            f"{r['duration_sec'] / 60:.1f} min"
        )

    # Table 2: Throughput
    print("\n" + "-" * 90)
    print(f"{'max_num_seqs':<14} {'Tokens/sec':<16} {'Questions/sec':<16}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['max_num_seqs']:<14} {r['tokens_per_sec']:<16.2f} "
            f"{r['questions_per_sec']:<16.4f}"
        )

    # Table 3: Context Length Analysis (for max_model_len optimization)
    print("\n" + "-" * 90)
    print("CONTEXT LENGTH ANALYSIS (for max_model_len optimization)")
    print("-" * 90)
    print(
        f"{'max_num_seqs':<14} {'max_model_len':<14} {'Max Prompt':<14} {'Max Combined':<14}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r['max_num_seqs']:<14} {r['max_model_len']:<14} "
            f"{r['max_tokens_prompt']:<14} {r['max_tokens_combined']:<14}"
        )

    print("=" * 90)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="vLLM parameter sweep tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to base config file",
    )

    parser.add_argument(
        "--sweep",
        nargs="+",
        choices=["max_num_seqs", "gpu_memory_utilization", "max_model_len"],
        help="Parameters to sweep",
    )

    parser.add_argument(
        "--values",
        nargs="+",
        type=str,
        help="Values to sweep (for single parameter sweep with --sweep)",
    )

    parser.add_argument(
        "--max-num-seqs",
        nargs="+",
        type=int,
        help="max_num_seqs values to test",
    )

    parser.add_argument(
        "--gpu-memory-util",
        nargs="+",
        type=float,
        help="gpu_memory_utilization values to test",
    )

    parser.add_argument(
        "--max-model-len",
        nargs="+",
        type=int,
        help="max_model_len values to test",
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Benchmark subcategory (required for --clean without --config)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be tested without running",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clear previous experiment data for the benchmark before running",
    )

    args = parser.parse_args()

    # Detect GPU
    gpu_info = get_gpu_info()
    if gpu_info:
        print(f"Detected GPU: {gpu_info['name']} ({gpu_info['memory_gb']:.0f} GB)")
    else:
        print("Warning: Could not detect GPU")

    # Handle --clean without --config (clean-only mode)
    if args.clean and not args.config:
        if not args.benchmark:
            parser.error("--benchmark is required when using --clean without --config")
        if not args.benchmark.startswith("dev_"):
            parser.error(
                f"--clean only allowed for dev_* benchmarks, got '{args.benchmark}'"
            )
        clean_benchmark_data(args.benchmark)
        sys.exit(0)

    # Require config for sweep
    if not args.config:
        parser.error("--config is required")

    config_path = Path(args.config)
    base_config = load_base_config(config_path)

    # Handle --clean with --config: clean the benchmark from config
    if args.clean:
        benchmark = base_config.get("experiment_metadata", {}).get(
            "benchmark_subcategory"
        )
        if not benchmark:
            parser.error(
                "Config missing experiment_metadata.benchmark_subcategory, "
                "cannot determine what to clean"
            )
        if not benchmark.startswith("dev_"):
            parser.error(
                f"--clean only allowed for dev_* benchmarks, got '{benchmark}'"
            )
        clean_benchmark_data(benchmark)

    # Determine parameter ranges
    param_ranges: Dict[str, List[Any]] = {}

    if args.max_num_seqs:
        param_ranges["max_num_seqs"] = args.max_num_seqs
    if args.gpu_memory_util:
        param_ranges["gpu_memory_utilization"] = args.gpu_memory_util
    if args.max_model_len:
        param_ranges["max_model_len"] = args.max_model_len

    # Handle --sweep with --values
    if args.sweep and args.values and len(args.sweep) == 1:
        param_name = args.sweep[0]
        if param_name == "gpu_memory_utilization":
            param_ranges[param_name] = [float(v) for v in args.values]
        else:
            param_ranges[param_name] = [int(v) for v in args.values]

    if not param_ranges:
        parser.error(
            "No parameters to sweep. Use --max-num-seqs, "
            "--gpu-memory-util, --max-model-len, or --sweep with --values"
        )

    # Calculate questions count: 2x the max max_num_seqs value
    max_num_seqs_values = param_ranges.get("max_num_seqs", [32])
    questions_count = max(max_num_seqs_values) * 2

    # Cap by available questions in the dataset
    questions_file = base_config.get("experiment_metadata", {}).get("questions_file")
    if questions_file:
        questions_path = Path(questions_file)
        if not questions_path.is_absolute():
            questions_path = project_root / questions_file
        if questions_path.exists():
            with open(questions_path) as f:
                available_questions = sum(1 for line in f if line.strip())
            questions_count = min(questions_count, available_questions)

    # Generate sweep combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    combinations = list(product(*param_values))

    print("\n" + "=" * 60)
    print("PARAMETER SWEEP")
    print("=" * 60)
    print(f"\nSweeping: {param_names}")
    print(f"Combinations: {len(combinations)}")
    print(f"Questions per run: {questions_count}")

    # Show combinations
    print("\nConfigurations to test:")
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        print(f"  {i + 1}. {params}")

    if args.dry_run:
        print("\n[Dry run - no tests executed]")
        sys.exit(0)

    # Confirm
    response = input("\nContinue? [y/N]: ")
    if response.lower() != "y":
        print("Aborted")
        sys.exit(0)

    # Run sweep
    experiments = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        sweep_id = f"{timestamp}_{i:03d}"

        config = create_sweep_config(base_config, params, sweep_id)
        success, exp_name = run_experiment(
            config,
            questions_count,
        )

        if success:
            experiments.append(exp_name)
            print(f"✓ Completed: {params}")
        else:
            print(f"✗ Failed: {params}")

    # Collect and report results
    benchmark = base_config.get("experiment_metadata", {}).get(
        "benchmark_subcategory", args.benchmark
    )
    results = collect_results(experiments, benchmark)
    print_report(results, gpu_info)


if __name__ == "__main__":
    main()
