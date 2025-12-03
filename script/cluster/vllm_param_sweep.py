#!/usr/bin/env python3
"""Unified vLLM parameter sweep tool for finding optimal configuration.

This script helps discover optimal vLLM parameters (max_num_seqs, gpu_memory_utilization,
max_model_len) through systematic testing. It is GPU-agnostic and model-agnostic.

The tool dynamically generates configs based on:
1. A base config template
2. Parameter ranges to sweep
3. Detected GPU capabilities

Usage:
    # Quick validation test (10 questions, single config)
    python script/cluster/vllm_param_sweep.py --config config/dev_snap/base_sweep.yaml --quick-test

    # Sweep max_num_seqs with detected GPU (default: 512 questions per config)
    python script/cluster/vllm_param_sweep.py --config config/dev_snap/base_sweep.yaml \\
        --sweep max_num_seqs --values 8 16 32 64

    # Sweep multiple parameters
    python script/cluster/vllm_param_sweep.py --config config/dev_snap/base_sweep.yaml \\
        --sweep max_num_seqs gpu_memory_utilization \\
        --max-num-seqs 8 16 32 \\
        --gpu-memory-util 0.85 0.9

    # Generate report from previous runs
    python script/cluster/vllm_param_sweep.py --report-only --benchmark dev_snap

Environment Variables:
    MAC_FAIRNESS_WORKSPACE - Project root directory (required)
    MAC_FAIRNESS_DEBUG_FLAG - Enable debug output
    MAC_FAIRNESS_EXPERIMENT_ROOT - Override experiment output directory
"""

import argparse
import json
import os
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

# Debug flag
DEBUG = os.environ.get("MAC_FAIRNESS_DEBUG_FLAG")


def _debug_print(msg: str) -> None:
    """Print debug message if debug flag is set."""
    if DEBUG:
        print(f"[DEBUG] {msg}")


def _info_print(msg: str) -> None:
    """Print info message (always shown)."""
    print(msg)


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
        _debug_print(f"GPU detection failed: {e}")
        return None


def suggest_param_ranges(gpu_info: Optional[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Suggest parameter ranges based on GPU capabilities.

    Args:
        gpu_info: GPU information dict from get_gpu_info()

    Returns:
        Dictionary of suggested parameter ranges
    """
    if not gpu_info:
        # Conservative defaults for unknown GPU
        return {
            "max_num_seqs": [4, 8, 16],
            "gpu_memory_utilization": [0.85],
            "max_model_len": [4096],
        }

    memory_gb = gpu_info["memory_gb"]

    # Suggest ranges based on GPU memory
    if memory_gb >= 160:  # H200, B200 class
        return {
            "max_num_seqs": [32, 64, 128],
            "gpu_memory_utilization": [0.9],
            "max_model_len": [8192],
        }
    elif memory_gb >= 80:  # H100, A100-80GB class
        return {
            "max_num_seqs": [16, 32, 64],
            "gpu_memory_utilization": [0.9],
            "max_model_len": [8192],
        }
    elif memory_gb >= 48:  # L40S, A6000 class
        return {
            "max_num_seqs": [12, 16, 24],
            "gpu_memory_utilization": [0.85, 0.9],
            "max_model_len": [8192],
        }
    elif memory_gb >= 32:  # V100-32GB class
        return {
            "max_num_seqs": [8, 12, 16],
            "gpu_memory_utilization": [0.85],
            "max_model_len": [8192],
        }
    elif memory_gb >= 16:  # RTX 4090, A4000 class
        return {
            "max_num_seqs": [4, 8, 12],
            "gpu_memory_utilization": [0.85],
            "max_model_len": [4096],
        }
    else:  # < 16GB - not recommended for 8B models
        raise ValueError(
            f"GPU memory ({memory_gb:.0f} GB) is below 16 GB minimum. "
            f"Use a smaller model (1B-3B) or a GPU with more memory."
        )


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

    _debug_print(f"Loaded base config: {config_path.name}")
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

    # Find the model config and update vllm_config
    model_config = config.get("model_config", {})
    models = model_config.get("models", {})

    for model_name, model_def in models.items():
        if "vllm_config" in model_def:
            for key, value in params.items():
                model_def["vllm_config"][key] = value
                _debug_print(f"Set {model_name}.vllm_config.{key} = {value}")

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

        _info_print(f"Running: {exp_name} ({questions} questions)")
        _debug_print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=not DEBUG,  # Show output if debug mode
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        success = result.returncode == 0
        if not success and not DEBUG:
            _debug_print(f"stdout: {result.stdout}")
            _debug_print(f"stderr: {result.stderr}")
        return success, exp_name
    except subprocess.TimeoutExpired:
        _info_print(f"Timeout: {exp_name}")
        return False, exp_name
    except Exception as e:
        _info_print(f"Error: {exp_name} - {e}")
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
        benchmark: Benchmark subcategory (e.g., dev_snap)

    Returns:
        Job summary dictionary or None if not found
    """
    exp_root = os.environ.get("MAC_FAIRNESS_EXPERIMENT_ROOT", "experiment")
    summary_dir = project_root / exp_root / benchmark / exp_name / "job_summary"

    if not summary_dir.exists():
        _debug_print(f"Summary dir not found: {summary_dir}")
        return None

    summaries = sorted(summary_dir.glob("*.json"))
    if not summaries:
        _debug_print(f"No summaries in: {summary_dir}")
        return None

    latest = summaries[-1]
    _debug_print(f"Loading summary: {latest}")

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

        hw = summary.get("hardware_utilization", {})
        tp = summary.get("throughput_performance", {})
        proc = summary.get("processing_statistics", {})
        vllm_cfg = summary.get("vllm_configuration", {})
        token_stats = summary.get("token_time_statistics", {})

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
                "peak_mem_gb": hw.get("peak_gpu_memory_gb", 0),
                "avg_mem_gb": hw.get("average_gpu_memory_gb", 0),
                "kv_hit_rate": hw.get("kv_cache_stats", {}).get("cache_hit_rate", 0)
                * 100,
                "tokens_per_sec": tp.get("tokens_per_second", 0),
                "questions_per_sec": tp.get("questions_per_second", 0),
                "duration_sec": summary.get("duration_seconds", 0),
                # Context length optimization stats
                "max_prompt_tokens": token_stats.get("max_prompt_tokens", 0),
                "max_combined_tokens": token_stats.get("max_combined_tokens", 0),
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
        _info_print("\nNo results to report.")
        return

    _info_print("\n" + "=" * 90)
    _info_print("PARAMETER SWEEP REPORT")
    _info_print("=" * 90)

    if gpu_info:
        _info_print(f"\nGPU: {gpu_info['name']} ({gpu_info['memory_gb']:.0f} GB)")

    # Table 1: Configuration and Success
    _info_print("\n" + "-" * 90)
    _info_print(
        f"{'max_num_seqs':<14} {'gpu_mem':<10} {'Questions':<12} {'Success':<10} {'Duration':<12}"
    )
    _info_print("-" * 90)
    for r in results:
        _info_print(
            f"{r['max_num_seqs']:<14} {r['gpu_memory_util']:<10} "
            f"{r['questions']:<12} {r['success_rate']:.1f}%{'':<6} "
            f"{r['duration_sec'] / 60:.1f} min"
        )

    # Table 2: Memory Usage
    _info_print("\n" + "-" * 90)
    _info_print(
        f"{'max_num_seqs':<14} {'Peak Mem':<14} {'Avg Mem':<14} {'KV Hit Rate':<14}"
    )
    _info_print("-" * 90)
    for r in results:
        _info_print(
            f"{r['max_num_seqs']:<14} {r['peak_mem_gb']:.1f} GB{'':<8} "
            f"{r['avg_mem_gb']:.1f} GB{'':<8} {r['kv_hit_rate']:.1f}%"
        )

    # Table 3: Throughput
    _info_print("\n" + "-" * 90)
    _info_print(f"{'max_num_seqs':<14} {'Tokens/sec':<16} {'Questions/sec':<16}")
    _info_print("-" * 90)
    for r in results:
        _info_print(
            f"{r['max_num_seqs']:<14} {r['tokens_per_sec']:.2f}{'':<12} "
            f"{r['questions_per_sec']:.4f}"
        )

    # Table 4: Context Length Analysis (for max_model_len optimization)
    _info_print("\n" + "-" * 90)
    _info_print("CONTEXT LENGTH ANALYSIS (for max_model_len optimization)")
    _info_print("-" * 90)
    _info_print(
        f"{'max_num_seqs':<14} {'max_model_len':<14} {'Max Prompt':<14} {'Max Combined':<14}"
    )
    _info_print("-" * 90)
    for r in results:
        _info_print(
            f"{r['max_num_seqs']:<14} {r['max_model_len']:<14} "
            f"{r['max_prompt_tokens']:<14} {r['max_combined_tokens']:<14}"
        )

    # Context length recommendation
    if results:
        max_combined = max(r["max_combined_tokens"] for r in results)
        if max_combined > 0:
            # Recommend 20% headroom above observed max
            recommended_len = int(max_combined * 1.2)
            # Round up to nearest power of 2 boundary for efficiency
            power_of_2 = 1
            while power_of_2 < recommended_len:
                power_of_2 *= 2
            _info_print(f"\nMax combined tokens observed: {max_combined}")
            _info_print(f"Recommended max_model_len: {power_of_2} (with 20% headroom)")

    # Recommendations
    if len(results) > 1:
        # Filter successful runs
        successful = [r for r in results if r["success_rate"] >= 95]
        if successful:
            best_tps = max(successful, key=lambda r: r["tokens_per_sec"])
            _info_print("\n" + "=" * 90)
            _info_print("RECOMMENDATION")
            _info_print("=" * 90)
            _info_print(f"\nBest throughput with high success rate:")
            _info_print(f"  max_num_seqs: {best_tps['max_num_seqs']}")
            _info_print(f"  gpu_memory_utilization: {best_tps['gpu_memory_util']}")
            _info_print(f"  tokens/sec: {best_tps['tokens_per_sec']:.2f}")
            _info_print(f"  peak memory: {best_tps['peak_mem_gb']:.1f} GB")

            # Check if still improving
            sorted_by_batch = sorted(successful, key=lambda r: r["max_num_seqs"])
            if len(sorted_by_batch) >= 2:
                last = sorted_by_batch[-1]
                second_last = sorted_by_batch[-2]
                if last["tokens_per_sec"] > second_last["tokens_per_sec"] * 1.05:
                    _info_print(
                        "\nNote: Throughput still improving. Consider testing higher max_num_seqs."
                    )
        else:
            _info_print("\nNo runs with >= 95% success rate. Check for OOM errors.")

    _info_print("=" * 90)


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
        "--quick-test",
        action="store_true",
        help="Run quick validation test (10 questions, single config)",
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
        "--questions",
        type=int,
        default=512,
        help="Number of questions per sweep run (default: 512)",
    )

    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report from existing runs without running new tests",
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default="dev_snap",
        help="Benchmark subcategory for finding results (default: dev_snap)",
    )

    parser.add_argument(
        "--auto-suggest",
        action="store_true",
        help="Auto-suggest parameter ranges based on detected GPU",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be tested without running",
    )

    args = parser.parse_args()

    # Detect GPU
    gpu_info = get_gpu_info()
    if gpu_info:
        _info_print(
            f"Detected GPU: {gpu_info['name']} ({gpu_info['memory_gb']:.0f} GB)"
        )
    else:
        _info_print("Warning: Could not detect GPU")

    # Handle report-only mode
    if args.report_only:
        # Find all experiments in benchmark
        exp_root = os.environ.get("MAC_FAIRNESS_EXPERIMENT_ROOT", "experiment")
        benchmark_dir = project_root / exp_root / args.benchmark

        if not benchmark_dir.exists():
            _info_print(f"No experiments found in: {benchmark_dir}")
            sys.exit(1)

        experiments = [d.name for d in benchmark_dir.iterdir() if d.is_dir()]
        results = collect_results(experiments, args.benchmark)
        print_report(results, gpu_info)
        sys.exit(0)

    # Require config for non-report modes
    if not args.config:
        parser.error("--config required (unless using --report-only)")

    config_path = Path(args.config)
    base_config = load_base_config(config_path)

    # Handle quick test
    if args.quick_test:
        _info_print("\n" + "=" * 60)
        _info_print("QUICK VALIDATION TEST")
        _info_print("=" * 60)

        success, exp_name = run_experiment(
            base_config,
            questions=10,
        )

        if success:
            _info_print("\n✓ Quick test passed!")
            _info_print("\nNext steps:")
            _info_print("  1. Run parameter sweep (modify max_num_seqs accordingly):")
            _info_print(
                f"     python script/cluster/vllm_param_sweep.py --config {args.config} \\"
            )
            _info_print("         --sweep max_num_seqs --max-num-seqs 8 16 32")
        else:
            _info_print("\n✗ Quick test failed. Check configuration and logs.")
            sys.exit(1)

        sys.exit(0)

    # Determine parameter ranges
    param_ranges: Dict[str, List[Any]] = {}

    if args.auto_suggest:
        param_ranges = suggest_param_ranges(gpu_info)
        _info_print(f"\nAuto-suggested ranges: {param_ranges}")

    # Override with explicit args
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
            "No parameters to sweep. Use --auto-suggest, --max-num-seqs, "
            "--gpu-memory-util, --max-model-len, or --sweep with --values"
        )

    # Generate sweep combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    combinations = list(product(*param_values))

    _info_print("\n" + "=" * 60)
    _info_print("PARAMETER SWEEP")
    _info_print("=" * 60)
    _info_print(f"\nSweeping: {param_names}")
    _info_print(f"Combinations: {len(combinations)}")
    _info_print(f"Questions per run: {args.questions}")
    _info_print(f"Estimated runs: {len(combinations)}")

    # Show combinations
    _info_print("\nConfigurations to test:")
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        _info_print(f"  {i + 1}. {params}")

    if args.dry_run:
        _info_print("\n[Dry run - no tests executed]")
        sys.exit(0)

    # Confirm
    response = input("\nContinue? [y/N]: ")
    if response.lower() != "y":
        _info_print("Aborted")
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
            args.questions,
        )

        if success:
            experiments.append(exp_name)
            _info_print(f"✓ Completed: {params}")
        else:
            _info_print(f"✗ Failed: {params}")

    # Collect and report results
    benchmark = base_config.get("experiment_metadata", {}).get(
        "benchmark_subcategory", args.benchmark
    )
    results = collect_results(experiments, benchmark)
    print_report(results, gpu_info)


if __name__ == "__main__":
    main()
