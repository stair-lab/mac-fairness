#!/usr/bin/env python3
"""Check progress of running / completed grid jobs.

Usage:
    python script/check_progress.py             # one-shot summary
    python script/check_progress.py --watch     # refresh every 30s
    python script/check_progress.py --watch 10  # refresh every 10s
    python script/check_progress.py --plot      # save bar chart PNG
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("MAC_FAIRNESS_WORKSPACE", Path(__file__).parent.parent))
MANIFEST_DIR = PROJECT_ROOT / "bookkeeping" / "grid_manifest"


def parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def fmt_duration(seconds: float) -> str:
    if seconds < 0:
        return "?"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def bar(done: int, total: int, width: int = 30) -> str:
    if total == 0:
        return "[" + " " * width + "]"
    filled = int(width * done / total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def summarize_manifest(path: Path) -> dict:
    try:
        with open(path) as f:
            m = json.load(f)
    except Exception as e:
        return {"error": str(e), "path": path.name}

    tasks = m.get("tasks", [])
    total = m.get("num_tasks_planned", len(tasks))

    succeeded = [t for t in tasks if t.get("status") == "succeeded"]
    in_progress = [t for t in tasks if t.get("status") == "started" and not t.get("completed_at")]
    failed = [t for t in tasks if t.get("status") not in ("succeeded", "started", None)]

    # Rate from completed tasks: time between first and last completion
    completion_times = sorted(
        [parse_ts(t.get("completed_at")) for t in succeeded if t.get("completed_at")]
    )
    first_started = parse_ts(tasks[0].get("started_at")) if tasks else None

    rate_per_min = None
    if len(completion_times) >= 2:
        elapsed = (completion_times[-1] - completion_times[0]).total_seconds()
        if elapsed > 0:
            rate_per_min = (len(completion_times) - 1) / elapsed * 60

    # ETA
    remaining = total - len(succeeded)
    eta_secs = None
    if rate_per_min and rate_per_min > 0:
        eta_secs = remaining / rate_per_min * 60

    # Elapsed since job started
    elapsed_secs = None
    if first_started:
        elapsed_secs = (now_utc() - first_started).total_seconds()

    # Experiment name from first task
    exp_name = tasks[0].get("experiment_name", path.stem) if tasks else path.stem

    # Current task
    current = None
    if in_progress:
        t = in_progress[0]
        current = f"{t.get('benchmark_subcategory', '?')} [{t.get('grid_sweep_specs', {}).get('prompt_template_config.for_participant.choice_display_format', '?')} / {t.get('grid_sweep_specs', {}).get('prompt_template_config.for_participant.json_field_order', '?')}]"

    return {
        "exp_name": exp_name,
        "total": total,
        "succeeded": len(succeeded),
        "in_progress": len(in_progress),
        "failed": len(failed),
        "remaining": remaining,
        "rate_per_min": rate_per_min,
        "eta_secs": eta_secs,
        "elapsed_secs": elapsed_secs,
        "current": current,
        "pid": m.get("pid"),
    }


def print_summary():
    if not MANIFEST_DIR.exists():
        print("No manifest directory found.")
        return

    manifests = sorted(MANIFEST_DIR.glob("*.json"))
    if not manifests:
        print("No active grid manifests found.")
        return

    now = now_utc().strftime("%H:%M:%S UTC")
    print(f"\n{'='*70}")
    print(f"  Grid Job Progress  —  {now}")
    print(f"{'='*70}")

    for path in manifests:
        s = summarize_manifest(path)
        if "error" in s:
            print(f"\n[{path.name}]  ERROR: {s['error']}")
            continue

        pct = s["succeeded"] / s["total"] * 100 if s["total"] else 0
        b = bar(s["succeeded"], s["total"])

        # Model name: strip timestamp prefix and suffix
        exp = s["exp_name"]
        # e.g. "20260222T231931.676Z_gemma2-9b_1agent_as-ai_v2025-12-10"
        parts = exp.split("_", 1)
        label = parts[1] if len(parts) > 1 else exp

        print(f"\n  {label}  (pid {s['pid']})")
        print(f"  {b} {s['succeeded']}/{s['total']} tasks ({pct:.0f}%)")

        stats = []
        if s["elapsed_secs"] is not None:
            stats.append(f"elapsed {fmt_duration(s['elapsed_secs'])}")
        if s["rate_per_min"] is not None:
            stats.append(f"{s['rate_per_min']:.1f} tasks/min")
        if s["eta_secs"] is not None:
            stats.append(f"ETA {fmt_duration(s['eta_secs'])}")
        if s["in_progress"]:
            stats.append(f"{s['in_progress']} running")
        if s["failed"]:
            stats.append(f"{s['failed']} failed")
        if stats:
            print(f"  {' | '.join(stats)}")
        if s["current"]:
            print(f"  current: {s['current']}")

    print(f"\n{'='*70}\n")


def load_summaries() -> list[dict]:
    if not MANIFEST_DIR.exists():
        return []
    return [summarize_manifest(p) for p in sorted(MANIFEST_DIR.glob("*.json"))]


def short_label(exp_name: str) -> str:
    """Strip timestamp prefix, e.g. '20260222T_gemma2-9b_1agent_as-ai_v...' → 'gemma2-9b'."""
    import re
    m = re.match(r"^\d{8}T\d{6}\.\d+Z_(.+?)_\d+agent", exp_name)
    return m.group(1) if m else exp_name.split("_", 1)[-1]


def plot_progress(out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    summaries = [s for s in load_summaries() if "error" not in s]
    if not summaries:
        print("No manifests found.")
        return

    labels = [short_label(s["exp_name"]) for s in summaries]
    totals    = np.array([s["total"]       for s in summaries], dtype=float)
    succeeded = np.array([s["succeeded"]   for s in summaries], dtype=float)
    running   = np.array([s["in_progress"] for s in summaries], dtype=float)
    pending   = totals - succeeded - running

    x = np.arange(len(summaries))
    fig, ax = plt.subplots(figsize=(max(7, len(summaries) * 1.6), 5))

    ax.bar(x, succeeded, label="Succeeded", color="#4C9A52")
    ax.bar(x, running,   bottom=succeeded, label="Running",   color="#F0A500")
    ax.bar(x, pending,   bottom=succeeded + running, label="Pending", color="#CCCCCC")

    # Annotate each bar with "N / total  (pct%)"
    for i, s in enumerate(summaries):
        pct = s["succeeded"] / s["total"] * 100 if s["total"] else 0
        eta = f"  ETA {fmt_duration(s['eta_secs'])}" if s["eta_secs"] else ""
        ax.text(i, s["total"] + totals.max() * 0.01,
                f"{s['succeeded']}/{s['total']} ({pct:.0f}%){eta}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Grid tasks")
    ax.set_ylim(0, totals.max() * 1.18)
    ax.set_title(f"Job progress  —  {now_utc().strftime('%Y-%m-%d %H:%M UTC')}")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to: {out_path}")


def main():
    args = sys.argv[1:]
    plot  = "--plot"  in args
    watch = "--watch" in args
    args  = [a for a in args if a not in ("--plot", "--watch")]

    interval = 30
    if args:
        try:
            interval = int(args[0])
        except ValueError:
            pass

    if plot:
        out = PROJECT_ROOT / "progress_chart.png"
        plot_progress(out)
    elif watch:
        try:
            while True:
                os.system("clear")
                print_summary()
                print(f"  Refreshing every {interval}s — Ctrl+C to stop\n")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print_summary()


if __name__ == "__main__":
    main()
