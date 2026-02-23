#!/usr/bin/env python3
"""Plot accuracy results as grouped bar charts.

Each bar = one (model, dataset) cell.
X axis = BBQ subcategory; grouped bars = models.

Usage:
    python script/plot_results.py                      # all data in bookkeeping/
    python script/plot_results.py --out results.png    # custom output path
    python script/plot_results.py --min-n 10           # skip cells with < 10 questions

Output: PNG saved to <project_root>/results_chart.png (or --out path)
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


PROJECT_ROOT = Path(os.environ.get("MAC_FAIRNESS_WORKSPACE", Path(__file__).parent.parent))
BOOKKEEPING = PROJECT_ROOT / "bookkeeping"
DATA_DIR = PROJECT_ROOT / "data"


# ── helpers ─────────────────────────────────────────────────────────────────

def resolve_path(p: str) -> Path:
    p = p.replace("$MAC_FAIRNESS_WORKSPACE", str(PROJECT_ROOT))
    p = p.replace("$MAC_FAIRNESS_EXPERIMENT_ROOT", str(PROJECT_ROOT / "experiment"))
    return Path(p)


def extract_model_name(experiment_name: str) -> str:
    """Extract short model name from experiment name.

    e.g. '20260222T231931.676Z_gemma2-9b_1agent_as-ai_v2025-12-10' → 'gemma2-9b'
    """
    # Strip leading timestamp (e.g. '20260222T231931.676Z_')
    m = re.match(r"^\d{8}T\d{6}\.\d+Z_(.+?)_\d+agent", experiment_name)
    if m:
        return m.group(1)
    # Fallback: second underscore-delimited token
    parts = experiment_name.split("_")
    return parts[1] if len(parts) > 1 else experiment_name


def extract_dataset(transcript_path: str) -> str:
    """Extract dataset (benchmark_subcategory) from transcript path.

    e.g. '.../experiment/bbq_age_sampled/...transcript/foo.json' → 'bbq_age_sampled'
    """
    p = Path(transcript_path)
    # Walk up until we find the 'experiment' parent
    for part in p.parts:
        if part.startswith("bbq_") or part.startswith("discrim") or part.startswith("diff"):
            return part
    return "unknown"


def load_questions() -> dict:
    """Load all questions from data/ into a dict keyed by question_id."""
    questions = {}
    for jsonl in DATA_DIR.rglob("*.jsonl"):
        try:
            with open(jsonl) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    q = json.loads(line)
                    if "question_id" in q:
                        questions[q["question_id"]] = q
        except Exception:
            pass
    return questions


def load_transcript_answer(transcript_path: str) -> dict | None:
    """Load final answers from a transcript file. Returns dict of agent_id→answer."""
    try:
        p = resolve_path(transcript_path)
        with open(p) as f:
            t = json.load(f)
        return t.get("conversation_summary", {}).get("final_answers", {})
    except Exception:
        return None


# ── data collection ──────────────────────────────────────────────────────────

def collect_results(min_n: int = 1) -> dict:
    """Scan all index files and collect (model, dataset) → accuracy data.

    Returns:
        {
            (model, dataset): {
                "correct": int,
                "total": int,
                "n_agents": int,
            }
        }
    """
    print("Loading questions...")
    questions = load_questions()
    print(f"  Loaded {len(questions)} questions")

    # (model, dataset) → {correct, total}
    cells: dict = defaultdict(lambda: {"correct": 0, "total": 0, "n_agents": 1})

    index_files = sorted(BOOKKEEPING.glob("*_index.jsonl"))
    if not index_files:
        print("No index files found in bookkeeping/")
        return {}

    for idx_path in index_files:
        print(f"  Reading {idx_path.name}...")
        count = 0
        with open(idx_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("status") != "succeeded":
                    continue

                qid = entry["question_id"]
                exp_name = idx_path.stem.replace("_index", "")  # experiment_name from filename
                model = extract_model_name(exp_name)
                dataset = extract_dataset(entry.get("transcript_path", ""))

                q = questions.get(qid)
                if not q:
                    continue
                correct_answer = q.get("correct_answer_id")
                if not correct_answer:
                    continue

                answers = load_transcript_answer(entry["transcript_path"])
                if not answers:
                    continue

                # Use first agent's answer (for 1-agent setup)
                agent_answer = next(iter(answers.values()), None) if answers else None
                if not agent_answer:
                    continue

                key = (model, dataset)
                cells[key]["total"] += 1
                if agent_answer == correct_answer:
                    cells[key]["correct"] += 1
                count += 1

        print(f"    → {count} answered questions processed")

    # Filter cells with too few observations
    result = {}
    for (model, dataset), data in cells.items():
        if data["total"] >= min_n:
            result[(model, dataset)] = data

    return result


# ── plotting ─────────────────────────────────────────────────────────────────

DATASET_LABELS = {
    "bbq_age_sampled": "Age",
    "bbq_disability_status_sampled": "Disability",
    "bbq_gender_identity_sampled": "Gender",
    "bbq_nationality_sampled": "Nationality",
    "bbq_physical_appearance_sampled": "Appearance",
    "bbq_race_ethnicity_sampled": "Race/Ethnicity",
    "bbq_race_x_gender_sampled": "Race×Gender",
    "bbq_race_x_ses_sampled": "Race×SES",
    "bbq_religion_sampled": "Religion",
    "bbq_ses_sampled": "SES",
    "bbq_sexual_orientation_sampled": "Sexual Orient.",
    # Non-sampled variants
    "bbq_age": "Age",
    "bbq_disability_status": "Disability",
    "bbq_gender_identity": "Gender",
    "bbq_nationality": "Nationality",
    "bbq_physical_appearance": "Appearance",
    "bbq_race_ethnicity": "Race/Ethnicity",
    "bbq_race_x_gender": "Race×Gender",
    "bbq_race_x_ses": "Race×SES",
    "bbq_religion": "Religion",
    "bbq_ses": "SES",
    "bbq_sexual_orientation": "Sexual Orient.",
}

MODEL_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


def plot_results(cells: dict, out_path: Path):
    if not cells:
        print("No data to plot.")
        return

    # Determine all models and datasets
    models = sorted({m for (m, _) in cells})
    datasets = sorted({d for (_, d) in cells})

    # Order datasets by a canonical list if possible
    canonical_order = list(DATASET_LABELS.keys())
    datasets.sort(key=lambda d: canonical_order.index(d) if d in canonical_order else 999)

    n_datasets = len(datasets)
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(max(12, n_datasets * 1.5), 6))

    x = np.arange(n_datasets)
    total_width = 0.8
    bar_width = total_width / n_models
    offsets = np.linspace(-total_width / 2 + bar_width / 2, total_width / 2 - bar_width / 2, n_models)

    for mi, (model, offset) in enumerate(zip(models, offsets)):
        accs = []
        ns = []
        for dataset in datasets:
            data = cells.get((model, dataset))
            if data and data["total"] > 0:
                accs.append(data["correct"] / data["total"] * 100)
                ns.append(data["total"])
            else:
                accs.append(float("nan"))
                ns.append(0)

        color = MODEL_COLORS[mi % len(MODEL_COLORS)]
        bars = ax.bar(x + offset, accs, bar_width * 0.9, label=model, color=color, alpha=0.85)

        # Annotate with n if small
        for xi, (acc, n) in enumerate(zip(accs, ns)):
            if not np.isnan(acc) and n > 0:
                ax.text(
                    x[xi] + offset, acc + 0.8,
                    f"n={n}",
                    ha="center", va="bottom", fontsize=6, color="#333333",
                    rotation=90 if n_models > 3 else 0,
                )

    # Reference line at 33.3% (random for 3-choice MCQ)
    ax.axhline(100 / 3, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Random (33%)")
    # Reference line at 100%
    ax.axhline(100, color="black", linestyle=":", linewidth=0.5, alpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.set_title("BBQ Accuracy by Model and Dataset\n(partial results — jobs still running)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to: {out_path}")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot BBQ accuracy results as bar chart")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "results_chart.png"), help="Output PNG path")
    parser.add_argument("--min-n", type=int, default=1, help="Minimum questions per cell to include")
    args = parser.parse_args()

    print("Collecting results...")
    cells = collect_results(min_n=args.min_n)

    if not cells:
        print("No results found. Are the jobs running and have they completed at least one task?")
        sys.exit(1)

    # Summary
    print(f"\nResults summary ({len(cells)} model×dataset cells):")
    models = sorted({m for (m, _) in cells})
    datasets = sorted({d for (_, d) in cells})
    header = f"{'Dataset':<30}" + "".join(f"{m:>20}" for m in models)
    print(header)
    print("-" * len(header))
    for d in datasets:
        row = f"{DATASET_LABELS.get(d, d):<30}"
        for m in models:
            data = cells.get((m, d))
            if data and data["total"] > 0:
                acc = data["correct"] / data["total"] * 100
                row += f"{acc:>17.1f}% ({data['total']})"
            else:
                row += f"{'—':>20}"
        print(row)

    print()
    plot_results(cells, Path(args.out))


if __name__ == "__main__":
    main()
