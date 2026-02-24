#!/usr/bin/env python3
import argparse
import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DISPLAY_ORDER_GROUPS = {
    "arabic_colon": "arabic",
    "arabic_dot": "arabic",
    "arabic_paren": "arabic",
    "letter_colon": "letter",
    "letter_dot": "letter",
    "letter_paren": "letter",
    "roman_colon": "roman",
    "roman_dot": "roman",
    "roman_paren": "roman",
    "none": "none",
    "bullet": "bullet",
}
DISPLAY_ORDER_GROUP_ORDER = ["arabic", "letter", "roman", "none", "bullet"]


def resolve_metric_root() -> Path:
    env_root = os.environ.get("METRIC_ROOT")
    if env_root:
        p = Path(env_root)
        if p.exists():
            return p

    local_root = Path(__file__).resolve().parents[1] / "metric_data"
    if local_root.exists():
        return local_root

    scratch_root = Path("/scratch/users/deonnao/mac-fairness/metric_data")
    if scratch_root.exists():
        return scratch_root

    raise FileNotFoundError(
        "Could not find metric_data. Set METRIC_ROOT or place metric_data under the repo."
    )


def load_scores(metric_root: Path) -> pd.DataFrame:
    pattern = str(metric_root / "*" / "*" / "*" / "*" / "*_scores.csv")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No score files found under {metric_root}")

    rows = []
    for fp in files:
        p = Path(fp)
        # metric_data/{social_group}/{display}/{json_field}/{run}/{file_scores.csv}
        parts = p.parts
        try:
            idx = parts.index("metric_data")
            social_group = parts[idx + 1]
            display_order = parts[idx + 2]
            json_field_order = parts[idx + 3]
            run_folder = parts[idx + 4]
        except (ValueError, IndexError):
            social_group = display_order = json_field_order = run_folder = "unknown"

        df = pd.read_csv(fp)
        df["social_group"] = social_group
        df["display_order"] = display_order
        df["json_field_order"] = json_field_order
        df["run_folder"] = run_folder
        df["file_path"] = fp
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    out["display_order_group"] = out["display_order"].map(DISPLAY_ORDER_GROUPS)
    out = out.dropna(subset=["display_order_group"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot BBQ bias scores as scatter plots using grouped display-order bins."
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Optional list of social groups to include (e.g., bbq_age bbq_race_ethnicity).",
    )
    parser.add_argument(
        "--facet",
        action="store_true",
        help="Facet plots by social group in a grid for selected groups.",
    )
    args = parser.parse_args()

    metric_root = resolve_metric_root()
    df = load_scores(metric_root)

    metrics = [
        "ambig_bias_score",
        "disambig_bias_score",
        "ambig_accuracy",
    ]

    sns.set_theme(style="whitegrid")
    out_dir = metric_root / "plots_grouped_display_order_scatter"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.groups:
        df = df[df["social_group"].isin(args.groups)]
    if df.empty:
        print("No rows available after filtering. Check --groups values and grouped display orders.")
        return 1

    if args.facet:
        for metric in metrics:
            if metric not in df.columns:
                continue
            metric_df = df.copy()
            metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")
            metric_df = metric_df.dropna(subset=[metric])
            if metric_df.empty:
                continue
            g = sns.catplot(
                data=metric_df,
                x="display_order_group",
                y=metric,
                hue="json_field_order",
                col="social_group",
                col_wrap=3,
                kind="strip",
                dodge=True,
                jitter=0.25,
                alpha=0.75,
                marker="x",
                size=6,
                linewidth=1.2,
                order=DISPLAY_ORDER_GROUP_ORDER,
                height=4,
                aspect=1.2,
            )
            g.set_titles("{col_name}")
            g.set_axis_labels("display_order_group", metric)
            for ax in g.axes.flatten():
                ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            out_path = out_dir / f"facet_{metric}_scatterplot_grouped_display_order.png"
            plt.savefig(out_path, dpi=200)
            plt.close()
        return 0

    for social_group in sorted(df["social_group"].unique()):
        df_sg = df[df["social_group"] == social_group]
        sg_out_dir = out_dir / social_group
        sg_out_dir.mkdir(parents=True, exist_ok=True)

        for metric in metrics:
            if metric not in df_sg.columns:
                continue
            plot_df = df_sg.copy()
            plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
            plot_df = plot_df.dropna(subset=[metric])
            if plot_df.empty:
                continue
            plt.figure(figsize=(10, 6))
            sns.stripplot(
                data=plot_df,
                x="display_order_group",
                y=metric,
                hue="json_field_order",
                dodge=True,
                jitter=0.25,
                alpha=0.75,
                marker="x",
                size=6,
                linewidth=1.2,
                order=DISPLAY_ORDER_GROUP_ORDER,
            )
            plt.title(f"{social_group}: {metric} by grouped display_order and json_field_order")
            plt.xlabel("display_order_group")
            plt.ylabel(metric)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            out_path = sg_out_dir / f"{metric}_scatterplot_grouped_display_order.png"
            plt.savefig(out_path, dpi=200)
            plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
