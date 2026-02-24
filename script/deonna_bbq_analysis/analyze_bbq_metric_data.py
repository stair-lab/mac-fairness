# #!/usr/bin/env python3
# import argparse
# import itertools
# import os
# from glob import glob
# from pathlib import Path

# import numpy as np
# import pandas as pd


# def higher_is_better(metric: str) -> bool:
#     return "accuracy" in metric.lower()


# def resolve_metric_root() -> Path:
#     env_root = os.environ.get("METRIC_ROOT")
#     if env_root:
#         p = Path(env_root)
#         if p.exists():
#             return p

#     local_root = Path(__file__).resolve().parents[1] / "metric_data"
#     if local_root.exists():
#         return local_root

#     scratch_root = Path("/scratch/users/deonnao/mac-fairness/metric_data")
#     if scratch_root.exists():
#         return scratch_root

#     raise FileNotFoundError(
#         "Could not find metric_data. Set METRIC_ROOT or place metric_data under the repo."
#     )


# def load_scores(metric_root: Path) -> pd.DataFrame:
#     pattern = str(metric_root / "*" / "*" / "*" / "*" / "*_scores.csv")
#     files = sorted(glob(pattern))
#     if not files:
#         raise FileNotFoundError(f"No score files found under {metric_root}")

#     rows = []
#     for fp in files:
#         p = Path(fp)
#         parts = p.parts
#         try:
#             idx = parts.index("metric_data")
#             social_group = parts[idx + 1]
#             display_order = parts[idx + 2]
#             json_field_order = parts[idx + 3]
#             run_folder = parts[idx + 4]
#         except (ValueError, IndexError):
#             social_group = display_order = json_field_order = run_folder = "unknown"

#         df = pd.read_csv(fp)
#         df["social_group"] = social_group
#         df["display_order"] = display_order
#         df["json_field_order"] = json_field_order
#         df["run_folder"] = run_folder
#         df["file_path"] = fp
#         rows.append(df)

#     return pd.concat(rows, ignore_index=True)


# def prepare_metric(df: pd.DataFrame, metric: str, use_absolute: bool) -> pd.DataFrame:
#     working = df.copy()
#     working["metric_value"] = pd.to_numeric(working[metric], errors="coerce")
#     working = working.dropna(subset=["metric_value"])
#     if use_absolute and "bias" in metric.lower():
#         working["analysis_value"] = working["metric_value"].abs()
#     else:
#         working["analysis_value"] = working["metric_value"]
#     return working


# def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
#     arr = np.asarray(values, dtype=float)
#     arr = arr[np.isfinite(arr)]
#     if arr.size == 0:
#         return np.nan, np.nan
#     if arr.size == 1:
#         return float(arr[0]), float(arr[0])
#     rng = np.random.default_rng(seed)
#     samples = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
#     low, high = np.percentile(samples, [2.5, 97.5])
#     return float(low), float(high)


# def summarize_condition(
#     working: pd.DataFrame,
#     condition_col: str,
#     ascending: bool,
#     n_boot: int,
#     seed: int,
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     grouped = (
#         working.groupby(["social_group", condition_col], as_index=False)["analysis_value"]
#         .agg(group_mean_score="mean", group_median_score="median", n_runs="count")
#     )
#     run_counts = working.groupby(condition_col, as_index=False).size().rename(
#         columns={"size": "n_runs_total"}
#     )

#     rows = []
#     for cond, chunk in grouped.groupby(condition_col):
#         vals = chunk["group_mean_score"].to_numpy(dtype=float)
#         q1 = float(np.percentile(vals, 25)) if vals.size else np.nan
#         q3 = float(np.percentile(vals, 75)) if vals.size else np.nan
#         ci_low, ci_high = bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed)
#         rows.append(
#             {
#                 condition_col: cond,
#                 "group_mean_score_mean": float(np.mean(vals)) if vals.size else np.nan,
#                 "group_mean_score_median": float(np.median(vals)) if vals.size else np.nan,
#                 "group_mean_score_std": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
#                 "group_mean_score_q1": q1,
#                 "group_mean_score_q3": q3,
#                 "group_mean_score_iqr": q3 - q1 if vals.size else np.nan,
#                 "group_mean_score_ci95_low": ci_low,
#                 "group_mean_score_ci95_high": ci_high,
#                 "n_social_groups": int(chunk["social_group"].nunique()),
#             }
#         )

#     summary = pd.DataFrame(rows).merge(run_counts, on=condition_col, how="left")
#     summary = summary.sort_values("group_mean_score_mean", ascending=ascending).reset_index(drop=True)
#     return summary, grouped


# def compute_win_rates(grouped: pd.DataFrame, condition_col: str, ascending: bool) -> pd.DataFrame:
#     wins: dict[str, float] = {}
#     n_groups = 0
#     for _, gdf in grouped.groupby("social_group"):
#         if gdf.empty:
#             continue
#         n_groups += 1
#         best_val = gdf["group_mean_score"].min() if ascending else gdf["group_mean_score"].max()
#         winners = gdf[gdf["group_mean_score"] == best_val][condition_col].tolist()
#         if not winners:
#             continue
#         share = 1.0 / len(winners)
#         for w in winners:
#             wins[w] = wins.get(w, 0.0) + share

#     rows = []
#     for cond, score in sorted(wins.items(), key=lambda x: x[1], reverse=True):
#         rows.append(
#             {
#                 condition_col: cond,
#                 "wins_weighted": score,
#                 "n_groups_compared": n_groups,
#                 "win_rate": score / n_groups if n_groups else np.nan,
#             }
#         )
#     return pd.DataFrame(rows)


# def compute_pairwise_deltas(
#     grouped: pd.DataFrame, condition_col: str, ascending: bool, n_boot: int, seed: int
# ) -> pd.DataFrame:
#     pivot = grouped.pivot(index="social_group", columns=condition_col, values="group_mean_score")
#     conds = [c for c in pivot.columns.tolist() if pd.notna(c)]
#     rows = []
#     for a, b in itertools.combinations(conds, 2):
#         pair = pivot[[a, b]].dropna()
#         if pair.empty:
#             continue
#         delta = pair[a] - pair[b]
#         vals = delta.to_numpy(dtype=float)
#         ci_low, ci_high = bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed)
#         if ascending:
#             a_better = (vals < 0).sum()
#             b_better = (vals > 0).sum()
#         else:
#             a_better = (vals > 0).sum()
#             b_better = (vals < 0).sum()
#         ties = (vals == 0).sum()
#         rows.append(
#             {
#                 "condition_a": a,
#                 "condition_b": b,
#                 "delta_a_minus_b_mean": float(np.mean(vals)),
#                 "delta_a_minus_b_median": float(np.median(vals)),
#                 "delta_a_minus_b_std": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
#                 "delta_a_minus_b_ci95_low": ci_low,
#                 "delta_a_minus_b_ci95_high": ci_high,
#                 "n_social_groups_paired": int(len(vals)),
#                 "a_better_count": int(a_better),
#                 "b_better_count": int(b_better),
#                 "tie_count": int(ties),
#                 "a_better_rate": float(a_better / len(vals)),
#                 "b_better_rate": float(b_better / len(vals)),
#             }
#         )
#     return pd.DataFrame(rows)


# def compute_joint_tradeoff(
#     df: pd.DataFrame,
#     condition_col: str,
#     use_absolute: bool,
# ) -> pd.DataFrame:
#     candidate_metrics = ["ambig_bias_score", "disambig_bias_score", "ambig_accuracy", "refusal_rate"]
#     rows = []
#     for metric in candidate_metrics:
#         if metric not in df.columns:
#             continue
#         working = prepare_metric(df, metric, use_absolute=use_absolute)
#         if working.empty:
#             continue
#         grouped = (
#             working.groupby(["social_group", condition_col], as_index=False)["analysis_value"]
#             .mean()
#             .rename(columns={"analysis_value": "group_mean_score"})
#         )
#         summary = grouped.groupby(condition_col, as_index=False)["group_mean_score"].mean()
#         summary["metric"] = metric
#         summary = summary.rename(columns={"group_mean_score": "condition_mean"})
#         rows.append(summary)

#     if not rows:
#         return pd.DataFrame()

#     wide = pd.concat(rows, ignore_index=True).pivot(
#         index=condition_col, columns="metric", values="condition_mean"
#     )
#     wide = wide.reset_index()

#     bias_cols = [c for c in ["ambig_bias_score", "disambig_bias_score"] if c in wide.columns]
#     if bias_cols:
#         wide["bias_mean"] = wide[bias_cols].mean(axis=1)
#     if "ambig_accuracy" in wide.columns:
#         wide["accuracy"] = wide["ambig_accuracy"]
#     if "refusal_rate" in wide.columns:
#         wide["refusal_rate_mean"] = wide["refusal_rate"]
#     if "bias_mean" in wide.columns and "accuracy" in wide.columns:
#         wide["pareto_optimal_bias_acc"] = False
#         for i, row_i in wide.iterrows():
#             dominated = False
#             for j, row_j in wide.iterrows():
#                 if i == j:
#                     continue
#                 better_or_equal = (
#                     row_j["bias_mean"] <= row_i["bias_mean"] and row_j["accuracy"] >= row_i["accuracy"]
#                 )
#                 strictly_better = (
#                     row_j["bias_mean"] < row_i["bias_mean"] or row_j["accuracy"] > row_i["accuracy"]
#                 )
#                 if better_or_equal and strictly_better:
#                     dominated = True
#                     break
#             wide.at[i, "pareto_optimal_bias_acc"] = not dominated
#     return wide.sort_values(condition_col)


# def main() -> int:
#     parser = argparse.ArgumentParser(
#         description="Analyze BBQ metric_data with robust cross-group summaries."
#     )
#     parser.add_argument(
#         "--metric-root",
#         default=None,
#         help="Path to metric_data. Defaults to METRIC_ROOT env var, repo metric_data, then /scratch path.",
#     )
#     parser.add_argument(
#         "--metrics",
#         nargs="+",
#         default=["ambig_bias_score", "disambig_bias_score", "ambig_accuracy", "refusal_rate"],
#         help="Metric column(s) to analyze.",
#     )
#     parser.add_argument(
#         "--groups",
#         nargs="*",
#         default=None,
#         help="Optional list of social groups to include.",
#     )
#     parser.add_argument(
#         "--raw-bias",
#         action="store_true",
#         help="Use raw bias metric values. Default uses absolute values for bias metrics only.",
#     )
#     parser.add_argument(
#         "--bootstrap-samples",
#         type=int,
#         default=1000,
#         help="Number of bootstrap samples for 95% confidence intervals.",
#     )
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42,
#         help="Random seed for bootstrap sampling.",
#     )
#     parser.add_argument(
#         "--output-dir",
#         default=None,
#         help="Directory to write CSV summaries. Defaults to <metric_root>/analysis.",
#     )
#     args = parser.parse_args()

#     metric_root = Path(args.metric_root) if args.metric_root else resolve_metric_root()
#     out_dir = Path(args.output_dir) if args.output_dir else metric_root / "analysis"
#     out_dir.mkdir(parents=True, exist_ok=True)

#     df = load_scores(metric_root)
#     if args.groups:
#         df = df[df["social_group"].isin(args.groups)]
#     if df.empty:
#         print("No rows available after filtering.")
#         return 1

#     use_absolute = not args.raw_bias
#     mode = "abs_for_bias_only" if use_absolute else "raw_for_all_metrics"
#     print(f"Metric root: {metric_root}")
#     print(f"Analysis mode: {mode}")
#     print(f"Bootstrap samples: {args.bootstrap_samples}")

#     condition_configs = [
#         ("json_field_order", "json_field_order"),
#         ("display_order", "display_order"),
#     ]

#     for metric in args.metrics:
#         if metric not in df.columns:
#             print(f"Skipping '{metric}' (column not found).")
#             continue
#         working = prepare_metric(df, metric, use_absolute=use_absolute)
#         if working.empty:
#             print(f"Skipping '{metric}' (no numeric values).")
#             continue

#         ascending = not higher_is_better(metric)
#         direction = "lower is better" if ascending else "higher is better"
#         print("")
#         print(f"=== Metric: {metric} ({direction}) ===")

#         for condition_col, label in condition_configs:
#             summary, grouped = summarize_condition(
#                 working,
#                 condition_col=condition_col,
#                 ascending=ascending,
#                 n_boot=args.bootstrap_samples,
#                 seed=args.seed,
#             )
#             wins = compute_win_rates(grouped, condition_col=condition_col, ascending=ascending)
#             deltas = compute_pairwise_deltas(
#                 grouped,
#                 condition_col=condition_col,
#                 ascending=ascending,
#                 n_boot=args.bootstrap_samples,
#                 seed=args.seed,
#             )

#             summary_out = out_dir / f"{metric}_{label}_summary.csv"
#             wins_out = out_dir / f"{metric}_{label}_win_rates.csv"
#             deltas_out = out_dir / f"{metric}_{label}_pairwise_deltas.csv"
#             grouped_out = out_dir / f"{metric}_{label}_group_means.csv"
#             summary.to_csv(summary_out, index=False)
#             wins.to_csv(wins_out, index=False)
#             deltas.to_csv(deltas_out, index=False)
#             grouped.to_csv(grouped_out, index=False)

#             print(f"{label}:")
#             print(summary[[condition_col, "group_mean_score_mean", "group_mean_score_median", "n_social_groups"]].to_string(index=False))
#             if not summary.empty:
#                 best = summary.iloc[0]
#                 print(f"Best {label}: {best[condition_col]} ({best['group_mean_score_mean']:.6f})")
#             print(f"Saved: {summary_out}")
#             print(f"Saved: {wins_out}")
#             print(f"Saved: {deltas_out}")
#             print(f"Saved: {grouped_out}")

#     for condition_col, label in condition_configs:
#         tradeoff = compute_joint_tradeoff(df, condition_col=condition_col, use_absolute=use_absolute)
#         if tradeoff.empty:
#             continue
#         tradeoff_out = out_dir / f"joint_bias_accuracy_refusal_{label}.csv"
#         tradeoff.to_csv(tradeoff_out, index=False)
#         print("")
#         print(f"Joint bias/accuracy/refusal summary by {label}:")
#         cols = [c for c in [condition_col, "bias_mean", "accuracy", "refusal_rate_mean", "pareto_optimal_bias_acc"] if c in tradeoff.columns]
#         print(tradeoff[cols].to_string(index=False))
#         print(f"Saved: {tradeoff_out}")

#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())



#!/usr/bin/env python3
import argparse
import itertools
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd


def higher_is_better(metric: str) -> bool:
    return "accuracy" in metric.lower()


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

    return pd.concat(rows, ignore_index=True)


def prepare_metric(df: pd.DataFrame, metric: str, use_absolute: bool) -> pd.DataFrame:
    working = df.copy()
    working["metric_value"] = pd.to_numeric(working[metric], errors="coerce")
    working = working.dropna(subset=["metric_value"])
    if use_absolute and "bias" in metric.lower():
        working["analysis_value"] = working["metric_value"].abs()
    else:
        working["analysis_value"] = working["metric_value"]
    return working


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def summarize_condition(
    working: pd.DataFrame,
    condition_col: str,
    ascending: bool,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Respect folder structure by balancing across the "other" axis:
    # - For json_field_order analysis, first average within each display_order.
    # - For display_order analysis, first average within each json_field_order.
    strata_col = "display_order" if condition_col == "json_field_order" else "json_field_order"

    per_stratum = (
        working.groupby(["social_group", condition_col, strata_col], as_index=False)["analysis_value"]
        .agg(stratum_mean_score="mean", n_runs_in_stratum="count")
    )
    grouped = (
        per_stratum.groupby(["social_group", condition_col], as_index=False)["stratum_mean_score"]
        .agg(group_mean_score="mean", group_median_score="median")
        .merge(
            per_stratum.groupby(["social_group", condition_col], as_index=False).agg(
                n_runs=("n_runs_in_stratum", "sum"),
                n_strata=(strata_col, "nunique"),
            ),
            on=["social_group", condition_col],
            how="left",
        )
    )
    run_counts = working.groupby(condition_col, as_index=False).size().rename(
        columns={"size": "n_runs_total"}
    )

    rows = []
    for cond, chunk in grouped.groupby(condition_col):
        vals = chunk["group_mean_score"].to_numpy(dtype=float)
        q1 = float(np.percentile(vals, 25)) if vals.size else np.nan
        q3 = float(np.percentile(vals, 75)) if vals.size else np.nan
        ci_low, ci_high = bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed)
        rows.append(
            {
                condition_col: cond,
                "group_mean_score_mean": float(np.mean(vals)) if vals.size else np.nan,
                "group_mean_score_median": float(np.median(vals)) if vals.size else np.nan,
                "group_mean_score_std": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
                "group_mean_score_q1": q1,
                "group_mean_score_q3": q3,
                "group_mean_score_iqr": q3 - q1 if vals.size else np.nan,
                "group_mean_score_ci95_low": ci_low,
                "group_mean_score_ci95_high": ci_high,
                "n_social_groups": int(chunk["social_group"].nunique()),
            }
        )

    summary = pd.DataFrame(rows).merge(run_counts, on=condition_col, how="left")
    summary = summary.sort_values("group_mean_score_mean", ascending=ascending).reset_index(drop=True)
    return summary, grouped


def compute_win_rates(grouped: pd.DataFrame, condition_col: str, ascending: bool) -> pd.DataFrame:
    wins: dict[str, float] = {}
    n_groups = 0
    for _, gdf in grouped.groupby("social_group"):
        if gdf.empty:
            continue
        n_groups += 1
        best_val = gdf["group_mean_score"].min() if ascending else gdf["group_mean_score"].max()
        winners = gdf[gdf["group_mean_score"] == best_val][condition_col].tolist()
        if not winners:
            continue
        share = 1.0 / len(winners)
        for w in winners:
            wins[w] = wins.get(w, 0.0) + share

    rows = []
    for cond, score in sorted(wins.items(), key=lambda x: x[1], reverse=True):
        rows.append(
            {
                condition_col: cond,
                "wins_weighted": score,
                "n_groups_compared": n_groups,
                "win_rate": score / n_groups if n_groups else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_pairwise_deltas(
    grouped: pd.DataFrame, condition_col: str, ascending: bool, n_boot: int, seed: int
) -> pd.DataFrame:
    pivot = grouped.pivot(index="social_group", columns=condition_col, values="group_mean_score")
    conds = [c for c in pivot.columns.tolist() if pd.notna(c)]
    rows = []
    for a, b in itertools.combinations(conds, 2):
        pair = pivot[[a, b]].dropna()
        if pair.empty:
            continue
        delta = pair[a] - pair[b]
        vals = delta.to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed)
        if ascending:
            a_better = (vals < 0).sum()
            b_better = (vals > 0).sum()
        else:
            a_better = (vals > 0).sum()
            b_better = (vals < 0).sum()
        ties = (vals == 0).sum()
        rows.append(
            {
                "condition_a": a,
                "condition_b": b,
                "delta_a_minus_b_mean": float(np.mean(vals)),
                "delta_a_minus_b_median": float(np.median(vals)),
                "delta_a_minus_b_std": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
                "delta_a_minus_b_ci95_low": ci_low,
                "delta_a_minus_b_ci95_high": ci_high,
                "n_social_groups_paired": int(len(vals)),
                "a_better_count": int(a_better),
                "b_better_count": int(b_better),
                "tie_count": int(ties),
                "a_better_rate": float(a_better / len(vals)),
                "b_better_rate": float(b_better / len(vals)),
            }
        )
    return pd.DataFrame(rows)


def compute_joint_tradeoff(
    df: pd.DataFrame,
    condition_col: str,
    use_absolute: bool,
) -> pd.DataFrame:
    candidate_metrics = ["ambig_bias_score", "disambig_bias_score", "ambig_accuracy", "refusal_rate"]
    rows = []
    for metric in candidate_metrics:
        if metric not in df.columns:
            continue
        working = prepare_metric(df, metric, use_absolute=use_absolute)
        if working.empty:
            continue
        grouped = (
            working.groupby(["social_group", condition_col], as_index=False)["analysis_value"]
            .mean()
            .rename(columns={"analysis_value": "group_mean_score"})
        )
        summary = grouped.groupby(condition_col, as_index=False)["group_mean_score"].mean()
        summary["metric"] = metric
        summary = summary.rename(columns={"group_mean_score": "condition_mean"})
        rows.append(summary)

    if not rows:
        return pd.DataFrame()

    wide = pd.concat(rows, ignore_index=True).pivot(
        index=condition_col, columns="metric", values="condition_mean"
    )
    wide = wide.reset_index()

    bias_cols = [c for c in ["ambig_bias_score", "disambig_bias_score"] if c in wide.columns]
    if bias_cols:
        wide["bias_mean"] = wide[bias_cols].mean(axis=1)
    if "ambig_accuracy" in wide.columns:
        wide["accuracy"] = wide["ambig_accuracy"]
    if "refusal_rate" in wide.columns:
        wide["refusal_rate_mean"] = wide["refusal_rate"]
    if "bias_mean" in wide.columns and "accuracy" in wide.columns:
        wide["pareto_optimal_bias_acc"] = False
        for i, row_i in wide.iterrows():
            dominated = False
            for j, row_j in wide.iterrows():
                if i == j:
                    continue
                better_or_equal = (
                    row_j["bias_mean"] <= row_i["bias_mean"] and row_j["accuracy"] >= row_i["accuracy"]
                )
                strictly_better = (
                    row_j["bias_mean"] < row_i["bias_mean"] or row_j["accuracy"] > row_i["accuracy"]
                )
                if better_or_equal and strictly_better:
                    dominated = True
                    break
            wide.at[i, "pareto_optimal_bias_acc"] = not dominated
    return wide.sort_values(condition_col)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze BBQ metric_data with robust cross-group summaries."
    )
    parser.add_argument(
        "--metric-root",
        default=None,
        help="Path to metric_data. Defaults to METRIC_ROOT env var, repo metric_data, then /scratch path.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["ambig_bias_score", "disambig_bias_score", "ambig_accuracy", "refusal_rate"],
        help="Metric column(s) to analyze.",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Optional list of social groups to include.",
    )
    parser.add_argument(
        "--raw-bias",
        action="store_true",
        help="Use raw bias metric values. Default uses absolute values for bias metrics only.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap samples for 95% confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap sampling.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write CSV summaries. Defaults to <metric_root>/analysis.",
    )
    args = parser.parse_args()

    metric_root = Path(args.metric_root) if args.metric_root else resolve_metric_root()
    out_dir = Path(args.output_dir) if args.output_dir else metric_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_scores(metric_root)
    if args.groups:
        df = df[df["social_group"].isin(args.groups)]
    if df.empty:
        print("No rows available after filtering.")
        return 1

    use_absolute = not args.raw_bias
    mode = "abs_for_bias_only" if use_absolute else "raw_for_all_metrics"
    print(f"Metric root: {metric_root}")
    print(f"Analysis mode: {mode}")
    print(f"Bootstrap samples: {args.bootstrap_samples}")

    condition_configs = [
        ("json_field_order", "json_field_order"),
        ("display_order", "display_order"),
    ]

    for metric in args.metrics:
        if metric not in df.columns:
            print(f"Skipping '{metric}' (column not found).")
            continue
        working = prepare_metric(df, metric, use_absolute=use_absolute)
        if working.empty:
            print(f"Skipping '{metric}' (no numeric values).")
            continue

        ascending = not higher_is_better(metric)
        direction = "lower is better" if ascending else "higher is better"
        print("")
        print(f"=== Metric: {metric} ({direction}) ===")

        for condition_col, label in condition_configs:
            summary, grouped = summarize_condition(
                working,
                condition_col=condition_col,
                ascending=ascending,
                n_boot=args.bootstrap_samples,
                seed=args.seed,
            )
            wins = compute_win_rates(grouped, condition_col=condition_col, ascending=ascending)
            deltas = compute_pairwise_deltas(
                grouped,
                condition_col=condition_col,
                ascending=ascending,
                n_boot=args.bootstrap_samples,
                seed=args.seed,
            )

            summary_out = out_dir / f"{metric}_{label}_summary.csv"
            wins_out = out_dir / f"{metric}_{label}_win_rates.csv"
            deltas_out = out_dir / f"{metric}_{label}_pairwise_deltas.csv"
            grouped_out = out_dir / f"{metric}_{label}_group_means.csv"
            summary.to_csv(summary_out, index=False)
            wins.to_csv(wins_out, index=False)
            deltas.to_csv(deltas_out, index=False)
            grouped.to_csv(grouped_out, index=False)

            print(f"{label}:")
            print(summary[[condition_col, "group_mean_score_mean", "group_mean_score_median", "n_social_groups"]].to_string(index=False))
            if not summary.empty:
                best = summary.iloc[0]
                print(f"Best {label}: {best[condition_col]} ({best['group_mean_score_mean']:.6f})")
            print(f"Saved: {summary_out}")
            print(f"Saved: {wins_out}")
            print(f"Saved: {deltas_out}")
            print(f"Saved: {grouped_out}")

    for condition_col, label in condition_configs:
        tradeoff = compute_joint_tradeoff(df, condition_col=condition_col, use_absolute=use_absolute)
        if tradeoff.empty:
            continue
        tradeoff_out = out_dir / f"joint_bias_accuracy_refusal_{label}.csv"
        tradeoff.to_csv(tradeoff_out, index=False)
        print("")
        print(f"Joint bias/accuracy/refusal summary by {label}:")
        cols = [c for c in [condition_col, "bias_mean", "accuracy", "refusal_rate_mean", "pareto_optimal_bias_acc"] if c in tradeoff.columns]
        print(tradeoff[cols].to_string(index=False))
        print(f"Saved: {tradeoff_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())