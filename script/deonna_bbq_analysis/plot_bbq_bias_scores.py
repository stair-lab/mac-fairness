# #!/usr/bin/env python3
# import os
# from glob import glob
# from pathlib import Path

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns


# def load_scores(metric_root: Path) -> pd.DataFrame:
#     pattern = str(metric_root / "*" / "*" / "*" / "*" / "*_scores.csv")
#     files = sorted(glob(pattern))
#     if not files:
#         raise FileNotFoundError(f"No score files found under {metric_root}")

#     rows = []
#     for fp in files:
#         p = Path(fp)
#         # metric_data/{social_group}/{display}/{json_field}/{run}/{file_scores.csv}
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


# def main() -> int:
#     metric_root = Path(f"/scratch/users/deonnao/mac-fairness/metric_data")

#     df = load_scores(metric_root)

#     metrics = [
#         "ambig_bias_score",
#         "disambig_bias_score",
#         "ambig_accuracy",
#     ]

#     sns.set_theme(style="whitegrid")
#     out_dir = metric_root / "plots"
#     out_dir.mkdir(parents=True, exist_ok=True)

#     for social_group in sorted(df["social_group"].unique()):
#         df_sg = df[df["social_group"] == social_group]
#         sg_out_dir = out_dir / social_group
#         sg_out_dir.mkdir(parents=True, exist_ok=True)

#         for metric in metrics:
#             if metric not in df_sg.columns:
#                 continue
#             plt.figure(figsize=(12, 6))
#             sns.boxplot(
#                 data=df_sg,
#                 x="display_order",
#                 y=metric,
#                 hue="json_field_order",
#                 showfliers=False,
#             )
#             plt.title(f"{social_group}: {metric} by display_order and json_field_order")
#             plt.xlabel("display_order")
#             plt.ylabel(metric)
#             plt.xticks(rotation=30, ha="right")
#             plt.tight_layout()
#             out_path = sg_out_dir / f"{metric}_boxplot.png"
#             plt.savefig(out_path, dpi=200)
#             plt.close()

#     return 0

# if __name__ == "__main__":
#     raise SystemExit(main())

# # !/usr/bin/env python3
# import argparse
# import os
# from glob import glob
# from pathlib import Path

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns


# def load_scores(metric_root: Path) -> pd.DataFrame:
#     pattern = str(metric_root / "*" / "*" / "*" / "*" / "*_scores.csv")
#     files = sorted(glob(pattern))
#     if not files:
#         raise FileNotFoundError(f"No score files found under {metric_root}")

#     rows = []
#     for fp in files:
#         p = Path(fp)
#         # metric_data/{social_group}/{display}/{json_field}/{run}/{file_scores.csv}
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


# def main() -> int:
#     parser = argparse.ArgumentParser(
#         description="Plot BBQ bias scores with optional social-group faceting."
#     )
#     parser.add_argument(
#         "--groups",
#         nargs="*",
#         default=None,
#         help="Optional list of social groups to include (e.g., bbq_age bbq_race_ethnicity).",
#     )
#     parser.add_argument(
#         "--facet",
#         action="store_true",
#         help="Facet plots by social group in a grid for selected groups.",
#     )
#     args = parser.parse_args()

#     metric_root = Path(f"/scratch/users/deonnao/mac-fairness/metric_data")

#     df = load_scores(metric_root)

#     metrics = [
#         "ambig_bias_score",
#         "disambig_bias_score",
#         "ambig_accuracy",
#     ]

#     sns.set_theme(style="whitegrid")
#     out_dir = metric_root / "plots"
#     out_dir.mkdir(parents=True, exist_ok=True)

#     if args.groups:
#         df = df[df["social_group"].isin(args.groups)]

#     if args.facet:
#         for metric in metrics:
#             if metric not in df.columns:
#                 continue
#             g = sns.catplot(
#                 data=df,
#                 x="display_order",
#                 y=metric,
#                 hue="json_field_order",
#                 col="social_group",
#                 col_wrap=3,
#                 kind="box",
#                 showfliers=False,
#                 height=4,
#                 aspect=1.2,
#             )
#             g.set_titles("{col_name}")
#             g.set_axis_labels("display_order", metric)
#             for ax in g.axes.flatten():
#                 ax.tick_params(axis="x", rotation=30)
#             plt.tight_layout()
#             out_path = out_dir / f"facet_{metric}_boxplot.png"
#             plt.savefig(out_path, dpi=200)
#             plt.close()
#         return 0

#     for social_group in sorted(df["social_group"].unique()):
#         df_sg = df[df["social_group"] == social_group]
#         sg_out_dir = out_dir / social_group
#         sg_out_dir.mkdir(parents=True, exist_ok=True)

#         for metric in metrics:
#             if metric not in df_sg.columns:
#                 continue
#             plt.figure(figsize=(12, 6))
#             sns.boxplot(
#                 data=df_sg,
#                 x="display_order",
#                 y=metric,
#                 hue="json_field_order",
#                 showfliers=False,
#             )
#             plt.title(f"{social_group}: {metric} by display_order and json_field_order")
#             plt.xlabel("display_order")
#             plt.ylabel(metric)
#             plt.xticks(rotation=30, ha="right")
#             plt.tight_layout()
#             out_path = sg_out_dir / f"{metric}_boxplot.png"
#             plt.savefig(out_path, dpi=200)
#             plt.close()

#     return 0

# if __name__ == "__main__":
#     raise SystemExit(main())


#!/usr/bin/env python3
import argparse
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

    return pd.concat(rows, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot BBQ bias scores with scatter plots and optional social-group faceting."
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

    metric_root = Path("/scratch/users/deonnao/mac-fairness/metric_data")
    df = load_scores(metric_root)

    metrics = [
        "ambig_bias_score",
        "disambig_bias_score",
        "ambig_accuracy",
    ]

    sns.set_theme(style="whitegrid")
    out_dir = metric_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.groups:
        df = df[df["social_group"].isin(args.groups)]

    if args.facet:
        for metric in metrics:
            if metric not in df.columns:
                continue
            g = sns.catplot(
                data=df,
                x="display_order",
                y=metric,
                hue="json_field_order",
                col="social_group",
                col_wrap=3,
                kind="strip",
                dodge=True,
                jitter=0.25,
                alpha=0.75,
                height=4,
                aspect=1.2,
            )
            g.set_titles("{col_name}")
            g.set_axis_labels("display_order", metric)
            for ax in g.axes.flatten():
                ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            out_path = out_dir / f"facet_{metric}_scatterplot.png"
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
            plt.figure(figsize=(12, 6))
            sns.stripplot(
                data=df_sg,
                x="display_order",
                y=metric,
                hue="json_field_order",
                dodge=True,
                jitter=0.25,
                alpha=0.75,
            )
            plt.title(f"{social_group}: {metric} by display_order and json_field_order")
            plt.xlabel("display_order")
            plt.ylabel(metric)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            out_path = sg_out_dir / f"{metric}_scatterplot.png"
            plt.savefig(out_path, dpi=200)
            plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())