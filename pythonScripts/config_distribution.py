"""
config_distribution.py

Generates, per benchmark, a bar chart and a summary table of the
predicted post-silicon-customization configuration distribution.

A "configuration" is defined as the combination of the 7 predicted
columns:
    l2_core0, l2_core1, l3, btb_core0, btb_core1,
    prefetch_core0, prefetch_core1

Usage:
    python config_distribution.py path/to/predictions.csv \
        [--outdir OUTPUT_DIR] [--top N] [--format csv|xlsx]

Outputs (per benchmark, written to OUTPUT_DIR):
    <benchmark>_config_distribution.png   -- bar chart, top-N configs
    <benchmark>_config_distribution.csv   -- full ranked table
    all_benchmarks_config_summary.csv     -- one row per benchmark:
                                              top config + its share
"""

import argparse
import os
import re
import sys

import pandas as pd
import matplotlib.pyplot as plt

CONFIG_COLS = [
    "l2_core0",
    "l2_core1",
    "l3",
    "btb_core0",
    "btb_core1",
    "prefetch_core0",
    "prefetch_core1",
]

# Short display names used when labeling the config combo on the chart
COL_ABBR = {
    "l2_core0": "L2c0",
    "l2_core1": "L2c1",
    "l3": "L3",
    "btb_core0": "BTBc0",
    "btb_core1": "BTBc1",
    "prefetch_core0": "PFc0",
    "prefetch_core1": "PFc1",
}


def format_config_label(row: pd.Series) -> str:
    """Build a compact human-readable label for one configuration row."""
    parts = [f"{COL_ABBR[c]}={row[c]}" for c in CONFIG_COLS]
    return "\n".join(parts)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def compute_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Group rows by the config columns, count occurrences, sort by count."""
    missing = [c for c in CONFIG_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    grouped = (
        df.groupby(CONFIG_COLS, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    grouped["share_pct"] = (grouped["count"] / grouped["count"].sum() * 100).round(1)
    grouped.insert(0, "rank", grouped.index + 1)
    return grouped


def plot_distribution(dist: pd.DataFrame, benchmark: str, top_n: int, outpath: str):
    top = dist.head(top_n).copy()
    labels = [format_config_label(row) for _, row in top.iterrows()]

    fig_height = max(3, 0.55 * len(top) + 1.2)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    y_pos = range(len(top))
    ax.barh(y_pos, top["count"], color="#4C72B0")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()  # highest count on top
    ax.set_xlabel("Count")
    ax.set_title(f"{benchmark}: predicted configuration distribution (top {len(top)})")

    for i, (count, pct) in enumerate(zip(top["count"], top["share_pct"])):
        ax.text(count, i, f"  {count} ({pct}%)", va="center", fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to predictions file (.csv, .tsv, or .xlsx)")
    parser.add_argument("--outdir", default="config_dist_output", help="Output directory")
    parser.add_argument("--top", type=int, default=10, help="Top-N configs to plot per benchmark")
    parser.add_argument(
        "--format", choices=["csv", "xlsx"], default="csv", help="Format for the per-benchmark tables"
    )
    args = parser.parse_args()

    if args.input.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(args.input)
    elif args.input.lower().endswith(".tsv"):
        df = pd.read_csv(args.input, sep="\t")
    else:
        # auto-detect delimiter (handles comma or tab)
        df = pd.read_csv(args.input, sep=None, engine="python")

    if "benchmark" not in df.columns:
        print("ERROR: no 'benchmark' column found in input file.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    summary_rows = []

    for benchmark, sub_df in df.groupby("benchmark"):
        dist = compute_distribution(sub_df)
        safe_name = sanitize_filename(benchmark)

        # full ranked table
        table_path = os.path.join(args.outdir, f"{safe_name}_config_distribution.{args.format}")
        if args.format == "xlsx":
            dist.to_excel(table_path, index=False)
        else:
            dist.to_csv(table_path, index=False)

        # bar chart
        chart_path = os.path.join(args.outdir, f"{safe_name}_config_distribution.png")
        plot_distribution(dist, benchmark, args.top, chart_path)

        top_row = dist.iloc[0]
        summary_rows.append(
            {
                "benchmark": benchmark,
                "n_predictions": int(dist["count"].sum()),
                "n_distinct_configs": len(dist),
                "top_config": format_config_label(top_row).replace("\n", ", "),
                "top_config_count": int(top_row["count"]),
                "top_config_share_pct": top_row["share_pct"],
            }
        )

        print(f"[{benchmark}] {len(dist)} distinct configs, top config = "
              f"{top_row['share_pct']}% ({int(top_row['count'])} of {int(dist['count'].sum())})")

    summary_df = pd.DataFrame(summary_rows).sort_values("benchmark")
    summary_path = os.path.join(args.outdir, "all_benchmarks_config_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\nDone. Outputs written to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
