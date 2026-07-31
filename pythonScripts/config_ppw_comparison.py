"""
config_ppw_comparison.py
=========================
Compare achieved PPW (the model's predicted-config output), oracle PPW
(best-possible-per-interval, from the full config sweep), and best-static
PPW (single fixed config for the whole run, from the full config sweep)
across multiple benchmarks. Produces a summary CSV and a comparison chart.

Data sources per benchmark:
  --predictions <path>   Output of predict_config.py. One row per interval.
                          Its 'PPW_best' column is the ACHIEVED PPW of the
                          model's predicted config for that interval.
  --sweep <path>         train_with_top3_<bench>.csv. One row per
                          (period, candidate config) pair.
                            - 'ppw_prev' = that candidate config's own
                              ACHIEVED PPW during that period (despite the
                              "_prev" naming -- it is not "previous
                              interval", it's this row's own config/period).
                            - 'PPW_best' = the ORACLE best-possible PPW for
                              that period, repeated on every config-row for
                              that period.
                            - the 7 "*_prev"-suffixed config columns
                              (l2Core0_prev, l2Core1_prev, l3Size_prev,
                              prefetchCore0_prev, prefetchCore1_prev,
                              btbCore0_prev, btbCore1_prev) identify which
                              candidate config that row swept.

Computed per benchmark:
  Achieved (Model) PPW  = mean of predictions['PPW_best'], finite values only
  Oracle PPW            = mean of sweep's per-period PPW_best (deduplicated
                           by period_start), finite values only
  Best Static PPW       = max over candidate configs of
                           (mean of sweep['ppw_prev'] for that config,
                           finite values only)

Usage:
    python config_ppw_comparison.py \\
        --benchmark cholesky --predictions predict_cholesky.csv --sweep train_with_top3_cholesky.csv \\
        --benchmark fft      --predictions predict_fft.csv      --sweep train_with_top3_fft.csv \\
        --benchmark barnes   --predictions predict_barnes.csv   --sweep train_with_top3_barnes.csv \\
        --benchmark radiosity --predictions predict_radiosity.csv --sweep train_with_top3_radiosity.csv \\
        --outdir ppw_comparison_output
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==============================================================================
# COLUMN RESOLUTION (same normalization approach as costAnalysis.py, so this
# script tracks whatever naming variant the pipeline produces)
# ==============================================================================

def _normalize_name(s):
    return s.lower().replace('_', '').replace(' ', '').replace('.', '')


def find_col(columns, expected_name, exclude_substr=None):
    norm = _normalize_name
    excl = norm(exclude_substr) if exclude_substr else None

    def ok(col):
        return excl is None or excl not in norm(col)

    for col in columns:
        if norm(col) == norm(expected_name) and ok(col):
            return col
    for col in columns:
        if norm(expected_name) in norm(col) and ok(col):
            return col
    return None


def resolve_first_present(columns, candidates, exclude_substr=None):
    for c in candidates:
        found = find_col(columns, c, exclude_substr=exclude_substr)
        if found:
            return found
    return None


PRED_PPW_CANDIDATES = ['PPW_best', 'PPW__best', 'ppw_best']
PRED_PERIOD_CANDIDATES = ['period_start']

SWEEP_PERIOD_CANDIDATES = ['period_start']
SWEEP_ACHIEVED_PPW_CANDIDATES = ['ppw_prev']
SWEEP_ORACLE_PPW_CANDIDATES = ['PPW_best', 'PPW__best']

SWEEP_CONFIG_DIMENSIONS = {
    'l2_core0':       ['l2Core0_prev', 'L2core0_prev', 'L2 core 0_prev'],
    'l2_core1':       ['l2Core1_prev', 'L2core1_prev', 'L2 core 1_prev'],
    'l3':             ['l3Size_prev', 'L3_prev'],
    'btb_core0':      ['btbCore0_prev', 'BTBcore0_prev', 'BTB core 0_prev'],
    'btb_core1':      ['btbCore1_prev', 'BTBcore1_prev', 'BTB core 1_prev'],
    'prefetch_core0': ['prefetchCore0_prev', 'Prefetchcore0_prev', 'Prefetch core 0_prev'],
    'prefetch_core1': ['prefetchCore1_prev', 'Prefetchcore1_prev', 'Prefetch core 1_prev'],
}


def _finite(series):
    s = pd.to_numeric(series, errors='coerce')
    return s[np.isfinite(s)]


# ==============================================================================
# PER-BENCHMARK COMPUTATION
# ==============================================================================

def compute_achieved_ppw(predictions_csv, benchmark):
    """Mean of predictions['PPW_best'] (the achieved PPW under the model's
    predicted config each interval), finite values only. Filters to rows
    matching `benchmark` if a benchmark column is present and contains
    other values (defensive, in case a file holds more than one benchmark)."""
    df = pd.read_csv(predictions_csv)
    ppw_col = resolve_first_present(df.columns, PRED_PPW_CANDIDATES)
    if ppw_col is None:
        raise ValueError(f"No PPW column found in {predictions_csv}. Columns: {list(df.columns)}")

    bench_col = find_col(df.columns, 'benchmark')
    if bench_col is not None:
        other_values = set(df[bench_col].unique()) - {benchmark}
        if other_values:
            print(f"  [WARN] {predictions_csv} contains rows for other benchmark(s) {other_values} "
                  f"-- filtering to only '{benchmark}'.")
            df = df[df[bench_col] == benchmark]

    vals = _finite(df[ppw_col])
    n_dropped = len(df) - len(vals)
    return {
        'achieved_ppw_mean': vals.mean() if len(vals) else np.nan,
        'achieved_n_intervals': len(df),
        'achieved_n_finite': len(vals),
        'achieved_n_dropped_nonfinite': n_dropped,
    }


def compute_oracle_and_static_ppw(sweep_csv, chunksize=200_000):
    """
    Single streaming pass over the (potentially large) sweep file.
      - Oracle: dedup PPW_best per period_start (it's repeated across every
        config-row for that period), then mean the finite per-period values.
      - Best static: accumulate sum/count of ppw_prev per candidate config
        (identified by the 7 config-dimension columns), then take the config
        with the highest mean.
    """
    header = pd.read_csv(sweep_csv, nrows=0)
    columns = list(header.columns)

    period_col = resolve_first_present(columns, SWEEP_PERIOD_CANDIDATES)
    achieved_col = resolve_first_present(columns, SWEEP_ACHIEVED_PPW_CANDIDATES)
    oracle_col = resolve_first_present(columns, SWEEP_ORACLE_PPW_CANDIDATES)
    dim_cols = {}
    for key, cands in SWEEP_CONFIG_DIMENSIONS.items():
        col = resolve_first_present(columns, cands)
        if col is None:
            raise ValueError(f"Could not resolve sweep column for dimension '{key}' in {sweep_csv}. "
                              f"Columns available: {columns}")
        dim_cols[key] = col

    if period_col is None or achieved_col is None or oracle_col is None:
        raise ValueError(
            f"Missing required column(s) in {sweep_csv}. "
            f"period={period_col}, ppw_prev={achieved_col}, oracle PPW_best={oracle_col}"
        )

    usecols = [period_col, achieved_col, oracle_col] + list(dim_cols.values())
    usecols = list(dict.fromkeys(usecols))  # de-dup while preserving order

    oracle_by_period = {}          # period_start -> oracle PPW (first finite value seen)
    static_sum = {}                # config_key -> running sum of ppw_prev
    static_count = {}              # config_key -> running finite-value count
    n_rows = 0
    n_static_dropped = 0

    for chunk in pd.read_csv(sweep_csv, usecols=usecols, chunksize=chunksize):
        n_rows += len(chunk)

        # --- oracle: dedup by period ---
        oracle_vals = pd.to_numeric(chunk[oracle_col], errors='coerce')
        finite_oracle = np.isfinite(oracle_vals)
        for period, val in zip(chunk.loc[finite_oracle, period_col], oracle_vals[finite_oracle]):
            if period not in oracle_by_period:
                oracle_by_period[period] = val

        # --- best static: accumulate per config ---
        config_key = pd.Series(list(zip(*[chunk[dim_cols[k]].astype(str) for k in SWEEP_CONFIG_DIMENSIONS])))
        achieved_vals = pd.to_numeric(chunk[achieved_col], errors='coerce')
        finite_static = np.isfinite(achieved_vals)
        n_static_dropped += int((~finite_static).sum())

        for key, val in zip(config_key[finite_static], achieved_vals[finite_static]):
            static_sum[key] = static_sum.get(key, 0.0) + val
            static_count[key] = static_count.get(key, 0) + 1

    if not oracle_by_period:
        oracle_mean = np.nan
    else:
        oracle_mean = float(np.mean(list(oracle_by_period.values())))

    if not static_count:
        best_static_mean = np.nan
        best_static_config = None
        n_candidate_configs = 0
    else:
        means = {k: static_sum[k] / static_count[k] for k in static_count}
        best_key = max(means, key=means.get)
        best_static_mean = means[best_key]
        n_candidate_configs = len(means)
        best_static_config = dict(zip(SWEEP_CONFIG_DIMENSIONS.keys(), best_key))

    return {
        'oracle_ppw_mean': oracle_mean,
        'oracle_n_periods': len(oracle_by_period),
        'best_static_ppw_mean': best_static_mean,
        'best_static_config': best_static_config,
        'n_candidate_configs': n_candidate_configs,
        'sweep_n_rows': n_rows,
        'sweep_n_dropped_nonfinite': n_static_dropped,
    }


def format_config(cfg):
    if cfg is None:
        return ""
    return (f"L2({cfg['l2_core0']}/{cfg['l2_core1']}) "
            f"L3({cfg['l3']}) "
            f"BTB({cfg['btb_core0']}/{cfg['btb_core1']}) "
            f"PF({cfg['prefetch_core0']}/{cfg['prefetch_core1']})")


# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_comparison(summary_df, outpath):
    """
    Grouped bar chart, normalized to % of oracle per benchmark (raw PPW
    magnitudes vary by orders of magnitude across benchmarks, so plotting
    absolute values on one axis would make smaller benchmarks unreadable).
    """
    benchmarks = summary_df['benchmark'].tolist()
    achieved_pct = (summary_df['achieved_ppw_mean'] / summary_df['oracle_ppw_mean'] * 100).tolist()
    static_pct = (summary_df['best_static_ppw_mean'] / summary_df['oracle_ppw_mean'] * 100).tolist()
    oracle_pct = [100.0] * len(benchmarks)

    x = np.arange(len(benchmarks))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width, oracle_pct, width, label='Oracle (per-interval best)', color='#55A868')
    ax.bar(x, achieved_pct, width, label='Achieved (model)', color='#4C72B0')
    ax.bar(x + width, static_pct, width, label='Best static config', color='#C44E52')

    for xi, vals in zip(x, [oracle_pct, achieved_pct, static_pct]):
        pass  # labels added below per-series for clarity

    for offset, series in zip([-width, 0, width], [oracle_pct, achieved_pct, static_pct]):
        for xi, v in zip(x, series):
            ax.text(xi + offset, v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=10)
    ax.set_ylabel("PPW as % of oracle", fontsize=10)
    ax.set_title("Achieved vs. Oracle vs. Best-Static-Config PPW", fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(max(achieved_pct, default=100), max(static_pct, default=100), 100) * 1.15)
    ax.axhline(100, color='#55A868', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(loc='lower right', fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches='tight')
    plt.close(fig)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--benchmark', action='append', required=True, help="Benchmark name (repeat per benchmark)")
    ap.add_argument('--predictions', action='append', required=True, help="Path to predict_config.py output CSV (repeat, aligned with --benchmark)")
    ap.add_argument('--sweep', action='append', required=True, help="Path to train_with_top3_<bench>.csv (repeat, aligned with --benchmark)")
    ap.add_argument('--outdir', default='ppw_comparison_output')
    ap.add_argument('--chunksize', type=int, default=200_000, help="Rows per chunk when streaming sweep files")
    args = ap.parse_args()

    if not (len(args.benchmark) == len(args.predictions) == len(args.sweep)):
        print("ERROR: --benchmark, --predictions, and --sweep must all be given the same number of times, "
              "in matching order.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for bench, pred_path, sweep_path in zip(args.benchmark, args.predictions, args.sweep):
        print(f"\n=== {bench} ===")
        achieved = compute_achieved_ppw(pred_path, bench)
        print(f"  Achieved (model): {achieved['achieved_ppw_mean']:.4e} "
              f"({achieved['achieved_n_finite']}/{achieved['achieved_n_intervals']} finite intervals)")

        sweep_stats = compute_oracle_and_static_ppw(sweep_path, chunksize=args.chunksize)
        print(f"  Oracle:           {sweep_stats['oracle_ppw_mean']:.4e} "
              f"({sweep_stats['oracle_n_periods']} periods)")
        print(f"  Best static:      {sweep_stats['best_static_ppw_mean']:.4e} "
              f"across {sweep_stats['n_candidate_configs']} candidate configs -> "
              f"{format_config(sweep_stats['best_static_config'])}")

        row = {
            'benchmark': bench,
            'achieved_ppw_mean': achieved['achieved_ppw_mean'],
            'oracle_ppw_mean': sweep_stats['oracle_ppw_mean'],
            'best_static_ppw_mean': sweep_stats['best_static_ppw_mean'],
            'best_static_config': format_config(sweep_stats['best_static_config']),
            'achieved_pct_of_oracle': achieved['achieved_ppw_mean'] / sweep_stats['oracle_ppw_mean'] * 100
                if sweep_stats['oracle_ppw_mean'] else np.nan,
            'best_static_pct_of_oracle': sweep_stats['best_static_ppw_mean'] / sweep_stats['oracle_ppw_mean'] * 100
                if sweep_stats['oracle_ppw_mean'] else np.nan,
            'achieved_gain_over_static_pct': (
                (achieved['achieved_ppw_mean'] - sweep_stats['best_static_ppw_mean'])
                / sweep_stats['best_static_ppw_mean'] * 100
            ) if sweep_stats['best_static_ppw_mean'] else np.nan,
            'n_periods': sweep_stats['oracle_n_periods'],
            'n_candidate_configs': sweep_stats['n_candidate_configs'],
            'n_intervals_predicted': achieved['achieved_n_intervals'],
            'n_dropped_nonfinite_achieved': achieved['achieved_n_dropped_nonfinite'],
            'n_dropped_nonfinite_sweep': sweep_stats['sweep_n_dropped_nonfinite'],
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    csv_path = os.path.join(args.outdir, 'ppw_comparison_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSummary CSV saved to: {os.path.abspath(csv_path)}")

    chart_path = os.path.join(args.outdir, 'ppw_comparison_chart.png')
    plot_comparison(summary_df, chart_path)
    print(f"Chart saved to: {os.path.abspath(chart_path)}")


if __name__ == "__main__":
    main()
