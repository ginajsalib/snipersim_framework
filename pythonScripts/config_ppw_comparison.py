"""
config_ppw_comparison.py
=========================
Compare achieved PPW (what the model's predicted config actually measured),
oracle PPW (best-possible-per-interval), and best-static PPW (single fixed
config for the whole run) across multiple benchmarks. Produces a summary CSV
and a comparison chart.

Data sources per benchmark:
  --predictions <path>   Output of predict_config.py. NOTE: this file has
                          one row per *starting configuration* for a given
                          interval, so it is completely normal for the same
                          period_start to appear many times -- that is not
                          a data problem. What must actually hold across
                          those repeated rows is that they agree on the
                          model's predicted config for that period; if they
                          don't, that's a real anomaly (see loader below).
                          Its lowercase, underscore-separated columns
                          (l2_core0, l2_core1, l3, btb_core0, btb_core1,
                          prefetch_core0, prefetch_core1) are the model's
                          PREDICTED config for that interval. NOTE: this
                          file's own 'PPW_best' column is NOT used as
                          achieved PPW -- it is the oracle label carried
                          through from training (it sits next to a separate,
                          unprefixed camelCase config block -- L2core0,
                          btbCore0, prefetcher -- which is the oracle-best
                          config, not the prediction).
  --sweep <path>         train_with_top3_<bench>.csv. One row per
                          (period, candidate config) pair.
                            - 'ppw_prev' = that candidate config's own
                              measured PPW during that period (despite the
                              "_prev" naming -- it is this row's own
                              period/config measurement).
                            - 'PPW_best' on a given row describes the
                              interval identified by that SAME row's own
                              'period_start' column -- it is one checkpoint
                              LATER than the interval described by that
                              row's 'ppw_prev' (which is keyed off
                              'period_start_val_prev'). In other words,
                              'ppw_prev' and 'PPW_best' on the same physical
                              row are NOT describing the same interval, even
                              though earlier code assumed they were. The
                              oracle value for a given prediction period must
                              therefore come from a *different* row than the
                              achieved value: the row whose own period_start
                              equals that period, not the row whose
                              period_start_val_prev equals that period.
                            - the 7 "*_prev"-suffixed config columns
                              (l2Core0_prev, l2Core1_prev, l3Size_prev,
                              prefetchCore0_prev, prefetchCore1_prev,
                              btbCore0_prev, btbCore1_prev) identify which
                              candidate config that row swept.

Computed per benchmark, over a common matched set of periods (finite oracle
+ the model's predicted config actually found with a finite measurement):
  Achieved (Model) PPW  = mean of ppw_prev from the sweep row whose
                           period_start_val_prev matches the period and
                           whose *_prev config matches the model's
                           predicted config for that period
  Oracle PPW             = mean of PPW_best from the sweep row whose OWN
                           period_start matches the period (a different row
                           than the one used for achieved PPW -- see note
                           above)
  Best Static PPW        = max over candidate configs of
                            (mean of that config's ppw_prev across periods)

Because the oracle-row and achieved-row for the same period can appear in
different chunks (or even different relative order) while streaming the
sweep file, this script does a genuine two-pass scan: pass 1 resolves every
period's oracle value first (from the full file), and only once that's
complete does pass 2 resolve achieved values and best-static accumulation.
This avoids order-dependent bugs where a row read early would otherwise be
incorrectly dropped just because its matching oracle row hadn't streamed
by yet.

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

# The predicted (model-chosen) config columns in predictions.csv are the
# lowercase, underscore-separated ones with NO suffix -- e.g. 'l2_core0'.
# This predictions file *also* contains a separate unprefixed camelCase
# block ('L2core0', 'btbCore0', 'prefetcher') which normalizes to the exact
# same key as 'l2_core0' under the generic fuzzy matcher above, so for THIS
# specific resolution we require an exact, case-sensitive name match first
# -- fuzzy/normalized matching would silently pick either block.
PRED_CONFIG_EXACT_CANDIDATES = {
    'l2_core0':       ['l2_core0'],
    'l2_core1':       ['l2_core1'],
    'l3':             ['l3'],
    'btb_core0':      ['btb_core0'],
    'btb_core1':      ['btb_core1'],
    'prefetch_core0': ['prefetch_core0'],
    'prefetch_core1': ['prefetch_core1'],
}


def resolve_predicted_config_columns(columns):
    """Resolve the model's predicted-config columns using an exact,
    case-sensitive match first (see note above); only falls back to fuzzy
    normalized matching -- with a loud warning -- if no exact match exists,
    since fuzzy matching here risks silently picking the oracle-label block
    instead of the predicted-config block."""
    resolved = {}
    used_fuzzy = []
    for key, exact_names in PRED_CONFIG_EXACT_CANDIDATES.items():
        col = next((c for c in exact_names if c in columns), None)
        if col is None:
            col = find_col(columns, exact_names[0], exclude_substr='prev')
            if col is not None:
                used_fuzzy.append((key, col))
        resolved[key] = col
    if used_fuzzy:
        print(f"  [WARN] Predicted-config columns resolved via fuzzy fallback (verify these are correct, "
              f"not the oracle-label columns): {used_fuzzy}")
    missing = [k for k, v in resolved.items() if v is None]
    if missing:
        raise ValueError(f"Could not resolve predicted-config column(s): {missing}. Columns available: {columns}")
    return resolved


def normalize_component(val):
    """Normalize one config-dimension value for equality comparison across
    files that may format the same value as '512', '512.0', 512.0, 'None',
    'none', etc."""
    try:
        f = float(val)
        if np.isnan(f):
            return None
        return round(f, 6)
    except (TypeError, ValueError):
        s = str(val).strip().lower()
        return None if s in ('nan', '') else s


SWEEP_PERIOD_CANDIDATES = ['period_start']
SWEEP_PREV_PERIOD_CANDIDATES = ['period_start_val_prev']
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


def parse_period_numeric(period_val):
    """Extract the numeric instruction-count id from a period label.
    Handles 'periodicins-100000002' -> 100000002, bare numbers (int/float/
    numeric string) -> as-is, and non-numeric sentinels ('roi-end',
    'roi-begin', etc.) -> None (unmatchable)."""
    if period_val is None:
        return None
    if isinstance(period_val, (int, float)) and not isinstance(period_val, bool):
        return None if (isinstance(period_val, float) and np.isnan(period_val)) else float(period_val)
    s = str(period_val).strip()
    if s.lower() in ('nan', ''):
        return None
    if '-' in s:
        tail = s.rsplit('-', 1)[-1]
        if tail.replace('.', '', 1).isdigit():
            return float(tail)
        return None  # e.g. 'roi-end', 'roi-begin' -- no numeric id
    if s.replace('.', '', 1).isdigit():
        return float(s)
    return None


def _finite(series):
    s = pd.to_numeric(series, errors='coerce')
    return s[np.isfinite(s)]


# ==============================================================================
# PER-BENCHMARK COMPUTATION
# ==============================================================================

def load_predicted_config_by_period(predictions_csv, benchmark):
    """Load the predictions file and return {period_start: normalized_config_tuple}
    -- the model's actual predicted config per interval -- plus basic counts.
    Filters to `benchmark` if a benchmark column is present and contains
    other values. Does NOT use predictions.csv's own 'PPW_best' column as
    achieved PPW -- that column is the oracle label carried through from
    training, not a measurement of what the predicted config achieved (see
    header note).

    IMPORTANT: this file has one row per *starting configuration* for a
    given interval, so a period_start repeating many times is completely
    normal and is NOT, by itself, worth a warning. What actually matters is
    whether the rows sharing a period_start agree on the model's predicted
    config for that period -- if they don't, that's a genuine inconsistency
    (a real loader/data issue) and is worth flagging; if they do agree,
    there's nothing wrong and picking any one of them is fine."""
    df = pd.read_csv(predictions_csv)
    period_col = resolve_first_present(df.columns, PRED_PERIOD_CANDIDATES)
    if period_col is None:
        raise ValueError(f"No period column found in {predictions_csv}. Columns: {list(df.columns)}")

    bench_col = find_col(df.columns, 'benchmark')
    if bench_col is not None:
        other_values = set(df[bench_col].unique()) - {benchmark}
        if other_values:
            print(f"  [WARN] {predictions_csv} contains rows for other benchmark(s) {other_values} "
                  f"-- filtering to only '{benchmark}'.")
            df = df[df[bench_col] == benchmark]

    cfg_cols = resolve_predicted_config_columns(list(df.columns))

    predicted_by_period = {}
    inconsistent_periods = {}  # period_num -> set of distinct cfg_tuples seen
    n_unparseable_period = 0
    n_incomplete_config = 0
    for _, row in df.iterrows():
        period_num = parse_period_numeric(row[period_col])
        if period_num is None:
            n_unparseable_period += 1
            continue
        cfg_tuple = tuple(normalize_component(row[cfg_cols[k]]) for k in SWEEP_CONFIG_DIMENSIONS)
        if any(c is None for c in cfg_tuple):
            n_incomplete_config += 1
            continue
        if period_num not in predicted_by_period:
            predicted_by_period[period_num] = cfg_tuple
        elif predicted_by_period[period_num] != cfg_tuple:
            inconsistent_periods.setdefault(period_num, {predicted_by_period[period_num]}).add(cfg_tuple)

    if n_unparseable_period:
        print(f"  [WARN] {n_unparseable_period} row(s) in {predictions_csv} had a non-numeric "
              f"period_start (e.g. 'roi-end') and could not be joined -- excluded.")
    if n_incomplete_config:
        print(f"  [WARN] {n_incomplete_config} row(s) in {predictions_csv} had an incomplete "
              f"predicted config and were skipped.")
    if inconsistent_periods:
        print(f"  [WARN] {len(inconsistent_periods)} period(s) in {predictions_csv} had rows (different "
              f"starting configurations) that DISAGREE on the model's predicted config for that period "
              f"-- this is a genuine inconsistency (not just repeated period_start values, which are "
              f"expected). Kept the first-seen config for these periods; investigate predict_config.py's "
              f"output for these periods: {sorted(inconsistent_periods.keys())[:10]}"
              f"{' ...' if len(inconsistent_periods) > 10 else ''}")

    return predicted_by_period, {
        'n_intervals': len(df),
        'n_with_complete_predicted_config': len(predicted_by_period),
    }


def compute_matched_metrics(sweep_csv, predicted_by_period, chunksize=200_000):
    """
    Two-pass streaming scan over the sweep file.

    Pass 1 (oracle): join key is the sweep row's OWN 'period_start' column
    against the prediction period. That row's 'PPW_best' correctly
    describes that exact interval. This is deliberately a different row
    from the one used for achieved PPW below (see header note on the
    checkpoint offset between 'ppw_prev' and 'PPW_best' on the same row).

    Pass 2 (achieved + best-static): join key is 'period_start_val_prev' --
    the numeric field marking the start of the interval that this row's
    '*_prev' columns (ppw_prev, config, etc.) actually describe.
      - If that row's '*_prev' config matches this period's model-predicted
        config, its 'ppw_prev' value IS the achieved PPW for that interval.
      - Every row also feeds the best-static accumulation for its own
        config, restricted to periods with a finite oracle value (already
        fully known by the time pass 2 runs, since pass 1 completed first),
        so all three metrics end up computed over the same period universe
        and achieved/static can never mathematically exceed oracle.

    Doing this as two full passes (rather than one interleaved pass) is
    what makes the result independent of chunk/row order: pass 1 finishes
    resolving every period's oracle value before pass 2 ever needs it, so a
    row read early in pass 2 is never incorrectly dropped just because its
    matching oracle row happened to stream later.
    """
    header = pd.read_csv(sweep_csv, nrows=0)
    columns = list(header.columns)

    period_col = resolve_first_present(columns, SWEEP_PERIOD_CANDIDATES, exclude_substr='prev')
    prev_period_col = resolve_first_present(columns, SWEEP_PREV_PERIOD_CANDIDATES)
    achieved_col = resolve_first_present(columns, SWEEP_ACHIEVED_PPW_CANDIDATES)
    oracle_col = resolve_first_present(columns, SWEEP_ORACLE_PPW_CANDIDATES)
    dim_cols = {}
    for key, cands in SWEEP_CONFIG_DIMENSIONS.items():
        col = resolve_first_present(columns, cands)
        if col is None:
            raise ValueError(f"Could not resolve sweep column for dimension '{key}' in {sweep_csv}. "
                              f"Columns available: {columns}")
        dim_cols[key] = col

    if period_col is None or prev_period_col is None or achieved_col is None or oracle_col is None:
        raise ValueError(
            f"Missing required column(s) in {sweep_csv}. "
            f"period_start={period_col}, period_start_val_prev={prev_period_col}, "
            f"ppw_prev={achieved_col}, oracle PPW_best={oracle_col}"
        )

    oracle_by_period = {}     # period_num -> oracle PPW (finite only)
    achieved_by_period = {}   # period_num -> achieved PPW of the model's predicted config (finite only)
    static_sum = {}
    static_count = {}
    n_rows = 0
    n_static_dropped = 0
    prev_periods_seen = set()

    # ---------------- Pass 1: oracle, keyed off the row's OWN period_start ----------------
    oracle_usecols = list(dict.fromkeys([period_col, oracle_col]))
    for chunk in pd.read_csv(sweep_csv, usecols=oracle_usecols, chunksize=chunksize):
        n_rows += len(chunk)
        period_nums = chunk[period_col].map(parse_period_numeric)
        in_predicted = period_nums.isin(predicted_by_period.keys())
        if not in_predicted.any():
            continue
        sub = chunk.loc[in_predicted]
        sub_periods = period_nums.loc[in_predicted]

        oracle_vals = pd.to_numeric(sub[oracle_col], errors='coerce')
        finite_oracle = np.isfinite(oracle_vals)
        for period, val in zip(sub_periods[finite_oracle], oracle_vals[finite_oracle]):
            if period not in oracle_by_period:
                oracle_by_period[period] = val

    # ------- Pass 2: achieved + best-static, keyed off period_start_val_prev -------
    achieved_usecols = list(dict.fromkeys([prev_period_col, achieved_col] + list(dim_cols.values())))
    n_rows_pass2 = 0
    for chunk in pd.read_csv(sweep_csv, usecols=achieved_usecols, chunksize=chunksize):
        n_rows_pass2 += len(chunk)
        period_nums = pd.to_numeric(chunk[prev_period_col], errors='coerce')
        prev_periods_seen.update(period_nums.dropna().unique())

        in_predicted = period_nums.isin(predicted_by_period.keys())
        if not in_predicted.any():
            continue
        sub = chunk.loc[in_predicted]
        sub_periods = period_nums.loc[in_predicted]

        # oracle_by_period is already fully resolved (pass 1 is complete), so
        # this filter is no longer order-dependent.
        row_has_finite_oracle = sub_periods.isin(oracle_by_period.keys())
        sub2 = sub.loc[row_has_finite_oracle]
        sub2_periods = sub_periods.loc[row_has_finite_oracle]

        config_key = pd.Series(
            list(zip(*[sub2[dim_cols[k]].map(normalize_component) for k in SWEEP_CONFIG_DIMENSIONS])),
            index=sub2.index,
        )
        achieved_vals = pd.to_numeric(sub2[achieved_col], errors='coerce')
        finite_achieved = np.isfinite(achieved_vals)
        n_static_dropped += int((~finite_achieved).sum())

        # --- achieved: rows whose config matches this period's predicted config ---
        is_predicted_config = pd.Series(
            [config_key.iat[i] == predicted_by_period.get(sub2_periods.iat[i])
             for i in range(len(sub2))],
            index=sub2.index,
        )
        match_mask = is_predicted_config & finite_achieved
        for period, val in zip(sub2_periods[match_mask], achieved_vals[match_mask]):
            if period not in achieved_by_period:
                achieved_by_period[period] = val

        # --- best static: every candidate config's own measured PPW ---
        for key, val in zip(config_key[finite_achieved], achieved_vals[finite_achieved]):
            static_sum[key] = static_sum.get(key, 0.0) + val
            static_count[key] = static_count.get(key, 0) + 1

    n_rows = max(n_rows, n_rows_pass2)

    # matched periods = finite oracle AND the predicted config was actually
    # found (with a finite measurement) in the sweep for that period
    matched_periods = [p for p in oracle_by_period if p in achieved_by_period]
    n_matched = len(matched_periods)

    periods_predicted_not_in_sweep = set(predicted_by_period.keys()) - prev_periods_seen
    if periods_predicted_not_in_sweep:
        print(f"  [WARN] {len(periods_predicted_not_in_sweep)} period(s) from the predictions file "
              f"were not found at all in the sweep file via period_start_val_prev (format mismatch?).")

    periods_config_not_found = (set(predicted_by_period.keys()) & set(oracle_by_period.keys())) - set(achieved_by_period.keys())
    if periods_config_not_found:
        print(f"  [WARN] {len(periods_config_not_found)} period(s) had a finite oracle value but the "
              f"model's predicted config was not found (or non-finite) among that period's swept rows "
              f"-- excluded from the achieved/oracle comparison.")

    if n_matched == 0:
        oracle_mean = np.nan
        achieved_mean_matched = np.nan
    else:
        oracle_mean = float(np.mean([oracle_by_period[p] for p in matched_periods]))
        achieved_mean_matched = float(np.mean([achieved_by_period[p] for p in matched_periods]))

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
        'achieved_ppw_mean': achieved_mean_matched,
        'oracle_ppw_mean': oracle_mean,
        'best_static_ppw_mean': best_static_mean,
        'best_static_config': best_static_config,
        'n_matched_periods': n_matched,
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

    for offset, series in zip([-width, 0, width], [oracle_pct, achieved_pct, static_pct]):
        for xi, v in zip(x, series):
            ax.text(xi + offset, v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=10)
    ax.set_ylabel("PPW as % of oracle", fontsize=10)
    ax.set_title("Achieved vs. Oracle vs. Best-Static-Config PPW", fontsize=12, fontweight='bold', pad=40)
    ax.set_ylim(0, max(max(achieved_pct, default=100), max(static_pct, default=100), 100) * 1.15)
    ax.axhline(100, color='#55A868', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=9, frameon=False)
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
        predicted_by_period, pred_meta = load_predicted_config_by_period(pred_path, bench)
        print(f"  Predictions: {pred_meta['n_with_complete_predicted_config']} distinct period(s) with a "
              f"complete predicted config, from {pred_meta['n_intervals']} row(s) total")

        metrics = compute_matched_metrics(sweep_path, predicted_by_period, chunksize=args.chunksize)
        print(f"  Matched periods (predicted config found + finite oracle in sweep): {metrics['n_matched_periods']}")
        print(f"  Achieved (model): {metrics['achieved_ppw_mean']:.4e}")
        print(f"  Oracle:           {metrics['oracle_ppw_mean']:.4e}")
        print(f"  Best static:      {metrics['best_static_ppw_mean']:.4e} "
              f"across {metrics['n_candidate_configs']} candidate configs -> "
              f"{format_config(metrics['best_static_config'])}")

        oracle_mean = metrics['oracle_ppw_mean']
        row = {
            'benchmark': bench,
            'achieved_ppw_mean': metrics['achieved_ppw_mean'],
            'oracle_ppw_mean': oracle_mean,
            'best_static_ppw_mean': metrics['best_static_ppw_mean'],
            'best_static_config': format_config(metrics['best_static_config']),
            'achieved_pct_of_oracle': metrics['achieved_ppw_mean'] / oracle_mean * 100 if oracle_mean else np.nan,
            'best_static_pct_of_oracle': metrics['best_static_ppw_mean'] / oracle_mean * 100 if oracle_mean else np.nan,
            'achieved_gain_over_static_pct': (
                (metrics['achieved_ppw_mean'] - metrics['best_static_ppw_mean'])
                / metrics['best_static_ppw_mean'] * 100
            ) if metrics['best_static_ppw_mean'] else np.nan,
            'n_matched_periods': metrics['n_matched_periods'],
            'n_candidate_configs': metrics['n_candidate_configs'],
            'n_intervals_predicted': pred_meta['n_intervals'],
            'n_dropped_nonfinite_sweep': metrics['sweep_n_dropped_nonfinite'],
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
