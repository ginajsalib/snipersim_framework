"""
Config Stability Analysis
=========================
Measures how often the optimal configuration stays the same vs. changes
between consecutive periods, for each target individually and jointly.

Since the CSV does not contain a direct "prev best config" column, this
script reconstructs it by:
  1. Sorting each benchmark's rows by period_start (or row order).
  2. Shifting the best-config columns by one row (within benchmark) to get
     the previous interval's best config.
  3. Computing stability from that reconstructed "prev best" vs current best.

Targets analysed:
  btbCore0, btbCore1, prefetcher_core0, prefetcher_core1, L2, L3

Outputs
-------
  stability_analysis/
    stability_overview.png
    stability_by_benchmark.png
    stability_heatmap.png
    transition_<target>.png    (one per target)
    stability_summary.csv
    stability_by_benchmark.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION — adjust to match your actual column names
# ==============================================================================

CSV_FILES = {
    'barnes':    '/home/gina/Desktop/snipersim_framework/pythonScripts/finalTrainingData/barnes_train_with_top3.csv',
    'fft':       '/home/gina/Desktop/snipersim_framework/pythonScripts/finalTrainingData/fft_train_with_top3.csv',
    'cholesky':  '/home/gina/Desktop/snipersim_framework/pythonScripts/finalTrainingData/cholesky_train_with_top3.csv',
    'radiosity': '/home/gina/Desktop/snipersim_framework/pythonScripts/finalTrainingData/radiosity_train_with_top3.csv',
}

# Column used to sort rows chronologically within each benchmark.
# Common candidates: 'period_start', 'period_start_val', 'interval_id'
# If the column is not found the script falls back to CSV row order.
PERIOD_COL = 'period_start'

# Best-config columns (what we predict).
# Update the dict values to match your CSV column names exactly.
BEST_COLS = {
    'btbCore0':      'btbCore0_best',
    'btbCore1':      'btbCore1_best',
    'prefetcher_c0': 'prefetcher_core0_best',
    'prefetcher_c1': 'prefetcher_core1_best',
    'L2':            'L2_best',
    'L3':            'L3_best',
}

OUTPUT_DIR = 'stability_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# HELPER — reconstruct prev-best via within-benchmark shift
# ==============================================================================

def reconstruct_prev_best(df, period_col, best_cols, benchmark_col='benchmark'):
    """
    Sort rows by (benchmark, period_col), then for each benchmark group
    shift each best-config column by 1 to create *_prev_inferred columns.
    The first interval in each benchmark gets NaN (no predecessor).

    Returns (modified_df, prev_col_map) where prev_col_map maps
    target_name -> new prev-column name.
    """
    df = df.copy()
    prev_col_map = {}

    # Pre-create output columns
    for name, col in best_cols.items():
        if col not in df.columns:
            continue
        prev_col = f'{col}_prev_inferred'
        prev_col_map[name] = prev_col
        df[prev_col] = np.nan

    df = df.sort_values([benchmark_col, period_col]).reset_index(drop=True)

    for _bench, group_idx in df.groupby(benchmark_col).groups.items():
        group = df.loc[group_idx]
        for name, col in best_cols.items():
            if col not in df.columns:
                continue
            prev_col = f'{col}_prev_inferred'
            df.loc[group_idx, prev_col] = group[col].shift(1).values

    return df, prev_col_map


# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print("CONFIG STABILITY ANALYSIS  (prev-best inferred from consecutive intervals)")
print("=" * 70)

dfs = []
for bench, path in CSV_FILES.items():
    try:
        tmp = pd.read_csv(path)
        tmp['benchmark'] = bench
        dfs.append(tmp)
        print(f"Loaded {bench:12s}: {tmp.shape[0]:5d} rows, {tmp.shape[1]:4d} cols")
    except FileNotFoundError:
        print(f"WARNING: File not found — {bench}: {path}")

if not dfs:
    print("ERROR: No data loaded. Check your CSV paths.")
    exit()

df_raw = pd.concat(dfs, ignore_index=True)
print(f"\nCombined dataset: {df_raw.shape[0]} rows")

# Resolve PERIOD_COL — fall back to row order if not present
if PERIOD_COL not in df_raw.columns:
    candidates = [c for c in df_raw.columns
                  if any(kw in c.lower() for kw in ('period', 'start', 'interval', 'time'))]
    print(f"\nWARNING: '{PERIOD_COL}' not found in columns.")
    print(f"  Columns that might be the sort key: {candidates}")
    print("  Falling back to positional ordering (CSV row order within each benchmark).")
    df_raw['_row_order'] = df_raw.groupby('benchmark').cumcount()
    PERIOD_COL = '_row_order'

# Filter to best-config columns that actually exist
present_best_cols = {n: c for n, c in BEST_COLS.items() if c in df_raw.columns}
missing_best_cols = {n: c for n, c in BEST_COLS.items() if c not in df_raw.columns}

if missing_best_cols:
    print(f"\nWARNING: These best-config columns were not found in the data:")
    for name, col in missing_best_cols.items():
        suggestions = [c for c in df_raw.columns
                       if name.lower().replace('_', '') in c.lower().replace('_', '')]
        print(f"  '{col}'  (target='{name}')  — possible matches: {suggestions}")

print(f"\nTargets available for stability analysis: {list(present_best_cols.keys())}")

if not present_best_cols:
    print("ERROR: No valid best-config columns found. Update BEST_COLS in the config block.")
    exit()


# ==============================================================================
# 2. RECONSTRUCT PREVIOUS-INTERVAL BEST CONFIGS
# ==============================================================================
print(f"\nReconstructing previous best configs "
      f"(sorting by '{PERIOD_COL}' within each benchmark)...")

df, prev_col_map = reconstruct_prev_best(df_raw, PERIOD_COL, present_best_cols)

first_prev_col = list(prev_col_map.values())[0]
n_with_prev  = df[first_prev_col].notna().sum()
n_first_rows = df[first_prev_col].isna().sum()
print(f"  Rows with a valid previous interval:  {n_with_prev}  ({n_with_prev/len(df)*100:.1f}%)")
print(f"  First-interval rows (no predecessor): {n_first_rows}  — excluded from stability counts")

# Quick sanity-check: show the first 6 rows of the first benchmark so the
# user can verify the shift looks right (current best == prev of next row)
sample_cols = (
    ['benchmark', PERIOD_COL] +
    list(present_best_cols.values())[:2] +
    [prev_col_map[n] for n in list(present_best_cols.keys())[:2] if n in prev_col_map]
)
first_bench = df['benchmark'].iloc[0]
print(f"\nSanity check — first 6 rows of '{first_bench}':")
print(f"(Column <x>_prev_inferred should equal the <x>_best of the row above it)")
print(df[df['benchmark'] == first_bench][sample_cols].head(6).to_string(index=False))


# ==============================================================================
# 3. COMPUTE STABILITY PER TARGET
# ==============================================================================
print("\n" + "=" * 70)
print("PER-TARGET STABILITY")
print("=" * 70)

stability_results = {}

for name, best_col in present_best_cols.items():
    prev_col = prev_col_map.get(name)
    if prev_col is None:
        continue

    best = df[best_col].astype(str)
    prev = df[prev_col].astype(str)
    valid_mask = df[best_col].notna() & df[prev_col].notna()
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        print(f"\n{name}: No valid rows — skipping")
        continue

    n_same    = int((best[valid_mask] == prev[valid_mask]).sum())
    n_changed = n_valid - n_same
    rate      = n_same / n_valid

    stability_results[name] = dict(
        best_col=best_col, prev_col=prev_col,
        n_valid=n_valid, n_same=n_same, n_changed=n_changed, stability=rate,
    )

    print(f"\n{name}  [{best_col}]")
    print(f"  Valid rows (has predecessor):    {n_valid}")
    print(f"  Unchanged from previous best:    {n_same:5d}  ({rate*100:.1f}%)")
    print(f"  Changed from previous best:      {n_changed:5d}  ({(1-rate)*100:.1f}%)")

if not stability_results:
    print("ERROR: Could not compute stability for any target.")
    exit()


# ==============================================================================
# 4. JOINT STABILITY
# ==============================================================================
print("\n" + "=" * 70)
print("JOINT STABILITY  (all targets unchanged simultaneously)")
print("=" * 70)

change_flags = pd.DataFrame(index=df.index)
for name, res in stability_results.items():
    best = df[res['best_col']].astype(str)
    prev = df[res['prev_col']].astype(str)
    both_valid = df[res['best_col']].notna() & df[res['prev_col']].notna()
    # True = "changed" (or missing predecessor)
    change_flags[name] = (~both_valid) | (best != prev)

change_flags['n_changed_targets'] = change_flags[list(stability_results.keys())].sum(axis=1)
change_flags['any_changed'] = change_flags['n_changed_targets'] > 0
change_flags['all_same']    = change_flags['n_changed_targets'] == 0
change_flags['benchmark']   = df['benchmark'].values
change_flags['has_prev']    = df[first_prev_col].notna().values

has_prev_mask   = change_flags['has_prev']
n_valid_joint   = int(has_prev_mask.sum())
n_all_same      = int(change_flags.loc[has_prev_mask, 'all_same'].sum())
n_any_changed   = int(change_flags.loc[has_prev_mask, 'any_changed'].sum())
joint_stability = n_all_same / n_valid_joint if n_valid_joint > 0 else 0.0

print(f"\n  Evaluated on {n_valid_joint} rows that have a previous interval\n")
print(f"  All {len(stability_results)} targets unchanged:  "
      f"{n_all_same}/{n_valid_joint}  ({joint_stability*100:.1f}%)")
print(f"  At least 1 target changed:        "
      f"{n_any_changed}/{n_valid_joint}  ({(1-joint_stability)*100:.1f}%)")

dist = change_flags.loc[has_prev_mask, 'n_changed_targets'].value_counts().sort_index()
print(f"\n  Distribution — how many targets changed per period:")
for n_ch, count in dist.items():
    bar = '█' * int(count / n_valid_joint * 40)
    print(f"    {int(n_ch)} target(s) changed: {count:5d} rows  "
          f"({count/n_valid_joint*100:5.1f}%)  {bar}")


# ==============================================================================
# 5. PER-BENCHMARK STABILITY
# ==============================================================================
print("\n" + "=" * 70)
print("STABILITY BY BENCHMARK")
print("=" * 70)

bench_rows = change_flags[has_prev_mask]
bench_stability = bench_rows.groupby('benchmark').agg(
    total=('all_same', 'count'),
    all_same=('all_same', 'sum'),
    any_changed=('any_changed', 'sum'),
).assign(joint_stability_pct=lambda x: x['all_same'] / x['total'] * 100)

for name in stability_results:
    bench_stability[f'{name}_pct'] = (
        bench_rows.groupby('benchmark')[name]
        .apply(lambda s: (~s).mean() * 100)   # ~changed means "stayed same"
    )

display_cols = (['total', 'all_same', 'joint_stability_pct'] +
                [f'{n}_pct' for n in stability_results])
print(bench_stability[display_cols].round(1).to_string())


# ==============================================================================
# 6. TRANSITION ANALYSIS
# ==============================================================================
print("\n" + "=" * 70)
print("TRANSITION ANALYSIS  (when a target DID change, which values did it move between?)")
print("=" * 70)

for name, res in stability_results.items():
    best = df.loc[has_prev_mask, res['best_col']].astype(str)
    prev = df.loc[has_prev_mask, res['prev_col']].astype(str)
    changed = best != prev
    n_changed = int(changed.sum())

    if n_changed == 0:
        print(f"\n{name}: Never changed — perfectly stable.")
        continue

    top5 = Counter(zip(prev[changed], best[changed])).most_common(5)
    print(f"\n{name}  ({n_changed} changes out of {n_valid_joint} periods):")
    for (frm, to), cnt in top5:
        print(f"  {frm:>14} → {to:<14}  {cnt:5d}×  ({cnt/n_changed*100:.1f}% of changes)")


# ==============================================================================
# 7. GATING MODEL RECOMMENDATION
# ==============================================================================
print("\n" + "=" * 70)
print("GATING MODEL RECOMMENDATION")
print("=" * 70)

print(f"\n  Joint stability (all targets same): {joint_stability*100:.1f}%")
print(f"  Change rate    (any target changes): {(1-joint_stability)*100:.1f}%")

if joint_stability >= 0.75:
    verdict = "STRONGLY RECOMMENDED"
    reason  = (f"Configs are stable {joint_stability*100:.0f}% of the time. A gating model "
               "will skip RF prediction for the vast majority of periods, "
               "avoiding unnecessary switches and overhead.")
elif joint_stability >= 0.50:
    verdict = "RECOMMENDED"
    reason  = (f"Configs are stable {joint_stability*100:.0f}% of the time. "
               "Roughly half of periods need no new prediction — a clear win.")
elif joint_stability >= 0.30:
    verdict = "CONSIDER WITH CAUTION"
    reason  = (f"Configs change {(1-joint_stability)*100:.0f}% of the time. "
               "A gating model adds complexity; profile reconfiguration cost first.")
else:
    verdict = "LOW BENEFIT"
    reason  = (f"Configs are highly volatile ({(1-joint_stability)*100:.0f}% change rate). "
               "A gating model would rarely fire. Focus on improving RF accuracy.")

print(f"\n  Verdict: {verdict}")
print(f"  Reason:  {reason}")

print(f"\n  Change contribution per target (how often each target alone changes):")
for name, res in stability_results.items():
    pct = (1 - res['stability']) * 100
    bar = '█' * int(pct / 2)
    print(f"    {name:24s}: {pct:5.1f}%  {bar}")


# ==============================================================================
# 8. PLOTS
# ==============================================================================
print(f"\nGenerating plots → {OUTPUT_DIR}/")

# ── Plot 1: Per-target stability + change-count distribution ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    'Configuration Stability Analysis\n'
    '(prev-best inferred from consecutive intervals)',
    fontsize=13, fontweight='bold')

tnames      = list(stability_results.keys())
stabilities = [stability_results[n]['stability'] * 100 for n in tnames]
changes     = [100 - s for s in stabilities]
x, w        = np.arange(len(tnames)), 0.35

axes[0].bar(x - w/2, stabilities, w, label='Stable (same)', color='steelblue', alpha=0.85)
axes[0].bar(x + w/2, changes,     w, label='Changed',       color='tomato',    alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(tnames, rotation=20, ha='right')
axes[0].set_ylabel('% of periods')
axes[0].set_title('Per-target stability')
axes[0].set_ylim(0, 115)
axes[0].axhline(50, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
axes[0].legend()
for i, s in enumerate(stabilities):
    axes[0].text(i - w/2, s + 1.5, f'{s:.1f}%', ha='center', va='bottom', fontsize=8)

colors = ['steelblue' if int(i) == 0 else 'tomato' for i in dist.index]
axes[1].bar([str(int(i)) for i in dist.index],
            dist.values / n_valid_joint * 100,
            color=colors, alpha=0.85, edgecolor='white')
axes[1].set_xlabel('Number of targets that changed')
axes[1].set_ylabel('% of periods')
axes[1].set_title('How many targets change per period')
for i, (idx, val) in enumerate(zip(dist.index, dist.values)):
    axes[1].text(i, val/n_valid_joint*100 + 0.5,
                 f'{val/n_valid_joint*100:.1f}%', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/stability_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: stability_overview.png")

# ── Plot 2: Joint stability by benchmark ─────────────────────────────────────
benchmarks = bench_stability.index.tolist()
jvals      = bench_stability['joint_stability_pct'].values
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(benchmarks, jvals, color='steelblue', alpha=0.85, edgecolor='white')
ax.axhline(75, color='green',  linestyle='--', linewidth=1,
           label='75% — strongly recommend gating')
ax.axhline(50, color='orange', linestyle='--', linewidth=1,
           label='50% — recommend gating')
ax.set_ylabel('Joint stability (%)\n(all targets unchanged)')
ax.set_title('Joint Configuration Stability by Benchmark')
ax.set_ylim(0, 110)
ax.legend(fontsize=9)
for bar, val in zip(bars, jvals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1,
            f'{val:.1f}%', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/stability_by_benchmark.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: stability_by_benchmark.png")

# ── Plot 3: Per-target × benchmark heatmap ───────────────────────────────────
hmap_data = pd.DataFrame({n: bench_stability[f'{n}_pct'] for n in stability_results})
fig, ax = plt.subplots(figsize=(max(7, len(stability_results) * 1.5),
                                 max(4, len(benchmarks) * 0.9)))
sns.heatmap(hmap_data, annot=True, fmt='.1f', cmap='RdYlGn',
            vmin=0, vmax=100, linewidths=0.5, ax=ax,
            cbar_kws={'label': 'Stability (%)'})
ax.set_title('Per-target Stability by Benchmark (%)')
ax.set_xlabel('Target')
ax.set_ylabel('Benchmark')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/stability_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: stability_heatmap.png")

# ── Plot 4: Transition heatmaps ──────────────────────────────────────────────
for name, res in stability_results.items():
    best = df.loc[has_prev_mask, res['best_col']].astype(str)
    prev = df.loc[has_prev_mask, res['prev_col']].astype(str)
    tmat = pd.crosstab(
        prev.rename('Previous config'),
        best.rename('Next best config'),
        normalize='index'
    ) * 100
    if tmat.shape[0] <= 1:
        continue
    fig, ax = plt.subplots(figsize=(max(5, tmat.shape[1] * 1.2),
                                     max(4, tmat.shape[0] * 0.9)))
    sns.heatmap(tmat, annot=True, fmt='.1f', cmap='Blues',
                vmin=0, vmax=100, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Row-normalised %'})
    ax.set_title(f'Transition matrix — {name}\n'
                 f'(row = previous best, col = next best;  diagonal = stayed same)')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/transition_{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: transition_{name}.png")


# ==============================================================================
# 9. SAVE SUMMARY CSVs
# ==============================================================================
summary_rows = [{
    'target':            name,
    'best_col':          res['best_col'],
    'inferred_prev_col': res['prev_col'],
    'n_valid':           res['n_valid'],
    'n_same':            res['n_same'],
    'n_changed':         res['n_changed'],
    'stability_pct':     round(res['stability'] * 100, 2),
    'change_rate_pct':   round((1 - res['stability']) * 100, 2),
} for name, res in stability_results.items()]

pd.DataFrame(summary_rows).to_csv(f'{OUTPUT_DIR}/stability_summary.csv', index=False)
bench_stability.to_csv(f'{OUTPUT_DIR}/stability_by_benchmark.csv')
print(f"\n  Saved: stability_summary.csv")
print(f"  Saved: stability_by_benchmark.csv")


# ==============================================================================
# FINAL SUMMARY TABLE
# ==============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"\n{'Target':<24} {'Valid':>8}  {'Stable':>8}  {'Changed':>8}  {'Stability':>10}")
print("-" * 64)
for name, res in stability_results.items():
    print(f"{name:<24} {res['n_valid']:>8}  {res['n_same']:>8}  "
          f"{res['n_changed']:>8}  {res['stability']*100:>9.1f}%")

print(f"\n{'Joint (all same)':<24} {n_valid_joint:>8}  {n_all_same:>8}  "
      f"{n_any_changed:>8}  {joint_stability*100:>9.1f}%")

print(f"\nGating model verdict: {verdict}")
print(f"Outputs saved to:     ./{OUTPUT_DIR}/")
