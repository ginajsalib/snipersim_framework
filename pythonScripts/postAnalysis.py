""""
post_hoc_analysis.py  (vectorised rewrite)
===========================================
Standalone analysis script that loads a previously trained Random Forest model
and runs the following analyses on the full dataset:

  1. PPW % LOSS ANALYSIS          — how much PPW do we lose vs optimal?
  2. STATIC BASELINE COMPARISON   — vs best-static, worst-static, optimal
  3. BEST vs 2ND-BEST GAP         — how much headroom is there to be wrong?
  4. NO-CHANGE BASELINE           — if prev config kept, is it still top-3?

All per-row logic is fully vectorised with numpy/pandas — no Python loops
over rows, so 300k samples runs in seconds not hours.

Usage:
    python post_hoc_analysis.py
    python post_hoc_analysis.py --model-dir saved_models/
    python post_hoc_analysis.py --model-dir saved_models/ --model-timestamp 20240101_120000
"""

import os
import sys
import glob
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description='Post-hoc RF model analysis')
parser.add_argument('--model-dir', type=str, default='saved_models/')
parser.add_argument('--model-timestamp', type=str, default=None,






                    help='Specific timestamp to load (e.g. 20240101_120000). '
                         'If omitted, loads the most recent model.')
parser.add_argument('--output-dir', type=str, default='analysis_reports/',
                    help='Directory to save the text report')
parser.add_argument('--cost-csv', type=str, default=None,
                    help='Optional CSV from costAnalysis.py with inference/reconfig costs')

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA FILES
# ══════════════════════════════════════════════════════════════════════════════
DATA_FILES = {
    'barnes':    '/home/gina/Desktop/snipersim_framework/pythonScripts/finalTrainingData/barnes_train_with_top3_fixed.csv',
    'cholesky':  '/home/gina/Desktop/snipersim_framework/pythonScripts/finalTrainingData/cholesky_train_with_top3_fixed.csv',
    'radiosity': '/home/gina/Desktop/snipersim_framework/pythonScripts/finalTrainingData/radiosity_train_with_top3_fixed.csv',
    # fft excluded — all PPW values are NaN
}

TARGET_COLUMNS = [
    'btbCore0_best', 'btbCore1_best',
    'prefetcher_core0_best', 'prefetcher_core1_best'
]

ALL_CONFIG_COLUMNS = [
    'btbCore0_best', 'btbCore1_best',
    'prefetcher_core0_best', 'prefetcher_core1_best', 'PPW_best',
    'btbCore0_2nd', 'btbCore1_2nd',
    'prefetcher_core0_2nd', 'prefetcher_core1_2nd', 'PPW_2nd', 'Diff_best_2nd',
    'btbCore0_3rd', 'btbCore1_3rd',
    'prefetcher_core0_3rd', 'prefetcher_core1_3rd', 'PPW_3rd', 'Diff_best_3rd',
]

METADATA_COLUMNS_TO_DROP = [
    'best-config', 'file', 'file_prev', 'period_start',
    'period_end', 'period_start_prev', 'period_end_prev',
    'directory_perf_prev', 'leaf_dir_prev', 'directory_power_prev',
    'leaf_dir_perf_prev', 'leaf_dir_power_prev', 'period_start_val_prev',
    'period_end_val_perf_prev', 'period_start_val_perf_prev',
    'period_start_val_power_prev', 'period_end_val_power_prev'
]

PREV_BTB0_COL = 'BTB core 0_prev'
PREV_BTB1_COL = 'BTB core 1_prev'
PREV_PF0_COL  = 'prefetcher_core0_prev'
PREV_PF1_COL  = 'prefetcher_core1_prev'

RANDOM_STATE = 42

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
report_lines = []

def out(line=''):
    print(line)
    report_lines.append(str(line))

def section(title):
    out(); out('═' * 65); out(f'  {title}'); out('═' * 65)

def subsection(title):
    out(); out(f'  ── {title}'); out('  ' + '─' * 55)

def pct_str(n, total):
    return f'{int(n):>6,}  ({n/total*100:5.1f}%)'

def stats_str(series, unit='%'):
    s = series.dropna()
    if len(s) == 0:
        return '  no data'
    return (f'  mean={s.mean():.2f}{unit}  median={s.median():.2f}{unit}'
            f'  p90={np.percentile(s, 90):.2f}{unit}  max={s.max():.2f}{unit}')

def loss_buckets(series, n_total, label=''):
    s = series.dropna()
    out(f'    {label}Loss distribution (% of {n_total:,} total samples):')
    boundaries = [
        (None, 0,   '0%  (exact match)     '),
        (0,    1,   '0-1%                  '),
        (1,    5,   '1-5%                  '),
        (5,    10,  '5-10%                 '),
        (10,   20,  '10-20%                '),
        (20,   None,'>20%                  '),
    ]
    for lo, hi, lbl in boundaries:
        if lo is None:
            cnt = (s == 0.0).sum()
        elif hi is None:
            cnt = (s > lo).sum()
        else:
            cnt = ((s > lo) & (s <= hi)).sum()
        out(f'      {lbl}  {cnt:>6,}  ({cnt/n_total*100:5.1f}%)')


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
section('LOADING SAVED MODEL')

model_dir = args.model_dir
if args.model_timestamp:
    ts = args.model_timestamp
else:
    pkls = glob.glob(os.path.join(model_dir, 'rf_4way_config_predictor_*.pkl'))
    pkls = [p for p in pkls if not any(x in p for x in
                                        ['_scaler', '_encoder', '_prefetcher'])]
    if not pkls:
        out(f'ERROR: No model files found in {model_dir}'); sys.exit(1)
    ts = os.path.basename(sorted(pkls)[-1]) \
           .replace('rf_4way_config_predictor_', '').replace('.pkl', '')

def mpath(suffix=''):
    return os.path.join(model_dir, f'rf_4way_config_predictor_{ts}{suffix}.pkl')

for p in [mpath(), mpath('_scaler'),
          mpath('_prefetcher_core0_encoder'),
          mpath('_prefetcher_core1_encoder')]:
    if not os.path.exists(p):
        out(f'ERROR: File not found: {p}'); sys.exit(1)

model   = joblib.load(mpath())
scaler  = joblib.load(mpath('_scaler'))
enc_pf0 = joblib.load(mpath('_prefetcher_core0_encoder'))
enc_pf1 = joblib.load(mpath('_prefetcher_core1_encoder'))

out(f'  Timestamp:    {ts}')
out(f'  n_estimators: {model.n_estimators}  '
    f'max_depth: {model.max_depth}  '
    f'max_features: {model.max_features}')


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
section('LOADING & PREPROCESSING DATA')

frames = []
for bench, path in DATA_FILES.items():
    if not os.path.exists(path):
        out(f'  [WARN] File not found, skipping: {path}'); continue
    df_b = pd.read_csv(path, low_memory=False)
    df_b['benchmark'] = bench
    frames.append(df_b)
    out(f'  Loaded {bench}: {df_b.shape[0]:,} rows')

df_full = pd.concat(frames, ignore_index=True)
out(f'  Combined: {df_full.shape[0]:,} rows')

prev_config_available = all(c in df_full.columns
                             for c in [PREV_BTB0_COL, PREV_BTB1_COL,
                                       PREV_PF0_COL,  PREV_PF1_COL])
if not prev_config_available:
    out('  [WARN] Previous-config columns not all found — Analysis 4 will be skipped.')

# preserve columns needed for analysis BEFORE any dropping
top3_raw = df_full[ALL_CONFIG_COLUMNS + ['benchmark']].copy()

if prev_config_available:
    prev_raw = df_full[[PREV_BTB0_COL, PREV_BTB1_COL,
                         PREV_PF0_COL, PREV_PF1_COL]].copy()

# build feature matrix
df    = df_full.drop(columns=METADATA_COLUMNS_TO_DROP, errors='ignore')
X_raw = df.drop(ALL_CONFIG_COLUMNS, axis=1)
y_raw = df[TARGET_COLUMNS].copy()

benchmark_col = X_raw['benchmark'].copy() if 'benchmark' in X_raw.columns else None

# encode categoricals
for col in X_raw.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_raw[col] = le.fit_transform(X_raw[col].astype(str))

# encode targets using saved encoders
def safe_encode_series(encoder, series):
    valid = set(encoder.classes_)
    s = series.fillna('none').astype(str).str.strip().str.lower()
    s = s.apply(lambda x: x if x in valid else 'none')
    return encoder.transform(s).astype(int)

for col in ['btbCore0_best', 'btbCore1_best']:
    y_raw[col] = pd.to_numeric(y_raw[col], errors='coerce')
y_raw['prefetcher_core0_best'] = safe_encode_series(enc_pf0, y_raw['prefetcher_core0_best'])
y_raw['prefetcher_core1_best'] = safe_encode_series(enc_pf1, y_raw['prefetcher_core1_best'])

X_num = X_raw.select_dtypes(include=[np.number])

mask_valid  = y_raw.notna().all(axis=1).values
imputer     = SimpleImputer(strategy='mean')
X_scaled    = imputer.fit_transform(X_num)
X_scaled    = scaler.transform(X_scaled)
X_scaled    = X_scaled[mask_valid]

y_valid     = y_raw[mask_valid].reset_index(drop=True)
top3_valid  = top3_raw[mask_valid].reset_index(drop=True)
bench_valid = benchmark_col[mask_valid].reset_index(drop=True) \
              if benchmark_col is not None else None
if prev_config_available:
    prev_valid = prev_raw[mask_valid].reset_index(drop=True)

out(f'  Valid samples after NaN drop: {len(y_valid):,}')


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PREDICT
# ══════════════════════════════════════════════════════════════════════════════
section('RUNNING PREDICTIONS')
t0     = datetime.now()
y_pred = model.predict(X_scaled)
out(f'  Predicted {len(y_pred):,} samples in '
    f'{(datetime.now()-t0).total_seconds():.1f}s')


# ══════════════════════════════════════════════════════════════════════════════
# 4.  VECTORISED RESULTS TABLE  (no per-row Python loops)
# ══════════════════════════════════════════════════════════════════════════════
section('BUILDING RESULTS TABLE  (vectorised)')

# encode 2nd and 3rd prefetcher string columns -> int once, up front
def encode_pf_col(encoder, series):
    valid = set(encoder.classes_)
    s = series.fillna('none').astype(str).str.strip().str.lower()
    s = s.apply(lambda x: x if x in valid else 'none')
    return encoder.transform(s).astype(int)

for suffix in ['2nd', '3rd']:
    top3_valid[f'pf0_{suffix}_enc'] = encode_pf_col(enc_pf0, top3_valid[f'prefetcher_core0_{suffix}'])
    top3_valid[f'pf1_{suffix}_enc'] = encode_pf_col(enc_pf1, top3_valid[f'prefetcher_core1_{suffix}'])

# numpy arrays for all columns (fast element-wise ops)
pred_btb0 = y_pred[:, 0].astype(float)
pred_btb1 = y_pred[:, 1].astype(float)
pred_pf0  = y_pred[:, 2].astype(int)
pred_pf1  = y_pred[:, 3].astype(int)

# ground truth — best from y_valid (already encoded, same path as pred)
true_btb0 = y_valid['btbCore0_best'].values.astype(float)
true_btb1 = y_valid['btbCore1_best'].values.astype(float)
true_pf0  = y_valid['prefetcher_core0_best'].values.astype(int)
true_pf1  = y_valid['prefetcher_core1_best'].values.astype(int)

# 2nd / 3rd from top3_valid
btb0_2nd = pd.to_numeric(top3_valid['btbCore0_2nd'], errors='coerce').values.astype(float)
btb1_2nd = pd.to_numeric(top3_valid['btbCore1_2nd'], errors='coerce').values.astype(float)
pf0_2nd  = top3_valid['pf0_2nd_enc'].values
pf1_2nd  = top3_valid['pf1_2nd_enc'].values

btb0_3rd = pd.to_numeric(top3_valid['btbCore0_3rd'], errors='coerce').values.astype(float)
btb1_3rd = pd.to_numeric(top3_valid['btbCore1_3rd'], errors='coerce').values.astype(float)
pf0_3rd  = top3_valid['pf0_3rd_enc'].values
pf1_3rd  = top3_valid['pf1_3rd_enc'].values

ppw_best = pd.to_numeric(top3_valid['PPW_best'], errors='coerce').values
ppw_2nd  = pd.to_numeric(top3_valid['PPW_2nd'],  errors='coerce').values
ppw_3rd  = pd.to_numeric(top3_valid['PPW_3rd'],  errors='coerce').values

# match flags — all vectorised
match_best = ((pred_btb0 == true_btb0) & (pred_btb1 == true_btb1) &
              (pred_pf0  == true_pf0)  & (pred_pf1  == true_pf1))
match_2nd  = ((pred_btb0 == btb0_2nd)  & (pred_btb1 == btb1_2nd) &
              (pred_pf0  == pf0_2nd)   & (pred_pf1  == pf1_2nd))
match_3rd  = ((pred_btb0 == btb0_3rd)  & (pred_btb1 == btb1_3rd) &
              (pred_pf0  == pf0_3rd)   & (pred_pf1  == pf1_3rd))
match_top3 = match_best | match_2nd | match_3rd

out(f'  Exact match:  {match_best.sum():,}  ({match_best.mean()*100:.2f}%)')
out(f'  Top-3 match:  {match_top3.sum():,}  ({match_top3.mean()*100:.2f}%)')
assert match_top3.sum() >= match_best.sum(), 'BUG: top3 < exact'

numpy_exact = (y_valid.values == y_pred).all(axis=1).sum()
out(f'  Numpy exact check: {numpy_exact:,}  '
    f'{"[OK]" if numpy_exact == match_best.sum() else "[WARN mismatch]"}')

# PPW % loss — vectorised
has_ppw = ~np.isnan(ppw_best) & (ppw_best != 0)

# predicted PPW = whichever rank matched, else 3rd-best as conservative estimate
pred_ppw = np.where(match_best, ppw_best,
           np.where(match_2nd,  ppw_2nd,
           np.where(match_3rd,  ppw_3rd,
                                ppw_3rd)))

pred_ppw_loss_pct = np.where(has_ppw,
                              (ppw_best - pred_ppw) / ppw_best * 100,
                              np.nan)
pred_ppw_loss_pct = np.where(match_best, 0.0, pred_ppw_loss_pct)

# gap best vs 2nd
has_ppw_2nd      = ~np.isnan(ppw_2nd) & has_ppw
gap_best_2nd_pct = np.where(has_ppw_2nd,
                              (ppw_best - ppw_2nd) / ppw_best * 100,
                              np.nan)

# no-change baseline
if prev_config_available:
    prev_btb0_arr = pd.to_numeric(prev_valid[PREV_BTB0_COL], errors='coerce').values.astype(float)
    prev_btb1_arr = pd.to_numeric(prev_valid[PREV_BTB1_COL], errors='coerce').values.astype(float)
    prev_pf0_arr  = encode_pf_col(enc_pf0, prev_valid[PREV_PF0_COL])
    prev_pf1_arr  = encode_pf_col(enc_pf1, prev_valid[PREV_PF1_COL])

    prev_match_best = ((prev_btb0_arr == true_btb0) & (prev_btb1_arr == true_btb1) &
                       (prev_pf0_arr  == true_pf0)  & (prev_pf1_arr  == true_pf1))
    prev_match_2nd  = ((prev_btb0_arr == btb0_2nd)  & (prev_btb1_arr == btb1_2nd) &
                       (prev_pf0_arr  == pf0_2nd)   & (prev_pf1_arr  == pf1_2nd))
    prev_match_3rd  = ((prev_btb0_arr == btb0_3rd)  & (prev_btb1_arr == btb1_3rd) &
                       (prev_pf0_arr  == pf0_3rd)   & (prev_pf1_arr  == pf1_3rd))
    prev_match_top3 = prev_match_best | prev_match_2nd | prev_match_3rd

    prev_ppw = np.where(prev_match_best, ppw_best,
               np.where(prev_match_2nd,  ppw_2nd,
               np.where(prev_match_3rd,  ppw_3rd,
                                         ppw_3rd)))
    prev_ppw_loss_pct = np.where(has_ppw,
                                  (ppw_best - prev_ppw) / ppw_best * 100,
                                  np.nan)
    prev_ppw_loss_pct = np.where(prev_match_best, 0.0, prev_ppw_loss_pct)

# assemble final results dataframe
results_df = pd.DataFrame({
    'benchmark':         bench_valid.values if bench_valid is not None else 'unknown',
    'best_ppw':          ppw_best,
    'ppw_2nd':           ppw_2nd,
    'ppw_3rd':           ppw_3rd,
    'exact_match':       match_best,
    'top3_match':        match_top3,
    'match_2nd':         match_2nd,
    'match_3rd':         match_3rd,
    'pred_ppw_loss_pct': pred_ppw_loss_pct,
    'gap_best_2nd_pct':  gap_best_2nd_pct,
})
if prev_config_available:
    results_df['prev_is_best']      = prev_match_best
    results_df['prev_in_top3']      = prev_match_top3
    results_df['prev_ppw_loss_pct'] = prev_ppw_loss_pct

out(f'  Results table: {len(results_df):,} rows x {len(results_df.columns)} columns')

benchmarks = ['overall'] + sorted(results_df['benchmark'].unique().tolist())

def subset(bench):
    return results_df if bench == 'overall' \
           else results_df[results_df['benchmark'] == bench]


# ══════════════════════════════════════════════════════════════════════════════
# 5.  STATIC CONFIGS  (groupby — no iterrows)
# ══════════════════════════════════════════════════════════════════════════════
section('COMPUTING STATIC BASELINES')

static_configs = {}
for bench in DATA_FILES.keys():
    mask_b   = top3_valid['benchmark'] == bench
    bench_df = top3_valid[mask_b].copy()
    if bench_df.empty:
        continue
    bench_df['_ppw'] = pd.to_numeric(bench_df['PPW_best'], errors='coerce')
    bench_df = bench_df.dropna(subset=['_ppw'])
    if bench_df.empty:
        continue

    # NaN in prefetcher_core1_best means both cores share the same setting
    bench_df['prefetcher_core1_best'] = bench_df['prefetcher_core1_best'].fillna(
        bench_df['prefetcher_core0_best'])

    grp = (bench_df
           .groupby(['btbCore0_best', 'btbCore1_best',
                     'prefetcher_core0_best', 'prefetcher_core1_best'],
                    dropna=False)['_ppw']
           .sum()
           .reset_index())

    best_row  = grp.loc[grp['_ppw'].idxmax()]
    worst_row = grp.loc[grp['_ppw'].idxmin()]
    n_periods = len(bench_df)

    static_configs[bench] = {
        'best_static':          (best_row['btbCore0_best'],  best_row['btbCore1_best'],
                                 best_row['prefetcher_core0_best'], best_row['prefetcher_core1_best']),
        'worst_static':         (worst_row['btbCore0_best'], worst_row['btbCore1_best'],
                                 worst_row['prefetcher_core0_best'], worst_row['prefetcher_core1_best']),
        'best_static_avg_ppw':  float(best_row['_ppw'])  / n_periods,
        'worst_static_avg_ppw': float(worst_row['_ppw']) / n_periods,
        'optimal_avg_ppw':      bench_df['_ppw'].mean(),
    }
    out(f'  {bench}: best_static  = {static_configs[bench]["best_static"]}')
    out(f'  {"":10}  worst_static = {static_configs[bench]["worst_static"]}')


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PRINT ANALYSES
# ══════════════════════════════════════════════════════════════════════════════
out(f'\n  Analysis run: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}   |   Model: {ts}')

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 1 — PPW % LOSS
# ─────────────────────────────────────────────────────────────────────────────
section('ANALYSIS 1 — PPW % LOSS  (Model Prediction vs Optimal)')

for bench in benchmarks:
    df_b = subset(bench)
    n    = len(df_b)
    subsection(bench.upper())
    out(f'    Samples: {n:,}')

    exact = df_b['exact_match'].sum()
    top3  = df_b['top3_match'].sum()
    m2    = df_b['match_2nd'].sum()
    m3    = df_b['match_3rd'].sum()

    out(f'    Exact match (1st):  {pct_str(exact, n)}')
    out(f'    Match 2nd best:     {pct_str(m2, n)}')
    out(f'    Match 3rd best:     {pct_str(m3, n)}')
    out(f'    Top-3 total:        {pct_str(top3, n)}')
    out(f'    Miss:               {pct_str(n - top3, n)}')

    loss = df_b['pred_ppw_loss_pct'].dropna()
    out(f'    PPW % loss stats:   {stats_str(loss)}')
    loss_buckets(df_b['pred_ppw_loss_pct'], n, label='Model prediction ')


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 2 — STATIC BASELINE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
section('ANALYSIS 2 — STATIC BASELINE COMPARISON')
out('  Metric: average PPW per period  (higher = better)')
out()
out(f'  {"Benchmark":<12} {"Optimal":>14} {"Best Static":>14} '
    f'{"Model Pred":>14} {"Worst Static":>14}')
out('  ' + '-' * 62)

for bench in sorted(static_configs.keys()):
    sc      = static_configs[bench]
    df_b    = subset(bench)
    vm      = df_b['pred_ppw_loss_pct'].notna() & df_b['best_ppw'].notna()
    model_avg = (df_b.loc[vm, 'best_ppw'] *
                 (1 - df_b.loc[vm, 'pred_ppw_loss_pct'] / 100)).mean()

    out(f'  {bench:<12} '
        f'{sc["optimal_avg_ppw"]:>14,.0f} '
        f'{sc["best_static_avg_ppw"]:>14,.0f} '
        f'{model_avg:>14,.0f} '
        f'{sc["worst_static_avg_ppw"]:>14,.0f}')

out()
out('  % gap vs optimal  (negative = worse than optimal):')
out(f'  {"Benchmark":<12} {"Best Static":>14} {"Model Pred":>14} {"Worst Static":>14}')
out('  ' + '-' * 48)

for bench in sorted(static_configs.keys()):
    sc   = static_configs[bench]
    df_b = subset(bench)
    opt  = sc['optimal_avg_ppw']
    if opt == 0 or np.isnan(opt): continue
    vm   = df_b['pred_ppw_loss_pct'].notna() & df_b['best_ppw'].notna()
    model_avg = (df_b.loc[vm, 'best_ppw'] *
                 (1 - df_b.loc[vm, 'pred_ppw_loss_pct'] / 100)).mean()

    out(f'  {bench:<12} '
        f'{(sc["best_static_avg_ppw"]  - opt)/opt*100:>+13.2f}% '
        f'{(model_avg                  - opt)/opt*100:>+13.2f}% '
        f'{(sc["worst_static_avg_ppw"] - opt)/opt*100:>+13.2f}%')

out()
out('  Static configs (btb0, btb1, pf_core0, pf_core1):')
for bench, sc in sorted(static_configs.items()):
    out(f'    {bench:<12}  best:  {sc["best_static"]}')
    out(f'    {"":<12}  worst: {sc["worst_static"]}')


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 3 — BEST vs 2ND-BEST GAP
# ─────────────────────────────────────────────────────────────────────────────
section('ANALYSIS 3 — GAP BETWEEN BEST AND 2ND-BEST CONFIG')
out('  How much PPW do you lose by picking 2nd instead of 1st?')
out('  (Small gap -> 2nd is nearly as good; large gap -> 1st is critical)')

for bench in benchmarks:
    df_b = subset(bench)
    gap  = df_b['gap_best_2nd_pct'].dropna()
    subsection(bench.upper())
    if gap.empty:
        out('    No data.'); continue

    n_gap = len(gap)
    out(f'    Samples with gap data: {n_gap:,}')
    out(f'    Stats: {stats_str(gap)}')
    out()
    out(f'    Gap distribution (% of {n_gap:,} samples with PPW data):')

    for lo, hi, lbl in [
        (None, 0.5, '< 0.5%   (nearly identical)'),
        (0.5,  2,   '0.5-2%   (minor difference) '),
        (2,    5,   '2-5%     (moderate)          '),
        (5,    10,  '5-10%    (significant)        '),
        (10,   20,  '10-20%   (large)              '),
        (20,   None,'>20%     (critical choice)    '),
    ]:
        cnt = (gap <= hi).sum()           if lo is None  else \
              (gap >  lo).sum()           if hi is None  else \
              ((gap > lo) & (gap <= hi)).sum()
        out(f'      {lbl}  {cnt:>6,}  ({cnt/n_gap*100:5.1f}%)')

    df_b2 = df_b[df_b['gap_best_2nd_pct'].notna()]
    if len(df_b2) > 10:
        corr = df_b2['gap_best_2nd_pct'].corr(df_b2['exact_match'].astype(float))
        direction = '(larger gap -> easier to get right)' if corr >= 0 \
                    else '(larger gap -> harder to get right — CONCERNING)'
        out(f'    Corr(gap, exact_match): {corr:.3f}  {direction}')


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 4 — NO-CHANGE BASELINE
# ─────────────────────────────────────────────────────────────────────────────
section('ANALYSIS 4 — NO-CHANGE BASELINE  (Keep Previous Config)')
out("  If we always use the previous period's config instead of predicting,")
out('  how often is it still in the top 3, and how much PPW do we lose?')

if not prev_config_available:
    out('  [SKIP] Previous-config columns not found in data.')
else:
    for bench in benchmarks:
        df_b = subset(bench)
        n    = len(df_b)
        n_p  = int(df_b['prev_in_top3'].notna().sum())
        subsection(bench.upper())

        if n_p == 0:
            out('    No prev-config data.'); continue

        out(f'    Samples: {n:,}  (prev-config available: {n_p:,})')

        prev_best = df_b['prev_is_best'].sum()
        prev_top3 = df_b['prev_in_top3'].sum()

        out(f'    Prev config IS best:       {pct_str(prev_best, n_p)}')
        out(f'    Prev config in top-3:      {pct_str(prev_top3, n_p)}')
        out(f'    Prev config NOT in top-3:  {pct_str(n_p - prev_top3, n_p)}')

        loss = df_b['prev_ppw_loss_pct'].dropna()
        out(f'    PPW % loss stats: {stats_str(loss)}')
        loss_buckets(df_b['prev_ppw_loss_pct'], n_p, label='No-change baseline ')

        out()
        out('    Model vs No-Change comparison:')
        m_loss  = df_b['pred_ppw_loss_pct'].dropna().mean()
        nc_loss = loss.mean() if len(loss) > 0 else float('nan')

        out(f'      {"":32} {"Model":>10}   {"No-Change":>10}')
        out(f'      {"Exact / Is-Best match":32} '
            f'{pct_str(df_b["exact_match"].sum(), n)}   '
            f'{pct_str(prev_best, n_p)}')
        out(f'      {"Top-3 match":32} '
            f'{pct_str(df_b["top3_match"].sum(), n)}   '
            f'{pct_str(prev_top3, n_p)}')
        out(f'      {"Avg PPW % loss":32} {m_loss:>9.2f}%   {nc_loss:>9.2f}%')
        winner = 'MODEL' if m_loss < nc_loss else 'NO-CHANGE'


        out(f'      -> {winner} has lower average PPW loss '
            f'(delta = {abs(m_loss - nc_loss):.2f}%)')





# ==============================================================================
# 5. INFERENCE & RECONFIGURATION COST ANALYSIS
# ==============================================================================
section('ANALYSIS 5 - INFERENCE & RECONFIGURATION OVERHEAD')

# GAINESTOWN/McPAT PARAMETERS (from power.xml, 45nm HP)
TECH_PARAMS = {
    'tech_node_nm': 45,
    'clock_rate_mhz': 2660,
    'vdd': 1.2,
    'device_type': 'HP',
    'btb_entries_default': 18944,
    'l2_capacity_kb': 256,
    'l3_capacity_kb': 8192,
    'E_gate_per_btb_entry_pJ': 0.05,
    'E_gate_per_l2_kb_pJ': 2.5,
    'E_gate_per_l3_kb_pJ': 1.8,
    'E_gate_per_prefetcher_pJ': 15.0,
    'E_rf_inst_pJ': 0.5,
    'E_rf_cmp_pJ': 0.1,
    'E_rf_mem_pJ': 1.0,
}

out('  McPAT/GAINESTOWN PARAMETERS USED:')
out('  ' + '-' * 50)
out(f"  Technology Node:      {TECH_PARAMS['tech_node_nm']} nm")
out(f"  Core Clock:           {TECH_PARAMS['clock_rate_mhz']} MHz")
out(f"  Supply Voltage (Vdd): {TECH_PARAMS['vdd']} V")
out(f"  Device Type:          {TECH_PARAMS['device_type']} (High Performance)")
out()
out('  Component Parameters:')
out(f"    BTB entries (default):  {TECH_PARAMS['btb_entries_default']}")
out(f"    L2 capacity:            {TECH_PARAMS['l2_capacity_kb']} KB")
out(f"    L3 capacity:            {TECH_PARAMS['l3_capacity_kb']} KB")
out()
out('  Energy Parameters (45nm HP, 1.2V):')
out(f"    E_gate_per_btb_entry:   {TECH_PARAMS['E_gate_per_btb_entry_pJ']:.3f} pJ")
out(f"    E_gate_per_l2_kb:       {TECH_PARAMS['E_gate_per_l2_kb_pJ']:.3f} pJ")
out(f"    E_gate_per_l3_kb:       {TECH_PARAMS['E_gate_per_l3_kb_pJ']:.3f} pJ")
out(f"    E_gate_per_prefetcher:  {TECH_PARAMS['E_gate_per_prefetcher_pJ']:.3f} pJ")

# Load cost analysis CSV if provided
if args.cost_csv and os.path.exists(args.cost_csv):
    out()
    out(f"  Cost analysis loaded from: {args.cost_csv}")
    cost_df = pd.read_csv(args.cost_csv)

    if 'inference_energy_pJ' in cost_df.columns:
        inf_e = cost_df['inference_energy_pJ'].iloc[0]
        out()
        out('  INFERENCE COST (per 500K inst interval):')
        out('  ' + '-' * 50)
        out(f"    Total inference:  {inf_e:.2f} pJ ({inf_e/1e6:.4f} uJ)")

    if 'reconfig_energy_pJ' in cost_df.columns:
        n_changes = (cost_df['config_changed'] == True).sum() if 'config_changed' in cost_df.columns else 0
        total_reconfig = cost_df['reconfig_energy_pJ'].sum()
        avg_reconfig = cost_df[cost_df['reconfig_energy_pJ'] > 0]['reconfig_energy_pJ'].mean() if n_changes > 0 else 0

        out()
        out('  RECONFIGURATION COST (only on config change):')
        out('  ' + '-' * 50)

        # Component breakdown
        out('  Component     | Energy/change | % intervals changed')
        out('  ' + '-' * 55)

        if 'btb_changed' in cost_df.columns:
            btb_changes = cost_df['btb_changed'].sum()
            btb_pct = btb_changes / (len(cost_df) - 1) * 100 if len(cost_df) > 1 else 0
            out(f"    BTB (c0+c1)   | N/A           | {btb_pct:.1f}%")

        if 'l2_changed' in cost_df.columns:
            l2_changes = cost_df['l2_changed'].sum()
            l2_pct = l2_changes / (len(cost_df) - 1) * 100 if len(cost_df) > 1 else 0
            out(f"    L2 cache      | {TECH_PARAMS['E_gate_per_l2_kb_pJ']:.1f} pJ/KB  | {l2_pct:.1f}%")

        if 'l3_changed' in cost_df.columns:
            l3_changes = cost_df['l3_changed'].sum()
            l3_pct = l3_changes / (len(cost_df) - 1) * 100 if len(cost_df) > 1 else 0
            out(f"    L3 cache      | {TECH_PARAMS['E_gate_per_l3_kb_pJ']:.1f} pJ/KB  | {l3_pct:.1f}%")

        if 'prefetcher_changed' in cost_df.columns:
            pf_changes = cost_df['prefetcher_changed'].sum()
            pf_pct = pf_changes / (len(cost_df) - 1) * 100 if len(cost_df) > 1 else 0
            out(f"    Prefetcher    | {TECH_PARAMS['E_gate_per_prefetcher_pJ']:.1f} pJ      | {pf_pct:.1f}%")

        out()
        out(f"    Total reconfig:      {total_reconfig:.2f} pJ")
        out(f"    Avg per change:      {avg_reconfig:.2f} pJ" if n_changes > 0 else "    Avg per change:      N/A (no changes)")
        out(f"    Avg per interval:    {total_reconfig/(len(cost_df)-1):.2f} pJ" if len(cost_df) > 1 else "    N/A")

    # Net PPW impact
    if 'net_ppw' in cost_df.columns and 'PPW_best' in cost_df.columns:
        raw_ppw = cost_df['PPW_best'].dropna().mean()
        net_ppw = cost_df['net_ppw'].dropna().mean()

        out()
        out('  NET PPW IMPACT:')
        out('  ' + '-' * 50)
        out(f"    Raw PPW (best config):     {raw_ppw:.4e}")
        out(f"    Net PPW (with overhead):   {net_ppw:.4e}")
        if raw_ppw > 0:
            overhead_pct = (raw_ppw - net_ppw) / raw_ppw * 100
            out(f"    PPW overhead:              {overhead_pct:.2f}%")
else:
    out()
    out('  [SKIP] No cost analysis CSV provided.')
    out('  Run costAnalysis.py first: python costAnalysis.py --benchmark <name> --model-dir saved_models/')


# ==============================================================================
# SAVE REPORT
# ==============================================================================
>>>>>>> refs/remotes/origin/main
# ══════════════════════════════════════════════════════════════════════════════
# SAVE REPORT
# ══════════════════════════════════════════════════════════════════════════════
report_ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
report_path = os.path.join(args.output_dir, f'analysis_report_{report_ts}.txt')
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))

out()
out(f'Report saved -> {report_path}')
