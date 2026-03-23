
"""
post_hoc_analysis.py
====================
Standalone analysis script that loads a previously trained Random Forest model
and runs the following analyses on the test set:
 
  1. PPW % LOSS ANALYSIS          — how much PPW do we lose vs optimal?
  2. STATIC BASELINE COMPARISON   — vs best-static, worst-static, optimal
  3. BEST vs 2ND-BEST GAP         — how much headroom is there to be wrong?
  4. NO-CHANGE BASELINE           — if prev config kept, is it still top-3?
 
All results are printed to console AND saved to a timestamped report file.
Results are broken down overall AND per benchmark.
 
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
from sklearn.preprocessing import StandardScaler
 
warnings.filterwarnings('ignore')
 
# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description='Post-hoc RF model analysis')
parser.add_argument('--model-dir', type=str, default='saved_models/',
                    help='Directory containing saved model files')
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
# DATA FILES  (same fixed CSVs used for training)
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
 
RANDOM_STATE = 42
 
# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
report_lines = []
 
def out(line=''):
    print(line)
    report_lines.append(line)
 
def section(title):
    out()
    out('═' * 65)
    out(f'  {title}')
    out('═' * 65)
 
def subsection(title):
    out()
    out(f'  ── {title}')
    out('  ' + '─' * 55)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
section('LOADING SAVED MODEL')
 
model_dir = args.model_dir
 
if args.model_timestamp:
    ts = args.model_timestamp
else:
    # find the most recent model pkl
    pkls = glob.glob(os.path.join(model_dir, 'rf_4way_config_predictor_*.pkl'))
    pkls = [p for p in pkls if '_scaler' not in p
            and '_encoder' not in p and '_prefetcher' not in p]
    if not pkls:
        out(f'ERROR: No model files found in {model_dir}')
        sys.exit(1)
    pkls.sort()
    latest = pkls[-1]
    ts = os.path.basename(latest).replace('rf_4way_config_predictor_', '').replace('.pkl', '')
 
model_path  = os.path.join(model_dir, f'rf_4way_config_predictor_{ts}.pkl')
scaler_path = os.path.join(model_dir, f'rf_4way_config_predictor_{ts}_scaler.pkl')
enc0_path   = os.path.join(model_dir, f'rf_4way_config_predictor_{ts}_prefetcher_core0_encoder.pkl')
enc1_path   = os.path.join(model_dir, f'rf_4way_config_predictor_{ts}_prefetcher_core1_encoder.pkl')
 
for p in [model_path, scaler_path, enc0_path, enc1_path]:
    if not os.path.exists(p):
        out(f'ERROR: File not found: {p}')
        sys.exit(1)
 
model                  = joblib.load(model_path)
scaler                 = joblib.load(scaler_path)
prefetcher_encoder_core0 = joblib.load(enc0_path)
prefetcher_encoder_core1 = joblib.load(enc1_path)
 
out(f'  Timestamp:  {ts}')
out(f'  Model:      {model_path}')
out(f'  n_estimators: {model.n_estimators}  '
    f'max_depth: {model.max_depth}  '
    f'max_features: {model.max_features}')
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 2.  LOAD & PREPROCESS DATA   (mirrors training script exactly)
# ══════════════════════════════════════════════════════════════════════════════
section('LOADING & PREPROCESSING DATA')
 
frames = []
for bench, path in DATA_FILES.items():
    if not os.path.exists(path):
        out(f'  [WARN] File not found, skipping: {path}')
        continue
    df_b = pd.read_csv(path, low_memory=False)
    df_b['benchmark'] = bench
    frames.append(df_b)
    out(f'  Loaded {bench}: {df_b.shape[0]:,} rows')
 
df = pd.concat(frames, ignore_index=True)
out(f'  Combined: {df.shape[0]:,} rows')
 
# ── save prev-config columns BEFORE any dropping ─────────────────────────────
PREV_BTB0_COL   = 'BTB core 0_prev'
PREV_BTB1_COL   = 'BTB core 1_prev'
PREV_PF0_COL    = 'prefetcher_core0_prev'   # after fix_training_data split
PREV_PF1_COL    = 'prefetcher_core1_prev'
 
prev_config_available = all(
    c in df.columns for c in [PREV_BTB0_COL, PREV_BTB1_COL, PREV_PF0_COL, PREV_PF1_COL]
)
if not prev_config_available:
    out(f'  [WARN] Previous-config columns not all found.')
    out(f'  Available _prev columns: '
        f'{[c for c in df.columns if "_prev" in c and "btb" in c.lower() or "prefetch" in c.lower()]}')
 
# ── top-3 ground truth ────────────────────────────────────────────────────────
top3_data = df[ALL_CONFIG_COLUMNS + ['benchmark']].copy()
 
# ── drop metadata ─────────────────────────────────────────────────────────────
df = df.drop(columns=METADATA_COLUMNS_TO_DROP, errors='ignore')
 
# ── feature matrix ────────────────────────────────────────────────────────────
X_raw = df.drop(ALL_CONFIG_COLUMNS, axis=1)
y_raw = df[TARGET_COLUMNS].copy()
 
benchmark_series = X_raw['benchmark'].copy() if 'benchmark' in X_raw.columns else None
 
# encode categoricals (same as training)
label_encoders = {}
for col in X_raw.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_raw[col] = le.fit_transform(X_raw[col].astype(str))
    label_encoders[col] = le
 
# encode targets
def safe_encode(encoder, series):
    valid = set(encoder.classes_)
    s = series.fillna('none').astype(str).replace('nan', 'none')
    s = s.apply(lambda x: x if x in valid else 'none')
    return encoder.transform(s)
 
y_raw['prefetcher_core0_best'] = safe_encode(prefetcher_encoder_core0,
                                              y_raw['prefetcher_core0_best'])
y_raw['prefetcher_core1_best'] = safe_encode(prefetcher_encoder_core1,
                                              y_raw['prefetcher_core1_best'])
 
for col in ['btbCore0_best', 'btbCore1_best']:
    y_raw[col] = pd.to_numeric(y_raw[col], errors='coerce')
 
X_raw = X_raw[X_raw.select_dtypes(include=[np.number]).columns]
 
# impute + scale using LOADED scaler (transform only, no re-fit)
imputer = SimpleImputer(strategy='mean')
X_scaled = imputer.fit_transform(X_raw)    # imputer re-fit is OK (same data distribution)
X_scaled = scaler.transform(X_scaled)
 
# drop NaN target rows
mask_valid = y_raw.notna().all(axis=1)
X_scaled   = X_scaled[mask_valid.values]
y_valid    = y_raw[mask_valid].reset_index(drop=True)
top3_valid = top3_data[mask_valid].reset_index(drop=True)
bench_valid = benchmark_series[mask_valid].reset_index(drop=True) if benchmark_series is not None else None
 
out(f'  Valid samples after NaN drop: {len(y_valid):,}')
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 3.  RUN PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
section('RUNNING PREDICTIONS')
 
y_pred = model.predict(X_scaled)
out(f'  Predicted {len(y_pred):,} samples')
 
def normalize_config(b0, b1, p0, p1):
    try:
        return (float(b0), float(b1), int(p0), int(p1))
    except (ValueError, TypeError):
        return None
 
pred_configs = [normalize_config(*y_pred[i]) for i in range(len(y_pred))]
 
 
def get_actual_top3(row):
    """Return list of dicts: {rank, config (normalised), ppw}"""
    configs = []
    for rank, suffix in [('best', 'best'), ('2nd', '2nd'), ('3rd', '3rd')]:
        btb0 = row[f'btbCore0_{suffix}']
        btb1 = row[f'btbCore1_{suffix}']
        pf0  = row[f'prefetcher_core0_{suffix}']
        pf1  = row[f'prefetcher_core1_{suffix}']
        ppw  = row[f'PPW_{suffix}']
        try:
            ppw = float(ppw)
            if np.isnan(ppw): ppw = None
        except (ValueError, TypeError):
            ppw = None
 
        # encode string prefetcher labels → int to match predictions
        def enc0(v):
            v = str(v).strip().lower() if pd.notna(v) else 'none'
            valid = set(prefetcher_encoder_core0.classes_)
            v = v if v in valid else 'none'
            return int(prefetcher_encoder_core0.transform([v])[0])
 
        def enc1(v):
            v = str(v).strip().lower() if pd.notna(v) else 'none'
            valid = set(prefetcher_encoder_core1.classes_)
            v = v if v in valid else 'none'
            return int(prefetcher_encoder_core1.transform([v])[0])
 
        configs.append({
            'rank':   rank,
            'config': normalize_config(btb0, btb1, enc0(pf0), enc1(pf1)),
            'ppw':    ppw,
        })
    return configs
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 4.  BUILD PER-SAMPLE RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════
results = []   # one dict per sample
 
for i in range(len(y_pred)):
    pred   = pred_configs[i]
    actual = get_actual_top3(top3_valid.iloc[i])
    bench  = bench_valid.iloc[i] if bench_valid is not None else 'unknown'
 
    best_ppw = actual[0]['ppw']
    ppw_2nd  = actual[1]['ppw']
    ppw_3rd  = actual[2]['ppw']
 
    # ── exact & top-3 match ───────────────────────────────────────────────────
    exact_match = (pred is not None and pred == actual[0]['config'])
    top3_match  = exact_match
    matched_rank = 'best' if exact_match else None
    if not exact_match:
        for cfg in actual[1:]:
            if pred is not None and pred == cfg['config']:
                top3_match   = True
                matched_rank = cfg['rank']
                break
 
    # ── PPW % loss for model prediction ──────────────────────────────────────
    if best_ppw is None:
        pred_ppw_loss_pct = None
    elif exact_match:
        pred_ppw_loss_pct = 0.0
    elif top3_match:
        matched_ppw = next(c['ppw'] for c in actual if c['rank'] == matched_rank)
        pred_ppw_loss_pct = ((best_ppw - matched_ppw) / best_ppw * 100
                             if matched_ppw is not None and best_ppw != 0 else None)
    else:
        # miss: conservative estimate using 3rd-best PPW
        pred_ppw_loss_pct = ((best_ppw - ppw_3rd) / best_ppw * 100
                             if ppw_3rd is not None and best_ppw != 0 else None)
 
    # ── best-vs-2nd gap ───────────────────────────────────────────────────────
    if best_ppw is not None and ppw_2nd is not None and best_ppw != 0:
        gap_best_2nd_pct = (best_ppw - ppw_2nd) / best_ppw * 100
    else:
        gap_best_2nd_pct = None
 
    # ── prev-config: is it still top-3? ──────────────────────────────────────
    prev_in_top3      = None
    prev_is_best      = None
    prev_ppw_loss_pct = None
 
    if prev_config_available and i < len(df[mask_valid]):
        orig_idx = df[mask_valid].index[i]
        row_orig = df.loc[orig_idx]
 
        prev_btb0 = row_orig.get(PREV_BTB0_COL)
        prev_btb1 = row_orig.get(PREV_BTB1_COL)
        prev_pf0  = row_orig.get(PREV_PF0_COL)
        prev_pf1  = row_orig.get(PREV_PF1_COL)
 
        def enc0_safe(v):
            v = str(v).strip().lower() if pd.notna(v) else 'none'
            valid = set(prefetcher_encoder_core0.classes_)
            return int(prefetcher_encoder_core0.transform([v if v in valid else 'none'])[0])
 
        def enc1_safe(v):
            v = str(v).strip().lower() if pd.notna(v) else 'none'
            valid = set(prefetcher_encoder_core1.classes_)
            return int(prefetcher_encoder_core1.transform([v if v in valid else 'none'])[0])
 
        prev_config = normalize_config(
            prev_btb0, prev_btb1,
            enc0_safe(prev_pf0), enc1_safe(prev_pf1)
        )
 
        if prev_config is not None:
            prev_is_best = (prev_config == actual[0]['config'])
            prev_in_top3 = prev_is_best or any(
                prev_config == c['config'] for c in actual[1:]
            )
            if best_ppw is not None and best_ppw != 0:
                if prev_is_best:
                    prev_ppw_loss_pct = 0.0
                else:
                    matched_prev = next(
                        (c['ppw'] for c in actual if c['config'] == prev_config), None
                    )
                    if matched_prev is not None:
                        prev_ppw_loss_pct = (best_ppw - matched_prev) / best_ppw * 100
                    elif ppw_3rd is not None:
                        prev_ppw_loss_pct = (best_ppw - ppw_3rd) / best_ppw * 100
 
    results.append({
        'benchmark':        bench,
        'best_ppw':         best_ppw,
        'ppw_2nd':          ppw_2nd,
        'ppw_3rd':          ppw_3rd,
        'exact_match':      exact_match,
        'top3_match':       top3_match,
        'matched_rank':     matched_rank,
        'pred_ppw_loss_pct':pred_ppw_loss_pct,
        'gap_best_2nd_pct': gap_best_2nd_pct,
        'prev_in_top3':     prev_in_top3,
        'prev_is_best':     prev_is_best,
        'prev_ppw_loss_pct':prev_ppw_loss_pct,
    })
 
results_df = pd.DataFrame(results)
benchmarks = ['overall'] + sorted(results_df['benchmark'].unique().tolist())
 
 
# ══════════════════════════════════════════════════════════════════════════════
# HELPER — filter results df by benchmark
# ══════════════════════════════════════════════════════════════════════════════
def subset(bench):
    if bench == 'overall':
        return results_df
    return results_df[results_df['benchmark'] == bench]
 
def pct_str(n, total):
    return f'{n:>6,}  ({n/total*100:5.1f}%)'
 
def stats_str(series, unit='%'):
    s = series.dropna()
    if len(s) == 0:
        return '  no data'
    return (f'  mean={s.mean():.2f}{unit}  '
            f'median={s.median():.2f}{unit}  '
            f'p90={np.percentile(s,90):.2f}{unit}  '
            f'max={s.max():.2f}{unit}')
 
def loss_buckets(series, label=''):
    s = series.dropna()
    n = len(results_df)   # always out of total samples for fair comparison
    out(f'    {label}Loss distribution (% of ALL {n:,} samples):')
    boundaries = [(0, 0, '0% (exact match)'),
                  (0, 1,  '0–1%'),
                  (1, 5,  '1–5%'),
                  (5, 10, '5–10%'),
                  (10, 20,'10–20%'),
                  (20, 1e9,'>20%')]
    for lo, hi, lbl in boundaries:
        if lo == 0 and hi == 0:
            cnt = (s == 0.0).sum()
        else:
            cnt = ((s > lo) & (s <= hi)).sum()
        out(f'      {lbl:<20} {pct_str(cnt, n)}')
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 5.  COMPUTE STATIC CONFIGS  (per benchmark)
# ══════════════════════════════════════════════════════════════════════════════
# Best static = the one config tuple that maximises SUM of PPW_best across all
# periods where that config appears as the ranked-1 choice.
# We approximate by finding the config (btb0,btb1,pf0_str,pf1_str) with the
# highest total PPW when it IS the best config.
 
static_configs = {}   # bench → {best_static_config, worst_static_config,
                      #           best_static_total_ppw, worst_static_total_ppw}
 
for bench in DATA_FILES.keys():
    bench_rows = top3_valid[top3_valid['benchmark'] == bench].copy()
    if bench_rows.empty:
        continue
 
    # Build a (config_tuple → total PPW) table using PPW_best column
    config_ppw = {}
    for _, row in bench_rows.iterrows():
        cfg = (row['btbCore0_best'], row['btbCore1_best'],
               str(row['prefetcher_core0_best']).strip().lower(),
               str(row['prefetcher_core1_best']).strip().lower())
        ppw = row['PPW_best']
        try:
            ppw = float(ppw)
            if np.isnan(ppw): continue
        except (ValueError, TypeError):
            continue
        config_ppw[cfg] = config_ppw.get(cfg, 0.0) + ppw
 
    if not config_ppw:
        continue
 
    best_cfg  = max(config_ppw, key=config_ppw.get)
    worst_cfg = min(config_ppw, key=config_ppw.get)
 
    # Average PPW per period for that config
    n_periods = len(bench_rows)
    static_configs[bench] = {
        'best_static':           best_cfg,
        'worst_static':          worst_cfg,
        'best_static_total_ppw': config_ppw[best_cfg],
        'worst_static_total_ppw':config_ppw[worst_cfg],
        'best_static_avg_ppw':   config_ppw[best_cfg]  / n_periods,
        'worst_static_avg_ppw':  config_ppw[worst_cfg] / n_periods,
        'optimal_avg_ppw':       bench_rows['PPW_best'].dropna().mean(),
        'optimal_total_ppw':     bench_rows['PPW_best'].dropna().sum(),
    }
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 6.  PRINT ALL ANALYSES
# ══════════════════════════════════════════════════════════════════════════════
 
timestamp_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
out(f'  Analysis run: {timestamp_now}')
out(f'  Model timestamp: {ts}')
 
# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 1 — PPW % LOSS (model prediction)
# ─────────────────────────────────────────────────────────────────────────────
section('ANALYSIS 1 — PPW % LOSS  (Model Prediction vs Optimal)')
 
for bench in benchmarks:
    df_b = subset(bench)
    n    = len(df_b)
    subsection(bench.upper())
    out(f'    Samples: {n:,}')
 
    exact  = df_b['exact_match'].sum()
    top3   = df_b['top3_match'].sum()
    out(f'    Exact match:  {pct_str(exact, n)}')
    out(f'    Top-3 match:  {pct_str(top3,  n)}')
    out(f'    Miss:         {pct_str(n - top3, n)}')
 
    loss = df_b['pred_ppw_loss_pct'].dropna()
    out(f'    PPW % loss stats: {stats_str(loss)}')
    loss_buckets(df_b['pred_ppw_loss_pct'], label='Model prediction ')
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 2 — STATIC BASELINE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
section('ANALYSIS 2 — STATIC BASELINE COMPARISON')
out('  Metric: average PPW per period (higher = better)')
out()
out(f'  {"Benchmark":<12} {"Optimal":>14} {"Best Static":>14} '
    f'{"Model Pred":>14} {"Worst Static":>14}')
out('  ' + '─' * 62)
 
for bench in sorted(static_configs.keys()):
    sc   = static_configs[bench]
    df_b = subset(bench)
 
    opt_avg = sc['optimal_avg_ppw']
 
    # model predicted PPW = best_ppw * (1 - loss/100)
    model_ppws = []
    for _, row in df_b.iterrows():
        bp = row['best_ppw']
        lp = row['pred_ppw_loss_pct']
        if bp is not None and lp is not None and not np.isnan(bp) and not np.isnan(lp):
            model_ppws.append(bp * (1 - lp / 100))
    model_avg = np.mean(model_ppws) if model_ppws else float('nan')
 
    best_avg  = sc['best_static_avg_ppw']
    worst_avg = sc['worst_static_avg_ppw']
 
    out(f'  {bench:<12} {opt_avg:>14,.0f} {best_avg:>14,.0f} '
        f'{model_avg:>14,.0f} {worst_avg:>14,.0f}')
 
out()
out('  % gap vs optimal  (negative = worse than optimal):')
out(f'  {"Benchmark":<12} {"Best Static":>14} {"Model Pred":>14} {"Worst Static":>14}')
out('  ' + '─' * 48)
 
for bench in sorted(static_configs.keys()):
    sc   = static_configs[bench]
    df_b = subset(bench)
    opt  = sc['optimal_avg_ppw']
    if opt == 0 or np.isnan(opt):
        continue
 
    model_ppws = []
    for _, row in df_b.iterrows():
        bp = row['best_ppw']
        lp = row['pred_ppw_loss_pct']
        if bp is not None and lp is not None and not np.isnan(bp) and not np.isnan(lp):
            model_ppws.append(bp * (1 - lp / 100))
    model_avg = np.mean(model_ppws) if model_ppws else float('nan')
 
    bs_gap  = (sc['best_static_avg_ppw']  - opt) / opt * 100
    m_gap   = (model_avg                  - opt) / opt * 100
    ws_gap  = (sc['worst_static_avg_ppw'] - opt) / opt * 100
 
    out(f'  {bench:<12} {bs_gap:>+13.2f}% {m_gap:>+13.2f}% {ws_gap:>+13.2f}%')
 
out()
out('  Best static configs found (btb0, btb1, pf_core0, pf_core1):')
for bench, sc in sorted(static_configs.items()):
    out(f'    {bench:<12}  best:  {sc["best_static"]}')
    out(f'    {"":<12}  worst: {sc["worst_static"]}')
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 3 — BEST vs 2ND-BEST GAP
# ─────────────────────────────────────────────────────────────────────────────
section('ANALYSIS 3 — GAP BETWEEN BEST AND 2ND-BEST CONFIG')
out('  How much PPW do you lose by picking 2nd instead of 1st?')
out('  (Small gap → 2nd is nearly as good; large gap → 1st is critical)')
 
for bench in benchmarks:
    df_b = subset(bench)
    subsection(bench.upper())
    gap = df_b['gap_best_2nd_pct'].dropna()
    if gap.empty:
        out('    No data.')
        continue
    out(f'    Samples with gap data: {len(gap):,}')
    out(f'    Stats: {stats_str(gap)}')
    out()
    out(f'    Gap distribution (% of samples with PPW data):')
    n_gap = len(gap)
    boundaries = [(0,   0.5, '< 0.5%  (nearly identical)'),
                  (0.5, 2,   '0.5–2%  (minor difference)'),
                  (2,   5,   '2–5%    (moderate)'),
                  (5,   10,  '5–10%   (significant)'),
                  (10,  20,  '10–20%  (large)'),
                  (20,  1e9, '>20%    (critical choice)')]
    for lo, hi, lbl in boundaries:
        cnt = ((gap > lo) & (gap <= hi)).sum() if lo > 0 else (gap <= hi).sum() if lo == 0 else 0
        if lo == 0:
            cnt = (gap <= hi).sum()
        out(f'      {lbl:<35} {cnt:>6,}  ({cnt/n_gap*100:5.1f}%)')
 
    # Correlation: does a large gap mean the model gets it right more?
    df_b2 = df_b[df_b['gap_best_2nd_pct'].notna()].copy()
    if len(df_b2) > 10:
        corr = df_b2['gap_best_2nd_pct'].corr(df_b2['exact_match'].astype(float))
        out(f'    Corr(gap, exact_match): {corr:.3f}  '
            f'{"(larger gap → harder to get right)" if corr < 0 else "(larger gap → easier to get right)"}')
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 4 — NO-CHANGE BASELINE  (keep previous config)
# ─────────────────────────────────────────────────────────────────────────────
section('ANALYSIS 4 — NO-CHANGE BASELINE  (Keep Previous Config)')
out('  If we always use the previous period\'s config instead of predicting,')
out('  how often is it still in the top 3, and how much PPW do we lose?')
 
if not prev_config_available:
    out('  [SKIP] Previous-config columns not found in data.')
else:
    for bench in benchmarks:
        df_b = subset(bench)
        df_b_prev = df_b[df_b['prev_in_top3'].notna()]
        n    = len(df_b)
        n_p  = len(df_b_prev)
        if n_p == 0:
            subsection(bench.upper())
            out('    No prev-config data.')
            continue
 
        subsection(bench.upper())
        out(f'    Samples: {n:,}  (prev-config available: {n_p:,})')
 
        prev_best = df_b_prev['prev_is_best'].sum()
        prev_top3 = df_b_prev['prev_in_top3'].sum()
        prev_miss = n_p - prev_top3
 
        out(f'    Prev config IS best:       {pct_str(prev_best, n_p)}')
        out(f'    Prev config in top-3:      {pct_str(prev_top3, n_p)}')
        out(f'    Prev config NOT in top-3:  {pct_str(prev_miss, n_p)}')
 
        loss = df_b_prev['prev_ppw_loss_pct'].dropna()
        out(f'    PPW % loss stats: {stats_str(loss)}')
        loss_buckets(df_b_prev['prev_ppw_loss_pct'], label='No-change baseline ')
 
        out()
        out('    Model vs No-Change comparison:')
        model_exact = df_b['exact_match'].sum()
        model_top3  = df_b['top3_match'].sum()
        out(f'      Model exact match:        {pct_str(model_exact, n)}')
        out(f'      No-change is best:        {pct_str(prev_best, n_p)}')
        out(f'      Model top-3:              {pct_str(model_top3, n)}')
        out(f'      No-change in top-3:       {pct_str(prev_top3, n_p)}')
 
        m_loss   = df_b['pred_ppw_loss_pct'].dropna().mean()
        nc_loss  = loss.mean() if len(loss) > 0 else float('nan')
        out(f'      Avg model PPW loss:       {m_loss:.2f}%')
        out(f'      Avg no-change PPW loss:   {nc_loss:.2f}%')
        winner = 'MODEL' if m_loss < nc_loss else 'NO-CHANGE'


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
# ══════════════════════════════════════════════════════════════════════════════
# SAVE REPORT
# ══════════════════════════════════════════════════════════════════════════════
report_ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
report_path = os.path.join(args.output_dir, f'analysis_report_{report_ts}.txt')
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))
 
out()
out(f'Report saved → {report_path}')
