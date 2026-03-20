# Trains one model to predict btbCore0_best, btbCore1_best,
# prefetcher_core0_best, prefetcher_core1_best, l2_size_best, l3_size_best simultaneously.
 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import warnings
import joblib
from datetime import datetime
from collections import Counter
import os
import argparse
warnings.filterwarnings('ignore')
 
 
# ==============================================================================
# Helpers
# ==============================================================================
 
def split_prefetcher(series, suffix=''):
    """
    Split a 'type0-type1' prefetcher column into two per-core columns.
    e.g. 'simple-none' -> core0='simple', core1='none'
    If there is no '-', both cores get the same value.
    """
    split = series.fillna('none-none').astype(str).str.split('-', n=1, expand=True)
    core0 = split[0].str.strip()
    core1 = split[1].str.strip() if split.shape[1] > 1 else core0
    return core0, core1
 
 
def safe_encode(encoder, series):
    """Encode a Series with a fitted LabelEncoder; unseen/NaN → 'none'."""
    valid  = set(encoder.classes_)
    mapped = series.fillna('none').astype(str).apply(lambda x: x if x in valid else 'none')
    return encoder.transform(mapped)
 
 
def normalize_config(btb0, btb1, pf0, pf1, l2, l3):
    try:
        return (float(btb0), float(btb1), int(pf0), int(pf1), int(l2), int(l3))
    except (ValueError, TypeError):
        return None
 
 
# ==============================================================================
# GPU check
# ==============================================================================
USE_GPU = False
try:
    from cuml.ensemble import RandomForestClassifier as cuRF
    import cupy as cp
    USE_GPU = True
    print("GPU detected — using cuML")
except ImportError:
    print("GPU libraries not available — using CPU (sklearn)")
 
# ==============================================================================
# CLI
# ==============================================================================
parser = argparse.ArgumentParser(description='Random Forest 6-target Configuration Predictor')
parser.add_argument('--split-method', type=str, default='random',
                    choices=['random', 'benchmark', 'temporal'])
parser.add_argument('--test-size',  type=float, default=0.2)
parser.add_argument('--tune',       action='store_true', default=True)
parser.add_argument('--no-tune',    dest='tune', action='store_false')
args = parser.parse_args()
 
SPLIT_METHOD               = args.split_method
TEST_SIZE                  = args.test_size
ENABLE_HYPERPARAMETER_TUNING = args.tune
RANDOM_STATE               = 42
 
# ==============================================================================
# Data loading
# ==============================================================================
CSV_FILES = {
    'barnes':    '/home/gina/Desktop/snipersim_framework/pythonScripts/train_with_top3_barnes.csv',
    'cholesky':  '/home/gina/Desktop/snipersim_framework/pythonScripts/train_with_top3_cholesky.csv',
    'fft':       '/home/gina/Desktop/snipersim_framework/pythonScripts/train_with_top3_fft.csv',
    'radiosity': '/home/gina/Desktop/snipersim_framework/pythonScripts/train_with_top3_radiosity.csv',
}
 
print("6-Target Random Forest Configuration Predictor")
print("=" * 60)
print(f"Split: {SPLIT_METHOD} | Test size: {TEST_SIZE} | Tuning: {ENABLE_HYPERPARAMETER_TUNING}")
print("=" * 60)
 
dfs = []
try:
    for bench, path in CSV_FILES.items():
        d = pd.read_csv(path)
        d['benchmark'] = bench
        print(f"  {bench}: {d.shape}")
        dfs.append(d)
except FileNotFoundError as e:
    print(f"File not found: {e}")
    exit()
 
df = pd.concat(dfs, ignore_index=True)
print(f"\nCombined shape: {df.shape}")
 
# ==============================================================================
# Split prefetcher columns for best / 2nd / 3rd
# ==============================================================================
for rank in ['best', '2nd', '3rd']:
    col = f'prefetcher_{rank}'
    if col in df.columns:
        c0, c1 = split_prefetcher(df[col], suffix=rank)
        df[f'prefetcher_core0_{rank}'] = c0
        df[f'prefetcher_core1_{rank}'] = c1
        df.drop(columns=[col], inplace=True)
        print(f"Split '{col}' → prefetcher_core0_{rank} / prefetcher_core1_{rank}")
    else:
        # Already split or missing — ensure columns exist
        if f'prefetcher_core0_{rank}' not in df.columns:
            df[f'prefetcher_core0_{rank}'] = 'none'
        if f'prefetcher_core1_{rank}' not in df.columns:
            df[f'prefetcher_core1_{rank}'] = 'none'
 
# ==============================================================================
# Targets and config columns
# ==============================================================================
TARGET_COLUMNS = [
    'btbCore0_best', 'btbCore1_best',
    'prefetcher_core0_best', 'prefetcher_core1_best',
    'l2_size_best', 'l3_size_best',
]
 
ALL_CONFIG_COLUMNS = [
    # Best
    'btbCore0_best', 'btbCore1_best',
    'prefetcher_core0_best', 'prefetcher_core1_best',
    'l2_size_best', 'l3_size_best', 'PPW_best',
    # 2nd
    'btbCore0_2nd', 'btbCore1_2nd',
    'prefetcher_core0_2nd', 'prefetcher_core1_2nd',
    'l2_size_2nd', 'l3_size_2nd', 'PPW_2nd', 'Diff_best_2nd',
    # 3rd
    'btbCore0_3rd', 'btbCore1_3rd',
    'prefetcher_core0_3rd', 'prefetcher_core1_3rd',
    'l2_size_3rd', 'l3_size_3rd', 'PPW_3rd', 'Diff_best_3rd',
]
 
METADATA_COLUMNS_TO_DROP = [
    'best-config', 'file', 'file_prev', 'period_start', 'period_end',
    'period_start_prev', 'period_end_prev',
    'directory_perf_prev', 'leaf_dir_prev', 'directory_power_prev',
    'leaf_dir_perf_prev', 'leaf_dir_power_prev', 'period_start_val_prev',
    'period_end_val_perf_prev', 'period_start_val_perf_prev',
    'period_start_val_power_prev', 'period_end_val_power_prev',
]
 
df = df.drop(columns=METADATA_COLUMNS_TO_DROP, errors='ignore')
 
missing = [c for c in ALL_CONFIG_COLUMNS if c not in df.columns]
if missing:
    print(f"\nMissing config columns: {missing}")
    print("Available columns with btb/ppw/prefetch/l2/l3:")
    print([c for c in df.columns if any(k in c.lower() for k in ['btb','ppw','prefetch','l2','l3'])])
    exit()
print("All required columns found.")
 
# ==============================================================================
# Feature / target split
# ==============================================================================
X = df.drop(columns=ALL_CONFIG_COLUMNS, errors='ignore')
y = df[TARGET_COLUMNS].copy()
 
top3_configs = {
    'best': df[['btbCore0_best','btbCore1_best',
                'prefetcher_core0_best','prefetcher_core1_best',
                'l2_size_best','l3_size_best','PPW_best']].copy(),
    '2nd':  df[['btbCore0_2nd','btbCore1_2nd',
                'prefetcher_core0_2nd','prefetcher_core1_2nd',
                'l2_size_2nd','l3_size_2nd','PPW_2nd']].copy(),
    '3rd':  df[['btbCore0_3rd','btbCore1_3rd',
                'prefetcher_core0_3rd','prefetcher_core1_3rd',
                'l2_size_3rd','l3_size_3rd','PPW_3rd']].copy(),
}
 
print(f"\nFeatures: {X.shape} | Targets: {y.shape}")
for col in TARGET_COLUMNS:
    print(f"  {col}: {y[col].nunique()} unique → {sorted(y[col].dropna().unique())}")
 
# ==============================================================================
# Encoding
# ==============================================================================
benchmark_original = X['benchmark'].copy() if 'benchmark' in X.columns else None
 
categorical_cols = X.select_dtypes(include=['object']).columns
label_encoders   = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
 
# One encoder per categorical target
prefetcher_encoder_core0 = LabelEncoder()
prefetcher_encoder_core1 = LabelEncoder()
l2_size_encoder          = LabelEncoder()
l3_size_encoder          = LabelEncoder()
 
y['prefetcher_core0_best'] = prefetcher_encoder_core0.fit_transform(y['prefetcher_core0_best'].astype(str))
y['prefetcher_core1_best'] = prefetcher_encoder_core1.fit_transform(y['prefetcher_core1_best'].astype(str))
y['l2_size_best']          = l2_size_encoder.fit_transform(y['l2_size_best'].astype(str))
y['l3_size_best']          = l3_size_encoder.fit_transform(y['l3_size_best'].astype(str))
 
print(f"\nEncodings:")
print(f"  prefetcher_core0: {dict(zip(prefetcher_encoder_core0.classes_, prefetcher_encoder_core0.transform(prefetcher_encoder_core0.classes_)))}")
print(f"  prefetcher_core1: {dict(zip(prefetcher_encoder_core1.classes_, prefetcher_encoder_core1.transform(prefetcher_encoder_core1.classes_)))}")
print(f"  l2_size:          {dict(zip(l2_size_encoder.classes_, l2_size_encoder.transform(l2_size_encoder.classes_)))}")
print(f"  l3_size:          {dict(zip(l3_size_encoder.classes_, l3_size_encoder.transform(l3_size_encoder.classes_)))}")
 
X = X[X.select_dtypes(include=[np.number]).columns]
 
# ==============================================================================
# Train / test split
# ==============================================================================
print(f"\nSplit method: {SPLIT_METHOD}")
 
if SPLIT_METHOD == 'benchmark' and benchmark_original is not None and benchmark_original.nunique() >= 2:
    benchmarks         = benchmark_original.unique()
    n_test             = max(1, int(len(benchmarks) * TEST_SIZE))
    test_benchmarks    = np.random.RandomState(RANDOM_STATE).choice(benchmarks, n_test, replace=False)
    mask               = benchmark_original.isin(test_benchmarks)
    X_train, X_test    = X[~mask].copy(), X[mask].copy()
    y_train, y_test    = y[~mask].copy(), y[mask].copy()
    print(f"  Test benchmarks: {test_benchmarks}")
 
elif SPLIT_METHOD == 'temporal':
    cut                = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test    = X.iloc[:cut].copy(), X.iloc[cut:].copy()
    y_train, y_test    = y.iloc[:cut].copy(), y.iloc[cut:].copy()
 
else:
    stratify_key = y[TARGET_COLUMNS].astype(str).apply('_'.join, axis=1)
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        for tr, te in sss.split(X, stratify_key):
            X_train, X_test = X.iloc[tr].copy(), X.iloc[te].copy()
            y_train, y_test = y.iloc[tr].copy(), y.iloc[te].copy()
    except Exception as e:
        print(f"  Stratification failed ({e}) — random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
 
test_indices   = X_test.index
test_top3      = {rank: data.loc[test_indices].copy() for rank, data in top3_configs.items()}
test_benchmarks_series = benchmark_original[test_indices].copy() if benchmark_original is not None else None
 
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
 
# ==============================================================================
# Scale / impute
# ==============================================================================
scaler         = StandardScaler()
imputer        = SimpleImputer(strategy='mean')
X_train_scaled = imputer.fit_transform(scaler.fit_transform(X_train))
X_test_scaled  = imputer.transform(scaler.transform(X_test))
 
# Clean NaN targets
for col in ['btbCore0_best','btbCore1_best']:
    y_train[col] = pd.to_numeric(y_train[col], errors='coerce')
    y_test[col]  = pd.to_numeric(y_test[col],  errors='coerce')
 
mask_tr        = y_train.notna().all(axis=1)
X_train_scaled = X_train_scaled[mask_tr.values]
y_train        = y_train[mask_tr]
 
mask_te        = y_test.notna().all(axis=1)
X_test_scaled  = X_test_scaled[mask_te.values]
y_test         = y_test[mask_te].reset_index(drop=True)
 
# Align test_top3 with cleaned test set
def encode_top3_rank(rank_data, rank):
    d = rank_data[mask_te.values].reset_index(drop=True).copy()
    for col in [f'btbCore0_{rank}', f'btbCore1_{rank}']:
        d[col] = pd.to_numeric(d[col], errors='coerce')
    d[f'prefetcher_core0_{rank}'] = safe_encode(prefetcher_encoder_core0, d[f'prefetcher_core0_{rank}'])
    d[f'prefetcher_core1_{rank}'] = safe_encode(prefetcher_encoder_core1, d[f'prefetcher_core1_{rank}'])
    d[f'l2_size_{rank}']          = safe_encode(l2_size_encoder,          d[f'l2_size_{rank}'].astype(str))
    d[f'l3_size_{rank}']          = safe_encode(l3_size_encoder,          d[f'l3_size_{rank}'].astype(str))
    d[f'PPW_{rank}']              = pd.to_numeric(d[f'PPW_{rank}'], errors='coerce')
    return d
 
test_top3 = {rank: encode_top3_rank(data, rank) for rank, data in test_top3.items()}
 
print(f"After cleaning — train: {y_train.shape} | test: {y_test.shape}")
print(f"Removed {(~mask_tr).sum()} train and {(~mask_te).sum()} test rows with NaN")
 
# ==============================================================================
# Training
# ==============================================================================
print(f"\nMODEL TRAINING (6 targets)")
print("-" * 40)
 
def exact_match_scorer(estimator, X, y_true):
    return (y_true.values == estimator.predict(X)).all(axis=1).mean()
 
if ENABLE_HYPERPARAMETER_TUNING:
    stratify_key_tune = y_train[TARGET_COLUMNS].astype(str).apply('_'.join, axis=1)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.6, random_state=RANDOM_STATE)
    for ti, vi in sss.split(X_train_scaled, stratify_key_tune):
        X_tune, y_tune = X_train_scaled[ti], y_train.iloc[ti]
        X_val,  y_val  = X_train_scaled[vi], y_train.iloc[vi]
 
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
 
    print("Stage 1: Coarse search...")
    coarse = RandomizedSearchCV(rf, {
        'n_estimators':    [200, 400, 600],
        'max_depth':       [15, 25, None],
        'min_samples_split': [2, 10, 20],
        'max_features':    ['sqrt', 0.5, None],
        'class_weight':    ['balanced', 'balanced_subsample', None],
    }, cv=3, n_iter=20, scoring=exact_match_scorer,
       n_jobs=-1, random_state=RANDOM_STATE, verbose=1)
    coarse.fit(X_tune, y_tune)
    bp = coarse.best_params_
    print(f"  Best score: {coarse.best_score_:.4f} | Params: {bp}")
 
    def adj(v, d, lo=5):
        return [None] if v is None else [max(lo, v-d), v, v+d]
 
    print("Stage 2: Fine search...")
    fine = RandomizedSearchCV(rf, {
        'n_estimators':      [max(50, bp['n_estimators']-50), bp['n_estimators'], bp['n_estimators']+50],
        'max_depth':         adj(bp['max_depth'], 3),
        'min_samples_split': adj(bp['min_samples_split'], 5),
        'min_samples_leaf':  [2, 5, 10],
        'max_features':      [bp['max_features']],
        'class_weight':      ['balanced', None],
        'bootstrap':         [True],
    }, cv=2, n_iter=15, scoring=exact_match_scorer,
       n_jobs=-1, random_state=RANDOM_STATE, verbose=1)
    fine.fit(X_tune, y_tune)
    best_model = fine.best_estimator_
    print(f"  Final score: {fine.best_score_:.4f} | Params: {fine.best_params_}")
    val_score = (y_val.values == best_model.predict(X_val)).all(axis=1).mean()
    print(f"  Validation score: {val_score:.4f}")
 
else:
    best_model = RandomForestClassifier(
        n_estimators=200, max_depth=25, min_samples_split=5,
        min_samples_leaf=2, max_features='sqrt',
        random_state=RANDOM_STATE, n_jobs=-1)
    best_model.fit(X_train_scaled, y_train)
 
# ==============================================================================
# Predictions
# ==============================================================================
y_pred = best_model.predict(X_test_scaled)
predictions = {col: y_pred[:, i] for i, col in enumerate(TARGET_COLUMNS)}
 
# ==============================================================================
# Evaluation
# ==============================================================================
print(f"\nEVALUATION (6-target)")
print("-" * 40)
 
print("Individual accuracies:")
for i, col in enumerate(TARGET_COLUMNS):
    acc = accuracy_score(y_test[col], predictions[col])
    print(f"  {col}: {acc:.4f}")
 
num_samples    = len(y_pred)
exact_matches  = 0
top3_matches   = 0
ppw_diffs      = []
matches_by_rank = {'best': 0, '2nd': 0, '3rd': 0, 'miss': 0}
detailed_results = []
 
for i in range(num_samples):
    pred = normalize_config(
        predictions[TARGET_COLUMNS[0]][i], predictions[TARGET_COLUMNS[1]][i],
        predictions[TARGET_COLUMNS[2]][i], predictions[TARGET_COLUMNS[3]][i],
        predictions[TARGET_COLUMNS[4]][i], predictions[TARGET_COLUMNS[5]][i],
    )
 
    def get_ranked(rank):
        row = test_top3[rank].iloc[i]
        cfg = normalize_config(
            row[f'btbCore0_{rank}'],    row[f'btbCore1_{rank}'],
            row[f'prefetcher_core0_{rank}'], row[f'prefetcher_core1_{rank}'],
            row[f'l2_size_{rank}'],     row[f'l3_size_{rank}'],
        )
        try:
            ppw = float(row[f'PPW_{rank}'])
            ppw = None if np.isnan(ppw) else ppw
        except (ValueError, TypeError):
            ppw = None
        return cfg, ppw
 
    best_cfg,  best_ppw  = get_ranked('best')
    cfg_2nd,   ppw_2nd   = get_ranked('2nd')
    cfg_3rd,   ppw_3rd   = get_ranked('3rd')
 
    actual_configs = [
        {'rank': 'best', 'config': best_cfg,  'ppw': best_ppw},
        {'rank': '2nd',  'config': cfg_2nd,   'ppw': ppw_2nd},
        {'rank': '3rd',  'config': cfg_3rd,   'ppw': ppw_3rd},
    ]
 
    exact_match      = pred is not None and pred == best_cfg
    top3_match       = exact_match
    matched_rank_idx = 0
 
    if not exact_match:
        for idx, cfg in enumerate(actual_configs):
            if pred is not None and pred == cfg['config']:
                top3_match       = True
                matched_rank_idx = idx
                break
 
    if exact_match:
        exact_matches += 1
        matches_by_rank['best'] += 1
    if top3_match:
        top3_matches += 1
        if not exact_match:
            matches_by_rank['2nd' if matched_rank_idx == 1 else '3rd'] += 1
    else:
        matches_by_rank['miss'] += 1
 
    # PPW loss
    if best_ppw is None:
        ppw_diffs.append(None)
    elif exact_match:
        ppw_diffs.append(0.0)
    elif top3_match:
        matched_ppw = actual_configs[matched_rank_idx]['ppw']
        if matched_ppw is not None and best_ppw != 0:
            ppw_diffs.append((best_ppw - matched_ppw) / best_ppw * 100)
        else:
            ppw_diffs.append(None)
    else:
        if ppw_3rd is not None and best_ppw != 0:
            ppw_diffs.append((best_ppw - ppw_3rd) / best_ppw * 100)
        else:
            ppw_diffs.append(None)
 
    detailed_results.append({
        'sample': i, 'predicted': pred,
        'actual_best': best_cfg, 'actual_2nd': cfg_2nd, 'actual_3rd': cfg_3rd,
        'ppw_best': best_ppw, 'ppw_2nd': ppw_2nd, 'ppw_3rd': ppw_3rd,
        'exact_match': exact_match, 'top3_match': top3_match,
    })
 
exact_accuracy = exact_matches / num_samples
top3_accuracy  = top3_matches  / num_samples
 
print(f"\nCombined accuracies (all 6 must match):")
print(f"  Exact match (best config): {exact_accuracy:.4f} ({exact_matches}/{num_samples})")
print(f"  Top-3 match:               {top3_accuracy:.4f} ({top3_matches}/{num_samples})")
print(f"  Top-3 uplift:              +{top3_accuracy - exact_accuracy:.4f}")
 
print(f"\nPredictions by rank:")
for rank in ['best','2nd','3rd','miss']:
    n = matches_by_rank[rank]
    print(f"  {'No match' if rank=='miss' else rank+' config':<14} {n:>6}  ({n/num_samples*100:.2f}%)")
 
valid_diffs = [d for d in ppw_diffs if d is not None and not np.isnan(d)]
if valid_diffs:
    print(f"\nPPW loss (% below best config):")
    print(f"  Average:    {np.mean(valid_diffs):.2f}%")
    print(f"  Median:     {np.median(valid_diffs):.2f}%")
    print(f"  90th pct:   {np.percentile(valid_diffs, 90):.2f}%")
    print(f"  Worst:      {np.max(valid_diffs):.2f}%")
 
# ==============================================================================
# Feature importance
# ==============================================================================
print(f"\nTOP 15 FEATURE IMPORTANCES")
print("-" * 40)
fi = pd.DataFrame({
    'feature':    X_train.columns.tolist(),
    'importance': best_model.feature_importances_,
}).sort_values('importance', ascending=False)
print(fi.head(15).to_string(index=False))
 
# ==============================================================================
# Save model
# ==============================================================================
os.makedirs('saved_models', exist_ok=True)
ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"rf_6way_config_predictor_{ts}"
 
paths = {
    'model':   f"saved_models/{model_name}.pkl",
    'scaler':  f"saved_models/{model_name}_scaler.pkl",
    'imputer': f"saved_models/{model_name}_imputer.pkl",
    'enc_pf0': f"saved_models/{model_name}_prefetcher_core0_encoder.pkl",
    'enc_pf1': f"saved_models/{model_name}_prefetcher_core1_encoder.pkl",
    'enc_l2':  f"saved_models/{model_name}_l2_size_encoder.pkl",
    'enc_l3':  f"saved_models/{model_name}_l3_size_encoder.pkl",
    'meta':    f"saved_models/{model_name}_metadata.txt",
}
 
joblib.dump(best_model,             paths['model'])
joblib.dump(scaler,                 paths['scaler'])
joblib.dump(imputer,                paths['imputer'])
joblib.dump(prefetcher_encoder_core0, paths['enc_pf0'])
joblib.dump(prefetcher_encoder_core1, paths['enc_pf1'])
joblib.dump(l2_size_encoder,        paths['enc_l2'])
joblib.dump(l3_size_encoder,        paths['enc_l3'])
 
with open(paths['meta'], 'w') as f:
    f.write(f"RF 6-target Config Predictor\n{'='*60}\n")
    f.write(f"Timestamp:    {ts}\n")
    f.write(f"Split:        {SPLIT_METHOD}\n")
    f.write(f"Test size:    {TEST_SIZE}\n")
    f.write(f"Train rows:   {len(X_train_scaled)}\n")
    f.write(f"Test rows:    {num_samples}\n")
    f.write(f"\nModel params:\n")
    for k in ['n_estimators','max_depth','min_samples_split','min_samples_leaf','max_features']:
        f.write(f"  {k}: {getattr(best_model, k)}\n")
    f.write(f"\nAccuracies:\n")
    f.write(f"  Exact match: {exact_accuracy:.4f}\n")
    f.write(f"  Top-3 match: {top3_accuracy:.4f}\n")
    for i, col in enumerate(TARGET_COLUMNS):
        f.write(f"  {col}: {accuracy_score(y_test[col], predictions[col]):.4f}\n")
    if valid_diffs:
        f.write(f"\nPPW loss: avg={np.mean(valid_diffs):.4f}% median={np.median(valid_diffs):.4f}%\n")
 
print(f"\nSaved:")
for k, p in paths.items():
    print(f"  {p}")
 
print(f"\nTo load:")
print(f"  model   = joblib.load('{paths['model']}')")
print(f"  scaler  = joblib.load('{paths['scaler']}')")
print(f"  imputer = joblib.load('{paths['imputer']}')")
print(f"  enc_pf0 = joblib.load('{paths['enc_pf0']}')")
print(f"  enc_pf1 = joblib.load('{paths['enc_pf1']}')")
print(f"  enc_l2  = joblib.load('{paths['enc_l2']}')")
print(f"  enc_l3  = joblib.load('{paths['enc_l3']}')")







