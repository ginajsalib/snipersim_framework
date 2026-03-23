# Trains one model to predict btbCore0_best, btbCore1_best,
# prefetcher_core0_best, prefetcher_core1_best, l2_size_best, l3_size_best simultaneously.
# Supports --load-model to skip training and reuse a saved model for evaluation.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
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

def split_prefetcher(series):
    """Split 'type0-type1' into two per-core columns. e.g. 'simple-none' -> ('simple','none')"""
    split = series.fillna('none-none').astype(str).str.split('-', n=1, expand=True)
    core0 = split[0].str.strip()
    core1 = split[1].str.strip() if split.shape[1] > 1 else core0
    return core0, core1


def safe_encode(encoder, series):
    """Encode a Series with a fitted LabelEncoder; unseen/NaN → 'none'."""
    valid  = set(encoder.classes_)
    mapped = series.fillna('none').astype(str).apply(lambda x: x if x in valid else 'none')
    return encoder.transform(mapped)


def norm_size(series):
    """Normalize numeric size values to int-string to avoid '256' vs '256.0' mismatch."""
    return series.apply(lambda x: str(int(float(x))) if str(x) not in ('nan', '', 'None') else 'nan')


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
parser.add_argument(
    '--load-model', type=str, default=None, metavar='MODEL_PREFIX',
    help=(
        'Skip training and load a previously saved model. '
        'Pass the filename prefix inside saved_models/ '
        '(e.g. rf_6way_config_predictor_20240101_120000). '
        'All 7 artifacts (model, scaler, imputer, 4 encoders) are loaded automatically.'
    )
)
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
print(f"Split: {SPLIT_METHOD} | Test size: {TEST_SIZE} | "
      f"{'LOAD: ' + args.load_model if args.load_model else 'Tuning: ' + str(ENABLE_HYPERPARAMETER_TUNING)}")
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
        c0, c1 = split_prefetcher(df[col])
        df[f'prefetcher_core0_{rank}'] = c0
        df[f'prefetcher_core1_{rank}'] = c1
        df.drop(columns=[col], inplace=True)
        print(f"Split '{col}' → prefetcher_core0_{rank} / prefetcher_core1_{rank}")
    else:
        for sfx in ['core0', 'core1']:
            if f'prefetcher_{sfx}_{rank}' not in df.columns:
                df[f'prefetcher_{sfx}_{rank}'] = 'none'

# ==============================================================================
# Targets and config columns
# ==============================================================================
TARGET_COLUMNS = [
    'btbCore0_best', 'btbCore1_best',
    'prefetcher_core0_best', 'prefetcher_core1_best',
    'l2_size_best', 'l3_size_best',
]

ALL_CONFIG_COLUMNS = [
    'btbCore0_best', 'btbCore1_best',
    'prefetcher_core0_best', 'prefetcher_core1_best',
    'l2_size_best', 'l3_size_best', 'PPW_best',
    'btbCore0_2nd', 'btbCore1_2nd',
    'prefetcher_core0_2nd', 'prefetcher_core1_2nd',
    'l2_size_2nd', 'l3_size_2nd', 'PPW_2nd', 'Diff_best_2nd',
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
    print([c for c in df.columns if any(k in c.lower() for k in ['btb','ppw','prefetch','l2','l3'])])
    exit()
print("All required columns found.")

# ==============================================================================
# Feature / target split  (keep raw values — encoding happens per-path below)
# ==============================================================================
X = df.drop(columns=ALL_CONFIG_COLUMNS, errors='ignore')
y = df[TARGET_COLUMNS].copy()

# Store raw top3 data (unencoded) — both paths need this
top3_configs_raw = {
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

benchmark_original = X['benchmark'].copy() if 'benchmark' in X.columns else None

# Encode categorical features
categorical_cols = X.select_dtypes(include=['object']).columns
label_encoders   = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

X = X[X.select_dtypes(include=[np.number]).columns]

# ==============================================================================
# Shared helper: encode top3 slice using already-fitted encoders
# (called after mask_te is known in each path)
# ==============================================================================
def encode_top3_slice(rank_data, rank, mask):
    d = rank_data.iloc[mask].reset_index(drop=True).copy() if isinstance(mask, np.ndarray) \
        else rank_data[mask].reset_index(drop=True).copy()
    for col in [f'btbCore0_{rank}', f'btbCore1_{rank}']:
        d[col] = pd.to_numeric(d[col], errors='coerce')
    d[f'prefetcher_core0_{rank}'] = safe_encode(prefetcher_encoder_core0, d[f'prefetcher_core0_{rank}'])
    d[f'prefetcher_core1_{rank}'] = safe_encode(prefetcher_encoder_core1, d[f'prefetcher_core1_{rank}'])
    d[f'l2_size_{rank}'] = safe_encode(l2_size_encoder, norm_size(d[f'l2_size_{rank}']))
    d[f'l3_size_{rank}'] = safe_encode(l3_size_encoder, norm_size(d[f'l3_size_{rank}']))
    d[f'PPW_{rank}']     = pd.to_numeric(d[f'PPW_{rank}'], errors='coerce')
    return d

# ==============================================================================
# Encoder instances (fitted in train path, loaded in load path)
# ==============================================================================
prefetcher_encoder_core0 = LabelEncoder()
prefetcher_encoder_core1 = LabelEncoder()
l2_size_encoder          = LabelEncoder()
l3_size_encoder          = LabelEncoder()

# ==============================================================================
# Deterministic train/test split (same seed → same split regardless of path)
# ==============================================================================
def do_split(X, y):
    if SPLIT_METHOD == 'benchmark' and benchmark_original is not None and benchmark_original.nunique() >= 2:
        benchmarks      = benchmark_original.unique()
        n_test          = max(1, int(len(benchmarks) * TEST_SIZE))
        test_benchmarks = np.random.RandomState(RANDOM_STATE).choice(benchmarks, n_test, replace=False)
        mask            = benchmark_original.isin(test_benchmarks)
        print(f"  Test benchmarks: {test_benchmarks}")
        return X[~mask].copy(), X[mask].copy(), y[~mask].copy(), y[mask].copy()
    elif SPLIT_METHOD == 'temporal':
        cut = int(len(X) * (1 - TEST_SIZE))
        return X.iloc[:cut].copy(), X.iloc[cut:].copy(), y.iloc[:cut].copy(), y.iloc[cut:].copy()
    else:
        stratify_key = y[TARGET_COLUMNS].astype(str).apply('_'.join, axis=1)
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
            for tr, te in sss.split(X, stratify_key):
                return X.iloc[tr].copy(), X.iloc[te].copy(), y.iloc[tr].copy(), y.iloc[te].copy()
        except Exception as e:
            print(f"  Stratification failed ({e}) — random split")
            return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# ==============================================================================
# LOAD PATH
# ==============================================================================
if args.load_model:
    prefix = os.path.join('saved_models', args.load_model)
    print(f"\nLOADING MODEL: {prefix}")

    best_model               = joblib.load(f"{prefix}.pkl")
    scaler                   = joblib.load(f"{prefix}_scaler.pkl")
    imputer                  = joblib.load(f"{prefix}_imputer.pkl")
    prefetcher_encoder_core0 = joblib.load(f"{prefix}_prefetcher_core0_encoder.pkl")
    prefetcher_encoder_core1 = joblib.load(f"{prefix}_prefetcher_core1_encoder.pkl")
    l2_size_encoder          = joblib.load(f"{prefix}_l2_size_encoder.pkl")
    l3_size_encoder          = joblib.load(f"{prefix}_l3_size_encoder.pkl")
    print("  All 7 artifacts loaded.")

    # Encode y with loaded encoders
    y['prefetcher_core0_best'] = safe_encode(prefetcher_encoder_core0, y['prefetcher_core0_best'].astype(str))
    y['prefetcher_core1_best'] = safe_encode(prefetcher_encoder_core1, y['prefetcher_core1_best'].astype(str))
    y['l2_size_best']          = safe_encode(l2_size_encoder, norm_size(y['l2_size_best']))
    y['l3_size_best']          = safe_encode(l3_size_encoder, norm_size(y['l3_size_best']))

    # Same deterministic split as training
    X_train, X_test, y_train, y_test = do_split(X, y)
    test_indices  = X_test.index
    top3_test_raw = {rank: data.loc[test_indices].copy() for rank, data in top3_configs_raw.items()}

    # Scale with loaded scaler/imputer
    X_train_scaled = imputer.transform(scaler.transform(X_train))
    X_test_scaled  = imputer.transform(scaler.transform(X_test))

    # Clean NaN targets
    for col in ['btbCore0_best', 'btbCore1_best']:
        y_test[col] = pd.to_numeric(y_test[col], errors='coerce')
    mask_te       = y_test.notna().all(axis=1)
    X_test_scaled = X_test_scaled[mask_te.values]
    y_test        = y_test[mask_te].reset_index(drop=True)
    test_top3     = {rank: encode_top3_slice(data, rank, mask_te.values)
                     for rank, data in top3_test_raw.items()}

    print(f"  Test set: {y_test.shape[0]} rows")

# ==============================================================================
# TRAIN PATH
# ==============================================================================
else:
    print(f"\nMODEL TRAINING (6 targets)")
    print("-" * 40)

    # Fit encoders on full y before split
    y['prefetcher_core0_best'] = prefetcher_encoder_core0.fit_transform(y['prefetcher_core0_best'].astype(str))
    y['prefetcher_core1_best'] = prefetcher_encoder_core1.fit_transform(y['prefetcher_core1_best'].astype(str))
    y['l2_size_best']          = l2_size_encoder.fit_transform(norm_size(y['l2_size_best']))
    y['l3_size_best']          = l3_size_encoder.fit_transform(norm_size(y['l3_size_best']))

    print(f"\nEncodings:")
    print(f"  prefetcher_core0: {dict(zip(prefetcher_encoder_core0.classes_, prefetcher_encoder_core0.transform(prefetcher_encoder_core0.classes_)))}")
    print(f"  prefetcher_core1: {dict(zip(prefetcher_encoder_core1.classes_, prefetcher_encoder_core1.transform(prefetcher_encoder_core1.classes_)))}")
    print(f"  l2_size:          {dict(zip(l2_size_encoder.classes_, l2_size_encoder.transform(l2_size_encoder.classes_)))}")
    print(f"  l3_size:          {dict(zip(l3_size_encoder.classes_, l3_size_encoder.transform(l3_size_encoder.classes_)))}")

    X_train, X_test, y_train, y_test = do_split(X, y)
    test_indices  = X_test.index
    top3_test_raw = {rank: data.loc[test_indices].copy() for rank, data in top3_configs_raw.items()}
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    scaler  = StandardScaler()
    imputer = SimpleImputer(strategy='mean')
    X_train_scaled = imputer.fit_transform(scaler.fit_transform(X_train))
    X_test_scaled  = imputer.transform(scaler.transform(X_test))

    for col in ['btbCore0_best', 'btbCore1_best']:
        y_train[col] = pd.to_numeric(y_train[col], errors='coerce')
        y_test[col]  = pd.to_numeric(y_test[col],  errors='coerce')

    mask_tr        = y_train.notna().all(axis=1)
    X_train_scaled = X_train_scaled[mask_tr.values]
    y_train        = y_train[mask_tr]

    mask_te       = y_test.notna().all(axis=1)
    X_test_scaled = X_test_scaled[mask_te.values]
    y_test        = y_test[mask_te].reset_index(drop=True)
    test_top3     = {rank: encode_top3_slice(data, rank, mask_te.values)
                     for rank, data in top3_test_raw.items()}

    print(f"After cleaning — train: {y_train.shape} | test: {y_test.shape}")
    print(f"Removed {(~mask_tr).sum()} train and {(~mask_te).sum()} test rows with NaN")

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
            'n_estimators':      [200, 400, 600],
            'max_depth':         [15, 25, None],
            'min_samples_split': [2, 10, 20],
            'max_features':      ['sqrt', 0.5, None],
            'class_weight':      ['balanced', 'balanced_subsample', None],
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
        print(f"  Validation score: {(y_val.values == best_model.predict(X_val)).all(axis=1).mean():.4f}")
    else:
        best_model = RandomForestClassifier(
            n_estimators=200, max_depth=25, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt',
            random_state=RANDOM_STATE, n_jobs=-1)
        best_model.fit(X_train_scaled, y_train)

# ==============================================================================
# Predictions
# ==============================================================================
y_pred      = best_model.predict(X_test_scaled)
predictions = {col: y_pred[:, i] for i, col in enumerate(TARGET_COLUMNS)}
num_samples = len(y_pred)

# ==============================================================================
# Evaluation
# ==============================================================================
print(f"\nEVALUATION (6-target)")
print("-" * 40)

print("Individual accuracies:")
for i, col in enumerate(TARGET_COLUMNS):
    print(f"  {col}: {accuracy_score(y_test[col], predictions[col]):.4f}")

# Diagnostic: first 3 samples to confirm tuple consistency
print("\nDiagnostic (first 3 test samples):")
for di in range(min(3, num_samples)):
    p  = tuple(y_pred[di].tolist())
    r  = test_top3['best'].iloc[di]
    b  = (float(r['btbCore0_best']), float(r['btbCore1_best']),
          int(r['prefetcher_core0_best']), int(r['prefetcher_core1_best']),
          int(r['l2_size_best']),          int(r['l3_size_best']))
    yt = tuple(y_test.iloc[di][TARGET_COLUMNS].tolist())
    print(f"  [{di}] pred={p}")
    print(f"       best_cfg={b}  match={p==b}")
    print(f"       y_test  ={yt} match={p==yt}")

exact_matches   = 0
top3_matches    = 0
ppw_diffs       = []
matches_by_rank = {'best': 0, '2nd': 0, '3rd': 0, 'miss': 0}
detailed_results = []

for i in range(num_samples):
    pred = normalize_config(*[predictions[col][i] for col in TARGET_COLUMNS])

    def get_ranked(rank):
        row = test_top3[rank].iloc[i]
        cfg = normalize_config(
            row[f'btbCore0_{rank}'],         row[f'btbCore1_{rank}'],
            row[f'prefetcher_core0_{rank}'], row[f'prefetcher_core1_{rank}'],
            row[f'l2_size_{rank}'],          row[f'l3_size_{rank}'],
        )
        try:
            ppw = float(row[f'PPW_{rank}'])
            ppw = None if np.isnan(ppw) else ppw
        except (ValueError, TypeError):
            ppw = None
        return cfg, ppw

    best_cfg, best_ppw = get_ranked('best')
    cfg_2nd,  ppw_2nd  = get_ranked('2nd')
    cfg_3rd,  ppw_3rd  = get_ranked('3rd')

    actual_configs = [
        {'rank': 'best', 'config': best_cfg, 'ppw': best_ppw},
        {'rank': '2nd',  'config': cfg_2nd,  'ppw': ppw_2nd},
        {'rank': '3rd',  'config': cfg_3rd,  'ppw': ppw_3rd},
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

    if best_ppw is None:
        ppw_diffs.append(None)
    elif exact_match:
        ppw_diffs.append(0.0)
    elif top3_match:
        matched_ppw = actual_configs[matched_rank_idx]['ppw']
        ppw_diffs.append((best_ppw - matched_ppw) / best_ppw * 100
                         if matched_ppw is not None and best_ppw != 0 else None)
    else:
        ppw_diffs.append((best_ppw - ppw_3rd) / best_ppw * 100
                         if ppw_3rd is not None and best_ppw != 0 else None)

    detailed_results.append({
        'sample': i, 'predicted': pred,
        'actual_best': best_cfg, 'actual_2nd': cfg_2nd, 'actual_3rd': cfg_3rd,
        'ppw_best': best_ppw, 'exact_match': exact_match, 'top3_match': top3_match,
    })

exact_accuracy = exact_matches / num_samples
top3_accuracy  = top3_matches  / num_samples

print(f"\nCombined accuracies (all 6 must match):")
print(f"  Exact match (best config): {exact_accuracy:.4f} ({exact_matches}/{num_samples})")
print(f"  Top-3 match:               {top3_accuracy:.4f} ({top3_matches}/{num_samples})")
print(f"  Top-3 uplift:              +{top3_accuracy - exact_accuracy:.4f}")

print(f"\nPredictions by rank:")
for rank in ['best', '2nd', '3rd', 'miss']:
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
# Save model (train path only)
# ==============================================================================
if not args.load_model:
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

    joblib.dump(best_model,               paths['model'])
    joblib.dump(scaler,                   paths['scaler'])
    joblib.dump(imputer,                  paths['imputer'])
    joblib.dump(prefetcher_encoder_core0, paths['enc_pf0'])
    joblib.dump(prefetcher_encoder_core1, paths['enc_pf1'])
    joblib.dump(l2_size_encoder,          paths['enc_l2'])
    joblib.dump(l3_size_encoder,          paths['enc_l3'])

    with open(paths['meta'], 'w') as f:
        f.write(f"RF 6-target Config Predictor\n{'='*60}\n")
        f.write(f"Timestamp: {ts}\nSplit: {SPLIT_METHOD}\nTest size: {TEST_SIZE}\n")
        f.write(f"Train rows: {len(X_train_scaled)}\nTest rows: {num_samples}\n\nModel params:\n")
        for k in ['n_estimators','max_depth','min_samples_split','min_samples_leaf','max_features']:
            f.write(f"  {k}: {getattr(best_model, k)}\n")
        f.write(f"\nAccuracies:\n  Exact: {exact_accuracy:.4f}\n  Top-3: {top3_accuracy:.4f}\n")
        for i, col in enumerate(TARGET_COLUMNS):
            f.write(f"  {col}: {accuracy_score(y_test[col], predictions[col]):.4f}\n")
        if valid_diffs:
            f.write(f"\nPPW: avg={np.mean(valid_diffs):.4f}% median={np.median(valid_diffs):.4f}%\n")

    print(f"\nModel saved as prefix: {model_name}")
    print(f"To rerun evaluation without retraining:")
    print(f"  python train_rf_gpu.py --load-model {model_name}")
