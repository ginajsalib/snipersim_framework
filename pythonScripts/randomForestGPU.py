# Trains one model to predict btbCore0_best, btbCore1_best, prefetcher_core0_best, and prefetcher_core1_best simultaneously.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import joblib
from datetime import datetime
import os
import argparse
warnings.filterwarnings('ignore')

def normalize_config(btb0, btb1, pf0, pf1):
    """
    Normalise a config 4-tuple to (float, float, int, int) so that
    numpy int64 / float64 / Python int / float all compare equal.
    Returns None for any element that cannot be converted.
    """
    try:
        return (float(btb0), float(btb1), int(pf0), int(pf1))
    except (ValueError, TypeError):
        return None

# Check for GPU availability (cuML for GPU-accelerated RF)
USE_GPU = False
try:
    from cuml.ensemble import RandomForestClassifier as cuRF
    import cupy as cp
    USE_GPU = True
    print("GPU detected! Will use cuML for GPU-accelerated training")
except ImportError:
    print("GPU libraries not available. Using CPU (sklearn)")
    print("To enable GPU: pip install cuml-cu11 cupy-cuda11x")

# ==============================================================================
# Command Line Arguments
# ==============================================================================
parser = argparse.ArgumentParser(description='Random Forest Configuration Predictor')
parser.add_argument('--split-method', type=str, default='random',
                    choices=['random', 'benchmark', 'temporal'],
                    help='Method to split train/test data: random, benchmark, or temporal')
parser.add_argument('--test-size', type=float, default=0.2,
                    help='Proportion of data to use for testing (default: 0.2)')
parser.add_argument('--tune', action='store_true', default=True,
                    help='Enable hyperparameter tuning')
parser.add_argument('--no-tune', dest='tune', action='store_false',
                    help='Disable hyperparameter tuning')
args = parser.parse_args()

SPLIT_METHOD = args.split_method
TEST_SIZE = args.test_size
ENABLE_HYPERPARAMETER_TUNING = args.tune

# ==============================================================================
# Data Loading, Merging & Initial Cleaning
# ==============================================================================

csv_filename1 = '/home/gina/Desktop/snipersim_framework/pythonScripts/finalTrainingData/barnes_train_with_top3_fixed.csv'
csv_filename2 = '/home/gina/Desktop/snipersim_framework/pythonScripts/finalTrainingData/cholesky_train_with_top3_fixed.csv'
csv_filename3 = '/home/gina/Desktop/snipersim_framework/pythonScripts/finalTrainingData/fft_train_with_top3.csv'
csv_filename4 = '/home/gina/Desktop/snipersim_framework/pythonScripts/finalTrainingData/radiosity_train_with_top3_fixed.csv'

print("OPTIMIZED Top-3 Random Forest Configuration Predictor (with Per-Core Prefetcher)")
print("=" * 60)
print(f"Split Method: {SPLIT_METHOD}")
print(f"Test Size: {TEST_SIZE}")
print(f"Hyperparameter Tuning: {'Enabled' if ENABLE_HYPERPARAMETER_TUNING else 'Disabled'}")
print("=" * 60)

try:
    df1 = pd.read_csv(csv_filename1)
    df1['benchmark'] = 'barnes'
    df2 = pd.read_csv(csv_filename2)
    df2['benchmark'] = 'fft'
    df3 = pd.read_csv(csv_filename3)
    df3['benchmark'] = 'cholesky'
    df4 = pd.read_csv(csv_filename4)
    df4['benchmark'] = 'radiosity'

    print(f"Data 1 loaded successfully! Shape: {df1.shape}")
    print(f"Data 2 loaded successfully! Shape: {df2.shape}")
    print(f"Data 3 loaded successfully! Shape: {df3.shape}")
    print(f"Data 4 loaded successfully! Shape: {df4.shape}")

#    df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    df = pd.concat([df1, df2, df4], ignore_index=True)
    print("\nDatasets combined successfully!")
    print(f"Combined Data Shape: {df.shape}")
    print(f"Combined Columns: {list(df.columns)}")
    print("\nFirst few rows of the combined data:")
    print(df.head())

except FileNotFoundError:
    print("One or more CSV files not found. Please check the file paths.")
    exit()

# ==============================================================================
# Configuration — 4 targets: btbCore0, btbCore1, prefetcher_core0, prefetcher_core1
# ==============================================================================
TARGET_COLUMNS = [
    'btbCore0_best', 'btbCore1_best',
    'prefetcher_core0_best', 'prefetcher_core1_best'
]

ALL_CONFIG_COLUMNS = [
    # Best
    'btbCore0_best', 'btbCore1_best',
    'prefetcher_core0_best', 'prefetcher_core1_best', 'PPW_best',
    # 2nd
    'btbCore0_2nd', 'btbCore1_2nd',
    'prefetcher_core0_2nd', 'prefetcher_core1_2nd', 'PPW_2nd', 'Diff_best_2nd',
    # 3rd
    'btbCore0_3rd', 'btbCore1_3rd',
    'prefetcher_core0_3rd', 'prefetcher_core1_3rd', 'PPW_3rd', 'Diff_best_3rd'
]

METADATA_COLUMNS_TO_DROP = [
    'best-config', 'file', 'file_prev', 'period_start',
    'period_end', 'period_start_prev', 'period_end_prev',
    'directory_perf_prev', 'leaf_dir_prev', 'directory_power_prev',
    'leaf_dir_perf_prev', 'leaf_dir_power_prev', 'period_start_val_prev',
    'period_end_val_perf_prev', 'period_start_val_perf_prev',
    'period_start_val_power_prev', 'period_end_val_power_prev'
]

SEARCH_TYPE = 'random'
N_ITER_RANDOM_SEARCH = 200
CV_FOLDS = 5
RANDOM_STATE = 42

print(f"\nDropping metadata columns: {METADATA_COLUMNS_TO_DROP}")
df = df.drop(columns=METADATA_COLUMNS_TO_DROP, errors='ignore')
print(f"New data shape: {df.shape}")

missing_cols = [col for col in ALL_CONFIG_COLUMNS if col not in df.columns]
if missing_cols:
    print(f"Missing columns: {missing_cols}")
    print(f"Available columns: {[col for col in df.columns if 'btb' in col.lower() or 'ppw' in col.lower() or 'prefetch' in col.lower()]}")
    exit()
else:
    print("All required columns found!")

# ==============================================================================
# 1. DATA PREPARATION
# ==============================================================================
print(f"\nDATA PREPARATION")
print("-" * 40)

X = df.drop(ALL_CONFIG_COLUMNS, axis=1)
y = df[TARGET_COLUMNS].copy()

# Store top-3 configs (now with per-core prefetcher)
top3_configs = {
    'best': df[['btbCore0_best', 'btbCore1_best',
                'prefetcher_core0_best', 'prefetcher_core1_best', 'PPW_best']].copy(),
    '2nd':  df[['btbCore0_2nd',  'btbCore1_2nd',
                'prefetcher_core0_2nd',  'prefetcher_core1_2nd',  'PPW_2nd']].copy(),
    '3rd':  df[['btbCore0_3rd',  'btbCore1_3rd',
                'prefetcher_core0_3rd',  'prefetcher_core1_3rd',  'PPW_3rd']].copy(),
}

print("\nPreparing PPW lookup from training data...")
df_original_for_lookup = pd.concat([df1, df2, df3, df4], ignore_index=True)
print(df_original_for_lookup.columns.tolist())

lookup_columns = ['benchmark']
if 'BTB core 0_prev' in df_original_for_lookup.columns:
    lookup_columns.extend(['BTB core 0_prev', 'BTB core 1_prev'])
elif 'BTB core 0' in df_original_for_lookup.columns:
    lookup_columns.extend(['BTB core 0', 'BTB core 1'])

# Per-core prefetcher lookup columns
for candidate in ['Prefetch_core0_prev', 'prefetcher_core0_prev', 'Prefetch_prev']:
    if candidate in df_original_for_lookup.columns:
        lookup_columns.append(candidate)
        break
for candidate in ['Prefetch_core1_prev', 'prefetcher_core1_prev']:
    if candidate in df_original_for_lookup.columns:
        lookup_columns.append(candidate)
        break

if 'ppw' in df_original_for_lookup.columns:
    lookup_columns.append('ppw')

print(f"Using columns for lookup: {lookup_columns}")

print(f"Features shape: {X.shape}")
print(f"Targets shape: {y.shape}")
print(f"Target 1 (btbCore0) unique values: {y['btbCore0_best'].nunique()}")
print(f"Target 2 (btbCore1) unique values: {y['btbCore1_best'].nunique()}")
print(f"Target 3 (prefetcher_core0) unique values: {y['prefetcher_core0_best'].nunique()} — {y['prefetcher_core0_best'].unique()}")
print(f"Target 4 (prefetcher_core1) unique values: {y['prefetcher_core1_best'].nunique()} — {y['prefetcher_core1_best'].unique()}")

print(f"\nTarget Distribution Analysis:")
print("-" * 40)
print(f"  btbCore0:         {y['btbCore0_best'].nunique()} unique values")
print(f"  btbCore1:         {y['btbCore1_best'].nunique()} unique values")
print(f"  prefetcherCore0:  {y['prefetcher_core0_best'].nunique()} unique values")
print(f"  prefetcherCore1:  {y['prefetcher_core1_best'].nunique()} unique values")
total_combos = (y['btbCore0_best'].nunique() * y['btbCore1_best'].nunique() *
                y['prefetcher_core0_best'].nunique() * y['prefetcher_core1_best'].nunique())
print(f"  Total possible combinations: {total_combos}")

if 'btbCore0_prev' in X.columns and 'btbCore1_prev' in X.columns:
    same_btb0 = (X['btbCore0_prev'] == y['btbCore0_best']).sum()
    same_btb1 = (X['btbCore1_prev'] == y['btbCore1_best']).sum()
    print(f"\nConfiguration Stability:")
    print(f"  btbCore0 stays same: {same_btb0}/{len(y)} ({same_btb0/len(y)*100:.1f}%)")
    print(f"  btbCore1 stays same: {same_btb1}/{len(y)} ({same_btb1/len(y)*100:.1f}%)")

# ==============================================================================
# 2. FEATURE PREPROCESSING
# ==============================================================================
print(f"\nFEATURE PREPROCESSING")
print("-" * 40)

benchmark_original = None
if 'benchmark' in X.columns:
    benchmark_original = X['benchmark'].copy()
    print(f"Saved original benchmark column for splitting")
    print(f"Unique benchmarks: {benchmark_original.unique()}")

categorical_cols = X.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded: {col}")

# Encode per-core prefetcher targets — use a shared encoder so classes are consistent
prefetcher_encoder_core0 = LabelEncoder()
prefetcher_encoder_core1 = LabelEncoder()

y['prefetcher_core0_best'] = prefetcher_encoder_core0.fit_transform(
    y['prefetcher_core0_best'].astype(str))
print(f"\nEncoded prefetcher_core0_best: "
      f"{dict(zip(prefetcher_encoder_core0.classes_, prefetcher_encoder_core0.transform(prefetcher_encoder_core0.classes_)))}")

y['prefetcher_core1_best'] = prefetcher_encoder_core1.fit_transform(
    y['prefetcher_core1_best'].astype(str))
print(f"Encoded prefetcher_core1_best: "
      f"{dict(zip(prefetcher_encoder_core1.classes_, prefetcher_encoder_core1.transform(prefetcher_encoder_core1.classes_)))}")

numeric_cols = X.select_dtypes(include=[np.number]).columns
X = X[numeric_cols]

print(f"Final feature set: {X.shape}")
print(f"\nFirst 20 feature columns:")
for i, col in enumerate(X.columns):
    if i < 20:
        print(f"  {i+1}. {col}")
if len(X.columns) > 20:
    print(f"  ... and {len(X.columns) - 20} more")

# ==============================================================================
# 3. TRAIN-TEST SPLIT
# ==============================================================================
print(f"\nTRAIN-TEST SPLIT")
print("-" * 40)
print(f"Split method: {SPLIT_METHOD}")

if SPLIT_METHOD == 'benchmark':
    if benchmark_original is None:
        print("ERROR: 'benchmark' column not found. Falling back to random split...")
        SPLIT_METHOD = 'random'
    else:
        benchmarks = benchmark_original.unique()
        print(f"Found {len(benchmarks)} unique benchmarks: {benchmarks}")

        if len(benchmarks) < 2:
            print(f"WARNING: Only {len(benchmarks)} benchmark(s). Falling back to random split...")
            SPLIT_METHOD = 'random'
        else:
            n_test_benchmarks = max(1, int(len(benchmarks) * TEST_SIZE))
            np.random.seed(RANDOM_STATE)
            test_benchmarks_selected = np.random.choice(benchmarks, n_test_benchmarks, replace=False)

            print(f"Test benchmarks ({n_test_benchmarks}): {test_benchmarks_selected}")
            print(f"Train benchmarks: {[b for b in benchmarks if b not in test_benchmarks_selected]}")

            test_mask = benchmark_original.isin(test_benchmarks_selected)
            X_train = X[~test_mask].copy()
            X_test  = X[test_mask].copy()
            y_train = y[~test_mask].copy()
            y_test  = y[test_mask].copy()

            benchmark_train = benchmark_original[~test_mask]
            benchmark_test  = benchmark_original[test_mask]

            print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

elif SPLIT_METHOD == 'temporal':
    print("Using temporal split (chronological order)")
    split_point = int(len(X) * (1 - TEST_SIZE))
    X_train = X.iloc[:split_point].copy()
    X_test  = X.iloc[split_point:].copy()
    y_train = y.iloc[:split_point].copy()
    y_test  = y.iloc[split_point:].copy()
    print(f"Train: first {len(X_train)}, Test: last {len(X_test)}")

else:  # random
    print("Using random stratified split")
    stratify_key = (y[TARGET_COLUMNS[0]].astype(str) + '_' +
                    y[TARGET_COLUMNS[1]].astype(str) + '_' +
                    y[TARGET_COLUMNS[2]].astype(str) + '_' +
                    y[TARGET_COLUMNS[3]].astype(str))
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        for train_idx, test_idx in sss.split(X, stratify_key):
            X_train = X.iloc[train_idx].copy()
            X_test  = X.iloc[test_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_test  = y.iloc[test_idx].copy()
        print("Using stratified split to balance target distribution")
    except Exception as e:
        print(f"Stratification failed ({e}), using simple random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

# Get corresponding top-3 data for the test set
test_indices = X_test.index
test_top3 = {}
for rank, data in top3_configs.items():
    test_top3[rank] = data.loc[test_indices].copy()

if benchmark_original is not None:
    test_benchmarks = benchmark_original[test_indices].copy()
else:
    test_benchmarks = None

# ==============================================================================
# Build PPW lookup table: (benchmark, btb0, btb1, prefetch_core0, prefetch_core1) -> ppw
# ==============================================================================
print("\nBuilding PPW lookup table from training data...")
print(f"Available columns in original data: {df_original_for_lookup.columns.tolist()[:20]}")

btb0_col = btb1_col = prefetch_core0_col = prefetch_core1_col = ppw_col = bench_col = None

for col in ['btbCore0', 'BTB core 0_prev', 'BTB_core_0']:
    if col in df_original_for_lookup.columns:
        btb0_col = col
        btb1_col = col.replace('0', '1')
        break

for col in ['prefetcher_core0_prev', 'Prefetch_core0_prev', 'Prefetch_prev']:
    if col in df_original_for_lookup.columns:
        prefetch_core0_col = col
        break

for col in ['prefetcher_core1_prev', 'Prefetch_core1_prev']:
    if col in df_original_for_lookup.columns:
        prefetch_core1_col = col
        break

for col in ['ppw_prev', 'ppw', 'PPW']:
    if col in df_original_for_lookup.columns:
        ppw_col = col
        break

for col in ['benchmark', 'Benchmark']:
    if col in df_original_for_lookup.columns:
        bench_col = col
        break

print(f"Identified columns: btb0={btb0_col}, btb1={btb1_col}, "
      f"prefetch_core0={prefetch_core0_col}, prefetch_core1={prefetch_core1_col}, "
      f"ppw={ppw_col}, bench={bench_col}")

ppw_lookup = {}

if all(c is not None for c in [btb0_col, btb1_col, ppw_col, bench_col]):
    for _, row in df_original_for_lookup.iterrows():
        bench  = row[bench_col]
        btb0   = row[btb0_col]
        btb1   = row[btb1_col]
        ppw    = row[ppw_col]

        pf0 = str(row[prefetch_core0_col]).lower().strip() if prefetch_core0_col and pd.notna(row.get(prefetch_core0_col)) else 'none'
        pf1 = str(row[prefetch_core1_col]).lower().strip() if prefetch_core1_col and pd.notna(row.get(prefetch_core1_col)) else 'none'

        if pd.notna(btb0) and pd.notna(btb1) and pd.notna(ppw):
            try:
                key = (str(bench), float(btb0), float(btb1), pf0, pf1)
                ppw_lookup[key] = (ppw_lookup[key] + float(ppw)) / 2 if key in ppw_lookup else float(ppw)
            except (ValueError, TypeError):
                pass

    print(f"PPW lookup table created with {len(ppw_lookup)} entries")
    if ppw_lookup:
        print("Sample entries:")
        for k, v in list(ppw_lookup.items())[:3]:
            print(f"  {k} -> {v}")
else:
    ppw_lookup = None
    print("Warning: Cannot create PPW lookup table — missing required columns")

# ==============================================================================
# Scale & impute features
# ==============================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

feature_names_for_model = X_train.columns.tolist()
print(f"\nFeatures being used in model: {len(feature_names_for_model)}")

print(f"\nCleaning data...")
imputer = SimpleImputer(strategy='mean')
X_train_scaled = imputer.fit_transform(X_train_scaled)
X_test_scaled  = imputer.transform(X_test_scaled)

print(f"Before cleaning — y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

for col in ['btbCore0_best', 'btbCore1_best']:
    y_train[col] = pd.to_numeric(y_train[col], errors='coerce')
    y_test[col]  = pd.to_numeric(y_test[col],  errors='coerce')

mask_train = y_train.notna().all(axis=1)
y_train        = y_train[mask_train]
X_train_scaled = X_train_scaled[mask_train.values]

mask_test     = y_test.notna().all(axis=1)
y_test        = y_test[mask_test]
X_test_scaled = X_test_scaled[mask_test.values]

# Update test_top3 to match cleaned test set
def _safe_prefetcher_transform(encoder, series):
    """Encode a prefetcher Series, mapping unseen labels to 'none'."""
    valid = set(encoder.classes_)
    series = series.fillna('none').astype(str).replace('nan', 'none')
    series = series.apply(lambda x: x if x in valid else 'none')
    return encoder.transform(series)

test_top3_cleaned = {}
for rank, data in test_top3.items():
    cleaned = data[mask_test.values].reset_index(drop=True)

    for col in [f'btbCore0_{rank}', f'btbCore1_{rank}']:
        cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')

    cleaned[f'prefetcher_core0_{rank}'] = _safe_prefetcher_transform(
        prefetcher_encoder_core0, cleaned[f'prefetcher_core0_{rank}'])
    cleaned[f'prefetcher_core1_{rank}'] = _safe_prefetcher_transform(
        prefetcher_encoder_core1, cleaned[f'prefetcher_core1_{rank}'])

    cleaned[f'PPW_{rank}'] = pd.to_numeric(cleaned[f'PPW_{rank}'], errors='coerce')
    test_top3_cleaned[rank] = cleaned

test_top3 = test_top3_cleaned
y_test = y_test.reset_index(drop=True)

print(f"After cleaning — y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
print(f"After cleaning — test_top3['best'] shape: {test_top3['best'].shape}")
print(f"Removed {(~mask_train).sum()} training and {(~mask_test).sum()} test samples with NaN")

# ==============================================================================
# 5. HYPERPARAMETER TUNING AND MODEL TRAINING
# ==============================================================================
print(f"\nOPTIMIZED MODEL TRAINING (Single Multi-Output for 4 targets)")
print("-" * 40)

if ENABLE_HYPERPARAMETER_TUNING:
    print(f"Advanced Hyperparameter Optimization...")

    y_train_combined = (y_train[TARGET_COLUMNS[0]].astype(str) + '_' +
                        y_train[TARGET_COLUMNS[1]].astype(str) + '_' +
                        y_train[TARGET_COLUMNS[2]].astype(str) + '_' +
                        y_train[TARGET_COLUMNS[3]].astype(str))

    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.6, random_state=RANDOM_STATE)
    for tune_idx, val_idx in sss.split(X_train_scaled, y_train_combined):
        X_tune = X_train_scaled[tune_idx]
        y_tune = y_train.iloc[tune_idx]
        X_val  = X_train_scaled[val_idx]
        y_val  = y_train.iloc[val_idx]

    print(f"Tuning on {len(X_tune)} samples ({len(X_tune)/len(X_train_scaled)*100:.1f}% of training data)")

    def exact_match_scorer(estimator, X, y_true):
        y_pred = estimator.predict(X)
        return (y_true.values == y_pred).all(axis=1).mean()

    coarse_params = {
        'n_estimators': [200, 400, 600],
        'max_depth': [15, 25, None],
        'min_samples_split': [2, 10, 20],
        'max_features': ['sqrt', 0.5, None],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }

    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')

    print("\nStage 1: Coarse search...")
    coarse_search = RandomizedSearchCV(
        rf, coarse_params,
        cv=3, n_iter=20,
        scoring=exact_match_scorer,
        n_jobs=-1, random_state=RANDOM_STATE, verbose=1
    )
    coarse_search.fit(X_tune, y_tune)

    print(f"Coarse best score: {coarse_search.best_score_:.4f}")
    print(f"Coarse best params: {coarse_search.best_params_}")

    print("\nStage 2: Fine-tuning...")
    best_coarse = coarse_search.best_params_

    def _adj(val, delta, lo=5):
        if val is None:
            return [None]
        return [max(lo, val - delta), val, val + delta]

    fine_params = {
        'n_estimators': [max(50, best_coarse['n_estimators'] - 50),
                         best_coarse['n_estimators'],
                         best_coarse['n_estimators'] + 50],
        'max_depth': _adj(best_coarse['max_depth'], 3),
        'min_samples_split': _adj(best_coarse['min_samples_split'], 5),
        'min_samples_leaf': [max(2, best_coarse.get('min_samples_leaf', 5) - 3),
                             best_coarse.get('min_samples_leaf', 5),
                             best_coarse.get('min_samples_leaf', 5) + 3],
        'max_features': [best_coarse['max_features']],
        'class_weight': ['balanced', None],
        'bootstrap': [True]
    }

    fine_search = RandomizedSearchCV(
        rf, fine_params,
        cv=2, n_iter=15,
        scoring=exact_match_scorer,
        n_jobs=-1, random_state=RANDOM_STATE, verbose=1
    )
    fine_search.fit(X_tune, y_tune)

    best_model = fine_search.best_estimator_
    print(f"\nFinal best parameters: {fine_search.best_params_}")
    print(f"Final best CV score: {fine_search.best_score_:.4f}")

    val_pred  = best_model.predict(X_val)
    val_score = (y_val.values == val_pred).all(axis=1).mean()
    print(f"Validation set score: {val_score:.4f}")

else:
    print("Using default parameters with regularization...")
    best_model = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE,
        max_depth=25, min_samples_split=5,
        min_samples_leaf=2, max_features='sqrt', n_jobs=-1
    )
    best_model.fit(X_train_scaled, y_train)

# Predictions — 4 output columns
y_pred = best_model.predict(X_test_scaled)
predictions = {
    TARGET_COLUMNS[0]: y_pred[:, 0],
    TARGET_COLUMNS[1]: y_pred[:, 1],
    TARGET_COLUMNS[2]: y_pred[:, 2],
    TARGET_COLUMNS[3]: y_pred[:, 3],
}

# ==============================================================================
# 6. TOP-K ACCURACY EVALUATION (4-way config: btb0, btb1, pf_core0, pf_core1)
# ==============================================================================
print(f"\nENHANCED TOP-K ACCURACY EVALUATION (4-way config)")
print("-" * 40)
 
print(f"Debug: ppw_lookup available: {ppw_lookup is not None}")
if ppw_lookup:
    print(f"Debug: ppw_lookup size: {len(ppw_lookup)}")
    print(f"Debug: Sample lookup keys: {list(ppw_lookup.keys())[:3]}")
print(f"Debug: test_benchmarks available: {test_benchmarks is not None}")
 
# ── Normalise all predictions once up-front ───────────────────────────────────
norm_predictions = [
    normalize_config(
        predictions[TARGET_COLUMNS[0]][i],
        predictions[TARGET_COLUMNS[1]][i],
        predictions[TARGET_COLUMNS[2]][i],
        predictions[TARGET_COLUMNS[3]][i],
    )
    for i in range(len(y_pred))
]
 
# ── Config frequency (decoded for readability) ────────────────────────────────
config_frequency = Counter(norm_predictions)
print(f"Most frequently predicted configurations (btbCore0, btbCore1, pf_core0, pf_core1):")
for config, count in config_frequency.most_common(10):
    if config is None:
        continue
    try:
        pf0_dec = prefetcher_encoder_core0.inverse_transform([config[2]])[0]
        pf1_dec = prefetcher_encoder_core1.inverse_transform([config[3]])[0]
    except Exception:
        pf0_dec, pf1_dec = config[2], config[3]
    print(f"  ({config[0]}, {config[1]}, {pf0_dec}, {pf1_dec}): "
          f"{count} times ({count/len(y_pred)*100:.1f}%)")
 
# ── Main evaluation loop ──────────────────────────────────────────────────────
exact_matches     = 0
top3_matches      = 0
ppw_diffs         = []          # one entry per sample, None if no PPW data
ppw_costs         = {'exact': [], 'top3_miss': [], 'top3_hit': []}
ppw_lookup_hits   = 0
ppw_lookup_misses = 0
detailed_results  = []
 
num_samples = len(y_pred)
 
for i in range(num_samples):
    pred_config = norm_predictions[i]
 
    # ── Build actual top-3 normalised configs + PPW values ────────────────────
    actual_configs = []
    for rank in ['best', '2nd', '3rd']:
        r_btb0 = test_top3[rank].iloc[i][f'btbCore0_{rank}']
        r_btb1 = test_top3[rank].iloc[i][f'btbCore1_{rank}']
        r_pf0  = test_top3[rank].iloc[i][f'prefetcher_core0_{rank}']
        r_pf1  = test_top3[rank].iloc[i][f'prefetcher_core1_{rank}']
        r_ppw  = test_top3[rank].iloc[i][f'PPW_{rank}']
 
        try:
            r_ppw = float(r_ppw)
            if np.isnan(r_ppw):
                r_ppw = None
        except (ValueError, TypeError):
            r_ppw = None
 
        actual_configs.append({
            'rank':   rank,
            'config': normalize_config(r_btb0, r_btb1, r_pf0, r_pf1),
            'ppw':    r_ppw,
        })
 
    best_ppw = actual_configs[0]['ppw']   # None if not available
 
    # ── Match logic ───────────────────────────────────────────────────────────
    # FIX: compare normalised tuples; exact match → always also a top-3 match
    exact_match = (pred_config is not None and
                   pred_config == actual_configs[0]['config'])
 
    top3_match  = exact_match   # exact is a subset of top-3 by definition
    matched_rank_idx = 0        # index into actual_configs (0=best,1=2nd,2=3rd)
 
    if not exact_match:
        for idx, cfg in enumerate(actual_configs):
            if pred_config is not None and pred_config == cfg['config']:
                top3_match       = True
                matched_rank_idx = idx
                break
 
    # ── Counters ──────────────────────────────────────────────────────────────
    if exact_match:
        exact_matches += 1
 
    if top3_match:
        top3_matches += 1
 
    # ── PPW difference (Best PPW − Predicted PPW, so >0 means we lost perf) ──
    if best_ppw is None:
        # No PPW data for this sample at all
        ppw_diffs.append(None)
 
    elif exact_match:
        # Perfect prediction → zero loss
        ppw_costs['exact'].append(best_ppw)
        ppw_diffs.append(0.0)
 
    elif top3_match:
        # Predicted 2nd or 3rd best config
        matched_ppw = actual_configs[matched_rank_idx]['ppw']
        if matched_ppw is not None and best_ppw != 0:
            loss_pct = (best_ppw - matched_ppw) / best_ppw * 100   # % below best
            ppw_costs['top3_hit'].append(matched_ppw)
            ppw_diffs.append(loss_pct)
        else:
            ppw_diffs.append(None)
 
    else:
        # Complete miss — try lookup table first, fall back to PPW_3rd estimate
        ppw_costs['top3_miss'].append(best_ppw)
        predicted_ppw = None
 
        if ppw_lookup is not None and test_benchmarks is not None and i < len(test_benchmarks):
            bench = test_benchmarks.iloc[i]
            try:
                pred_pf0_str = prefetcher_encoder_core0.inverse_transform(
                    [int(predictions[TARGET_COLUMNS[2]][i])])[0]
            except Exception:
                pred_pf0_str = 'none'
            try:
                pred_pf1_str = prefetcher_encoder_core1.inverse_transform(
                    [int(predictions[TARGET_COLUMNS[3]][i])])[0]
            except Exception:
                pred_pf1_str = 'none'
 
            lookup_key = (
                str(bench),
                float(predictions[TARGET_COLUMNS[0]][i]),
                float(predictions[TARGET_COLUMNS[1]][i]),
                str(pred_pf0_str).lower().strip(),
                str(pred_pf1_str).lower().strip(),
            )
 
            # Debug first 3 misses only
            if ppw_lookup_misses < 3:
                print(f"\n=== Debug miss {ppw_lookup_misses + 1} ===")
                print(f"  Lookup key: {lookup_key}")
                print(f"  Key in lookup: {lookup_key in ppw_lookup}")
 
            if lookup_key in ppw_lookup:
                predicted_ppw = ppw_lookup[lookup_key]
                ppw_lookup_hits += 1
            else:
                ppw_lookup_misses += 1
        else:
            ppw_lookup_misses += 1
 
        if predicted_ppw is not None and best_ppw != 0:
            ppw_diffs.append((best_ppw - predicted_ppw) / best_ppw * 100)
        elif actual_configs[2]['ppw'] is not None and best_ppw != 0:
            # Conservative estimate: assume we got 3rd best
            ppw_diffs.append((best_ppw - actual_configs[2]['ppw']) / best_ppw * 100)
        else:
            ppw_diffs.append(None)
 
    detailed_results.append({
        'sample':       i,
        'predicted':    pred_config,
        'actual_best':  actual_configs[0]['config'],
        'actual_2nd':   actual_configs[1]['config'],
        'actual_3rd':   actual_configs[2]['config'],
        'ppw_best':     best_ppw,
        'ppw_2nd':      actual_configs[1]['ppw'],
        'ppw_3rd':      actual_configs[2]['ppw'],
        'exact_match':  exact_match,
        'top3_match':   top3_match,
    })
 
# ── Accuracy reporting ────────────────────────────────────────────────────────
exact_accuracy = exact_matches / num_samples
top3_accuracy  = top3_matches  / num_samples
 
assert top3_accuracy >= exact_accuracy, \
    f"BUG: top3={top3_accuracy:.4f} < exact={exact_accuracy:.4f}"
 
print(f"\nOPTIMIZED MODEL RESULTS:")
print(f"Individual Accuracies:")
acc_0 = accuracy_score(y_test[TARGET_COLUMNS[0]], predictions[TARGET_COLUMNS[0]])
acc_1 = accuracy_score(y_test[TARGET_COLUMNS[1]], predictions[TARGET_COLUMNS[1]])
acc_2 = accuracy_score(y_test[TARGET_COLUMNS[2]], predictions[TARGET_COLUMNS[2]])
acc_3 = accuracy_score(y_test[TARGET_COLUMNS[3]], predictions[TARGET_COLUMNS[3]])
print(f"  {TARGET_COLUMNS[0]} accuracy: {acc_0:.4f}")
print(f"  {TARGET_COLUMNS[1]} accuracy: {acc_1:.4f}")
print(f"  {TARGET_COLUMNS[2]} accuracy: {acc_2:.4f}")
print(f"  {TARGET_COLUMNS[3]} accuracy: {acc_3:.4f}")
 
exact_match_numpy = (y_test.values == y_pred).all(axis=1)
print(f"\nExact Match Debug:")
print(f"  Via numpy (raw types): {exact_match_numpy.sum()} | "
      f"Via loop (normalised): {exact_matches}")
if exact_match_numpy.sum() != exact_matches:
    print(f"  [NOTE] Difference = {abs(exact_match_numpy.sum() - exact_matches)} "
          f"— caused by type coercion (float vs int); normalised loop is correct.")
 
print(f"\nCombined Accuracies (all 4 must match):")
print(f"  Exact Match (Best Config):  {exact_accuracy:.4f} ({exact_matches}/{num_samples})")
print(f"  Top-3 Match (Any of 3):     {top3_accuracy:.4f} ({top3_matches}/{num_samples})")
improvement = top3_accuracy - exact_accuracy
print(f"  Top-3 Uplift:               +{improvement:.4f} "
      f"(+{top3_matches - exact_matches} more correct vs exact)")
 
# ── PPW difference analysis ───────────────────────────────────────────────────
print(f"\nPPW DIFFERENCE ANALYSIS:")
print("-" * 40)
 
matches_by_rank = {'best': 0, '2nd': 0, '3rd': 0, 'miss': 0}
for result in detailed_results:
    if result['exact_match']:
        matches_by_rank['best'] += 1
    elif result['top3_match']:
        if result['predicted'] == result['actual_2nd']:
            matches_by_rank['2nd'] += 1
        else:
            matches_by_rank['3rd'] += 1
    else:
        matches_by_rank['miss'] += 1
 
print(f"Predictions by rank:")
for rank in ['best', '2nd', '3rd', 'miss']:
    n = matches_by_rank[rank]
    label = 'No match (miss)' if rank == 'miss' else f'{rank} config predicted'
    print(f"  {label:<26} {n:>6}  ({n/num_samples*100:.2f}%)")
 
# Split ppw_diffs by outcome
correct_diffs   = []
near_miss_diffs = []
miss_diffs      = []
 
for i, result in enumerate(detailed_results):
    d = ppw_diffs[i]
    if d is None or np.isnan(d):
        continue
    if result['exact_match']:
        correct_diffs.append(d)
    elif result['top3_match']:
        near_miss_diffs.append(d)
    else:
        miss_diffs.append(d)
 
print(f"\nPPW Loss Analysis (% below best config):")
print(f"  Exact matches (0% loss): {len(correct_diffs)} samples")
print(f"  Top-3 near misses:       {len(near_miss_diffs)} samples")
if near_miss_diffs:
    print(f"    Avg loss:    {np.mean(near_miss_diffs):.2f}%")
    print(f"    Median loss: {np.median(near_miss_diffs):.2f}%")
    print(f"    Max loss:    {np.max(near_miss_diffs):.2f}%")
print(f"  Complete misses:         {len(miss_diffs)} samples")
if miss_diffs:
    print(f"    Avg estimated loss:    {np.mean(miss_diffs):.2f}%")
    print(f"    Median estimated loss: {np.median(miss_diffs):.2f}%")
    print(f"    Max estimated loss:    {np.max(miss_diffs):.2f}%")
    print(f"    PPW lookup hits:       {ppw_lookup_hits}")
    print(f"    PPW lookup misses:     {ppw_lookup_misses} (used 3rd-config estimate)")
 
valid_ppw_diffs = [d for d in ppw_diffs if d is not None and not np.isnan(d)]
 
if valid_ppw_diffs:
    avg_ppw_diff    = np.mean(valid_ppw_diffs)
    median_ppw_diff = np.median(valid_ppw_diffs)
    max_ppw_diff    = np.max(valid_ppw_diffs)
    p90_ppw_diff    = np.percentile(valid_ppw_diffs, 90)
    p95_ppw_diff    = np.percentile(valid_ppw_diffs, 95)
 
    print(f"\nOverall PPW Loss Statistics (% below best config):")
    print(f"  Average loss:   {avg_ppw_diff:.2f}%")
    print(f"  Median loss:    {median_ppw_diff:.2f}%")
    print(f"  90th pct loss:  {p90_ppw_diff:.2f}%")
    print(f"  95th pct loss:  {p95_ppw_diff:.2f}%")
    print(f"  Worst loss:     {max_ppw_diff:.2f}%")
    print(f"  Samples with data: {len(valid_ppw_diffs)}/{num_samples} "
          f"({len(valid_ppw_diffs)/num_samples*100:.1f}%)")
 
    # Breakdown: what % of samples fall into loss buckets
    buckets = [0, 1, 5, 10, 20, float('inf')]
    labels  = ['0% (exact)', '0–1%', '1–5%', '5–10%', '10–20%', '>20%']
    print(f"\n  Loss distribution:")
    for lo, hi, label in zip(buckets, buckets[1:], labels[1:]):
        if lo == 0:
            n = sum(1 for d in valid_ppw_diffs if d == 0.0)
            print(f"    {labels[0]:<14} {n:>6} samples  ({n/num_samples*100:.1f}%)")
        n = sum(1 for d in valid_ppw_diffs if lo < d <= hi)
        print(f"    {label:<14} {n:>6} samples  ({n/num_samples*100:.1f}%)")
 
    samples_with_ppw = [r for r in detailed_results if r['ppw_best'] is not None]
    if samples_with_ppw:
        avg_best_ppw = np.mean([r['ppw_best'] for r in samples_with_ppw])
        predicted_ppws = []
        for i, r in enumerate(samples_with_ppw):
            orig_i = r['sample']
            d = ppw_diffs[orig_i]
            if d is not None and not np.isnan(d):
                predicted_ppws.append(r['ppw_best'] * (1 - d / 100))
        if predicted_ppws:
            avg_predicted_ppw = np.mean(predicted_ppws)
            overall_loss_pct  = (avg_best_ppw - avg_predicted_ppw) / avg_best_ppw * 100
            print(f"\nOverall Performance Impact (samples with PPW data):")
            print(f"  Avg best config PPW:      {avg_best_ppw:.2f}")
            print(f"  Avg predicted config PPW: {avg_predicted_ppw:.2f}")
            print(f"  Weighted PPW loss:        {overall_loss_pct:.2f}%")
            print(f"  NOTE: misses ({matches_by_rank['miss']} samples) use "
                  f"3rd-config PPW as conservative estimate.")
else:
    avg_ppw_diff = median_ppw_diff = None
    print("No valid PPW difference data available")
 

# ==============================================================================
# 7. FEATURE IMPORTANCE ANALYSIS
# ==============================================================================
print(f"\nFEATURE IMPORTANCE ANALYSIS")
print("-" * 40)

feature_names = X_train.columns.tolist()
if len(feature_names) != len(best_model.feature_importances_):
    print(f"WARNING: Feature count mismatch — using indices")
    feature_names = [f"feature_{i}" for i in range(len(best_model.feature_importances_))]

feature_importance_df = pd.DataFrame({
    'feature':    feature_names,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 15 Most Important Features:")
print(feature_importance_df.head(15).to_string(index=False))

# ==============================================================================
# 8. HYPERPARAMETER SUMMARY
# ==============================================================================
if ENABLE_HYPERPARAMETER_TUNING:
    print(f"\nOPTIMAL HYPERPARAMETERS FOUND")
    print("-" * 40)
    print(f"  n_estimators:      {best_model.n_estimators}")
    print(f"  max_depth:         {best_model.max_depth}")
    print(f"  min_samples_split: {best_model.min_samples_split}")
    print(f"  min_samples_leaf:  {best_model.min_samples_leaf}")
    print(f"  max_features:      {best_model.max_features}")
    if hasattr(best_model, 'bootstrap'):
        print(f"  bootstrap:         {best_model.bootstrap}")

print(f"\nOPTIMIZED ANALYSIS COMPLETE!")
print(f"Key Results:")
print(f"  Exact match accuracy (all 4 params): {exact_accuracy:.4f}")
print(f"  Top-3 match accuracy:                {top3_accuracy:.4f}")
print(f"  Improvement: +{(top3_accuracy - exact_accuracy):.4f} "
      f"({top3_matches - exact_matches} more correct)")
if valid_ppw_diffs:
    print(f"  Average PPW loss: {avg_ppw_diff:.4f}")

# ==============================================================================
# 9. MODEL PERSISTENCE
# ==============================================================================
print(f"\nMODEL PERSISTENCE")
print("-" * 40)

os.makedirs('saved_models', exist_ok=True)
timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"rf_4way_config_predictor_{timestamp}"

model_path    = f"saved_models/{model_name}.pkl"
scaler_path   = f"saved_models/{model_name}_scaler.pkl"
enc0_path     = f"saved_models/{model_name}_prefetcher_core0_encoder.pkl"
enc1_path     = f"saved_models/{model_name}_prefetcher_core1_encoder.pkl"
metadata_path = f"saved_models/{model_name}_metadata.txt"

joblib.dump(best_model,             model_path)
joblib.dump(scaler,                 scaler_path)
joblib.dump(prefetcher_encoder_core0, enc0_path)
joblib.dump(prefetcher_encoder_core1, enc1_path)

print(f"Model saved:             {model_path}")
print(f"Scaler saved:            {scaler_path}")
print(f"Core-0 encoder saved:    {enc0_path}")
print(f"Core-1 encoder saved:    {enc1_path}")

with open(metadata_path, 'w') as f:
    f.write("Random Forest Configuration Predictor (4-way per-core)\n")
    f.write("=" * 60 + "\n")
    f.write(f"Timestamp:        {timestamp}\n")
    f.write(f"Split Method:     {SPLIT_METHOD}\n")
    f.write(f"Test Size:        {TEST_SIZE}\n")
    f.write(f"Training samples: {len(X_train_scaled)}\n")
    f.write(f"Test samples:     {num_samples}\n")
    f.write(f"\nModel Parameters:\n")
    f.write(f"  n_estimators:      {best_model.n_estimators}\n")
    f.write(f"  max_depth:         {best_model.max_depth}\n")
    f.write(f"  min_samples_split: {best_model.min_samples_split}\n")
    f.write(f"  min_samples_leaf:  {best_model.min_samples_leaf}\n")
    f.write(f"  max_features:      {best_model.max_features}\n")
    f.write(f"\nPerformance Metrics:\n")
    f.write(f"  Exact match accuracy: {exact_accuracy:.4f}\n")
    f.write(f"  Top-3 match accuracy: {top3_accuracy:.4f}\n")
    f.write(f"  btbCore0 accuracy:           {acc_0:.4f}\n")
    f.write(f"  btbCore1 accuracy:           {acc_1:.4f}\n")
    f.write(f"  prefetcher_core0 accuracy:   {acc_2:.4f}\n")
    f.write(f"  prefetcher_core1 accuracy:   {acc_3:.4f}\n")
    if valid_ppw_diffs:
        f.write(f"  Average PPW difference: {avg_ppw_diff:.4f}\n")
        f.write(f"  Median PPW difference:  {median_ppw_diff:.4f}\n")

print(f"Metadata saved:          {metadata_path}")

print("\nTo load the model:")
print(f"  import joblib")
print(f"  model    = joblib.load('{model_path}')")
print(f"  scaler   = joblib.load('{scaler_path}')")
print(f"  enc_pf0  = joblib.load('{enc0_path}')")
print(f"  enc_pf1  = joblib.load('{enc1_path}')")
