# OPTIMIZED TOP-3 CONFIGURATION PREDICTOR WITH A SINGLE MULTI-OUTPUT MODEL
# Trains one model to predict btbCore0_best, btbCore1_best, and prefetcher_best simultaneously.

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

# For local testing, specify your CSV file paths
csv_filename1 = 'train_with_top3_barnes_merged_prefetcher_combined.csv'
csv_filename2 = 'train_with_top3_cholesky_merged_prefetcher_combined.csv'
csv_filename3 = 'train_with_top3_fft_merged_prefetcher_combined.csv'
csv_filename4 = 'train_with_top3_radiosityy_merged_prefetcher_combined.csv'

print("OPTIMIZED Top-3 Random Forest Configuration Predictor (with Prefetcher)")
print("=" * 60)
print(f"Split Method: {SPLIT_METHOD}")
print(f"Test Size: {TEST_SIZE}")
print(f"Hyperparameter Tuning: {'Enabled' if ENABLE_HYPERPARAMETER_TUNING else 'Disabled'}")
print("=" * 60)

# Load the datasets
try:
    df1 = pd.read_csv(csv_filename1)
    df2 = pd.read_csv(csv_filename2)
    df3 = pd.read_csv(csv_filename3)
    df4 = pd.read_csv(csv_filename4)
    
    print(f"Data 1 loaded successfully! Shape: {df1.shape}")
    print(f"Data 2 loaded successfully! Shape: {df2.shape}")
    print(f"Data 3 loaded successfully! Shape: {df3.shape}")
    print(f"Data 4 loaded successfully! Shape: {df4.shape}")

    # Combine the dataframes
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)

    print("\nDatasets combined successfully!")
    print(f"Combined Data Shape: {df.shape}")
    print(f"Combined Columns: {list(df.columns)}")
    print("\nFirst few rows of the combined data:")
    print(df.head())

except FileNotFoundError:
    print("One or more CSV files not found. Please check the file paths.")
    exit()

# Configuration - NOW INCLUDING PREFETCHER
TARGET_COLUMNS = ['btbCore0_best', 'btbCore1_best', 'prefetcher_best']  # What we're trying to predict
ALL_CONFIG_COLUMNS = [
    'btbCore0_best', 'btbCore1_best', 'prefetcher_best', 'PPW_best',
    'btbCore0_2nd', 'btbCore1_2nd', 'prefetcher_2nd', 'PPW_2nd', 'Diff_best_2nd',
    'btbCore0_3rd', 'btbCore1_3rd', 'prefetcher_3rd', 'PPW_3rd', 'Diff_best_3rd'
]

# Columns to drop
METADATA_COLUMNS_TO_DROP = ['best-config', 'file', 'file_prev', 'period_start',
                            'period_end', 'period_start_prev', 'period_end_prev',
                            'directory_perf_prev', 'leaf_dir_prev', 'directory_power_prev',
                            'leaf_dir_perf_prev', 'leaf_dir_power_prev', 'period_start_val_prev', 
                            'period_end_val_perf_prev', 'period_start_val_perf_prev', 'period_start_val_power_prev',
                            'period_end_val_power_prev']

# Hyperparameter tuning options
SEARCH_TYPE = 'random'  # 'grid' or 'random'
N_ITER_RANDOM_SEARCH = 50  # Number of iterations for random search
CV_FOLDS = 3
RANDOM_STATE = 42

# Drop the specified metadata columns
print(f"\nDropping metadata columns: {METADATA_COLUMNS_TO_DROP}")
df = df.drop(columns=METADATA_COLUMNS_TO_DROP, errors='ignore')
print(f"New data shape: {df.shape}")

# Check if we have the required columns
missing_cols = [col for col in ALL_CONFIG_COLUMNS if col not in df.columns]
if missing_cols:
    print(f"Missing columns: {missing_cols}")
    print(f"Available columns: {[col for col in df.columns if 'btb' in col.lower() or 'ppw' in col.lower() or 'prefetch' in col.lower()]}")
    exit()
else:
    print("All required columns found!")

# 1. DATA PREPARATION
print(f"\nDATA PREPARATION")
print("-" * 40)

# Exclude target and performance columns from features
X = df.drop(ALL_CONFIG_COLUMNS, axis=1)
y = df[TARGET_COLUMNS].copy()

# Store top-3 configurations and performance data
top3_configs = {
    'best': df[['btbCore0_best', 'btbCore1_best', 'prefetcher_best', 'PPW_best']].copy(),
    '2nd': df[['btbCore0_2nd', 'btbCore1_2nd', 'prefetcher_2nd', 'PPW_2nd']].copy(),
    '3rd': df[['btbCore0_3rd', 'btbCore1_3rd', 'prefetcher_3rd', 'PPW_3rd']].copy()
}

print(f"Features shape: {X.shape}")
print(f"Targets shape: {y.shape}")
print(f"Target 1 (btbCore0) unique values: {len(y.iloc[:, 0].unique())}")
print(f"Target 2 (btbCore1) unique values: {len(y.iloc[:, 1].unique())}")
print(f"Target 3 (prefetcher) unique values: {len(y.iloc[:, 2].unique())}")
print(f"Prefetcher values: {y.iloc[:, 2].unique()}")

# Analyze target distribution
print(f"\nTarget Distribution Analysis:")
print("-" * 40)
print(f"Unique configurations:")
print(f"  btbCore0: {y['btbCore0_best'].nunique()} unique values")
print(f"  btbCore1: {y['btbCore1_best'].nunique()} unique values") 
print(f"  prefetcher: {y['prefetcher_best'].nunique()} unique values")
print(f"  Total possible combinations: {y['btbCore0_best'].nunique() * y['btbCore1_best'].nunique() * y['prefetcher_best'].nunique()}")

# Check stability - how often best config matches previous config
if 'btbCore0_prev' in X.columns and 'btbCore1_prev' in X.columns:
    same_btb0 = (X['btbCore0_prev'] == y['btbCore0_best']).sum()
    same_btb1 = (X['btbCore1_prev'] == y['btbCore1_best']).sum()
    print(f"\nConfiguration Stability:")
    print(f"  btbCore0 stays same: {same_btb0}/{len(y)} ({same_btb0/len(y)*100:.1f}%)")
    print(f"  btbCore1 stays same: {same_btb1}/{len(y)} ({same_btb1/len(y)*100:.1f}%)")
    if 'Prefetch_prev' in X.columns or 'prefetcher_prev' in X.columns:
        prefetch_col = 'Prefetch_prev' if 'Prefetch_prev' in X.columns else 'prefetcher_prev'
        # Note: This comparison may not work directly since prefetcher_best is encoded
        print(f"  (prefetcher stability check requires decoding)")


# 2. FEATURE PREPROCESSING
print(f"\nFEATURE PREPROCESSING")
print("-" * 40)

# Handle categorical features
categorical_cols = X.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded: {col}")

# Encode prefetcher in targets (it's categorical)
prefetcher_encoder = LabelEncoder()
y['prefetcher_best'] = prefetcher_encoder.fit_transform(y['prefetcher_best'].astype(str))
print(f"\nEncoded prefetcher_best: {dict(zip(prefetcher_encoder.classes_, prefetcher_encoder.transform(prefetcher_encoder.classes_)))}")

# Remove non-numeric columns that couldn't be encoded
numeric_cols = X.select_dtypes(include=[np.number]).columns
X = X[numeric_cols]

print(f"Final feature set: {X.shape}")

print(f"\nFirst 20 feature columns:")
for i, col in enumerate(X.columns):
    if i < 20:
        print(f"  {i+1}. {col}")
if len(X.columns) > 20:
    print(f"  ... and {len(X.columns) - 20} more")

# 3. TRAIN-TEST SPLIT
print(f"\nTRAIN-TEST SPLIT")
print("-" * 40)
print(f"Split method: {SPLIT_METHOD}")

if SPLIT_METHOD == 'benchmark':
    # Split by benchmark - test on unseen benchmarks
    if 'benchmark' not in X.columns:
        print("ERROR: 'benchmark' column not found in features!")
        print("Available columns:", X.columns.tolist())
        print("Falling back to random split...")
        SPLIT_METHOD = 'random'
    else:
        benchmarks = X['benchmark'].unique()
        print(f"Found {len(benchmarks)} unique benchmarks: {benchmarks}")
        
        # Use specified test size to determine number of test benchmarks
        n_test_benchmarks = max(1, int(len(benchmarks) * TEST_SIZE))
        np.random.seed(RANDOM_STATE)
        test_benchmarks = np.random.choice(benchmarks, n_test_benchmarks, replace=False)
        
        print(f"Test benchmarks ({n_test_benchmarks}): {test_benchmarks}")
        print(f"Train benchmarks ({len(benchmarks) - n_test_benchmarks}): {[b for b in benchmarks if b not in test_benchmarks]}")
        
        # Split by benchmark
        test_mask = X['benchmark'].isin(test_benchmarks)
        X_train = X[~test_mask].copy()
        X_test = X[test_mask].copy()
        y_train = y[~test_mask].copy()
        y_test = y[test_mask].copy()
        
        print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        print(f"This tests generalization to UNSEEN benchmarks")

elif SPLIT_METHOD == 'temporal':
    # Temporal split - train on early data, test on later data
    print("Using temporal split (chronological order)")
    print("Assuming data is already sorted by time/period")
    
    split_point = int(len(X) * (1 - TEST_SIZE))
    X_train = X.iloc[:split_point].copy()
    X_test = X.iloc[split_point:].copy()
    y_train = y.iloc[:split_point].copy()
    y_test = y.iloc[split_point:].copy()
    
    print(f"Train set: first {len(X_train)} samples ({(1-TEST_SIZE)*100:.0f}%)")
    print(f"Test set: last {len(X_test)} samples ({TEST_SIZE*100:.0f}%)")
    print(f"This tests generalization to FUTURE time periods")

else:  # random
    # Random split with stratification
    print("Using random stratified split")
    
    # Create stratification key (all 3 targets)
    stratify_key = (y[TARGET_COLUMNS[0]].astype(str) + '_' + 
                    y[TARGET_COLUMNS[1]].astype(str) + '_' + 
                    y[TARGET_COLUMNS[2]].astype(str))
    
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        for train_idx, test_idx in sss.split(X, stratify_key):
            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_test = y.iloc[test_idx].copy()
        print("Using stratified split to balance target distribution")
    except:
        # Fallback if stratification fails
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        print("Stratification failed, using simple random split")
    
    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

# Check for data overlap (potential leakage indicator)
print(f"\nChecking for potential data leakage...")
if 'config' in X_train.columns:
    train_configs = set(X_train['config'].astype(str))
    test_configs = set(X_test['config'].astype(str))
    config_overlap = len(train_configs & test_configs)
    print(f"Config values appearing in both train/test: {config_overlap}/{len(test_configs)} ({config_overlap/len(test_configs)*100:.1f}%)")
    if config_overlap / len(test_configs) > 0.9:
        print("  WARNING: High overlap detected - model may appear to overfit")


# Get corresponding top-3 data for test set
test_indices = X_test.index
test_top3 = {}
for rank, data in top3_configs.items():
    test_top3[rank] = data.loc[test_indices].copy()

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle NaN values in features
print(f"\nCleaning data...")
imputer = SimpleImputer(strategy='mean')
X_train_scaled = imputer.fit_transform(X_train_scaled)
X_test_scaled = imputer.transform(X_test_scaled)

# Handle NaN values in targets
print(f"Before cleaning - y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

# Convert to numeric where needed
for col in ['btbCore0_best', 'btbCore1_best']:
    y_train[col] = pd.to_numeric(y_train[col], errors='coerce')
    y_test[col] = pd.to_numeric(y_test[col], errors='coerce')

# Remove rows with NaN in training data
mask_train = y_train.notna().all(axis=1)
y_train = y_train[mask_train]
X_train_scaled = X_train_scaled[mask_train.values]

# Remove rows with NaN in test data
mask_test = y_test.notna().all(axis=1)
y_test = y_test[mask_test]
X_test_scaled = X_test_scaled[mask_test.values]

# Update test_top3 to match cleaned test set
test_top3_cleaned = {}
for rank, data in test_top3.items():
    test_top3_cleaned[rank] = data[mask_test.values].reset_index(drop=True)
    # Convert to numeric
    for col in [f'btbCore0_{rank}', f'btbCore1_{rank}']:
        test_top3_cleaned[rank][col] = pd.to_numeric(test_top3_cleaned[rank][col], errors='coerce')
    
    # Handle prefetcher encoding - clean NaN values first
    prefetch_col = test_top3_cleaned[rank][f'prefetcher_{rank}']
    prefetch_col = prefetch_col.fillna('none')  # Fill NaN with 'none'
    prefetch_col = prefetch_col.astype(str)
    prefetch_col = prefetch_col.replace('nan', 'none')  # Replace string 'nan' with 'none'
    
    # Handle any unseen categories
    valid_categories = set(prefetcher_encoder.classes_)
    prefetch_col = prefetch_col.apply(lambda x: x if x in valid_categories else 'none')
    
    test_top3_cleaned[rank][f'prefetcher_{rank}'] = prefetcher_encoder.transform(prefetch_col)
    
    # Convert PPW to numeric
    test_top3_cleaned[rank][f'PPW_{rank}'] = pd.to_numeric(test_top3_cleaned[rank][f'PPW_{rank}'], errors='coerce')
test_top3 = test_top3_cleaned
y_test = y_test.reset_index(drop=True)

print(f"After cleaning - y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
print(f"After cleaning - test_top3['best'] shape: {test_top3['best'].shape}")
print(f"Removed {(~mask_train).sum()} training samples and {(~mask_test).sum()} test samples with NaN")

# 5. HYPERPARAMETER TUNING AND MODEL TRAINING (Single Multi-Output Model)
print(f"\nOPTIMIZED MODEL TRAINING (Single Multi-Output for 3 targets)")
print("-" * 40)

if ENABLE_HYPERPARAMETER_TUNING:
    print(f"Advanced Hyperparameter Optimization...")
    
    # Create combined label for stratification (all 3 targets)
    y_train_combined = (y_train[TARGET_COLUMNS[0]].astype(str) + '_' + 
                        y_train[TARGET_COLUMNS[1]].astype(str) + '_' + 
                        y_train[TARGET_COLUMNS[2]].astype(str))
    
    # Use 60% for tuning with stratification
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.6, random_state=RANDOM_STATE)
    for tune_idx, val_idx in sss.split(X_train_scaled, y_train_combined):
        X_tune = X_train_scaled[tune_idx]
        y_tune = y_train.iloc[tune_idx]
        X_val = X_train_scaled[val_idx]
        y_val = y_train.iloc[val_idx]
    
    print(f"Tuning on {len(X_tune)} samples ({len(X_tune)/len(X_train_scaled)*100:.1f}% of training data)")
    
    # Define custom scorer for 3 targets
    def exact_match_scorer(estimator, X, y_true):
        y_pred = estimator.predict(X)
        return (y_true.values == y_pred).all(axis=1).mean()
    
    # Expanded parameter space with regularization to prevent overfitting
    param_dict = {
        'n_estimators': [100, 200, 300],  # Reduced from 600
        'max_depth': [10, 15, 20, 25],  # Added limits to prevent overfitting
        'min_samples_split': [10, 20, 30, 40],  # Increased minimums
        'min_samples_leaf': [5, 10, 15, 20],  # Increased minimums
        'max_features': ['sqrt', 'log2', 0.3],  # Reduced options
        'bootstrap': [True],
        'class_weight': ['balanced', None],
        'max_samples': [0.7, 0.8]  # Bootstrap sample size to add variance
    }
    
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    
    # Multi-stage search
    print("\nStage 1: Coarse search...")
    coarse_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_split': [10, 20, 40],
        'min_samples_leaf': [5, 10, 20],
        'max_features': ['sqrt', 'log2']
    }
    
    coarse_search = RandomizedSearchCV(
        rf, coarse_params,
        cv=3,
        n_iter=20,
        scoring=exact_match_scorer,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    coarse_search.fit(X_tune, y_tune)
    
    print(f"Coarse best score: {coarse_search.best_score_:.4f}")
    print(f"Coarse best params: {coarse_search.best_params_}")
    
    # Stage 2: Fine-tune with RandomizedSearchCV for speed
    print("\nStage 2: Fine-tuning...")
    best_coarse = coarse_search.best_params_
    
    fine_params = {
        'n_estimators': [
            max(50, best_coarse['n_estimators'] - 50),
            best_coarse['n_estimators'],
            best_coarse['n_estimators'] + 50
        ],
        'max_depth': [
            max(5, best_coarse['max_depth'] - 3),
            best_coarse['max_depth'],
            min(25, best_coarse['max_depth'] + 3)
        ],
        'min_samples_split': [
            max(5, best_coarse['min_samples_split'] - 5),
            best_coarse['min_samples_split'],
            best_coarse['min_samples_split'] + 5
        ],
        'min_samples_leaf': [
            max(2, best_coarse.get('min_samples_leaf', 5) - 3),
            best_coarse.get('min_samples_leaf', 5),
            best_coarse.get('min_samples_leaf', 5) + 3
        ],
        'max_features': [best_coarse['max_features']],
        'class_weight': ['balanced', None],
        'bootstrap': [True]
    }
    
    # Use RandomizedSearchCV instead of GridSearchCV
    fine_search = RandomizedSearchCV(
        rf, fine_params,
        cv=2,  # Reduced from 3
        n_iter=15,  # Only test 15 combinations instead of all
        scoring=exact_match_scorer,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    fine_search.fit(X_tune, y_tune)
    
    best_model = fine_search.best_estimator_
    
    print(f"\nFinal best parameters: {fine_search.best_params_}")
    print(f"Final best CV score: {fine_search.best_score_:.4f}")
    
    # Validate on held-out validation set
    val_pred = best_model.predict(X_val)
    val_score = (y_val.values == val_pred).all(axis=1).mean()
    print(f"Validation set score: {val_score:.4f}")

else:
    # Use default parameters with regularization
    print(f"Using default parameters with regularization...")
    best_model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        n_jobs=-1
    )
    best_model.fit(X_train_scaled, y_train)

# Make predictions with the single model (now 3 outputs)
y_pred = best_model.predict(X_test_scaled)
predictions = {
    TARGET_COLUMNS[0]: y_pred[:, 0],
    TARGET_COLUMNS[1]: y_pred[:, 1],
    TARGET_COLUMNS[2]: y_pred[:, 2]
}

# 6. TOP-K ACCURACY EVALUATION (Enhanced with Prefetcher)
print(f"\nENHANCED TOP-K ACCURACY EVALUATION (3-way config)")
print("-" * 40)

exact_matches = 0
top3_matches = 0
ppw_costs = {'exact': [], 'top3_miss': [], 'top3_hit': []}
ppw_diffs = []  # Track PPW differences
detailed_results = []

# Configuration frequency analysis (now includes prefetcher)
predicted_configs = list(zip(predictions[TARGET_COLUMNS[0]], 
                             predictions[TARGET_COLUMNS[1]],
                             predictions[TARGET_COLUMNS[2]]))
config_frequency = Counter(predicted_configs)

print(f"Most frequently predicted configurations (btbCore0, btbCore1, prefetcher):")
for config, count in config_frequency.most_common(10):
    # Decode prefetcher
    prefetch_decoded = prefetcher_encoder.inverse_transform([int(config[2])])[0]
    print(f"  ({config[0]}, {config[1]}, {prefetch_decoded}): {count} times ({count/len(y_pred)*100:.1f}%)")

# Use len(y_pred) to ensure we match the predictions
num_samples = len(y_pred)

for i in range(num_samples):
    # Predicted configuration (3-tuple now)
    pred_core0 = predictions[TARGET_COLUMNS[0]][i]
    pred_core1 = predictions[TARGET_COLUMNS[1]][i]
    pred_prefetch = predictions[TARGET_COLUMNS[2]][i]
    pred_config = (pred_core0, pred_core1, pred_prefetch)

    # Actual configuration from y_test
    actual_core0 = y_test.iloc[i, 0]
    actual_core1 = y_test.iloc[i, 1]
    actual_prefetch = y_test.iloc[i, 2]
    actual_best_config = (actual_core0, actual_core1, actual_prefetch)

    # Actual configurations from top-3
    actual_configs = []
    for rank in ['best', '2nd', '3rd']:
        rank_core0 = test_top3[rank].iloc[i][f'btbCore0_{rank}']
        rank_core1 = test_top3[rank].iloc[i][f'btbCore1_{rank}']
        rank_prefetch = test_top3[rank].iloc[i][f'prefetcher_{rank}']
        rank_ppw = test_top3[rank].iloc[i][f'PPW_{rank}']
        
        # Convert PPW to numeric
        try:
            rank_ppw = float(rank_ppw)
        except (ValueError, TypeError):
            rank_ppw = np.nan
        
        actual_configs.append({
            'rank': rank,
            'config': (rank_core0, rank_core1, rank_prefetch),
            'ppw': rank_ppw
        })

    # Check matches - compare all 3 elements
    exact_match = (pred_core0 == actual_core0) and (pred_core1 == actual_core1) and (pred_prefetch == actual_prefetch)
    
    # Try element-wise comparison for top-3
    top3_match = False
    for cfg in actual_configs:
        cfg_core0, cfg_core1, cfg_prefetch = cfg['config']
        if (pred_core0 == cfg_core0) and (pred_core1 == cfg_core1) and (pred_prefetch == cfg_prefetch):
            top3_match = True
            break

    if exact_match:
        exact_matches += 1
        ppw_costs['exact'].append(actual_configs[0]['ppw'])
        ppw_diffs.append(0)  # No difference for exact match

    if top3_match:
        top3_matches += 1
        matched_ppw = next(cfg['ppw'] for cfg in actual_configs 
                          if (pred_core0 == cfg['config'][0]) and 
                             (pred_core1 == cfg['config'][1]) and
                             (pred_prefetch == cfg['config'][2]))
        ppw_costs['top3_hit'].append(matched_ppw)
        # Calculate PPW difference (best - predicted)
        ppw_diff = actual_configs[0]['ppw'] - matched_ppw
        ppw_diffs.append(ppw_diff)
    else:
        ppw_costs['top3_miss'].append(actual_configs[0]['ppw'])
        # For misses, we don't know the predicted config's PPW
        # Just record that it missed
        ppw_diffs.append(None)

    # Store detailed result
    detailed_results.append({
        'sample': i,
        'predicted': pred_config,
        'actual_best': actual_best_config,
        'actual_2nd': actual_configs[1]['config'],
        'actual_3rd': actual_configs[2]['config'],
        'ppw_best': actual_configs[0]['ppw'],
        'ppw_2nd': actual_configs[1]['ppw'],
        'ppw_3rd': actual_configs[2]['ppw'],
        'exact_match': exact_match,
        'top3_match': top3_match
    })

# Calculate accuracies
exact_accuracy = exact_matches / num_samples
top3_accuracy = top3_matches / num_samples

print(f"\nOPTIMIZED MODEL RESULTS:")
print(f"Individual Accuracies:")
acc_0 = accuracy_score(y_test[TARGET_COLUMNS[0]], predictions[TARGET_COLUMNS[0]])
acc_1 = accuracy_score(y_test[TARGET_COLUMNS[1]], predictions[TARGET_COLUMNS[1]])
acc_2 = accuracy_score(y_test[TARGET_COLUMNS[2]], predictions[TARGET_COLUMNS[2]])
print(f"  {TARGET_COLUMNS[0]} individual accuracy: {acc_0:.4f}")
print(f"  {TARGET_COLUMNS[1]} individual accuracy: {acc_1:.4f}")
print(f"  {TARGET_COLUMNS[2]} individual accuracy: {acc_2:.4f}")

# Debug exact match calculation
exact_match_check = (y_test.values == y_pred).all(axis=1)
print(f"\nExact Match Debug:")
print(f"Number of exact matches (numpy comparison): {exact_match_check.sum()}")
print(f"Number of exact matches (loop count): {exact_matches}")

print(f"\nCombined Accuracies (all 3 must match):")
print(f"  Exact Match (Best Config):  {exact_accuracy:.4f} ({exact_matches}/{num_samples})")
print(f"  Top-3 Match (Any of 3):     {top3_accuracy:.4f} ({top3_matches}/{num_samples})")
print(f"  Improvement:                +{(top3_accuracy - exact_accuracy):.4f} ({top3_matches - exact_matches} more correct)")

# PPW Difference Analysis
print(f"\nPPW DIFFERENCE ANALYSIS:")
print("-" * 40)
valid_ppw_diffs = [d for d in ppw_diffs if d is not None]
if valid_ppw_diffs:
    avg_ppw_diff = np.mean(valid_ppw_diffs)
    median_ppw_diff = np.median(valid_ppw_diffs)
    max_ppw_diff = np.max(valid_ppw_diffs)
    min_ppw_diff = np.min(valid_ppw_diffs)
    
    print(f"PPW Difference Statistics (Best PPW - Predicted PPW):")
    print(f"  Average PPW difference:  {avg_ppw_diff:.4f}")
    print(f"  Median PPW difference:   {median_ppw_diff:.4f}")
    print(f"  Max PPW difference:      {max_ppw_diff:.4f}")
    print(f"  Min PPW difference:      {min_ppw_diff:.4f}")
    print(f"  Samples with PPW data:   {len(valid_ppw_diffs)}/{num_samples}")
    
    # Percentage loss
    if ppw_costs['exact'] and ppw_costs['top3_hit']:
        exact_costs = [x for x in ppw_costs['exact'] if not np.isnan(x)]
        top3_costs = [x for x in ppw_costs['top3_hit'] if not np.isnan(x)]
        
        if exact_costs and top3_costs:
            avg_exact_ppw = np.mean(exact_costs)
            avg_top3_ppw = np.mean(top3_costs)
            ppw_loss_pct = ((avg_exact_ppw - avg_top3_ppw) / avg_exact_ppw) * 100
            print(f"  Average exact match PPW: {avg_exact_ppw:.4f}")
            print(f"  Average top-3 match PPW: {avg_top3_ppw:.4f}")
            print(f"  PPW loss percentage:     {ppw_loss_pct:.2f}%")
else:
    print("No valid PPW difference data available")

# 7. FEATURE IMPORTANCE ANALYSIS
print(f"\nFEATURE IMPORTANCE ANALYSIS")
print("-" * 40)

feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"Top 15 Most Important Features:")
print(feature_importance_df.head(15))

# 8. HYPERPARAMETER ANALYSIS
if ENABLE_HYPERPARAMETER_TUNING:
    print(f"\nOPTIMAL HYPERPARAMETERS FOUND")
    print("-" * 40)
    print(f"Optimal parameters:")
    print(f"  n_estimators: {best_model.n_estimators}")
    print(f"  max_depth: {best_model.max_depth}")
    print(f"  min_samples_split: {best_model.min_samples_split}")
    print(f"  min_samples_leaf: {best_model.min_samples_leaf}")
    print(f"  max_features: {best_model.max_features}")
    if hasattr(best_model, 'bootstrap'):
        print(f"  bootstrap: {best_model.bootstrap}")

print(f"\nOPTIMIZED ANALYSIS COMPLETE!")
print(f"Key Results:")
print(f"  Exact match accuracy (all 3 params): {exact_accuracy:.4f}")
print(f"  Top-3 match accuracy: {top3_accuracy:.4f}")
print(f"  Improvement: +{(top3_accuracy - exact_accuracy):.4f} ({top3_matches - exact_matches} more correct)")

if valid_ppw_diffs:
    print(f"  Average PPW loss: {avg_ppw_diff:.4f}")

# Save the optimized model and encoders with timestamp
print(f"\nMODEL PERSISTENCE")
print("-" * 40)

# Create models directory if it doesn't exist
os.makedirs('saved_models', exist_ok=True)

# Generate timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"rf_3way_config_predictor_{timestamp}"

# Save files
model_path = f"saved_models/{model_name}.pkl"
scaler_path = f"saved_models/{model_name}_scaler.pkl"
encoder_path = f"saved_models/{model_name}_prefetcher_encoder.pkl"
metadata_path = f"saved_models/{model_name}_metadata.txt"

joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(prefetcher_encoder, encoder_path)

print(f"Model saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")
print(f"Encoder saved to: {encoder_path}")

# Save metadata
with open(metadata_path, 'w') as f:
    f.write(f"Random Forest Configuration Predictor\n")
    f.write(f"=" * 60 + "\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Split Method: {SPLIT_METHOD}\n")
    f.write(f"Test Size: {TEST_SIZE}\n")
    f.write(f"Training samples: {len(X_train_scaled)}\n")
    f.write(f"Test samples: {num_samples}\n")
    f.write(f"\nModel Parameters:\n")
    f.write(f"  n_estimators: {best_model.n_estimators}\n")
    f.write(f"  max_depth: {best_model.max_depth}\n")
    f.write(f"  min_samples_split: {best_model.min_samples_split}\n")
    f.write(f"  min_samples_leaf: {best_model.min_samples_leaf}\n")
    f.write(f"  max_features: {best_model.max_features}\n")
    f.write(f"\nPerformance Metrics:\n")
    f.write(f"  Exact match accuracy: {exact_accuracy:.4f}\n")
    f.write(f"  Top-3 match accuracy: {top3_accuracy:.4f}\n")
    f.write(f"  btbCore0 accuracy: {acc_0:.4f}\n")
    f.write(f"  btbCore1 accuracy: {acc_1:.4f}\n")
    f.write(f"  prefetcher accuracy: {acc_2:.4f}\n")
    if valid_ppw_diffs:
        f.write(f"  Average PPW difference: {avg_ppw_diff:.4f}\n")
        f.write(f"  Median PPW difference: {median_ppw_diff:.4f}\n")

print(f"Metadata saved to: {metadata_path}")

print("\nTo load the model:")
print(f"import joblib")
print(f"model = joblib.load('{model_path}')")
print(f"scaler = joblib.load('{scaler_path}')")
print(f"prefetcher_encoder = joblib.load('{encoder_path}')")
