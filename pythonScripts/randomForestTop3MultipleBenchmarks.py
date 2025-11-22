# OPTIMIZED TOP-3 CONFIGURATION PREDICTOR WITH A SINGLE MULTI-OUTPUT MODEL
# Trains one model to predict both btbCore0_best and btbCore1_best simultaneously.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
#  Data Loading, Merging & Initial Cleaning
# ==============================================================================

# For Colab file upload (uncomment and run this section first)
# from google.colab import files
# print("Please upload your first CSV file:")
# uploaded1 = files.upload()
# csv_filename1 = list(uploaded1.keys())[0]
#
# print("Please upload your second CSV file:")
# uploaded2 = files.upload()
# csv_filename2 = list(uploaded2.keys())[0]

# For local testing, specify your CSV file paths
csv_filename1 = '/content/train_with_top3_barnes_simple.csv'  # Replace with your first file path
csv_filename2 = '/content/train_with_top3_cholesky.csv'
csv_filename3 = '/content/train_with_top3_fft.csv'   # Replace with your second file path
csv_filename4 = '/content/train_with_top3_radiosity.csv'

print(" OPTIMIZED Top-3 Random Forest Configuration Predictor")
print("=" * 60)

# Load the datasets
try:
    df1 = pd.read_csv(csv_filename1)
    df2 = pd.read_csv(csv_filename2)
    df3 = pd.read_csv(csv_filename3)
    df4 = pd.read_csv(csv_filename4)
    print(f" Data 1 loaded successfully! Shape: {df1.shape}")
    print(f" Data 2 loaded successfully! Shape: {df2.shape}")
    print(f"Data 3 loaded successfully! Shape: {df3.shape}")
    print(f"Data 4 loaded successfully! Shape: {df4.shape}")

    # Combine the two dataframes
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)

    #df = df2
    print("\n Datasets combined successfully!")
    print(f" Combined Data Shape: {df.shape}")
    print(f" Combined Columns: {list(df.columns)}")
    print("\n First few rows of the combined data:")
    print(df.head())

except FileNotFoundError:
    print(" One or both CSV files not found. Please check the file paths.")
    exit()

# Configuration
TARGET_COLUMNS = ['btbCore0_best', 'btbCore1_best']  # What we're trying to predict
ALL_CONFIG_COLUMNS = [
    'btbCore0_best', 'btbCore1_best', 'PPW_best',
    'btbCore0_2nd', 'btbCore1_2nd', 'PPW_2nd', 'Diff_best_2nd',
    'btbCore0_3rd', 'btbCore1_3rd', 'PPW_3rd', 'Diff_best_3rd'
]

#  Columns to drop
METADATA_COLUMNS_TO_DROP = ['best-config', 'file', 'file_prev', 'period_start',
                            'period_end', 'period_start_prev', 'period_end_prev',
                            'directory_perf_prev', 'leaf_dir_prev', 'directory_power_prev',
                            'leaf_dir_perf_prev', 'leaf_dir_power_prev']

# Hyperparameter tuning options
ENABLE_HYPERPARAMETER_TUNING = True
SEARCH_TYPE = 'random'  # 'grid' or 'random'
N_ITER_RANDOM_SEARCH = 30  # Number of iterations for random search
CV_FOLDS = 3
RANDOM_STATE = 42

# Drop the specified metadata columns
print(f"\n Dropping metadata columns: {METADATA_COLUMNS_TO_DROP}")
df = df.drop(columns=METADATA_COLUMNS_TO_DROP, errors='ignore')
print(f" New data shape: {df.shape}")

# Check if we have the required columns
missing_cols = [col for col in ALL_CONFIG_COLUMNS if col not in df.columns]
if missing_cols:
    print(f" Missing columns: {missing_cols}")
    print(f"Available columns: {[col for col in df.columns if 'btb' in col.lower() or 'ppw' in col.lower()]}")
    exit()
else:
    print(" All required columns found!")

# 1. DATA PREPARATION
print(f"\n DATA PREPARATION")
print("-" * 40)

# Exclude target and performance columns from features
X = df.drop(ALL_CONFIG_COLUMNS, axis=1)
y = df[TARGET_COLUMNS].copy()

# Store top-3 configurations and performance data
top3_configs = {
    'best': df[['btbCore0_best', 'btbCore1_best', 'PPW_best']].copy(),
    '2nd': df[['btbCore0_2nd', 'btbCore1_2nd', 'PPW_2nd']].copy(),
    '3rd': df[['btbCore0_3rd', 'btbCore1_3rd', 'PPW_3rd']].copy()
}

print(f"Features shape: {X.shape}")
print(f"Targets shape: {y.shape}")
print(f"Target 1 unique values: {len(y.iloc[:, 0].unique())}")
print(f"Target 2 unique values: {len(y.iloc[:, 1].unique())}")

# 2. FEATURE PREPROCESSING
print(f"\n FEATURE PREPROCESSING")
print("-" * 40)

# Handle categorical features
categorical_cols = X.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded: {col}")

# Remove non-numeric columns that couldn't be encoded
numeric_cols = X.select_dtypes(include=[np.number]).columns
X = X[numeric_cols]

print(f"Final feature set: {X.shape}")

# 3. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Get corresponding top-3 data for test set
test_indices = X_test.index
test_top3 = {}
for rank, data in top3_configs.items():
    test_top3[rank] = data.loc[test_indices]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 5. HYPERPARAMETER TUNING AND MODEL TRAINING (Single Multi-Output Model)
print(f"\n OPTIMIZED MODEL TRAINING (Single Multi-Output)")
print("-" * 40)

if ENABLE_HYPERPARAMETER_TUNING:
    print(f" Optimizing hyperparameters...")

    # Define a custom scorer for multi-output accuracy
    # This checks if both predictions are correct for a single sample
    def exact_match_scorer(y_true, y_pred):
        return (y_true == y_pred).all(axis=1).mean()

    if SEARCH_TYPE == 'grid':
        # Grid search with reasonable parameter space
        param_search = GridSearchCV
        param_dict = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    else:  # Random search
        param_search = RandomizedSearchCV
        # Broader parameter space for random search
        param_dict = {
            'n_estimators': [50, 100, 150, 200, 250, 300],
            'max_depth': [5, 10, 15, 20, 25, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
            'bootstrap': [True, False]
        }

    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    search = param_search(
        rf, param_dict,
        cv=CV_FOLDS,
        scoring=exact_match_scorer, # Score on exact match of both targets
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    y_train_clean = y_train.copy()
    y_test_clean = y_test.copy()

    # Remove dashes from both columns
    for col in ['btbCore0_best', 'btbCore1_best']:
        y_train_clean[col] = pd.to_numeric(y_train_clean[col], errors='coerce')
        y_test_clean[col] = pd.to_numeric(y_test_clean[col], errors='coerce')

    # Remove rows with any NaN
    mask_train = y_train_clean.notna().all(axis=1)
    y_train_clean = y_train_clean[mask_train]
    X_train_scaled_filtered = X_train_scaled[mask_train]
    mask_test = y_test_clean.notna().all(axis=1)
    y_test_clean = y_test_clean[mask_test]
    X_test_scaled_filtered = X_test_scaled[mask_test]

    X_tune, _, y_tune, _ = train_test_split(
    X_train_scaled_filtered,
    y_train_clean,
    train_size=0.3,  # Use 30% for tuning
    random_state=42)

    search.fit(X_tune, y_tune)

    best_model = search.best_estimator_
    print(f" Best parameters: {search.best_params_}")
    print(f" Best CV score: {search.best_score_:.4f}")

else:
    # Use default parameters (your original static setup)
    print(f" Using default parameters...")
    best_model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1
    )
    best_model.fit(X_train_scaled, y_train)

# Make predictions with the single model
y_pred = best_model.predict(X_test_scaled_filtered)
predictions = {
    TARGET_COLUMNS[0]: y_pred[:, 0],
    TARGET_COLUMNS[1]: y_pred[:, 1]
}


# 6. TOP-K ACCURACY EVALUATION (Enhanced)
print(f"\n ENHANCED TOP-K ACCURACY EVALUATION")
print("-" * 40)

exact_matches = 0
top3_matches = 0
ppw_costs = {'exact': [], 'top3_miss': [], 'top3_hit': []}
detailed_results = []

# Configuration frequency analysis
predicted_configs = list(zip(predictions[TARGET_COLUMNS[0]], predictions[TARGET_COLUMNS[1]]))
config_frequency = Counter(predicted_configs)

print(f" Most frequently predicted configurations:")
for config, count in config_frequency.most_common(10):
    print(f"  {config}: {count} times ({count/len(X_test)*100:.1f}%)")

for i in range(len(predictions[TARGET_COLUMNS[0]])):
    # Predicted configuration
    pred_core0 = predictions[TARGET_COLUMNS[0]][i]
    pred_core1 = predictions[TARGET_COLUMNS[1]][i]
    pred_config = (pred_core0, pred_core1)

    # Actual configurations (best, 2nd, 3rd)
    actual_configs = []
    for rank in ['best', '2nd', '3rd']:
        actual_core0 = test_top3[rank].iloc[i][f'btbCore0_{rank}']
        actual_core1 = test_top3[rank].iloc[i][f'btbCore1_{rank}']
        actual_ppw = test_top3[rank].iloc[i][f'PPW_{rank}']
        actual_configs.append({
            'rank': rank,
            'config': (actual_core0, actual_core1),
            'ppw': actual_ppw
        })

    # Check matches
    exact_match = pred_config == actual_configs[0]['config']
    top3_match = any(pred_config == cfg['config'] for cfg in actual_configs)

    if exact_match:
        exact_matches += 1
        ppw_costs['exact'].append(actual_configs[0]['ppw'])

    if top3_match:
        top3_matches += 1
        matched_ppw = next(cfg['ppw'] for cfg in actual_configs if pred_config == cfg['config'])
        ppw_costs['top3_hit'].append(matched_ppw)
    else:
        ppw_costs['top3_miss'].append(actual_configs[0]['ppw'])

    # Store detailed result
    detailed_results.append({
        'sample': i,
        'predicted': pred_config,
        'actual_best': actual_configs[0]['config'],
        'actual_2nd': actual_configs[1]['config'],
        'actual_3rd': actual_configs[2]['config'],
        'ppw_best': actual_configs[0]['ppw'],
        'ppw_2nd': actual_configs[1]['ppw'],
        'ppw_3rd': actual_configs[2]['ppw'],
        'exact_match': exact_match,
        'top3_match': top3_match
    })

# Calculate accuracies
exact_accuracy = exact_matches / len(X_test)
top3_accuracy = top3_matches / len(X_test)

print(f"\ OPTIMIZED MODEL RESULTS:")
print(f"Individual Accuracies:")
print(f"  {TARGET_COLUMNS[0]} individual accuracy: {accuracy_score(y_test_clean[TARGET_COLUMNS[0]], predictions[TARGET_COLUMNS[0]]):.4f}")
print(f"  {TARGET_COLUMNS[1]} individual accuracy: {accuracy_score(y_test_clean[TARGET_COLUMNS[1]], predictions[TARGET_COLUMNS[1]]):.4f}")

print(f"\nCombined Accuracies:")
print(f"  Exact Match (Best Config):  {exact_accuracy:.4f} ({exact_matches}/{len(X_test)})")
print(f"  Top-3 Match (Any of 3):     {top3_accuracy:.4f} ({top3_matches}/{len(X_test)})")
print(f"  Improvement:                +{(top3_accuracy - exact_accuracy):.4f} ({top3_matches - exact_matches} more correct)")

# 7. COMPARISON WITH STATIC MODEL
print(f"\n COMPARISON: OPTIMIZED vs STATIC PARAMETERS")
print("-" * 50)

# Train a single static model for comparison
print("Training static multi-output model...")
static_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1
)
static_rf.fit(X_train_scaled, y_train)

# Evaluate static model
static_predictions = static_rf.predict(X_test_scaled_filtered)
static_exact_matches = (y_test_clean.values == static_predictions).all(axis=1).sum()
static_top3_matches = 0

for i in range(len(X_test)):
    pred_config = (static_predictions[i, 0], static_predictions[i, 1])

    actual_configs = []
    for rank in ['best', '2nd', '3rd']:
        actual_core0 = test_top3[rank].iloc[i][f'btbCore0_{rank}']
        actual_core1 = test_top3[rank].iloc[i][f'btbCore1_{rank}']
        actual_configs.append((actual_core0, actual_core1))

    if pred_config in actual_configs:
        static_top3_matches += 1

static_exact_accuracy = static_exact_matches / len(X_test_scaled_filtered)
static_top3_accuracy = static_top3_matches / len(X_test_scaled_filtered)

print(f" STATIC MODEL RESULTS:")
print(f"  Exact Match (Best Config):  {static_exact_accuracy:.4f} ({static_exact_matches}/{len(X_test_scaled_filtered)})")
print(f"  Top-3 Match (Any of 3):     {static_top3_accuracy:.4f} ({static_top3_matches}/{len(X_test_scaled_filtered)})")

print(f"\n IMPROVEMENT WITH OPTIMIZATION:")
exact_improvement = exact_accuracy - static_exact_accuracy
top3_improvement = top3_accuracy - static_top3_accuracy
print(f"  Exact Match Improvement:    +{exact_improvement:.4f} ({exact_improvement*100:.1f}% points)")
print(f"  Top-3 Match Improvement:    +{top3_improvement:.4f} ({top3_improvement*100:.1f}% points)")

# 8. FEATURE IMPORTANCE ANALYSIS
print(f"\n FEATURE IMPORTANCE ANALYSIS")
print("-" * 40)

# Feature importance from the single optimized model
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"🔝 Top 15 Most Important Features:")
print(feature_importance_df.head(15))

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance_df.head(20)
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Feature Importance')
plt.title('Top 20 Feature Importances (Optimized Model)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 9. HYPERPARAMETER ANALYSIS
if ENABLE_HYPERPARAMETER_TUNING:
    print(f"\n OPTIMAL HYPERPARAMETERS FOUND")
    print("-" * 40)
    print(f"Optimal parameters:")
    print(f"  n_estimators: {best_model.n_estimators}")
    print(f"  max_depth: {best_model.max_depth}")
    print(f"  min_samples_split: {best_model.min_samples_split}")
    print(f"  min_samples_leaf: {best_model.min_samples_leaf}")
    print(f"  max_features: {best_model.max_features}")
    if hasattr(best_model, 'bootstrap'):
        print(f"  bootstrap: {best_model.bootstrap}")

print(f"\n OPTIMIZED ANALYSIS COMPLETE!")
print(f"Key Results:")
print(f"• Optimized top-3 accuracy: {top3_accuracy:.4f} vs Static: {static_top3_accuracy:.4f}")
print(f"• Improvement from optimization: +{top3_improvement*100:.1f} percentage points")
print(f"• Total additional correct predictions: {int(top3_improvement * len(X_test))}")

if ppw_costs['exact'] and ppw_costs['top3_hit']:
    cost_pct = (np.mean(ppw_costs['top3_hit']) - np.mean(ppw_costs['exact'])) / np.mean(ppw_costs['exact']) * 100
    print(f"• Performance cost of top-3 matching: {cost_pct:.2f}% PPW reduction")

# Save the optimized model and scaler
print(f"\n MODEL PERSISTENCE")
print("-" * 40)
print("To save the optimized model:")
print("import joblib")
print("joblib.dump(best_model, 'optimized_multi_output_model.pkl')")
print("joblib.dump(scaler, 'feature_scaler.pkl')")
print("\nTo load and use:")
print("model = joblib.load('optimized_multi_output_model.pkl')")
print("scaler = joblib.load('feature_scaler.pkl')")
