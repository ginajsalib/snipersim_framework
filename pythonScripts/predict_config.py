"""
predict_config.py
=================
Load a saved RF model and run predictions on new data, applying the same
preprocessing pipeline as train_rf_gpu.py.  Outputs a CSV ready to be
consumed by costAnalysis.py.
 
Usage:
    python predict_config.py \\
        --input data/my_new_data.csv \\
        --model-dir saved_models/ \\
        --output predictions.csv \\
        [--benchmark barnes]
"""
 
import os
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
 
warnings.filterwarnings('ignore')
 
 
# ==============================================================================
# Column definitions  (must match train_rf_gpu.py)
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
 
# Columns costAnalysis.py expects in the output CSV
COST_ANALYSIS_PASSTHROUGH = [
    'benchmark', 'period_start', 'period_end',
    'btbCore0_prev', 'btbCore1_prev',
    'prefetcher_prev', 'L2_prev', 'L3_prev',
    'PPW_best', 'ips',
]
 
 
# ==============================================================================
# Helpers  (identical to train_rf_gpu.py)
# ==============================================================================
 
def split_prefetcher(series, suffix=''):
    """Split a 'type0-type1' prefetcher column into two per-core columns."""
    split = series.fillna('none-none').astype(str).str.split('-', n=1, expand=True)
    core0 = split[0].str.strip()
    core1 = split[1].str.strip() if split.shape[1] > 1 else core0
    return core0, core1
 
 
def safe_encode(encoder, series):
    """Encode a Series with a fitted LabelEncoder; unseen/NaN -> 'none'."""
    valid = set(encoder.classes_)
    mapped = series.fillna('none').astype(str).apply(lambda x: x if x in valid else 'none')
    return encoder.transform(mapped)
 
 
# ==============================================================================
# Model loading
# ==============================================================================
 
def load_model(model_dir):
    """Load the most recent saved RF model and all its preprocessors."""
    pkls = glob.glob(os.path.join(model_dir, 'rf_6way_config_predictor_*.pkl'))
    pkls += glob.glob(os.path.join(model_dir, 'rf_4way_config_predictor_*.pkl'))
    # Exclude preprocessor files
    pkls = [p for p in pkls
            if not any(tag in p for tag in ['_scaler', '_encoder', '_imputer'])]
 
    if not pkls:
        raise FileNotFoundError(f"No model .pkl found in '{model_dir}'")
 
    pkls.sort()
    model_path = pkls[-1]
    base = model_path.replace('.pkl', '')
 
    print(f"  Loading model : {model_path}")
    model = joblib.load(model_path)
 
    def _load(suffix):
        p = f"{base}{suffix}.pkl"
        if os.path.exists(p):
            print(f"  Loading        : {p}")
            return joblib.load(p)
        print(f"  WARNING: missing {p}")
        return None
 
    scaler  = _load('_scaler')
    imputer = _load('_imputer')
    enc_pf0 = _load('_prefetcher_core0_encoder')
    enc_pf1 = _load('_prefetcher_core1_encoder')
    enc_l2  = _load('_l2_size_encoder')
    enc_l3  = _load('_l3_size_encoder')
 
    return model, scaler, imputer, enc_pf0, enc_pf1, enc_l2, enc_l3
 
 
# ==============================================================================
# Preprocessing  (mirrors train_rf_gpu.py)
# ==============================================================================
 
def preprocess(df_raw, scaler, imputer, enc_pf0, enc_pf1, enc_l2, enc_l3,
               label_encoders_path=None):
    """
    Apply the same preprocessing steps as in train_rf_gpu.py to raw input data.
 
    Returns
    -------
    X_scaled : np.ndarray  - ready to pass to model.predict()
    passthrough : pd.DataFrame - columns to keep for the output CSV
    feature_cols : list[str] - ordered feature names (for diagnostics)
    """
    df = df_raw.copy()
 
    # Save passthrough columns before any modification
    passthrough = df[[c for c in COST_ANALYSIS_PASSTHROUGH if c in df.columns]].copy()
 
    # Split combined prefetcher columns (best / 2nd / 3rd)
    for rank in ['best', '2nd', '3rd']:
        col = f'prefetcher_{rank}'
        if col in df.columns:
            c0, c1 = split_prefetcher(df[col], suffix=rank)
            df[f'prefetcher_core0_{rank}'] = c0
            df[f'prefetcher_core1_{rank}'] = c1
            df.drop(columns=[col], inplace=True)
        else:
            for sub in ['core0', 'core1']:
                key = f'prefetcher_{sub}_{rank}'
                if key not in df.columns:
                    df[key] = 'none'
 
    # Drop metadata / config columns that were dropped during training
    df = df.drop(columns=METADATA_COLUMNS_TO_DROP, errors='ignore')
    df = df.drop(columns=ALL_CONFIG_COLUMNS, errors='ignore')
 
    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if label_encoders_path:
            enc_path = os.path.join(label_encoders_path, f'{col}_encoder.pkl')
            if os.path.exists(enc_path):
                le = joblib.load(enc_path)
                df[col] = safe_encode(le, df[col])
                continue
        # Fallback: plain ordinal encode
        df[col] = df[col].astype('category').cat.codes
 
    # Keep only numeric columns (same as training)
    df = df[df.select_dtypes(include=[np.number]).columns]
 
    # Align columns to exactly what the scaler saw at fit time.
    # Handles extra columns in new data, missing columns, and different order.
    if scaler is not None and hasattr(scaler, 'feature_names_in_'):
        expected = list(scaler.feature_names_in_)
        missing_cols = [c for c in expected if c not in df.columns]
        extra_cols   = [c for c in df.columns if c not in expected]
        if missing_cols:
            print(f"      WARNING: {len(missing_cols)} feature(s) missing from input -- "
                  f"filling with 0: {missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}")
            for c in missing_cols:
                df[c] = 0.0
        if extra_cols:
            print(f"      INFO: dropping {len(extra_cols)} extra column(s) not seen at "
                  f"fit time: {extra_cols[:5]}{'...' if len(extra_cols) > 5 else ''}")
        df = df[expected]  # reorder + drop extras in one shot
 
    feature_cols = df.columns.tolist()
 
    # Scale / impute
    if scaler is not None:
        X_scaled = scaler.transform(df)
    else:
        X_scaled = df.values
 
    if imputer is not None:
        X_scaled = imputer.transform(X_scaled)
 
    return X_scaled, passthrough, feature_cols
 
 
# ==============================================================================
# Decode predictions back to human-readable values
# ==============================================================================
 
def decode_predictions(y_pred, enc_pf0, enc_pf1, enc_l2, enc_l3):
    """
    Convert raw model output back to human-readable config values.
 
    Handles both model variants:
      - 4-target: btbCore0, btbCore1, prefetcher_core0, prefetcher_core1
      - 6-target: above + l2_size, l3_size
 
    L2/L3 are set to NaN for 4-target models.
    """
    n_targets = y_pred.shape[1]
    results = {}
 
    print(f"      Model outputs {n_targets} target(s) -- "
          + ("4-way (no L2/L3)" if n_targets == 4 else "6-way (with L2/L3)"))
 
    # btbCore0 / btbCore1 -- always present, always numeric
    results['btbCore0_best'] = y_pred[:, 0].astype(float)
    results['btbCore1_best'] = y_pred[:, 1].astype(float)
    results['btbCore0']      = results['btbCore0_best']
    results['btbCore1']      = results['btbCore1_best']
 
    # Prefetcher core0
    if enc_pf0 is not None:
        idx = np.clip(y_pred[:, 2].astype(int), 0, len(enc_pf0.classes_) - 1)
        results['prefetcher_core0_best'] = enc_pf0.classes_[idx]
    else:
        results['prefetcher_core0_best'] = y_pred[:, 2].astype(str)
 
    # Prefetcher core1
    if enc_pf1 is not None:
        idx = np.clip(y_pred[:, 3].astype(int), 0, len(enc_pf1.classes_) - 1)
        results['prefetcher_core1_best'] = enc_pf1.classes_[idx]
    else:
        results['prefetcher_core1_best'] = y_pred[:, 3].astype(str)
 
    # Combined prefetcher string expected by costAnalysis.py
    results['prefetcher'] = [
        f"{p0}-{p1}"
        for p0, p1 in zip(results['prefetcher_core0_best'],
                          results['prefetcher_core1_best'])
    ]
 
    # L2 -- only present in 6-target models
    if n_targets >= 5:
        if enc_l2 is not None:
            idx = np.clip(y_pred[:, 4].astype(int), 0, len(enc_l2.classes_) - 1)
            results['l2_size_best'] = enc_l2.classes_[idx]
            results['L2'] = pd.to_numeric(pd.Series(results['l2_size_best']),
                                          errors='coerce')
        else:
            results['l2_size_best'] = y_pred[:, 4]
            results['L2'] = y_pred[:, 4].astype(float)
    else:
        results['l2_size_best'] = np.nan
        results['L2'] = np.nan
 
    # L3 -- only present in 6-target models
    if n_targets >= 6:
        if enc_l3 is not None:
            idx = np.clip(y_pred[:, 5].astype(int), 0, len(enc_l3.classes_) - 1)
            results['l3_size_best'] = enc_l3.classes_[idx]
            results['L3'] = pd.to_numeric(pd.Series(results['l3_size_best']),
                                          errors='coerce')
        else:
            results['l3_size_best'] = y_pred[:, 5]
            results['L3'] = y_pred[:, 5].astype(float)
    else:
        results['l3_size_best'] = np.nan
        results['L3'] = np.nan
 
    return pd.DataFrame(results)
 
 
# ==============================================================================
# Main prediction pipeline
# ==============================================================================
 
def predict(input_csv, model_dir, output_csv, benchmark=None):
    print("\n" + "=" * 60)
    print("  RF CONFIG PREDICTOR -- inference mode")
    print("=" * 60)
 
    # Load raw data
    print(f"\n[1/4] Loading data from '{input_csv}' ...")
    df_raw = pd.read_csv(input_csv)
    if benchmark:
        df_raw['benchmark'] = benchmark
    print(f"      Shape: {df_raw.shape}")
 
    # Load model + preprocessors
    print(f"\n[2/4] Loading model from '{model_dir}' ...")
    model, scaler, imputer, enc_pf0, enc_pf1, enc_l2, enc_l3 = load_model(model_dir)
    print(f"      n_estimators : {model.n_estimators}")
    print(f"      max_depth    : {model.max_depth}")
 
    # Preprocess
    print("\n[3/4] Preprocessing ...")
    X_scaled, passthrough, feature_cols = preprocess(
        df_raw, scaler, imputer, enc_pf0, enc_pf1, enc_l2, enc_l3
    )
    print(f"      Feature matrix : {X_scaled.shape}")
 
    # Predict
    print("\n[4/4] Running predictions ...")
    y_pred = model.predict(X_scaled)
 
    pred_df = decode_predictions(y_pred, enc_pf0, enc_pf1, enc_l2, enc_l3)
 
    # Assemble output
    output = pd.concat([passthrough.reset_index(drop=True),
                        pred_df.reset_index(drop=True)], axis=1)
 
    # Ensure costAnalysis.py columns come first
    priority_cols = [
        'benchmark', 'period_start', 'period_end',
        'btbCore0', 'btbCore1', 'prefetcher', 'L2', 'L3',
        'btbCore0_prev', 'btbCore1_prev', 'prefetcher_prev', 'L2_prev', 'L3_prev',
        'PPW_best', 'ips',
        'btbCore0_best', 'btbCore1_best',
        'prefetcher_core0_best', 'prefetcher_core1_best',
        'l2_size_best', 'l3_size_best',
    ]
    existing_priority = [c for c in priority_cols if c in output.columns]
    remaining = [c for c in output.columns if c not in existing_priority]
    output = output[existing_priority + remaining]
 
    output.to_csv(output_csv, index=False)
 
    print(f"\n{'=' * 60}")
    print(f"  Done!  {len(output)} predictions saved to '{output_csv}'")
    print(f"  Columns: {list(output.columns)}")
    print("=" * 60)
 
    # Quick prediction distribution summary
    print("\nPrediction distribution:")
    for col in ['btbCore0', 'btbCore1', 'prefetcher']:
        if col in output.columns:
            print(f"  {col}: {output[col].value_counts().to_dict()}")
    for col in ['L2', 'L3']:
        if col in output.columns and output[col].notna().any():
            print(f"  {col}: {output[col].value_counts().to_dict()}")
 
    return output
 
 
# ==============================================================================
# CLI
# ==============================================================================
 
def main():
    parser = argparse.ArgumentParser(
        description='Run RF config predictions and output a costAnalysis-ready CSV'
    )
    parser.add_argument('--input',      required=True,
                        help='Path to input CSV (same format as training data)')
    parser.add_argument('--model-dir',  required=True,
                        help='Directory containing saved RF model (.pkl files)')
    parser.add_argument('--output',     required=True,
                        help='Output CSV path')
    parser.add_argument('--benchmark',  default=None,
                        help='Benchmark label to tag rows with (e.g. barnes)')
    args = parser.parse_args()
 
    predict(
        input_csv=args.input,
        model_dir=args.model_dir,
        output_csv=args.output,
        benchmark=args.benchmark,
    )
 
 
if __name__ == '__main__':
    main()
 
