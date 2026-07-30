"""
predict_config.py  (7-way)
=================
Load a saved 7-way RF model (L2 core0/1, L3, BTB core0/1, Prefetcher core0/1)
and run predictions on new data, applying the same preprocessing pipeline as
rf_7way_config_predictor.py. Outputs a CSV ready to be consumed by
costAnalysis.py.

Column naming is resolved the same flexible, case/separator-insensitive way
as the training and post-hoc-analysis scripts (find_col / CONFIG_DIMENSIONS),
so this stays in sync if the naming convention drifts across pipeline runs.

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
# Column name resolution helpers  (must match rf_7way_config_predictor.py /
# post_hoc_analysis.py)
# ==============================================================================

def _normalize_name(s):
    return s.lower().replace('_', '').replace(' ', '').replace('.', '')


def find_col(df, expected_name):
    """Find a column matching expected_name, ignoring case/separators."""
    norm = _normalize_name
    for col in df.columns:
        if norm(col) == norm(expected_name):
            return col
    for col in df.columns:
        if norm(expected_name) in norm(col):
            return col
    return None


CONFIG_DIMENSIONS = {
    'l2_core0':        'L2core0',
    'l2_core1':        'L2core1',
    'l3':              'L3',
    'btb_core0':       'BTBcore0',
    'btb_core1':       'BTBcore1',
    'prefetch_core0':  'Prefetchcore0',
    'prefetch_core1':  'Prefetchcore1',
}
RANKS = ['best', '2nd', '3rd']

# Canonical target order — MUST match TARGET_KEYS in rf_7way_config_predictor.py,
# since that's the column order the model's .predict() returns.
TARGET_KEYS = ['l2_core0', 'l2_core1', 'l3', 'btb_core0', 'btb_core1',
               'prefetch_core0', 'prefetch_core1']
NUMERIC_TARGET_KEYS = ['l2_core0', 'l2_core1', 'l3', 'btb_core0', 'btb_core1']
PREFETCH_KEYS = ['prefetch_core0', 'prefetch_core1']


def canonical_name(key, rank):
    return f"{key}__{rank}"


def try_resolve_config_columns(df):
    """
    Lenient version of the training script's resolve_config_columns: returns
    whatever (key, rank) -> actual_column mappings it can find, WITHOUT
    raising if some are missing. New/production data being scored typically
    won't have '_best' / '_2nd' / '_3rd' ground-truth columns at all — that's
    expected and fine, we just won't have anything to drop as a feature leak
    for those.
    """
    resolved = {}
    for key, template in CONFIG_DIMENSIONS.items():
        for rank in RANKS:
            actual = find_col(df, f"{template}_{rank}")
            if actual is not None:
                resolved[(key, rank)] = actual
    ppw_cols = {}
    for rank in RANKS:
        actual = find_col(df, f"PPW_{rank}")
        if actual is not None:
            ppw_cols[rank] = actual
    return resolved, ppw_cols


# "_prev" (previous-interval actual config) column candidates — passed
# through to costAnalysis.py, which needs them to detect reconfiguration
# (config changed vs kept) and compute reconfig energy per dimension.
PREV_DIMENSIONS = {
    'l2_core0':       ['L2 core 0_prev', 'L2core0_prev', 'l2Core0_prev'],
    'l2_core1':       ['L2 core 1_prev', 'L2core1_prev', 'l2Core1_prev'],
    'l3':             ['L3_prev', 'l3Size_prev'],
    'btb_core0':      ['BTB core 0_prev', 'BTBcore0_prev', 'btbCore0_prev'],
    'btb_core1':      ['BTB core 1_prev', 'BTBcore1_prev', 'btbCore1_prev'],
    'prefetch_core0': ['Prefetch core 0_prev', 'prefetcher_core0_prev', 'Prefetch_core0_prev', 'Prefetch_prev'],
    'prefetch_core1': ['Prefetch core 1_prev', 'prefetcher_core1_prev', 'Prefetch_core1_prev'],
}

# Other fields costAnalysis.py expects, resolved flexibly rather than assumed
# to have one exact spelling.
OTHER_PASSTHROUGH_CANDIDATES = {
    'benchmark':    ['benchmark'],
    'period_start': ['period_start'],
    'period_end':   ['period_end'],
    'PPW_best':     ['PPW_best', 'PPW__best'],
    'ips_prev':     ['ips_prev'],
}

METADATA_COLUMNS_TO_DROP = [
    'best-config', 'file', 'file_prev', 'period_start', 'period_end',
    'period_start_prev', 'period_end_prev',
    'directory_perf_prev', 'leaf_dir_prev', 'directory_power_prev',
    'leaf_dir_perf_prev', 'leaf_dir_power_prev', 'period_start_val_prev',
    'period_end_val_perf_prev', 'period_start_val_perf_prev',
    'period_start_val_power_prev', 'period_end_val_power_prev',
    'Diff_best_2nd', 'Diff_best_3rd',
]


# ==============================================================================
# Helpers
# ==============================================================================

def split_prefetcher(series):
    """
    Back-compat: split a legacy combined 'type0-type1' prefetcher column into
    two per-core columns, in case an older-format input CSV is passed in.
    """
    split = series.fillna('none-none').astype(str).str.split('-', n=1, expand=True)
    core0 = split[0].str.strip()
    core1 = split[1].str.strip() if split.shape[1] > 1 else core0
    return core0, core1


def safe_encode(encoder, series):
    """Encode a Series with a fitted LabelEncoder; unseen/NaN -> 'none'."""
    valid = set(encoder.classes_)
    mapped = (series.fillna('none').astype(str).str.strip().str.lower()
              .apply(lambda x: x if x in valid else 'none'))
    return encoder.transform(mapped)


# ==============================================================================
# Model loading
# ==============================================================================

def load_model(model_dir):
    """Load the most recent saved 7-way RF model and its preprocessors."""
    pkls = glob.glob(os.path.join(model_dir, 'rf_7way_config_predictor_*.pkl'))
    if not pkls:
        # fall back to an older-generation model so this script degrades
        # gracefully rather than hard-failing on a stale saved_models dir
        pkls = glob.glob(os.path.join(model_dir, 'rf_4way_config_predictor_*.pkl'))
    pkls = [p for p in pkls
            if not any(tag in p for tag in ['_scaler', '_encoder', '_imputer'])]

    if not pkls:
        raise FileNotFoundError(f"No model .pkl found in '{model_dir}'")

    pkls.sort()
    model_path = pkls[-1]
    base = model_path.replace('.pkl', '')

    print(f"  Loading model : {model_path}")
    model = joblib.load(model_path)

    def _load(suffix, required=False):
        p = f"{base}{suffix}.pkl"
        if os.path.exists(p):
            print(f"  Loading        : {p}")
            return joblib.load(p)
        msg = f"  {'ERROR' if required else 'WARNING'}: missing {p}"
        print(msg)
        if required:
            raise FileNotFoundError(p)
        return None

    scaler  = _load('_scaler', required=True)
    # rf_7way_config_predictor.py fits a SimpleImputer at train time but does
    # NOT joblib.dump it, so this will typically come back None — we handle
    # that below by falling back to a fresh mean-imputer fit on the scored
    # data itself, which is the closest available approximation.
    imputer = _load('_imputer')
    enc_pf0 = _load('_prefetcher_core0_encoder', required=True)
    enc_pf1 = _load('_prefetcher_core1_encoder', required=True)

    return model, scaler, imputer, enc_pf0, enc_pf1


# ==============================================================================
# Preprocessing  (mirrors rf_7way_config_predictor.py)
# ==============================================================================

def build_passthrough(df):
    """Assemble the columns costAnalysis.py needs, resolved flexibly."""
    passthrough = pd.DataFrame(index=df.index)

    for out_name, candidates in OTHER_PASSTHROUGH_CANDIDATES.items():
        col = None
        for c in candidates:
            found = find_col(df, c)
            if found:
                col = found
                break
        if col is not None:
            passthrough[out_name] = df[col]

    prev_resolved = {}
    for key, candidates in PREV_DIMENSIONS.items():
        col = None
        for c in candidates:
            found = find_col(df, c)
            if found:
                col = found
                break
        prev_resolved[key] = col
        if col is not None:
            passthrough[f'{key}_prev'] = df[col]

    missing_prev = [k for k, v in prev_resolved.items() if v is None]
    if missing_prev:
        print(f"      INFO: previous-config columns not found for: {missing_prev} "
              f"-- costAnalysis.py reconfig-cost analysis for those dims will be skipped.")

    return passthrough


def preprocess(df_raw, scaler, imputer, enc_pf0, enc_pf1):
    """
    Apply the same preprocessing steps as rf_7way_config_predictor.py to raw
    input data.

    Returns
    -------
    X_scaled : np.ndarray  - ready to pass to model.predict()
    passthrough : pd.DataFrame - columns to keep for the output CSV
    feature_cols : list[str] - ordered feature names (for diagnostics)
    """
    df = df_raw.copy()

    # Save passthrough columns before any modification
    passthrough = build_passthrough(df)

    # Back-compat: split a legacy combined 'prefetcher_<rank>' column into
    # per-core columns if the new per-core columns aren't already present.
    for rank in RANKS:
        combined_col = find_col(df, f'prefetcher_{rank}')
        core0_present = find_col(df, f'Prefetchcore0_{rank}') is not None
        core1_present = find_col(df, f'Prefetchcore1_{rank}') is not None
        if combined_col is not None and not (core0_present and core1_present):
            c0, c1 = split_prefetcher(df[combined_col])
            df[f'prefetcher_core0_{rank}'] = c0
            df[f'prefetcher_core1_{rank}'] = c1
            df.drop(columns=[combined_col], inplace=True)

    # Drop metadata and any resolvable *_best/*_2nd/*_3rd config columns so
    # ground-truth (if present in this input, e.g. for offline evaluation)
    # can't leak into the feature set.
    resolved_cols, ppw_cols = try_resolve_config_columns(df)
    config_cols_to_drop = list(resolved_cols.values()) + list(ppw_cols.values())
    df = df.drop(columns=METADATA_COLUMNS_TO_DROP, errors='ignore')
    df = df.drop(columns=config_cols_to_drop, errors='ignore')

    # Encode categorical features (plain ordinal encode — matches training's
    # per-column LabelEncoder-at-fit-time behavior closely enough for
    # inference-time features, which aren't targets)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes

    # Keep only numeric columns (same as training)
    df = df[df.select_dtypes(include=[np.number]).columns]

    # Align columns to exactly what the scaler saw at fit time.
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

    # Scale
    if scaler is not None:
        X_scaled = scaler.transform(df)
    else:
        X_scaled = df.values

    # Impute
    if imputer is not None:
        X_scaled = imputer.transform(X_scaled)
    elif np.isnan(X_scaled).any():
        print("      WARNING: no saved imputer found (rf_7way_config_predictor.py "
              "doesn't persist one) -- fitting a mean-imputer on this batch as a "
              "best-effort fallback. Results may differ slightly from a training-time "
              "imputer fit on the full training distribution.")
        from sklearn.impute import SimpleImputer
        X_scaled = SimpleImputer(strategy='mean').fit_transform(X_scaled)

    return X_scaled, passthrough, feature_cols


# ==============================================================================
# Decode predictions back to human-readable values
# ==============================================================================

def decode_predictions(y_pred, enc_pf0, enc_pf1):
    """
    Convert raw model output back to human-readable config values.

    Handles:
      - 7-target (current): l2_core0, l2_core1, l3, btb_core0, btb_core1,
        prefetch_core0, prefetch_core1
      - 4-target (legacy):  btb_core0, btb_core1, prefetch_core0, prefetch_core1
        -- L2/L3 are set to NaN for these older models.
    """
    n_targets = y_pred.shape[1]
    results = {}

    if n_targets not in (4, 7):
        print(f"      WARNING: model outputs {n_targets} target(s) -- expected 4 (legacy) "
              f"or 7 (current). Attempting best-effort decode assuming the current "
              f"7-target order truncated/padded; verify results carefully.")

    print(f"      Model outputs {n_targets} target(s) -- "
          + ("7-way (L2 x2, L3, BTB x2, Prefetcher x2)" if n_targets == 7
             else "4-way legacy (BTB x2, Prefetcher x2 only)"))

    if n_targets == 7:
        idx = {key: i for i, key in enumerate(TARGET_KEYS)}
        results['l2_core0'] = y_pred[:, idx['l2_core0']].astype(float)
        results['l2_core1'] = y_pred[:, idx['l2_core1']].astype(float)
        results['l3']       = y_pred[:, idx['l3']].astype(float)
        results['btb_core0'] = y_pred[:, idx['btb_core0']].astype(float)
        results['btb_core1'] = y_pred[:, idx['btb_core1']].astype(float)
        pf0_raw = y_pred[:, idx['prefetch_core0']].astype(int)
        pf1_raw = y_pred[:, idx['prefetch_core1']].astype(int)
    else:  # legacy 4-target: btb_core0, btb_core1, prefetch_core0, prefetch_core1
        results['l2_core0'] = np.nan
        results['l2_core1'] = np.nan
        results['l3']       = np.nan
        results['btb_core0'] = y_pred[:, 0].astype(float)
        results['btb_core1'] = y_pred[:, 1].astype(float)
        pf0_raw = y_pred[:, 2].astype(int)
        pf1_raw = y_pred[:, 3].astype(int)

    if enc_pf0 is not None:
        idx0 = np.clip(pf0_raw, 0, len(enc_pf0.classes_) - 1)
        results['prefetch_core0'] = enc_pf0.classes_[idx0]
    else:
        results['prefetch_core0'] = pf0_raw.astype(str)

    if enc_pf1 is not None:
        idx1 = np.clip(pf1_raw, 0, len(enc_pf1.classes_) - 1)
        results['prefetch_core1'] = enc_pf1.classes_[idx1]
    else:
        results['prefetch_core1'] = pf1_raw.astype(str)

    # Combined prefetcher string, kept for any downstream tooling that still
    # expects the older single-column 'type0-type1' format.
    results['prefetcher'] = [
        f"{p0}-{p1}"
        for p0, p1 in zip(results['prefetch_core0'], results['prefetch_core1'])
    ]

    df_out = pd.DataFrame(results)

    # costAnalysis.py-friendly aliases (short, capitalized names matching the
    # rest of the pipeline's convention)
    df_out['L2core0']  = df_out['l2_core0']
    df_out['L2core1']  = df_out['l2_core1']
    df_out['L3']       = df_out['l3']
    df_out['btbCore0'] = df_out['btb_core0']
    df_out['btbCore1'] = df_out['btb_core1']

    return df_out


# ==============================================================================
# Main prediction pipeline
# ==============================================================================

def predict(input_csv, model_dir, output_csv, benchmark=None):
    print("\n" + "=" * 60)
    print("  RF CONFIG PREDICTOR (7-way) -- inference mode")
    print("=" * 60)

    # Load raw data
    print(f"\n[1/4] Loading data from '{input_csv}' ...")
    df_raw = pd.read_csv(input_csv)
    if benchmark:
        df_raw['benchmark'] = benchmark
    print(f"      Shape: {df_raw.shape}")

    # Load model + preprocessors
    print(f"\n[2/4] Loading model from '{model_dir}' ...")
    model, scaler, imputer, enc_pf0, enc_pf1 = load_model(model_dir)
    print(f"      n_estimators : {model.n_estimators}")
    print(f"      max_depth    : {model.max_depth}")

    # Preprocess
    print("\n[3/4] Preprocessing ...")
    X_scaled, passthrough, feature_cols = preprocess(
        df_raw, scaler, imputer, enc_pf0, enc_pf1
    )
    print(f"      Feature matrix : {X_scaled.shape}")

    # Predict
    print("\n[4/4] Running predictions ...")
    y_pred = model.predict(X_scaled)

    pred_df = decode_predictions(y_pred, enc_pf0, enc_pf1)

    # Assemble output
    output = pd.concat([passthrough.reset_index(drop=True),
                        pred_df.reset_index(drop=True)], axis=1)

    # Ensure costAnalysis.py columns come first
    priority_cols = [
        'benchmark', 'period_start', 'period_end',
        'L2core0', 'L2core1', 'L3', 'btbCore0', 'btbCore1', 'prefetcher',
        'l2_core0_prev', 'l2_core1_prev', 'l3_prev',
        'btb_core0_prev', 'btb_core1_prev',
        'prefetch_core0_prev', 'prefetch_core1_prev',
        'PPW_best', 'ips_prev',
        'l2_core0', 'l2_core1', 'l3', 'btb_core0', 'btb_core1',
        'prefetch_core0', 'prefetch_core1',
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
    for col in ['L2core0', 'L2core1', 'L3']:
        if col in output.columns and output[col].notna().any():
            print(f"  {col}: {output[col].value_counts().to_dict()}")

    return output


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run 7-way RF config predictions and output a costAnalysis-ready CSV'
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
