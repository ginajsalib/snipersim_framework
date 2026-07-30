"""
costAnalysis.py  (7-way)
===============
Calculate RF model inference cost and reconfiguration cost for post-silicon
customization using clock gating (McPAT/Gainestown parameters), for the
7-way model (L2 core0/1, L3, BTB core0/1, Prefetcher core0/1).

Reconfiguration cost is now computed per-row against that row's own actual
previous-interval config (the split per-core "_prev" columns), rather than by
diffing consecutive rows in the sorted dataframe -- this reflects what's
really "currently running" before the model's new prediction takes effect,
and works whether the input comes straight from predict_config.py's output
or from a raw training-style CSV with the older "_prev" naming.

Also reports a Net PPW Gain % -- how much PPW you actually keep (or lose) by
switching to the model's chosen config and paying inference + reconfig
overhead, versus just staying on the previous config.

Based on: Weston, K., et al. (2023). Post-Silicon Customization Using Deep
Neural Networks. ARCS.

Usage:
    python costAnalysis.py --model-dir saved_models/ --benchmark barnes --output cost_analysis_barnes.csv
    python costAnalysis.py --cost-csv cost_analysis_barnes.csv --report cost_report.txt
"""

import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ==============================================================================
# COLUMN NAME RESOLUTION HELPERS  (matches rf_7way_config_predictor.py /
# post_hoc_analysis.py / predict_config.py)
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


def resolve_first_present(df, candidates):
    for c in candidates:
        found = find_col(df, c)
        if found:
            return found
    return None


# Candidates for the model's CHOSEN/predicted config for the current interval.
# Covers both predict_config.py's output naming and raw training-CSV naming.
CURRENT_DIMENSIONS = {
    'l2_core0':       ['L2core0', 'l2_core0', 'L2 core 0', 'l2Core0'],
    'l2_core1':       ['L2core1', 'l2_core1', 'L2 core 1', 'l2Core1'],
    'l3':             ['L3', 'l3', 'l3Size'],
    'btb_core0':      ['btbCore0', 'btb_core0', 'BTB core 0', 'BTBcore0'],
    'btb_core1':      ['btbCore1', 'btb_core1', 'BTB core 1', 'BTBcore1'],
    'prefetch_core0': ['prefetch_core0', 'prefetcher_core0', 'Prefetch core 0', 'Prefetchcore0'],
    'prefetch_core1': ['prefetch_core1', 'prefetcher_core1', 'Prefetch core 1', 'Prefetchcore1'],
}

# Candidates for the split per-core "_prev" columns -- the actual config that
# was running immediately before this interval. predict_config.py's own
# passthrough naming ('l2_core0_prev', etc.) is tried first.
PREV_DIMENSIONS = {
    'l2_core0':       ['l2_core0_prev', 'L2 core 0_prev', 'L2core0_prev', 'l2Core0_prev'],
    'l2_core1':       ['l2_core1_prev', 'L2 core 1_prev', 'L2core1_prev', 'l2Core1_prev'],
    'l3':             ['l3_prev', 'L3_prev', 'l3Size_prev'],
    'btb_core0':      ['btb_core0_prev', 'BTB core 0_prev', 'BTBcore0_prev', 'btbCore0_prev'],
    'btb_core1':      ['btb_core1_prev', 'BTB core 1_prev', 'BTBcore1_prev', 'btbCore1_prev'],
    'prefetch_core0': ['prefetch_core0_prev', 'Prefetch core 0_prev', 'prefetcher_core0_prev', 'Prefetch_core0_prev'],
    'prefetch_core1': ['prefetch_core1_prev', 'Prefetch core 1_prev', 'prefetcher_core1_prev', 'Prefetch_core1_prev'],
}

PPW_BEST_CANDIDATES = ['PPW_best', 'PPW__best', 'ppw_best']
PPW_PREV_CANDIDATES = ['ppw_prev', 'PPW_prev', 'PPW__prev']
IPS_PREV_CANDIDATES = ['ips_prev']


# ==============================================================================
# GAINESTOWN/McPAT TECHNOLOGY PARAMETERS (from power.xml)
# ==============================================================================

TECH_PARAMS = {
    # Technology node
    'tech_node_nm': 45,
    'clock_rate_mhz': 2660,
    'vdd': 1.2,
    'device_type': 'HP',  # High Performance
    'power_gating_enabled': True,

    # Component defaults (from Gainestown power.xml)
    'btb_entries_default': 18944,
    'l2_capacity_kb': 256,       # 262144 bytes
    'l3_capacity_kb': 8192,      # 8388608 bytes
    'l2_assoc': 8,
    'l3_assoc': 16,
    'cache_line_bytes': 64,

    # Energy parameters (45nm HP, 1.2V)
    # Derived from McPAT gate capacitance values
    'E_gate_per_btb_entry_pJ': 0.05,     # Clock gating per BTB entry
    'E_gate_per_l2_kb_pJ': 2.5,          # Clock gating per KB of L2
    'E_gate_per_l3_kb_pJ': 1.8,          # Clock gating per KB of L3
    'E_gate_per_prefetcher_pJ': 15.0,    # Prefetcher state machine gating (per core)

    # RF inference energy (per operation)
    'E_rf_inst_pJ': 0.5,                 # Instruction fetch
    'E_rf_cmp_pJ': 0.1,                  # Tree node comparison
    'E_rf_mem_pJ': 1.0,                  # SRAM access (feature/memo lookup)
}


def print_tech_params():
    """Print all technology parameters used in cost calculation."""
    print("\n" + "=" * 70)
    print("  McPAT/GAINESTOWN PARAMETERS USED")
    print("=" * 70)
    print(f"\n  Technology Node:      {TECH_PARAMS['tech_node_nm']} nm")
    print(f"  Core Clock:           {TECH_PARAMS['clock_rate_mhz']} MHz")
    print(f"  Supply Voltage (Vdd): {TECH_PARAMS['vdd']} V")
    print(f"  Device Type:          {TECH_PARAMS['device_type']} (High Performance)")
    print(f"  Power Gating:         {'Enabled' if TECH_PARAMS['power_gating_enabled'] else 'Disabled'}")

    print(f"\n  Component Parameters (from Gainestown power.xml):")
    print(f"    BTB entries (default):  {TECH_PARAMS['btb_entries_default']}")
    print(f"    L2 capacity:            {TECH_PARAMS['l2_capacity_kb']} KB ({TECH_PARAMS['l2_capacity_kb']*1024} bytes)")
    print(f"    L3 capacity:            {TECH_PARAMS['l3_capacity_kb']} KB ({TECH_PARAMS['l3_capacity_kb']*1024} bytes)")
    print(f"    L2 associativity:       {TECH_PARAMS['l2_assoc']}-way")
    print(f"    L3 associativity:       {TECH_PARAMS['l3_assoc']}-way")
    print(f"    Cache line size:        {TECH_PARAMS['cache_line_bytes']} B")

    print(f"\n  Energy Parameters (45nm HP, {TECH_PARAMS['vdd']}V):")
    print(f"    E_gate_per_btb_entry:   {TECH_PARAMS['E_gate_per_btb_entry_pJ']:.3f} pJ")
    print(f"    E_gate_per_l2_kb:       {TECH_PARAMS['E_gate_per_l2_kb_pJ']:.3f} pJ")
    print(f"    E_gate_per_l3_kb:       {TECH_PARAMS['E_gate_per_l3_kb_pJ']:.3f} pJ")
    print(f"    E_gate_per_prefetcher:  {TECH_PARAMS['E_gate_per_prefetcher_pJ']:.3f} pJ  (per core)")
    print(f"    E_rf_inst (inference):  {TECH_PARAMS['E_rf_inst_pJ']:.3f} pJ")
    print(f"    E_rf_cmp (tree node):   {TECH_PARAMS['E_rf_cmp_pJ']:.3f} pJ")
    print(f"    E_rf_mem (SRAM access): {TECH_PARAMS['E_rf_mem_pJ']:.3f} pJ")


# ==============================================================================
# Reconfiguration energy — vectorised, per-row vs that row's own "_prev" cols
# ==============================================================================

def resolve_reconfig_columns(df):
    """
    Resolve both the current-chosen-config columns and the split per-core
    "_prev" columns. Returns (curr_col_map, prev_col_map), each
    {dimension_key: actual_column_name_or_None}.
    """
    curr_map = {key: resolve_first_present(df, cands)
                for key, cands in CURRENT_DIMENSIONS.items()}
    prev_map = {key: resolve_first_present(df, cands)
                for key, cands in PREV_DIMENSIONS.items()}
    return curr_map, prev_map


def calc_reconfig_energy_vectorized(df, curr_map, prev_map, tech_params):
    """
    Vectorised reconfiguration-energy calculation: for every row, compare the
    model's chosen config for that interval against that row's own actual
    previous-interval config (the split per-core "_prev" columns) -- NOT the
    previous row in the sorted dataframe.

    Returns a dict of numpy arrays (one entry per row of df):
      reconfig_energy_pJ, config_changed, l2_changed, l3_changed,
      btb_changed, prefetcher_changed, has_prev_data
    """
    n = len(df)
    energy   = np.zeros(n)
    l2_changed  = np.zeros(n, dtype=bool)
    l3_changed  = np.zeros(n, dtype=bool)
    btb_changed = np.zeros(n, dtype=bool)
    pf_changed  = np.zeros(n, dtype=bool)
    has_prev    = np.ones(n, dtype=bool)

    def numeric(col):
        return pd.to_numeric(df[col], errors='coerce').values.astype(float) if col else None

    def categorical(col):
        return df[col].astype(str).str.strip().str.lower().values if col else None

    # --- L2 (core0 + core1) ---
    for core in ['l2_core0', 'l2_core1']:
        c_col, p_col = curr_map.get(core), prev_map.get(core)
        if c_col is None or p_col is None:
            has_prev &= False if p_col is None else has_prev
            continue
        curr_v, prev_v = numeric(c_col), numeric(p_col)
        valid = ~np.isnan(curr_v) & ~np.isnan(prev_v)
        diff  = np.where(valid, np.abs(curr_v - prev_v), 0.0)
        changed_here = valid & (diff > 0)
        energy += np.where(changed_here, tech_params['E_gate_per_l2_kb_pJ'] * diff, 0.0)
        l2_changed |= changed_here

    # --- L3 (single) ---
    c_col, p_col = curr_map.get('l3'), prev_map.get('l3')
    if c_col is not None and p_col is not None:
        curr_v, prev_v = numeric(c_col), numeric(p_col)
        valid = ~np.isnan(curr_v) & ~np.isnan(prev_v)
        diff  = np.where(valid, np.abs(curr_v - prev_v), 0.0)
        changed_here = valid & (diff > 0)
        energy += np.where(changed_here, tech_params['E_gate_per_l3_kb_pJ'] * diff, 0.0)
        l3_changed |= changed_here
    else:
        has_prev &= False if p_col is None else has_prev

    # --- BTB (core0 + core1) ---
    for core in ['btb_core0', 'btb_core1']:
        c_col, p_col = curr_map.get(core), prev_map.get(core)
        if c_col is None or p_col is None:
            has_prev &= False if p_col is None else has_prev
            continue
        curr_v, prev_v = numeric(c_col), numeric(p_col)
        valid = ~np.isnan(curr_v) & ~np.isnan(prev_v)
        diff  = np.where(valid, np.abs(curr_v - prev_v), 0.0)
        changed_here = valid & (diff > 0)
        energy += np.where(changed_here, tech_params['E_gate_per_btb_entry_pJ'] * diff, 0.0)
        btb_changed |= changed_here

    # --- Prefetcher (core0 + core1, discrete -- each changed core is its own
    #     gating event, so both cores switching costs 2x one core switching) ---
    for core in ['prefetch_core0', 'prefetch_core1']:
        c_col, p_col = curr_map.get(core), prev_map.get(core)
        if c_col is None or p_col is None:
            has_prev &= False if p_col is None else has_prev
            continue
        curr_v, prev_v = categorical(c_col), categorical(p_col)
        valid = (curr_v != 'nan') & (prev_v != 'nan')
        changed_here = valid & (curr_v != prev_v)
        energy += np.where(changed_here, tech_params['E_gate_per_prefetcher_pJ'], 0.0)
        pf_changed |= changed_here

    config_changed = l2_changed | l3_changed | btb_changed | pf_changed

    return {
        'reconfig_energy_pJ': energy,
        'config_changed':     config_changed,
        'l2_changed':         l2_changed,
        'l3_changed':         l3_changed,
        'btb_changed':        btb_changed,
        'prefetcher_changed': pf_changed,
        'has_prev_data':      has_prev,
    }


def calc_inference_energy(model, tech_params):
    """
    Calculate RF model inference energy per prediction, introspected from the
    actual loaded model rather than hardcoded stand-ins.

    Args:
        model: Trained RandomForest model
        tech_params: Technology parameters dict

    Returns:
        (inference_energy_pJ, breakdown_dict)
    """
    n_estimators = model.n_estimators
    max_depth = model.max_depth if model.max_depth else 25
    if hasattr(model, 'n_features_in_'):
        n_features = model.n_features_in_
    elif hasattr(model, 'estimators_') and len(model.estimators_) > 0:
        n_features = len(model.estimators_[0].feature_importances_)
    else:
        n_features = 130  # fallback if model internals aren't introspectable

    # Estimate operations per inference
    n_comparisons = n_estimators * max_depth  # Each tree traverses depth nodes
    n_mem_accesses = n_estimators * n_features  # Feature lookups
    n_instructions = n_comparisons * 3  # Rough estimate: 3 insts per comparison

    # Energy calculation
    e_inst = n_instructions * tech_params['E_rf_inst_pJ']
    e_cmp = n_comparisons * tech_params['E_rf_cmp_pJ']
    e_mem = n_mem_accesses * tech_params['E_rf_mem_pJ']

    return e_inst + e_cmp + e_mem, {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'n_features': n_features,
        'n_comparisons': n_comparisons,
        'n_mem_accesses': n_mem_accesses,
        'e_inst': e_inst,
        'e_cmp': e_cmp,
        'e_mem': e_mem,
    }


def load_model(model_dir):
    """Load the most recent saved RF model from directory (7-way, falling
    back to a legacy 4-way model if no 7-way model is found)."""
    pkls = glob.glob(os.path.join(model_dir, 'rf_7way_config_predictor_*.pkl'))
    prefix = 'rf_7way_config_predictor_'
    if not pkls:
        pkls = glob.glob(os.path.join(model_dir, 'rf_4way_config_predictor_*.pkl'))
        prefix = 'rf_4way_config_predictor_'
    pkls = [p for p in pkls if '_scaler' not in p and '_encoder' not in p and '_imputer' not in p]

    if not pkls:
        raise FileNotFoundError(f"No model files found in {model_dir}")

    pkls.sort()
    latest = pkls[-1]
    model = joblib.load(latest)

    ts = os.path.basename(latest).replace(prefix, '').replace('.pkl', '')
    return model, ts, latest


def load_config_data(benchmark, input_csv=None):
    """Load configuration data for a benchmark."""
    if input_csv and os.path.exists(input_csv):
        df = pd.read_csv(input_csv)
        print(f"  Loaded {len(df)} rows from {input_csv}")
        return df

    # Default paths for training data
    default_paths = {
        'barnes': '/home/gina/Desktop/snipersim_framework/pythonScripts/barnes/train_with_top3_barnes.csv',
        'cholesky': '/home/gina/Desktop/snipersim_framework/pythonScripts/cholesky/train_with_top3_cholesky.csv',
        'fft': '/home/gina/Desktop/snipersim_framework/pythonScripts/fft/train_with_top3_fft.csv',
    }

    if benchmark in default_paths and os.path.exists(default_paths[benchmark]):
        df = pd.read_csv(default_paths[benchmark])
        print(f"  Loaded {len(df)} rows from default path")
        return df

    raise FileNotFoundError(f"No data found for benchmark '{benchmark}'")


def run_cost_analysis(benchmark, model_dir, output_csv, input_csv=None):
    """
    Main cost analysis function.

    Args:
        benchmark: Benchmark name
        model_dir: Directory containing saved model
        output_csv: Output CSV path
        input_csv: Optional input CSV with config data
    """
    print(f"\n{'='*70}")
    print(f"  COST ANALYSIS: {benchmark.upper()}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*70)

    # Print technology parameters
    print_tech_params()

    # Load model
    print(f"\n{'='*70}")
    print("  LOADING MODEL")
    print('='*70)
    model, ts, model_path = load_model(model_dir)
    print(f"  Model timestamp: {ts}")
    print(f"  Model path: {model_path}")
    print(f"  n_estimators: {model.n_estimators}")
    print(f"  max_depth: {model.max_depth}")

    # Load config data
    print(f"\n{'='*70}")
    print("  LOADING CONFIGURATION DATA")
    print('='*70)
    df = load_config_data(benchmark, input_csv)

    # Sort by period for sequential analysis
    if 'period_start' in df.columns:
        df = df.sort_values('period_start').reset_index(drop=True)
    else:
        df = df.sort_index().reset_index(drop=True)

    # Resolve current-config and split per-core "_prev" columns
    curr_map, prev_map = resolve_reconfig_columns(df)
    print(f"\n  Resolved current-config columns: {curr_map}")
    print(f"  Resolved previous-config columns (per core): {prev_map}")
    missing_prev = [k for k, v in prev_map.items() if v is None]
    if missing_prev:
        print(f"  [WARN] No previous-config column found for: {missing_prev} "
              f"-- reconfig cost for those dimensions will be treated as 0/unchanged.")

    ppw_best_col = resolve_first_present(df, PPW_BEST_CANDIDATES)
    ppw_prev_col = resolve_first_present(df, PPW_PREV_CANDIDATES)
    ips_prev_col = resolve_first_present(df, IPS_PREV_CANDIDATES)

    # Calculate inference energy
    print(f"\n{'='*70}")
    print("  INFERENCE ENERGY CALCULATION")
    print('='*70)
    inf_energy, inf_breakdown = calc_inference_energy(model, TECH_PARAMS)

    print(f"\n  RF Model Structure:")
    print(f"    n_estimators:     {inf_breakdown['n_estimators']}")
    print(f"    max_depth:        {inf_breakdown['max_depth']}")
    print(f"    n_features:       {inf_breakdown['n_features']}")
    print(f"    n_comparisons:    {inf_breakdown['n_comparisons']:,}")
    print(f"    n_mem_accesses:   {inf_breakdown['n_mem_accesses']:,}")

    print(f"\n  Inference Energy Breakdown:")
    print(f"    Instruction fetch:  {inf_breakdown['e_inst']:.2f} pJ")
    print(f"    Tree comparisons:   {inf_breakdown['e_cmp']:.2f} pJ")
    print(f"    Memory accesses:    {inf_breakdown['e_mem']:.2f} pJ")
    print(f"    Total inference:    {inf_energy:.2f} pJ ({inf_energy/1e6:.4f} μJ)")

    # Calculate reconfiguration energy — vectorised, per row vs that row's
    # own "_prev" columns (NOT the previous row in the dataframe)
    print(f"\n{'='*70}")
    print("  RECONFIGURATION ENERGY CALCULATION")
    print('='*70)

    df['inference_energy_pJ'] = inf_energy

    reconfig = calc_reconfig_energy_vectorized(df, curr_map, prev_map, TECH_PARAMS)
    for col, arr in reconfig.items():
        df[col] = arr

    valid_mask = df['has_prev_data'].values
    n_intervals = int(valid_mask.sum())
    n_changed = int((df['config_changed'] & df['has_prev_data']).sum())

    print(f"\n  Reconfiguration Summary:")
    print(f"    Total rows:              {len(df)}")
    print(f"    Rows with prev-config data: {n_intervals}")
    if n_intervals > 0:
        print(f"    Intervals changed:   {n_changed} ({n_changed/n_intervals*100:.1f}%)")
        print(f"    Intervals unchanged: {n_intervals - n_changed} ({(n_intervals-n_changed)/n_intervals*100:.1f}%)")

        print(f"\n  Component Change Frequency:")
        for comp_col, label in [('l2_changed', 'L2 cache (either core)'),
                                 ('l3_changed', 'L3 cache'),
                                 ('btb_changed', 'BTB (either core)'),
                                 ('prefetcher_changed', 'Prefetcher (either core)')]:
            n_comp = int((df[comp_col] & df['has_prev_data']).sum())
            print(f"    {label:<26} {n_comp:5d} ({n_comp/n_intervals*100:.1f}%)")

        total_reconfig_energy = float(df.loc[valid_mask, 'reconfig_energy_pJ'].sum())
        print(f"\n  Reconfiguration Energy:")
        print(f"    Total:            {total_reconfig_energy:.2f} pJ")
        if n_changed > 0:
            avg_reconfig = df.loc[valid_mask & df['config_changed'], 'reconfig_energy_pJ'].mean()
            print(f"    Avg per change:   {avg_reconfig:.2f} pJ")
        else:
            print(f"    Avg per change:   N/A (no changes)")
        print(f"    Avg per interval: {total_reconfig_energy/n_intervals:.2f} pJ")
    else:
        print("    [WARN] No rows had complete previous-config data -- "
              "reconfiguration energy is 0 for all rows.")

    # Calculate net PPW and Net PPW Gain %
    print(f"\n{'='*70}")
    print("  NET PPW CALCULATION")
    print('='*70)

    if ppw_best_col is not None:
        interval_inst = 500000  # 500K instructions per interval
        if ips_prev_col is not None:
            interval_time = interval_inst / df[ips_prev_col].mean()
        else:
            interval_time = interval_inst / 1e9
            print("  [WARN] No ips_prev column found -- using a fixed 1e9 IPS "
                  "assumption for interval_time.")

        overhead_pJ = df['inference_energy_pJ'] + df['reconfig_energy_pJ']
        df['net_ppw'] = df[ppw_best_col] - overhead_pJ / interval_time

        raw_ppw_mean = df[ppw_best_col].mean()
        net_ppw_mean = df['net_ppw'].mean()
        overhead_pct = (raw_ppw_mean - net_ppw_mean) / raw_ppw_mean * 100 if raw_ppw_mean else np.nan

        print(f"\n  Raw PPW (best config, avg):  {raw_ppw_mean:.4e}")
        print(f"  Net PPW (after overhead, avg): {net_ppw_mean:.4e}")
        print(f"  PPW overhead (cost of ML control): {overhead_pct:.2f}%")

        # --- Net PPW Gain % --------------------------------------------------
        # How much better (or worse) off are you, after paying inference +
        # reconfig overhead, versus simply staying on the previous config?
        if ppw_prev_col is not None:
            baseline = df[ppw_prev_col]
            gain_source = f"'{ppw_prev_col}' column (actual previous-config PPW)"
        else:
            # Fallback: approximate the "stayed on previous config" baseline
            # using the prior row's own best-PPW, sorted by period. This is
            # an approximation (previous row's optimal PPW under ITS config,
            # not necessarily what that config would net during this row's
            # interval) -- flagged clearly below and in the output.
            baseline = df[ppw_best_col].shift(1)
            gain_source = ("previous row's PPW_best, shifted by one period "
                           "(approximation -- no explicit ppw_prev column found)")

        valid_gain = baseline.notna() & (baseline != 0) & df['net_ppw'].notna()
        df['net_ppw_gain_pct'] = np.where(
            valid_gain, (df['net_ppw'] - baseline) / baseline * 100, np.nan
        )

        gain_series = df.loc[valid_gain, 'net_ppw_gain_pct']
        print(f"\n  Net PPW Gain % baseline: {gain_source}")
        if len(gain_series) > 0:
            print(f"  Net PPW Gain %  (mean):    {gain_series.mean():+.2f}%")
            print(f"  Net PPW Gain %  (median):  {gain_series.median():+.2f}%")
            print(f"  Rows with net gain > 0:    {(gain_series > 0).sum():,} "
                  f"({(gain_series > 0).mean()*100:.1f}%)")
            print(f"  Rows with net loss < 0:    {(gain_series < 0).sum():,} "
                  f"({(gain_series < 0).mean()*100:.1f}%)")
        else:
            print("  [WARN] Could not compute Net PPW Gain % -- no valid baseline rows.")
    else:
        print("  [WARN] No PPW_best-style column found -- skipping net PPW / "
              "Net PPW Gain % calculation entirely.")
        df['net_ppw'] = np.nan
        df['net_ppw_gain_pct'] = np.nan

    # Save output
    print(f"\n{'='*70}")
    print("  SAVING OUTPUT")
    print('='*70)

    # Select columns for output — current config, split per-core prev config,
    # change flags, energy, and both PPW metrics.
    output_cols = ['benchmark', 'period_start', 'period_end']
    for key in CURRENT_DIMENSIONS:
        if curr_map.get(key):
            output_cols.append(curr_map[key])
    for key in PREV_DIMENSIONS:
        if prev_map.get(key):
            output_cols.append(prev_map[key])
    output_cols += [
        'config_changed', 'l2_changed', 'l3_changed', 'btb_changed', 'prefetcher_changed',
        'inference_energy_pJ', 'reconfig_energy_pJ',
    ]
    if ppw_best_col:
        output_cols.append(ppw_best_col)
    if ppw_prev_col:
        output_cols.append(ppw_prev_col)
    output_cols += ['net_ppw', 'net_ppw_gain_pct']

    # Filter to existing columns, de-duplicated, preserving order
    seen = set()
    output_cols = [c for c in output_cols if c in df.columns and not (c in seen or seen.add(c))]
    df[output_cols].to_csv(output_csv, index=False)

    print(f"  Output saved to: {output_csv}")
    print(f"  Rows: {len(df)}, Columns: {len(output_cols)}")

    return df, inf_breakdown, model_path


def generate_report(df, model_info, output_path):
    """Generate detailed text report."""
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("  COST ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")

        f.write(f"  Benchmark:     {df.get('benchmark', 'unknown').iloc[0] if 'benchmark' in df.columns else 'unknown'}\n")
        f.write(f"  Timestamp:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Model:         {model_info['path']}\n")
        f.write(f"  n_estimators:  {model_info['n_estimators']}\n")
        f.write(f"  max_depth:     {model_info['max_depth']}\n\n")

        f.write("="*70 + "\n")
        f.write("  McPAT/GAINESTOWN PARAMETERS USED\n")
        f.write("="*70 + "\n\n")

        f.write(f"  Technology Node:      {TECH_PARAMS['tech_node_nm']} nm\n")
        f.write(f"  Core Clock:           {TECH_PARAMS['clock_rate_mhz']} MHz\n")
        f.write(f"  Supply Voltage (Vdd): {TECH_PARAMS['vdd']} V\n")
        f.write(f"  Device Type:          {TECH_PARAMS['device_type']}\n\n")

        f.write("  Component Parameters:\n")
        f.write(f"    BTB entries:         {TECH_PARAMS['btb_entries_default']}\n")
        f.write(f"    L2 capacity:         {TECH_PARAMS['l2_capacity_kb']} KB\n")
        f.write(f"    L3 capacity:         {TECH_PARAMS['l3_capacity_kb']} KB\n\n")

        f.write("  Energy Parameters (45nm HP, 1.2V):\n")
        f.write(f"    E_gate_per_btb_entry:   {TECH_PARAMS['E_gate_per_btb_entry_pJ']:.3f} pJ\n")
        f.write(f"    E_gate_per_l2_kb:       {TECH_PARAMS['E_gate_per_l2_kb_pJ']:.3f} pJ\n")
        f.write(f"    E_gate_per_l3_kb:       {TECH_PARAMS['E_gate_per_l3_kb_pJ']:.3f} pJ\n")
        f.write(f"    E_gate_per_prefetcher:  {TECH_PARAMS['E_gate_per_prefetcher_pJ']:.3f} pJ (per core)\n\n")

        f.write("="*70 + "\n")
        f.write("  RESULTS SUMMARY\n")
        f.write("="*70 + "\n\n")

        if 'config_changed' in df.columns and 'has_prev_data' in df.columns:
            valid = df['has_prev_data']
            n_changed = int((df['config_changed'] & valid).sum())
            n_total = int(valid.sum())
            if n_total > 0:
                f.write(f"  Config changes:      {n_changed}/{n_total} ({n_changed/n_total*100:.1f}%)\n")

        if 'inference_energy_pJ' in df.columns:
            f.write(f"  Inference energy:    {df['inference_energy_pJ'].iloc[0]:.2f} pJ/interval\n")

        if 'reconfig_energy_pJ' in df.columns:
            total_reconfig = df['reconfig_energy_pJ'].sum()
            avg_reconfig = (df[df['config_changed']]['reconfig_energy_pJ'].mean()
                            if df['config_changed'].any() else 0)
            f.write(f"  Total reconfig:      {total_reconfig:.2f} pJ\n")
            f.write(f"  Avg per change:      {avg_reconfig:.2f} pJ\n")

        if 'net_ppw' in df.columns:
            ppw_best_col = resolve_first_present(df, PPW_BEST_CANDIDATES)
            raw_ppw = df[ppw_best_col].mean() if ppw_best_col else np.nan
            net_ppw = df['net_ppw'].mean()
            f.write(f"  Raw PPW (avg):       {raw_ppw:.4e}\n")
            f.write(f"  Net PPW (avg):       {net_ppw:.4e}\n")
            if raw_ppw:
                f.write(f"  PPW overhead:        {(raw_ppw-net_ppw)/raw_ppw*100:.2f}%\n")

        if 'net_ppw_gain_pct' in df.columns and df['net_ppw_gain_pct'].notna().any():
            gain = df['net_ppw_gain_pct'].dropna()
            f.write(f"\n  Net PPW Gain % (mean):    {gain.mean():+.2f}%\n")
            f.write(f"  Net PPW Gain % (median):  {gain.median():+.2f}%\n")
            f.write(f"  Rows with net gain > 0:   {(gain > 0).sum():,} ({(gain > 0).mean()*100:.1f}%)\n")
            f.write(f"  Rows with net loss < 0:   {(gain < 0).sum():,} ({(gain < 0).mean()*100:.1f}%)\n")


def main():
    parser = argparse.ArgumentParser(description='Cost analysis for RF-based post-silicon customization (7-way)')
    parser.add_argument('--benchmark', required=True, help='Benchmark name (barnes, cholesky, radiosity)')
    parser.add_argument('--model-dir', required=True, help='Directory containing saved RF model')
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--input-csv', default=None, help='Input CSV with config data')
    parser.add_argument('--report', default=None, help='Output report path')

    args = parser.parse_args()

    # Run analysis
    df, inf_breakdown, model_path = run_cost_analysis(
        args.benchmark, args.model_dir, args.output, args.input_csv
    )

    # Generate report if requested
    if args.report:
        model_info = {
            'path': model_path,
            'n_estimators': inf_breakdown['n_estimators'],
            'max_depth': inf_breakdown['max_depth'],
        }
        generate_report(df, model_info, args.report)
        print(f"  Report saved to: {args.report}")


if __name__ == "__main__":
    main()
