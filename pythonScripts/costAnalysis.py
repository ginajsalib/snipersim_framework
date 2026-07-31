"""
costAnalysis.py  (7-way)
===============
Calculate RF model inference cost and reconfiguration cost for post-silicon
customization using clock gating (McPAT/Gainestown parameters), for the
7-way model (L2 core0/1, L3, BTB core0/1, Prefetcher core0/1).

IMPORTANT -- this version requires the REAL predict_config.py output as
--input-csv, plus the matching --sweep train_with_top3_<bench>.csv. Passing
train_with_top3_<bench>.csv itself as --input-csv does NOT work: this
script never runs predict_config.py for you, and train_with_top3's own
"*_best" columns (the oracle label) collide, under naive normalization,
with the model's real predicted-config columns -- silently treating the
oracle label as if it were the model's decision. See resolve_reconfig_columns.

Net PPW Gain % and the "current"/achieved PPW are now computed by joining
each interval's model-predicted config (and, for the gain baseline, the
config that was actually running going into that interval) against the
sweep file via period_start_val_prev -- NOT by reading predictions.csv's
own 'PPW_best' column directly, because that column is an oracle label
carried through from training, not a measurement of what the predicted
config achieved (confirmed empirically: it is identical to the sweep's
oracle value to 11+ significant digits). See build_achieved_lookup.

Reconfiguration cost is computed per-row against that row's own actual
previous-interval config (the split per-core "_prev" columns in
predictions.csv) vs the model's real predicted config for that row (the
lowercase, underscore-separated columns -- resolved via an exact,
case-sensitive match first, so it can never collide with the oracle-label
block that normalizes to the same key).

Based on: Weston, K., et al. (2023). Post-Silicon Customization Using Deep
Neural Networks. ARCS.

Usage:
    python costAnalysis.py --input-csv predict_barnes.csv --sweep train_with_top3_barnes.csv \\
        --model-dir saved_models/ --benchmark barnes --output cost_analysis_barnes.csv \\
        --report cost_report_barnes.txt
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


def find_col(df, expected_name, exclude_substr=None):
    """
    Find a column matching expected_name, ignoring case/separators.
    If exclude_substr is given, any column whose normalized name contains it
    is skipped entirely.
    """
    norm = _normalize_name
    excl = norm(exclude_substr) if exclude_substr else None

    def ok(col):
        return excl is None or excl not in norm(col)

    for col in df.columns:
        if norm(col) == norm(expected_name) and ok(col):
            return col
    for col in df.columns:
        if norm(expected_name) in norm(col) and ok(col):
            return col
    return None


def resolve_first_present(df, candidates, exclude_substr=None):
    for c in candidates:
        found = find_col(df, c, exclude_substr=exclude_substr)
        if found:
            return found
    return None


# The model's REAL predicted-config columns in predict_config.py's output
# are the lowercase, underscore-separated ones with NO suffix (l2_core0,
# btb_core0, prefetch_core0, ...). That same file also carries a SEPARATE
# unprefixed camelCase block (L2core0, btbCore0, prefetcher) which is the
# ORACLE label, not the prediction -- and it normalizes to the exact same
# key as the real predicted column under naive matching. So resolution here
# tries an EXACT, case-sensitive name first; fuzzy/normalized matching is
# only a last-resort fallback, with a loud warning, since it risks silently
# picking the oracle-label block instead.
CURRENT_DIMENSIONS_EXACT = {
    'l2_core0':       'l2_core0',
    'l2_core1':       'l2_core1',
    'l3':             'l3',
    'btb_core0':      'btb_core0',
    'btb_core1':      'btb_core1',
    'prefetch_core0': 'prefetch_core0',
    'prefetch_core1': 'prefetch_core1',
}
# Legacy/fuzzy fallback candidates, only used if the exact name isn't present
# (e.g. a raw training-style CSV with ranked "_best" columns instead).
CURRENT_DIMENSIONS_FUZZY = {
    'l2_core0':       ['L2core0_best', 'L2 core 0_best', 'l2_core0_best', 'l2Core0_best'],
    'l2_core1':       ['L2core1_best', 'L2 core 1_best', 'l2_core1_best', 'l2Core1_best'],
    'l3':             ['L3_best', 'l3_best', 'l3Size_best'],
    'btb_core0':      ['BTBcore0_best', 'BTB core 0_best', 'btb_core0_best', 'btbCore0_best'],
    'btb_core1':      ['BTBcore1_best', 'BTB core 1_best', 'btb_core1_best', 'btbCore1_best'],
    'prefetch_core0': ['Prefetchcore0_best', 'Prefetch core 0_best', 'prefetch_core0_best', 'prefetcher_core0_best'],
    'prefetch_core1': ['Prefetchcore1_best', 'Prefetch core 1_best', 'prefetch_core1_best', 'prefetcher_core1_best'],
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

PERIOD_CANDIDATES = ['period_start']
PPW_BEST_CANDIDATES = ['PPW_best', 'PPW__best', 'ppw_best']  # oracle label -- NOT used as achieved anymore
IPS_PREV_CANDIDATES = ['ips_prev']

# Sweep-file (train_with_top3) column candidates
SWEEP_PERIOD_CANDIDATES = ['period_start']
SWEEP_PREV_PERIOD_CANDIDATES = ['period_start_val_prev']
SWEEP_ACHIEVED_PPW_CANDIDATES = ['ppw_prev']
SWEEP_CONFIG_DIMENSIONS = {
    'l2_core0':       ['l2Core0_prev', 'L2core0_prev', 'L2 core 0_prev'],
    'l2_core1':       ['l2Core1_prev', 'L2core1_prev', 'L2 core 1_prev'],
    'l3':             ['l3Size_prev', 'L3_prev'],
    'btb_core0':      ['btbCore0_prev', 'BTBcore0_prev', 'BTB core 0_prev'],
    'btb_core1':      ['btbCore1_prev', 'BTBcore1_prev', 'BTB core 1_prev'],
    'prefetch_core0': ['prefetchCore0_prev', 'Prefetchcore0_prev', 'Prefetch core 0_prev'],
    'prefetch_core1': ['prefetchCore1_prev', 'Prefetchcore1_prev', 'Prefetch core 1_prev'],
}


def parse_period_numeric(period_val):
    """Extract the numeric instruction-count id from a period label.
    Handles 'periodicins-100000002' -> 100000002, bare numbers -> as-is,
    and non-numeric sentinels ('roi-end', 'roi-begin', etc.) -> None."""
    if period_val is None:
        return None
    if isinstance(period_val, (int, float)) and not isinstance(period_val, bool):
        return None if (isinstance(period_val, float) and np.isnan(period_val)) else float(period_val)
    s = str(period_val).strip()
    if s.lower() in ('nan', ''):
        return None
    if '-' in s:
        tail = s.rsplit('-', 1)[-1]
        if tail.replace('.', '', 1).isdigit():
            return float(tail)
        return None
    if s.replace('.', '', 1).isdigit():
        return float(s)
    return None


def normalize_component(val):
    """Normalize one config-dimension value for equality comparison across
    files that may format the same value as '512', '512.0', 512.0, 'none', etc."""
    try:
        f = float(val)
        if np.isnan(f):
            return None
        return round(f, 6)
    except (TypeError, ValueError):
        s = str(val).strip().lower()
        return None if s in ('nan', '') else s


def resolve_reconfig_columns(df):
    """
    Resolve both the current-chosen-config columns (the model's REAL
    predicted config, exact-name-first -- see CURRENT_DIMENSIONS_EXACT) and
    the split per-core "_prev" columns (the config actually running before
    this interval). Returns (curr_col_map, prev_col_map).
    """
    curr_map = {}
    used_fuzzy = []
    for key, exact_name in CURRENT_DIMENSIONS_EXACT.items():
        if exact_name in df.columns:
            curr_map[key] = exact_name
        else:
            col = resolve_first_present(df, CURRENT_DIMENSIONS_FUZZY[key], exclude_substr='prev')
            curr_map[key] = col
            if col is not None:
                used_fuzzy.append((key, col))
    if used_fuzzy:
        print(f"  [WARN] Current-config columns resolved via fuzzy fallback (verify these are the model's "
              f"real predictions, not an oracle label): {used_fuzzy}")

    prev_map = {key: resolve_first_present(df, cands)
                for key, cands in PREV_DIMENSIONS.items()}
    return curr_map, prev_map


def build_config_tuple(row, col_map):
    """Build a normalized 7-tuple config identity from a row + column map.
    Returns None if any dimension is missing/unresolvable for this row."""
    vals = []
    for key in SWEEP_CONFIG_DIMENSIONS:  # canonical dimension order
        col = col_map.get(key)
        if col is None:
            return None
        v = normalize_component(row[col])
        if v is None:
            return None
        vals.append(v)
    return tuple(vals)


def build_achieved_lookup(sweep_csv, needed_pairs, chunksize=200_000):
    """
    Stream the sweep file once, returning {(period_num, cfg_tuple): ppw_prev}
    for every (period, config) pair present in needed_pairs (a set) with a
    finite ppw_prev value. Join key: period_start_val_prev (numeric) ==
    period_num -- the field marking the start of the interval this row's
    '*_prev' columns (ppw_prev, config) actually describe.
    """
    header = pd.read_csv(sweep_csv, nrows=0)
    columns = list(header.columns)

    prev_period_col = resolve_first_present(header, SWEEP_PREV_PERIOD_CANDIDATES)
    achieved_col = resolve_first_present(header, SWEEP_ACHIEVED_PPW_CANDIDATES)
    dim_cols = {key: resolve_first_present(header, cands) for key, cands in SWEEP_CONFIG_DIMENSIONS.items()}
    missing = [k for k, v in dim_cols.items() if v is None]
    if prev_period_col is None or achieved_col is None or missing:
        raise ValueError(f"Missing required sweep column(s) in {sweep_csv}: "
                          f"period_start_val_prev={prev_period_col}, ppw_prev={achieved_col}, "
                          f"missing config dims={missing}")

    usecols = list(dict.fromkeys([prev_period_col, achieved_col] + list(dim_cols.values())))
    periods_needed = {p for p, _ in needed_pairs}

    lookup = {}
    for chunk in pd.read_csv(sweep_csv, usecols=usecols, chunksize=chunksize):
        period_nums = pd.to_numeric(chunk[prev_period_col], errors='coerce')
        in_needed = period_nums.isin(periods_needed)
        if not in_needed.any():
            continue
        sub = chunk.loc[in_needed]
        sub_periods = period_nums.loc[in_needed]

        config_key = pd.Series(
            list(zip(*[sub[dim_cols[k]].map(normalize_component) for k in SWEEP_CONFIG_DIMENSIONS])),
            index=sub.index,
        )
        vals = pd.to_numeric(sub[achieved_col], errors='coerce')
        finite = np.isfinite(vals)

        for period, cfg, val in zip(sub_periods[finite], config_key[finite], vals[finite]):
            key = (period, cfg)
            if key in needed_pairs and key not in lookup:
                lookup[key] = val

    return lookup


# ==============================================================================
# GAINESTOWN/McPAT TECHNOLOGY PARAMETERS (from power.xml)
# ==============================================================================

TECH_PARAMS = {
    'tech_node_nm': 45,
    'clock_rate_mhz': 2660,
    'vdd': 1.2,
    'device_type': 'HP',
    'power_gating_enabled': True,
    'btb_entries_default': 18944,
    'l2_capacity_kb': 256,
    'l3_capacity_kb': 8192,
    'l2_assoc': 8,
    'l3_assoc': 16,
    'cache_line_bytes': 64,
    'E_gate_per_btb_entry_pJ': 0.05,
    'E_gate_per_l2_kb_pJ': 2.5,
    'E_gate_per_l3_kb_pJ': 1.8,
    'E_gate_per_prefetcher_pJ': 15.0,
    'E_rf_inst_pJ': 0.5,
    'E_rf_cmp_pJ': 0.1,
    'E_rf_mem_pJ': 1.0,
}


def print_tech_params():
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
    print(f"    L2 capacity:            {TECH_PARAMS['l2_capacity_kb']} KB")
    print(f"    L3 capacity:            {TECH_PARAMS['l3_capacity_kb']} KB")
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

def calc_reconfig_energy_vectorized(df, curr_map, prev_map, tech_params):
    n = len(df)
    energy = np.zeros(n)
    l2_changed = np.zeros(n, dtype=bool)
    l3_changed = np.zeros(n, dtype=bool)
    btb_changed = np.zeros(n, dtype=bool)
    pf_changed = np.zeros(n, dtype=bool)
    has_prev = np.ones(n, dtype=bool)

    def numeric(col):
        return pd.to_numeric(df[col], errors='coerce').values.astype(float) if col else None

    def categorical(col):
        return df[col].astype(str).str.strip().str.lower().values if col else None

    for core in ['l2_core0', 'l2_core1']:
        c_col, p_col = curr_map.get(core), prev_map.get(core)
        if c_col is None or p_col is None:
            has_prev &= False if p_col is None else has_prev
            continue
        curr_v, prev_v = numeric(c_col), numeric(p_col)
        valid = ~np.isnan(curr_v) & ~np.isnan(prev_v)
        diff = np.where(valid, np.abs(curr_v - prev_v), 0.0)
        changed_here = valid & (diff > 0)
        energy += np.where(changed_here, tech_params['E_gate_per_l2_kb_pJ'] * diff, 0.0)
        l2_changed |= changed_here

    c_col, p_col = curr_map.get('l3'), prev_map.get('l3')
    if c_col is not None and p_col is not None:
        curr_v, prev_v = numeric(c_col), numeric(p_col)
        valid = ~np.isnan(curr_v) & ~np.isnan(prev_v)
        diff = np.where(valid, np.abs(curr_v - prev_v), 0.0)
        changed_here = valid & (diff > 0)
        energy += np.where(changed_here, tech_params['E_gate_per_l3_kb_pJ'] * diff, 0.0)
        l3_changed |= changed_here
    else:
        has_prev &= False if p_col is None else has_prev

    for core in ['btb_core0', 'btb_core1']:
        c_col, p_col = curr_map.get(core), prev_map.get(core)
        if c_col is None or p_col is None:
            has_prev &= False if p_col is None else has_prev
            continue
        curr_v, prev_v = numeric(c_col), numeric(p_col)
        valid = ~np.isnan(curr_v) & ~np.isnan(prev_v)
        diff = np.where(valid, np.abs(curr_v - prev_v), 0.0)
        changed_here = valid & (diff > 0)
        energy += np.where(changed_here, tech_params['E_gate_per_btb_entry_pJ'] * diff, 0.0)
        btb_changed |= changed_here

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
        'config_changed': config_changed,
        'l2_changed': l2_changed,
        'l3_changed': l3_changed,
        'btb_changed': btb_changed,
        'prefetcher_changed': pf_changed,
        'has_prev_data': has_prev,
    }


def calc_inference_energy(model, tech_params):
    n_estimators = model.n_estimators
    max_depth = model.max_depth if model.max_depth else 25
    if hasattr(model, 'n_features_in_'):
        n_features = model.n_features_in_
    elif hasattr(model, 'estimators_') and len(model.estimators_) > 0:
        n_features = len(model.estimators_[0].feature_importances_)
    else:
        n_features = 130

    n_comparisons = n_estimators * max_depth
    n_mem_accesses = n_estimators * n_features
    n_instructions = n_comparisons * 3

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


def run_cost_analysis(benchmark, model_dir, output_csv, input_csv, sweep_csv):
    print(f"\n{'='*70}")
    print(f"  COST ANALYSIS: {benchmark.upper()}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*70)

    print_tech_params()

    print(f"\n{'='*70}")
    print("  LOADING MODEL")
    print('='*70)
    model, ts, model_path = load_model(model_dir)
    print(f"  Model timestamp: {ts}")
    print(f"  Model path: {model_path}")
    print(f"  n_estimators: {model.n_estimators}")
    print(f"  max_depth: {model.max_depth}")

    print(f"\n{'='*70}")
    print("  LOADING CONFIGURATION DATA")
    print('='*70)
    df = pd.read_csv(input_csv)
    print(f"  Loaded {len(df)} rows from {input_csv}")

    if 'period_start' in df.columns:
        df = df.sort_values('period_start').reset_index(drop=True)
    else:
        df = df.sort_index().reset_index(drop=True)

    curr_map, prev_map = resolve_reconfig_columns(df)
    print(f"\n  Resolved current (predicted) config columns: {curr_map}")
    print(f"  Resolved previous (entering) config columns:  {prev_map}")
    missing_prev = [k for k, v in prev_map.items() if v is None]
    if missing_prev:
        print(f"  [WARN] No previous-config column found for: {missing_prev} "
              f"-- reconfig cost for those dimensions will be treated as 0/unchanged.")

    period_col = resolve_first_present(df, PERIOD_CANDIDATES)
    if period_col is None:
        raise ValueError(f"No period_start column found in {input_csv}")

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
    print(f"    Total inference:    {inf_energy:.2f} pJ ({inf_energy/1e6:.4f} uJ)")

    # Calculate reconfiguration energy
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

    # ---------------------------------------------------------------
    # Achieved / baseline PPW via the sweep join (replaces reading
    # predictions.csv's own PPW_best directly, since that column is an
    # oracle label, not a measurement of what was actually achieved)
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  ACHIEVED / BASELINE PPW (via sweep join)")
    print('='*70)

    period_nums = df[period_col].map(parse_period_numeric)
    n_unparseable = int(period_nums.isna().sum())
    if n_unparseable:
        print(f"  [WARN] {n_unparseable} row(s) had a non-numeric period_start and are excluded "
              f"from the achieved/baseline PPW join.")

    predicted_cfg = df.apply(lambda r: build_config_tuple(r, curr_map), axis=1)
    entering_cfg = df.apply(lambda r: build_config_tuple(r, prev_map), axis=1)

    needed_pairs = set()
    for p, c in zip(period_nums, predicted_cfg):
        if p is not None and c is not None:
            needed_pairs.add((p, c))
    for p, c in zip(period_nums, entering_cfg):
        if p is not None and c is not None:
            needed_pairs.add((p, c))

    lookup = build_achieved_lookup(sweep_csv, needed_pairs)

    achieved_ppw = [lookup.get((p, c)) if (p is not None and c is not None) else None
                    for p, c in zip(period_nums, predicted_cfg)]
    baseline_ppw = [lookup.get((p, c)) if (p is not None and c is not None) else None
                    for p, c in zip(period_nums, entering_cfg)]
    df['achieved_ppw'] = pd.array(achieved_ppw, dtype='float64')
    df['baseline_ppw'] = pd.array(baseline_ppw, dtype='float64')

    n_achieved_found = int(df['achieved_ppw'].notna().sum())
    n_baseline_found = int(df['baseline_ppw'].notna().sum())
    print(f"  Achieved PPW found for: {n_achieved_found}/{len(df)} rows "
          f"({n_achieved_found/len(df)*100:.1f}%)")
    print(f"  Baseline (entering-config) PPW found for: {n_baseline_found}/{len(df)} rows "
          f"({n_baseline_found/len(df)*100:.1f}%)")

    ips_prev_col = resolve_first_present(df, IPS_PREV_CANDIDATES)

    print(f"\n{'='*70}")
    print("  NET PPW CALCULATION")
    print('='*70)

    if ips_prev_col is not None:
        ips_clean = df[ips_prev_col].replace([np.inf, -np.inf], np.nan)
        ips_mean = ips_clean.mean()
        interval_inst = 500000
        if not ips_mean or np.isnan(ips_mean):
            print("  [WARN] ips_prev average is 0/NaN -- falling back to a fixed 1e9 IPS assumption.")
            interval_time = 500000 / 1e9
        else:
            interval_time = interval_inst / ips_mean
    else:
        interval_time = 500000 / 1e9
        print("  [WARN] No ips_prev column found -- using a fixed 1e9 IPS assumption for interval_time.")

    overhead_pJ = df['inference_energy_pJ'] + df['reconfig_energy_pJ']
    df['net_ppw'] = df['achieved_ppw'] - overhead_pJ / interval_time
    df['net_ppw'] = df['net_ppw'].replace([np.inf, -np.inf], np.nan)

    raw_ppw_mean = df['achieved_ppw'].mean()
    net_ppw_mean = df['net_ppw'].mean()
    overhead_pct = ((raw_ppw_mean - net_ppw_mean) / raw_ppw_mean * 100
                    if raw_ppw_mean and not np.isnan(raw_ppw_mean) else np.nan)

    print(f"\n  Raw achieved PPW (avg):        {raw_ppw_mean:.4e}")
    print(f"  Net PPW (after overhead, avg): {net_ppw_mean:.4e}")
    print(f"  PPW overhead (cost of ML control): {overhead_pct:.2f}%")

    baseline = df['baseline_ppw'].replace([np.inf, -np.inf], np.nan)
    valid_gain = baseline.notna() & (baseline != 0) & df['net_ppw'].notna()
    df['net_ppw_gain_pct'] = np.where(
        valid_gain, (df['net_ppw'] - baseline) / baseline * 100, np.nan
    )

    gain_series = df.loc[valid_gain, 'net_ppw_gain_pct']
    print(f"\n  Net PPW Gain % baseline: achieved PPW of the ENTERING config (what you'd have kept "
          f"getting had you not switched), via sweep join")
    if len(gain_series) > 0:
        print(f"  Net PPW Gain %  (mean):    {gain_series.mean():+.2f}%")
        print(f"  Net PPW Gain %  (median):  {gain_series.median():+.2f}%")
        print(f"  Rows with net gain > 0:    {(gain_series > 0).sum():,} ({(gain_series > 0).mean()*100:.1f}%)")
        print(f"  Rows with net loss < 0:    {(gain_series < 0).sum():,} ({(gain_series < 0).mean()*100:.1f}%)")
    else:
        print("  [WARN] Could not compute Net PPW Gain % -- no valid baseline rows.")

    # Save output
    print(f"\n{'='*70}")
    print("  SAVING OUTPUT")
    print('='*70)

    output_cols = ['benchmark', 'period_start'] if 'benchmark' in df.columns else ['period_start']
    if 'period_end' in df.columns:
        output_cols.append('period_end')
    for key in CURRENT_DIMENSIONS_EXACT:
        if curr_map.get(key):
            output_cols.append(curr_map[key])
    for key in PREV_DIMENSIONS:
        if prev_map.get(key):
            output_cols.append(prev_map[key])
    output_cols += [
        'config_changed', 'l2_changed', 'l3_changed', 'btb_changed', 'prefetcher_changed',
        'inference_energy_pJ', 'reconfig_energy_pJ',
        'achieved_ppw', 'baseline_ppw', 'net_ppw', 'net_ppw_gain_pct',
    ]

    seen = set()
    output_cols = [c for c in output_cols if c in df.columns and not (c in seen or seen.add(c))]
    df[output_cols].to_csv(output_csv, index=False)

    print(f"  Output saved to: {output_csv}")
    print(f"  Rows: {len(df)}, Columns: {len(output_cols)}")

    return df, inf_breakdown, model_path


def generate_report(df, model_info, output_path):
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
            raw_ppw = df['achieved_ppw'].mean() if 'achieved_ppw' in df.columns else np.nan
            net_ppw = df['net_ppw'].mean()
            f.write(f"  Raw achieved PPW (avg): {raw_ppw:.4e}\n")
            f.write(f"  Net PPW (avg):          {net_ppw:.4e}\n")
            if raw_ppw:
                f.write(f"  PPW overhead:           {(raw_ppw-net_ppw)/raw_ppw*100:.2f}%\n")

        if 'net_ppw_gain_pct' in df.columns and df['net_ppw_gain_pct'].notna().any():
            gain = df['net_ppw_gain_pct'].dropna()
            f.write(f"\n  Net PPW Gain % (mean):    {gain.mean():+.2f}%\n")
            f.write(f"  Net PPW Gain % (median):  {gain.median():+.2f}%\n")
            f.write(f"  Rows with net gain > 0:   {(gain > 0).sum():,} ({(gain > 0).mean()*100:.1f}%)\n")
            f.write(f"  Rows with net loss < 0:   {(gain < 0).sum():,} ({(gain < 0).mean()*100:.1f}%)\n")


def main():
    parser = argparse.ArgumentParser(description='Cost analysis for RF-based post-silicon customization (7-way)')
    parser.add_argument('--benchmark', required=True, help='Benchmark name (barnes, cholesky, radiosity, fft)')
    parser.add_argument('--model-dir', required=True, help='Directory containing saved RF model')
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--input-csv', required=True,
                         help='Path to the REAL predict_config.py output (NOT train_with_top3_<bench>.csv)')
    parser.add_argument('--sweep', required=True,
                         help='Path to train_with_top3_<bench>.csv, used to look up achieved/baseline PPW')
    parser.add_argument('--report', default=None, help='Output report path')

    args = parser.parse_args()

    df, inf_breakdown, model_path = run_cost_analysis(
        args.benchmark, args.model_dir, args.output, args.input_csv, args.sweep
    )

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
