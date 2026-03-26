"""
costAnalysis.py
===============
Calculate RF model inference cost and reconfiguration cost for post-silicon
customization using clock gating (McPAT/Gainestown parameters).

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
    'E_gate_per_prefetcher_pJ': 15.0,    # Prefetcher state machine gating

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
    print(f"    E_gate_per_prefetcher:  {TECH_PARAMS['E_gate_per_prefetcher_pJ']:.3f} pJ")
    print(f"    E_rf_inst (inference):  {TECH_PARAMS['E_rf_inst_pJ']:.3f} pJ")
    print(f"    E_rf_cmp (tree node):   {TECH_PARAMS['E_rf_cmp_pJ']:.3f} pJ")
    print(f"    E_rf_mem (SRAM access): {TECH_PARAMS['E_rf_mem_pJ']:.3f} pJ")


def calc_reconfig_energy(row, prev_row, tech_params):
    """
    Calculate reconfiguration energy for one interval.

    Args:
        row: Current row with config columns
        prev_row: Previous row with config columns
        tech_params: Technology parameters dict

    Returns:
        (total_energy_pJ, breakdown_dict, changed_components)
    """
    energy = 0.0
    breakdown = {}
    changed = []

    # BTB core0 change
    if pd.notna(row.get('btbCore0')) and pd.notna(prev_row.get('btbCore0')):
        if row['btbCore0'] != prev_row['btbCore0']:
            # Energy proportional to BTB size difference
            btb_diff = abs(int(row['btbCore0']) - int(prev_row['btbCore0']))
            e_btb0 = tech_params['E_gate_per_btb_entry_pJ'] * btb_diff
            energy += e_btb0
            breakdown['btbCore0'] = e_btb0
            changed.append('btbCore0')

    # BTB core1 change
    if pd.notna(row.get('btbCore1')) and pd.notna(prev_row.get('btbCore1')):
        if row['btbCore1'] != prev_row['btbCore1']:
            btb_diff = abs(int(row['btbCore1']) - int(prev_row['btbCore1']))
            e_btb1 = tech_params['E_gate_per_btb_entry_pJ'] * btb_diff
            energy += e_btb1
            breakdown['btbCore1'] = e_btb1
            changed.append('btbCore1')

    # L2 change
    if pd.notna(row.get('L2')) and pd.notna(prev_row.get('L2')):
        if row['L2'] != prev_row['L2']:
            l2_diff = abs(int(row['L2']) - int(prev_row['L2']))
            e_l2 = tech_params['E_gate_per_l2_kb_pJ'] * l2_diff
            energy += e_l2
            breakdown['L2'] = e_l2
            changed.append('L2')

    # L3 change
    if pd.notna(row.get('L3')) and pd.notna(prev_row.get('L3')):
        if row['L3'] != prev_row['L3']:
            l3_diff = abs(int(row['L3']) - int(prev_row['L3']))
            e_l3 = tech_params['E_gate_per_l3_kb_pJ'] * l3_diff
            energy += e_l3
            breakdown['L3'] = e_l3
            changed.append('L3')

    # Prefetcher change (discrete: none→simple→ghb)
    if pd.notna(row.get('prefetcher')) and pd.notna(prev_row.get('prefetcher')):
        if row['prefetcher'] != prev_row['prefetcher']:
            e_pref = tech_params['E_gate_per_prefetcher_pJ']
            energy += e_pref
            breakdown['prefetcher'] = e_pref
            changed.append('prefetcher')

    return energy, breakdown, changed


def calc_inference_energy(model, tech_params):
    """
    Calculate RF model inference energy per prediction.

    Args:
        model: Trained RandomForest model
        tech_params: Technology parameters dict

    Returns:
        inference_energy_pJ
    """
    n_estimators = model.n_estimators
    max_depth = model.max_depth if model.max_depth else 15
    n_features = len(model.estimators_[0].feature_importances_)

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
    """Load the most recent RF model from directory."""
    pkls = glob.glob(os.path.join(model_dir, 'rf_4way_config_predictor_*.pkl'))
    pkls = [p for p in pkls if '_scaler' not in p and '_encoder' not in p]

    if not pkls:
        raise FileNotFoundError(f"No model files found in {model_dir}")

    pkls.sort()
    latest = pkls[-1]
    model = joblib.load(latest)

    ts = os.path.basename(latest).replace('rf_4way_config_predictor_', '').replace('.pkl', '')
    return model, ts, latest


def load_config_data(benchmark, input_csv=None):
    """Load configuration data for a benchmark."""
    if input_csv and os.path.exists(input_csv):
        df = pd.read_csv(input_csv)
        print(f"  Loaded {len(df)} rows from {input_csv}")
        return df

    # Default paths for training data
    default_paths = {
        'barnes': '/home/gina/Desktop/snipersim_framework/pythonScripts/finalTrainingData/barnes_train_with_top3_fixed.csv',
        'cholesky': '/home/gina/Desktop/snipersim_framework/pythonScripts/finalTrainingData/cholesky_train_with_top3_fixed.csv',
        'radiosity': '/home/gina/Desktop/snipersim_framework/pythonScripts/finalTrainingData/radiosity_train_with_top3_fixed.csv',
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

    # Calculate reconfiguration energy per row
    print(f"\n{'='*70}")
    print("  RECONFIGURATION ENERGY CALCULATION")
    print('='*70)

    # Initialize columns
    df['inference_energy_pJ'] = inf_energy
    df['reconfig_energy_pJ'] = 0.0
    df['config_changed'] = False
    df['btb_changed'] = False
    df['l2_changed'] = False
    df['l3_changed'] = False
    df['prefetcher_changed'] = False
    df['reconfig_breakdown'] = ''

    # Process each row (compare with previous)
    changed_counts = {'btbCore0': 0, 'btbCore1': 0, 'L2': 0, 'L3': 0, 'prefetcher': 0}
    total_reconfig_energy = 0.0

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]

        reconfig_e, breakdown, changed = calc_reconfig_energy(row, prev_row, TECH_PARAMS)

        df.at[i, 'reconfig_energy_pJ'] = reconfig_e
        df.at[i, 'config_changed'] = len(changed) > 0
        df.at[i, 'reconfig_breakdown'] = str(breakdown)

        for comp in changed:
            if comp in ['btbCore0', 'btbCore1']:
                df.at[i, 'btb_changed'] = True
                changed_counts['btbCore0'] += 1 if comp == 'btbCore0' else 0
                changed_counts['btbCore1'] += 1 if comp == 'btbCore1' else 0
            elif comp == 'L2':
                df.at[i, 'l2_changed'] = True
                changed_counts['L2'] += 1
            elif comp == 'L3':
                df.at[i, 'l3_changed'] = True
                changed_counts['L3'] += 1
            elif comp == 'prefetcher':
                df.at[i, 'prefetcher_changed'] = True
                changed_counts['prefetcher'] += 1

        total_reconfig_energy += reconfig_e

    # Calculate net PPW
    if 'PPW_best' in df.columns:
        # Convert PPW to energy-aware metric
        interval_inst = 500000  # 500K instructions per interval
        interval_time = interval_inst / (df['ips'].mean() if 'ips' in df.columns else 1e9)

        # Net PPW = raw PPW - overhead
        df['net_ppw'] = df['PPW_best'] - (df['inference_energy_pJ'] + df['reconfig_energy_pJ']) / interval_time

    # Print reconfiguration summary
    n_intervals = len(df) - 1  # Exclude first row
    n_changed = df['config_changed'].sum()

    print(f"\n  Reconfiguration Summary:")
    print(f"    Total intervals:     {n_intervals}")
    print(f"    Intervals changed:   {n_changed} ({n_changed/n_intervals*100:.1f}%)")
    print(f"    Intervals unchanged: {n_intervals - n_changed} ({(n_intervals-n_changed)/n_intervals*100:.1f}%)")

    print(f"\n  Component Change Frequency:")
    print(f"    BTB core0:    {changed_counts['btbCore0']:5d} ({changed_counts['btbCore0']/n_intervals*100:.1f}%)")
    print(f"    BTB core1:    {changed_counts['btbCore1']:5d} ({changed_counts['btbCore1']/n_intervals*100:.1f}%)")
    print(f"    L2 cache:     {changed_counts['L2']:5d} ({changed_counts['L2']/n_intervals*100:.1f}%)")
    print(f"    L3 cache:     {changed_counts['L3']:5d} ({changed_counts['L3']/n_intervals*100:.1f}%)")
    print(f"    Prefetcher:   {changed_counts['prefetcher']:5d} ({changed_counts['prefetcher']/n_intervals*100:.1f}%)")

    print(f"\n  Reconfiguration Energy:")
    print(f"    Total:          {total_reconfig_energy:.2f} pJ")
    print(f"    Avg per change: {total_reconfig_energy/n_changed:.2f} pJ" if n_changed > 0 else "    N/A (no changes)")
    print(f"    Avg per interval: {total_reconfig_energy/n_intervals:.2f} pJ")

    # Save output
    print(f"\n{'='*70}")
    print("  SAVING OUTPUT")
    print('='*70)

    # Select columns for output
    output_cols = [
        'benchmark', 'period_start', 'period_end',
        'btbCore0', 'btbCore1', 'prefetcher', 'L2', 'L3',
        'btbCore0_prev' if 'btbCore0_prev' in df.columns else 'btbCore0',
        'btbCore1_prev' if 'btbCore1_prev' in df.columns else 'btbCore1',
        'config_changed', 'btb_changed', 'l2_changed', 'l3_changed', 'prefetcher_changed',
        'inference_energy_pJ', 'reconfig_energy_pJ', 'reconfig_breakdown',
        'PPW_best' if 'PPW_best' in df.columns else 'ppw',
        'net_ppw' if 'net_ppw' in df.columns else 'ppw',
    ]

    # Filter to existing columns
    output_cols = [c for c in output_cols if c in df.columns]
    df[output_cols].to_csv(output_csv, index=False)

    print(f"  Output saved to: {output_csv}")
    print(f"  Rows: {len(df)}, Columns: {len(output_cols)}")

    return df


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
        f.write(f"    E_gate_per_prefetcher:  {TECH_PARAMS['E_gate_per_prefetcher_pJ']:.3f} pJ\n\n")

        f.write("="*70 + "\n")
        f.write("  RESULTS SUMMARY\n")
        f.write("="*70 + "\n\n")

        if 'config_changed' in df.columns:
            n_changed = df['config_changed'].sum()
            n_total = len(df) - 1
            f.write(f"  Config changes:      {n_changed}/{n_total} ({n_changed/n_total*100:.1f}%)\n")

        if 'inference_energy_pJ' in df.columns:
            f.write(f"  Inference energy:    {df['inference_energy_pJ'].iloc[0]:.2f} pJ/interval\n")

        if 'reconfig_energy_pJ' in df.columns:
            total_reconfig = df['reconfig_energy_pJ'].sum()
            avg_reconfig = df[df['config_changed']]['reconfig_energy_pJ'].mean() if df['config_changed'].any() else 0
            f.write(f"  Total reconfig:      {total_reconfig:.2f} pJ\n")
            f.write(f"  Avg per change:      {avg_reconfig:.2f} pJ\n")

        if 'net_ppw' in df.columns:
            raw_ppw = df['PPW_best'].mean() if 'PPW_best' in df.columns else df.get('ppw', 0).mean()
            net_ppw = df['net_ppw'].mean()
            f.write(f"  Raw PPW (avg):       {raw_ppw:.4e}\n")
            f.write(f"  Net PPW (avg):       {net_ppw:.4e}\n")
            f.write(f"  PPW overhead:        {(raw_ppw-net_ppw)/raw_ppw*100:.2f}%\n")


def main():
    parser = argparse.ArgumentParser(description='Cost analysis for RF-based post-silicon customization')
    parser.add_argument('--benchmark', required=True, help='Benchmark name (barnes, cholesky, radiosity)')
    parser.add_argument('--model-dir', required=True, help='Directory containing saved RF model')
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--input-csv', default=None, help='Input CSV with config data')
    parser.add_argument('--report', default=None, help='Output report path')

    args = parser.parse_args()

    # Run analysis
    df = run_cost_analysis(args.benchmark, args.model_dir, args.output, args.input_csv)

    # Generate report if requested
    if args.report:
        model_info = {
            'path': args.model_dir,
            'n_estimators': TECH_PARAMS.get('n_estimators', 200),
            'max_depth': TECH_PARAMS.get('max_depth', 15),
        }
        generate_report(df, model_info, args.report)
        print(f"  Report saved to: {args.report}")


if __name__ == "__main__":
    main()
