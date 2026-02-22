import pandas as pd
import sys
import os
from pathlib import Path


def merge_benchmark_csvs(file_paths, output_path='merged_top3_configs.csv'):
    print("Loading CSV files...")
    dfs = []
    for i, fp in enumerate(file_paths):
        try:
            df = pd.read_csv(fp)
            print(f"File {i+1} ({fp}): {len(df)} rows loaded")
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {fp}: {e}")
            return

    all_configs = pd.concat(dfs, ignore_index=True)
    print(f"Total combined rows: {len(all_configs)}")

    all_configs.columns = all_configs.columns.str.strip()

    def get_column_value(df, possible_names):
        for name in possible_names:
            if name in df.columns:
                return df[name]
        return None

    btb_core0 = get_column_value(all_configs, ['BTB core 0_prev', 'btbCore0_prev', 'btbCore0'])
    btb_core1 = get_column_value(all_configs, ['BTB core 1_prev', 'btbCore1_prev', 'btbCore1'])
    benchmark  = get_column_value(all_configs, ['leaf_dir_prev', 'directory_perf_prev', 'benchmark'])
    ppw        = get_column_value(all_configs, ['ppw'])

    # Try per-core prefetcher columns first, fall back to shared column for both cores
    prefetcher_core0 = get_column_value(all_configs, ['Prefetch_core0', 'prefetcher_core0', 'PrefetchCore0'])
    prefetcher_core1 = get_column_value(all_configs, ['Prefetch_core1', 'prefetcher_core1', 'PrefetchCore1'])

    if prefetcher_core0 is None or prefetcher_core1 is None:
        shared = get_column_value(all_configs, ['Prefetch_prev', 'prefetcher', 'Prefetcher'])
        if prefetcher_core0 is None:
            prefetcher_core0 = shared
        if prefetcher_core1 is None:
            prefetcher_core1 = shared

    all_configs['btbCore0']         = btb_core0
    all_configs['btbCore1']         = btb_core1
    all_configs['prefetcher_core0'] = prefetcher_core0.fillna('none') if prefetcher_core0 is not None else 'none'
    all_configs['prefetcher_core1'] = prefetcher_core1.fillna('none') if prefetcher_core1 is not None else 'none'
    all_configs['benchmark']        = benchmark
    all_configs['ppw']              = pd.to_numeric(ppw, errors='coerce')

    all_configs['config_full'] = (
        all_configs['btbCore0'].astype(str) + '_' +
        all_configs['btbCore1'].astype(str) + '_' +
        all_configs['prefetcher_core0'].astype(str) + '_' +
        all_configs['prefetcher_core1'].astype(str)
    )

    all_configs['group_key'] = (
        all_configs['benchmark'].astype(str) + '|' +
        all_configs['period_start'].astype(str) + '|' +
        all_configs['period_end'].astype(str)
    )

    print(f"\nProcessing {all_configs['group_key'].nunique()} unique benchmark-interval combinations...")

    skip_cols = {'btbCore0', 'btbCore1', 'prefetcher_core0', 'prefetcher_core1',
                 'ppw', 'config_full', 'group_key'}

    results = []

    for group_key, group in all_configs.groupby('group_key'):
        # Sort by PPW descending (higher is better)
        group_sorted = group.sort_values('ppw', ascending=False)
        top3 = group_sorted.head(3).reset_index(drop=True)

        if len(top3) > 0:
            result_row = {}

            # Copy common columns from the best row
            for col in group.columns:
                if col not in skip_cols:
                    result_row[col] = top3.iloc[0][col]

            # Add best configuration
            result_row['btbCore0_best']         = top3.iloc[0]['btbCore0']
            result_row['btbCore1_best']         = top3.iloc[0]['btbCore1']
            result_row['prefetcher_core0_best'] = top3.iloc[0]['prefetcher_core0']
            result_row['prefetcher_core1_best'] = top3.iloc[0]['prefetcher_core1']
            result_row['PPW_best']              = top3.iloc[0]['ppw']

            # Add 2nd configuration (if exists)
            if len(top3) > 1:
                result_row['btbCore0_2nd']         = top3.iloc[1]['btbCore0']
                result_row['btbCore1_2nd']         = top3.iloc[1]['btbCore1']
                result_row['prefetcher_core0_2nd'] = top3.iloc[1]['prefetcher_core0']
                result_row['prefetcher_core1_2nd'] = top3.iloc[1]['prefetcher_core1']
                result_row['PPW_2nd']              = top3.iloc[1]['ppw']
                result_row['Diff_best_2nd']        = top3.iloc[0]['ppw'] - top3.iloc[1]['ppw']
            else:
                result_row['btbCore0_2nd']         = None
                result_row['btbCore1_2nd']         = None
                result_row['prefetcher_core0_2nd'] = None
                result_row['prefetcher_core1_2nd'] = None
                result_row['PPW_2nd']              = None
                result_row['Diff_best_2nd']        = None

            # Add 3rd configuration (if exists)
            if len(top3) > 2:
                result_row['btbCore0_3rd']         = top3.iloc[2]['btbCore0']
                result_row['btbCore1_3rd']         = top3.iloc[2]['btbCore1']
                result_row['prefetcher_core0_3rd'] = top3.iloc[2]['prefetcher_core0']
                result_row['prefetcher_core1_3rd'] = top3.iloc[2]['prefetcher_core1']
                result_row['PPW_3rd']              = top3.iloc[2]['ppw']
                result_row['Diff_best_3rd']        = top3.iloc[0]['ppw'] - top3.iloc[2]['ppw']
            else:
                result_row['btbCore0_3rd']         = None
                result_row['btbCore1_3rd']         = None
                result_row['prefetcher_core0_3rd'] = None
                result_row['prefetcher_core1_3rd'] = None
                result_row['PPW_3rd']              = None
                result_row['Diff_best_3rd']        = None

            results.append(result_row)

    output_df = pd.DataFrame(results)

    # Organize columns in a logical order
    cols_order = [
        'benchmark', 'period_start', 'period_end',
        'btbCore0_best', 'btbCore1_best', 'prefetcher_core0_best', 'prefetcher_core1_best', 'PPW_best',
        'btbCore0_2nd',  'btbCore1_2nd',  'prefetcher_core0_2nd',  'prefetcher_core1_2nd',  'PPW_2nd',  'Diff_best_2nd',
        'btbCore0_3rd',  'btbCore1_3rd',  'prefetcher_core0_3rd',  'prefetcher_core1_3rd',  'PPW_3rd',  'Diff_best_3rd',
    ]

    other_cols = [col for col in output_df.columns if col not in cols_order and col != 'group_key']
    final_cols = [col for col in cols_order + other_cols if col in output_df.columns]
    output_df  = output_df[final_cols]

    if 'group_key' in output_df.columns:
        output_df = output_df.drop('group_key', axis=1)

    output_df.to_csv(output_path, index=False)
    print(f"\nMerged CSV saved to: {output_path}")
    print(f"Total rows in output: {len(output_df)}")

    print("\n" + "=" * 80)
    print("SAMPLE RESULTS (first 3 benchmark-interval combinations):")
    print("=" * 80)

    for idx, (_, row) in enumerate(output_df.iterrows()):
        if idx >= 3:
            break
        bench_name = Path(str(row['benchmark'])).name if pd.notna(row['benchmark']) else 'unknown'
        print(f"\nBenchmark: {bench_name}")
        print(f"Period: {row['period_start']} to {row['period_end']}")
        print(f"  Best: {row['btbCore0_best']}/{row['btbCore1_best']} | core0_pf={row['prefetcher_core0_best']} core1_pf={row['prefetcher_core1_best']} -> PPW: {row['PPW_best']:.2f}")
        if pd.notna(row['PPW_2nd']):
            print(f"  2nd:  {row['btbCore0_2nd']}/{row['btbCore1_2nd']} | core0_pf={row['prefetcher_core0_2nd']} core1_pf={row['prefetcher_core1_2nd']} -> PPW: {row['PPW_2nd']:.2f} (diff: {row['Diff_best_2nd']:.2f})")
        if pd.notna(row['PPW_3rd']):
            print(f"  3rd:  {row['btbCore0_3rd']}/{row['btbCore1_3rd']} | core0_pf={row['prefetcher_core0_3rd']} core1_pf={row['prefetcher_core1_3rd']} -> PPW: {row['PPW_3rd']:.2f} (diff: {row['Diff_best_3rd']:.2f})")

    return output_df


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_csvs.py <csv1> <csv2> [csv3] [csv4] ... [output_file.csv]")
        print("\nExample:")
        print("  python merge_csvs.py pf_none_none.csv pf_simple_simple.csv pf_simple_none.csv pf_none_simple.csv merged_output.csv")
        sys.exit(1)

    *input_files, last = sys.argv[1:]
    if last.endswith('.csv') and not os.path.exists(last):
        output = last
    else:
        input_files.append(last)
        output = 'merged_top3_configs.csv'

    merge_benchmark_csvs(input_files, output)
