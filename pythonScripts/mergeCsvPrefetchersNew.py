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

    benchmark = get_column_value(all_configs, ['directory_perf_prev', 'leaf_dir_prev', 'benchmark'])
    all_configs['benchmark'] = benchmark

    # Get the shared prefetcher column and split on '-' into per-core
    prefetch_col = get_column_value(all_configs, ['Prefetch_prev', 'prefetcher', 'Prefetcher'])
    if prefetch_col is not None:
        split_pf = prefetch_col.fillna('none-none').str.split('-', expand=True)
        all_configs['prefetcher_core0'] = split_pf[0]
        all_configs['prefetcher_core1'] = split_pf[1] if split_pf.shape[1] > 1 else split_pf[0]
    else:
        all_configs['prefetcher_core0'] = 'none'
        all_configs['prefetcher_core1'] = 'none'

    all_configs['btbCore0'] = get_column_value(all_configs, ['btbCore0_prev', 'BTB core 0_prev', 'btbCore0'])
    all_configs['btbCore1'] = get_column_value(all_configs, ['btbCore1_prev', 'BTB core 1_prev', 'btbCore1'])

    all_configs['group_key'] = (
        all_configs['benchmark'].astype(str) + '|' +
        all_configs['period_start'].astype(str) + '|' +
        all_configs['period_end'].astype(str)
    )

    print(f"\nProcessing {all_configs['group_key'].nunique()} unique benchmark-interval combinations...")

    results = []

    for group_key, group in all_configs.groupby('group_key'):
        # Explode each row's best/2nd/3rd into individual candidate entries
        candidates = []
        for _, row in group.iterrows():
            for rank, label in [('best', 'best'), ('2nd', '2nd'), ('3rd', '3rd')]:
                ppw_val = pd.to_numeric(row.get(f'PPW_{label}'), errors='coerce')
                btb0    = row.get(f'btbCore0_{label}', row.get('btbCore0'))
                btb1    = row.get(f'btbCore1_{label}', row.get('btbCore1'))
                if pd.notna(ppw_val):
                    candidates.append({
                        'btbCore0':         btb0,
                        'btbCore1':         btb1,
                        'prefetcher_core0': row['prefetcher_core0'],
                        'prefetcher_core1': row['prefetcher_core1'],
                        'ppw':              ppw_val,
                    })

        if not candidates:
            continue

        cand_df  = pd.DataFrame(candidates).sort_values('ppw', ascending=False)
        # Drop duplicate configs, keep highest PPW for each
        cand_df  = cand_df.drop_duplicates(subset=['btbCore0', 'btbCore1', 'prefetcher_core0', 'prefetcher_core1'])
        top3     = cand_df.head(3).reset_index(drop=True)

        result_row = {}

        # Copy common columns from the first row of the group
        skip_cols = {'btbCore0', 'btbCore1', 'prefetcher_core0', 'prefetcher_core1', 'group_key',
                     'btbCore0_best', 'btbCore1_best', 'PPW_best',
                     'btbCore0_2nd',  'btbCore1_2nd',  'PPW_2nd',  'Diff_best_2nd',
                     'btbCore0_3rd',  'btbCore1_3rd',  'PPW_3rd',  'Diff_best_3rd'}
        for col in group.columns:
            if col not in skip_cols:
                result_row[col] = group.iloc[0][col]

        result_row['benchmark'] = group.iloc[0]['benchmark']

        for rank, label in enumerate(['best', '2nd', '3rd']):
            if rank < len(top3):
                result_row[f'btbCore0_{label}']         = top3.iloc[rank]['btbCore0']
                result_row[f'btbCore1_{label}']         = top3.iloc[rank]['btbCore1']
                result_row[f'prefetcher_core0_{label}'] = top3.iloc[rank]['prefetcher_core0']
                result_row[f'prefetcher_core1_{label}'] = top3.iloc[rank]['prefetcher_core1']
                result_row[f'PPW_{label}']              = top3.iloc[rank]['ppw']
                if rank > 0:
                    result_row[f'Diff_best_{label}'] = top3.iloc[0]['ppw'] - top3.iloc[rank]['ppw']
            else:
                result_row[f'btbCore0_{label}']         = None
                result_row[f'btbCore1_{label}']         = None
                result_row[f'prefetcher_core0_{label}'] = None
                result_row[f'prefetcher_core1_{label}'] = None
                result_row[f'PPW_{label}']              = None
                if rank > 0:
                    result_row[f'Diff_best_{label}'] = None

        results.append(result_row)

    output_df = pd.DataFrame(results)

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
        print(f"  Best: {row['btbCore0_best']}/{row['btbCore1_best']} | core0_pf={row['prefetcher_core0_best']} core1_pf={row['prefetcher_core1_best']} -> PPW: {row['PPW_best']:.4e}")
        if pd.notna(row.get('PPW_2nd')):
            print(f"  2nd:  {row['btbCore0_2nd']}/{row['btbCore1_2nd']} | core0_pf={row['prefetcher_core0_2nd']} core1_pf={row['prefetcher_core1_2nd']} -> PPW: {row['PPW_2nd']:.4e} (diff: {row['Diff_best_2nd']:.4e})")
        if pd.notna(row.get('PPW_3rd')):
            print(f"  3rd:  {row['btbCore0_3rd']}/{row['btbCore1_3rd']} | core0_pf={row['prefetcher_core0_3rd']} core1_pf={row['prefetcher_core1_3rd']} -> PPW: {row['PPW_3rd']:.4e} (diff: {row['Diff_best_3rd']:.4e})")

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
