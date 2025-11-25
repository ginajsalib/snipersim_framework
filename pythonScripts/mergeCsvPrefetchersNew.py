import pandas as pd
import sys
from pathlib import Path


def merge_benchmark_csvs(file1_path, file2_path, output_path='merged_top3_configs.csv'):
    print("Loading CSV files...")
    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        print(f"File 1: {len(df1)} rows loaded")
        print(f"File 2: {len(df2)} rows loaded")
    except Exception as e:
        print(f"Error loading files: {e}")
        return
    
    all_configs = pd.concat([df1, df2], ignore_index=True)
    print(f"Total combined rows: {len(all_configs)}")
    
    all_configs.columns = all_configs.columns.str.strip()
    
    def get_column_value(df, possible_names):
        for name in possible_names:
            if name in df.columns:
                return df[name]
        return None
    
    btb_core0 = get_column_value(all_configs, ['BTB core 0_prev', 'btbCore0_prev', 'btbCore0'])
    btb_core1 = get_column_value(all_configs, ['BTB core 1_prev', 'btbCore1_prev', 'btbCore1'])
    prefetcher = get_column_value(all_configs, ['Prefetch_prev', 'prefetcher', 'Prefetcher'])
    benchmark = get_column_value(all_configs, ['leaf_dir_prev', 'directory_perf_prev', 'benchmark'])
    ppw = get_column_value(all_configs, ['ppw_prev', 'PPW_best', 'PPW'])
    
    all_configs['btbCore0'] = btb_core0
    all_configs['btbCore1'] = btb_core1
    all_configs['prefetcher'] = prefetcher.fillna('none')
    all_configs['benchmark'] = benchmark
    all_configs['ppw'] = pd.to_numeric(ppw, errors='coerce')
    
    all_configs['config_full'] = (all_configs['btbCore0'].astype(str) + '_' + 
                                   all_configs['btbCore1'].astype(str) + '_' + 
                                   all_configs['prefetcher'].astype(str))
    
    all_configs['group_key'] = (all_configs['benchmark'].astype(str) + '|' + 
                                 all_configs['period_start'].astype(str) + '|' + 
                                 all_configs['period_end'].astype(str))
    
    print(f"\nProcessing {all_configs['group_key'].nunique()} unique benchmark-interval combinations...")
    
    results = []
    
    for group_key, group in all_configs.groupby('group_key'):
        # Sort by PPW descending (higher is better)
        group_sorted = group.sort_values('ppw', ascending=False)
        top3 = group_sorted.head(3).reset_index(drop=True)
        
        if len(top3) > 0:
            # Create a single row with all top-3 configurations
            result_row = {}
            
            # Copy common columns from the first row
            for col in group.columns:
                if col not in ['btbCore0', 'btbCore1', 'prefetcher', 'ppw', 'config_full', 'group_key']:
                    result_row[col] = top3.iloc[0][col]
            
            # Add best configuration
            result_row['btbCore0_best'] = top3.iloc[0]['btbCore0']
            result_row['btbCore1_best'] = top3.iloc[0]['btbCore1']
            result_row['prefetcher_best'] = top3.iloc[0]['prefetcher']
            result_row['PPW_best'] = top3.iloc[0]['ppw']
            
            # Add 2nd configuration (if exists)
            if len(top3) > 1:
                result_row['btbCore0_2nd'] = top3.iloc[1]['btbCore0']
                result_row['btbCore1_2nd'] = top3.iloc[1]['btbCore1']
                result_row['prefetcher_2nd'] = top3.iloc[1]['prefetcher']
                result_row['PPW_2nd'] = top3.iloc[1]['ppw']
                result_row['Diff_best_2nd'] = top3.iloc[0]['ppw'] - top3.iloc[1]['ppw']
            else:
                result_row['btbCore0_2nd'] = None
                result_row['btbCore1_2nd'] = None
                result_row['prefetcher_2nd'] = None
                result_row['PPW_2nd'] = None
                result_row['Diff_best_2nd'] = None
            
            # Add 3rd configuration (if exists)
            if len(top3) > 2:
                result_row['btbCore0_3rd'] = top3.iloc[2]['btbCore0']
                result_row['btbCore1_3rd'] = top3.iloc[2]['btbCore1']
                result_row['prefetcher_3rd'] = top3.iloc[2]['prefetcher']
                result_row['PPW_3rd'] = top3.iloc[2]['ppw']
                result_row['Diff_best_3rd'] = top3.iloc[0]['ppw'] - top3.iloc[2]['ppw']
            else:
                result_row['btbCore0_3rd'] = None
                result_row['btbCore1_3rd'] = None
                result_row['prefetcher_3rd'] = None
                result_row['PPW_3rd'] = None
                result_row['Diff_best_3rd'] = None
            
            results.append(result_row)
    
    output_df = pd.DataFrame(results)
    
    # Organize columns in a logical order
    cols_order = [
        'benchmark', 'period_start', 'period_end',
        'btbCore0_best', 'btbCore1_best', 'prefetcher_best', 'PPW_best',
        'btbCore0_2nd', 'btbCore1_2nd', 'prefetcher_2nd', 'PPW_2nd', 'Diff_best_2nd',
        'btbCore0_3rd', 'btbCore1_3rd', 'prefetcher_3rd', 'PPW_3rd', 'Diff_best_3rd'
    ]
    
    # Add remaining columns
    other_cols = [col for col in output_df.columns if col not in cols_order and col != 'group_key']
    final_cols = cols_order + other_cols
    final_cols = [col for col in final_cols if col in output_df.columns]
    
    output_df = output_df[final_cols]
    
    if 'group_key' in output_df.columns:
        output_df = output_df.drop('group_key', axis=1)
    
    output_df.to_csv(output_path, index=False)
    print(f"\nMerged CSV saved to: {output_path}")
    print(f"Total rows in output: {len(output_df)}")
    
    print("\n" + "="*80)
    print("SAMPLE RESULTS (first 3 benchmark-interval combinations):")
    print("="*80)
    
    for idx, (_, row) in enumerate(output_df.iterrows()):
        if idx >= 3:
            break
        bench_name = Path(str(row['benchmark'])).name if pd.notna(row['benchmark']) else 'unknown'
        print(f"\nBenchmark: {bench_name}")
        print(f"Period: {row['period_start']} to {row['period_end']}")
        print(f"  Best: {row['btbCore0_best']}/{row['btbCore1_best']}/{row['prefetcher_best']} -> PPW: {row['PPW_best']:.2f}")
        if pd.notna(row['PPW_2nd']):
            print(f"  2nd:  {row['btbCore0_2nd']}/{row['btbCore1_2nd']}/{row['prefetcher_2nd']} -> PPW: {row['PPW_2nd']:.2f} (diff: {row['Diff_best_2nd']:.2f})")
        if pd.notna(row['PPW_3rd']):
            print(f"  3rd:  {row['btbCore0_3rd']}/{row['btbCore1_3rd']}/{row['prefetcher_3rd']} -> PPW: {row['PPW_3rd']:.2f} (diff: {row['Diff_best_3rd']:.2f})")
    
    return output_df


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_csvs.py <csv_file1> <csv_file2> [output_file]")
        print("\nExample:")
        print("  python merge_csvs.py prefetch_none.csv prefetch_simple.csv merged_output.csv")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else 'merged_top3_configs.csv'
    
    merge_benchmark_csvs(file1, file2, output)
