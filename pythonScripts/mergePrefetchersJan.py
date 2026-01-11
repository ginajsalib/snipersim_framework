import pandas as pd
import numpy as np
from collections import defaultdict

def extract_period_number(period_str):
    """Extract numeric value from period string like 'periodicins-100000028'"""
    if pd.isna(period_str):
        return None
    try:
        return int(str(period_str).split('-')[-1])
    except:
        return None

def create_period_bucket(period_num, bucket_size=100):
    """Create bucket ID for period grouping"""
    if period_num is None:
        return None
    return period_num // bucket_size

def merge_and_compare_prefetchers(file_none, file_simple, output_file, period_tolerance=100):
    """
    Merge two CSV files with different prefetcher configurations and 
    determine best configs considering prefetcher type.
    
    Args:
        file_none: Path to CSV with prefetcher=none
        file_simple: Path to CSV with prefetcher=simple
        output_file: Path for output CSV
        period_tolerance: Allowed difference in period values (default: 100)
    """
    
    print("Reading CSV files...")
    # Read both CSV files
    df_none = pd.read_csv(file_none)
    df_simple = pd.read_csv(file_simple)
    
    # Add prefetcher identifier column
    df_none['prefetcher_type'] = 'none'
    df_simple['prefetcher_type'] = 'simple'
    
    # Combine both dataframes
    df_combined = pd.concat([df_none, df_simple], ignore_index=True)
    
    print(f"Total rows to process: {len(df_combined)}")
    
    # Convert PPW columns to numeric upfront
    ppw_cols = ['PPW_best', 'PPW_2nd', 'PPW_3rd']
    for col in ppw_cols:
        df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
    
    # Extract period numbers and create buckets for faster matching
    print("Preprocessing periods...")
    df_combined['period_start_num'] = df_combined['period_start'].apply(extract_period_number)
    df_combined['period_end_num'] = df_combined['period_end'].apply(extract_period_number)
    df_combined['period_start_bucket'] = df_combined['period_start_num'].apply(
        lambda x: create_period_bucket(x, period_tolerance))
    df_combined['period_end_bucket'] = df_combined['period_end_num'].apply(
        lambda x: create_period_bucket(x, period_tolerance))
    
    # Create grouping key (excluding periods which need lenient matching)
    print("Creating grouping keys...")
    df_combined['group_key'] = (
        df_combined['L2_prev'].astype(str) + '|' +
        df_combined['L3_prev'].astype(str) + '|' +
        df_combined['BTB core 0_prev'].astype(str) + '|' +
        df_combined['BTB core 1_prev'].astype(str) + '|' +
        df_combined['period_start_bucket'].astype(str) + '|' +
        df_combined['period_end_bucket'].astype(str)
    )
    
    # Group by the key
    print("Grouping rows...")
    grouped = df_combined.groupby('group_key')
    
    results = []
    total_groups = len(grouped)
    
    print(f"Processing {total_groups} groups...")
    
    for i, (group_key, group_df) in enumerate(grouped):
        if i % 100 == 0:
            print(f"Progress: {i}/{total_groups} groups processed")
        
        # Further filter within bucket for exact period tolerance
        if len(group_df) > 1:
            final_group = []
            processed = set()
            
            for idx, row in group_df.iterrows():
                if idx in processed:
                    continue
                
                matching = [row]
                processed.add(idx)
                
                for idx2, row2 in group_df.iterrows():
                    if idx2 in processed:
                        continue
                    
                    # Check period tolerance
                    start_match = abs(row['period_start_num'] - row2['period_start_num']) <= period_tolerance
                    end_match = abs(row['period_end_num'] - row2['period_end_num']) <= period_tolerance
                    
                    if start_match and end_match:
                        matching.append(row2)
                        processed.add(idx2)
                
                final_group.append(matching)
        else:
            final_group = [[group_df.iloc[0]]]
        
        # Process each sub-group
        for matching_rows in final_group:
            if len(matching_rows) == 1:
                # Single row - add prefetcher info to best/2nd/3rd columns
                row_dict = matching_rows[0].to_dict()
                prefetcher = row_dict['prefetcher_type']
                row_dict['prefetcher_best'] = prefetcher
                row_dict['prefetcher_2nd'] = prefetcher
                row_dict['prefetcher_3rd'] = prefetcher
                results.append(row_dict)
            else:
                # Collect all candidates using vectorized operations
                candidates = []
                
                for row in matching_rows:
                    prefetcher = row['prefetcher_type']
                    
                    candidates.extend([
                        {
                            'btbCore0': row['btbCore0_best'],
                            'btbCore1': row['btbCore1_best'],
                            'PPW': row['PPW_best'] if not pd.isna(row['PPW_best']) else float('inf'),
                            'prefetcher': prefetcher,
                            'source_row': row
                        },
                        {
                            'btbCore0': row['btbCore0_2nd'],
                            'btbCore1': row['btbCore1_2nd'],
                            'PPW': row['PPW_2nd'] if not pd.isna(row['PPW_2nd']) else float('inf'),
                            'prefetcher': prefetcher,
                            'source_row': row
                        },
                        {
                            'btbCore0': row['btbCore0_3rd'],
                            'btbCore1': row['btbCore1_3rd'],
                            'PPW': row['PPW_3rd'] if not pd.isna(row['PPW_3rd']) else float('inf'),
                            'prefetcher': prefetcher,
                            'source_row': row
                        }
                    ])
                
                # Sort by PPW (lower is better)
                candidates_sorted = sorted(candidates, key=lambda x: x['PPW'])
                top3 = candidates_sorted[:3]
                
                # Build merged row
                base_row = matching_rows[0].to_dict()
                
                base_row['btbCore0_best'] = top3[0]['btbCore0']
                base_row['btbCore1_best'] = top3[0]['btbCore1']
                base_row['PPW_best'] = top3[0]['PPW']
                base_row['prefetcher_best'] = top3[0]['prefetcher']
                
                if len(top3) > 1:
                    base_row['btbCore0_2nd'] = top3[1]['btbCore0']
                    base_row['btbCore1_2nd'] = top3[1]['btbCore1']
                    base_row['PPW_2nd'] = top3[1]['PPW']
                    base_row['prefetcher_2nd'] = top3[1]['prefetcher']
                    base_row['Diff_best_2nd'] = top3[1]['PPW'] - top3[0]['PPW']
                
                if len(top3) > 2:
                    base_row['btbCore0_3rd'] = top3[2]['btbCore0']
                    base_row['btbCore1_3rd'] = top3[2]['btbCore1']
                    base_row['PPW_3rd'] = top3[2]['PPW']
                    base_row['prefetcher_3rd'] = top3[2]['prefetcher']
                    base_row['Diff_best_3rd'] = top3[2]['PPW'] - top3[0]['PPW']
                
                results.append(base_row)
    
    print("Creating final dataframe...")
    df_result = pd.DataFrame(results)
    
    # Remove temporary columns
    cols_to_drop = ['prefetcher_type', 'period_start_num', 'period_end_num', 
                    'period_start_bucket', 'period_end_bucket', 'group_key']
    df_result = df_result.drop(columns=[col for col in cols_to_drop if col in df_result.columns])
    
    print("Saving to CSV...")
    df_result.to_csv(output_file, index=False)
    print(f"✓ Merged CSV saved to {output_file}")
    print(f"✓ Total rows: {len(df_result)}")
    
    return df_result


# Example usage:
if __name__ == "__main__":
    # Replace these with your actual file paths
    file_none = "/home/gina/Desktop/snipersim_framework/pythonScripts/train_with_top3_barnes.csv"
    file_simple = "/home/gina/Desktop/snipersim_framework/pythonScripts/train_with_top3_barnes_simple.csv"
    output_file = "barnes_merged_prefetcher_configs.csv"
    
    df_merged = merge_and_compare_prefetchers(file_none, file_simple, output_file)
    
    # Display first few rows
    print("\nFirst few rows of merged data:")
    print(df_merged.head())
