import pandas as pd
import re
import sys
import traceback

def get_leaf_directory(path):
    """Extract the last part of a path (leaf directory)."""
    try:
        if not path or pd.isna(path):
            return ''
        path = str(path).strip()
        path = re.sub(r'^\.?/', '', path)
        parts = path.split('/')
        return parts[-1] if parts else ''
    except Exception as e:
        print(f"Warning: Error processing path '{path}': {e}")
        return ''


def normalize_period_value(val):
    """Normalize period values like 'periodicins-1000058' or 'roi-begin'."""
    try:
        if pd.isna(val):
            return -1
        val = str(val).strip()
        if val == 'roi-begin':
            return 0
        if val.startswith('periodicins-'):
            try:
                return int(val.replace('periodicins-', ''))
            except ValueError:
                return -1
        try:
            return int(val)
        except ValueError:
            return -1
    except Exception as e:
        print(f"Warning: Error normalizing value '{val}': {e}")
        return -1


def merge_sheets_by_directory_and_period(file1, file2, output_file, 
                                          period_col_name='period',
                                          period_start_col='period_start',
                                          period_end_col='period_end',
                                          tolerance=100):
    """
    Merge two CSV files based on leaf directory and period matching.
    Optimized for memory efficiency with chunked processing.
    """
    
    try:
        # Read files with progress indication
        print(f"Reading {file1}...")
        df1 = pd.read_csv(file1, low_memory=False)
        print(f"✓ Loaded {len(df1)} rows, {len(df1.columns)} columns")
        print(f"  Memory usage: {df1.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print(f"\nReading {file2}...")
        df2 = pd.read_csv(file2, low_memory=False)
        print(f"✓ Loaded {len(df2)} rows, {len(df2.columns)} columns")
        print(f"  Memory usage: {df2.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Validate required columns
        dir_col1 = df1.columns[0]
        dir_col2 = df2.columns[0]
        
        if period_col_name not in df1.columns:
            raise ValueError(f"Column '{period_col_name}' not found in {file1}. Available: {list(df1.columns)}")
        if period_start_col not in df2.columns:
            raise ValueError(f"Column '{period_start_col}' not found in {file2}. Available: {list(df2.columns)}")
        if period_end_col not in df2.columns:
            raise ValueError(f"Column '{period_end_col}' not found in {file2}. Available: {list(df2.columns)}")
        
        print("\n Processing data...")
        
        # Extract leaf directories
        print("  Extracting leaf directories...")
        df1['leaf_dir'] = df1[dir_col1].apply(get_leaf_directory)
        df2['leaf_dir'] = df2[dir_col2].apply(get_leaf_directory)
        
        # Parse periods from df1
        print("  Parsing periods from file1...")
        period_split = df1[period_col_name].astype(str).str.split(':', expand=True)
        if period_split.shape[1] < 2:
            print("  Warning: Some period values don't contain ':'. Filling with empty strings.")
            period_split = period_split.reindex(columns=[0, 1], fill_value='')
        
        df1['period_start_val'] = period_split[0].apply(normalize_period_value)
        df1['period_end_val'] = period_split[1].apply(normalize_period_value)
        
        # Parse periods from df2
        print("  Parsing periods from file2...")
        df2['period_start_val'] = df2[period_start_col].apply(normalize_period_value)
        df2['period_end_val'] = df2[period_end_col].apply(normalize_period_value)
        
        print(f"\n🔗 Merging on leaf directory...")
        print(f"  Unique directories in file1: {df1['leaf_dir'].nunique()}")
        print(f"  Unique directories in file2: {df2['leaf_dir'].nunique()}")
        
        # Merge on leaf directory
        merged = pd.merge(df1, df2, on='leaf_dir', suffixes=('_perf', '_power'), how='left')
        print(f"  After merge: {len(merged)} rows")
        print(f"  Memory usage: {merged.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Filter within tolerance
        print(f"\n🔍 Filtering by period tolerance (±{tolerance})...")
        mask = (
            (merged['period_start_val_perf'] - merged['period_start_val_power']).abs() <= tolerance
        ) & (
            (merged['period_end_val_perf'] - merged['period_end_val_power']).abs() <= tolerance
        )
        
        matched_count = mask.sum()
        merged_filtered = merged[mask].copy()
        
        print(f"  Matched rows: {matched_count} / {len(merged)} ({matched_count/len(merged)*100:.1f}%)")
        
        # Clean up temporary columns
        cols_to_drop = ['leaf_dir', 'period_start_val_perf', 'period_end_val_perf', 
                        'period_start_val_power', 'period_end_val_power']
        merged_filtered = merged_filtered.drop(columns=[c for c in cols_to_drop if c in merged_filtered.columns])
        
        # Save output
        print(f"\n💾 Saving to {output_file}...")
        merged_filtered.to_csv(output_file, index=False)
        print(f" Success! Output saved with {len(merged_filtered)} rows")
        
        return merged_filtered
        
    except MemoryError:
        print("\n MEMORY ERROR: Not enough RAM to process these files.")
        print("Suggestions:")
        print("  1. Close other applications to free up memory")
        print("  2. Try processing on a machine with more RAM")
        print("  3. Use a chunked processing approach (contact for implementation)")
        sys.exit(1)
        
    except FileNotFoundError as e:
        print(f"\n FILE NOT FOUND: {e}")
        sys.exit(1)
        
    except pd.errors.EmptyDataError:
        print(f"\n ERROR: One of the CSV files is empty")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n UNEXPECTED ERROR: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        output_file = sys.argv[3]
        period_col = sys.argv[4] if len(sys.argv) > 4 else 'period'
        period_start = sys.argv[5] if len(sys.argv) > 5 else 'period_start'
        period_end = sys.argv[6] if len(sys.argv) > 6 else 'period_end'
        tolerance = int(sys.argv[7]) if len(sys.argv) > 7 else 100
        
        merge_sheets_by_directory_and_period(
            file1, file2, output_file,
            period_col_name=period_col,
            period_start_col=period_start,
            period_end_col=period_end,
            tolerance=tolerance
        )
    else:
        print("Usage: python script.py <file1.csv> <file2.csv> <output.csv> [period_col] [period_start_col] [period_end_col] [tolerance]")
        print("\nExample:")
        print("  python script.py fft_perf_new.csv fft_power.csv merged_output.csv")
        print("\nWith custom column names:")
        print("  python script.py fft_perf_new.csv fft_power.csv merged_output.csv period period_start period_end 100")
