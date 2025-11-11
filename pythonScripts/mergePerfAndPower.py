import pandas as pd
import re

def get_leaf_directory(path):
    if not path or pd.isna(path):
        return ''
    path = str(path).strip()
    path = re.sub(r'^\\.?/', '', path)
    parts = path.split('/')
    return parts[-1] if parts else ''

def normalize_period_value(val):
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

def merge_sheets_by_directory_and_period(file1, file2, output_file, 
                                          period_col_name='period',
                                          period_start_col='period_start',
                                          period_end_col='period_end',
                                          tolerance=100):
    """
    Optimized version — merges using vectorized pandas operations instead of nested loops.
    """
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    print(f"Loaded {len(df1)} rows from {file1}")
    print(f"Loaded {len(df2)} rows from {file2}")

    dir_col1 = df1.columns[0]
    dir_col2 = df2.columns[0]

    # Normalize directories
    df1['leaf_dir'] = df1[dir_col1].apply(get_leaf_directory)
    df2['leaf_dir'] = df2[dir_col2].apply(get_leaf_directory)

    # Extract and normalize periods
    df1[['period_start_val', 'period_end_val']] = df1[period_col_name].astype(str).str.split(':', expand=True).fillna('')
    df1['period_start_val'] = df1['period_start_val'].apply(normalize_period_value)
    df1['period_end_val'] = df1['period_end_val'].apply(normalize_period_value)

    df2['period_start_val'] = df2[period_start_col].apply(normalize_period_value)
    df2['period_end_val'] = df2[period_end_col].apply(normalize_period_value)

    # Merge on leaf directory
    merged = pd.merge(df1, df2, on='leaf_dir', suffixes=('_perf', '_power'))

    # Filter within tolerance
    mask = (
        (merged['period_start_val_perf'] - merged['period_start_val_power']).abs() <= tolerance
    ) & (
        (merged['period_end_val_perf'] - merged['period_end_val_power']).abs() <= tolerance
    )
    merged = merged[mask]

    print(f" Filtered merged rows: {len(merged)}")
    merged.to_csv(output_file, index=False)
    print(f"Output saved to {output_file}")
    return merged

if __name__ == "__main__":
    import sys
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
        print("\\nExample:")
        print("  python script.py fft_perf_new.csv fft_power.csv merged_output.csv")
        print("\\nWith custom column names:")
        print("  python script.py fft_perf_new.csv fft_power.csv merged_output.csv period period_start period_end 100")
