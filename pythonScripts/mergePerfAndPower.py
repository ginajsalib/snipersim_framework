import pandas as pd
import re

def get_leaf_directory(path):
    """
    Extract the last part of a path (leaf directory).
    
    Args:
        path: File path string
    
    Returns:
        Last directory/file name in path
    """
    if not path or pd.isna(path):
        return ''
    
    path = str(path).strip()
    # Remove leading "./" or "/"
    path = re.sub(r'^\.?/', '', path)
    
    parts = path.split('/')
    return parts[-1] if parts else ''


def normalize_period_value(val):
    """
    Normalize period values like "periodicins-1000058" or "roi-begin".
    
    Args:
        val: Period value to normalize
    
    Returns:
        Integer representation of the period value
    """
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
    
    # Try to parse as integer
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
    Merge two CSV files based on leaf directory and period matching.
    
    Args:
        file1: Path to first CSV file (e.g., fft_perf_new.csv)
        file2: Path to second CSV file (e.g., fft_power.csv)
        output_file: Path to output merged CSV file
        period_col_name: Name of the period column in file1 (default: 'period')
        period_start_col: Name of period_start column in file2 (default: 'period_start')
        period_end_col: Name of period_end column in file2 (default: 'period_end')
        tolerance: Matching tolerance for period values (default: 100)
    
    Returns:
        Merged DataFrame
    """
    
    # Read both CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    print(f"Loaded {len(df1)} rows from {file1}")
    print(f"Loaded {len(df2)} rows from {file2}")
    
    # Get first column name (assumed to be directory)
    dir_col1 = df1.columns[0]
    dir_col2 = df2.columns[0]
    
    # Find period column in df1
    if period_col_name not in df1.columns:
        raise ValueError(f"❌ Column '{period_col_name}' not found in {file1}")
    
    # Check for period columns in df2
    if period_start_col not in df2.columns or period_end_col not in df2.columns:
        raise ValueError(f"❌ Columns '{period_start_col}' or '{period_end_col}' not found in {file2}")
    
    # Create combined headers (exclude first column from df2 to avoid duplication)
    combined_columns = list(df1.columns) + list(df2.columns[1:])
    
    # Prepare output rows
    output_rows = []
    
    # Process each row in df1
    for idx, row1 in df1.iterrows():
        if idx % 100 == 0:
            print(f"Processing row {idx}/{len(df1)}")
        
        dir1 = get_leaf_directory(row1[dir_col1])
        period = str(row1[period_col_name])
        
        # Split period into start and end
        if ':' in period:
            parts = period.split(':')
            if len(parts) == 2:
                start_raw, end_raw = parts
                start = normalize_period_value(start_raw)
                end = normalize_period_value(end_raw)
            else:
                # Invalid period format, add empty columns
                output_row = list(row1) + [None] * (len(df2.columns) - 1)
                output_rows.append(output_row)
                continue
        else:
            # No colon in period, add empty columns
            output_row = list(row1) + [None] * (len(df2.columns) - 1)
            output_rows.append(output_row)
            continue
        
        # Find matching row in df2
        matched_row = None
        
        for idx2, row2 in df2.iterrows():
            dir2 = get_leaf_directory(row2[dir_col2])
            
            ps = normalize_period_value(row2[period_start_col])
            pe = normalize_period_value(row2[period_end_col])
            
            # Check if directories match and periods are within tolerance
            if (dir1 == dir2 and 
                abs(start - ps) <= tolerance and 
                abs(end - pe) <= tolerance):
                matched_row = row2
                break
        
        # Combine rows
        if matched_row is not None:
            output_row = list(row1) + list(matched_row[1:])  # Exclude first column of row2
        else:
            output_row = list(row1) + [None] * (len(df2.columns) - 1)
        
        output_rows.append(output_row)
    
    # Create output DataFrame
    df_merged = pd.DataFrame(output_rows, columns=combined_columns)
    
    # Save to CSV
    df_merged.to_csv(output_file, index=False)
    
    print(f"✅ Merged {len(df1)} rows")
    print(f"Output saved to: {output_file}")
    
    return df_merged


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 4:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        output_file = sys.argv[3]
        
        # Optional parameters
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
        print("\nThis script merges two CSV files based on:")
        print("  - Matching leaf directory names")
        print("  - Matching period ranges (with tolerance of ±100 by default)")
