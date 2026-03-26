import pandas as pd
import re

def delete_invalid_period_rows(input_file, output_file=None):
    """
    Delete rows where the 'period' column has matching start and end values.
    
    For example, deletes rows where period is '5:5' or 'a10:a10' (same numeric part)
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional, will overwrite input if not provided)
    
    Returns:
        DataFrame with invalid rows removed
    """
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Check if 'period' column exists
    if 'period' not in df.columns:
        print("❌ No 'period' column found.")
        return df
    
    # Track which rows to keep
    rows_to_keep = []
    rows_to_delete_count = 0
    
    for idx, row in df.iterrows():
        if idx % 100 == 0 and idx > 0:
            print(f"Processed {idx} rows")
        
        period = row['period']
        
        # Check if period is a string and contains ':'
        if isinstance(period, str) and ':' in period:
            parts = period.split(':')
            if len(parts) == 2:
                start, end = parts
                
                # Extract numeric parts only
                numeric_start = re.sub(r'[^\d]', '', start)
                numeric_end = re.sub(r'[^\d]', '', end)
                
                # If numeric parts are the same, mark for deletion
                if numeric_start == numeric_end and numeric_start != '':
                    rows_to_delete_count += 1
                    continue  # Skip this row (don't add to rows_to_keep)
        
        rows_to_keep.append(idx)
    
    print("Found all invalid rows")
    print(f"Will delete {rows_to_delete_count} invalid rows.")
    
    # Keep only valid rows
    df_cleaned = df.loc[rows_to_keep].reset_index(drop=True)
    
    # Save the file
    output_file = output_file or input_file
    df_cleaned.to_csv(output_file, index=False)
    
    print(f"✅ Deleted {rows_to_delete_count} invalid rows.")
    print(f"Output saved to: {output_file}")
    print(f"Rows remaining: {len(df_cleaned)}")
    
    return df_cleaned

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        delete_invalid_period_rows(input_file, output_file)
    else:
        print("Usage: python script.py <input_file.csv> [output_file.csv]")
        print("Example: python script.py fft_perf_new.csv")
        print("\nThis script deletes rows where the 'period' column has matching start/end values.")
        print("E.g., '5:5', 'a10:a10' will be deleted, but '5:10' will be kept.")
