import re
import pandas as pd

def parse_benchmark_data(input_file, output_file=None):
    """
    Parse benchmark configuration strings from column A and extract values into separate columns.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional, will overwrite input if not provided)
    
    Returns:
        DataFrame with parsed data
    """
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Get the first column name (assuming config strings are in the first column)
    first_col = df.columns[0]
    
    # Add new columns after the first column if they don't exist
    new_columns = ['L2', 'L3', 'Prefetch', 'BTB core 0', 'BTB core 1']
    
    # Insert columns at position 1, 2, 3, 4, 5 (right after first column)
    for i, col in enumerate(new_columns, start=1):
        if col not in df.columns:
            df.insert(i, col, None)
    
    # Process each row
    for idx, row in df.iterrows():
        raw_string = str(row[first_col])
        
        # Extract values using regex
        l2_match = re.search(r'l2_(\d+)', raw_string)
        l3_match = re.search(r'l3MB_(\d+)', raw_string)
        prefetch_match = re.search(r'prefetch_([^_]+)', raw_string)
        branch_match = re.search(r'branch_(\d+)', raw_string)
        branch_match2 = re.search(r'branch_\d+-(\d+)', raw_string)
        
        # Set values (convert to appropriate types)
        df.at[idx, 'L2'] = float(l2_match.group(1)) if l2_match else None
        df.at[idx, 'L3'] = float(l3_match.group(1)) if l3_match else None
        df.at[idx, 'Prefetch'] = prefetch_match.group(1) if prefetch_match else None
        df.at[idx, 'BTB core 0'] = float(branch_match.group(1)) if branch_match else None
        df.at[idx, 'BTB core 1'] = float(branch_match2.group(1)) if branch_match2 else None
    
    # Save the file
    output_file = output_file or input_file
    df.to_csv(output_file, index=False)
    
    print(f"Successfully processed {len(df)} rows")
    print(f"Output saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    # Example usage:
    # parse_benchmark_data("benchmark_perf_new.csv")
    # Or specify different output file:
    # parse_benchmark_data("benchmark_perf_new.csv", "benchmark_perf_processed.csv")
    
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        parse_benchmark_data(input_file, output_file)
    else:
        print("Usage: python script.py <input_file.csv> [output_file.csv]")
        print("Example: python script.py benchmark_perf_new.csv")
