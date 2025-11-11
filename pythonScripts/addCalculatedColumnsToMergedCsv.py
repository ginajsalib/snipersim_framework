import pandas as pd

def add_calculated_columns(input_file, output_file=None):
    """
    Add calculated columns for time, IPS, IPS^3, PPW, and config.
    
    Columns added:
    - time_seconds: performance_model.elapsed_time_core0 / 1000000000
    - ips: (core.instructions_core0 + core.instructions_core1) / time_seconds
    - ips_cubed: ips^3
    - ppw: ips^3 / total_power
    - config: concatenation of btbCore0, btbCore1, and prefetcher
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional, will overwrite input if not provided)
    
    Returns:
        DataFrame with new columns added
    """
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"Loaded {len(df)} rows from {input_file}")
    print(f"Columns in file: {list(df.columns)}")
    
    # Check required columns exist
    required_cols = {
        'elapsed_time': 'performance_model.elapsed_time_core0',
        'instructions_core0': 'core.instructions_core0',
        'instructions_core1': 'core.instructions_core1',
        'total_power': 'total_power',
        'btb_core0': 'BTB core 0',
        'btb_core1': 'BTB core 1',
        'prefetcher': 'Prefetch'
    }
    
    # Find actual column names (case-insensitive matching)
    col_mapping = {}
    for key, expected_name in required_cols.items():
        found = False
        for col in df.columns:
            if col.lower() == expected_name.lower():
                col_mapping[key] = col
                found = True
                break
        if not found:
            print(f"⚠️  Warning: Column '{expected_name}' not found. Looking for alternatives...")
            # Try partial match
            for col in df.columns:
                if expected_name.lower().replace('.', '').replace('_', '') in col.lower().replace('.', '').replace('_', ''):
                    col_mapping[key] = col
                    print(f"   Using '{col}' for '{expected_name}'")
                    found = True
                    break
            if not found:
                print(f"❌ Error: Could not find column for '{expected_name}'")
                print(f"   Available columns: {list(df.columns)}")
                return df
    
    print("\n✅ Column mapping:")
    for key, col in col_mapping.items():
        print(f"   {key}: {col}")
    
    # Calculate time_seconds
    print("\nCalculating time_seconds...")
    df['time_seconds'] = df[col_mapping['elapsed_time']] / 1_000_000_000
    
    # Calculate IPS (Instructions Per Second)
    print("Calculating ips...")
    df['ips'] = (df[col_mapping['instructions_core0']] + df[col_mapping['instructions_core1']]) / df['time_seconds']
    
    # Calculate IPS^3
    print("Calculating ips_cubed...")
    df['ips_cubed'] = df['ips'] ** 3
    
    # Calculate PPW (Performance Per Watt)
    print("Calculating ppw...")
    df['ppw'] = df['ips_cubed'] / df[col_mapping['total_power']]
    
    # Create config column
    print("Creating config column...")
    df['config'] = (
        df[col_mapping['btb_core0']].astype(str) + '_' + 
        df[col_mapping['btb_core1']].astype(str) + '_' + 
        df[col_mapping['prefetcher']].astype(str)
    )
    
    # Handle NaN and inf values
    print("\nHandling invalid values...")
    for col in ['time_seconds', 'ips', 'ips_cubed', 'ppw']:
        nan_count = df[col].isna().sum()
        inf_count = (df[col] == float('inf')).sum()
        if nan_count > 0:
            print(f"   {col}: {nan_count} NaN values")
        if inf_count > 0:
            print(f"   {col}: {inf_count} infinite values")
    
    # Save the file
    output_file = output_file or input_file
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Added columns: time_seconds, ips, ips_cubed, ppw, config")
    print(f"Output saved to: {output_file}")
    print(f"\nSample of new columns:")
    print(df[['time_seconds', 'ips', 'ips_cubed', 'ppw', 'config']].head())
    
    return df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        add_calculated_columns(input_file, output_file)
    else:
        print("Usage: python script.py <input_file.csv> [output_file.csv]")
        print("Example: python script.py merged_data.csv")
        print("\nThis script adds the following calculated columns:")
        print("  - time_seconds: elapsed_time / 1,000,000,000")
        print("  - ips: (instructions_core0 + instructions_core1) / time_seconds")
        print("  - ips_cubed: ips^3")
        print("  - ppw: ips^3 / total_power")
        print("  - config: btbCore0_btbCore1_prefetcher")
