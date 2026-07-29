import re
import pandas as pd

def parse_benchmark_data(input_file, output_file=None):
    """
    Parse benchmark configuration strings from column A and extract values into separate columns.

    Handles both old (single value = symmetric across cores) and new
    (explicit per-core value) directory naming formats for L2, prefetcher, and BTB.

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
    new_columns = ['L2 core 0', 'L2 core 1', 'L3', 'Prefetch core 0', 'Prefetch core 1', 'BTB core 0', 'BTB core 1']

    # Insert columns at position 1, 2, 3, ... (right after first column)
    for i, col in enumerate(new_columns, start=1):
        if col not in df.columns:
            df.insert(i, col, None)

    # Regex patterns - each captures an optional second (core 1) value,
    # falling back to the core 0 value when only one is present (old / symmetric format)
    l2_pattern = re.compile(r'l2_(\d+)(?:_(\d+))?')
    l3_pattern = re.compile(r'l3MB_(\d+)')
    prefetch_pattern = re.compile(r'prefetch_([a-zA-Z]+)(?:[-_]([a-zA-Z]+))?')
    btb_pattern = re.compile(r'branch_(\d+)(?:-(\d+))?')

    unmatched_rows = []

    # Process each row
    for idx, row in df.iterrows():
        raw_string = str(row[first_col])

        l2_match = l2_pattern.search(raw_string)
        l3_match = l3_pattern.search(raw_string)
        prefetch_match = prefetch_pattern.search(raw_string)
        btb_match = btb_pattern.search(raw_string)

        # L2: core 0 required, core 1 falls back to core 0 if not split
        if l2_match:
            l2_core0 = float(l2_match.group(1))
            l2_core1 = float(l2_match.group(2)) if l2_match.group(2) is not None else l2_core0
        else:
            l2_core0 = l2_core1 = None

        # L3 is shared across cores - single value
        l3_size = float(l3_match.group(1)) if l3_match else None

        # Prefetch: core 0 required, core 1 falls back to core 0 if not split
        if prefetch_match:
            prefetch_core0 = prefetch_match.group(1)
            prefetch_core1 = prefetch_match.group(2) if prefetch_match.group(2) is not None else prefetch_core0
        else:
            prefetch_core0 = prefetch_core1 = None

        # BTB: core 0 required, core 1 falls back to core 0 if not split
        if btb_match:
            btb_core0 = float(btb_match.group(1))
            btb_core1 = float(btb_match.group(2)) if btb_match.group(2) is not None else btb_core0
        else:
            btb_core0 = btb_core1 = None

        df.at[idx, 'L2 core 0'] = l2_core0
        df.at[idx, 'L2 core 1'] = l2_core1
        df.at[idx, 'L3'] = l3_size
        df.at[idx, 'Prefetch core 0'] = prefetch_core0
        df.at[idx, 'Prefetch core 1'] = prefetch_core1
        df.at[idx, 'BTB core 0'] = btb_core0
        df.at[idx, 'BTB core 1'] = btb_core1

        # flag rows where anything failed to parse, for a post-run sanity check
        if None in (l2_core0, l3_size, prefetch_core0, btb_core0):
            unmatched_rows.append((idx, raw_string))

    # Save the file
    output_file = output_file or input_file
    df.to_csv(output_file, index=False)

    print(f"Successfully processed {len(df)} rows")
    if unmatched_rows:
        print(f"[Warning] {len(unmatched_rows)} row(s) had at least one field that failed to parse:")
        for idx, raw in unmatched_rows[:10]:
            print(f"  row {idx}: {raw}")
        if len(unmatched_rows) > 10:
            print(f"  ... and {len(unmatched_rows) - 10} more")
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
