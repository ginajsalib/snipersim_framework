import pandas as pd
import math
import os
import sys

def normalize_period_value(val):
    """Normalize 'period_start' values similar to the Apps Script version."""
    if isinstance(val, str):
        val = val.strip()
        if val == 'roi-begin':
            return 0
        elif val.startswith('periodicins-'):
            try:
                return int(val.replace('periodicins-', ''))
            except ValueError:
                return -1
        elif val.isdigit():
            return int(val)
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return -1

def find_top3_configs_by_ppw(input_csv, output_csv="Top3ConfigsPPW.csv"):
    """
    Reads a local CSV (similar to 'MergedFull' sheet) and computes top 3 configs by PPW.
    Output is written to output_csv.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"File not found: {input_csv}")

    # Read input data
    df = pd.read_csv(input_csv)

    # Flexible column matching - handles both space-separated (L2 core 0)
    # and camelCase (l2Core0) naming depending on which file this runs against.
    wanted_cols = {
        'period_start': 'period_start',
        'ppw': 'ppw',
        'l2_core0': 'L2 core 0',
        'l2_core1': 'L2 core 1',
        'l3': 'L3',
        'prefetch_core0': 'Prefetch core 0',
        'prefetch_core1': 'Prefetch core 1',
        'btb_core0': 'BTB core 0',
        'btb_core1': 'BTB core 1',
    }

    def find_col(expected_name):
        norm = lambda s: s.lower().replace('.', '').replace('_', '').replace(' ', '')
        for col in df.columns:
            if norm(col) == norm(expected_name):
                return col
        for col in df.columns:
            if norm(expected_name) in norm(col):
                return col
        return None

    col_mapping = {}
    missing = []
    for key, expected_name in wanted_cols.items():
        found = find_col(expected_name)
        if found:
            col_mapping[key] = found
        else:
            missing.append(expected_name)
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Available columns: {list(df.columns)}")

    # CRITICAL: force ppw to numeric. If the column was read as object dtype
    # (e.g. due to stray non-numeric values), sort_values would silently fall
    # back to lexicographic (string) comparison, producing wrong "top" rows -
    # this is the most likely cause of the second-higher-than-best symptom.
    df[col_mapping['ppw']] = pd.to_numeric(df[col_mapping['ppw']], errors='coerce')

    # Normalize period_start
    df["norm_period"] = df[col_mapping['period_start']].apply(normalize_period_value)
    df["norm_period"] = (df["norm_period"] // 100) * 100  # group into 100-intervals

    # Group by normalized period
    results = []
    grouped = df.groupby("norm_period")
    for norm_period, group in grouped:
        # Drop rows with NaN/invalid PPW
        group = group.dropna(subset=[col_mapping['ppw']])
        if group.empty:
            continue

        # Sort by PPW descending (numeric, guaranteed by the to_numeric conversion above)
        group_sorted = group.sort_values(by=col_mapping['ppw'], ascending=False).reset_index(drop=True)

        # Sanity check: verify sort actually produced a non-increasing sequence
        ppw_values = group_sorted[col_mapping['ppw']].tolist()
        if any(ppw_values[i] < ppw_values[i + 1] for i in range(len(ppw_values) - 1)):
            print(f"[Warning] Sort order violated for period {norm_period} - "
                  f"PPW values not properly descending: {ppw_values[:5]}...")

        # Get top 3 configs (fill with best if fewer)
        best = group_sorted.iloc[0]
        second = group_sorted.iloc[1] if len(group_sorted) > 1 else best
        third = group_sorted.iloc[2] if len(group_sorted) > 2 else second

        # Diff now represents the drop-off from best (positive = best is better,
        # as the column name implies). Previously computed as (other - best),
        # which was always negative/zero and had the sign backwards.
        diff_best_second = best[col_mapping['ppw']] - second[col_mapping['ppw']]
        diff_best_third = best[col_mapping['ppw']] - third[col_mapping['ppw']]

        def config_fields(row, suffix):
            return {
                f"L2core0_{suffix}": row[col_mapping['l2_core0']],
                f"L2core1_{suffix}": row[col_mapping['l2_core1']],
                f"L3_{suffix}": row[col_mapping['l3']],
                f"Prefetchcore0_{suffix}": row[col_mapping['prefetch_core0']],
                f"Prefetchcore1_{suffix}": row[col_mapping['prefetch_core1']],
                f"BTBcore0_{suffix}": row[col_mapping['btb_core0']],
                f"BTBcore1_{suffix}": row[col_mapping['btb_core1']],
                f"PPW_{suffix}": row[col_mapping['ppw']],
            }

        result_row = {"period_start": norm_period}
        result_row.update(config_fields(best, "best"))
        result_row.update(config_fields(second, "2nd"))
        result_row["Diff_best_2nd"] = diff_best_second
        result_row.update(config_fields(third, "3rd"))
        result_row["Diff_best_3rd"] = diff_best_third

        results.append(result_row)

    # Convert results to DataFrame and save
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    print(f"Top 3 configs per period saved to: {output_csv}")
    print(f"Total periods processed: {len(result_df)}")

# Example usage
if __name__ == "__main__":
    if len(sys.argv) >= 3:
        input_csv = sys.argv[1]
        output_csv = sys.argv[2]
        find_top3_configs_by_ppw(input_csv, output_csv)
    else:
        print("Usage: python script.py <input_file.csv> <output_file.csv>")
