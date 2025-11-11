import pandas as pd
import re

def find_best_btbsizes_per_interval(
    input_csv: str,
    output_csv: str,
    tolerance: int = 100
):
    # Load CSV
    df = pd.read_csv(input_csv)
    
    # Check required columns
    required_cols = ["btbCore0", "btbCore1", "period_start", "period_end", "PPW"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Prefetcher column (optional)
    has_prefetcher = "prefetcher" in df.columns

    # Parse "periodicins-12345" → 12345
    def parse_ins(val):
        if isinstance(val, str):
            m = re.search(r"periodicins-(\d+)", val)
            return int(m.group(1)) if m else None
        return None

    df["start_num"] = df["period_start"].apply(parse_ins)
    df["end_num"] = df["period_end"].apply(parse_ins)

    # Drop invalid rows
    df = df.dropna(subset=["start_num", "end_num", "PPW"])

    # Sort for grouping stability
    df = df.sort_values(by=["start_num", "end_num"]).reset_index(drop=True)

    # Group rows by interval (within tolerance)
    groups = []
    for _, row in df.iterrows():
        start, end = row["start_num"], row["end_num"]
        found = False
        for group in groups:
            if abs(group["start"] - start) <= tolerance and abs(group["end"] - end) <= tolerance:
                group["rows"].append(row)
                found = True
                break
        if not found:
            groups.append({"start": start, "end": end, "rows": [row]})

    # Find best config per group (max PPW)
    output_rows = []
    for group in groups:
        group_df = pd.DataFrame(group["rows"])
        best_row = group_df.loc[group_df["PPW"].idxmax()]
        output = {
            "interval_start": group["start"],
            "interval_end": group["end"],
            "btbCore0": best_row["btbCore0"],
            "btbCore1": best_row["btbCore1"],
            "PPW": best_row["PPW"]
        }
        if has_prefetcher:
            output["prefetcher"] = best_row["prefetcher"]
        output_rows.append(output)

    # Create output dataframe
    result_df = pd.DataFrame(output_rows)

    # Save to output CSV
    result_df.to_csv(output_csv, index=False)
    print(f"✅ Best configurations saved to {output_csv}")


# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        input_csv = sys.argv[1] 
        output_csv = sys.argv[2]
    find_best_btbsizes_per_interval(
        input_csv,
        output_csv,
        tolerance=100
    )
