import pandas as pd
import re
import sys


def find_best_btbsizes_per_interval(
    input_csv: str,
    output_csv: str,
    tolerance: int = 100
):
    """
    For each time interval, find the single globally best (btbCore0, btbCore1,
    Prefetch, L2, L3) combination by highest PPW across ALL rows in that interval.
    Outputs one row per interval.

    input_csv may be a single path or a comma-separated list of paths
    (one per prefetcher/cache setting) — concatenated automatically.
    """

    # --- Load one or multiple CSVs ---
    if ',' in input_csv:
        paths = [p.strip() for p in input_csv.split(',')]
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        print(f" Concatenated {len(paths)} CSVs → {len(df)} total rows")
    else:
        df = pd.read_csv(input_csv)

    # --- Validate required columns ---
    required_cols = ["btbCore0", "btbCore1", "period_start", "period_end", "ppw", "Prefetch", "L2", "L3"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # --- Parse "periodicins-12345" → 12345 ---
    def parse_ins(val):
        if isinstance(val, str):
            m = re.search(r"periodicins-(\d+)", val)
            return int(m.group(1)) if m else None
        return None

    df["start_num"] = df["period_start"].apply(parse_ins)
    df["end_num"]   = df["period_end"].apply(parse_ins)

    # --- Drop rows with unparseable interval or missing PPW ---
    df = df.dropna(subset=["start_num", "end_num", "ppw"])
    df = df.sort_values(by=["start_num", "end_num"]).reset_index(drop=True)

    # --- Group rows into intervals (within tolerance) ---
    # Assign a canonical (start, end) label to each row
    canonical_intervals = []
    interval_label_start = []
    interval_label_end   = []

    for _, row in df.iterrows():
        start, end = row["start_num"], row["end_num"]
        matched = None
        for cs, ce in canonical_intervals:
            if abs(cs - start) <= tolerance and abs(ce - end) <= tolerance:
                matched = (cs, ce)
                break
        if matched is None:
            matched = (start, end)
            canonical_intervals.append(matched)
        interval_label_start.append(matched[0])
        interval_label_end.append(matched[1])

    df["interval_start"] = interval_label_start
    df["interval_end"]   = interval_label_end

    # --- For each interval pick the single row with globally highest PPW ---
    best_idx = df.groupby(["interval_start", "interval_end"])["ppw"].idxmax()
    best_df  = df.loc[best_idx].reset_index(drop=True)

    # --- Build output ---
    config_str = (
        best_df["btbCore0"].astype(str) + "_" +
        best_df["btbCore1"].astype(str) + "_" +
        best_df["Prefetch"].astype(str) + "_" +
        best_df["L2"].astype(str)       + "_" +
        best_df["L3"].astype(str)
    )

    result_df = pd.DataFrame({
        "interval_start": best_df["interval_start"],
        "interval_end":   best_df["interval_end"],
        "btbCore0":       best_df["btbCore0"],
        "btbCore1":       best_df["btbCore1"],
        "prefetcher":     best_df["Prefetch"],
        "l2_size":        best_df["L2"],
        "l3_size":        best_df["L3"],
        "config":         config_str,
        "best-config":    config_str,
        "PPW":            best_df["ppw"],
    })

    result_df = result_df.sort_values(["interval_start", "interval_end"]).reset_index(drop=True)
    result_df.to_csv(output_csv, index=False)

    print(f" Best configurations saved to: {output_csv}")
    print(f" Intervals processed: {len(result_df)}")
    print(f" Unique winning configs: {result_df['config'].nunique()}")
    print(f" Unique L2 sizes seen:  {result_df['l2_size'].nunique()} → {sorted(result_df['l2_size'].unique())}")
    print(f" Unique L3 sizes seen:  {result_df['l3_size'].nunique()} → {sorted(result_df['l3_size'].unique())}")
    print(f" Unique prefetchers:    {result_df['prefetcher'].nunique()} → {sorted(result_df['prefetcher'].unique())}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python findBestConfigUsingPPW.py <input_csv_or_comma_list> <output_csv>")
        sys.exit(1)
    find_best_btbsizes_per_interval(sys.argv[1], sys.argv[2], tolerance=100)
