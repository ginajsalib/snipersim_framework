import pandas as pd
import re
import sys


def find_best_btbsizes_per_interval(
    input_csv: str,
    output_csv: str,
    tolerance: int = 100
):
    """
    For each time interval (grouped within tolerance), find the best PPW row
    for every unique (btbCore0, btbCore1, Prefetch, L2, L3) combination.
    Outputs one row per (interval, config_combo).
    """
    # --- Load CSV (accepts a single path or a comma-separated list of paths) ---
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

    # --- Group rows by interval (within tolerance) ---
    interval_labels    = {}
    canonical_intervals = []

    for idx, row in df.iterrows():
        start, end = row["start_num"], row["end_num"]
        matched_interval = None
        for cs, ce in canonical_intervals:
            if abs(cs - start) <= tolerance and abs(ce - end) <= tolerance:
                matched_interval = (cs, ce)
                break
        if matched_interval is None:
            matched_interval = (start, end)
            canonical_intervals.append(matched_interval)
        interval_labels[idx] = matched_interval

    df["interval_start"] = [interval_labels[i][0] for i in df.index]
    df["interval_end"]   = [interval_labels[i][1] for i in df.index]

    # --- For each (interval, btbCore0, btbCore1, Prefetch, L2, L3) pick max PPW row ---
    combo_cols = ["interval_start", "interval_end", "btbCore0", "btbCore1", "Prefetch", "L2", "L3"]
    best_idx = df.groupby(combo_cols)["ppw"].idxmax()
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
        "best-config":    config_str,   # alias used by createTrainingDataWithLabels.py
        "PPW":            best_df["ppw"],
    })

    result_df = result_df.sort_values(["interval_start", "interval_end"]).reset_index(drop=True)
    result_df.to_csv(output_csv, index=False)

    print(f" Best configurations saved to: {output_csv}")
    print(f" Intervals processed : {result_df[['interval_start','interval_end']].drop_duplicates().shape[0]}")
    print(f" Config combinations : {result_df.shape[0]} rows total")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python findBestConfigUsingPPW.py <input_csv_or_comma_list> <output_csv>")
        sys.exit(1)
    input_csv  = sys.argv[1]
    output_csv = sys.argv[2]
    find_best_btbsizes_per_interval(input_csv, output_csv, tolerance=100)
