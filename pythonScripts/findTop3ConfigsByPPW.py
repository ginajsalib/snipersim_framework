import pandas as pd
import sys
import os


def normalize_period_value(val):
    """Normalize 'period_start' values to integers."""
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
    For each time interval, rank ALL unique (btbCore0, btbCore1, Prefetch, L2, L3)
    combinations by their best PPW and output the top 3.

    input_csv may be a single path or a comma-separated list of paths
    (one per prefetcher/cache setting) — they will be concatenated automatically.
    """

    # --- Load one or multiple CSVs ---
    if ',' in input_csv:
        paths = [p.strip() for p in input_csv.split(',')]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Files not found: {missing}")
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        print(f" Concatenated {len(paths)} CSVs → {len(df)} total rows")
    else:
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"File not found: {input_csv}")
        df = pd.read_csv(input_csv)

    # --- Validate required columns ---
    required_cols = ["period_start", "ppw", "btbCore0", "btbCore1", "Prefetch", "L2", "L3"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # --- Normalize and bucket period_start into 100-instruction intervals ---
    df["norm_period"] = df["period_start"].apply(normalize_period_value)
    df["norm_period"] = (df["norm_period"] // 100) * 100

    # --- For each (interval, full config combo) keep only the best PPW row ---
    combo_cols = ["norm_period", "btbCore0", "btbCore1", "Prefetch", "L2", "L3"]
    df = df.dropna(subset=["ppw"])
    best_per_combo = df.loc[df.groupby(combo_cols)["ppw"].idxmax()].reset_index(drop=True)

    # --- Per interval, rank combos by PPW and pick top 3 ---
    results = []

    for norm_period, group in best_per_combo.groupby("norm_period"):
        group_sorted = group.sort_values("ppw", ascending=False).reset_index(drop=True)

        best   = group_sorted.iloc[0]
        second = group_sorted.iloc[1] if len(group_sorted) > 1 else best
        third  = group_sorted.iloc[2] if len(group_sorted) > 2 else second

        results.append({
            "period_start":      norm_period,
            # --- Best ---
            "btbCore0_best":     best["btbCore0"],
            "btbCore1_best":     best["btbCore1"],
            "prefetcher_best":   best["Prefetch"],
            "l2_size_best":      best["L2"],
            "l3_size_best":      best["L3"],
            "PPW_best":          best["ppw"],
            # --- 2nd ---
            "btbCore0_2nd":      second["btbCore0"],
            "btbCore1_2nd":      second["btbCore1"],
            "prefetcher_2nd":    second["Prefetch"],
            "l2_size_2nd":       second["L2"],
            "l3_size_2nd":       second["L3"],
            "PPW_2nd":           second["ppw"],
            "Diff_best_2nd":     second["ppw"] - best["ppw"],
            # --- 3rd ---
            "btbCore0_3rd":      third["btbCore0"],
            "btbCore1_3rd":      third["btbCore1"],
            "prefetcher_3rd":    third["Prefetch"],
            "l2_size_3rd":       third["L2"],
            "l3_size_3rd":       third["L3"],
            "PPW_3rd":           third["ppw"],
            "Diff_best_3rd":     third["ppw"] - best["ppw"],
        })

    result_df = pd.DataFrame(results, columns=[
        "period_start",
        "btbCore0_best", "btbCore1_best", "prefetcher_best", "l2_size_best", "l3_size_best", "PPW_best",
        "btbCore0_2nd",  "btbCore1_2nd",  "prefetcher_2nd",  "l2_size_2nd",  "l3_size_2nd",  "PPW_2nd",  "Diff_best_2nd",
        "btbCore0_3rd",  "btbCore1_3rd",  "prefetcher_3rd",  "l2_size_3rd",  "l3_size_3rd",  "PPW_3rd",  "Diff_best_3rd",
    ])

    result_df.to_csv(output_csv, index=False)
    print(f" Top 3 configs per period saved to: {output_csv}")
    print(f" Total periods processed: {len(result_df)}")
    print(f" Unique config combos seen: {len(best_per_combo)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python findTop3ConfigsByPPW.py <input_csv_or_comma_list> <output_csv>")
        sys.exit(1)
    input_csv  = sys.argv[1]
    output_csv = sys.argv[2]
    find_top3_configs_by_ppw(input_csv, output_csv)
