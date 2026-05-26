import pandas as pd
import os
import numpy as np
import sys
 
 
def normalize_period_value(val):
    """Normalize 'period_start' to integer values."""
    if isinstance(val, str):
        val = val.strip()
        if val == "roi-begin":
            return 0
        elif val.startswith("periodicins-"):
            try:
                return int(val.replace("periodicins-", ""))
            except ValueError:
                return -1
        elif val.isdigit():
            return int(val)
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return -1
 
 
def merge_train_with_top3(
    train_csv="Training_Data_Complete.csv",
    top3_csv="Top3ConfigsPPW.csv",
    output_csv="TrainWithTop3.csv"
):
    """Merge training data with Top3ConfigsPPW results based on fuzzy period matching."""
 
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training data file not found: {train_csv}")
    if not os.path.exists(top3_csv):
        raise FileNotFoundError(f"Top3 configs file not found: {top3_csv}")
 
    train_df = pd.read_csv(train_csv)
    top3_df  = pd.read_csv(top3_csv)
 
    if "period_start" not in train_df.columns:
        raise ValueError("Missing 'period_start' column in training data.")
    if "period_start" not in top3_df.columns:
        raise ValueError("Missing 'period_start' column in Top3ConfigsPPW data.")
 
    # --- All top3 columns to carry into training data ---
    top3_cols = [
        "btbCore0_best",   "btbCore1_best",   "prefetcher_best", "l2_size_best", "l3_size_best", "PPW_best",
        "btbCore0_2nd",    "btbCore1_2nd",    "prefetcher_2nd",  "l2_size_2nd",  "l3_size_2nd",  "PPW_2nd",  "Diff_best_2nd",
        "btbCore0_3rd",    "btbCore1_3rd",    "prefetcher_3rd",  "l2_size_3rd",  "l3_size_3rd",  "PPW_3rd",  "Diff_best_3rd",
    ]
    # Only include columns that exist in the top3 CSV
    top3_cols = [c for c in top3_cols if c in top3_df.columns]
 
    for col in top3_cols:
        train_df[col] = np.nan
 
    # --- Normalize period values ---
    train_df["norm_period"] = train_df["period_start"].apply(normalize_period_value)
    top3_df["norm_period"]  = top3_df["period_start"].apply(normalize_period_value)
 
    top3_records = top3_df.to_dict(orient="records")
 
    matched, unmatched = 0, 0
 
    for idx, row in train_df.iterrows():
        period    = row["norm_period"]
        tolerance = max(1000, abs(period) * 0.1)
 
        best_match = None
        best_diff  = float("inf")
 
        for rec in top3_records:
            diff = abs(rec["norm_period"] - period)
            if diff <= tolerance and diff < best_diff:
                best_diff  = diff
                best_match = rec
 
        if best_match is not None:
            for col in top3_cols:
                train_df.at[idx, col] = best_match.get(col, np.nan)
            matched += 1
        else:
            unmatched += 1
 
    train_df.drop(columns=["norm_period"], inplace=True, errors="ignore")
    train_df.to_csv(output_csv, index=False)
 
    print(f" Merge completed! Saved to {output_csv}")
    print(f" Matched rows  : {matched}")
    print(f" Unmatched rows: {unmatched}")
 
 
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python mergeTrainWithTop3.py <train_csv> <top3_csv> <output_csv>")
        sys.exit(1)
    merge_train_with_top3(sys.argv[1], sys.argv[2], sys.argv[3])
 
