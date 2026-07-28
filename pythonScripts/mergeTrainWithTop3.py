import pandas as pd
import os
import numpy as np

def normalize_period_value(val):
    """Normalize 'period_start' to integer values similar to Apps Script."""
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

def merge_train_with_top3(train_csv="Training_Data_Complete.csv", top3_csv="Top3ConfigsPPW.csv", output_csv="TrainWithTop3.csv"):
    """Merge training data with Top3ConfigsPPW results based on fuzzy matching of period_start."""
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training data file not found: {train_csv}")
    if not os.path.exists(top3_csv):
        raise FileNotFoundError(f"Top3 configs file not found: {top3_csv}")

    # Load CSVs
    train_df = pd.read_csv(train_csv)
    top3_df = pd.read_csv(top3_csv)

    if "period_start" not in train_df.columns:
        raise ValueError("Missing 'period_start' column in training data.")
    if "period_start" not in top3_df.columns:
        raise ValueError("Missing 'period_start' column in Top3ConfigsPPW data.")

    # Normalize periods
    train_df["norm_period"] = train_df["period_start"].apply(normalize_period_value)
    top3_df["norm_period"] = top3_df["period_start"].apply(normalize_period_value)

    # Prepare output columns - now includes L2, L3, and per-core prefetch
    # alongside BTB, matching the expanded find_top3_configs_by_ppw output.
    top3_cols = [
        "L2core0_best", "L2core1_best", "L3_best", "Prefetchcore0_best", "Prefetchcore1_best",
        "BTBcore0_best", "BTBcore1_best", "PPW_best",
        "L2core0_2nd", "L2core1_2nd", "L3_2nd", "Prefetchcore0_2nd", "Prefetchcore1_2nd",
        "BTBcore0_2nd", "BTBcore1_2nd", "PPW_2nd", "Diff_best_2nd",
        "L2core0_3rd", "L2core1_3rd", "L3_3rd", "Prefetchcore0_3rd", "Prefetchcore1_3rd",
        "BTBcore0_3rd", "BTBcore1_3rd", "PPW_3rd", "Diff_best_3rd"
    ]

    # Verify all expected columns actually exist in top3_df before proceeding -
    # fail loudly here rather than silently writing NaN for every row if the
    # top3 CSV came from an older/different version of the script.
    missing_top3_cols = [c for c in top3_cols if c not in top3_df.columns]
    if missing_top3_cols:
        raise ValueError(
            f"Top3 configs file is missing expected column(s): {missing_top3_cols}. "
            f"Available columns: {list(top3_df.columns)}. "
            f"Was this file generated with the updated find_top3_configs_by_ppw script?"
        )

    # Create empty columns in train_df
    for col in top3_cols:
        train_df[col] = np.nan

    # Perform fuzzy matching
    unmatched = []
    for idx, row in train_df.iterrows():
        period = row["norm_period"]
        if np.isnan(period):
            unmatched.append(period)
            continue
        # Compute absolute differences
        top3_df["diff"] = abs(top3_df["norm_period"] - period)
        # Dynamic tolerance (like in Apps Script)
        tolerance = max(1000, abs(period) * 0.1)
        # Find best match within tolerance
        match_row = top3_df[top3_df["diff"] <= tolerance].sort_values("diff").head(1)
        if not match_row.empty:
            # Assign matched values
            for col in top3_cols:
                train_df.at[idx, col] = match_row.iloc[0][col]
        else:
            unmatched.append(period)

    # Drop helper columns
    train_df.drop(columns=["norm_period"], inplace=True, errors="ignore")

    # Save merged result
    train_df.to_csv(output_csv, index=False)
    print(f"Merge completed! Saved to {output_csv}")
    print(f"Matched rows: {len(train_df) - len(unmatched)}")
    print(f"Unmatched rows: {len(unmatched)}")
    if unmatched:
        print(f"No Top3 match found for periods (showing up to 10): {unmatched[:10]}")

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        train_csv = sys.argv[1]
        top3_csv = sys.argv[2]
        output_csv = sys.argv[3]
        merge_train_with_top3(train_csv, top3_csv, output_csv)
    else:
        print("Usage: python script.py <train_file.csv> <top3_file.csv> <output_file.csv>")        raise ValueError("Missing 'period_start' column in training data.")
    if "period_start" not in top3_df.columns:
        raise ValueError("Missing 'period_start' column in Top3ConfigsPPW data.")

    # Normalize periods
    train_df["norm_period"] = train_df["period_start"].apply(normalize_period_value)
    top3_df["norm_period"] = top3_df["period_start"].apply(normalize_period_value)

    # Prepare output columns
    top3_cols = [
        "btbCore0_best", "btbCore1_best", "PPW_best",
        "btbCore0_2nd", "btbCore1_2nd", "PPW_2nd", "Diff_best_2nd",
        "btbCore0_3rd", "btbCore1_3rd", "PPW_3rd", "Diff_best_3rd"
    ]

    # Create empty columns in train_df
    for col in top3_cols:
        train_df[col] = np.nan

    # Perform fuzzy matching
    unmatched = []
    for idx, row in train_df.iterrows():
        period = row["norm_period"]
        if np.isnan(period):
            unmatched.append(period)
            continue

        # Compute absolute differences
        top3_df["diff"] = abs(top3_df["norm_period"] - period)

        # Dynamic tolerance (like in Apps Script)
        tolerance = max(1000, abs(period) * 0.1)

        # Find best match within tolerance
        match_row = top3_df[top3_df["diff"] <= tolerance].sort_values("diff").head(1)

        if not match_row.empty:
            # Assign matched values
            for col in top3_cols:
                train_df.at[idx, col] = match_row.iloc[0][col]
        else:
            unmatched.append(period)

    # Drop helper columns
    train_df.drop(columns=["norm_period"], inplace=True, errors="ignore")

    # Save merged result
    train_df.to_csv(output_csv, index=False)
    print(f" Merge completed! Saved to {output_csv}")
    print(f"Matched rows: {len(train_df) - len(unmatched)}")
    print(f"Unmatched rows: {len(unmatched)}")

    if unmatched:
        print(f" No Top3 match found for periods (showing up to 10): {unmatched[:10]}")


# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        train_csv = sys.argv[1] 
        top3_csv = sys.argv[2]
        output_csv = sys.argv[3]
    merge_train_with_top3(train_csv, top3_csv, output_csv)
