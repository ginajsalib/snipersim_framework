import pandas as pd
import re

def normalize_period_value(val):
    """Convert period labels (like 'periodicins-12345' or 'roi-begin') into integers."""
    if pd.isna(val):
        return -1
    if isinstance(val, str):
        val = val.strip()
        if val == "roi-begin":
            return 0
        if val.startswith("periodicins-"):
            return int(val.replace("periodicins-", ""))
        if re.match(r"^\d+$", val):
            return int(val)
    try:
        return int(val)
    except Exception:
        return -1


def merge_best_config_with_training_data(
    training_csv: str,
    best_config_csv: str,
    output_csv: str,
    tolerance: int = 100
):
    """Merge best configuration info into lagged training data."""
    
    # --- Load CSVs ---
    training_df = pd.read_csv(training_csv)
    best_df = pd.read_csv(best_config_csv)

    # --- Validate required columns ---
    required_training_cols = ["period_start", "period_end"]
    required_best_cols = ["interval_start", "interval_end"]
    
    for col in required_training_cols:
        if col not in training_df.columns:
            raise ValueError(f"Missing required column in training data: {col}")
    for col in required_best_cols:
        if col not in best_df.columns:
            raise ValueError(f"Missing required column in best config data: {col}")

    # --- Detect best config column ---
    # Support either "best-config" or "best_config"
    if "best-config" in best_df.columns:
        best_col_name = "best-config"
    elif "best_config" in best_df.columns:
        best_col_name = "best_config"
    else:
        # fallback to any config-like column if no explicit one
        candidates = [c for c in best_df.columns if "config" in c.lower()]
        if not candidates:
            raise ValueError("No best-config column found in best config CSV.")
        best_col_name = candidates[0]

    # --- Normalize numeric period values ---
    best_df["interval_start_num"] = best_df["interval_start"].apply(normalize_period_value)
    best_df["interval_end_num"] = best_df["interval_end"].apply(normalize_period_value)
    training_df["period_start_num"] = training_df["period_start"].apply(normalize_period_value)
    training_df["period_end_num"] = training_df["period_end"].apply(normalize_period_value)

    # --- Prepare for merge ---
    merged_rows = []
    matched, unmatched = 0, 0

    # Convert best configs into dict for quick fuzzy lookup
    best_records = best_df.to_dict(orient="records")

    for _, row in training_df.iterrows():
        t_start = row["period_start_num"]
        t_end = row["period_end_num"]
        best_config_val = None

        # Find matching best config (within tolerance)
        for config in best_records:
            start_diff = abs(config["interval_start_num"] - t_start)
            end_diff = abs(config["interval_end_num"] - t_end)
            if start_diff <= tolerance and end_diff <= tolerance:
                best_config_val = config[best_col_name]
                matched += 1
                break

        if best_config_val is None:
            best_config_val = ""
            unmatched += 1

        new_row = row.to_dict()
        new_row["best_config"] = best_config_val
        merged_rows.append(new_row)

    merged_df = pd.DataFrame(merged_rows)

    # --- Save result ---
    merged_df.to_csv(output_csv, index=False)

    print("✅ Merge complete!")
    print(f"• {matched} rows matched with best config")
    print(f"• {unmatched} rows without best config")
    print(f"📄 Output saved to: {output_csv}")


# Example usage
if __name__ == "__main__":
    merge_best_config_with_training_data(
        training_csv="Training_Data_Lagged.csv",
        best_config_csv="BestConfigs.csv",
        output_csv="Training_Data_Complete.csv",
        tolerance=100
    )
