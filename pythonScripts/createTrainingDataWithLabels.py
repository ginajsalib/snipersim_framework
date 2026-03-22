import pandas as pd
import re
import sys


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
    """
    Merge best configuration info into lagged training data.

    best_config_csv has ONE row per interval (the globally best config
    across all BTB/prefetcher/L2/L3 combinations for that interval).
    Each training row is matched to the best config for its interval.
    """

    training_df = pd.read_csv(training_csv)
    best_df     = pd.read_csv(best_config_csv)

    for col in ["period_start", "period_end"]:
        if col not in training_df.columns:
            raise ValueError(f"Missing required column in training data: {col}")
    for col in ["interval_start", "interval_end"]:
        if col not in best_df.columns:
            raise ValueError(f"Missing required column in best config data: {col}")

    # --- Detect best-config label column ---
    if "best-config" in best_df.columns:
        best_col_name = "best-config"
    elif "best_config" in best_df.columns:
        best_col_name = "best_config"
    else:
        candidates = [c for c in best_df.columns if "config" in c.lower()]
        if not candidates:
            raise ValueError("No best-config column found in best config CSV.")
        best_col_name = candidates[0]

    # --- Normalize period values ---
    best_df["interval_start_num"] = best_df["interval_start"].apply(normalize_period_value)
    best_df["interval_end_num"]   = best_df["interval_end"].apply(normalize_period_value)
    training_df["period_start_num"] = training_df["period_start"].apply(normalize_period_value)
    training_df["period_end_num"]   = training_df["period_end"].apply(normalize_period_value)

    best_records = best_df.to_dict(orient="records")
    merged_rows  = []
    matched, unmatched = 0, 0

    for _, row in training_df.iterrows():
        t_start = row["period_start_num"]
        t_end   = row["period_end_num"]

        best_config_val = None
        best_l2_size    = None
        best_l3_size    = None
        best_prefetcher = None

        # Interval-only match — best_configs has one globally best row per interval
        for cfg in best_records:
            if abs(cfg["interval_start_num"] - t_start) <= tolerance and \
               abs(cfg["interval_end_num"]   - t_end)   <= tolerance:
                best_config_val = cfg[best_col_name]
                best_l2_size    = cfg.get("l2_size", "")
                best_l3_size    = cfg.get("l3_size", "")
                best_prefetcher = cfg.get("prefetcher", "")
                matched += 1
                break

        if best_config_val is None:
            best_config_val = ""
            best_l2_size    = ""
            best_l3_size    = ""
            best_prefetcher = ""
            unmatched += 1

        new_row = row.to_dict()
        new_row["best_config"]     = best_config_val
        new_row["best_l2_size"]    = best_l2_size
        new_row["best_l3_size"]    = best_l3_size
        new_row["best_prefetcher"] = best_prefetcher
        merged_rows.append(new_row)

    merged_df = pd.DataFrame(merged_rows)
    merged_df.to_csv(output_csv, index=False)

    print(" Merge complete!")
    print(f" {matched} rows matched with best config")
    print(f" {unmatched} rows without best config")
    print(f" Output columns added: best_config, best_l2_size, best_l3_size, best_prefetcher")
    print(f" Output saved to: {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python createTrainingDataWithLabels.py <training_csv> <best_config_csv> <output_csv>")
        sys.exit(1)
    merge_best_config_with_training_data(sys.argv[1], sys.argv[2], sys.argv[3], tolerance=100)
