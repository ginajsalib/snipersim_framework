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
    Output is written to 'Top3ConfigsPPW.csv'.
    """

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"File not found: {input_csv}")

    # Read input data
    df = pd.read_csv(input_csv)

    required_cols = ["period_start", "ppw", "btbCore0", "btbCore1"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Normalize period_start
    df["norm_period"] = df["period_start"].apply(normalize_period_value)
    df["norm_period"] = (df["norm_period"] // 100) * 100  # group into 100-intervals

    # Group by normalized period
    results = []
    grouped = df.groupby("norm_period")

    for norm_period, group in grouped:
        # Drop rows with NaN PPW
        group = group.dropna(subset=["ppw"])
        if group.empty:
            continue



        # Sort by PPW descending
        group_sorted = group.sort_values(by="ppw", ascending=False).reset_index(drop=True)

        # Get top 3 configs (fill with best if fewer)
        best = group_sorted.iloc[0]
        second = group_sorted.iloc[1] if len(group_sorted) > 1 else best
        third = group_sorted.iloc[2] if len(group_sorted) > 2 else second

        diff_best_second = second["ppw"] - best["ppw"]
        diff_best_third = third["ppw"] - best["ppw"]

        results.append({
            "period_start": norm_period,
            "btbCore0_best": best["btbCore0"],
            "btbCore1_best": best["btbCore1"],
            "PPW_best": best["ppw"],
            "btbCore0_2nd": second["btbCore0"],
            "btbCore1_2nd": second["btbCore1"],
            "PPW_2nd": second["ppw"],
            "Diff_best_2nd": diff_best_second,
            "btbCore0_3rd": third["btbCore0"],
            "btbCore1_3rd": third["btbCore1"],
            "PPW_3rd": third["ppw"],
            "Diff_best_3rd": diff_best_third
        })

    # Convert results to DataFrame and save
    result_df = pd.DataFrame(results, columns=[
        "period_start",
        "btbCore0_best", "btbCore1_best", "PPW_best",
        "btbCore0_2nd", "btbCore1_2nd", "PPW_2nd", "Diff_best_2nd",
        "btbCore0_3rd", "btbCore1_3rd", "PPW_3rd", "Diff_best_3rd"
    ])

    result_df.to_csv(output_csv, index=False)
    print(f"Top 3 configs per period saved to: {output_csv}")
    print(f"Total periods processed: {len(result_df)}")


# Example usage
if __name__ == "__main__":

    if len(sys.argv) >= 2:
        input_csv = sys.argv[1] 
        output_csv = sys.argv[2]
    find_top3_configs_by_ppw(input_csv, output_csv)
