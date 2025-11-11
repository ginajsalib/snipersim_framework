import pandas as pd

def transform_data_with_lag(input_csv: str, output_csv: str):
    # Load CSV
    df = pd.read_csv(input_csv)
    
    # --- Validate required columns ---
    required = ["config", "period_start", "period_end"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Identify metric columns (exclude metadata columns)
    exclude_cols = set(required)
    for col in ["directory", "period"]:
        if col in df.columns:
            exclude_cols.add(col)
    
    metric_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Sort by config and period_start to help matching
    df = df.sort_values(by=["config", "period_start"]).reset_index(drop=True)
    
    # Create lookup: (config, period_end) -> row
    lookup = {
        (str(r.config), str(r.period_end)): r
        for _, r in df.iterrows()
    }

    # Build transformed rows
    transformed_rows = []

    for _, row in df.iterrows():
        config = str(row.config)
        current_start = str(row.period_start)
        current_end = str(row.period_end)

        # Find previous row: same config, period_end == current period_start
        prev_row = lookup.get((config, current_start))

        if prev_row is not None:
            new_row = {
                "config": row.config,
                "period_start": row.period_start,
                "period_end": row.period_end,
            }
            # Add all _prev metrics
            for metric in metric_cols:
                new_row[f"{metric}_prev"] = prev_row[metric]
            transformed_rows.append(new_row)
        else:
            # Skip if no previous period exists
            print(f"Info: No previous period found for config {config}, period {current_start}")

    # Create new DataFrame
    if transformed_rows:
        result_df = pd.DataFrame(transformed_rows)
        result_df.to_csv(output_csv, index=False)
        print(f"✅ Transformation complete! Created {len(result_df)} rows.")
        print(f"Saved to: {output_csv}")
    else:
        print("⚠️ No data transformed — no previous periods found.")

# Example usage
if __name__ == "__main__":
    transform_data_with_lag(
        input_csv="MergedFull.csv",
        output_csv="Training_Data_Lagged.csv"
    )
