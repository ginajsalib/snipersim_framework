import subprocess
import os
import sys
import argparse
from datetime import datetime


def run_script(script_name, *args):
    """Run a Python script with optional arguments and handle errors."""
    print(f"\n▶ Running {script_name} {' '.join(args)} ...")
    result = subprocess.run([sys.executable, script_name, *args])
    if result.returncode != 0:
        print(f" Error in {script_name}: Exit code {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f" {script_name} completed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Run the full SniperSim data pipeline from perf & power CSVs to training datasets."
    )
    parser.add_argument(
        "--perf_csv",
        required=True,
        nargs="+",
        help="One or more performance input CSV files (one per prefetcher/cache setting)."
    )
    parser.add_argument(
        "--power_csv",
        required=True,
        nargs="+",
        help="One or more power input CSV files (must match --perf_csv count and order)."
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Name of the benchmark (used for output file naming)."
    )
    args = parser.parse_args()

    if len(args.perf_csv) != len(args.power_csv):
        print(f" Error: --perf_csv has {len(args.perf_csv)} file(s) but --power_csv has {len(args.power_csv)}.")
        print("  Each --perf_csv must have a matching --power_csv in the same order.")
        sys.exit(1)

    base_dir  = os.path.dirname(os.path.abspath(__file__))
    n_configs = len(args.perf_csv)

    print(f"\n Pipeline starting with {n_configs} prefetcher/cache setting(s) for benchmark '{args.benchmark}'")

    # -------------------------------------------------------------------------
    # Stage 1: Preprocess each perf+power pair into its own merged_full CSV
    # -------------------------------------------------------------------------
    merged_full_files = []

    for i, (perf_csv, power_csv) in enumerate(zip(args.perf_csv, args.power_csv)):
        suffix = f"{args.benchmark}_cfg{i}"
        parsed_perf       = os.path.join(base_dir, f"parsed_perf_{suffix}.csv")
        cleaned_perf      = os.path.join(base_dir, f"cleaned_perf_{suffix}.csv")
        merged_perf_power = os.path.join(base_dir, f"merged_perf_power_{suffix}.csv")
        merged_full_i     = os.path.join(base_dir, f"merged_full_{suffix}.csv")

        print(f"\n── Config {i+1}/{n_configs}: {os.path.basename(perf_csv)}")
        run_script("parseInputConfigsOnPerf.py",          perf_csv,         parsed_perf)
        run_script("deleteEmptyRows.py",                   parsed_perf,      cleaned_perf)
        run_script("mergePerfAndPower.py",                 cleaned_perf,     power_csv, merged_perf_power)
        run_script("addCalculatedColumnsToMergedCsv.py",   merged_perf_power, merged_full_i)

        merged_full_files.append(merged_full_i)

    # Comma-separated list passed to scripts that accept multi-CSV input
    merged_full_arg = ",".join(merged_full_files)

    # Also produce a single concatenated merged_full for createTrainingDataLagged
    # (which does not need multi-CSV support — it just sees all rows)
    merged_full_combined = os.path.join(base_dir, f"merged_full_{args.benchmark}.csv")
    if n_configs == 1:
        merged_full_combined = merged_full_files[0]
    else:
        import pandas as pd
        print(f"\n── Concatenating {n_configs} merged_full CSVs → {os.path.basename(merged_full_combined)}")
        pd.concat(
            [pd.read_csv(f) for f in merged_full_files],
            ignore_index=True
        ).to_csv(merged_full_combined, index=False)
        print(f" Combined merged_full saved.")

    # -------------------------------------------------------------------------
    # Stage 2: Best config & training data
    # -------------------------------------------------------------------------
    best_configs           = os.path.join(base_dir, f"best_configs_{args.benchmark}.csv")
    training_data_lagged   = os.path.join(base_dir, f"training_data_lagged_{args.benchmark}.csv")
    training_data_complete = os.path.join(base_dir, f"training_data_complete_{args.benchmark}.csv")

    # findBestConfigUsingPPW accepts comma-separated list → groups by full combo
    run_script("findBestConfigUsingPPW.py",        merged_full_arg,      best_configs)
    run_script("createTrainingDataLagged.py",       merged_full_combined, training_data_lagged)
    run_script("createTrainingDataWithLabels.py",   training_data_lagged, best_configs, training_data_complete)

    # -------------------------------------------------------------------------
    # Stage 3: Top 3 configs path
    # -------------------------------------------------------------------------
    top3_configs   = os.path.join(base_dir, f"top3_configs_{args.benchmark}.csv")
    train_with_top3 = os.path.join(base_dir, f"train_with_top3_{args.benchmark}.csv")

    # findTop3ConfigsByPPW also accepts comma-separated list
    run_script("findTop3ConfigsByPPW.py",  merged_full_arg,      top3_configs)
    run_script("mergeTrainWithTop3.py",    training_data_lagged, top3_configs, train_with_top3)

    print("\n Pipeline finished successfully!")
    print(" Final outputs:")
    print(f"    {training_data_complete}   Training data with BEST config")
    print(f"    {train_with_top3}          Training data with TOP 3 configs")


if __name__ == "__main__":
    main()
