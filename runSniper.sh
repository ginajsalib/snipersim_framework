#!/bin/bash

# Set working directory
cd /root/benchmarks || exit 1

# Configuration parameters
cache_sizes_l2=(256)
cache_sizes_l3MB=(8192)
prefetchers=("none")
branch_predictor_sizes=(512 1024 2048 4096)
benchmark="fft"

# Loop through all combinations
for l2 in "${cache_sizes_l2[@]}"; do
  for l3 in "${cache_sizes_l3MB[@]}"; do
    for prefetch in "${prefetchers[@]}"; do
      for bp in "${branch_predictor_sizes[@]}"; do

        # Construct the output directory
        directory="/root/snipersim_framework/config_l2_${l2}_l3MB_${l3}_prefetch_${prefetch}_branch_${bp}_${benchmark}"
        
        # Create the directory if it doesn't exist
        mkdir -p "$directory"

        # Construct the benchmark name
        benchmark_name="splash2-${benchmark}-test-4"

        # Build the command
        cmd=(
          "/root/benchmarks/run-sniper"
          "--benchmarks" "$benchmark_name"
          "-n" "2"
          "-c" "gainestown"
          "-c" "rob"
          "-c" "big,LITTLE"
          "-d" "$directory"
          "-g" "perf_model/l2_cache/cache_size=${l2}"
          "-g" "perf_model/l3_cache/cache_size=${l3}"
          "-g" "perf_model/l2_cache/prefetcher=${prefetch}"
          "-g" "perf_model/branch_predictor/num_entries=${bp}"
          "-g" "perf_model/branch_predictor/num_ways=4"
        )

        # Print and run the command
        echo "Running command: ${cmd[*]}"
        "${cmd[@]}"
      done
    done
  done
done

