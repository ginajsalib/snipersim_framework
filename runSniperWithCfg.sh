#!/bin/bash

# Set working directory
cd /root/benchmarks || exit 1

# Configuration parameters
cache_sizes_l2=(256)
cache_sizes_l3MB=(8192)
prefetchers=("none")
branch_predictor_sizes=(512 1024 2048 4096)
benchmark="fft"

NUM_CORES=2
DISPATCH_WIDTH=4
CFG_DIR="/root/snipersim_framework"

# Loop through all combinations
for l2 in "${cache_sizes_l2[@]}"; do
  for l3 in "${cache_sizes_l3MB[@]}"; do
    for prefetch in "${prefetchers[@]}"; do
      for bp0 in "${branch_predictor_sizes[@]}"; do
        for bp1 in "${branch_predictor_sizes[@]}"; do

          # Create config files for core0 and core1
          cat <<EOF > "$CFG_DIR/core0.cfg"
[perf_model/core/interval_timer]
dispatch_width = ${DISPATCH_WIDTH}

[perf_model/core0/branch_predictor]
num_entries = ${bp0}

[perf_model/core1/branch_predictor]
num_entries = ${bp1}

EOF

          cat <<EOF > "$CFG_DIR/core1.cfg"
[perf_model/core/interval_timer]
dispatch_width = ${DISPATCH_WIDTH}

[perf_model/core0/branch_predictor]
num_entries = ${bp0}

[perf_model/core1/branch_predictor]
num_entries = ${bp1}
EOF

          # Construct the output directory name
          directory="/root/snipersim_framework/config_l2_${l2}_l3MB_${l3}_prefetch_${prefetch}_branch_${bp0}-${bp1}_${benchmark}"
          mkdir -p "$directory"

          # Build benchmark name
          benchmark_name="splash2-${benchmark}-test-4"

          # Build and run the sniper command
          cmd=(
            "/root/benchmarks/run-sniper"
            "-c" "gainestown"
            "-c" "rob"
            "-c" "core0,core1"
            "--benchmarks" "$benchmark_name"
            "-n" "$NUM_CORES"
            "-d" "$directory"
            "-g" "perf_model/l2_cache/cache_size=${l2}"
            "-g" "perf_model/l3_cache/cache_size=${l3}"
            "-g" "perf_model/l2_cache/prefetcher=${prefetch}"
            "-g" "perf_model/core0/branch_predictor/num_ways=4"
            "-g" "perf_model/core1/branch_predictor/num_ways=4"
          )

          echo "Running: ${cmd[*]}"
          "${cmd[@]}"

        done
      done
    done
  done
done

