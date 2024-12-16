import os
import subprocess
from itertools import product

# Define the parameter options
cache_sizes_l2 = [256, 512, 1024]
cache_sizes_l2MB = [4, 8, 12, 16]
prefetchers = ["none", "simple", "ghb"]
branch_predictor_sizes = [512, 1024, 2048, 4096]

# Function to run the command with specified options
def run_sniper_command(directory, cache_size_l2, cache_size_l2MB, prefetcher, branch_predictor):
    command = [
        './run-sniper',
        '--benchmarks', 'splash2-barnes-test-4',
        '-n', '4',
        '-d', directory,
        '-g', 'option'
    ]
    
    # Add the perf_model configurations
    command.append(f'perf_model/l2_cache/cache_size={cache_size_l2}')
    command.append(f'perf_model/l2_cache/cache_size={cache_size_l2MB}MB')
    command.append(f'perf_model/l2_cache/prefetcher={prefetcher}')
    command.append(f'perf_model/branch_predictor/size={branch_predictor}')
    
    # Execute the command
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command)

# Main function to generate configurations and run commands
def main():
    # Create all combinations of the parameters
    configurations = product(cache_sizes_l2, cache_sizes_l2MB, prefetchers, branch_predictor_sizes)
    
    # Iterate through each combination and run the command
    for cache_size_l2, cache_size_l2MB, prefetcher, branch_predictor in configurations:
        # Create a directory name based on the parameters
        directory = f"config_l2_{cache_size_l2}_l2MB_{cache_size_l2MB}_prefetch_{prefetcher}_branch_{branch_predictor}"
        
        # Ensure the directory exists (this is for logging, data collection, etc.)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Run the sniper command with the given configuration
        run_sniper_command(directory, cache_size_l2, cache_size_l2MB, prefetcher, branch_predictor)

# Run the script
if __name__ == "__main__":
    main()
