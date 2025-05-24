import os
import subprocess
from itertools import product

# Define the parameter options
#cache_sizes_l2 = [256, 512, 1024]
cache_sizes_l2=[256]
cache_sizes_l3MB=[8192]
prefetchers=["none"]
#cache_sizes_l3MB = [4096, 8192, 12288, 16384]
#prefetchers = ["none", "simple", "ghb"]
branch_predictor_sizes = [512, 1024, 2048, 4096]
benchmark = 'fft'
# Function to run the command with specified options
def run_sniper_command(directory, cache_size_l2, cache_size_l3MB, prefetcher, branch_predictor):
    os.chdir("/root/benchmarks")
    benchmark_name = 'splash2-' + benchmark + '-test-4'
    command = [
        '/root/benchmarks/run-sniper',
        '--benchmarks', benchmark_name,
        '-n',  '2', '-c', 'gainestown', '-c', 'rob',  '-c', 'big,LITTLE',
        '-d', directory,
        '-g'
    ]
    
    # Add the perf_model configurations using .format() for compatibility with Python 3.4
    command.append(' perf_model/l2_cache/cache_size={}'.format(cache_size_l2))
    command.append(' -g perf_model/l3_cache/cache_size={}'.format(cache_size_l3MB))
    command.append(' -g perf_model/l2_cache/prefetcher={}'.format(prefetcher))
    command.append(' -g perf_model/branch_predictor/num_entries={}'.format(branch_predictor))
    command.append(' -g perf_model/branch_predictor/num_ways=4 ')
    
    # Execute the command
    print("Running command: {}".format(" ".join(command)))
    subprocess.call(command, env=os.environ.copy())

# Main function to generate configurations and run commands
def main():
    # Create all combinations of the parameters
    configurations = product(cache_sizes_l2, cache_sizes_l3MB, prefetchers, branch_predictor_sizes)
    
    # Iterate through each combination and run the command
    for cache_size_l2, cache_size_l3MB, prefetcher, branch_predictor in configurations:
        # Create a directory name based on the parameters
        directory = '/root/snipersim_framework/config_l2_{}_l3MB_{}_prefetch_{}_branch_{}_{}'.format(
            cache_size_l2, cache_size_l3MB, prefetcher, branch_predictor, benchmark
        )
        
        # Ensure the directory exists (this is for logging, data collection, etc.)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Run the sniper command with the given configuration
        run_sniper_command(directory, cache_size_l2, cache_size_l3MB, prefetcher, branch_predictor)

# Run the script
if __name__ == "__main__":
    main()

