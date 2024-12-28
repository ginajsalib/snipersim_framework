import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Function to parse the 'sim.out' file and extract the required stats
def parse_sim_out(file_path):
    branch_mispredictions = []
    l2_cache_misses = []
    l3_cache_misses = []

    with open(file_path, 'r') as file:
        content = file.read()

        # Extract branch mispredictions
        branch_mispred_pattern = re.compile(r'num incorrect\s+\|\s+([\d,]+)')
        branch_mispredictions = [int(match.replace(',', '')) for match in branch_mispred_pattern.findall(content)]

        # Extract L2 cache misses
        l2_cache_misses_pattern = re.compile(r'Cache L2\s+[\s\w]+num cache misses\s+\|[\s\w]+([\d,]+)')
        l2_cache_misses = [int(match.replace(',', '')) for match in l2_cache_misses_pattern.findall(content)]

        # Extract L3 cache misses
        l3_cache_misses_pattern = re.compile(r'Cache L3\s+[\s\w]+num cache misses\s+\|[\s\w]+([\d,]+)')
        l3_cache_misses = [int(match.replace(',', '')) for match in l3_cache_misses_pattern.findall(content)]

    return branch_mispredictions, l2_cache_misses, l3_cache_misses

# Function to traverse all subdirectories and find 'sim.out' files
def traverse_and_parse(base_directory):
    all_data = {}
    
    # Traverse all subdirectories
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file == 'sim.out':
                file_path = os.path.join(root, file)
                print("Processing {}...".format(file_path))

                # Parse the sim.out file
                branch_mispredictions, l2_cache_misses, l3_cache_misses = parse_sim_out(file_path)

                # Store data with the directory as the key
                all_data[root] = {
                    'branch_mispredictions': branch_mispredictions,
                    'l2_cache_misses': l2_cache_misses,
                    'l3_cache_misses': l3_cache_misses
                }

    return all_data

# Function to plot the branch mispredictions and cache misses data
def plot_data(all_data):
    # Create subplots: one for branch mispredictions, one for cache misses (L2 and L3)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Prepare data for plotting branch mispredictions
    for dir_path, data in all_data.items():
        branch_mispredictions = data['branch_mispredictions']
        if branch_mispredictions:
            axs[0].plot(range(len(branch_mispredictions)), branch_mispredictions, label=dir_path)
    
    axs[0].set_title('Branch Mispredictions')
    axs[0].set_xlabel('Index (Different Timesteps or Trials)')
    axs[0].set_ylabel('Branch Mispredictions')
    axs[0].legend()

    # Prepare data for plotting L2 and L3 cache misses
    for dir_path, data in all_data.items():
        l2_cache_misses = data['l2_cache_misses']
        l3_cache_misses = data['l3_cache_misses']

        if l2_cache_misses and l3_cache_misses:
            axs[1].plot(range(len(l2_cache_misses)), l2_cache_misses, label=f'{dir_path} L2 Cache Misses', linestyle='--')
            axs[1].plot(range(len(l3_cache_misses)), l3_cache_misses, label=f'{dir_path} L3 Cache Misses', linestyle=':')

    axs[1].set_title('L2 and L3 Cache Misses')
    axs[1].set_xlabel('Index (Different Timesteps or Trials)')
    axs[1].set_ylabel('Cache Misses')
    axs[1].legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

# Main function to run the script
def main():
    base_directory = '/root/snipersim_framework/'  # Change to the directory you want to traverse
    all_data = traverse_and_parse(base_directory)
    plot_data(all_data)

if __name__ == "__main__":
    main()

