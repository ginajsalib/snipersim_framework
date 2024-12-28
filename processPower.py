import os
import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np

# Function to run mcpat.py in each subdirectory and capture the output
def run_mcpat(directory):
    result = subprocess.Popen(["python", "/root/sniper/tools/mcpat.py", "-d", directory], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()
     # Check if there was an error during execution
    if result.returncode != 0:
        print("Error executing mcpat")
        return None  # Handle error as needed (e.g., return None or raise an exception)

    # Return the stdout output as a decoded string
    return stdout.decode('utf-8')

# Function to parse the output and extract power values
def parse_power_data(output):
    power_data = {}
    power_regex = re.compile(r"(\S+)\s+(\d+\.\d+)\s+W")
    
    for line in output.splitlines():
        match = power_regex.search(line)
        if match:
            component = match.group(1)
            power = float(match.group(2))
            power_data[component] = power

    return power_data

# Function to traverse directories and process power data
def traverse_and_process(base_directory):
    power_data_dict = {}
    
    # Traverse all subdirectories
    for root, dirs, files in os.walk(base_directory):
        for dir in dirs:
            if "config" not in dir:
                continue
            dir_path = os.path.join(root, dir)
            print("Processing {}...".format(dir_path))
            output = run_mcpat(dir_path)
            power_data = parse_power_data(output)
            power_data_dict[dir_path] = power_data

    return power_data_dict

# Function to plot the power consumption data
def plot_power_consumption(power_data_dict):
    # Prepare data for plotting
    components = set()
    for power_data in power_data_dict.values():
        components.update(power_data.keys())

    # Sort components for plotting consistency
    components = sorted(components)
    
    # Prepare the plot
    plt.figure(figsize=(10, 6))

    # Iterate over directories and plot power data for each component
    for dir_path, power_data in power_data_dict.items():
        x = [dir_path]  # Directory path on the x-axis
        y = [power_data.get(component, 0) for component in components]  # Power for each component
        
        plt.plot(components, y, marker='o', label=dir_path)

    # Label the plot
    plt.xlabel('Components')
    plt.ylabel('Power (W)')
    plt.title('Power Consumption by Component')
    plt.xticks(rotation=90)
    plt.legend(loc='best')
    
    # Show the plot
    plt.tight_layout()
    save_path = "/root/snipersim_framework"
    plt.savefig(save_path)
    print("Plot saved to {}".format(save_path))
    plt.show()

# Main function
def main():
    base_directory = "/root/snipersim_framework/"  # Change this to your target directory
    power_data_dict = traverse_and_process(base_directory)
    plot_power_consumption(power_data_dict)

if __name__ == "__main__":
    main()

