import os
import re
import csv
import datetime

benchmark = ''
# Function to parse the 'sim.out' file and extract the required stats
def parse_sim_out(file_path):
    branch_mispredictions = []
    l2_cache_misses = []
    l3_cache_misses = []
    time_ns = []
    cycles = []
    with open(file_path, 'r') as file:
        content = file.read()

        # Extract branch mispredictions (num incorrect)
        branch_mispred_pattern = re.compile(r'num incorrect\s*\|\s*(\d[\d,]*)\s*\|\s*(\d[\d,]*)')
        matches = branch_mispred_pattern.findall(content)
        if matches:
            branch_mispredictions = [int(match.replace(',', '').strip()) for match in matches[0]]
        print("Branch mispredictions found:", branch_mispredictions)  # Debugging print statement

        # Extract L2 cache misses (num cache misses under Cache L2)
#        print(content)
        l2_cache_misses_pattern = re.compile(r'Cache L2\s+\|\s*\|\s*num cache accesses\s*\|\s*\d+\s*\|\s*\d+\s*num cache misses\s*\|\s*(\d+)\s*\|\s*(\d+)')

        matches_l2 = l2_cache_misses_pattern.findall(content)
        if matches_l2:
            print(matches_l2)
            l2_cache_misses = [int(match.replace(',', '').strip()) for match in matches_l2[0]]
        print("L2 Cache misses found:", l2_cache_misses)  # Debugging print statement

        # Extract L3 cache misses (num cache misses under Cache L3)
        l3_cache_misses_pattern = re.compile(r'Cache L3\s+\|\s*\|\s*num cache accesses\s*\|\s*\d+\s*\|\s*\d+\s*num cache misses\s*\|\s*(\d+)\s*\|\s*(\d+)')
        matches_l3 = l3_cache_misses_pattern.findall(content)
        if matches_l3:
            l3_cache_misses = [int(match.replace(',', '').strip()) for match in matches_l3[0]]
        print("L3 Cache misses found:", l3_cache_misses)  # Debugging print statement

        # Extract time_ns
        time_ns_pattern = re.compile(r'Time\s*\(ns\)\s*\|\s*(\d+)\s*\|\s*(\d+)')
        matches_time = time_ns_pattern.findall(content)
        if matches_time:
            time_ns  = [int(match.replace(',', '').strip()) for match in matches_time[0]]
        print("Time in nanosecond for each core found :", time_ns)
        
        # Extract cycles
        cycles_pattern = re.compile(r'Cycles\s*\|\s*(\d+)\s*\|\s*(\d+)')
        matches_cycles = cycles_pattern.findall(content)
        if matches_cycles:
             cycles = [int(match.replace(',', '').strip()) for match in matches_cycles[0]]
        print("Cycles for each core found: ", cycles)
    return branch_mispredictions, l2_cache_misses, l3_cache_misses, time_ns, cycles

# Function to traverse all subdirectories and find 'sim.out' files
def traverse_and_parse(base_directory):
    all_data = {}
    # Define the start and end of January 4th for the current year
    target_date_start = datetime.datetime(datetime.datetime.now().year, 1, 4, 0, 0, 0)
    target_date_end = datetime.datetime(datetime.datetime.now().year, 1, 4, 23, 59, 59)
 
 
    # Traverse all subdirectories
    for root, dirs, files in os.walk(base_directory):
        if benchmark in os.path.basename(root):
           for file in files:
                if file == 'sim.out':
                    file_path = os.path.join(root, file)
                    print("Processing {}...".format(file_path))

                    # Parse the sim.out file
                    branch_mispredictions, l2_cache_misses, l3_cache_misses, time_ns, cycles = parse_sim_out(file_path)

                    # Store data with the directory as the key
                    all_data[root] = {
                        'branch_mispredictions': branch_mispredictions,
                        'l2_cache_misses': l2_cache_misses,
                        'l3_cache_misses': l3_cache_misses,
                        'time_ns' : time_ns,
                        'cycles' : cycles
                    }
   

    return all_data

# Function to save data to CSV
def save_to_csv(data, filename):
    with open(filename, 'wb') as csvfile:  # 'wb' mode for Python 2
        # Dynamically generate the fieldnames for the CSV (columns for each value)
        fieldnames = ['directory']
        
        # Get the max length of values (the number of columns required for each type)
        max_len = 0
        for dir_path, row in data.items():
            max_len = max(max_len, len(row['branch_mispredictions']), len(row['l2_cache_misses']), len(row['l3_cache_misses']), len(row['time_ns']), len(row['cycles']))
        
        # Add columns for each branch misprediction, L2 cache miss, and L3 cache miss
        for i in range(max_len):
            fieldnames.append('branch_misprediction_{}'.format(i+1))
            fieldnames.append('l2_cache_miss_{}'.format(i+1))
            fieldnames.append('l3_cache_miss_{}'.format(i+1))
            fieldnames.append('time_ns_{}'.format(i+1))
            fieldnames.append('cycles_{}'.format(i+1))
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row to the CSV
        writer.writeheader()

        # Write the data rows to the CSV
        for dir_path, row in data.items():
            # Pad shorter lists with empty strings to match the max length
            branch_mispredictions = row['branch_mispredictions'] + [''] * (max_len - len(row['branch_mispredictions']))
            l2_cache_misses = row['l2_cache_misses'] + [''] * (max_len - len(row['l2_cache_misses']))
            l3_cache_misses = row['l3_cache_misses'] + [''] * (max_len - len(row['l3_cache_misses']))
            time_ns = row['time_ns']  + [''] * (max_len - len(row['time_ns'])) 
            cycles = row['cycles'] + [''] * (max_len - len(row['cycles']))
            # Create the row dictionary for this directory
            row_dict = {'directory': dir_path}
            
            # Assign values for each index
            for i in range(max_len):
                row_dict['branch_misprediction_{}'.format(i+1)] = branch_mispredictions[i]
                row_dict['l2_cache_miss_{}'.format(i+1)] = l2_cache_misses[i]
                row_dict['l3_cache_miss_{}'.format(i+1)] = l3_cache_misses[i]
                row_dict['time_ns_{}'.format(i+1)] = time_ns[i]
                row_dict['cycles_{}'.format(i+1)] = cycles[i]
            # Write the row to the CSV
            writer.writerow(row_dict)

    print "Data saved to {}".format(filename)

# Main function to run the script
def main():
    base_directory = '/root/snipersim_framework/'  # Change to the directory you want to traverse
    all_data = traverse_and_parse(base_directory)
    csv_name = 'sim_out_data_' + benchmark + '.csv'
    # Save the parsed data to a CSV file
    save_to_csv(all_data, csv_name)

if __name__ == "__main__":
    main()

