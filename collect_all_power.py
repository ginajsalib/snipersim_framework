import os
import re
import csv
import parse_mcpat_power  # assumes parse_mcpat_power.py is in the same dir

def collect_directories(base_dir):
    dirs = []
    pattern = re.compile(r'config_l2_256_l3MB_8192_prefetch_none_branch_(\d+)-(\d+)_barnes-intervals$')
    for entry in os.listdir(base_dir):
        path = os.path.join(base_dir, entry)
        if os.path.isdir(path):
            match = pattern.match(entry)
            if match:
                btb0 = int(match.group(1))
                btb1 = int(match.group(2))
                dirs.append((path, btb0, btb1))
    return dirs

def collect_all_power(base_dir, output_csv):
    dirs = collect_directories(base_dir)
    results = []
    for path, btb0, btb1 in dirs:
        files = [f for f in os.listdir(path) if f.startswith('power-') and f.endswith('.txt')]
        for file in files:
            filepath = os.path.join(path, file)
            data = parse_mcpat_power.parse_power_file(filepath)
            data['btbCore0'] = btb0
            data['btbCore1'] = btb1
            data['directory'] = path
            results.append(data)

    # Write CSV
    with open(output_csv, 'wb') as csvfile:
        fieldnames = ['directory', 'btbCore0', 'btbCore1', 'file', 'period_start', 'period_end', 'total_power', 'l2_power', 'l3_power']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("Done. Parsed {} power files from {} configurations.".format(len(results), len(dirs)))
    print("Output saved to {}".format(output_csv))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python collect_all_power.py <base_dir> <output_csv>")
    else:
        collect_all_power(sys.argv[1], sys.argv[2])

