import os
import re
import csv

def parse_power_file(filepath):
    result = {
        'file': os.path.basename(filepath),
        'period_start': '',
        'period_end': '',
        'total_power': 0.0,
        'l2_power': 0.0,
        'l3_power': 0.0
    }

    # Extract period from filename
    match = re.search(r'power-(periodicins-\d+)-(periodicins-\d+)-', result['file'])
    if match:
        result['period_start'] = match.group(1)
        result['period_end'] = match.group(2)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    in_processor = False
    in_l3 = False
    in_l2 = False

    for line in lines:
        line = line.strip()

        # Section flags
        if line.startswith("Processor:"):
            in_processor = True
            in_l3 = False
            in_l2 = False
        elif line.startswith("Total L3s:"):
            in_l3 = True
            in_processor = False
            in_l2 = False
        elif line.startswith("Total L2s:"):
            in_l2 = True
            in_processor = False
            in_l3 = False

        # Extract power values
        if in_processor and 'Runtime Dynamic' in line:
            result['total_power'] = float(line.split('=')[-1].replace('W', '').strip())
        elif in_l3 and 'Runtime Dynamic' in line:
            result['l3_power'] = float(line.split('=')[-1].replace('W', '').strip())
        elif in_l2 and 'Runtime Dynamic' in line:
            result['l2_power'] = float(line.split('=')[-1].replace('W', '').strip())

    return result

def parse_directory(directory, output_csv):
    files = [f for f in os.listdir(directory) if f.startswith('power-') and f.endswith('.txt')]
    results = []

    for file in files:
        filepath = os.path.join(directory, file)
        data = parse_power_file(filepath)
        results.append(data)

    # Write CSV
    with open(output_csv, 'wb') as csvfile:
        fieldnames = ['file', 'period_start', 'period_end', 'total_power', 'l2_power', 'l3_power']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("Parsed {} files. Results saved to {}".format(len(results), output_csv))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python parse_mcpat_power.py <directory> <output_csv>")
    else:
        parse_directory(sys.argv[1], sys.argv[2])

