import os
import re
import csv
import subprocess

METRICS = [
    'branch_predictor.num-correct',
    'branch_predictor.num-incorrect',
    'L2.load-misses',
    'L2.loads',
    'L2.stores',
    'L2.store-misses',
    'core.instructions',
    'performance_model.elapsed_time',
    'performance_model.idle_elapsed_time',
    'performance_model.instruction_count',
    'L3.load-misses',
    'L3.loads',
    'L3.stores',
    'L3.store-misses'
]

def run_command(command, cwd):
    """Run a shell command in a given directory and return stdout."""
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print('Command failed in %s: %s' % (cwd, stderr))
        return None
    return stdout

def get_available_periods_and_markers(directory):
    """Run dumpstats.py -l and extract available periods and markers."""
    output = run_command(['python', '/root/sniper/tools/dumpstats.py', '-l'], directory)
    if output is None:
        return [], {}

    periods = []
    markers = {}

    parts = output.strip().split(',')
    for part in parts:
        part = part.strip()
        if part in ['start', 'roi-begin', 'roi-end', 'stop']:
            markers[part] = part  # Save the marker name for CSV
        elif part.startswith('periodic-'):
            match = re.match(r'periodic-(\d+)', part)
            if match:
                periods.append(int(match.group(1)))

    periods.sort()
    return periods, markers

def parse_dumpstats_output(output):
    """Extract required metrics and markers from dumpstats output."""
    data = {'start': None, 'roi-begin': None, 'roi-end': None, 'stop': None}
    for metric in METRICS:
        data[metric] = [None, None]

    for line in output.splitlines():
        line = line.strip()
        if line.startswith('Start:'):
            data['start'] = int(line.split(':')[1].strip())
        elif line.startswith('ROI Begin:'):
            data['roi-begin'] = int(line.split(':')[1].strip())
        elif line.startswith('ROI End:'):
            data['roi-end'] = int(line.split(':')[1].strip())
        elif line.startswith('Stop:'):
            data['stop'] = int(line.split(':')[1].strip())
        else:
            for metric in METRICS:
                if metric in line:
                    values = re.findall(r'\d+', line)
                    if len(values) >= 2:
                        data[metric] = [int(values[0]), int(values[1])]
    return data

def process_interval(directory, start, end, label):
    """Run dumpstats.py for a specific interval and parse output."""
    command = [
        'python',
        '/root/sniper/tools/dumpstats.py',
        '--partial=%s:%s' % (start, end)
    ]
    dumpstats_output = run_command(command, directory)
    if dumpstats_output is None:
        return None

    # Save the output to dumpstats.out
    with open(os.path.join(directory, 'dumpstats.out'), 'a') as f:
        f.write(dumpstats_output)
        f.write('\n\n')

    data = parse_dumpstats_output(dumpstats_output)
    data['period'] = label
    return data

def process_directory(directory):
    """Process all available periods and the ROI interval in the given directory."""
    interval_data = []
    periods, markers = get_available_periods_and_markers(directory)
    if not periods or len(periods) < 2:
        print('No valid periods found in %s' % directory)
        return interval_data

    # Process periodic intervals
    for i in range(len(periods) - 1):
        start = 'periodic-%d' % periods[i]
        end = 'periodic-%d' % periods[i + 1]
        label = 'periodic-%d' % periods[i]

        data = process_interval(directory, start, end, label)
        if data:
            interval_data.append(data)

    # Process ROI interval if available
    if 'roi-begin' in markers and 'roi-end' in markers:
        data = process_interval(directory, 'roi-begin', 'roi-end', 'roi-interval')
        if data:
            interval_data.append(data)

    # Add the markers to all rows
    for interval in interval_data:
        for key in ['start', 'roi-begin', 'roi-end', 'stop']:
            if key in markers:
                interval[key] = markers[key]

    return interval_data

def traverse_and_process(base_directory):
    """Walk through all directories and process benchmarks."""
    results = {}
    for root, dirs, files in os.walk(base_directory):
        if 'sim.out' in files:
            print('Processing directory: %s' % root)
            results[root] = process_directory(root)
    return results

def save_to_csv(results, filename):
    """Save collected data to CSV."""
    fieldnames = ['directory', 'period', 'start', 'roi-begin', 'roi-end', 'stop']
    for metric in METRICS:
        fieldnames.append('%s_core0' % metric)
        fieldnames.append('%s_core1' % metric)

    with open(filename, 'wb') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for directory, intervals in results.items():
            for interval in intervals:
                row = {'directory': directory, 'period': interval.get('period', '')}
                for key in ['start', 'roi-begin', 'roi-end', 'stop']:
                    row[key] = interval.get(key, '')

                for metric in METRICS:
                    row['%s_core0' % metric] = interval[metric][0] if interval[metric][0] is not None else ''
                    row['%s_core1' % metric] = interval[metric][1] if interval[metric][1] is not None else ''
                writer.writerow(row)

    print('Results saved to %s' % filename)

def main():
    base_directory = '/root/snipersim_framework/'  # Update this if needed
    results = traverse_and_process(base_directory)
    save_to_csv(results, 'periodic_dumpstats.csv')

if __name__ == '__main__':
    main()

