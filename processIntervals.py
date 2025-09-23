import os
import re
import csv
import subprocess

BASE_DIR = '/root/snipersim_framework/'

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
    'L2.evict-E',
'L2.evict-I',
'L2.evict-M',
'L2.evict-O',
'L2.evict-prefetch',
'L2.evict-S',
'L2.evict-u',
'L2.evict-warmup',
'L1-D.loads', 'L1-D.stores'
'rob_timer.uop_fp_addsub', 'rob_timer.uop_fp_muldiv', 'rob_timer.uops_x87'
'rob_timer.uop_load', 'rob_timer.uop_store',
'rob_timer.uop_generic',
'rob_timer.uop_branch'
]

POWER_METRICS = [
    'power.core[0].total',
    'power.core[0].dynamic',
    'power.core[0].leakage',
    'power.core[1].total',
    'power.core[1].dynamic',
    'power.core[1].leakage'
]

def make_metric_regex(metric_name):
    pattern = r'^{} = ([\d.e+-]+), ([\d.e+-]+)'.format(re.escape(metric_name))
    return re.compile(pattern, re.MULTILINE)

def run_dumpstats(directory, start_marker, end_marker, power=False):
    partial_arg = '{}:{}'.format(start_marker, end_marker)
    cmd = [
        'python',
        '/root/sniper/tools/dumpstats.py',
        '--partial=' + partial_arg
    ]
    if power:
        cmd.append('--power')

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=directory)
    out, err = proc.communicate()
    if proc.returncode != 0:
        print("Error running dumpstats (power={}) in {}: {}".format(power, directory, err.decode('utf-8').strip()))
        return None
    return out.decode('utf-8')

def parse_metrics(output, metrics):
    data = {}
    for metric in metrics:
        regex = make_metric_regex(metric)
        match = regex.search(output)
        if match:
            data[metric] = (float(match.group(1)), float(match.group(2)))
        else:
            data[metric] = (0.0, 0.0)
    return data

def get_period_markers(output):
    markers = output.strip().split(',')
    return [m.strip() for m in markers]

def main():
    all_results = []

    for root, dirs, files in os.walk(BASE_DIR):
        if not ('sim.out' in files or 'stats.out' in files):
            continue

        print("=== Processing directory:", root)

        proc = subprocess.Popen(
            ['python', '/root/sniper/tools/dumpstats.py', '-l'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=root)
        out, err = proc.communicate()
        if proc.returncode != 0:
            print("Failed to list periods in {}: {}".format(root, err.decode('utf-8').strip()))
            continue

        periods = get_period_markers(out.decode('utf-8'))

        try:
            roi_begin_idx = periods.index('roi-begin')
            roi_end_idx = periods.index('roi-end')
        except ValueError:
            print("ROI markers not found in {}, skipping.".format(root))
            continue

        interval_markers = periods[roi_begin_idx:roi_end_idx + 1]

        for i in range(len(interval_markers) - 1):
            start_marker = interval_markers[i]
            end_marker = interval_markers[i + 1]

            print("  Processing interval: {} -> {}".format(start_marker, end_marker))

            output_metrics = run_dumpstats(root, start_marker, end_marker, power=False)
            if output_metrics is None:
                print("    [Warning] Failed to get metrics for interval, skipping.")
                continue

        #    output_power = run_dumpstats(root, start_marker, end_marker, power=True)
         #   if output_power is None:
         #       print("    [Warning] Failed to get power data for interval, skipping.")
         #       continue

            with open(os.path.join(root, 'dumpstats_{}_{}.out'.format(start_marker, end_marker)), 'w') as f:
                f.write(output_metrics)
          #  with open(os.path.join(root, 'dumpstats_power_{}_{}.out'.format(start_marker, end_marker)), 'w') as f:
          #      f.write(output_power)

            metrics_data = parse_metrics(output_metrics, METRICS)
       #     power_data = parse_metrics(output_power, POWER_METRICS)

            combined_data = {}
            combined_data.update(metrics_data)
        #    combined_data.update(power_data)
            combined_data['directory'] = root
            combined_data['period'] = '{}:{}'.format(start_marker, end_marker)
            all_results.append(combined_data)

    csv_file = os.path.join(BASE_DIR, 'interval_metrics_power.csv')
    with open(csv_file, 'w') as csvfile:
        fieldnames = ['directory', 'period']
        for m in METRICS + POWER_METRICS:
            fieldnames.append(m + '_core0')
            fieldnames.append(m + '_core1')

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in all_results:
            csv_row = {'directory': row['directory'], 'period': row['period']}
            for metric in METRICS + POWER_METRICS:
                val0, val1 = row.get(metric, (0.0, 0.0))
                csv_row[metric + '_core0'] = val0
                csv_row[metric + '_core1'] = val1
            writer.writerow(csv_row)

    print("\nAll data written to", csv_file)

if __name__ == '__main__':
    main()

