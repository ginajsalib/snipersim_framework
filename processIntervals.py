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
    'L3.store-misses',
    'L2.evict-E',
    'L2.evict-I',
    'L2.evict-M',
    'L2.evict-O',
    'L2.evict-prefetch',
    'L2.evict-S',
    'L2.evict-u',
    'L2.evict-warmup',
    'L1-D.loads',
    'L1-D.stores',
    'rob_timer.uop_fp_addsub',
    'rob_timer.uop_fp_muldiv',
    'rob_timer.uops_x87',
    'rob_timer.uop_load',
    'rob_timer.uop_store',
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

def calculate_derived_metrics(metrics_data):
    """Calculate the requested derived metrics for both cores"""
    derived_metrics = {}
    
    for core in [0, 1]:  # Calculate for both cores
        core_suffix = f'_core{core}'
        
        # Get core values (core 0 = index 0, core 1 = index 1)
        def get_metric_value(metric_name):
            return metrics_data.get(metric_name, (0.0, 0.0))[core]
        
        instructions = get_metric_value('core.instructions')
        
        # 1. L2 most usage
        l2_usage = get_metric_value('L2.loads') + get_metric_value('L2.stores')
        derived_metrics[f'l2_most_usage{core_suffix}'] = l2_usage
        
        # 2. Normalized commit float
        float_ops = (get_metric_value('rob_timer.uop_fp_addsub') + 
                    get_metric_value('rob_timer.uop_fp_muldiv') + 
                    get_metric_value('rob_timer.uops_x87'))
        norm_commit_float = float_ops / instructions if instructions > 0 else 0.0
        derived_metrics[f'normalized_commit_float{core_suffix}'] = norm_commit_float
        
        # 3. Normalized commit mem
        mem_ops = get_metric_value('rob_timer.uop_load') + get_metric_value('rob_timer.uop_store')
        norm_commit_mem = mem_ops / instructions if instructions > 0 else 0.0
        derived_metrics[f'normalized_commit_mem{core_suffix}'] = norm_commit_mem
        
        # 4. Normalized commit int
        int_ops = get_metric_value('rob_timer.uop_generic')
        norm_commit_int = int_ops / instructions if instructions > 0 else 0.0
        derived_metrics[f'normalized_commit_int{core_suffix}'] = norm_commit_int
        
        # 5. L1 data access
        l1_data_access = get_metric_value('L1-D.loads') + get_metric_value('L1-D.stores')
        derived_metrics[f'l1_data_access{core_suffix}'] = l1_data_access
        
        # 6. Normalized commit ctrl
        ctrl_ops = get_metric_value('rob_timer.uop_branch')
        norm_commit_ctrl = ctrl_ops / instructions if instructions > 0 else 0.0
        derived_metrics[f'normalized_commit_ctrl{core_suffix}'] = norm_commit_ctrl
        
        # 7. L2 avg eviction rate
        l2_evictions = (get_metric_value('L2.evict-E') + get_metric_value('L2.evict-I') + 
                       get_metric_value('L2.evict-M') + get_metric_value('L2.evict-O') + 
                       get_metric_value('L2.evict-S'))
        l2_total_accesses = get_metric_value('L2.loads') + get_metric_value('L2.stores')
        l2_eviction_rate = l2_evictions / l2_total_accesses if l2_total_accesses > 0 else 0.0
        derived_metrics[f'l2_avg_eviction_rate{core_suffix}'] = l2_eviction_rate
        
        # 8. L2 most hit rate
        l2_misses = get_metric_value('L2.load-misses') + get_metric_value('L2.store-misses')
        l2_hits = l2_total_accesses - l2_misses
        l2_hit_rate = l2_hits / l2_total_accesses if l2_total_accesses > 0 else 0.0
        derived_metrics[f'l2_most_hit_rate{core_suffix}'] = l2_hit_rate
        
        # 9. L3 usage
        l3_usage = get_metric_value('L3.loads') + get_metric_value('L3.stores')
        derived_metrics[f'l3_usage{core_suffix}'] = l3_usage
        
        # 10. Branch mispred rate
        branch_correct = get_metric_value('branch_predictor.num-correct')
        branch_incorrect = get_metric_value('branch_predictor.num-incorrect')
        total_branches = branch_correct + branch_incorrect
        branch_mispred_rate = branch_incorrect / total_branches if total_branches > 0 else 0.0
        derived_metrics[f'branch_mispred_rate{core_suffix}'] = branch_mispred_rate
    
    return derived_metrics

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

            with open(os.path.join(root, 'dumpstats_{}_{}.out'.format(start_marker, end_marker)), 'w') as f:
                f.write(output_metrics)

            metrics_data = parse_metrics(output_metrics, METRICS)
            
            # Calculate derived metrics
            derived_metrics = calculate_derived_metrics(metrics_data)

            combined_data = {}
            combined_data.update(metrics_data)
            combined_data.update(derived_metrics)
            combined_data['directory'] = root
            combined_data['period'] = '{}:{}'.format(start_marker, end_marker)
            all_results.append(combined_data)

    csv_file = os.path.join(BASE_DIR, 'interval_metrics_power.csv')
    with open(csv_file, 'w') as csvfile:
        # Define fieldnames including derived metrics
        fieldnames = ['directory', 'period']
        
        # Add original metrics
        for m in METRICS + POWER_METRICS:
            fieldnames.append(m + '_core0')
            fieldnames.append(m + '_core1')
        
        # Add derived metrics
        derived_metric_names = [
            'l2_most_usage',
            'normalized_commit_float',
            'normalized_commit_mem', 
            'normalized_commit_int',
            'l1_data_access',
            'normalized_commit_ctrl',
            'l2_avg_eviction_rate',
            'l2_most_hit_rate',
            'l3_usage',
            'branch_mispred_rate'
        ]
        
        for metric in derived_metric_names:
            fieldnames.append(metric + '_core0')
            fieldnames.append(metric + '_core1')

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in all_results:
            csv_row = {'directory': row['directory'], 'period': row['period']}
            
            # Add original metrics
            for metric in METRICS + POWER_METRICS:
                val0, val1 = row.get(metric, (0.0, 0.0))
                csv_row[metric + '_core0'] = val0
                csv_row[metric + '_core1'] = val1
            
            # Add derived metrics
            for metric in derived_metric_names:
                csv_row[metric + '_core0'] = row.get(metric + '_core0', 0.0)
                csv_row[metric + '_core1'] = row.get(metric + '_core1', 0.0)
                
            writer.writerow(csv_row)

    print("\nAll data written to", csv_file)

if __name__ == '__main__':
    main()
