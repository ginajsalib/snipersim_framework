import json
import re
import csv

DUMPSTATS_FILE = "dumpstats.out"
POWER_JSON_FILE = "viz/levels/level2/data/config_l2_256_l3MB_8192_prefetch_none_branch_512-512_barnes-intervals-power.json"
OUTPUT_CSV = "periodic_power.csv"

def parse_dumpstats_intervals(filepath):
    time_to_ins = {}
    pattern = re.compile(r'periodic-(\d+)\s*=\s*([\d,]+)')
    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                time = int(match.group(1))
                ins_str = match.group(2).split(',')[0].strip()  # handle cases with extra values
                ins = int(ins_str.replace(',', ''))
                time_to_ins[time] = ins
    return time_to_ins

def parse_power_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    power_data = []
    for entry in data:
        component = entry.get("name", "unknown")
        for point in entry["data"]:
            time = point["x"]
            power = point["y"]
            power_data.append({
                "component": component,
                "time": time,
                "power": power
            })
    return power_data

def map_time_to_ins(power_data, time_to_ins):
    time_to_ins_sorted = sorted(time_to_ins.items())
    results = []
    for p in power_data:
        t = p["time"]
        # Find closest lower bound time
        matching_time = None
        for tt, ins in time_to_ins_sorted:
            if tt / 1e9 <= t:
                matching_time = tt
            else:
                break
        if matching_time is not None:
            ins = time_to_ins[matching_time]
            results.append({
                "period_start_ins": ins,
                "period_end_ins": None,  # will fill later
                "component": p["component"],
                "power": p["power"],
                "time": t
            })
    # Fill end instructions
    for i in range(len(results)-1):
        results[i]["period_end_ins"] = results[i+1]["period_start_ins"]
    if results:
        results[-1]["period_end_ins"] = results[-1]["period_start_ins"]
    return results

def save_csv(data, output_file):
    with open(output_file, 'wb') as f:  # 'wb' needed for py2 csv on some systems
        writer = csv.DictWriter(f, fieldnames=["period_start_ins", "period_end_ins", "component", "power", "time"])
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def main():
    print("Parsing dumpstats...")
    time_to_ins = parse_dumpstats_intervals(DUMPSTATS_FILE)
    if not time_to_ins:
        raise RuntimeError("No periodic data found in dumpstats.out!")
    
    print("Parsing power JSON...")
    power_data = parse_power_json(POWER_JSON_FILE)
    if not power_data:
        raise RuntimeError("No data points found in power JSON!")

    print("Mapping time to instruction count...")
    mapped_data = map_time_to_ins(power_data, time_to_ins)

    print("Saving to {}".format(OUTPUT_CSV))
    save_csv(mapped_data, OUTPUT_CSV)

    print(" CSV file generated.")

if __name__ == "__main__":
    main()

