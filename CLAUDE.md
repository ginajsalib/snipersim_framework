# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Post-silicon customization framework for heterogeneous multi-core CPUs that optimizes power consumption without affecting performance. Uses ML (Random Forest, Deep Neural Networks) to predict optimal processor configurations (branch predictor size, prefetcher, cache settings) per workload phase.

The codebase runs Sniper multicore simulations with varying L2/L3 cache sizes, branch predictor sizes, and prefetcher configurations, then analyzes results to identify optimal configurations using PPW (Performance Per Watt) metrics.

**Research Context:** This work implements post-silicon customization techniques described in:
> Weston, K., Janfaza, V., Taur, A., Mungra, D., Kansal, A., Zahran, M., & Muzahid, A. (2023). Post-Silicon Customization Using Deep Neural Networks. ARCS.

## Key Commands

### Run Simulations
```bash
# Run all config combinations (bash entry point)
./runSniper.sh

# Run with per-core branch predictor configs (asymmetric workloads)
./runSniperWithCfg.sh
```

### Python Pipeline
```bash
# Full pipeline: perf+power CSVs → training data with best/top-3 configs
python pythonScripts/run_pipeline.py \
  --perf_csv <perf.csv> --power_csv <power.csv> --benchmark <name>

# Individual stages
python pythonScripts/parseInputConfigsOnPerf.py <input> <output>
python pythonScripts/deleteEmptyRows.py <input> <output>
python pythonScripts/mergePerfAndPower.py <perf> <power> <output>
python pythonScripts/addCalculatedColumnsToMergedCsv.py <input> <output>
python pythonScripts/findBestConfigUsingPPW.py <input> <output>
python pythonScripts/createTrainingDataLagged.py <input> <output>
python pythonScripts/createTrainingDataWithLabels.py <lagged> <best> <output>
python pythonScripts/findTop3ConfigsByPPW.py <input> <output>
python pythonScripts/mergeTrainWithTop3.py <lagged> <top3> <output>
```

### Analysis
```bash
# Post-hoc analysis of trained Random Forest model
python pythonScripts/postAnalysis.py --model-dir saved_models/

# Merge top-3 configs across prefetcher variants
python mergeCsvPrefetchersNew.py <csv1> <csv2> [csv3...] [output.csv]
```

## Architecture

### Simulation Layer
- `runSniper.py` / `runSniper.sh`: Generates Sniper simulation configs (L2, L3, prefetcher, branch predictor)
- Output dirs: `config_l2_<size>_l3MB_<size>_prefetch_<type>_branch_<bp0>-<bp1>_<benchmark>`

### Power Analysis
- `runSniper.py`: Calls mcpat.py for power estimation
- `collect_all_power.py`: Aggregates power traces across configs
- `parse_mcpat_power.py`: Parses MCPower output files

### Performance Analysis
- `processSimOut.py`: Extracts branch mispredictions, L2/L3 cache misses from sim.out

### Data Pipeline (pythonScripts/)
1. **Parse**: Extract config params from perf CSVs (`parseInputConfigsOnPerf.py`)
2. **Clean**: Remove empty rows (`deleteEmptyRows.py`)
3. **Merge**: Join perf+power by directory + period interval (`mergePerfAndPower.py`)
4. **Enrich**: Add calculated columns (`addCalculatedColumnsToMergedCsv.py`)
5. **Label**: Find best/top-3 configs by PPW (`findBestConfigUsingPPW.py`, `findTop3ConfigsByPPW.py`)
6. **Train**: Create lagged features + merge labels (`createTrainingDataLagged.py`, `createTrainingDataWithLabels.py`)

### ML Models
- `Top3RandomForest.py`: 4-way config predictor (btbCore0, btbCore1, prefetcher_core0, prefetcher_core1)
- `randomForestTop3MultipleBenchmarks.py`: Multi-benchmark training
- `randomForestGPU.py`: GPU-accelerated variant

### Key Metrics
- **PPW**: Performance Per Watt (higher = better)
- **Top-3 accuracy**: Relaxed matching (predicted config is any of best/2nd/3rd)
- **PPW % loss**: Degradation vs optimal config

## Data Conventions
- Period intervals: `periodicins-<start>:<end>` or `roi-begin`
- BTB sizes: 512, 1024, 2048, 4096
- Prefetchers: none, simple, ghb
- Core-asymmetric configs: `branch_<bp0>-<bp1>` (core0-core1)
