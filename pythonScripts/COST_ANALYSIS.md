# Cost Analysis for Post-Silicon Customization

## Overview

This document explains the energy and power calculations used in the cost analysis scripts (`costAnalysis.py` and the Section 5 additions to `postAnalysis.py`) for the post-silicon customization framework.

**Research Context:**
> Weston, K., Janfaza, V., Taur, A., Mungra, D., Kansal, A., Zahran, M., & Muzahid, A. (2023). Post-Silicon Customization Using Deep Neural Networks. ARCS.

---

## 1. Technology Parameters (Gainestown/McPAT)

All calculations use the Gainestown architecture parameters from McPAT configuration (`power.xml`):

| Parameter | Value | Source |
|-----------|-------|--------|
| **Technology Node** | 45 nm | `core_tech_node` |
| **Core Clock** | 2660 MHz | `target_core_clockrate` |
| **Supply Voltage (Vdd)** | 1.2 V | `vdd` |
| **Device Type** | HP (High Performance) | `device_type` |
| **Power Gating** | Enabled | `power_gating` |

### Component Defaults

| Component | Default Value | McPAT Param |
|-----------|---------------|-------------|
| BTB entries | 18,944 | `BTB_config` |
| L2 capacity | 256 KB (262,144 B) | `L2_config` |
| L3 capacity | 8,192 KB (8,388,608 B) | `L3_config` |
| L2 associativity | 8-way | `L2_config` |
| L3 associativity | 16-way | `L3_config` |
| Cache line size | 64 B | `L2_config`/`L3_config` |

---

## 2. Energy Parameters (45nm HP, 1.2V)

Energy values are derived from McPAT gate capacitance models for 45nm High Performance technology:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `E_gate_per_btb_entry` | 0.05 pJ | Clock gating energy per BTB entry |
| `E_gate_per_l2_kb` | 2.5 pJ | Clock gating energy per KB of L2 |
| `E_gate_per_l3_kb` | 1.8 pJ | Clock gating energy per KB of L3 |
| `E_gate_per_prefetcher` | 15.0 pJ | Prefetcher state machine gating |
| `E_rf_inst` | 0.5 pJ | RF inference instruction fetch |
| `E_rf_cmp` | 0.1 pJ | Tree node comparison |
| `E_rf_mem` | 1.0 pJ | SRAM access (feature lookup) |

---

## 3. Reconfiguration Energy Calculation

### 3.1 Clock Gating Model

The paper advocates using **clock gating** instead of cache invalidation to avoid writeback costs. When a component is reconfigured:

$$E_{reconfig} = \sum_{components} N_{units} \times C_{gate} \times V_{dd}^2$$

Where:
- $N_{units}$ = number of gated units (entries, KB, or state bits)
- $C_{gate}$ = gate capacitance for the component type
- $V_{dd}$ = supply voltage (1.2V)

### 3.2 Per-Component Energy

**BTB Reconfiguration:**
$$E_{BTB} = E_{gate\_per\_btb\_entry} \times |BTB_{new} - BTB_{old}|$$

Example: Changing BTB from 512 → 1024 entries:
$$E_{BTB} = 0.05 \text{ pJ} \times |1024 - 512| = 0.05 \times 512 = 25.6 \text{ pJ}$$

**L2 Cache Reconfiguration:**
$$E_{L2} = E_{gate\_per\_l2\_kb} \times |L2_{new} - L2_{old}|$$

Example: Changing L2 from 256 KB → 512 KB:
$$E_{L2} = 2.5 \text{ pJ} \times |512 - 256| = 2.5 \times 256 = 640 \text{ pJ}$$

**L3 Cache Reconfiguration:**
$$E_{L3} = E_{gate\_per\_l3\_kb} \times |L3_{new} - L3_{old}|$$

Example: Changing L3 from 4096 KB → 8192 KB:
$$E_{L3} = 1.8 \text{ pJ} \times |8192 - 4096| = 1.8 \times 4096 = 7,372.8 \text{ pJ}$$

**Prefetcher Reconfiguration:**
$$E_{prefetcher} = E_{gate\_per\_prefetcher} \times \delta_{type}$$

Where $\delta_{type} = 1$ if prefetcher type changes (none↔simple↔ghb), 0 otherwise.

### 3.3 Total Reconfiguration Energy

For each interval $i$:

$$E_{reconfig}(i) = \begin{cases}
E_{BTB}(i) + E_{L2}(i) + E_{L3}(i) + E_{prefetcher}(i) & \text{if config changed} \\
0 & \text{if config unchanged}
\end{cases}$$

**Config Change Detection:**
```python
config_changed = (
    btbCore0[i] != btbCore0_prev[i] or
    btbCore1[i] != btbCore1_prev[i] or
    L2[i] != L2_prev[i] or
    L3[i] != L3_prev[i] or
    prefetcher[i] != prefetcher_prev[i]
)
```

---

## 4. RF Model Inference Energy

### 4.1 Model Structure

The Random Forest model has:
- `n_estimators` = number of trees (e.g., 200)
- `max_depth` = maximum tree depth (e.g., 15)
- `n_features` = number of input features

### 4.2 Inference Operations

For each prediction:

| Operation | Count | Energy |
|-----------|-------|--------|
| Instruction fetch | $N_{inst} \approx 3 \times N_{comparisons}$ | $N_{inst} \times E_{rf\_inst}$ |
| Tree comparisons | $N_{trees} \times depth$ | $N_{comparisons} \times E_{rf\_cmp}$ |
| Memory access | $N_{trees} \times n_{features}$ | $N_{mem} \times E_{rf\_mem}$ |

### 4.3 Total Inference Energy

$$E_{inference} = E_{inst} + E_{cmp} + E_{mem}$$

$$E_{inference} = (N_{inst} \times 0.5 \text{ pJ}) + (N_{comparisons} \times 0.1 \text{ pJ}) + (N_{mem} \times 1.0 \text{ pJ})$$

**Example** (200 trees, depth 15, 60 features):
- $N_{comparisons} = 200 \times 15 = 3,000$
- $N_{inst} = 3 \times 3,000 = 9,000$
- $N_{mem} = 200 \times 60 = 12,000$

$$E_{inference} = (9000 \times 0.5) + (3000 \times 0.1) + (12000 \times 1.0) = 4500 + 300 + 12000 = 16,800 \text{ pJ} = 16.8 \text{ nJ}$$

---

## 5. Net PPW Calculation

### 5.1 Raw PPW

$$PPW_{raw} = \frac{IPS^3}{P_{total}}$$

Where:
- $IPS$ = Instructions Per Second
- $P_{total}$ = Total power (W)

### 5.2 Overhead-Adjusted PPW

Each interval has duration $T_{interval}$ based on 500K instructions:

$$T_{interval} = \frac{500,000}{IPS}$$

Net PPW accounts for inference and reconfiguration overhead:

$$PPW_{net} = PPW_{raw} - \frac{E_{inference} + E_{reconfig}}{T_{interval}}$$

### 5.3 PPW Overhead Percentage

$$Overhead_{\%} = \frac{PPW_{raw} - PPW_{net}}{PPW_{raw}} \times 100\%$$

---

## 6. Output Files

### 6.1 cost_analysis_{benchmark}.csv

| Column | Description |
|--------|-------------|
| `benchmark` | Benchmark name |
| `period_start`, `period_end` | Interval boundaries |
| `btbCore0`, `btbCore1` | Current BTB sizes |
| `prefetcher`, `L2`, `L3` | Current config |
| `*_prev` columns | Previous interval config |
| `config_changed` | Boolean: any change detected |
| `btb_changed`, `l2_changed`, `l3_changed`, `prefetcher_changed` | Per-component flags |
| `inference_energy_pJ` | RF inference energy (constant) |
| `reconfig_energy_pJ` | Reconfiguration energy (per interval) |
| `reconfig_breakdown` | JSON: per-component energy |
| `PPW_best` | Raw PPW of best config |
| `net_ppw` | PPW after overhead |

### 6.2 Text Report

The report includes:
1. All McPAT/Gainestown parameters used
2. Component parameters (BTB, L2, L3 defaults)
3. Energy parameters (45nm HP values)
4. Inference cost breakdown
5. Reconfiguration cost summary (per component)
6. Config change frequency per component
7. Net PPW impact

---

## 7. Usage

### Run Cost Analysis

```bash
python pythonScripts/costAnalysis.py \
  --benchmark barnes \
  --model-dir saved_models/ \
  --output cost_analysis_barnes.csv \
  --report cost_report.txt
```

### Run Post-Analysis with Cost Integration

```bash
python pythonScripts/postAnalysis.py \
  --model-dir saved_models/ \
  --cost-csv cost_analysis_barnes.csv
```

---

## 8. Key Design Decisions

1. **Clock Gating vs. Cache Invalidation**: Following the paper, we use clock gating to avoid cache writeback costs (~100 pJ/line for off-chip transfer).

2. **Per-Row Calculation**: Config values differ for each data row based on current vs. predicted config.

3. **Change-Only Reconfiguration**: Reconfiguration energy is only applied when the config actually changes from the previous interval.

4. **Component Breakdown**: Separate tracking of BTB, L2, L3, and prefetcher changes allows identifying which components dominate reconfiguration cost.

5. **Net PPW Metric**: The final metric accounts for both the ML overhead (inference) and the physical reconfiguration cost.
