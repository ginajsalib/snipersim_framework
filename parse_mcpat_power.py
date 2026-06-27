# parse_mcpat_power.py
import re
import os


def extract_metric(text, metric_name):
    """
    Extract the FIRST occurrence of 'metric_name = X W' from text.
    The strict '\\s*=' after the name prevents 'Subthreshold Leakage'
    from accidentally matching 'Subthreshold Leakage with power gating'.
    """
    pattern = re.compile(
        r'^\s*' + re.escape(metric_name) + r'\s*=\s*([\d.e+-]+)\s*W',
        re.MULTILINE
    )
    m = pattern.search(text)
    return float(m.group(1)) if m else None


def split_major_sections(content):
    """Split McPAT output by *** separator lines into named sections."""
    parts = re.split(r'\*{10,}', content)
    return [p.strip() for p in parts if p.strip()]


def split_core_and_l2(core_text):
    """
    Split a Core section into (core_body, l2_body) at the standalone L2 block.
    Matches '    L2' on its own line, not 'L2_Local Predictor:' etc.
    """
    m = re.search(r'\n[ \t]+L2[ \t]*\n', core_text)
    if m:
        return core_text[:m.start()], core_text[m.start():]
    return core_text, ''


def extract_subsection(text, header):
    """Return text from the first occurrence of header to end of text."""
    idx = text.find(header)
    return text[idx:] if idx != -1 else ''


def parse_power_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    filename = os.path.basename(filepath)

    # Filename format: power-<start_marker>-<end_marker>.txt
    period_match = re.match(r'power-(\S+?)-(\S+?)\.txt$', filename)
    period_start = period_match.group(1) if period_match else ''
    period_end   = period_match.group(2) if period_match else ''

    result = {
        'file':         filename,
        'period_start': period_start,
        'period_end':   period_end,
    }

    sections        = split_major_sections(content)
    core_sections   = []
    processor_section = None
    l3_section      = None

    for section in sections:
        if section.startswith('McPAT'):
            continue
        elif section.startswith('Processor:'):
            processor_section = section
        elif re.match(r'Core:', section):
            core_sections.append(section)
        elif re.match(r'L3', section):
            l3_section = section

    # ------------------------------------------------------------------ #
    # Processor-level totals
    # ------------------------------------------------------------------ #
    if processor_section:
        result['total_runtime_dynamic'] = extract_metric(processor_section, 'Runtime Dynamic')
        result['total_leakage']         = extract_metric(processor_section, 'Total Leakage')
        result['total_peak_dynamic']    = extract_metric(processor_section, 'Peak Dynamic')

    # ------------------------------------------------------------------ #
    # Per-core metrics (iterates over however many Core: sections exist)
    # ------------------------------------------------------------------ #
    for i, core_text in enumerate(core_sections):
        core_body, l2_body = split_core_and_l2(core_text)

        # Core totals
        result[f'core{i}_runtime_dynamic']      = extract_metric(core_body, 'Runtime Dynamic')
        result[f'core{i}_subthreshold_leakage'] = extract_metric(core_body, 'Subthreshold Leakage')
        result[f'core{i}_gate_leakage']         = extract_metric(core_body, 'Gate Leakage')

        # L2 cache (sits at the bottom of each Core section)
        if l2_body:
            result[f'l2_core{i}_runtime_dynamic']      = extract_metric(l2_body, 'Runtime Dynamic')
            result[f'l2_core{i}_subthreshold_leakage'] = extract_metric(l2_body, 'Subthreshold Leakage')
            result[f'l2_core{i}_peak_dynamic']         = extract_metric(l2_body, 'Peak Dynamic')

        # Branch Target Buffer — directly relevant to BTB size experiments
        btb_text = extract_subsection(core_body, 'Branch Target Buffer:')
        if btb_text:
            result[f'btb_core{i}_runtime_dynamic']      = extract_metric(btb_text, 'Runtime Dynamic')
            result[f'btb_core{i}_subthreshold_leakage'] = extract_metric(btb_text, 'Subthreshold Leakage')

        # Branch Predictor (aggregated)
        bp_text = extract_subsection(core_body, 'Branch Predictor:')
        if bp_text:
            result[f'branch_predictor_core{i}_runtime_dynamic'] = extract_metric(bp_text, 'Runtime Dynamic')

        # Instruction Fetch Unit (IFU total, includes BTB + BP + I-cache)
        ifu_text = extract_subsection(core_body, 'Instruction Fetch Unit:')
        if ifu_text:
            result[f'ifu_core{i}_runtime_dynamic'] = extract_metric(ifu_text, 'Runtime Dynamic')

        # Execution Unit
        eu_text = extract_subsection(core_body, 'Execution Unit:')
        if eu_text:
            result[f'execution_unit_core{i}_runtime_dynamic'] = extract_metric(eu_text, 'Runtime Dynamic')

        # Load Store Unit
        lsu_text = extract_subsection(core_body, 'Load Store Unit:')
        if lsu_text:
            result[f'load_store_unit_core{i}_runtime_dynamic'] = extract_metric(lsu_text, 'Runtime Dynamic')

    # ------------------------------------------------------------------ #
    # L3 cache
    # ------------------------------------------------------------------ #
    if l3_section:
        result['l3_runtime_dynamic']      = extract_metric(l3_section, 'Runtime Dynamic')
        result['l3_subthreshold_leakage'] = extract_metric(l3_section, 'Subthreshold Leakage')
        result['l3_peak_dynamic']         = extract_metric(l3_section, 'Peak Dynamic')

    return result
