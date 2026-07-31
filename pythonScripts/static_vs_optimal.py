#!/usr/bin/env python3
"""
static_vs_optimal.py
====================
How much PPW does a single FIXED configuration lose against the per-interval
optimal (oracle) configuration?  Barnes / Cholesky / FFT / Radiosity.

Per-interval loss for a static config s:
        loss(s, t) = (PPW_oracle(t) - PPW_s(t)) / PPW_oracle(t) * 100

Reported for:
    BEST static   -- the single config with the highest mean PPW (chosen with
                     full hindsight, so this is the *strongest possible* static
                     baseline; the real static loss can only be worse)
    MEDIAN static -- the middle config, i.e. a config picked without insight
    WORST static  -- the floor of the design space

CORRECTNESS
-----------
All configs are compared on a BALANCED PANEL: a set of configs that each have a
finite, positive PPW on every one of a common set of intervals. Without this,
configs get averaged over different intervals and a static config can appear to
beat the oracle. Rows with inf / NaN / non-positive PPW and unparseable periods
are dropped up front; duplicate (period, config) rows are collapsed to a mean.

USAGE
-----
    python static_vs_optimal.py --merged-full-dir /path/to/merged_full/
    python static_vs_optimal.py --merged-full-dir DIR --benchmarks barnes fft
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

TARGET_KEYS = ['l2_core0', 'l2_core1', 'l3', 'btb_core0', 'btb_core1',
               'prefetch_core0', 'prefetch_core1']
DIMS = {
    'l2_core0':       ['l2Core0', 'L2core0', 'L2 core 0'],
    'l2_core1':       ['l2Core1', 'L2core1', 'L2 core 1'],
    'l3':             ['l3Size', 'L3'],
    'btb_core0':      ['btbCore0', 'BTBcore0', 'BTB core 0'],
    'btb_core1':      ['btbCore1', 'BTBcore1', 'BTB core 1'],
    'prefetch_core0': ['prefetchCore0', 'Prefetchcore0', 'Prefetch core 0'],
    'prefetch_core1': ['prefetchCore1', 'Prefetchcore1', 'Prefetch core 1'],
}
PREFETCH = {'prefetch_core0', 'prefetch_core1'}


def _n(s):
    return str(s).lower().replace('_', '').replace(' ', '').replace('.', '')


def col(cols, cands):
    for c in cands:
        for k in cols:
            if _n(k) == _n(c):
                return k
    for c in cands:
        for k in cols:
            if _n(c) in _n(k):
                return k
    return None


def period_num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return None if (isinstance(v, float) and np.isnan(v)) else float(v)
    s = str(v).strip()
    if s.lower() in ('nan', ''):
        return None
    if '-' in s:
        t = s.rsplit('-', 1)[-1]
        return float(t) if t.replace('.', '', 1).isdigit() else None
    return float(s) if s.replace('.', '', 1).isdigit() else None


def canon(series, is_pf):
    if is_pf:
        return series.fillna('none').astype(str).str.strip().str.lower()
    num = pd.to_numeric(series, errors='coerce')
    out = num.map(lambda v: 'nan' if pd.isna(v) else f'{float(v):.6g}')
    bad = num.isna() & series.notna()
    if bad.any():
        out[bad] = series[bad].astype(str).str.strip().str.lower()
    return out


def load_panel(path, bench, chunksize, log):
    """merged_full_<bench>.csv -> dense [periods x configs] PPW matrix (NaN = absent)."""
    head = list(pd.read_csv(path, nrows=0).columns)
    pcol, vcol = col(head, ['period_start']), col(head, ['ppw'])
    dcol = {k: col(head, c) for k, c in DIMS.items()}
    bcol = col(head, ['benchmark'])
    miss = [k for k, v in dcol.items() if v is None]
    if pcol is None or vcol is None or miss:
        raise ValueError(f'unresolved columns (period={pcol}, ppw={vcol}, dims missing={miss})')

    use = list(dict.fromkeys([pcol, vcol] + list(dcol.values()) + ([bcol] if bcol else [])))
    cfg_id, per_id = {}, {}
    P_, C_, V_ = [], [], []
    n_read = n_drop = 0

    for ch in pd.read_csv(path, usecols=use, chunksize=chunksize, low_memory=False):
        n_read += len(ch)
        if bcol and bcol in ch.columns:
            ch = ch[ch[bcol].astype(str).str.lower() == bench.lower()]
            if ch.empty:
                continue
        per = ch[pcol].map(period_num)
        ppw = pd.to_numeric(ch[vcol], errors='coerce')
        good = per.notna() & np.isfinite(ppw) & (ppw > 0)
        n_drop += int((~good).sum())
        if not good.any():
            continue
        sub = ch.loc[good]

        key = None
        for k in TARGET_KEYS:
            part = canon(sub[dcol[k]], k in PREFETCH)
            key = part if key is None else key + '|' + part
        for x in key.unique():
            cfg_id.setdefault(x, len(cfg_id))
        C_.append(key.map(cfg_id).to_numpy(np.int32))

        pv = per.loc[good]
        for x in pv.unique():
            per_id.setdefault(x, len(per_id))
        P_.append(pv.map(per_id).to_numpy(np.int32))
        V_.append(ppw.loc[good].to_numpy(np.float64))

    if not V_:
        raise ValueError('no usable rows')

    pc, cc, vv = np.concatenate(P_), np.concatenate(C_), np.concatenate(V_)
    P, C = len(per_id), len(cfg_id)
    if P * C > 400_000_000:
        raise MemoryError(f'panel {P}x{C} too large')

    flat = pc.astype(np.int64) * C + cc.astype(np.int64)
    s = np.bincount(flat, weights=vv, minlength=P * C)
    n = np.bincount(flat, minlength=P * C)
    dups = int((n > 1).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        mat = np.where(n > 0, s / np.maximum(n, 1), np.nan).reshape(P, C)

    keys = [None] * C
    for k, i in cfg_id.items():
        keys[i] = tuple(k.split('|'))

    log(f'    rows {n_read:,} read, {n_drop:,} dropped (bad period / inf / NaN / PPW<=0)'
        + (f', {dups:,} duplicate cells averaged' if dups else ''))
    log(f'    {P:,} intervals x {C:,} configs, panel {np.isfinite(mat).mean()*100:.1f}% filled')
    return mat, keys


def balanced(mat, log):
    """Largest complete rectangle: configs sorted by coverage, keep the prefix
    maximising (#configs x #intervals with no missing cell)."""
    m = np.isfinite(mat)
    P, C = m.shape
    order = np.argsort(-m.mean(axis=0), kind='stable')
    cum = np.logical_and.accumulate(m[:, order], axis=1)
    per_k = cum.sum(axis=0).astype(np.int64)
    k = np.arange(1, C + 1, dtype=np.int64)
    ok = (k >= 2) & (per_k >= 1)
    if not ok.any():
        raise ValueError('no balanced panel with >=2 configs')
    kb = int(k[ok][np.argmax((k * per_k)[ok])])
    cols, rows = np.sort(order[:kb]), np.where(cum[:, kb - 1])[0]
    log(f'    balanced panel: {len(rows):,} intervals x {kb:,} configs '
        f'({len(rows)/P*100:.0f}% of intervals, {kb/C*100:.0f}% of configs)')
    if kb < C:
        log(f'    ({C-kb:,} config(s) excluded: not simulated in every interval)')
    return rows, cols


def report(bench, mat, keys, log):
    rows, cols = balanced(mat, log)
    M = mat[np.ix_(rows, cols)]
    assert np.isfinite(M).all() and (M > 0).all()
    T, N = M.shape

    oracle = M.max(axis=1)                       # per-interval optimal PPW
    loss = (oracle[:, None] - M) / oracle[:, None] * 100.0   # [T, N] % loss, >= 0
    assert (loss >= -1e-9).all(), 'negative loss: panel not balanced'

    # Rank by MEAN PER-INTERVAL % LOSS (ascending), not by mean PPW. Ranking by
    # mean PPW and then reporting mean loss gives two different orderings -- the
    # "best" config can then show a higher mean loss than a middling one, because
    # mean PPW is dominated by high-PPW intervals while mean loss weights every
    # interval equally. Ranking on the reported metric guarantees BEST <= MEDIAN
    # <= WORST and makes the baseline maximally favourable to the static case.
    mean_loss = loss.mean(axis=0)
    rank = np.argsort(mean_loss)
    picks = [('BEST static  (hindsight-chosen)', rank[0]),
             ('MEDIAN static (blind pick)     ', rank[len(rank) // 2]),
             ('WORST static                   ', rank[-1])]
    j_ppw = int(np.argmax(M.mean(axis=0)))

    def cfg(j):
        d = dict(zip(TARGET_KEYS, keys[cols[j]]))
        return (f"L2={d['l2_core0']}/{d['l2_core1']}  L3={d['l3']}  "
                f"BTB={d['btb_core0']}/{d['btb_core1']}  "
                f"PF={d['prefetch_core0']}/{d['prefetch_core1']}")

    log('')
    log(f'    {"":34}{"mean":>9}{"median":>9}{"p90":>9}{"p99":>9}{"MAX":>9}{"optimal":>9}')
    log(f'    {"":34}{"loss":>9}{"loss":>9}{"loss":>9}{"loss":>9}{"loss":>9}{"in":>9}')
    log('    ' + '-' * 88)
    stats = {}
    for label, j in picks:
        L = loss[:, j]
        log(f'    {label:<34}{L.mean():>8.2f}%{np.median(L):>8.2f}%'
            f'{np.percentile(L,90):>8.2f}%{np.percentile(L,99):>8.2f}%'
            f'{L.max():>8.2f}%{(L<1e-9).mean()*100:>8.1f}%')
        stats[label.strip()] = L
    log('')
    for label, j in picks:
        log(f'    {label.split("(")[0].strip():<14} {cfg(j)}')
    if j_ppw != rank[0]:
        log(f'    (the config with the highest mean PPW is a different one: {cfg(j_ppw)}, '
            f'mean loss {mean_loss[j_ppw]:.2f}% -- reported figures use the '
            f'lowest-mean-loss config, which is the more generous static baseline)')

    # how bad it gets -- distribution for the strongest static baseline
    Lb = stats[picks[0][0].strip()]
    log('')
    log(f'    Per-interval loss distribution for the BEST static config '
        f'({T:,} intervals):')
    for lo, hi, lbl in [(None, 1e-9, '  0%      (matches optimal)'),
                        (1e-9, 1,    '  0-1%                     '),
                        (1,    5,    '  1-5%                     '),
                        (5,    10,   '  5-10%                    '),
                        (10,   20,   ' 10-20%                    '),
                        (20,   50,   ' 20-50%                    '),
                        (50,   None, ' >50%                      ')]:
        c = (Lb <= hi).sum() if lo is None else (Lb > lo).sum() if hi is None \
            else ((Lb > lo) & (Lb <= hi)).sum()
        log(f'      {lbl}  {c:>8,}  ({c/T*100:5.1f}%)')

    worst_t = int(np.argmax(Lb))
    log('')
    log(f'    Worst single interval for the best static config: '
        f'{Lb[worst_t]:.2f}% below optimal')
    log(f'    Across ALL {N:,} static configs, mean loss ranges '
        f'{loss.mean(axis=0).min():.2f}% (best) .. {loss.mean(axis=0).max():.2f}% (worst)')

    return {
        'benchmark': bench, 'intervals': T, 'configs': N,
        'best_mean_loss': Lb.mean(), 'best_median_loss': float(np.median(Lb)),
        'best_p90_loss': float(np.percentile(Lb, 90)),
        'best_p99_loss': float(np.percentile(Lb, 99)), 'best_max_loss': Lb.max(),
        'best_frac_optimal_pct': float((Lb < 1e-9).mean() * 100),
        'median_cfg_mean_loss': stats['MEDIAN static (blind pick)'].mean(),
        'worst_mean_loss': stats['WORST static'].mean(),
        'worst_max_loss': stats['WORST static'].max(),
    }


def main():
    ap = argparse.ArgumentParser(description='Static config vs per-interval optimal PPW loss')
    ap.add_argument('--merged-full-dir', required=True)
    ap.add_argument('--benchmarks', nargs='+',
                    default=['barnes', 'cholesky', 'fft', 'radiosity'])
    ap.add_argument('--out', default='static_vs_optimal.txt')
    ap.add_argument('--chunksize', type=int, default=200_000)
    a = ap.parse_args()

    lines = []

    def log(m=''):
        print(m)
        lines.append(str(m))

    log('=' * 96)
    log('  STATIC CONFIGURATION vs PER-INTERVAL OPTIMAL  --  PPW % LOSS')
    log('=' * 96)

    res, bad = [], []
    for b in a.benchmarks:
        p = os.path.join(a.merged_full_dir, f'merged_full_{b}.csv')
        log('')
        log(f'  {b.upper()}')
        if not os.path.exists(p):
            log(f'    [SKIP] not found: {p}')
            bad.append(b)
            continue
        try:
            mat, keys = load_panel(p, b, a.chunksize, log)
            res.append(report(b, mat, keys, log))
        except Exception as e:
            log(f'    [FAIL] {type(e).__name__}: {e}')
            bad.append(b)

    if res:
        log('')
        log('=' * 96)
        log('  SUMMARY -- PPW % loss vs optimal (higher = static is worse)')
        log('=' * 96)
        log(f'  {"bench":<11}{"intervals":>11}{"cfgs":>7}'
            f'{"best mean":>11}{"best med":>10}{"best p99":>10}{"best MAX":>10}'
            f'{"blind mean":>12}{"worst mean":>12}')
        log('  ' + '-' * 92)
        for r in res:
            log(f'  {r["benchmark"]:<11}{r["intervals"]:>11,}{r["configs"]:>7,}'
                f'{r["best_mean_loss"]:>10.2f}%{r["best_median_loss"]:>9.2f}%'
                f'{r["best_p99_loss"]:>9.2f}%{r["best_max_loss"]:>9.2f}%'
                f'{r["median_cfg_mean_loss"]:>11.2f}%{r["worst_mean_loss"]:>11.2f}%')
        log('')
        log('  Sentences you can lift:')
        for r in res:
            log(f'    {r["benchmark"]:<10} even the best single static configuration loses '
                f'{r["best_mean_loss"]:.1f}% PPW on average vs per-interval optimal '
                f'(median {r["best_median_loss"]:.1f}%, up to {r["best_max_loss"]:.1f}% in the '
                f'worst interval); it is optimal in only {r["best_frac_optimal_pct"]:.1f}% of intervals.')
        pd.DataFrame(res).to_csv(a.out.replace('.txt', '.csv'), index=False)
        log('')
        log(f'  csv -> {a.out.replace(".txt", ".csv")}')

    with open(a.out, 'w') as f:
        f.write('\n'.join(lines))
    print(f'\nreport -> {a.out}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
