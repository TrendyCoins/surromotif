"""Rule R7 (docs/rules.md): warm-up of the line from the first bar of a chart.

A chart that starts at bar X0 is simulated by slicing the bars from X0 (norm, EMA, moves start from scratch).
Reference: the same windows computed on the whole file (>= 70 days of history before any X0: EMA converged).
For a burn-in B (bars from the first valid r), windows with T - first_r in [B, B + 2016) are compared;
B* = the smallest B where the 99th percentile of |dlogit| <= 0.01, for two EMA starts (adjust=False / True).

    python scripts/legacy/warmup.py [--starts 24] [--out docs/checks/legacy/f3g/warmup.json]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mri import BARS_DAY, L  # noqa: E402
from mri.batch import infer  # noqa: E402
from mri.features import bar_features, read_bars  # noqa: E402
from mri.params import load_npz  # noqa: E402

CTX_MIN = L + 1 + 7 * BARS_DAY       # T - first_r needed for the full week of moves before the window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--starts', type=int, default=24)
    ap.add_argument('--npz', default=os.path.join(ROOT, 'artifacts/weights/prod_F3_g_L64_a_slim.npz'))
    ap.add_argument('--floor', type=float, default=0.17530229687690735)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    bars = read_bars(os.path.join(ROOT, 'artifacts/bars/btc_old_tail.csv.gz'))
    p, _ = load_npz(a.npz)
    full = infer(bar_features(bars), p, a.floor)
    full_logit = np.full(len(bars), np.nan)
    full_logit[full['T']] = full['logit']
    x_lo = 70 * BARS_DAY                                    # >= 70 days of history behind every simulated chart start
    x_hi = len(bars) - 16000
    X0s = np.linspace(x_lo, x_hi, a.starts).astype(int)
    grid = np.arange(CTX_MIN, 14001, 100)
    res = {'grid': grid.tolist(), 'ctx_min': CTX_MIN, 'starts': X0s.tolist()}
    for adjust in (False, True):
        rel = {B: [] for B in grid}
        for X0 in X0s:
            sub = bars.iloc[X0:X0 + 16000].reset_index(drop=True)
            f = bar_features(sub, ema_adjust=adjust)
            r = infer(f, p, a.floor)
            age = r['T'] - f['first_r']
            d = np.abs(r['logit'] - full_logit[r['T'] + X0])
            for B in grid:
                m = (age >= B) & (age < B + 2016)
                rel[B].append(d[m])
        p99 = [float(np.percentile(np.concatenate(rel[B]), 99)) for B in grid]
        p50 = [float(np.percentile(np.concatenate(rel[B]), 50)) for B in grid]
        ok = [B for B, v in zip(grid, p99) if v <= 0.01]
        key = 'adjust' if adjust else 'pandas'
        res[key] = {'p99': p99, 'p50': p50, 'B_star': int(ok[0]) if ok else None}
        print(key, 'B* =', res[key]['B_star'], '| p99 at', [(int(B), round(v, 4)) for B, v in zip(grid, p99)][::10])
    if a.out:
        with open(a.out, 'w') as fh:
            json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
