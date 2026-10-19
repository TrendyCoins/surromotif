"""Rule R9 (docs/rules.md): calibration curve of the line M on the VAL of the 'refit' models, out-of-fold Platt by
half-years; bins by M per color; factual / expected multiple; ceiling per color (pool of 31 assets).

    python scripts/legacy/calib_curve.py [--out docs/checks/legacy/f3g/calib_curve.json]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mri.legacy.calibration import WEEK_NS, load_val, platt_fit, pooled  # noqa: E402

EDGES = [1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, np.inf]
TOL = (0.8, 1.25)
MIN_N = 1000
H_ASSETS = ('FILUSDT', 'LINKUSDT', 'XLMUSDT')
SEEDS = ('a', 'b')


def oof_logs(val, assets=None):
    """Out-of-fold log lift of real and surrogate windows (+ weeks), all seeds and both halves together."""
    cut = pd.Timestamp('2026-01-01').value // WEEK_NS
    halves = {'H1': lambda wk: wk < cut, 'H2': lambda wk: wk >= cut}
    platt = {}
    for h in halves:
        sp, sn = zip(*[pooled(val[s], None, halves[h]) for s in SEEDS])
        platt[h] = platt_fit(np.concatenate(sp), np.concatenate(sn))
    lp, ln, wp = [], [], []
    for h, other in (('H1', 'H2'), ('H2', 'H1')):
        a, b = platt[other]
        for s in SEEDS:
            for asset, (sp, sn, wk) in val[s].items():
                if assets is not None and asset not in assets:
                    continue
                m = halves[h](wk)
                lp.append(a * sp[m] + b)
                ln.append(a * sn[m] + b)
                wp.append(wk[m] * 10 + (s == 'b'))          # week x seed as the bootstrap unit
    return np.concatenate(lp), np.concatenate(ln), np.concatenate(wp), platt


def curve(lp, ln, wk=None, boot=0, seed=0):
    rows = []
    rng = np.random.default_rng(seed)
    if boot:
        weeks, idx = np.unique(wk, return_inverse=True)
    for color in ('orange', 'blue'):
        sg = 1.0 if color == 'orange' else -1.0
        xp, xn = sg * lp, sg * ln                             # log M on this color's side (> 0)
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            bp = (xp > 0) & (xp >= np.log(lo)) & (xp < np.log(hi))
            bn = (xn > 0) & (xn >= np.log(lo)) & (xn < np.log(hi))
            n_p, n_n = int(bp.sum()), int(bn.sum())
            row = {'color': color, 'lo': lo, 'hi': hi, 'n_real': n_p, 'n_surr': n_n}
            if n_p and n_n:
                if color == 'orange':
                    fact = (n_p / len(lp)) / (n_n / len(ln))
                    exp_ = float(np.exp(xn[bn]).mean())
                else:
                    fact = (n_n / len(ln)) / (n_p / len(lp))
                    exp_ = float(np.exp(xp[bp]).mean())
                row.update({'factual': fact, 'expected': exp_, 'ratio': fact / exp_})
                if boot:
                    # paired real / surrogate windows share the week: resample weeks
                    cp = np.bincount(idx, weights=bp, minlength=len(weeks))
                    cn = np.bincount(idx, weights=bn, minlength=len(weeks))
                    tot = np.bincount(idx, minlength=len(weeks))
                    ev = np.bincount(idx, weights=np.where(bn, np.exp(xn), 0.0) if color == 'orange'
                                     else np.where(bp, np.exp(xp), 0.0), minlength=len(weeks))
                    r = []
                    for _ in range(boot):
                        k = rng.integers(0, len(weeks), len(weeks))
                        P, N, T = cp[k].sum(), cn[k].sum(), tot[k].sum()
                        if color == 'orange':
                            f, e = (P / T) / (N / T), ev[k].sum() / max(N, 1)
                        else:
                            f, e = (N / T) / (P / T), ev[k].sum() / max(P, 1)
                        r.append(f / e)
                    row.update({'ratio_lo': float(np.percentile(r, 2.5)), 'ratio_hi': float(np.percentile(r, 97.5))})
            row['sparse'] = n_p < MIN_N or n_n < MIN_N
            row['pass'] = (not row['sparse']) and TOL[0] <= row.get('ratio', np.nan) <= TOL[1]
            rows.append(row)
    return rows


def ceilings(rows):
    out = {}
    for color in ('orange', 'blue'):
        bad = [r['lo'] for r in rows if r['color'] == color and not r['pass']]
        out[color] = min(bad) if bad else None
    return out


def show(title, rows):
    print(title)
    for r in rows:
        rng = f"[{r['lo']:g}; {r['hi']:g})"
        ci = f" [{r['ratio_lo']:.2f}; {r['ratio_hi']:.2f}]" if 'ratio_lo' in r else ''
        print(f"  {r['color']:<6} {rng:<12} real {r['n_real']:>8} surr {r['n_surr']:>8} | fact {r.get('factual', np.nan):8.3f} "
              f"exp {r.get('expected', np.nan):8.3f} ratio {r.get('ratio', np.nan):6.3f}{ci} "
              f"{'ok' if r['pass'] else ('SPARSE' if r['sparse'] else 'FAIL')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs/checks/legacy/f3g/calib_curve.json'))
    ap.add_argument('--boot', type=int, default=200)
    a = ap.parse_args()
    val = {s: load_val(os.path.join(ROOT, 'artifacts/calib'), s) for s in SEEDS}
    lp, ln, wk, platt = oof_logs(val)
    pool = curve(lp, ln, wk, boot=a.boot)
    show('POOL (31 assets, rule)', pool)
    lp_h, ln_h, wk_h, _ = oof_logs(val, H_ASSETS)
    hold = curve(lp_h, ln_h, wk_h, boot=a.boot)
    show('H (FIL, LINK, XLM, report)', hold)
    per_asset = {}
    for asset in sorted(val['a']):
        lpa, lna, _, _ = oof_logs(val, (asset,))
        per_asset[asset] = curve(lpa, lna)
    res = {'edges': [e if np.isfinite(e) else None for e in EDGES], 'tol': TOL, 'min_n': MIN_N, 'platt_oof': platt,
           'pool': pool, 'ceiling_pool': ceilings(pool), 'holdout': hold, 'ceiling_holdout': ceilings(hold),
           'per_asset': per_asset,
           'per_asset_fails': {k: [f"{r['color']} {r['lo']:g}" for r in v if not r['pass'] and not r['sparse']]
                               for k, v in per_asset.items()}}
    print('ceiling pool:', res['ceiling_pool'], '| ceiling H (report):', res['ceiling_holdout'])
    print('per asset fails (not sparse):', {k: v for k, v in res['per_asset_fails'].items() if v})
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else x)


if __name__ == '__main__':
    main()
