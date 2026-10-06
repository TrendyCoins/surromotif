"""Rule R11 (docs/rules.md): per-asset ceilings of the line M (one-sided: factual / expected >= 0.8), at most the pool
ceiling of R9; unknown symbol = the lowest of FIL, LINK, XLM. Reference: share of bars where the line hits the ceiling
(fresh BTC golden file; VAL 'refit' per asset with the exported Platt).

    python scripts/ceilings.py [--out docs/checks/ceilings.json]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from calib_curve import EDGES, MIN_N, SEEDS, curve, oof_logs  # noqa: E402
from mri.calibration import load_val  # noqa: E402

MIN_RATIO = 0.8
H_ASSETS = ('FILUSDT', 'LINKUSDT', 'XLMUSDT')


def asset_ceiling(rows, color, pool_ceiling):
    for r in rows:
        if r['color'] != color:
            continue
        ok = (not r['sparse']) and r.get('ratio', 0.0) >= MIN_RATIO
        if not ok:
            return min(r['lo'], pool_ceiling)
    return pool_ceiling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs/checks/ceilings.json'))
    a = ap.parse_args()
    with open(os.path.join(ROOT, 'docs/checks/calib_curve.json')) as fh:
        pool = json.load(fh)['ceiling_pool']
    with open(os.path.join(ROOT, 'export/mri_model.json'), encoding='utf-8') as fh:
        ex = json.load(fh)
    pa, pb = ex['calibration']['platt_a'], ex['calibration']['platt_b']
    val = {s: load_val(os.path.join(ROOT, 'artifacts/calib'), s) for s in SEEDS}
    table, bins = {}, {}
    for asset in sorted(val['a']):
        lp, ln, _, _ = oof_logs(val, (asset,))
        rows = curve(lp, ln)
        bins[asset] = rows
        table[asset] = {'more': asset_ceiling(rows, 'orange', pool['orange']),
                        'less': asset_ceiling(rows, 'blue', pool['blue'])}
    unknown = {k: min(table[h][k] for h in H_ASSETS) for k in ('more', 'less')}
    # reference: share of bars at the ceiling (exported Platt)
    hits = {}
    for asset in sorted(val['a']):
        ll = np.concatenate([pa * val[s][asset][0] + pb for s in SEEDS])
        c = table[asset]
        hits[asset] = {'more': float(np.mean(ll >= np.log(c['more']))), 'less': float(np.mean(-ll >= np.log(c['less'])))}
    g = pd.read_csv(os.path.join(ROOT, 'tests/golden/btc5m_fresh.csv.gz'))
    ll = (pa * g['logit'] + pb)[g['shown'] > 0].to_numpy()
    c = table['BTCUSDT']
    btc = {'bars': int(len(ll)), 'more': float(np.mean(ll >= np.log(c['more']))),
           'less': float(np.mean(-ll >= np.log(c['less'])))}
    res = {'rule': 'R11', 'min_ratio': MIN_RATIO, 'min_n': MIN_N, 'pool_ceiling': pool, 'edges': EDGES[:-1],
           'ceilings': table, 'unknown_symbol': unknown, 'share_at_ceiling_val': hits, 'share_at_ceiling_btc_fresh': btc,
           'bins': {k: [{kk: r.get(kk) for kk in ('color', 'lo', 'n_real', 'n_surr', 'ratio', 'sparse')} for r in v]
                    for k, v in bins.items()}}
    print(f"{'asset':<10} {'more':>5} {'less':>5} | at ceiling VAL: more {'':>2} less")
    for k, v in table.items():
        h = hits[k]
        print(f"{k:<10} {v['more']:>5g} {v['less']:>5g} | {100 * h['more']:6.2f}% {100 * h['less']:6.2f}%"
              + ('   (H)' if k in H_ASSETS else ''))
    print('unknown symbol:', unknown, '| BTC fresh at ceiling:', btc)
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else x)


if __name__ == '__main__':
    main()
