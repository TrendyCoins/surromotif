"""Rule R4 (docs/rules.md): global Platt on the VAL of the 'refit' models (seeds a, b, all 31 assets) and its check by
half-years (fit on one half, check on the other), per seed; K1 (pool), K2 (asset x half x seed cells), K3 (H assets).

    python scripts/legacy/calibrate.py [--out docs/checks/legacy/f3g/calibration.json]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mri.legacy.calibration import THR, WEEK_NS, calib_ratio, load_val, mark_lift, platt_fit, pooled  # noqa: E402

H_ASSETS = ('FILUSDT', 'LINKUSDT', 'XLMUSDT')
SEEDS = ('a', 'b')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs/checks/legacy/f3g/calibration.json'))
    a = ap.parse_args()
    root = os.path.join(ROOT, 'artifacts/calib')
    val = {s: load_val(root, s) for s in SEEDS}
    assets = sorted(val['a'])
    wk_cut = pd.Timestamp('2026-01-01').value // WEEK_NS          # 2026-01-01 is a Thursday: a week boundary
    halves = {'H1': lambda wk: wk < wk_cut, 'H2': lambda wk: wk >= wk_cut}
    res = {'assets': len(assets), 'thr': THR, 'wk_cut': int(wk_cut)}

    def fit(half=None, seeds=SEEDS):
        sp, sn = zip(*[pooled(val[s], None, halves.get(half)) for s in seeds])
        return platt_fit(np.concatenate(sp), np.concatenate(sn))

    final = fit()
    res['platt_final'] = {'a': final[0], 'b': final[1], 'fit_on': 'VAL refit a+b, 31 assets, both halves'}
    res['platt_by_seed'] = {s: fit(seeds=(s,)) for s in SEEDS}
    res['platt_by_half'] = {h: fit(h) for h in halves}
    print('final Platt a, b =', final, '| by seed', res['platt_by_seed'], '| by half', res['platt_by_half'])

    k1, k3, cells = [], [], []
    for h, other in (('H1', 'H2'), ('H2', 'H1')):
        pa, pb = res['platt_by_half'][other]                      # out of fold: fitted on the other half
        for s in SEEDS:
            sp, sn = pooled(val[s], None, halves[h])
            m = mark_lift(sp, sn, pa, pb)
            own = platt_fit(sp, sn)
            k1.append({'half': h, 'seed': s, 'orange': m['orange']['lift'], 'blue': m['blue']['lift'],
                       'band_orange': m['band_orange']['lift'], 'band_blue': m['band_blue']['lift'],
                       'marked_real': m['orange']['share_real'] + m['blue']['share_real'],
                       'marked_surr': m['orange']['share_surr'] + m['blue']['share_surr'],
                       'transfer': [calib_ratio((pa, pb), own, THR), calib_ratio((pa, pb), own, 1 / THR)]})
            sp, sn = pooled(val[s], H_ASSETS, halves[h])
            m = mark_lift(sp, sn, pa, pb)
            k3.append({'half': h, 'seed': s, 'orange': m['orange']['lift'], 'blue': m['blue']['lift'],
                       'band_orange': m['band_orange']['lift'], 'band_blue': m['band_blue']['lift']})
            for asset in assets:
                p_, n_, wk = val[s][asset]
                mk = halves[h](wk)
                if mk.sum() < 5000:
                    continue
                m = mark_lift(p_[mk], n_[mk], pa, pb, wk[mk], ci=True)
                cells.append({'asset': asset, 'half': h, 'seed': s, 'n': int(mk.sum()),
                              'orange': m['orange']['lift'], 'orange_lo': m['orange']['lo'],
                              'orange_hi': m['orange']['hi'], 'blue': m['blue']['lift'], 'blue_lo': m['blue']['lo'],
                              'blue_hi': m['blue']['hi'], 'marked_real': m['orange']['share_real']
                              + m['blue']['share_real']})
    k1_ok = all(r['orange'] >= THR and r['blue'] <= 1 / THR for r in k1)
    k3_ok = all(r['orange'] >= THR and r['blue'] <= 1 / THR for r in k3)
    fail_o = [c for c in cells if c['orange_hi'] < THR]
    fail_b = [c for c in cells if c['blue_lo'] > 1 / THR]
    k2_ok = len(fail_o) <= 0.05 * len(cells) and len(fail_b) <= 0.05 * len(cells)
    res.update({'K1': {'ok': k1_ok, 'rows': k1}, 'K3': {'ok': k3_ok, 'rows': k3},
                'K2': {'ok': k2_ok, 'cells': len(cells), 'fail_orange': fail_o, 'fail_blue': fail_b,
                       'point_below_orange': sum(c['orange'] < THR for c in cells),
                       'point_above_blue': sum(c['blue'] > 1 / THR for c in cells)},
                'cells': cells})
    for r in k1:
        print('K1', r['half'], r['seed'], f"orange {r['orange']:.3f} blue {r['blue']:.3f} | band {r['band_orange']:.3f} / "
              f"{r['band_blue']:.3f} | marked real {100 * r['marked_real']:.1f}% surr {100 * r['marked_surr']:.1f}% | "
              f"transfer x{r['transfer'][0]:.3f} / x{r['transfer'][1]:.3f}")
    for r in k3:
        print('K3', r['half'], r['seed'], f"orange {r['orange']:.3f} blue {r['blue']:.3f} | band {r['band_orange']:.3f} / "
              f"{r['band_blue']:.3f}")
    print(f"K1 {'OK' if k1_ok else 'FAIL'} | K3 {'OK' if k3_ok else 'FAIL'} | K2 {'OK' if k2_ok else 'FAIL'}: cells "
          f"{len(cells)}, significant fails orange {len(fail_o)}, blue {len(fail_b)}; point estimates on the wrong side: "
          f"orange {res['K2']['point_below_orange']}, blue {res['K2']['point_above_blue']}")
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
