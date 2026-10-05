"""Rule R5 (docs/rules.md): how jumpy the line M is, without smoothing.

Series: fresh BTC (golden file: prod a + Platt, shown bars) and VAL of the 'refit' models (seeds a, b) per asset
(consecutive windows in time order; the final Platt of R4 — in sample for the Platt, descriptive only).

    python scripts/display_metrics.py [--out docs/checks/display.json]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mri import BARS_DAY  # noqa: E402
from mri.calibration import THR, load_val  # noqa: E402


def segments(ll):
    """Split a log-lift series by NaN (line breaks) into finite runs."""
    fin = np.isfinite(ll)
    idx = np.where(np.diff(np.concatenate([[0], fin.astype(int), [0]])))[0].reshape(-1, 2)
    return [ll[a:b] for a, b in idx]


def runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return np.where(d == -1)[0] - np.where(d == 1)[0]


def metrics(ll):
    segs = [s for s in segments(np.asarray(ll, dtype=float)) if len(s) > 1]
    t = math.log(THR)
    dlm, flips, mflips, bars, run_l, above = [], 0, 0, 0, [], 0
    for s in segs:
        lm = np.abs(s)                                    # log M
        dlm.append(np.abs(np.diff(lm)))
        col = np.sign(s)
        flips += int((col[1:] * col[:-1] < 0).sum())
        mk = lm >= t
        mc = col[mk]
        mflips += int((mc[1:] != mc[:-1]).sum())
        run_l.append(runs(mk))
        above += int(mk.sum())
        bars += len(s)
    days = bars / BARS_DAY
    rl = np.concatenate(run_l) if run_l else np.array([])
    out = {'bars': bars, 'days': round(days, 1), 'median_abs_dlogM': float(np.median(np.concatenate(dlm))),
           'color_flips_per_day': flips / days, 'marked_color_flips_per_day': mflips / days,
           'mean_run_marked_bars': float(rl.mean()) if len(rl) else 0.0,
           'median_run_marked_bars': float(np.median(rl)) if len(rl) else 0.0,
           'share_marked': above / bars,
           'share_orange': float(sum(int((s >= t).sum()) for s in segs) / bars),
           'share_blue': float(sum(int((s <= -t).sum()) for s in segs) / bars)}
    out['jumpy'] = bool(out['mean_run_marked_bars'] < 3 or out['marked_color_flips_per_day'] > 24)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs/checks/display.json'))
    a = ap.parse_args()
    with open(os.path.join(ROOT, 'export/mri_model.json'), encoding='utf-8') as fh:
        ex = json.load(fh)
    pa, pb = ex['calibration']['platt_a'], ex['calibration']['platt_b']
    g = pd.read_csv(os.path.join(ROOT, 'tests/golden/btc5m_fresh.csv.gz'))
    ll = np.where(g['shown'] > 0, pa * g['logit'] + pb, np.nan)
    res = {'platt': [pa, pb], 'btc_fresh_prod_a': metrics(ll), 'val_refit': {}}
    print('BTC fresh (prod a):', json.dumps(res['btc_fresh_prod_a']))
    rows = []
    for s in ('a', 'b'):
        val = load_val(os.path.join(ROOT, 'artifacts/calib'), s)
        for asset, (sp, _, _) in val.items():
            m = metrics(pa * sp + pb)
            res['val_refit'][f'{asset}_{s}'] = m
            rows.append({'asset': asset, 'seed': s, **m})
    df = pd.DataFrame(rows)
    cols = ['median_abs_dlogM', 'color_flips_per_day', 'marked_color_flips_per_day', 'mean_run_marked_bars',
            'share_marked', 'share_orange', 'share_blue']
    res['val_refit_summary'] = {c: {'median': float(df[c].median()), 'min': float(df[c].min()), 'max': float(df[c].max())}
                                for c in cols}
    res['val_refit_jumpy'] = int(df['jumpy'].sum())
    print(df.groupby('asset')[cols].mean().round(3).to_string())
    print('summary', json.dumps(res['val_refit_summary']), '| jumpy cells', res['val_refit_jumpy'], 'of', len(df))
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
