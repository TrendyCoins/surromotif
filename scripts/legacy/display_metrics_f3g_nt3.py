"""Rule R5 for the legacy models (docs/rules.md): how jumpy the line M of F3_g (sessions 1-2, global Platt) and of
NT3_PRV_ch48 (session 3, Platt of the asset) was, without smoothing. The metrics are those of scripts/display_metrics.py.

    python scripts/legacy/display_metrics_f3g_nt3.py [--model F3_g | NT3_PRV_ch48] [--out ...]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from display_metrics import metrics  # noqa: E402
from mri.legacy.calibration import load_part, load_val  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='F3_g', choices=['F3_g', 'NT3_PRV_ch48'])
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    nt = a.model != 'F3_g'
    out = a.out or os.path.join(ROOT, 'docs/checks/legacy/nt3/display.json' if nt else 'docs/checks/legacy/f3g/display.json')
    with open(os.path.join(ROOT, 'export/legacy/mri_model_nt3.json' if nt else 'export/legacy/mri_model.json'), encoding='utf-8') as fh:
        ex = json.load(fh)
    if nt:
        platt = {k: (v['platt_a'], v['platt_b']) for k, v in ex['calibration']['per_symbol'].items()}
    else:
        platt = {None: (ex['calibration']['platt_a'], ex['calibration']['platt_b'])}
    pa, pb = platt.get('BTCUSDT', platt.get(None))
    g = pd.read_csv(os.path.join(ROOT, 'tests/golden/legacy/btc5m_fresh_nt3.csv.gz' if nt else 'tests/golden/legacy/btc5m_fresh.csv.gz'))
    ll = np.where(g['shown'] > 0, pa * g['logit'] + pb, np.nan)
    res = {'model': a.model, 'platt': {k or 'all': v for k, v in platt.items()}, 'btc_fresh_prod_a': metrics(ll),
           'val_refit': {}}
    print('BTC fresh (prod a):', json.dumps(res['btc_fresh_prod_a']))
    if nt:                                    # reference (R11): share of shown bars of fresh BTC at the ceiling of BTC
        c = ex['display']['ceiling']['by_symbol']['BTCUSDT']
        lv = ll[np.isfinite(ll)]
        res['btc_fresh_at_ceiling'] = {'ceiling': c, 'more': float(np.mean(lv >= np.log(c['more']))),
                                       'less': float(np.mean(-lv >= np.log(c['less'])))}
        print('BTC fresh at the ceiling:', res['btc_fresh_at_ceiling'])
    rows = []
    for s in ('a', 'b'):
        val = (load_part(os.path.join(ROOT, 'artifacts/calib'), f'pool_ALL_{a.model}_L64_{s}_refit') if nt
               else load_val(os.path.join(ROOT, 'artifacts/calib'), s))
        for asset, (sp, _, _) in val.items():
            pa, pb = platt.get(asset, platt.get(None))
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
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=1)



if __name__ == '__main__':
    main()
