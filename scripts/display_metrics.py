"""Rule R5 (docs/rules.md): how jumpy the line M is, without smoothing.

Series: fresh BTC (golden file: prod a + Platt, shown bars) and VAL of the 'refit' models (seeds a, b) per asset
(consecutive windows in time order; the final Platt of R4 — in sample for the Platt, descriptive only).

    python scripts/display_metrics.py [--out docs/checks/display.json]
    python scripts/display_metrics.py --model NT3_PRV_ch48      # NT3 (R5 amendment of session 3): fresh BTC from the NT3
        golden file with the Platt of BTC; VAL 'refit' on ALL (seeds a, b) per asset with the export Platt of that asset
    python scripts/display_metrics.py --model PB_s64h24_d1      # PB (R5 amendment of session 4): log M of the line M
        (colour = its sign, M = exp(|log M|) without the ceiling); fresh BTC from the PB golden file (prod a, ruler of prod
        a, live part from the start of the tail, shown windows); VAL 'refit' on ALL (seeds a, b) per asset as in R15 (ruler
        of the other half-year, live part over VAL of the asset, windows numbered >= 288), windows in time order
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
from mri import BARS_DAY, linem  # noqa: E402
from mri.calibration import THR, load_part, load_val  # noqa: E402


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


def val_line_pb(seed):
    """log M of the VAL windows of 'refit' on ALL (one seed) per asset, as R15: {asset: log M in time order}."""
    from mri import linem_check as LC
    from mri.export_pb import nocal_path
    with open(nocal_path(), encoding='utf-8') as fh:
        nc = json.load(fh)
    knots, zk = np.asarray(nc['knots'], dtype=np.float64), np.asarray(nc['zk'], dtype=np.float64)
    D = load_part(os.path.join(ROOT, 'artifacts/calib'), f'pool_ALL_PB_s64h24_d1_L64_{seed}_refit')
    D = {a: {'sp': sp, 'sn': sn, 'wk': wk} for a, (sp, sn, wk) in D.items()}
    R = LC.rulers({seed: D}, knots)
    out = {}
    for a, d in D.items():
        h1 = LC.half1(d['wk'])
        lm = np.full(len(d['sp']), np.nan)
        for h in (True, False):
            zr = linem.ruler_z(d['sp'], R[(seed, h)], zk)
            m, sd = linem.live(zr)
            ok = (h1 == h) & (np.arange(len(zr)) >= linem.WARM)
            lm[ok] = linem.log_m(zr[ok], m[ok], sd[ok])
        out[a] = lm[np.isfinite(lm)]
    return out


def main_pb(out):
    from mri.export_pb import load_export
    ex = load_export()
    g = pd.read_csv(os.path.join(ROOT, 'tests/golden/btc5m_fresh_pb.csv.gz'))
    lm = np.where(g['shown'] > 0, g['log_m'], np.nan)
    res = {'model': 'PB_s64h24_d1', 'line': 'log M of the line M (R15), no ceiling', 'btc_fresh_prod_a': metrics(lm),
           'val_refit': {}}
    print('BTC fresh (prod a):', json.dumps(res['btc_fresh_prod_a']))
    c = ex['line']['ceilings']['by_symbol']['BTCUSDT']
    lv = lm[np.isfinite(lm)]
    res['btc_fresh_above_ceiling'] = {'ceiling': c, 'orange': float(np.mean(lv > np.log(c['orange']))),
                                      'blue': float(np.mean(-lv > np.log(c['blue'])))}
    print('BTC fresh above the ceiling:', res['btc_fresh_above_ceiling'])
    rows = []
    for s in ('a', 'b'):
        for asset, v in val_line_pb(s).items():
            m = metrics(v)
            cc = ex['line']['ceilings']['by_symbol'][asset]
            m.update({'above_ceiling_orange': float(np.mean(v > np.log(cc['orange']))),
                      'above_ceiling_blue': float(np.mean(-v > np.log(cc['blue'])))})
            res['val_refit'][f'{asset}_{s}'] = m
            rows.append({'asset': asset, 'seed': s, **m})
    df = pd.DataFrame(rows)
    cols = ['median_abs_dlogM', 'color_flips_per_day', 'marked_color_flips_per_day', 'mean_run_marked_bars',
            'share_marked', 'share_orange', 'share_blue', 'above_ceiling_orange', 'above_ceiling_blue']
    res['val_refit_summary'] = {c: {'median': float(df[c].median()), 'min': float(df[c].min()), 'max': float(df[c].max())}
                                for c in cols}
    res['val_refit_jumpy'] = int(df['jumpy'].sum())
    res['val_refit_cells'] = int(len(df))
    print(df.groupby('asset')[cols].mean().round(3).to_string())
    print('summary', json.dumps(res['val_refit_summary']), '| jumpy cells', res['val_refit_jumpy'], 'of', len(df))
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='F3_g', choices=['F3_g', 'NT3_PRV_ch48', 'PB_s64h24_d1'])
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.model == 'PB_s64h24_d1':
        return main_pb(a.out or os.path.join(ROOT, 'docs/checks/pb/display.json'))
    nt = a.model != 'F3_g'
    out = a.out or os.path.join(ROOT, 'docs/checks/nt3/display.json' if nt else 'docs/checks/display.json')
    with open(os.path.join(ROOT, 'export/mri_model_nt3.json' if nt else 'export/mri_model.json'), encoding='utf-8') as fh:
        ex = json.load(fh)
    if nt:
        platt = {k: (v['platt_a'], v['platt_b']) for k, v in ex['calibration']['per_symbol'].items()}
    else:
        platt = {None: (ex['calibration']['platt_a'], ex['calibration']['platt_b'])}
    pa, pb = platt.get('BTCUSDT', platt.get(None))
    g = pd.read_csv(os.path.join(ROOT, 'tests/golden/btc5m_fresh_nt3.csv.gz' if nt else 'tests/golden/btc5m_fresh.csv.gz'))
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
