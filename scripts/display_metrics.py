"""Rule R5 (docs/rules.md, amendment of session 5): how jumpy the line M is, without smoothing (reference only).

Log M of the line M of session 5 (colour = its sign, M = exp(|log M|) without the ceiling):
    fresh BTC — the golden file (prod a, ruler R19, formula A from the start of the tail, shown windows from 4360);
    VAL 'refit' on ALL (seeds a, b) per asset as the count (Ц) of SuMo_Class_Ceil: scores with the floor 0.2 (the Colab
        export), cross ruler from the surrogates of the other half (П1 / П2 by 2026-01-01, without FIL, LINK, XLM),
        formula A over the windows of the asset in a row, windows from position 4096, in time order;
plus the share of windows above the ceiling 10. Metrics: median |d log M| bar to bar, colour changes per day (all and
marked, M >= 1.25), mean run of marked bars, shares; 'jumpy' — mean run < 3 bars or > 24 marked colour changes a day.

    python scripts/display_metrics.py [--out docs/checks/pb5/display.json]

The legacy models (F3_g, NT3_PRV_ch48): scripts/legacy/display_metrics_f3g_nt3.py.
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

THR = 1.25                         # marked: M >= 1.25 (rule R5)
HALF_CUT = '2026-01-01'            # VAL half-years П1 / П2 (research _nc_half)
WEEK_NS = 7 * 86400 * 10 ** 9


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


def val_line_pb():
    """log M of the VAL windows of 'refit' on ALL per (seed, asset), as the count (Ц) of SuMo_Class_Ceil with the floor 0.2:
       {(seed, asset): log M in time order}."""
    from mri.ceil_export import TAGS, load_export
    from mri.export_pb import nocal_path
    from mri.research5 import H
    with open(nocal_path(), encoding='utf-8') as fh:
        nc = json.load(fh)
    knots, zk = np.asarray(nc['knots'], dtype=np.float64), np.asarray(nc['zk'], dtype=np.float64)
    ex = load_export()
    out = {}
    for s in ('a', 'b'):
        D = {a: {'sp': d['sp'].astype(np.float64), 'sn': d['sn'].astype(np.float64), 'f': (d['wk'] >= pd.Timestamp(HALF_CUT).value // WEEK_NS).astype(int)}
             for a, d in ex[TAGS[f'refit_{s}']].items()}
        Q = {f: np.quantile(np.concatenate([d['sn'][d['f'] != f] for a, d in D.items() if a not in H]), knots)
             for f in (0, 1)}
        for a, d in D.items():
            n = len(d['sp'])
            lm = np.full(n, np.nan)
            for f in (0, 1):
                zr = linem.ruler_z(d['sp'], Q[f], zk)
                m, sd, _ = linem.live_win(zr)
                ok = (d['f'] == f) & (np.arange(n) >= linem.SHOW_FROM)
                lm[ok] = linem.log_m(zr[ok], m[ok], sd[ok])
            out[(s, a)] = lm[np.isfinite(lm)]
    return out


def main_pb(out):
    from mri.export_pb import load_export
    ex = load_export()
    C = ex['line']['ceiling']['value']
    lc = np.log(C)
    g = pd.read_csv(os.path.join(ROOT, 'tests/golden/btc5m_fresh_pb.csv.gz'))
    lm = np.where(g['shown'] > 0, g['log_m'], np.nan)
    res = {'model': 'PB_s64h24_d1', 'line': 'log M of the line M of session 5 (R17-R19), no ceiling', 'ceiling': C,
           'btc_fresh_prod_a': metrics(lm), 'val_refit': {}}
    print('BTC fresh (prod a):', json.dumps(res['btc_fresh_prod_a']))
    lv = lm[np.isfinite(lm)]
    res['btc_fresh_above_ceiling'] = {'orange': float(np.mean(lv > lc)), 'blue': float(np.mean(-lv > lc)),
                                      'shown_bars': int(len(lv))}
    print('BTC fresh above the ceiling 10:', res['btc_fresh_above_ceiling'])
    rows = []
    for (s, asset), v in val_line_pb().items():
        m = metrics(v)
        m.update({'above_ceiling_orange': float(np.mean(v > lc)), 'above_ceiling_blue': float(np.mean(-v > lc))})
        res['val_refit'][f'{asset}_{s}'] = m
        rows.append({'asset': asset, 'seed': s, **m})
    df = pd.DataFrame(rows)
    cols = ['median_abs_dlogM', 'color_flips_per_day', 'marked_color_flips_per_day', 'mean_run_marked_bars',
            'share_marked', 'share_orange', 'share_blue', 'above_ceiling_orange', 'above_ceiling_blue']
    res['val_refit_summary'] = {c: {'median': float(df[c].median()), 'min': float(df[c].min()), 'max': float(df[c].max())}
                                for c in cols}
    res['val_refit_jumpy'] = int(df['jumpy'].sum())
    res['val_refit_cells'] = int(len(df))
    by_asset = df.groupby('asset')[cols].mean()
    res['val_refit_by_asset'] = {a: {c: float(r[c]) for c in cols} for a, r in by_asset.iterrows()}
    print(by_asset.round(3).to_string())
    print('summary', json.dumps(res['val_refit_summary']), '| jumpy cells', res['val_refit_jumpy'], 'of', len(df))
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs/checks/pb5/display.json'))
    main_pb(ap.parse_args().out)


if __name__ == '__main__':
    main()
