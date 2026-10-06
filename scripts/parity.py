"""Rule R10 (docs/rules.md): parity of the Pine indicator (debug mode) with the Python reference.

Input: TradingView 'Export chart data' CSV of BINANCE:BTCUSDT 5M with the indicator in debug mode (columns: time, open,
high, low, close and the plots 'dbg ...'). The reference (mri.stream, EMA adjust=True, parameters of
export/mri_model.json) recomputes from the same OHLC from the first exported bar.

    python scripts/parity.py chart.csv [--symbol BTCUSDT] [--out docs/checks/parity.json]
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
from mri.params import SIDE_NAMES, from_dict  # noqa: E402
from mri.stream import run  # noqa: E402

INPUTS = ['norm', 'r', 'step_hl2', 'step_hlc3', 'g', 'ema7', 'ema1', 'D', 'x_end'] + [f'side_{n}' for n in SIDE_NAMES]
TOL_IN, TOL_LOGIT, TOL_M, TIE_BAND = 1e-9, 1e-6, 1e-6, 1e-6


def load_tv(path):
    """TradingView export -> (DataFrame with time ms, OHLC, dbg_* columns; decimals per column)."""
    raw = pd.read_csv(path, dtype=str)
    cols = {c: c.strip() for c in raw.columns}
    raw = raw.rename(columns=cols)
    t = raw['time']
    if t.str.fullmatch(r'\d+').all():
        ms = t.astype(np.int64) * (1 if t.str.len().max() >= 13 else 1000)
    else:
        ms = pd.to_datetime(t, utc=True).astype('datetime64[ms, UTC]').astype(np.int64)
    out = pd.DataFrame({'time': ms.to_numpy()})
    dec = {}
    for c in raw.columns:
        if c == 'time':
            continue
        key = c[4:].strip() if c.startswith('dbg ') else c
        s = raw[c].fillna('').str.strip()
        out[key] = pd.to_numeric(s.replace({'': 'nan', 'NaN': 'nan', '∅': 'nan'}), errors='coerce')
        frac = s.str.extract(r'\.(\d+)$')[0].dropna()
        dec[key] = int(frac.str.len().max()) if len(frac) else 0
    return out, dec


def reference(df, ex, symbol):
    p = from_dict(ex['model'])
    floor = ex['floors']['floor_L64'].get(symbol, ex['floors']['unknown_symbol'])
    res = run(df[['time', 'open', 'high', 'low', 'close']], p, floor, eps=ex['tie_rule']['eps'], ema_adjust=True,
              platt=(ex['calibration']['platt_a'], ex['calibration']['platt_b']))
    rows = []
    burn = ex['warmup']['burn_bars_from_first_valid_r']
    for x in res:
        side = x.get('side_raw', [np.nan] * 16)
        shown = np.isfinite(x['logit']) and x['first_r'] is not None and x['bar'] - x['first_r'] >= burn
        row = {'norm': x['norm'], 'r': x['r'], 'step_hl2': x['step_hl2'], 'step_hlc3': x['step_hlc3'], 'g': x['g'],
               'D': x.get('D', np.nan), 'x_end': x.get('x_end', np.nan), 'sgn': x.get('sgn', np.nan),
               'logit': x['logit'] if shown else np.nan, 'M': x.get('M', np.nan) if shown else np.nan}
        row.update({f'side_{n}': v for n, v in zip(SIDE_NAMES, side)})
        rows.append(row)
    ref = pd.DataFrame(rows)
    ref['ema7'], ref['ema1'] = ema_series(df, ex)
    for c in ('D', 'x_end', 'sgn') + tuple(f'side_{n}' for n in SIDE_NAMES):
        ref.loc[~np.isfinite(ref['logit']), c] = np.nan            # Pine fills these only on shown windows
    return ref


def ema_series(df, ex):
    """Normalized EMA |r| after each bar, as the Pine debug plots (s / w)."""
    from mri.features import bar_features
    f = bar_features(df[['time', 'open', 'high', 'low', 'close']], ema_adjust=True)
    out = []
    for span in (2016, 288):
        a, s, w, v = 2.0 / (span + 1.0), 0.0, 0.0, []
        for x in f['r']:
            if np.isfinite(x):
                s, w = abs(x) + (1 - a) * s, 1 + (1 - a) * w
            v.append(s / w if w > 0 else np.nan)
        out.append(np.array(v))
    return out


def live_last_bar(tv, ref):
    """The last exported row is the forming (unconfirmed) bar if Pine did not advance its state there: no logit while the
       reference has one, EMA values equal to the previous row. Pine computes only on confirmed bars (no repaint)."""
    if len(tv) < 2 or not {'logit', 'ema7', 'ema1'} <= set(tv.columns):
        return False
    a = tv.iloc[-1]
    b = tv.iloc[-2]
    return bool(np.isnan(a['logit']) and np.isfinite(ref['logit'].iloc[-1])
                and a['ema7'] == b['ema7'] and a['ema1'] == b['ema1'])


def compare(tv, dec, ref):
    res, ok = {'bars': int(len(tv))}, True
    res['live_last_bar_excluded'] = live_last_bar(tv, ref)
    if res['live_last_bar_excluded']:
        tv, ref = tv.iloc[:-1], ref.iloc[:-1]
    bi = tv['bar_index'] if 'bar_index' in tv else None
    res['bar_index_contiguous_from_0'] = bool(bi is not None and np.array_equal(bi.to_numpy(), np.arange(len(tv))))
    ok &= res['bar_index_contiguous_from_0']
    rnd = lambda c: 0.5 * 10.0 ** -dec.get(c, 10)
    for c in INPUTS + ['logit', 'M']:
        if c not in tv:
            res[c] = 'missing'
            ok = False
            continue
        a, b = tv[c].to_numpy(float), ref[c].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        na_mismatch = int((np.isfinite(a) != np.isfinite(b)).sum())
        d = np.abs(a - b)
        if c == 'M':
            tol = TOL_M * np.abs(b) + rnd(c)
        elif c == 'logit':
            tol = TOL_LOGIT + rnd(c)
        else:
            tol = TOL_IN * np.maximum(1.0, np.abs(b)) + rnd(c)
        bad = both & (d > tol)
        res[c] = {'compared': int(both.sum()), 'max_abs_diff': float(d[both].max()) if both.any() else None,
                  'over_tol': int(bad.sum()), 'na_mismatch': na_mismatch,
                  'first_bad_bar': int(np.argmax(bad | (np.isfinite(a) != np.isfinite(b))))
                  if (bad.any() or na_mismatch) else None}
        ok &= not bad.any() and na_mismatch == 0
    if 'sgn' in tv and 'x_end' in tv:
        z = ref['x_end'] / ref['D']
        near = np.abs(z + 1e-4) <= TIE_BAND
        a, b = tv['sgn'].to_numpy(float), ref['sgn'].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        res['sgn'] = {'mismatch': int((both & (a != b) & ~near).sum()), 'near_eps_windows': int((both & near).sum())}
        ok &= res['sgn']['mismatch'] == 0
    fs = lambda x: int(np.argmax(np.isfinite(x))) if np.isfinite(x).any() else None
    res['first_line_bar'] = {'pine': fs(tv['logit'].to_numpy(float)) if 'logit' in tv else None,
                             'reference': fs(ref['logit'].to_numpy(float))}
    ok &= res['first_line_bar']['pine'] == res['first_line_bar']['reference']
    res['pass_R10'] = bool(ok)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--symbol', default='BTCUSDT')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    with open(os.path.join(ROOT, 'export/mri_model.json'), encoding='utf-8') as fh:
        ex = json.load(fh)
    tv, dec = load_tv(a.csv)
    ref = reference(tv, ex, a.symbol)
    res = compare(tv, dec, ref)
    res['decimals'] = dec
    print(json.dumps({k: v for k, v in res.items() if k != 'decimals'}, indent=1))
    if a.out:
        with open(a.out, 'w') as fh:
            json.dump(res, fh, indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)


if __name__ == '__main__':
    main()
