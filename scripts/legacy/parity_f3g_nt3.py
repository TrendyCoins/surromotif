"""Rule R10 for the legacy models (docs/rules.md): parity of the F3_g (sessions 1-2) and NT3_PRV_ch48 (session 3) Pine
indicators (debug mode) with their Python references, from the same OHLC of the first exported bar:
    NT3_PRV_ch48 (export/legacy/mri_model_nt3.json; R10 addition 2): mri.nt3 stream, floor and Platt of the symbol;
        17 side inputs; no EMA — the unconfirmed last bar is recognized by 'dbg confirmed' (count of confirmed bars)
        equal to the previous row (amendment of session 3);
    F3_g (export/legacy/mri_model.json): mri.legacy.stream, EMA adjust=True.
The comparison itself (reading the export, tolerances, the unconfirmed last bar) is that of scripts/parity.py (PB).

    python scripts/legacy/parity_f3g_nt3.py chart.csv --export export/legacy/mri_model_nt3.json [--symbol BTCUSDT]
        [--calc-bars N] [--out ...]
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
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from parity import compare as compare_pb, live_last_bar, load_tv  # noqa: E402,F401 (load_tv, live_last_bar: re-exported)
from mri import nt3  # noqa: E402
from mri.legacy.stream import run  # noqa: E402
from mri.params import SIDE_NAMES, from_dict  # noqa: E402

INPUTS = ['norm', 'r', 'step_hl2', 'step_hlc3', 'g', 'ema7', 'ema1', 'D', 'x_end'] + [f'side_{n}' for n in SIDE_NAMES]
INPUTS_NT3 = ['norm', 'r', 'step_hl2', 'step_hlc3', 'g', 'D', 'x_end'] + [f'side_{n}' for n in nt3.SIDE_NAMES]


def is_nt3(ex):
    return ex['model'].get('id') == nt3.MODEL_ID


def reference_nt3(df, ex, symbol):
    p = from_dict(ex['model'])
    fl, cal = ex['floors'], ex['calibration']
    known = symbol in fl['floor_L64']
    floor = fl['floor_L64'][symbol] if known else fl['unknown_symbol']
    pl = cal['per_symbol'][symbol] if known else cal['unknown_symbol']
    res = nt3.run(df[['time', 'open', 'high', 'low', 'close']], p, floor, eps=ex['tie_rule']['eps'],
                  platt=(pl['platt_a'], pl['platt_b']))
    rows = []
    for x in res:
        side = x.get('side_raw', [np.nan] * len(nt3.SIDE_NAMES))
        row = {'norm': x['norm'], 'r': x['r'], 'step_hl2': x['step_hl2'], 'step_hlc3': x['step_hlc3'], 'g': x['g'],
               'D': x.get('D', np.nan), 'x_end': x.get('x_end', np.nan), 'sgn': x.get('sgn', np.nan),
               'logit': x['logit'], 'M': x.get('M', np.nan)}
        row.update({f'side_{n}': v for n, v in zip(nt3.SIDE_NAMES, side)})
        rows.append(row)
    return pd.DataFrame(rows)


def reference(df, ex, symbol, calc_bars=0, **kw):
    if is_nt3(ex):
        return reference_nt3(df, ex, symbol)
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


def compare(tv, dec, ref, inputs=None, **kw):
    """scripts/parity.py compare with the inputs of F3_g by default."""
    return compare_pb(tv, dec, ref, INPUTS if inputs is None else inputs, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--symbol', default='BTCUSDT')
    ap.add_argument('--export', default=os.path.join(ROOT, 'export/legacy/mri_model_nt3.json'))
    ap.add_argument('--calc-bars', type=int, default=0, help="the indicator's 'Compute only the last N bars' at export")
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    with open(a.export, encoding='utf-8') as fh:
        ex = json.load(fh)
    tv, dec = load_tv(a.csv)
    ref = reference(tv, ex, a.symbol)
    if a.calc_bars:      # Pine: the network starts at heavyFrom = last_bar_index - N - 64; first window 63 bars later
        heavy0 = max(len(tv) - 1 - a.calc_bars - 64, 0)
        cols = [c for c in ref.columns if c not in ('norm', 'r', 'step_hl2', 'step_hlc3', 'g', 'ema7', 'ema1')]
        ref.loc[:heavy0 + 62, cols] = np.nan
    res = compare(tv, dec, ref, INPUTS_NT3 if is_nt3(ex) else INPUTS)
    res['model'] = ex['model'].get('id', 'F3_g')
    res['symbol'] = a.symbol
    res['decimals'] = dec
    print(json.dumps({k: v for k, v in res.items() if k != 'decimals'}, indent=1))
    if a.out:
        with open(a.out, 'w') as fh:
            json.dump(res, fh, indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)


if __name__ == '__main__':
    main()
