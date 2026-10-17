"""Rule R10 (docs/rules.md): parity of the Pine indicator (debug mode) with the Python reference.

Input: TradingView 'Export chart data' CSV with the indicator in debug mode (columns: time, open, high, low, close and
the plots 'dbg ...'). The reference recomputes from the same OHLC from the first exported bar, with the parameters of the
export JSON (the model is taken from it):
    PB_s64h24_d1 (export/mri_model_pb.json, default; R10 addendum 3): mri.pb stream, floor of the symbol, the ruler of the
        export; the live part (mri.linem.LiveM, as Pine) from the first computed valid window; compared also z, m_ema,
        s_ema (|d| <= 1e-6), log M (|d| <= 1e-5), n (exactly); first line bar — the first value of the visible plot 'M'
        (R16; the visible M is reported, capped at the ceilings of the symbol);
    NT3_PRV_ch48 (export/mri_model_nt3.json; R10 addition 2): mri.nt3 stream, floor and Platt of the symbol;
        17 side inputs; no EMA — the unconfirmed last bar is recognized by 'dbg confirmed' (count of confirmed bars)
        equal to the previous row (amendment of session 3);
    F3_g (export/mri_model.json, legacy): mri.stream, EMA adjust=True.

    python scripts/parity.py chart.csv [--symbol BTCUSDT] [--export export/mri_model_pb.json] [--calc-bars N] [--out ...]
    python scripts/parity.py artifacts/parity/btc5m_pb_tv_debug.csv --symbol BTCUSDT --out docs/checks/pb/parity_btc.json

--partial (PB; docs/rules.md, «Справочная сверка по короткой выгрузке», not R10): the export holds only the last bars of the
chart. The model part (264 bars of memory) is compared from row 264 of the export; the live part continues from the Pine
state of the first compared window (n, m_ema, s_ema -> n1, n2, d), so its accumulation before that window is not checked.
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
from mri import linem, nt3, pb  # noqa: E402
from mri.params import SIDE_NAMES, from_dict  # noqa: E402
from mri.stream import run  # noqa: E402

INPUTS = ['norm', 'r', 'step_hl2', 'step_hlc3', 'g', 'ema7', 'ema1', 'D', 'x_end'] + [f'side_{n}' for n in SIDE_NAMES]
INPUTS_NT3 = ['norm', 'r', 'step_hl2', 'step_hlc3', 'g', 'D', 'x_end'] + [f'side_{n}' for n in nt3.SIDE_NAMES]
INPUTS_PB = INPUTS_NT3
LINE_PB = {'z': 1e-6, 'm_ema': 1e-6, 's_ema': 1e-6, 'logM': 1e-5}      # R10 addendum 3 (+ rounding of the export)
TOL_IN, TOL_LOGIT, TOL_M, TIE_BAND = 1e-9, 1e-6, 1e-6, 1e-6
TOL_M_PB = 1e-5                                                      # visible M of PB (reported): relative, as log M


def is_nt3(ex):
    return ex['model'].get('id') == nt3.MODEL_ID


def is_pb(ex):
    return ex['model'].get('id') == pb.MODEL_ID


def reference_pb(df, ex, symbol, calc_bars=0, span=None, warm=None):
    """PB reference per bar: inputs and logit by the stream (logits of windows on the computed bars equal the full
       stream), the live part by mri.linem.LiveM from the first computed valid window (R16), the visible M capped at the
       ceilings of the symbol (or of the unknown symbol)."""
    p = from_dict(ex['model'])
    fl, ln = ex['floors'], ex['line']
    known = symbol in fl['floor_L64']
    floor = fl['floor_L64'][symbol] if known else fl['unknown_symbol']
    ceil = ln['ceilings']['by_symbol'][symbol] if known else ln['ceilings']['unknown_symbol']
    span = span or ln['live']['span_default']
    warm = ln['warmup']['windows_default'] if warm is None else warm
    q, zk = ln['ruler']['q'], ln['ruler']['zk']
    res = nt3.run(df[['time', 'open', 'high', 'low', 'close']], p, floor, eps=ex['tie_rule']['eps'])
    first = 0 if not calc_bars else max(len(df) - 1 - calc_bars - 64, 0) + 63
    live = linem.LiveM(q, zk, span, warm, ln['live']['s_lo'])
    rows = []
    for x in res:
        side = x.get('side_raw', [np.nan] * len(nt3.SIDE_NAMES))
        row = {'norm': x['norm'], 'r': x['r'], 'step_hl2': x['step_hl2'], 'step_hlc3': x['step_hlc3'], 'g': x['g'],
               'D': x.get('D', np.nan), 'x_end': x.get('x_end', np.nan), 'sgn': x.get('sgn', np.nan),
               'logit': x['logit'] if x['bar'] >= first else np.nan, 'z': np.nan, 'm_ema': np.nan, 's_ema': np.nan,
               'logM': np.nan, 'n': np.nan, 'M': np.nan, 'shown': False}
        row.update({f'side_{n}': v for n, v in zip(nt3.SIDE_NAMES, side)})
        if np.isfinite(row['logit']):
            lv = live.update(row['logit'])
            row.update({'z': lv['z'], 'm_ema': lv['m_ema'], 's_ema': lv['s_ema'], 'logM': lv['log_m'], 'n': lv['n'],
                        'shown': lv['shown']})
            if lv['shown']:
                v, _, _ = linem.display(lv['log_m'], ceil['orange'], ceil['blue'])
                row['M'] = float(v)
        else:
            for c in ['D', 'x_end', 'sgn'] + [f'side_{n}' for n in nt3.SIDE_NAMES]:
                row[c] = np.nan                                      # Pine fills these only on computed windows
        rows.append(row)
    return pd.DataFrame(rows)


PARTIAL_START = 264        # rows of the export before the first window with its whole memory inside the export


def live_from_pine(ref, tv, ex, symbol, span=None):
    """--partial: the live part of the reference continued from the Pine state after the first compared window w0
       (rules: d = (1 - k^(n+1)) / (1 - k), n1 = m_ema d, n2 = (s_ema^2 + m_ema^2) d); returns (ref, w0)."""
    ln, fl = ex['line'], ex['floors']
    ceil = ln['ceilings']['by_symbol'][symbol] if symbol in fl['floor_L64'] else ln['ceilings']['unknown_symbol']
    k = 1.0 - 2.0 / ((span or ln['live']['span_default']) + 1.0)
    fin = np.isfinite(ref['logit'].to_numpy()) & np.isfinite(tv['logit'].to_numpy()) & np.isfinite(tv['n'].to_numpy())
    fin[:PARTIAL_START] = False
    if not fin.any():
        raise SystemExit('--partial: no window with its 264-bar memory inside the export')
    w0 = int(np.argmax(fin))
    n, m, sd = int(tv['n'].iloc[w0]), float(tv['m_ema'].iloc[w0]), float(tv['s_ema'].iloc[w0])
    d = (1.0 - k ** (n + 1)) / (1.0 - k)
    n1, n2 = m * d, (sd * sd + m * m) * d
    ref = ref.copy()
    ref.loc[:w0, ['m_ema', 's_ema', 'logM', 'n', 'M']] = np.nan
    ref.loc[w0, ['m_ema', 's_ema', 'n']] = [m, sd, n]
    for i in range(w0 + 1, len(ref)):
        if not np.isfinite(ref.at[i, 'logit']):
            continue
        z = ref.at[i, 'z']
        n += 1
        n1, n2, d = k * n1 + z, k * n2 + z * z, k * d + 1.0
        me = n1 / d
        se = float(np.sqrt(max(n2 / d - me * me, 0.0)))
        lm = linem.log_m1(z, me, se, ln['live']['s_lo'])
        ref.loc[i, ['m_ema', 's_ema', 'logM', 'n']] = [me, se, lm, n]
        if n >= ln['warmup']['windows_default']:
            ref.at[i, 'M'] = float(linem.display(lm, ceil['orange'], ceil['blue'])[0])
    return ref, w0


def compare_partial(tv, dec, ref, w0):
    """--partial: rows from w0 (the first compared window, row >= 264); the live part from w0 + 1."""
    tv_, ref_ = tv.iloc[w0:].reset_index(drop=True), ref.iloc[w0:].reset_index(drop=True)
    for c in ('logM', 'M'):                                     # the state of w0 itself is taken from Pine
        tv_.loc[0, c] = np.nan
        ref_.loc[0, c] = np.nan
    res = compare(tv_, dec, ref_, INPUTS_PB, LINE_PB, from0=False)
    res['partial'] = {'first_row': int(w0), 'first_bar_index': int(tv['bar_index'].iloc[w0]),
                      'export_first_bar_index': int(tv['bar_index'].iloc[0]), 'rows': int(len(tv)),
                      'live_state_from_pine': {'n': int(tv['n'].iloc[w0]), 'm_ema': float(tv['m_ema'].iloc[w0]),
                                               's_ema': float(tv['s_ema'].iloc[w0])},
                      'not_checked': 'accumulation of the live part before the first compared window; first line bar (R16)'}
    res['pass_partial'] = res.pop('pass_R10')
    res['pass_R10'] = None
    return res


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
    for c in sorted(raw.columns, key=lambda c: c.startswith('dbg ')):   # 'dbg M' wins over the visible plot 'M'
        if c == 'time':
            continue
        key = c[4:].strip() if c.startswith('dbg ') else c
        s = raw[c].fillna('').str.strip()
        out[key] = pd.to_numeric(s.replace({'': 'nan', 'NaN': 'nan', '∅': 'nan'}), errors='coerce')
        frac = s.str.extract(r'\.(\d+)$')[0].dropna()
        dec[key] = int(frac.str.len().max()) if len(frac) else 0
    return out, dec


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
    if is_pb(ex):
        return reference_pb(df, ex, symbol, calc_bars, **kw)
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


def live_last_bar(tv, ref):
    """The last exported row is the forming (unconfirmed) bar if Pine did not advance its state there: no logit while the
       reference has one, and the state equal to the previous row — EMA values (F3_g) or the count of confirmed bars
       'confirmed' (NT3, no EMA). Pine computes only on confirmed bars (no repaint)."""
    state = ['confirmed'] if 'confirmed' in tv.columns else ['ema7', 'ema1']
    if len(tv) < 2 or not {'logit', *state} <= set(tv.columns):
        return False
    a = tv.iloc[-1]
    b = tv.iloc[-2]
    return bool(np.isnan(a['logit']) and np.isfinite(ref['logit'].iloc[-1]) and all(a[k] == b[k] for k in state))


def compare(tv, dec, ref, inputs=None, line=None, from0=True):
    """line (PB): {column: tolerance} of the live part; then n is compared exactly, the visible M is reported and the
       first line bar is the first value of the visible M (R16). from0 = False (--partial): bar_index only contiguous,
       the first line bar is not checked."""
    inputs = INPUTS if inputs is None else inputs
    res, ok = {'bars': int(len(tv))}, True
    res['live_last_bar_excluded'] = live_last_bar(tv, ref)
    if res['live_last_bar_excluded']:
        tv, ref = tv.iloc[:-1], ref.iloc[:-1]
    bi = tv['bar_index'] if 'bar_index' in tv else None
    if from0:
        res['bar_index_contiguous_from_0'] = bool(bi is not None and np.array_equal(bi.to_numpy(), np.arange(len(tv))))
        ok &= res['bar_index_contiguous_from_0']
    else:
        res['bar_index_contiguous'] = bool(bi is not None and (np.diff(bi.to_numpy()) == 1).all())
        ok &= res['bar_index_contiguous']
    rnd = lambda c: 0.5 * 10.0 ** -dec.get(c, 10)
    cols = inputs + ['logit'] + (list(line) + ['n'] if line else []) + ['M']
    for c in cols:
        if c not in tv:
            res[c] = 'missing'
            ok = False
            continue
        a, b = tv[c].to_numpy(float), ref[c].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        na_mismatch = int((np.isfinite(a) != np.isfinite(b)).sum())
        d = np.abs(a - b)
        if c == 'M':
            tol = (TOL_M_PB if line else TOL_M) * np.abs(b) + rnd(c)
        elif c == 'logit':
            tol = TOL_LOGIT + rnd(c)
        elif line and c in line:
            tol = line[c] + rnd(c)
        elif c == 'n':
            tol = 0.0
        else:
            tol = TOL_IN * np.maximum(1.0, np.abs(b)) + rnd(c)
        bad = both & (d > tol)
        res[c] = {'compared': int(both.sum()), 'max_abs_diff': float(d[both].max()) if both.any() else None,
                  'over_tol': int(bad.sum()), 'na_mismatch': na_mismatch,
                  'first_bad_bar': int(np.argmax(bad | (np.isfinite(a) != np.isfinite(b))))
                  if (bad.any() or na_mismatch) else None}
        if c == 'M' and line:
            res[c]['reported_only'] = True                         # visible line: its first value is the R16 check
        else:
            ok &= not bad.any() and na_mismatch == 0
    if 'sgn' in tv and 'x_end' in tv:
        z = ref['x_end'] / ref['D']
        near = np.abs(z + 1e-4) <= TIE_BAND
        a, b = tv['sgn'].to_numpy(float), ref['sgn'].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        res['sgn'] = {'mismatch': int((both & (a != b) & ~near).sum()), 'near_eps_windows': int((both & near).sum()),
                      'ties_abs_z_lt_1e-9': int((both & (np.abs(z.to_numpy(float)) < 1e-9)).sum()),
                      'flipped': int((both & (b < 0)).sum()), 'compared': int(both.sum())}
        ok &= res['sgn']['mismatch'] == 0
    fs = lambda x: int(np.argmax(np.isfinite(x))) if np.isfinite(x).any() else None
    key = 'M' if line else 'logit'
    if from0:
        res['first_line_bar'] = {'pine': fs(tv[key].to_numpy(float)) if key in tv else None,
                                 'reference': fs(ref[key].to_numpy(float))}
        ok &= (res['first_line_bar']['pine'] is not None
               and res['first_line_bar']['pine'] == res['first_line_bar']['reference'])
    if line and from0:
        res['first_window_bar'] = {'pine': fs(tv['logit'].to_numpy(float)) if 'logit' in tv else None,
                                   'reference': fs(ref['logit'].to_numpy(float))}
        ok &= res['first_window_bar']['pine'] == res['first_window_bar']['reference']
    res['pass_R10'] = bool(ok)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--symbol', default='BTCUSDT')
    ap.add_argument('--export', default=os.path.join(ROOT, 'export/mri_model_pb.json'))
    ap.add_argument('--calc-bars', type=int, default=0, help="the indicator's 'Compute only the last N bars' at export")
    ap.add_argument('--span', type=int, default=None, help='PB: span of the live part set in the indicator (default 8064)')
    ap.add_argument('--warm', type=int, default=None, help='PB: line warm-up set in the indicator (default 288)')
    ap.add_argument('--partial', action='store_true', help='PB: the export holds only the last bars (reference check, not R10)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    with open(a.export, encoding='utf-8') as fh:
        ex = json.load(fh)
    tv, dec = load_tv(a.csv)
    if is_pb(ex) and a.partial:
        ref = reference_pb(tv, ex, a.symbol, 0, a.span, a.warm)
        ref, w0 = live_from_pine(ref, tv, ex, a.symbol, a.span)
        res = compare_partial(tv, dec, ref, w0)
    elif is_pb(ex):
        ref = reference_pb(tv, ex, a.symbol, a.calc_bars, a.span, a.warm)
        res = compare(tv, dec, ref, INPUTS_PB, LINE_PB)
    else:
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
