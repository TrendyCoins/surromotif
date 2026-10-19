"""Rule R10 (docs/rules.md, addendum 4): parity of the Pine indicator (the prototype pine/proto_pb.pine in debug mode) with
the Python reference.

Input: TradingView 'Export chart data' CSV with the indicator in debug mode (columns: time, open, high, low, close and
the plots 'dbg ...'). The reference recomputes from the same OHLC from the first exported bar, with the parameters of the
export JSON (export/mri_model_pb.json): mri.pb stream, the one floor of the export (R17), its ruler; the live part —
formula A (mri.linem.LiveWin, as Pine: ring of 4096 z, S1, S2) from the first computed valid window (R18); compared: the
bar inputs, D, X_end, sign, 17 side inputs, logit, z, m_win, s_win (|d| <= 1e-6), log M (|d| <= 1e-5), n and k (exactly);
first line bar — the first value of the visible plot 'M' (R18: 4360 on a full export from bar 0; the visible M is
reported, capped at the one ceiling). The unconfirmed last bar is recognized by 'dbg confirmed' equal to the previous row.

    python scripts/parity.py chart.csv [--symbol BTCUSDT] [--export export/mri_model_pb.json] [--calc-bars N] [--out ...]
    python scripts/parity.py artifacts/parity/btc5m_pb5_tv_debug.csv --symbol BTCUSDT --out docs/checks/pb5/parity_btc.json

The legacy models (F3_g, NT3_PRV_ch48): scripts/legacy/parity_f3g_nt3.py (the same comparison). The session-4 PB parity
(EMA live part, symbol tables) is in docs/checks/legacy/pb/, its code in the git history.
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
from mri.params import from_dict  # noqa: E402

INPUTS_PB = ['norm', 'r', 'step_hl2', 'step_hlc3', 'g', 'D', 'x_end'] + [f'side_{n}' for n in nt3.SIDE_NAMES]
LINE_PB = {'z': 1e-6, 'm_win': 1e-6, 's_win': 1e-6, 'logM': 1e-5}      # R10 addendum 4 (+ rounding of the export)
TOL_IN, TOL_LOGIT, TOL_M, TIE_BAND = 1e-9, 1e-6, 1e-6, 1e-6
TOL_M_PB = 1e-5                                                      # visible M of PB (reported): relative, as log M


def is_pb(ex):
    return ex['model'].get('id') == pb.MODEL_ID


def reference_pb(df, ex, symbol=None, calc_bars=0):
    """PB reference per bar (session 5): inputs and logit by the stream with the one floor of the export (logits of windows
       on the computed bars equal the full stream), the live part by formula A (mri.linem.LiveWin) from the first computed
       valid window (R18), the visible M capped at the one ceiling. The symbol does not change anything (no constants of
       the symbol); it is kept for the report."""
    p = from_dict(ex['model'])
    ln = ex['line']
    if (ln['live']['L'], ln['live']['n0'], ln['live']['lag'], ln['show']['from_window']) != (
            linem.L_WIN, linem.N0, linem.LAG, linem.SHOW_FROM):
        raise ValueError('export: live part not as mri.linem (formula A 4032 + 144, lag 64, shown from 4096)')
    floor, ceil = ex['floor']['value'], ln['ceiling']['value']
    q, zk = ln['ruler']['q'], ln['ruler']['zk']
    res = nt3.run(df[['time', 'open', 'high', 'low', 'close']], p, floor, eps=ex['tie_rule']['eps'])
    first = 0 if not calc_bars else max(len(df) - 1 - calc_bars - 64, 0) + 63
    live = linem.LiveWin(q, zk, ln['live']['s_lo'])
    rows = []
    for x in res:
        side = x.get('side_raw', [np.nan] * len(nt3.SIDE_NAMES))
        row = {'norm': x['norm'], 'r': x['r'], 'step_hl2': x['step_hl2'], 'step_hlc3': x['step_hlc3'], 'g': x['g'],
               'D': x.get('D', np.nan), 'x_end': x.get('x_end', np.nan), 'sgn': x.get('sgn', np.nan),
               'logit': x['logit'] if x['bar'] >= first else np.nan, 'z': np.nan, 'm_win': np.nan, 's_win': np.nan,
               'k': np.nan, 'logM': np.nan, 'n': np.nan, 'M': np.nan, 'shown': False}
        row.update({f'side_{n}': v for n, v in zip(nt3.SIDE_NAMES, side)})
        if np.isfinite(row['logit']):
            lv = live.update(row['logit'])
            row.update({'z': lv['z'], 'm_win': lv['m_win'], 's_win': lv['s_win'], 'k': lv['k'], 'logM': lv['log_m'],
                        'n': lv['n'], 'shown': lv['shown']})
            if lv['shown']:
                v, _, _ = linem.display(lv['log_m'], ceil, ceil)
                row['M'] = float(v)
        else:
            for c in ['D', 'x_end', 'sgn'] + [f'side_{n}' for n in nt3.SIDE_NAMES]:
                row[c] = np.nan                                      # Pine fills these only on computed windows
        rows.append(row)
    return pd.DataFrame(rows)


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
    """line (PB): {column: tolerance} of the live part; then n and k are compared exactly, the visible M is reported and
       the first line bar is the first value of the visible M (R18). from0 = False: bar_index only contiguous, the first
       line bar is not checked."""
    inputs = INPUTS_PB if inputs is None else inputs
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
    cols = inputs + ['logit'] + (list(line) + ['n', 'k'] if line else []) + ['M']
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
        elif c in ('n', 'k'):
            tol = 0.0
        else:
            tol = TOL_IN * np.maximum(1.0, np.abs(b)) + rnd(c)
        bad = both & (d > tol)
        res[c] = {'compared': int(both.sum()), 'max_abs_diff': float(d[both].max()) if both.any() else None,
                  'over_tol': int(bad.sum()), 'na_mismatch': na_mismatch,
                  'first_bad_bar': int(np.argmax(bad | (np.isfinite(a) != np.isfinite(b))))
                  if (bad.any() or na_mismatch) else None}
        if c == 'M' and line:
            res[c]['reported_only'] = True                         # visible line: its first value is the R18 check
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
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    with open(a.export, encoding='utf-8') as fh:
        ex = json.load(fh)
    if not is_pb(ex):
        raise SystemExit('not the PB export: the legacy models — scripts/legacy/parity_f3g_nt3.py')
    tv, dec = load_tv(a.csv)
    ref = reference_pb(tv, ex, a.symbol, a.calc_bars)
    res = compare(tv, dec, ref, INPUTS_PB, LINE_PB)
    res['model'] = ex['model'].get('id')
    res['symbol'] = a.symbol
    res['decimals'] = dec
    print(json.dumps({k: v for k, v in res.items() if k != 'decimals'}, indent=1))
    if a.out:
        with open(a.out, 'w') as fh:
            json.dump(res, fh, indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)


if __name__ == '__main__':
    main()
