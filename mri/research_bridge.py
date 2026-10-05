"""Bars -> research data files (smc_series / smc_prices / smc_range npz), so that research code (SeriesData,
to_shapes, side_feats, CNNBase of research/sumo_class_train.py) can score the same bars as the reference.
Used only by checks (torch vs reference); the indicator never needs it.

The research data scripts run their main() on import (Drive I/O), so the arithmetic is copied here:
    make_parts  <- research/sumo_class_data.py, make_parts (r, parts with a 200-bar warm-up each)
    e / d / d0  <- research/sumo_class_prices.py, prepare (shifts of HL/2, HLC/3)
    g           <- research/sumo_class_range.py, prepare (bar range)
tests/test_bridge.py checks the copy of make_parts against the original bit for bit.
"""
import json
import os
import zlib

import numpy as np
import pandas as pd

N = 200
PRICES = {'hl2': lambda h, l, c: (h + l) / 2.0,
          'hlc3': lambda h, l, c: (h + l + c) / 3.0}


def make_parts(df, N, t_start, t_valid):
    """Copy of research/sumo_class_data.py make_parts (df: 'timestamp' datetime, open/high/low/close)."""
    close = df['close'].to_numpy(dtype=np.float64)
    hl = (df['high'] - df['low']).to_numpy(dtype=np.float64)
    date = df['timestamp']
    roll = pd.Series(hl).rolling(N, closed='left').median().to_numpy()

    dt = date.diff().dt.total_seconds().to_numpy()
    step = float(np.nanmedian(dt[1:]))
    gap = np.zeros(len(df), dtype=bool)
    gap[1:] = dt[1:] > step * 1.5

    d_close = np.diff(close, prepend=close[0])
    d_close[0] = 0.0
    r = np.zeros_like(d_close)
    np.divide(d_close, roll, out=r, where=roll != 0)
    bad = ~np.isfinite(roll) | (roll <= 0) | ~np.isfinite(r) | gap
    r[bad] = np.nan
    r[0] = np.nan

    parts = {}
    for name, a, b in (('TRAINING', t_start, t_valid), ('VALIDATION', t_valid, None)):
        hits = np.where(date >= pd.Timestamp(a))[0]
        i0 = int(hits[0])
        if b is None:
            i1 = len(df) - 1
        else:
            i1 = int(np.where(date < pd.Timestamp(b))[0][-1])
        lo, hi = i0 + N, i1 + 1
        parts[name] = {'r': r[lo:hi].copy(), 'close': close[lo:hi].copy(),
                       'date': date.iloc[lo:hi].reset_index(drop=True),
                       'i0': lo, 'i1': i1, 'n_gap': int(gap[lo:hi].sum())}
    return parts, step


def write_research_files(bars, out_dir, asset, t_start, t_valid, floor_L64):
    """bars: DataFrame time (ms), open, high, low, close. Writes the three npz of one asset into out_dir.
       Returns {'tr': (lo, hi), 'va': (lo, hi)} — CSV row ranges of the parts."""
    df = pd.DataFrame({'timestamp': pd.to_datetime(bars['time'], unit='ms'), 'open': bars['open'],
                       'high': bars['high'], 'low': bars['low'], 'close': bars['close']})
    parts, _ = make_parts(df, N, t_start, t_valid)
    high, low, close = (df[k].to_numpy(dtype=np.float64) for k in ('high', 'low', 'close'))
    # prices / range: same bad mask as make_parts (research/sumo_class_prices.py, sumo_class_range.py)
    roll = pd.Series(high - low).rolling(N, closed='left').median().to_numpy()
    dt = df['timestamp'].diff().dt.total_seconds().to_numpy()
    step = float(np.nanmedian(dt[1:]))
    gap = np.zeros(len(df), dtype=bool)
    gap[1:] = dt[1:] > step * 1.5
    d_close = np.diff(close, prepend=close[0])
    d_close[0] = 0.0
    r = np.zeros_like(d_close)
    np.divide(d_close, roll, out=r, where=roll != 0)
    bad = ~np.isfinite(roll) | (roll <= 0) | ~np.isfinite(r) | gap
    good_roll = np.isfinite(roll) & (roll > 0)
    prev_close = np.concatenate([[np.nan], close[:-1]])
    e_all, d_all = {}, {}
    for nm, f in PRICES.items():
        p = f(high, low, close)
        e = np.full(len(close), np.nan)
        np.divide(p - prev_close, roll, out=e, where=good_roll)
        e[bad] = np.nan
        e[0] = np.nan
        dd = np.full(len(close), np.nan)
        np.divide(p - close, roll, out=dd, where=good_roll)
        e_all[nm], d_all[nm] = e, dd
    g = np.full(len(close), np.nan)
    np.divide(high - low, roll, out=g, where=~bad)

    ser, pri, rng, crc, rows = {}, {}, {}, {}, {}
    for name, key in (('TRAINING', 'tr'), ('VALIDATION', 'va')):
        pt = parts[name]
        lo, hi = pt['i0'], pt['i1'] + 1
        rows[key] = (lo, hi)
        r_part = pt['r'].astype(np.float32)
        ser[f'r_{key}'] = r_part
        ser[f't_{key}'] = pt['date'].to_numpy().astype('datetime64[ns]').astype(np.int64)
        ser[f'close_{key}'] = pt['close'].astype(np.float64)
        ser[f'fin_{key}'] = np.where(np.isfinite(pt['r']))[0].astype(np.int32)
        crc[key] = zlib.crc32(r_part.tobytes())
        for nm in PRICES:
            pri[f'e_{nm}_{key}'] = e_all[nm][lo:hi].astype(np.float32)
            pri[f'd_{nm}_{key}'] = d_all[nm][lo:hi].astype(np.float32)
            pri[f'd0_{nm}_{key}'] = np.array(np.float32(d_all[nm][lo - 1]))
        g_part = g[lo:hi].astype(np.float32)
        g_part[~np.isfinite(r_part)] = np.nan
        rng[f'g_{key}'] = g_part
    ser['floor_L64'] = np.array(floor_L64)
    meta = {'asset': asset, 'tf': '5M', 'N': N, 'parts': {
        'TRAINING': {'n': int(len(ser['r_tr'])), 'from': str(parts['TRAINING']['date'].iloc[0])},
        'VALIDATION': {'n': int(len(ser['r_va'])), 'from': str(parts['VALIDATION']['date'].iloc[0])}}}
    ser['meta'] = np.array(json.dumps(meta))
    pri['meta'] = np.array(json.dumps({'crc': crc}))
    rng['meta'] = np.array(json.dumps({'crc': crc}))
    os.makedirs(out_dir, exist_ok=True)
    for kind, arrs in (('series', ser), ('prices', pri), ('range', rng)):
        np.savez(os.path.join(out_dir, f"smc_{kind}_{asset}_5M_N{N}.npz"), **arrs)
    return rows
