"""Bars -> research data files (smc_series / smc_prices / smc_range npz), so that research code (SeriesData,
to_shapes, side_feats, CNNBase of research/sumo_class_train.py) can score the same bars as the reference.
Used only by checks (torch vs reference); the indicator never needs it.

The research data scripts run their main() on import (Drive I/O), so the arithmetic is copied here:
    make_parts  <- research/sumo_class_data.py, make_parts (r, parts with a 200-bar warm-up each)
    e / d / d0  <- research/sumo_class_prices.py, prepare (shifts of HL/2, HLC/3)
    g           <- research/sumo_class_range.py, prepare (bar range)
    smc_bars    <- research/sumo_class_bars.py, prepare (raw OHLC of the parts + 200 CSV bars before each part; NT3 history)
tests/test_bridge.py checks the copies of make_parts and of the smc_bars prepare against the originals bit for bit (the
original bars script runs as is in a subprocess on a synthetic CSV).

floor_override (session 5, rule R17): the divisor floor of every series of a research SeriesData set to one value for a
block, as research/sumo_class_ceil.py floor_override — that function itself (the Ceil file imports without side effects).
"""
import importlib.util
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

N = 200
FLOOR = float(np.float32(0.2))         # rule R17: one floor of the window divisor for every symbol
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


def same_array(a, b):
    """Copy of research/sumo_class_bars.py same_array: float32 series equal bit for bit, NaN at the same places."""
    if len(a) != len(b):
        return False
    na, nb = np.isnan(a), np.isnan(b)
    return bool(np.array_equal(na, nb) and np.array_equal(a[~na], b[~nb]))


def bars_arrays(df, d, asset, series_name, N=200, n_pre=200, tf='5M'):
    """Copy of the arithmetic of research/sumo_class_bars.py prepare (TrendyCoins_SMC_Bars), without file I/O and prints.
       df: CSV as read by the original (pd.read_csv(..., parse_dates=['timestamp'])); d: the smc_series npz of the asset.
       Returns the dict that the original saves (arrays o/h/l/c_{tr,va}, pre_*, and 'meta' with 'created')."""
    o, h, l, c = (df[k].to_numpy(dtype=np.float64) for k in ('open', 'high', 'low', 'close'))
    date = df['timestamp']
    ts = date.to_numpy().astype('datetime64[ns]').astype(np.int64)
    if not np.all(np.diff(ts) > 0):
        raise ValueError(f"{asset}: CSV timestamps are not increasing")
    roll = pd.Series(h - l).rolling(N, closed='left').median().to_numpy()
    dt = date.diff().dt.total_seconds().to_numpy()
    step = float(np.nanmedian(dt[1:]))
    gap = np.zeros(len(df), dtype=bool)
    gap[1:] = dt[1:] > step * 1.5
    d_close = np.diff(c, prepend=c[0])
    d_close[0] = 0.0
    r = np.zeros_like(d_close)
    np.divide(d_close, roll, out=r, where=roll != 0)
    bad = ~np.isfinite(roll) | (roll <= 0) | ~np.isfinite(r) | gap
    r[bad] = np.nan
    r[0] = np.nan
    out, crc = {}, {}
    for name, key in (('TRAINING', 'tr'), ('VALIDATION', 'va')):
        t_part, r_ser = d[f't_{key}'], d[f'r_{key}']
        idx = np.searchsorted(ts, t_part)
        idx_c = np.minimum(idx, len(ts) - 1)
        found = (idx < len(ts)) & (ts[idx_c] == t_part)
        if not found.all():
            raise ValueError(f"{asset} {name}: {int((~found).sum())} bars of smc_series are not in the CSV")
        if not np.all(np.diff(idx) == 1):
            raise ValueError(f"{asset} {name}: bars of smc_series are not consecutive in the CSV")
        if not same_array(r_ser, r[idx].astype(np.float32)):
            raise ValueError(f"{asset} {name}: r from the CSV differs from smc_series")
        if f'close_{key}' in d.files and not np.array_equal(c[idx], d[f'close_{key}'].astype(np.float64)):
            raise ValueError(f"{asset} {name}: close from the CSV differs from smc_series")
        i0 = int(idx[0])
        if i0 < n_pre:
            raise ValueError(f"{asset} {name}: {i0} bars before the part in the CSV, {n_pre} needed")
        pre = slice(i0 - n_pre, i0)
        for k_, v in (('o', o), ('h', h), ('l', l), ('c', c)):
            out[f'{k_}_{key}'] = v[idx].copy()
            out[f'pre_{k_}_{key}'] = v[pre].copy()
        crc[key] = zlib.crc32(r_ser.astype(np.float32).tobytes())
    out['meta'] = np.array(json.dumps({'asset': asset, 'tf': tf, 'N': N, 'n_pre': n_pre, 'ver': 1,
                                       'what': 'raw OHLC of smc_series bars + N_PRE bars before each part',
                                       'series': series_name, 'crc': crc, 'csv_last': str(date.iloc[-1]),
                                       'created': time.strftime('%Y-%m-%d %H:%M:%S')}, ensure_ascii=False))
    return out


def write_bars_file(df, data_dir, asset, N=200, tf='5M'):
    """smc_bars_<asset>_<tf>_N<N>.npz next to the asset's smc_series in data_dir (as research/sumo_class_bars.py)."""
    series = os.path.join(data_dir, f"smc_series_{asset}_{tf}_N{N}.npz")
    with np.load(series, allow_pickle=False) as d:
        out = bars_arrays(df, d, asset, os.path.basename(series), N=N, tf=tf)
    path = os.path.join(data_dir, f"smc_bars_{asset}_{tf}_N{N}.npz")
    np.savez_compressed(path, **out)
    return path


def write_research_files(bars, out_dir, asset, t_start, t_valid, floor_L64, with_bars=False):
    """bars: DataFrame time (ms), open, high, low, close. Writes the three npz of one asset into out_dir (with_bars: and the
       smc_bars file of the NT3 history inputs). Returns {'tr': (lo, hi), 'va': (lo, hi)} — CSV row ranges of the parts."""
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
    if with_bars:
        write_bars_file(df, out_dir, asset, N=N)
    return rows


def _ceil_module():
    """research/sumo_class_ceil.py as a module (only definitions: main() runs under __main__, drive.mount is in a try)."""
    if 'sumo_class_ceil' not in sys.modules:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'research', 'sumo_class_ceil.py')
        spec = importlib.util.spec_from_file_location('sumo_class_ceil', path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules['sumo_class_ceil'] = mod
        spec.loader.exec_module(mod)
    return sys.modules['sumo_class_ceil']


def floor_override(data, L=64, value=FLOOR):
    """Context manager: data.windows(L)['floor'] and ['floor_t'] (float32) of every series = value inside the block (research
       datasets made inside take it), afterwards the same objects as before — research/sumo_class_ceil.py floor_override."""
    return _ceil_module().floor_override(data, L, value)
