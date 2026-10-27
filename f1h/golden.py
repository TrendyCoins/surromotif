"""The golden file: candles and the expected values of every candle (gzip CSV, stdlib + numpy), and the end-to-end check
of rules R24 / R31 (docs/rules.md): the reference on the candles of the file gives the values of the file.

    python -m f1h [--tf 1H | 5M | 15M | 30M] [--golden PATH] [--model PATH]      # exit code 0 = ok
    # defaults: the export and the golden of the timeframe in this repository (1H: export/f1h_model.json,
    # tests/golden/btc1h_f1h.csv.gz; 15M: export/f1h_model_15m.json, tests/golden/btc15m_f1h.csv.gz; ...)
"""
import argparse
import csv
import gzip
import io
import json
import os

import numpy as np

from .model import ROOT, load_model
from .run import FIELDS, compute

GOLDEN = os.path.join(ROOT, 'tests', 'golden', 'btc1h_f1h.csv.gz')
GOLDENS = {tf: os.path.join(ROOT, 'tests', 'golden', f'btc{tf.lower()}_f1h.csv.gz') for tf in ('1H', '5M', '15M', '30M')}
CANDLES = ('time', 'open', 'high', 'low', 'close')
COLUMNS = CANDLES + FIELDS[1:]
INTS = ('time', 'n', 'n_live')
BOOLS = ('window', 'line')
FLOATS = ('logit', 'z', 'm', 's', 'log_m', 'strength')
TOL = 1e-9                       # rules R24 / R31: other numpy / BLAS versions


def _cell(x, kind):
    if kind == 'b':
        return '1' if x else '0'
    if kind == 'i':
        return str(int(x))
    return repr(float(x)) if np.isfinite(x) else ''


def write_golden(path, candles, out):
    """candles: dict time, open, high, low, close; out: compute() of them. Floats — shortest exact decimals."""
    kinds = {c: ('i' if c in INTS else 'b' if c in BOOLS else 'f') for c in COLUMNS}
    cols = {**{k: np.asarray(candles[k]) for k in CANDLES}, **{k: out[k] for k in FIELDS[1:]}}
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(COLUMNS)
    for i in range(len(cols['time'])):
        w.writerow([_cell(cols[c][i], kinds[c]) for c in COLUMNS])
    with open(path, 'wb') as fh, gzip.GzipFile(fileobj=fh, mode='wb', mtime=0, filename='') as gz:
        gz.write(buf.getvalue().encode('ascii'))


def read_golden(path=None):
    """Golden file -> dict of arrays (NaN for empty cells)."""
    with gzip.open(path or GOLDEN, 'rt', encoding='ascii', newline='') as fh:
        rows = list(csv.reader(fh))
    head, body = rows[0], rows[1:]
    if tuple(head) != COLUMNS:
        raise ValueError(f"golden columns {head} != {list(COLUMNS)}")
    cols = list(zip(*body)) if body else [()] * len(head)
    out = {}
    for name, vals in zip(head, cols):
        if name in INTS:
            out[name] = np.array([int(v) for v in vals], dtype=np.int64)
        elif name in BOOLS:
            out[name] = np.array([v == '1' for v in vals], dtype=bool)
        else:
            out[name] = np.array([float(v) if v else np.nan for v in vals], dtype=np.float64)
    return out


def compare(out, ref, tol=TOL):
    """Rule R24: out (compute) against ref (golden or another run) on the same candles. Returns a dict with ok."""
    res = {'n_bars': int(len(ref['time']))}
    ok = bool(np.array_equal(out['time'], ref['time']))
    for k in ('window', 'n', 'n_live', 'line'):
        same = bool(np.array_equal(out[k], ref[k]))
        res[f'{k}_same'] = same
        ok &= same
    for k in FLOATS:
        a, b = out[k], ref[k]
        nan_same = bool(np.array_equal(np.isnan(a), np.isnan(b)))
        fin = np.isfinite(a) & np.isfinite(b)
        d = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0
        res[f'max_d_{k}'] = d
        ok &= nan_same and d <= tol
    a, b = out['side'], ref['side']
    near0 = np.abs(np.nan_to_num(ref['log_m'], nan=np.inf)) <= tol
    diff = ~((a == b) | (np.isnan(a) & np.isnan(b))) & ~near0
    res['side_diff'] = int(diff.sum())
    ok &= res['side_diff'] == 0
    shown = np.where(out['line'])[0]
    res['first_window_bar'] = int(np.argmax(out['window'])) if out['window'].any() else None
    res['first_line_bar'] = int(shown[0]) if len(shown) else None
    res['bars_with_line'] = int(len(shown))
    res['ok'] = bool(ok)
    return res


def check_golden(model=None, path=None):
    """End-to-end check on the golden file (rules R24 / R31): candles of the file -> compute -> values of the file. Without
       a path — the golden of the timeframe of the model in this repository (GOLDENS)."""
    model = load_model() if model is None else model
    g = read_golden(path or GOLDENS[model.timeframe])
    out = compute(model, g['time'], g['open'], g['high'], g['low'], g['close'])
    res = compare(out, g)
    res['first_line_bar_expected'] = model.first_line_bar
    res['ok'] = res['ok'] and res['first_line_bar'] == model.first_line_bar
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--tf', default=None, help='1H, 5M, 15M or 30M: the export and the golden of this repository')
    ap.add_argument('--golden', default=None)
    ap.add_argument('--model', default=None)
    a = ap.parse_args()
    model = load_model(a.model, tf=a.tf)
    res = check_golden(model, a.golden)
    res['timeframe'] = model.timeframe
    print(json.dumps(res, indent=1))
    raise SystemExit(0 if res['ok'] else 1)
