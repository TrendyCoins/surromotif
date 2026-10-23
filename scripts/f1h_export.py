"""Export of F1 on 1H (rule R21, docs/rules.md) and the compact golden file (rule R24).

    python scripts/f1h_export.py      # writes export/f1h_model.json and tests/golden/btc1h_f1h.csv.gz

Sources (artifacts/, read only; sha256 checked against artifacts/pack/site1h_manifest.json): prod_SP_l025_d1_L64_a.json and
_slim.npz (weights, cfg), tfline21_20261022_063520.json (section 'export': ruler, live part, floor, ceiling, history). The
export is self-contained: f1h needs nothing else at run time. Weights are exact decimals of their float32 values.

The golden file: 11 480 continuous BTCUSDT 1H candles from artifacts/bars/smc_bars_BTCUSDT_1H_N200.npz — the last bar of
TRAINING, the 200 bars before VALIDATION and VALIDATION (2025-06-30 23:00 .. 2026-10-22 06:00 UTC; the windows and their
numbers are those of the research golden tfline21) — and the values of the reference on every candle.
research_candles() is shared with scripts/f1h_parity.py.
"""
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import f1h  # noqa: E402
from f1h.bars import SIDE_NAMES, SIDE_SIGNED  # noqa: E402
from f1h.golden import write_golden  # noqa: E402

MANIFEST = 'artifacts/pack/site1h_manifest.json'
PROD = 'artifacts/weights/prod_SP_l025_d1_L64_a'
TFLINE = 'artifacts/calib/tfline21_20261022_063520.json'
BARS = 'artifacts/bars/smc_{}_BTCUSDT_1H_N200.npz'
EXPORT = 'export/f1h_model.json'
GOLDEN = 'tests/golden/btc1h_f1h.csv.gz'
HOUR_MS = 3_600_000
FLOOR = float(np.float32(0.2))
ORIENT_EPS = 1e-4                       # research ORIENT_EPS (v6.0)
CFG_WANT = {'arch': 'cnn', 'L': 64, 'cnn_k': 5, 'cnn_dil': [1, 2, 4, 8], 'cnn_ch': 64, 'cnn_bias': True, 'cnn_valid': False,
            'cnn_bn': False, 'cnn_res': False, 'cnn_sep': False, 'cnn_sym': True, 'cnn_pool': 'mml', 'cnn_in': 'steps',
            'cnn_ma': 0, 'cnn_pyr': None, 'cnn_chan': None, 'lvl_log': False, 'price': 'multi_g', 'side': 'nt_PRV',
            'head_hidden': [128]}


def path(rel):
    return os.path.join(ROOT, rel)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def check_manifest():
    """Every file of the manifest: size and sha256. Returns {path: sha256}."""
    with open(path(MANIFEST), encoding='utf-8') as fh:
        man = json.load(fh)
    out = {}
    for f in man['files']:
        p = path(f['path'])
        if not os.path.isfile(p) or os.path.getsize(p) != f['bytes'] or sha256(p) != f['sha256']:
            raise SystemExit(f"{f['path']}: not the file of {MANIFEST} — no export")
        out[f['path']] = f['sha256']
    return man, out


def git_version():
    try:
        h = subprocess.run(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', ROOT, 'status', '--porcelain', 'f1h'], capture_output=True, text=True).stdout
        return h + ('+dirty' if dirty.strip() else '')
    except OSError:
        return 'unknown'


def num(x):
    """Exact decimal of a float32 value (as float64): parses back to the same float32."""
    return float(np.format_float_positional(np.float64(np.float32(x)), unique=True))


def nums(a):
    a = np.asarray(a)
    return [nums(v) for v in a] if a.ndim > 1 else [num(v) for v in a]


def build_export():
    man, shas = check_manifest()
    with open(path(PROD + '.json'), encoding='utf-8') as fh:
        prod = json.load(fh)
    with open(path(TFLINE), encoding='utf-8') as fh:
        tfl = json.load(fh)
    cfg = prod['cfg']
    bad = {k: (cfg.get(k), v) for k, v in CFG_WANT.items() if cfg.get(k) != v}
    if bad or prod['tag'] != man['prod'] or prod['model'] != 'SP_l025_d1' or prod['seed_key'] != 'a':
        raise SystemExit(f"prod a cfg is not the network of f1h: {bad} — no export")
    with np.load(path(PROD + '_slim.npz')) as z:
        sd = {k[3:]: z[k] for k in z.files}
    ex = tfl['export']
    ru, lv, hi = ex['ruler'], ex['live'], ex['history']
    if (tfl['mode'] != 'line21' or ex['tf'] != '1H' or ex['seeds'] != ['a'] or ex['floor'] != 0.2 or ex['ceiling'] != 8
            or lv['key'] != '4032·k0.5' or (lv['family'], lv['L'], lv['n0'], lv['k'], lv['tau'], lv['lag'],
                                           lv['show_from_window'], lv['s_lo']) != ('F3', 4032, 0, 0.5, 64, 64, 4096, 0.75)
            or (hi['first_window_bar'], hi['first_line_bar']) != (264, 4360) or len(ru['q']) != 37
            or not np.all(np.diff(ru['q']) > 0)):
        raise SystemExit("tfline21 export section is not the line of f1h (1H, floor 0.2, ceiling 8, 4032·k0.5) — no export")
    conv_w = [sd[f'blocks.{i}.0.weight'] for i in range(4)]
    conv_b = [sd[f'blocks.{i}.0.bias'] for i in range(4)]
    n_w = sum(v.size for k, v in sd.items() if k.startswith(('blocks.', 'head.')))
    return {
        'format': f1h.model.FORMAT, 'format_version': f1h.model.FORMAT_VERSION,
        'what': 'SuMo classifier F1 (SP_l025_d1 prod a) and its line M on 1H candles; reference: the f1h package',
        'passport': {
            'model': 'SP_l025_d1: CNN over the window steps (close, HL/2, HLC/3) and the bar range, 64 channels, kernel 5, '
                     'dilations 1, 2, 4, 8, zero padding, bias, mirror-symmetric; side inputs nt_PRV (window divisor and '
                     'the 200 bars before the window, in bars; no clock); head 209 -> 128 -> 1',
            'weights_source': prod['tag'], 'seed_key': prod['seed_key'], 'seed': cfg['seed'], 'n_weights': int(n_w),
            'training': f"prod: all data of {len(prod['train_assets'])} assets on 5M up to {prod['t_cut']} UTC, "
                        f"{prod['epochs_run']} epochs; research/sumo_class_train.py v6.6",
            'line': f"research/sumo_class_tf.py v{tfl['ver']} MODE '{tfl['mode']}' ({man['line21']}): own ruler of 1H, live "
                    f"part {lv['key']}, floor 0.2, ceiling {ex['ceiling']} (rules v2.1 in the header of the file)",
            'sources': {k: v for k, v in shas.items() if k.startswith((PROD, TFLINE))},
            'manifest': MANIFEST, 'code_version': git_version(),
            'exported': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
            'timeframe': '1H', 'exchange_of_training_data': 'Binance spot'},
        'model': {
            'id': prod['model'], 'source': prod['tag'],
            'conv_w': [nums(w) for w in conv_w], 'conv_b': [nums(b) for b in conv_b], 'dilations': [1, 2, 4, 8],
            'kernel': 5, 'padding': 'zeros, 2 d on each side (same length: 63 places)', 'channels': 64,
            'head0_w': nums(sd['head.0.weight']), 'head0_b': nums(sd['head.0.bias']),
            'head3_w': nums(sd['head.3.weight']), 'head3_b': nums(sd['head.3.bias']),
            's_sd': num(sd['s_sd']), 'lvl_mu': num(sd['lvl_mu'][0]), 'lvl_sd': num(sd['lvl_sd'][0]),
            'side_mu': nums(sd['side_mu']), 'side_sd': nums(sd['side_sd']), 'side_names': list(SIDE_NAMES),
            'side_signed': list(SIDE_SIGNED),
            'layout': {'conv_w[i]': 'torch Conv1d weight (out 64, in, tap 5); in = 4 for the first layer (r, step HL/2, '
                                    'step HLC/3, level), then 64',
                       'head': 'relu(head0_w @ [mean 64, max 64, last 64, side 17] + head0_b) -> head3_w @ . + head3_b'}},
        'bars': {'timeframe': '1H', 'bar_ms': HOUR_MS, 'gap_factor': 1.5, 'n_norm': 200, 'n_hist': 200,
                 'input': 'closed candles of one symbol, oldest first: open time (ms UTC, a multiple of the hour) and OHLC; '
                          'missing hours are not filled in',
                 'norm': 'median(high - low) over the 200 rows before the bar',
                 'valid': 'norm finite and > 0, no grid gap (interval to the previous candle > 1.5 h), previous candle '
                          'exists and has a norm'},
        'window': {'L': 64, 'floor': FLOOR, 'floor_of': 0.2, 'orient_eps': ORIENT_EPS,
                   'rule': '64 valid bars in a row; >= 200 rows before its first bar; D = max(median |r|, floor); '
                           'orientation: flip if x_end < -orient_eps * D (the mirror-symmetric logit does not depend on it)'},
        'line': {
            'ruler': {'kind': ru['kind'], 'variant': ru['variant'], 'knots': ru['knots'], 'zk': ru['zk'], 'q': ru['q'],
                      'n_null': ru['n_null'], 'from': ru['from'],
                      'formula': 'z = interp(logit, q, zk); beyond the outer knots the edge value'},
            'live': {'key': lv['key'], 'family': lv['family'], 'L': lv['L'], 'lag': lv['lag'], 'k': lv['k'],
                     'tau': lv['tau'], 's_lo': lv['s_lo'],
                     'formula': 'p — window number from 0 over the existing windows; the set [lo, hi), hi = p - 63, '
                                'lo = max(0, hi - 4032), n = hi - lo; m = max(s1 / n - k s sqrt(tau / n), 0), s^2 = '
                                'max(s2 / n - (s1 / n)^2, 0); n = 0: m = 0, s = 1'},
            'log_m': 'm+ = max(m, 0); s+ = min(max(s, s_lo), 1); zc = min(z, m+ / (1 - s+^2)) if s+ < 1 else z; '
                     'log M = -(zc - m+)^2 / (2 s+^2) - ln s+ + zc^2 / 2',
            'show_from_window': lv['show_from_window'], 'ceiling': float(ex['ceiling']),
            'output': 'strength = min(exp |log M|, ceiling) for both colours; side = sign of log M: +1 orange (more often '
                      'than random), -1 blue (less often), 0 — M = 1'},
        'history': {'first_window_bar': hi['first_window_bar'], 'first_line_bar': hi['first_line_bar'],
                    'bars_needed': 'first_line_bar + k candles for the line on the last k candles (continuous, from 0)'},
    }


def research_candles(part):
    """Continuous candles of a research part of BTCUSDT 1H with the rows before it (smc_bars + times of smc_series).
       'va': the last TRAINING bar + the 200 bars before VALIDATION + VALIDATION (201 h between the last TRAINING bar and
       the first VALIDATION bar: the rows before VALIDATION are hourly). 'tr': the 200 bars before TRAINING (times — hourly
       before its first bar) + TRAINING. Returns dict time (ms), open, high, low, close and 'first' — row of the first bar of
       the part."""
    with np.load(path(BARS.format('bars'))) as b, np.load(path(BARS.format('series'))) as s:
        t_part = s[f't_{part}'] // 10 ** 6
        pre = {k: b[f'pre_{k}_{part}'] for k in 'ohlc'}
        cur = {k: b[f'{k}_{part}'] for k in 'ohlc'}
        n_pre = len(pre['c'])
        t_pre = t_part[0] - HOUR_MS * np.arange(n_pre, 0, -1)
        if part == 'va':
            t_tr = s['t_tr'] // 10 ** 6
            if t_part[0] - t_tr[-1] != (n_pre + 1) * HOUR_MS:
                raise SystemExit("smc_bars: the 200 rows before VALIDATION are not hourly after the last TRAINING bar")
            last = {k: b[f'{k}_tr'][-1:] for k in 'ohlc'}
            t = np.concatenate([t_tr[-1:], t_pre, t_part])
            px = {k: np.concatenate([last[k], pre[k], cur[k]]) for k in 'ohlc'}
            first = n_pre + 1
        else:
            t = np.concatenate([t_pre, t_part])
            px = {k: np.concatenate([pre[k], cur[k]]) for k in 'ohlc'}
            first = n_pre
    return {'time': t.astype(np.int64), 'open': px['o'], 'high': px['h'], 'low': px['l'], 'close': px['c'], 'first': first}


def main():
    ex = build_export()
    with open(path(EXPORT), 'w', encoding='utf-8') as fh:
        json.dump(ex, fh, ensure_ascii=False, separators=(',', ':'))
    model = f1h.load_model(path(EXPORT))
    with np.load(path(PROD + '_slim.npz')) as z:            # rule R21: the weights back to float32 bit for bit
        same = all(np.array_equal(np.asarray(a, np.float32), z[f'sd_blocks.{i}.0.{k}'])
                   for i in range(4) for k, a in (('weight', model.conv_w[i]), ('bias', model.conv_b[i])))
        same &= all(np.array_equal(np.asarray(getattr(model, a), np.float32), z[f'sd_head.{k}'])
                    for a, k in (('head0_w', '0.weight'), ('head0_b', '0.bias'), ('head3_w', '3.weight'),
                                 ('head3_b', '3.bias')))
        same &= all(np.array_equal(np.float32(np.atleast_1d(getattr(model, a))), np.atleast_1d(z[f'sd_{k}']))
                    for a, k in (('s_sd', 's_sd'), ('lvl_mu', 'lvl_mu'), ('lvl_sd', 'lvl_sd'), ('side_mu', 'side_mu'),
                                 ('side_sd', 'side_sd')))
    if not same:
        raise SystemExit('export weights are not the float32 of the slim npz')
    print('export', EXPORT, os.path.getsize(path(EXPORT)) // 1024, 'KB | weights = slim npz bit for bit')
    c = research_candles('va')
    out = f1h.compute(model, c['time'], c['open'], c['high'], c['low'], c['close'])
    write_golden(path(GOLDEN), c, out)
    print('golden', GOLDEN, os.path.getsize(path(GOLDEN)) // 1024, 'KB | candles', len(c['time']), '| windows',
          int(out['window'].sum()), '| first window bar', int(np.argmax(out['window'])), '| first line bar',
          int(np.argmax(out['line'])), '| with line', int(out['line'].sum()))


if __name__ == '__main__':
    main()
