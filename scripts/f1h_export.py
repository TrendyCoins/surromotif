"""Export of F1 by timeframe (rules R21, R28, docs/rules.md) and the compact golden file (rules R24, R31).

    python scripts/f1h_export.py --tf 5M | 15M | 30M   # export/f1h_model_<tf>.json and tests/golden/btc<tf>_f1h.csv.gz
    python scripts/f1h_export.py --tf 1H --out DIR     # 1H only into another folder: export/f1h_model.json and
                                                       # tests/golden/btc1h_f1h.csv.gz are held by the lock of the site
                                                       # generator; DIR — to compare with them (regression)

Sources (artifacts/, read only; size and sha256 checked against the manifest of the timeframe before anything is counted):
the weights and cfg of prod a (prod_SP_l025_d1_L64_a.json and _slim.npz — artifacts/pack/site1h_manifest.json, the same files
on every timeframe) and the json of the line — section 'export': ruler, live part, floor, ceiling, history:
    1H        tfline21_20261022_063520.json (research/sumo_class_tf.py v2.1 'line21'), manifest site1h
    15M, 30M  tfline21_15m_ / _30m_<stamp>.json (the same 'line21' through scripts/f1h_tf_line21.py), manifests
              site15m, site30m
    5M        site_20261021_084916.json (research/sumo_class_site.py v1.1 'run'; no tf and history fields: the timeframe is
              5M, the history — rule R20), manifest site5m
The ceiling is the checked one (5M and 15M — 10; the site shows up to 8, the generator cuts). The export is self-contained:
f1h needs nothing else at run time. Weights are exact decimals of their float32 values.

The golden file: continuous BTCUSDT candles of the timeframe from its research files (smc_bars + times of smc_series) — the
last bar of TRAINING, the 200 bars before VALIDATION and VALIDATION (the windows and their numbers are those of the research
golden) — and the values of the reference on every candle; 5M — the piece from the start of length first_line_bar + 20 160
(rule R31). research_candles() is shared with scripts/f1h_parity.py.
"""
import argparse
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
from f1h.model import tf_ms  # noqa: E402

WEIGHTS_MANIFEST = 'artifacts/pack/site1h_manifest.json'
PROD = 'artifacts/weights/prod_SP_l025_d1_L64_a'
# per timeframe: manifest, json of the line, its kind, the golden of the source, research files BTC, outputs and what the
# line must be (the verdict of 'line21' / Site recorded in docs/progress.md; anything else — no export)
TFS = {
    '1H': {'manifest': WEIGHTS_MANIFEST, 'line': 'artifacts/calib/tfline21_20261022_063520.json', 'kind': 'line21',
           'golden_src': 'artifacts/calib/tfline21_golden_20261022_063520.npz',
           'bars': 'artifacts/bars/smc_{}_BTCUSDT_1H_N200.npz',
           'export': 'export/f1h_model.json', 'golden': 'tests/golden/btc1h_f1h.csv.gz',
           'live': ('4032·k0.5', 'F3', 4032, 0, 0.5, 64), 'ceiling': 8, 'golden_k': None},
    '15M': {'manifest': 'artifacts/pack/site15m_manifest.json', 'line': 'artifacts/calib/tfline21_15m_20261027_084749.json',
            'kind': 'line21', 'golden_src': 'artifacts/calib/tfline21_15m_golden_20261027_084749.npz',
            'bars': 'artifacts/bars/smc_{}_BTCUSDT_15M_N200.npz',
            'export': 'export/f1h_model_15m.json', 'golden': 'tests/golden/btc15m_f1h.csv.gz',
            'live': ('4032+288', 'F2', 4032, 288, None, None), 'ceiling': 10, 'golden_k': None},
    '30M': {'manifest': 'artifacts/pack/site30m_manifest.json', 'line': 'artifacts/calib/tfline21_30m_20261027_090105.json',
            'kind': 'line21', 'golden_src': 'artifacts/calib/tfline21_30m_golden_20261027_090105.npz',
            'bars': 'artifacts/bars/smc_{}_BTCUSDT_30M_N200.npz',
            'export': 'export/f1h_model_30m.json', 'golden': 'tests/golden/btc30m_f1h.csv.gz',
            'live': ('4032·k0.5', 'F3', 4032, 0, 0.5, 64), 'ceiling': 8, 'golden_k': None},
    '5M': {'manifest': 'artifacts/pack/site5m_manifest.json', 'line': 'artifacts/calib/site_20261021_084916.json',
           'kind': 'site', 'golden_src': 'artifacts/calib/site_golden_20261021_084916.npz',
           'bars': 'artifacts/bars/btc5m_compact/smc_{}_BTCUSDT_5M_N200.npz',
           'export': 'export/f1h_model_5m.json', 'golden': 'tests/golden/btc5m_f1h.csv.gz',
           'live': ('4032·k0.5', 'F3', 4032, 0, 0.5, 64), 'ceiling': 10, 'golden_k': 20160},
}
FLOOR = float(np.float32(0.2))
ORIENT_EPS = 1e-4                       # research ORIENT_EPS (v6.0)
FIRST_WINDOW_BAR = 264                  # rule R20: continuous candles
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


def check_manifest(manifest):
    """Every file of the manifest: size and sha256. Returns (manifest, {path: sha256})."""
    with open(path(manifest), encoding='utf-8') as fh:
        man = json.load(fh)
    out = {}
    for f in man['files']:
        p = path(f['path'])
        if not os.path.isfile(p) or os.path.getsize(p) != f['bytes'] or sha256(p) != f['sha256']:
            raise SystemExit(f"{f['path']}: not the file of {manifest} — no export")
        out[f['path']] = f['sha256']
    return man, out


def check_sources(tf):
    """Manifests of the timeframe and of the weights (rule R28). Returns (manifest of the tf, {path: sha256} of both)."""
    c = TFS[tf]
    wman, wshas = check_manifest(WEIGHTS_MANIFEST)
    if tf == '1H':
        return wman, wshas
    man, shas = check_manifest(c['manifest'])
    pj = PROD + '.json'
    if man['weights']['manifest'] != WEIGHTS_MANIFEST or man['weights']['prod_json_sha256'] != wshas[pj]:
        raise SystemExit(f"{c['manifest']}: the weights of prod a are not those of {WEIGHTS_MANIFEST} — no export")
    if man['tf'] != tf or man.get('line21', man.get('site')) != stamp(tf) or c['line'] not in shas \
            or c['golden_src'] not in shas or not all(c['bars'].format(k) in shas for k in ('series', 'prices', 'range',
                                                                                              'bars')):
        raise SystemExit(f"{c['manifest']}: not the files of {tf} — no export")
    return man, {**shas, **{k: v for k, v in wshas.items() if k.startswith(PROD)}}


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


def tf_words(tf):
    """(the multiple of the open time, the grid gap) in words: 1H — 'the hour', '1.5 h' (the texts of the 1H export)."""
    if tf == '1H':
        return 'the hour', '1.5 h', 'hours'
    mins = tf_ms(tf) // 60_000
    return f'{mins} minutes', f'{1.5 * mins:g} min', 'bars'


def live_formula(lv):
    """The live part in words (the 1H export: the text of F3 with its numbers)."""
    head = (f"p — window number from 0 over the existing windows; the set [lo, hi), hi = p - {lv['lag'] - 1}, "
            f"lo = max(0, hi - {lv['L']}), n = hi - lo; ")
    if lv['family'] == 'F3':
        return head + ('m = max(s1 / n - k s sqrt(tau / n), 0), s^2 = max(s2 / n - (s1 / n)^2, 0); '
                       'n = 0: m = 0, s = 1')
    n0 = lv['n0']
    return head + (f"m = s1 / (n + {n0}); s^2 = (max(s2 - s1^2 / n, 0) + {n0}) / (n + {n0}), no k correction "
                   f"({n0} virtual random windows with z ~ N(0, 1)); n = 0: m = 0, s = 1")


def stamp(tf):
    """Stamp of the run of the line: the end of the name of its json (YYYYMMDD_HHMMSS)."""
    return os.path.splitext(os.path.basename(TFS[tf]['line']))[0][-15:]


def line_source(tf):
    """Section 'export' of the json of the line, checked against what the line of the timeframe must be. Returns
       (json, export section, ruler, live, history, text of the passport)."""
    c = TFS[tf]
    with open(path(c['line']), encoding='utf-8') as fh:
        src = json.load(fh)
    ex = src['export']
    ru, lv = dict(ex['ruler']), ex['live']
    key, fam, L, n0, k, tau = c['live']
    want_live = (lv['key'], lv['family'], lv['L'], lv['n0'], lv['k'], lv['tau'], lv['lag'], lv['show_from_window'],
                 lv['s_lo']) == (key, fam, L, n0, k, tau, 64, 4096, 0.75)
    if c['kind'] == 'line21':
        hi = ex['history']
        ok = src['mode'] == 'line21' and ex['tf'] == tf and src['tf'] == tf and ex['rules'] == 'v2.1'
        text = (f"research/sumo_class_tf.py v{src['ver']} MODE '{src['mode']}' ({stamp(tf)})"
                + ('' if tf == '1H' else ' through scripts/f1h_tf_line21.py') +
                f": own ruler of {tf}, live part {lv['key']}, floor 0.2, ceiling {ex['ceiling']} (rules v2.1 in the header "
                f"of the file)")
        if tf != '1H':                  # research/sumo_class_tf.py writes 'VAL 1H' into this text on every timeframe
            ru['from'] = ru['from'].replace('VAL 1H', f'VAL {tf}')
    else:                               # SuMo_Class_Site v1.1 'run': no tf and history fields (rule R28)
        hi = {'first_window_bar': FIRST_WINDOW_BAR, 'first_line_bar': FIRST_WINDOW_BAR + lv['show_from_window']}
        single = src['rulers']['single']
        ok = (src['mode'] == 'run' and src['ver'] == '1.1' and ex['config'] == 'single' and single['q'] == ru['q']
              and single['n_null'] == ru['n_null'] and single['ceiling'] == ex['ceiling'])
        ru.setdefault('variant', 'own')
        ru['from'] = (f"prod_SP_l025_d1_L64_a: суррогаты VAL 5M ({single['n_assets']} ассет), пол {ex['floor']:g} "
                      f"(SuMo_Class_Site v1.1, rulers['single'])")
        text = (f"research/sumo_class_site.py v{src['ver']} MODE '{src['mode']}' ({stamp(tf)}): ruler of prod a on 5M, live "
                f"part {lv['key']}, floor 0.2, ceiling {ex['ceiling']} as checked (rules of SuMo_Class_Site v1.1)")
    ok = ok and want_live and ex['seeds'] == ['a'] and ex['floor'] == 0.2 and ex['ceiling'] == c['ceiling'] \
        and (hi['first_window_bar'], hi['first_line_bar']) == (FIRST_WINDOW_BAR, FIRST_WINDOW_BAR + 4096) \
        and len(ru['knots']) == 37 and len(ru['q']) == 37 and len(ru['zk']) == 37 and bool(np.all(np.diff(ru['q']) > 0))
    if not ok:
        raise SystemExit(f"{c['line']}: section 'export' is not the line of f1h on {tf} ({key}, floor 0.2, ceiling "
                         f"{c['ceiling']}) — no export")
    if tf != '1H':
        text += '; the site shows the strength up to 8 (its generator cuts)' if ex['ceiling'] > 8 else ''
    return src, ex, ru, lv, hi, text


def build_export(tf):
    man, shas = check_sources(tf)
    c = TFS[tf]
    with open(path(PROD + '.json'), encoding='utf-8') as fh:
        prod = json.load(fh)
    cfg = prod['cfg']
    bad = {k: (cfg.get(k), v) for k, v in CFG_WANT.items() if cfg.get(k) != v}
    if bad or prod['tag'] != man['prod'] or prod['model'] != 'SP_l025_d1' or prod['seed_key'] != 'a':
        raise SystemExit(f"prod a cfg is not the network of f1h: {bad} — no export")
    with np.load(path(PROD + '_slim.npz')) as z:
        sd = {k[3:]: z[k] for k in z.files}
    _, ex, ru, lv, hi, line_text = line_source(tf)
    conv_w = [sd[f'blocks.{i}.0.weight'] for i in range(4)]
    conv_b = [sd[f'blocks.{i}.0.bias'] for i in range(4)]
    n_w = sum(v.size for k, v in sd.items() if k.startswith(('blocks.', 'head.')))
    mult, gap, missing = tf_words(tf)
    live = {'key': lv['key'], 'family': lv['family'], 'L': lv['L'], 'lag': lv['lag']}
    if lv['family'] == 'F2':
        live['n0'] = lv['n0']
    live.update({'k': lv['k'], 'tau': lv['tau'], 's_lo': lv['s_lo'], 'formula': live_formula(lv)})
    passport = {
        'model': 'SP_l025_d1: CNN over the window steps (close, HL/2, HLC/3) and the bar range, 64 channels, kernel 5, '
                 'dilations 1, 2, 4, 8, zero padding, bias, mirror-symmetric; side inputs nt_PRV (window divisor and '
                 'the 200 bars before the window, in bars; no clock); head 209 -> 128 -> 1',
        'weights_source': prod['tag'], 'seed_key': prod['seed_key'], 'seed': cfg['seed'], 'n_weights': int(n_w),
        'training': f"prod: all data of {len(prod['train_assets'])} assets on 5M up to {prod['t_cut']} UTC, "
                    f"{prod['epochs_run']} epochs; research/sumo_class_train.py v6.6",
        'line': line_text,
        'sources': {k: v for k, v in shas.items() if k.startswith((PROD, c['line']))},
        'manifest': c['manifest']}
    if tf != '1H':
        passport['weights_manifest'] = WEIGHTS_MANIFEST
    passport.update({'code_version': git_version(),
                     'exported': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
                     'timeframe': tf, 'exchange_of_training_data': 'Binance spot'})
    return {
        'format': f1h.model.FORMAT, 'format_version': f1h.model.FORMAT_VERSION,
        'what': f'SuMo classifier F1 (SP_l025_d1 prod a) and its line M on {tf} candles; reference: the f1h package',
        'passport': passport,
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
        'bars': {'timeframe': tf, 'bar_ms': tf_ms(tf), 'gap_factor': 1.5, 'n_norm': 200, 'n_hist': 200,
                 'input': f'closed candles of one symbol, oldest first: open time (ms UTC, a multiple of {mult}) and OHLC; '
                          f'missing {missing} are not filled in',
                 'norm': 'median(high - low) over the 200 rows before the bar',
                 'valid': f'norm finite and > 0, no grid gap (interval to the previous candle > {gap}), previous candle '
                          'exists and has a norm'},
        'window': {'L': 64, 'floor': FLOOR, 'floor_of': 0.2, 'orient_eps': ORIENT_EPS,
                   'rule': '64 valid bars in a row; >= 200 rows before its first bar; D = max(median |r|, floor); '
                           'orientation: flip if x_end < -orient_eps * D (the mirror-symmetric logit does not depend on it)'},
        'line': {
            'ruler': {'kind': ru['kind'], 'variant': ru['variant'], 'knots': ru['knots'], 'zk': ru['zk'], 'q': ru['q'],
                      'n_null': ru['n_null'], 'from': ru['from'],
                      'formula': 'z = interp(logit, q, zk); beyond the outer knots the edge value'},
            'live': live,
            'log_m': 'm+ = max(m, 0); s+ = min(max(s, s_lo), 1); zc = min(z, m+ / (1 - s+^2)) if s+ < 1 else z; '
                     'log M = -(zc - m+)^2 / (2 s+^2) - ln s+ + zc^2 / 2',
            'show_from_window': lv['show_from_window'], 'ceiling': float(ex['ceiling']),
            'output': 'strength = min(exp |log M|, ceiling) for both colours; side = sign of log M: +1 orange (more often '
                      'than random), -1 blue (less often), 0 — M = 1'},
        'history': {'first_window_bar': hi['first_window_bar'], 'first_line_bar': hi['first_line_bar'],
                    'bars_needed': 'first_line_bar + k candles for the line on the last k candles (continuous, from 0)'},
    }


def research_candles(part, tf='1H'):
    """Continuous candles of a research part of BTCUSDT on tf with the rows before it (smc_bars + times of smc_series).
       'va': the last TRAINING bar + the 200 bars before VALIDATION + VALIDATION (201 bars between the last TRAINING bar
       and the first VALIDATION bar: the rows before VALIDATION are continuous). 'tr': the 200 bars before TRAINING (times —
       continuous before its first bar) + TRAINING (5M: the compact files — the tail of TRAINING and the 200 bars before
       it). Returns dict time (ms), open, high, low, close and 'first' — row of the first bar of the part."""
    bars, bar_ms = TFS[tf]['bars'], tf_ms(tf)
    with np.load(path(bars.format('bars'))) as b, np.load(path(bars.format('series'))) as s:
        t_part = s[f't_{part}'] // 10 ** 6
        pre = {k: b[f'pre_{k}_{part}'] for k in 'ohlc'}
        cur = {k: b[f'{k}_{part}'] for k in 'ohlc'}
        n_pre = len(pre['c'])
        t_pre = t_part[0] - bar_ms * np.arange(n_pre, 0, -1)
        if part == 'va':
            t_tr = s['t_tr'] // 10 ** 6
            if t_part[0] - t_tr[-1] != (n_pre + 1) * bar_ms:
                raise SystemExit(f"smc_bars {tf}: the 200 rows before VALIDATION are not continuous after the last "
                                 f"TRAINING bar")
            last = {k: b[f'{k}_tr'][-1:] for k in 'ohlc'}
            t = np.concatenate([t_tr[-1:], t_pre, t_part])
            px = {k: np.concatenate([last[k], pre[k], cur[k]]) for k in 'ohlc'}
            first = n_pre + 1
        else:
            t = np.concatenate([t_pre, t_part])
            px = {k: np.concatenate([pre[k], cur[k]]) for k in 'ohlc'}
            first = n_pre
    return {'time': t.astype(np.int64), 'open': px['o'], 'high': px['h'], 'low': px['l'], 'close': px['c'], 'first': first}


def golden_candles(tf, model):
    """Candles of the compact golden (rule R31): research_candles('va'); 5M — its piece from the start of length
       first_line_bar + golden_k."""
    c = research_candles('va', tf)
    k = TFS[tf]['golden_k']
    if k is not None:
        n = model.first_line_bar + k
        c = {**{key: v[:n] for key, v in c.items() if key != 'first'}, 'first': c['first']}
    return c


def check_weights(model):
    """Rule R21 / R28: the weights of the export back to float32 bit for bit."""
    with np.load(path(PROD + '_slim.npz')) as z:
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--tf', required=True, choices=sorted(TFS))
    ap.add_argument('--out', default=None, help='folder for the export and the golden (default: export/, tests/golden/)')
    a = ap.parse_args()
    c = TFS[a.tf]
    if a.out is None and a.tf == '1H':
        raise SystemExit("1H: export/f1h_model.json and tests/golden/btc1h_f1h.csv.gz are not rewritten — pass --out DIR")
    out_ex = path(c['export']) if a.out is None else os.path.join(a.out, os.path.basename(c['export']))
    out_g = path(c['golden']) if a.out is None else os.path.join(a.out, os.path.basename(c['golden']))
    ex = build_export(a.tf)
    os.makedirs(os.path.dirname(out_ex), exist_ok=True)
    with open(out_ex, 'w', encoding='utf-8') as fh:
        json.dump(ex, fh, ensure_ascii=False, separators=(',', ':'))
    model = f1h.load_model(out_ex, tf=a.tf)
    check_weights(model)
    print('export', out_ex, os.path.getsize(out_ex) // 1024, 'KB | weights = slim npz bit for bit | live',
          model.live_family, ex['line']['live']['key'], '| ceiling', model.ceiling)
    cd = golden_candles(a.tf, model)
    out = f1h.compute(model, cd['time'], cd['open'], cd['high'], cd['low'], cd['close'])
    write_golden(out_g, cd, out)
    print('golden', out_g, os.path.getsize(out_g) // 1024, 'KB | candles', len(cd['time']), '| windows',
          int(out['window'].sum()), '| first window bar', int(np.argmax(out['window'])), '| first line bar',
          int(np.argmax(out['line'])), '| with line', int(out['line'].sum()))


if __name__ == '__main__':
    main()
