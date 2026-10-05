"""Neutral export of the indicator model (JSON) and the golden file (bar-by-bar reference on BTC 5M).

    python -m mri.export            # writes export/mri_model.json and tests/golden/btc5m_fresh.csv.gz

The JSON is self-contained: weights, buffers, floors, rules (tie, unknown symbol, warm-up), Platt, thresholds,
passport. Any platform (Pine, MQL, web) computes the same numbers from it; the golden file lets it check bar by bar
from the first bar of the slice (warm-up included and marked).
Weights are written as the shortest decimals that round-trip their float32 values (the same literals as in Pine).
"""
import datetime as dt
import json
import os
import subprocess

import numpy as np
import pandas as pd

from . import BARS_DAY, CTX_BLOCK, CTX_NBLK, GAP_MS, L, N_NORM, SPAN_CTX, SPAN_SIDE, TIE_EPS
from .calibration import THR
from .features import read_bars
from .params import SIDE_NAMES, from_dict, load_npz
from .pine_gen import fmt
from .stream import run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 'a'                                    # rule R4: one seed, chosen in advance
BURN = 2981                                   # rule R7: docs/checks/warmup.json, EMA 'adjust'
POOL_ALL = ['BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'SOLUSDT', 'ADAUSDT', 'BNBUSDT', 'BCHUSDT', 'DOTUSDT', 'UNIUSDT',
            'XRPUSDT', 'LTCUSDT', 'DASHUSDT', 'ZECUSDT', 'SPELLUSDT', 'ICPUSDT', 'NEARUSDT', 'ZENUSDT', 'AVAXUSDT',
            'FETUSDT', 'RAYUSDT', 'CAKEUSDT', 'INJUSDT', 'RUNEUSDT', 'PHAUSDT', 'QNTUSDT', 'HBARUSDT', 'AAVEUSDT',
            'VTHOUSDT', 'FILUSDT', 'LINKUSDT', 'XLMUSDT']
FRESH = ('BTCUSDT', 'ETHUSDT')                # prod model: these two from the fresh data folder


def _num(x):
    return float(fmt(x))


def _nums(a):
    a = np.asarray(a)
    return [_nums(v) for v in a] if a.ndim > 1 else [_num(v) for v in a]


def git_version():
    try:
        h = subprocess.run(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', ROOT, 'status', '--porcelain', 'mri'], capture_output=True, text=True).stdout
        return h + ('+dirty' if dirty.strip() else '')
    except OSError:
        return 'unknown'


def build_export(seed=SEED):
    p, _ = load_npz(os.path.join(ROOT, f'artifacts/weights/prod_F3_g_L64_{seed}_slim.npz'))
    with open(os.path.join(ROOT, f'artifacts/weights/prod_F3_g_L64_{seed}.json'), encoding='utf-8') as fh:
        prod = json.load(fh)
    with open(os.path.join(ROOT, 'artifacts/pack/floors.json'), encoding='utf-8') as fh:
        fl_all = json.load(fh)
    with open(os.path.join(ROOT, 'docs/checks/calibration.json'), encoding='utf-8') as fh:
        cal = json.load(fh)
    floors = {a: _num(fl_all[f'smc_series_{a}_5M_N200.npz']['floor_L64']) for a in POOL_ALL}
    unknown_floor = _num(np.median(list(floors.values())))
    model = {'conv': [_nums(w) for w in p.conv], 'dilations': [1, 2, 4, 8],
             'head0_w': _nums(p.head0_w), 'head0_b': _nums(p.head0_b),
             'head3_w': _nums(p.head3_w), 'head3_b': _nums(p.head3_b),
             's_sd': _num(p.s_sd), 'lvl_mu': 0.0, 'lvl_sd': _num(p.lvl_sd),
             'side_mu': _nums(p.side_mu), 'side_sd': _nums(p.side_sd), 'side_names': list(SIDE_NAMES),
             'n_params': p.n_params}
    pa, pb = cal['platt_final']['a'], cal['platt_final']['b']
    return {
        'format': 'mri-export', 'format_version': 1,
        'indicator': 'Market Randomness Index (MRI)', 'classifier': 'SuMo classifier',
        'passport': {
            'model': 'F3_g (streamable CNN: close, HL/2, HLC/3 steps + bar range; side inputs v7tcp)',
            'weights_source': f'prod_F3_g_L64_{seed}', 'seed_key': seed, 'seed': prod['cfg']['seed'],
            'training': 'prod: all data of 31 assets (FIT + TUNE + VALIDATION), 15 epochs, cosine lr',
            'data_end': prod['t_cut'] + ' UTC', 'train_assets': prod['train_assets'],
            'code_version': git_version(), 'exported': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
            'timeframe': '5M', 'exchange_of_training_data': 'Binance spot'},
        'model': model,
        'inputs': {
            'bar_time': 'open time of the bar, UTC milliseconds',
            'norm': f'median(high - low) over the {N_NORM} bars before the bar (closed left); even count: mean of the two middle',
            'valid_bar': f'norm finite and > 0, no grid gap (interval > {int(GAP_MS)} ms), previous bar exists, '
                         'previous bar has a norm (for d of HL/2, HLC/3)',
            'r': '(close - close[1]) / norm',
            'step_p': '(p - close[1]) / norm - (p[1] - close[1]) / norm[1], p = HL/2 and HLC/3',
            'g': '(high - low) / norm',
            'conv_input_plus': '[r, step_hl2, step_hlc3] / s_sd, (g - lvl_mu) / lvl_sd',
            'conv_input_minus': '[-r, -step_hl2, -step_hlc3] / s_sd, (g - lvl_mu) / lvl_sd',
            'window': f'last {L} bars, all valid',
            'divisor_D': 'max(median |r| over the window (even: mean of two middle), floor of the symbol)',
            'pooling': 'layer 4 outputs at the last 33 bars of the chosen stream: mean, max, last; each / D',
            'ema': f'normalized EMA (pandas adjust=True, NaN skipped) of |r| over valid bars, spans {SPAN_SIDE} and {SPAN_CTX}',
            'side': ['vol_to_7d = log(D) - log(EMA|r| span 2016 at the last bar)',
                     'hour_sin, hour_cos = sin/cos(2 pi * (minutes of day / 60) / 24) of the last bar',
                     'dow_sin, dow_cos = sin/cos(2 pi * dow / 7), dow = (floor(time / 86400000) + 3) % 7, Monday = 0',
                     f'move_1d / 3d / 7d: sum of r over the {BARS_DAY} / {3 * BARS_DAY} / {7 * BARS_DAY} bars before the '
                     'bar preceding the window, / (sqrt(n) * EMA|r| span 288 at the bar preceding the window) * sgn',
                     f'path_*: same over {CTX_NBLK} blocks of {CTX_BLOCK} bars (last 128 bars before the bar preceding '
                     'the window), oldest first'],
            'side_standardize': '(side - side_mu) / side_sd',
            'head': 'logit = head3_w . relu(head0_w @ [pooled(96), side_std(16)] + head0_b) + head3_b'},
        'tie_rule': {'x_end': 'sum_{i=0..63} i * r_i / 64 over the window (end of the centered close path)',
                     'eps': TIE_EPS, 'rule': 'sgn = -1 if x_end < -eps * D else +1; one sign for the stream and the side inputs'},
        'floors': {'floor_L64': floors, 'unknown_symbol': unknown_floor,
                   'unknown_rule': 'median of floor_L64 of the 31 table symbols',
                   'provisional': list(FRESH),
                   'provisional_note': 'BTC / ETH: values of the research data folder (verified end to end on BTC VAL); '
                                       'the prod model trained BTC / ETH on the fresh folder - its floors to be exported '
                                       '(scripts/colab_export_floors.py)'},
        'calibration': {'platt_a': _num(pa), 'platt_b': _num(pb), 'lift': 'exp(platt_a * logit + platt_b)',
                        'M': 'max(lift, 1 / lift) = exp(|platt_a * logit + platt_b|)',
                        'fit_on': cal['platt_final']['fit_on'], 'rule': 'docs/rules.md R4',
                        'check': {'K1': cal['K1']['ok'], 'K2': cal['K2']['ok'], 'K3': cal['K3']['ok']},
                        'unknown_symbol': 'same Platt (global), note in the corner'},
        'thresholds': {'mark_M': THR, 'orange': f'lift >= {THR}', 'blue': f'lift <= {1 / THR:g}'},
        'warmup': {'burn_bars_from_first_valid_r': BURN, 'ema_start': 'normalized (adjust=True)',
                   'context_bars': L + 1 + 7 * BARS_DAY, 'rule': 'docs/rules.md R7, docs/checks/warmup.json'},
    }


GOLDEN_COLS = ['time', 'open', 'high', 'low', 'close', 'norm', 'ok', 'r', 'step_hl2', 'step_hlc3', 'g', 'D', 'x_end',
               'sgn'] + [f'side_{n}' for n in SIDE_NAMES] + ['logit', 'lift', 'M', 'shown']


def golden(ex, bars_path, floor):
    p = from_dict(ex['model'])
    bars = read_bars(bars_path)
    res = run(bars, p, floor, eps=ex['tie_rule']['eps'], ema_adjust=True,
              platt=(ex['calibration']['platt_a'], ex['calibration']['platt_b']))
    burn = ex['warmup']['burn_bars_from_first_valid_r']
    rows = []
    for i, x in enumerate(res):
        side = x.get('side_raw', [np.nan] * 16)
        fr = x['first_r']
        shown = int(np.isfinite(x['logit']) and fr is not None and x['bar'] - fr >= burn)
        rows.append([x['time'], bars['open'].iat[i], bars['high'].iat[i], bars['low'].iat[i], bars['close'].iat[i],
                     x['norm'], int(x['ok']), x['r'], x['step_hl2'], x['step_hlc3'], x['g'], x.get('D', np.nan),
                     x.get('x_end', np.nan), x.get('sgn', np.nan)] + list(side)
                    + [x['logit'], x.get('lift', np.nan), x.get('M', np.nan), shown])
    return pd.DataFrame(rows, columns=GOLDEN_COLS)


def main():
    ex = build_export()
    os.makedirs(os.path.join(ROOT, 'export'), exist_ok=True)
    path = os.path.join(ROOT, 'export/mri_model.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(ex, fh, ensure_ascii=False, separators=(',', ':'))
    print('export', path, os.path.getsize(path) // 1024, 'KB')
    g = golden(ex, os.path.join(ROOT, 'artifacts/bars/btc_fresh_tail.csv.gz'), ex['floors']['floor_L64']['BTCUSDT'])
    os.makedirs(os.path.join(ROOT, 'tests/golden'), exist_ok=True)
    gp = os.path.join(ROOT, 'tests/golden/btc5m_fresh.csv.gz')
    g.to_csv(gp, index=False, float_format='%.12g', compression={'method': 'gzip', 'mtime': 0})
    print('golden', gp, os.path.getsize(gp) // 1024, 'KB', '| bars', len(g), '| shown', int(g['shown'].sum()),
          '| first shown bar', int(np.argmax(g['shown'].to_numpy() > 0)))


if __name__ == '__main__':
    main()
