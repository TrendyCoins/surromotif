"""Neutral export of the NT3_PRV_ch48 indicator model (JSON) and its golden file (bar-by-bar reference on BTC 5M).

    python -m mri.export_nt3      # writes export/mri_model_nt3.json and tests/golden/btc5m_fresh_nt3.csv.gz

The JSON is self-contained: weights, buffers, the 17 side inputs (order, formulas, clips, sign, NaN), tie rule, floors (R6),
per-symbol Platt and ceilings (R12, R11), unknown symbol, warm-up (R13), passport. Weights are exact decimals of their
float32 values (Pine decodes the same bits from dense strings). The golden file lets any platform check bar by bar from
the first bar of the slice (warm-up included and marked).
The F3_g export (export/mri_model.json, mri/export.py) stays as the legacy model.
"""
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

from . import GAP_MS, L, N_NORM, TIE_EPS
from .export import FRESH, POOL_ALL, git_version
from .features import read_bars
from .nt3 import (MODEL_ID, NT_BODY_FLOOR, NT_CLIP_LOG, NT_CLIP_Z, NT_GEO, NT_HIST, NT_SHORT, SIDE_COLS, SIDE_NAMES,
                  SIDE_SIGNED, run)
from .params import from_dict, load_npz
from .pine_gen import fmt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 'a'                                   # rule R12: one seed, chosen in advance
EXPORT = 'export/mri_model_nt3.json'
GOLDEN = 'tests/golden/btc5m_fresh_nt3.csv.gz'
FIRST_LINE_BAR = 264                         # rule R13 (continuous data), confirmed by the reference


def _num(x):
    return float(fmt(x))


def _nums(a):
    a = np.asarray(a)
    return [_nums(v) for v in a] if a.ndim > 1 else [_num(v) for v in a]


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding='utf-8') as fh:
        return json.load(fh)


SIDE_SPEC = [
    'log_D = log(D), D = max(median |r| over the 64 bars of the window, floor of the symbol)',
    'log_mean_abs_r = log(max(mean |r| over the 64 bars of the window, floor of the symbol))',
] + [f'path_{n}{"ab"[k - 6] if n == 2 else ""}: chunk {k + 1}/8 of the path before the window ({n} bars; chunks of '
     f'{", ".join(map(str, NT_GEO))} bars over bars s-200 .. s-2, oldest first): (close of its last bar - open of its first '
     f'bar) / sqrt({n}) / rmed, clipped to +-{NT_CLIP_Z:g}, times the orientation sign' for k, n in enumerate(NT_GEO)] + [
    'range_pos = 2 * (close[s-2] - lmin) / (hmax - lmin) - 1 (0 if hmax = lmin), hmax / lmin = max high / min low over '
    'bars s-200 .. s-2 (199 bars), times the orientation sign',
    f'range_width = log(max(hmax - lmin, 1e-12) / (sqrt(199) * rmed)), clipped to +-{NT_CLIP_LOG:g}, no sign',
    f'close_dev = (close[s-2] - mean close over bars s-200 .. s-2) / (sqrt(199) * rmed), clipped to +-{NT_CLIP_Z:g}, times '
    'the orientation sign',
    f'vol_16 = log(max(mean(high - low) over bars s-16 .. s-1 / rmean, 0.001)), clipped to +-{NT_CLIP_LOG:g}, no sign',
    f'vol_64 = log(max(mean(high - low) over bars s-64 .. s-1 / rmean, 0.001)), clipped to +-{NT_CLIP_LOG:g}, no sign',
    f'range_mean_med = log(max(rmean / rmed, 0.001)), clipped to +-{NT_CLIP_LOG:g}, no sign',
    f'body_med = log(max(bmed / rmed, 0.001)), clipped to +-{NT_CLIP_LOG:g}, no sign']


def build_export(seed=SEED):
    p, _ = load_npz(os.path.join(ROOT, f'artifacts/weights/prod_{MODEL_ID}_L64_{seed}_slim.npz'))
    prod = _load(f'artifacts/weights/prod_{MODEL_ID}_L64_{seed}.json')
    fl_dir = _load('artifacts/pack/floors_by_folder.json')
    cal = _load('docs/checks/nt3/calibration.json')
    curve = _load('docs/checks/nt3/calib_curve.json')
    ceil = _load('docs/checks/nt3/ceilings.json')
    if p.lvl_mu != 0.0:
        raise ValueError('lvl_mu must be 0 (streaming)')
    # rule R6: the folder the prod model trained on — fresh for BTC / ETH, research for the rest (gate (c): same data end)
    floors = {a: _num(fl_dir['SMC/data_fresh/' if a in FRESH else 'SMC/data/'][a]['floor_L64']) for a in POOL_ALL}
    unknown_floor = _num(np.median(list(floors.values())))
    platt = {a: {'platt_a': _num(cal['platt_asset'][a]['all'][0]), 'platt_b': _num(cal['platt_asset'][a]['all'][1])}
             for a in POOL_ALL}
    ua, ub = cal['unknown_symbol']['platt']
    ch, hidden = p.conv[0].shape[0], p.head0_w.shape[0]
    model = {'id': MODEL_ID, 'conv': [_nums(w) for w in p.conv], 'dilations': [1, 2, 4, 8], 'kernel': 3, 'channels': ch,
             'head0_w': _nums(p.head0_w), 'head0_b': _nums(p.head0_b),
             'head3_w': _nums(p.head3_w), 'head3_b': _nums(p.head3_b),
             's_sd': _num(p.s_sd), 'lvl_mu': 0.0, 'lvl_sd': _num(p.lvl_sd),
             'side_mu': _nums(p.side_mu), 'side_sd': _nums(p.side_sd), 'side_names': list(SIDE_NAMES),
             'side_cols_research': list(SIDE_COLS), 'side_signed': list(SIDE_SIGNED), 'head_hidden': hidden,
             'n_params': p.n_params}
    return {
        'format': 'mri-export', 'format_version': 2,
        'indicator': 'Market Randomness Index (MRI)', 'classifier': 'SuMo classifier',
        'passport': {
            'model': f'{MODEL_ID} (streamable CNN: close, HL/2, HLC/3 steps + bar range, {ch} channels, head '
                     f'{3 * ch + len(SIDE_NAMES)} -> {hidden} -> 1; side inputs: window divisor and the 200 bars before '
                     'the window, in bars; no clock, no EMA)',
            'weights_source': f'prod_{MODEL_ID}_L64_{seed}', 'seed_key': seed, 'seed': prod['cfg']['seed'],
            'training': f"prod: all data of 31 assets (FIT + TUNE + VALIDATION), {prod['cfg']['epochs']} epochs, "
                        'cosine lr', 'research_code': 'research/sumo_class_train.py v6.2',
            'data_end': prod['t_cut'] + ' UTC', 'train_assets': prod['train_assets'],
            'code_version': git_version(), 'exported': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
            'timeframe': '5M', 'exchange_of_training_data': 'Binance spot'},
        'model': model,
        'inputs': {
            'bar_time': 'open time of the bar, UTC milliseconds (only for grid gaps)',
            'norm': f'median(high - low) over the {N_NORM} bars before the bar (closed left); even count: mean of the two middle',
            'valid_bar': f'norm finite and > 0, no grid gap (interval > {int(GAP_MS)} ms on 5M; 1.5 bars in general), '
                         'previous bar exists, previous bar has a norm (for d of HL/2, HLC/3)',
            'r': '(close - close[1]) / norm',
            'step_p': '(p - close[1]) / norm - (p[1] - close[1]) / norm[1], p = HL/2 and HLC/3',
            'g': '(high - low) / norm',
            'conv_input_plus': '[r, step_hl2, step_hlc3] / s_sd, (g - lvl_mu) / lvl_sd',
            'conv_input_minus': '[-r, -step_hl2, -step_hlc3] / s_sd, (g - lvl_mu) / lvl_sd',
            'window': f'last {L} bars t-63 .. t, all valid; first bar s = t - 63 needs >= {NT_HIST} bars before it',
            'divisor_D': 'max(median |r| over the window (even: mean of two middle), floor of the symbol)',
            'pooling': f'layer 4 outputs at the last 33 bars of the chosen stream: mean, max, last ({3 * ch}); each / D',
            'history': {'bars': f'{NT_HIST} bars s-200 .. s-1 before the window, consecutive bars (time and grid gaps are '
                                 'not looked at); nothing deeper; the bar s-1 enters only the scales and vol inputs',
                        'scales': f'rmed = median(high - low) over s-200 .. s-1 (= norm of bar s); rmean = mean(high - low) '
                                  f'over s-200 .. s-1; bmed = max(median |close - open| over s-200 .. s-1, '
                                  f'{NT_BODY_FLOOR} * rmed)',
                        'chunks': f'{list(NT_GEO)} bars, oldest first, over s-200 .. s-2', 'short_means': list(NT_SHORT)},
            'side': SIDE_SPEC,
            'side_nan': 'NaN -> 0 after the clips (cannot occur on a valid window: rmed > 0)',
            'side_sign': 'the orientation sign of the window (tie_rule) multiplies the signed inputs (model.side_signed)',
            'side_standardize': '(side - side_mu) / side_sd',
            'head': f'logit = head3_w . relu(head0_w @ [pooled({3 * ch}), side_std({len(SIDE_NAMES)})] + head0_b) + head3_b'},
        'tie_rule': {'x_end': 'sum_{i=0..63} i * r_i / 64 over the window (end of the centered close path)',
                     'eps': TIE_EPS, 'rule': 'sgn = -1 if x_end < -eps * D else +1; one sign for the stream and the side inputs'},
        'floors': {'floor_L64': floors, 'unknown_symbol': unknown_floor,
                   'unknown_rule': 'median of floor_L64 of the 31 table symbols',
                   'source': 'smc_series files of the prod training: fresh data folder for '
                             + ', '.join(FRESH) + ', research data folder for the rest'},
        'calibration': {'per_symbol': platt, 'lift': 'exp(platt_a * logit + platt_b), Platt of the symbol',
                        'M': 'max(lift, 1 / lift) = exp(|platt_a * logit + platt_b|)',
                        'fit_on': "VAL of the 'refit' models on the prod set of 31 assets (training up to 2025-06-30), "
                                  'seeds a + b together, whole VAL of the symbol', 'rule': 'docs/rules.md R12 (V2)',
                        'unknown_symbol': {'platt_a': _num(ua), 'platt_b': _num(ub),
                                           'fit_on': "VAL of the 'refit' models on set C (FIL, LINK, XLM unseen), 31 "
                                                     'assets, seeds a + b'},
                        'transfer_check': {'rule': "per symbol, 'stop' C -> 'refit' C (v5.2), |total - 1| <= 0.10 at 1.25 "
                                                   'and 0.8', 'not_passed': cal['transfer']['fails'],
                                           'borderline': cal['transfer']['borderline']}},
        'display': {'line': 'M = exp(|platt_a * logit + platt_b|); more frequent than random if platt_a * logit + platt_b > 0, '
                            'less frequent if < 0; floor 1.0',
                    'threshold': 'visual only: input >= 1, default 3 (bright above, pale below, dashed line, optional marks)',
                    'ceiling': {'pool': {'more': curve['ceiling_pool']['orange'], 'less': curve['ceiling_pool']['blue']},
                                'by_symbol': {a: ceil['ceilings'][a] for a in POOL_ALL},
                                'unknown_symbol': ceil['unknown_symbol']},
                    'ceiling_rule': 'docs/rules.md R12 (R9 on the pool with the Platt of each symbol from the other half; '
                                    'R11 per symbol: the lower edge of the first bin where factual / expected < 0.8 or < 1000 '
                                    'windows of a class; unknown symbol: the lowest of FIL, LINK, XLM with the global Platt); '
                                    'above the ceiling the line stays at it, "N+"',
                    'unknown_symbol': 'not Binance spot from the table (prefix BINANCE, no .P): floor, Platt and ceilings of '
                                      'the unknown symbol + note',
                    'timeframe': 'the same computation on any timeframe (bars only); 5M only checked: other timeframes '
                                 'drawn with a note'},
        'warmup': {'first_line_bar_continuous': FIRST_LINE_BAR, 'history_bars_before_window': NT_HIST,
                   'rule': 'docs/rules.md R13: no line while the window is invalid or has < 200 bars before its first bar; '
                           'first valid r and steps at bar_index 201, first window 201 .. 264'},
    }


GOLDEN_COLS = ['time', 'open', 'high', 'low', 'close', 'norm', 'ok', 'r', 'step_hl2', 'step_hlc3', 'g', 'D', 'x_end',
               'sgn'] + [f'side_{n}' for n in SIDE_NAMES] + ['logit', 'lift', 'M', 'shown']


def golden(ex, bars, symbol='BTCUSDT'):
    p = from_dict(ex['model'])
    pl = ex['calibration']['per_symbol'][symbol]
    res = run(bars, p, ex['floors']['floor_L64'][symbol], eps=ex['tie_rule']['eps'], platt=(pl['platt_a'], pl['platt_b']))
    rows = []
    for i, x in enumerate(res):
        side = x.get('side_raw', [np.nan] * len(SIDE_NAMES))
        rows.append([x['time'], bars['open'].iat[i], bars['high'].iat[i], bars['low'].iat[i], bars['close'].iat[i],
                     x['norm'], int(x['ok']), x['r'], x['step_hl2'], x['step_hlc3'], x['g'], x.get('D', np.nan),
                     x.get('x_end', np.nan), x.get('sgn', np.nan)] + list(side)
                    + [x['logit'], x.get('lift', np.nan), x.get('M', np.nan), int(np.isfinite(x['logit']))])
    return pd.DataFrame(rows, columns=GOLDEN_COLS)


def load_export(path=None):
    with open(path or os.path.join(ROOT, EXPORT), encoding='utf-8') as fh:
        return json.load(fh)


def main():
    ex = build_export()
    path = os.path.join(ROOT, EXPORT)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(ex, fh, ensure_ascii=False, separators=(',', ':'))
    print('export', path, os.path.getsize(path) // 1024, 'KB')
    g = golden(ex, read_bars(os.path.join(ROOT, 'artifacts/bars/btc_fresh_tail.csv.gz')))
    gp = os.path.join(ROOT, GOLDEN)
    g.to_csv(gp, index=False, float_format='%.12g', compression={'method': 'gzip', 'mtime': 0})
    first = int(np.argmax(g['shown'].to_numpy() > 0))
    print('golden', gp, os.path.getsize(gp) // 1024, 'KB', '| bars', len(g), '| shown', int(g['shown'].sum()),
          '| first shown bar', first)
    if first != FIRST_LINE_BAR:
        raise SystemExit(f'first line bar {first} != {FIRST_LINE_BAR} (rule R13): look into it')


if __name__ == '__main__':
    main()
