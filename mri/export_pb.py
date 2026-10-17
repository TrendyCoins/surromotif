"""Neutral export of the PB_s64h24_d1 indicator model (JSON) and its golden file (bar-by-bar reference on BTC 5M).

    python -m mri.export_pb       # writes export/mri_model_pb.json and tests/golden/btc5m_fresh_pb.csv.gz

The JSON is self-contained: weights (separable layers 2..4), buffers, the 17 side inputs (order, formulas, clips, sign,
NaN), tie rule, floors (R6), the line M without per-asset calibration (R15: ruler glob_e of prod a, live part, limits,
warm-up R16, ceilings of the pool, of 31 symbols and of the unknown symbol), passport. Weights are exact decimals of their
float32 values (Pine decodes the same bits from dense strings); the ruler is float64 as in the 'nocal' json. The golden file
lets any platform check bar by bar from the first bar of the slice (warm-up included and marked).
The NT3_PRV_ch48 (export/mri_model_nt3.json) and F3_g (export/mri_model.json) exports stay as legacy models.
"""
import datetime as dt
import glob
import json
import os

import numpy as np
import pandas as pd

from . import GAP_MS, L, N_NORM, TIE_EPS, linem
from .export import FRESH, POOL_ALL, git_version
from .export_nt3 import SIDE_SPEC, _num, _nums
from .features import read_bars
from .nt3 import NT_BODY_FLOOR, NT_GEO, NT_HIST, NT_SHORT, SIDE_COLS, SIDE_NAMES, SIDE_SIGNED
from .params import from_dict, load_npz
from .pb import MODEL_ID, run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 'a'                                   # rule R14: one seed, chosen in advance
EXPORT = 'export/mri_model_pb.json'
GOLDEN = 'tests/golden/btc5m_fresh_pb.csv.gz'
FIRST_WINDOW_END = 264                       # rule R13 (continuous data)
FIRST_LINE_BAR = FIRST_WINDOW_END + linem.WARM   # rule R16: 552, confirmed by the reference
UNKNOWN_FLOOR_NOTE = 'median of floor_L64 of the 31 table symbols'


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding='utf-8') as fh:
        return json.load(fh)


def nocal_path():
    return sorted(glob.glob(os.path.join(ROOT, f'artifacts/calib/nocal_{MODEL_ID}_*.json')))[-1]


def build_export(seed=SEED):
    p, _ = load_npz(os.path.join(ROOT, f'artifacts/weights/prod_{MODEL_ID}_L64_{seed}_slim.npz'))
    prod = _load(f'artifacts/weights/prod_{MODEL_ID}_L64_{seed}.json')
    fl_dir = _load('artifacts/pack/floors_by_folder.json')
    with open(nocal_path(), encoding='utf-8') as fh:
        nc = json.load(fh)
    ceil = _load('docs/checks/pb/ceilings.json')
    if p.lvl_mu != 0.0:
        raise ValueError('lvl_mu must be 0 (streaming)')
    if p.pw is None:
        raise ValueError('PB needs separable layers')
    if not ceil.get('reproduced'):
        raise SystemExit('the nocal json is not reproduced (docs/checks/pb/ceilings.json): rule R15 — no export')
    if nc['pine']['ruler'] != 'glob_e' or nc['prod'][seed]['tag'] != prod['tag'] or nc['prod'][seed]['t_cut'] != prod['t_cut']:
        raise ValueError('nocal json: ruler / prod model mismatch')
    q = [float(x) for x in nc['prod'][seed]['glob_e']['q']]
    if len(q) != 37 or not np.all(np.diff(q) > 0):
        raise ValueError('ruler: 37 strictly increasing quantiles expected')
    # rule R6: the folder the prod model trained on — fresh for BTC / ETH, research for the rest (gate (в): same data end)
    floors = {a: _num(fl_dir['SMC/data_fresh/' if a in FRESH else 'SMC/data/'][a]['floor_L64']) for a in POOL_ALL}
    unknown_floor = _num(np.median(list(floors.values())))
    ch, hidden = p.conv[0].shape[0], p.head0_w.shape[0]
    n_side = len(SIDE_NAMES)
    model = {'id': MODEL_ID, 'conv': [_nums(w) for w in p.conv], 'pw': [_nums(w) for w in p.pw],
             'separable': [False, True, True, True], 'dilations': [1, 2, 4, 8], 'kernel': 3, 'channels': ch,
             'head0_w': _nums(p.head0_w), 'head0_b': _nums(p.head0_b),
             'head3_w': _nums(p.head3_w), 'head3_b': _nums(p.head3_b),
             's_sd': _num(p.s_sd), 'lvl_mu': 0.0, 'lvl_sd': _num(p.lvl_sd),
             'side_mu': _nums(p.side_mu), 'side_sd': _nums(p.side_sd), 'side_names': list(SIDE_NAMES),
             'side_cols_research': list(SIDE_COLS), 'side_signed': list(SIDE_SIGNED), 'head_hidden': hidden,
             'n_params': p.n_params, 'mult_per_bar': prod['pine'],
             'layout': {'conv[0]': f'({ch}, 4, 3) layer 1, ordinary', 'conv[1..3]': f'({ch}, 1, 3) depthwise (per channel)',
                        'pw[0..2]': f'({ch}, {ch}) pointwise 1 x 1 of layers 2..4 (research blocks.i.1.weight[:, :, 0])'}}
    by_symbol = {a: {'orange': float(ceil['by_symbol'][a]['orange']), 'blue': float(ceil['by_symbol'][a]['blue'])}
                 for a in POOL_ALL}
    return {
        'format': 'mri-export', 'format_version': 3,
        'indicator': 'Market Randomness Index (MRI)', 'classifier': 'SuMo classifier',
        'passport': {
            'model': f'{MODEL_ID} (streamable CNN: close, HL/2, HLC/3 steps + bar range, {ch} channels, layers 2-4 '
                     f'separable, head {3 * ch + n_side} -> {hidden} -> 1; side inputs: window divisor and the 200 bars '
                     'before the window, in bars; no clock)',
            'weights_source': f'prod_{MODEL_ID}_L64_{seed}', 'seed_key': seed, 'seed': prod['cfg']['seed'],
            'training': f"prod: all data of 31 assets (FIT + TUNE + VALIDATION), {prod['cfg']['epochs']} epochs, "
                        'cosine lr', 'research_code': 'research/sumo_class_train.py v6.3',
            'data_end': prod['t_cut'] + ' UTC', 'train_assets': prod['train_assets'],
            'line_check': {'nocal_when': nc['when'], 'nocal_ver': nc['ver'], 'ruler': nc['pine']['ruler'],
                           'source': f"VAL of 'refit' on {nc['src'][0]}, seeds {' + '.join(nc['seeds'])}"},
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
            'layers': 'valid convolutions without bias, kernel 3, dilations 1, 2, 4, 8; layer 1: y = relu(sum_j conv[0][:, :, j] '
                      '@ x(t - (2 - j) d)); layers 2..4: dw = sum_j conv[i][:, 0, j] * h(t - (2 - j) d) (per channel), '
                      'y = relu(pw[i - 1] @ dw); two streams (plus / minus) computed once per bar',
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
            'head': f'logit = head3_w . relu(head0_w @ [pooled({3 * ch}), side_std({n_side})] + head0_b) + head3_b'},
        'tie_rule': {'x_end': 'sum_{i=0..63} i * r_i / 64 over the window (end of the centered close path)',
                     'eps': TIE_EPS, 'rule': 'sgn = -1 if x_end < -eps * D else +1; one sign for the stream and the side inputs'},
        'floors': {'floor_L64': floors, 'unknown_symbol': unknown_floor, 'unknown_rule': UNKNOWN_FLOOR_NOTE,
                   'source': 'smc_series files of the prod training: fresh data folder for '
                             + ', '.join(FRESH) + ', research data folder for the rest'},
        'line': {
            'rule': 'docs/rules.md R15 (line M without per-asset calibration), R16 (warm-up); research v6.3 NOCAL v2',
            'ruler': {'kind': 'glob_e', 'knots': [float(x) for x in nc['knots']], 'zk': [float(x) for x in nc['zk']], 'q': q,
                      'formula': 'z = interp(logit, q, zk): a straight line between the knots, beyond the outer knots the '
                                 'edge value (np.interp: knot j with q[j] <= x < q[j+1], z = (zk[j+1] - zk[j]) / '
                                 '(q[j+1] - q[j]) * (x - q[j]) + zk[j])',
                      'source': f"37 quantiles of the logits of the sflip4 surrogates of VAL of all 31 assets of "
                                f"prod_{MODEL_ID}_L64_{seed} ({nc['prod'][seed]['n_null']} windows); one ruler for every "
                                'symbol, known or not'},
            'live': {'span_default': linem.SPAN, 's_lo': linem.S_LO, 'm_floor': 0.0, 'peak_eps': linem.PEAK_EPS,
                     'formula': 'on every valid window, the current one included: a = 2 / (span + 1); n1 = (1 - a) n1 + z; '
                                'n2 = (1 - a) n2 + z^2; d = (1 - a) d + 1 (zeros before the first valid window of the '
                                'chart); m_ema = n1 / d; s_ema = sqrt(max(n2 / d - m_ema^2, 0))',
                     'limits': 'm = max(m_ema, 0); s = min(max(s_ema, s_lo), 1); zc = min(z, m / max(1 - s^2, peak_eps)) '
                               'if s < 1 else z'},
            'log_m': 'log M = -(zc - m)^2 / (2 s^2) - ln s + zc^2 / 2',
            'warmup': {'windows_default': linem.WARM, 'first_line_bar_continuous': FIRST_LINE_BAR,
                       'rule': 'R16: windows are numbered from 0 over the valid windows of the chart; the line is shown '
                               'from n >= warm-up; an invalid window does not change the state and gets no number; with '
                               '"Compute only the last N bars" the live part and the numbers start at the first computed '
                               'window'},
            'settings': 'span and warm-up are inputs in bars (defaults 8064 / 288); the promise is checked at the '
                        'defaults only: other values -> corner note "not checked"',
            'display': {'colour': 'sign of log M: > 0 orange (more often than random), < 0 blue (less often), 0 — floor',
                        'number': 'M = exp(|log M|), floor 1.0',
                        'threshold': 'visual only: input >= 1, default 3 (bright above, pale below, dashed line, optional marks)',
                        'ceiling': 'not above the ceiling of its colour; above — drawn at the ceiling, label "N+"'},
            'ceilings': {'pool': {k: float(v) for k, v in ceil['pool'].items()}, 'by_symbol': by_symbol,
                         'unknown_symbol': {k: float(v) for k, v in ceil['unseen_symbol'].items()},
                         'rule': "R15: pool — R9 two-sided [0.8; 1.25]; symbol — R11 one-sided >= 0.8, not above the pool; "
                                 "unknown symbol — the lowest of FIL, LINK, XLM by 'refit' on C (unseen there); VAL 'refit' "
                                 'on ALL, seeds a + b, ruler of the other half-year, live part over VAL of the asset'}},
        'unknown_symbol': {'when': 'not Binance spot from the table (prefix BINANCE, no .P)',
                           'floor': unknown_floor, 'ruler': 'the same ruler', 'ceilings': 'line.ceilings.unknown_symbol',
                           'note': 'corner note'},
        'timeframe': 'the same computation on any timeframe (bars only); 5M only checked: other timeframes drawn with a note',
        'warmup': {'first_valid_window_end_continuous': FIRST_WINDOW_END, 'history_bars_before_window': NT_HIST,
                   'first_line_bar_continuous': FIRST_LINE_BAR,
                   'rule': 'R13: no window while a bar of it is invalid or its first bar has < 200 bars before it (first '
                           'valid r and steps at bar_index 201, first window 201 .. 264); R16: the line from the window '
                           'number 288 (bar 552)'},
    }


GOLDEN_COLS = ['time', 'open', 'high', 'low', 'close', 'norm', 'ok', 'r', 'step_hl2', 'step_hlc3', 'g', 'D', 'x_end',
               'sgn'] + [f'side_{n}' for n in SIDE_NAMES] + ['logit', 'z', 'm_ema', 's_ema', 'log_m', 'n', 'shown']


def params_line(ex):
    """(Params, q, zk) from the export."""
    ln = ex['line']['ruler']
    return from_dict(ex['model']), ln['q'], ln['zk']


def golden(ex, bars, symbol='BTCUSDT'):
    p, q, zk = params_line(ex)
    res = run(bars, p, ex['floors']['floor_L64'][symbol], q, zk, eps=ex['tie_rule']['eps'])
    rows = []
    for i, x in enumerate(res):
        side = x.get('side_raw', [np.nan] * len(SIDE_NAMES))
        rows.append([x['time'], bars['open'].iat[i], bars['high'].iat[i], bars['low'].iat[i], bars['close'].iat[i],
                     x['norm'], int(x['ok']), x['r'], x['step_hl2'], x['step_hlc3'], x['g'], x.get('D', np.nan),
                     x.get('x_end', np.nan), x.get('sgn', np.nan)] + list(side)
                    + [x['logit'], x['z'], x['m_ema'], x['s_ema'], x['log_m'],
                       x['n'] if np.isfinite(x['logit']) else -1, int(x['shown'])])
    return pd.DataFrame(rows, columns=GOLDEN_COLS)


def write_golden(g, path):
    """OHLC with 12 significant digits (exact prices), computed columns with 11 (relative 5e-12; keeps the file ~5 MB)."""
    out = g.copy()
    for c in GOLDEN_COLS[5:]:
        if out[c].dtype.kind == 'f':
            v = out[c].to_numpy()
            out[c] = np.where(np.isfinite(v), np.char.mod('%.11g', np.nan_to_num(v)), '')
    out.to_csv(path, index=False, float_format='%.12g', compression={'method': 'gzip', 'mtime': 0})


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
    write_golden(g, gp)
    first = int(np.argmax(g['shown'].to_numpy() > 0))
    first_w = int(np.argmax(np.isfinite(g['logit'].to_numpy())))
    print('golden', gp, os.path.getsize(gp) // 1024, 'KB', '| bars', len(g), '| shown', int(g['shown'].sum()),
          '| first window end', first_w, '| first line bar', first)
    if first_w != FIRST_WINDOW_END or first != FIRST_LINE_BAR:
        raise SystemExit(f'first window {first_w} / line bar {first} != {FIRST_WINDOW_END} / {FIRST_LINE_BAR} (R13 / R16)')


if __name__ == '__main__':
    main()
