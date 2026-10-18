"""Neutral export of the PB_s64h24_d1 indicator model (JSON) and its golden file (bar-by-bar reference on BTC 5M).

    python -m mri.export_pb       # writes export/mri_model_pb.json and tests/golden/btc5m_fresh_pb.csv.gz

Session 5 (format_version 4; rules R17-R19): one floor of the window divisor for every symbol (FLOOR = float32(0.2)), the
ruler of prod a scored with that floor (SuMo_Class_Ceil v1), the live part — formula A of SuMo_Class_Live (window 4032 +
144 virtual random windows, lag 64; the line from the window 4096), one ceiling 10 for both colours. No constants of the
symbol, no settings of the live part. The export refuses to run unless the Ceil json is reproduced in the repo
(docs/checks/pb5/ceil_repro.json, scripts/pb5_ceil.py).

The JSON is self-contained: weights (separable layers 2..4), buffers, the 17 side inputs (order, formulas, clips, sign,
NaN), tie rule, floor, the line M (ruler, live part, limits, start of the line, ceiling), passport. Weights are exact
decimals of their float32 values (Pine decodes the same bits from dense strings); the ruler is float64 as in the Ceil json.
The golden file lets any platform check bar by bar from the first bar of the slice (the start of the line marked).
The NT3_PRV_ch48 (export/mri_model_nt3.json) and F3_g (export/mri_model.json) exports stay as legacy models; the session-4
PB export (EMA line, symbol tables) is in the git history.
"""
import datetime as dt
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from . import GAP_MS, L, N_NORM, TIE_EPS, linem
from .export import git_version
from .export_nt3 import SIDE_SPEC, _num, _nums
from .features import read_bars
from .nt3 import NT_BODY_FLOOR, NT_GEO, NT_HIST, NT_SHORT, SIDE_COLS, SIDE_NAMES, SIDE_SIGNED
from .params import from_dict, load_npz
from .pb import FLOOR, MODEL_ID, run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 'a'                                   # rule R14: one seed, chosen in advance
EXPORT = 'export/mri_model_pb.json'
GOLDEN = 'tests/golden/btc5m_fresh_pb.csv.gz'
CEIL_JSON = 'artifacts/calib/ceil_20261018_115426.json'
LIVE_JSON = 'artifacts/calib/live_20261018_093025.json'
REPRO = 'docs/checks/pb5/ceil_repro.json'
FIRST_WINDOW_END = 264                       # rule R13 (continuous data)
FIRST_LINE_BAR = FIRST_WINDOW_END + linem.SHOW_FROM   # rule R18: 4360, confirmed by the reference


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding='utf-8') as fh:
        return json.load(fh)


def nocal_path():
    return sorted(glob.glob(os.path.join(ROOT, f'artifacts/calib/nocal_{MODEL_ID}_*.json')))[-1]


def research_version(name, var):
    """Version constant of a research file (BELL_VER / LIVE_VER / CEIL_VER), read from its text."""
    with open(os.path.join(ROOT, 'research', name), encoding='utf-8') as fh:
        m = re.search(rf"^{var} = (.+)$", fh.read(), re.M)
    return m.group(1).strip().strip("'") if m else None


def build_export(seed=SEED):
    p, _ = load_npz(os.path.join(ROOT, f'artifacts/weights/prod_{MODEL_ID}_L64_{seed}_slim.npz'))
    prod = _load(f'artifacts/weights/prod_{MODEL_ID}_L64_{seed}.json')
    with open(nocal_path(), encoding='utf-8') as fh:
        nc = json.load(fh)
    cj, lj = _load(CEIL_JSON), _load(LIVE_JSON)
    if not os.path.isfile(os.path.join(ROOT, REPRO)):
        raise SystemExit(f'no {REPRO}: the Ceil json is not reproduced (scripts/pb5_ceil.py) — no export')
    rep = _load(REPRO)
    if not rep.get('reproduced') or not rep.get('ruler', {}).get('bit_exact'):
        raise SystemExit(f'the Ceil json is not reproduced ({REPRO}: reproduced {rep.get("reproduced")}, ruler '
                         f'{rep.get("ruler", {}).get("bit_exact")}) — no export')
    if p.lvl_mu != 0.0:
        raise ValueError('lvl_mu must be 0 (streaming)')
    if p.pw is None:
        raise ValueError('PB needs separable layers')
    ce = cj['export']
    live_want = {'L': linem.L_WIN, 'n0': linem.N0, 'lag': linem.LAG, 'show_from_window': linem.SHOW_FROM}
    if (np.float32(ce['floor']) != np.float32(FLOOR) or ce['ceiling'] != linem.CEIL or ce['live'] != live_want
            or cj['choice'] != ce['ceiling']):
        raise ValueError(f'Ceil json export: floor {ce["floor"]}, ceiling {ce["ceiling"]}, live {ce["live"]} — not as R17-R19')
    ru = ce['ruler']
    if ru['seed'] != seed or ru['tag'] != prod['tag'] or ru['t_cut'] != prod['t_cut'] or nc['prod'][seed]['tag'] != prod['tag']:
        raise ValueError('Ceil json ruler: not the prod model of the export')
    q = [float(x) for x in ru['glob_e']['q']]
    if len(q) != 37 or not np.all(np.diff(q) > 0):
        raise ValueError('ruler: 37 strictly increasing quantiles expected')
    if lj['verdict']['a_pick'] != '4032+144' or lj['verdict']['b_pick'] != '4032+144·B3':
        raise ValueError('Live json: the formula of the export is 4032+144 shown from the full window (B3)')
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
    vers = {f: research_version(f, v) for f, v in (('sumo_class_bell.py', 'BELL_VER'), ('sumo_class_live.py', 'LIVE_VER'),
                                                   ('sumo_class_ceil.py', 'CEIL_VER'))}
    stamp = lambda name: re.search(r'(\d{8}_\d{6})', name).group(1)
    return {
        'format': 'mri-export', 'format_version': 4,
        'indicator': 'Market Randomness Index (MRI)', 'classifier': 'SuMo classifier',
        'passport': {
            'model': f'{MODEL_ID} (streamable CNN: close, HL/2, HLC/3 steps + bar range, {ch} channels, layers 2-4 '
                     f'separable, head {3 * ch + n_side} -> {hidden} -> 1; side inputs: window divisor and the 200 bars '
                     'before the window, in bars; no clock)',
            'weights_source': f'prod_{MODEL_ID}_L64_{seed}', 'seed_key': seed, 'seed': prod['cfg']['seed'],
            'training': f"prod: all data of 31 assets (FIT + TUNE + VALIDATION), {prod['cfg']['epochs']} epochs, "
                        'cosine lr',
            'research_code': (f"research/sumo_class_train.py v6.3; research/sumo_class_bell.py SuMo_Class_Bell "
                              f"v{vers['sumo_class_bell.py']}, research/sumo_class_live.py SuMo_Class_Live "
                              f"v{vers['sumo_class_live.py']}, research/sumo_class_ceil.py SuMo_Class_Ceil "
                              f"v{vers['sumo_class_ceil.py']}"),
            'data_end': prod['t_cut'] + ' UTC', 'train_assets': prod['train_assets'],
            'line_check': {'live_json': stamp(LIVE_JSON), 'live_run_ver': lj['ver'],
                           'live_note': f"the 'run' was made by SuMo_Class_Live v{lj['ver']}; research v1.1 has the same "
                                        "'run' code (adds the mode 'opt_b' only)",
                           'live_pick': {'a': lj['verdict']['a_pick'], 'b': lj['verdict']['b_pick']},
                           'ceil_json': stamp(CEIL_JSON), 'ceil_ver': cj['ver'], 'ceil_choice': cj['choice'],
                           'nocal_json_knots': nc['when'], 'reproduced_in_repo': REPRO,
                           'source': "main source 'stop' on C (TUNE + VAL), confirmation 'refit' on ALL (VAL), seeds a + b"},
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
            'divisor_D': 'max(median |r| over the window (even: mean of two middle), floor.value)',
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
        'floor': {'value': FLOOR, 'float32_of': 0.2,
                  'rule': 'R17: one floor of the window divisor for every symbol, in both places of the model: D = max(median '
                          '|r| of the window, floor) (pooling / D, side input log D, tie threshold eps * D) and the side input '
                          'log max(mean |r| of the window, floor)',
                  'check': f"SuMo_Class_Ceil v{cj['ver']} (П): accepted (p_ok {cj['p_ok']}); the windows where it acts: "
                           f"{100 * cj['impact']['pool']['affected']:.2f}% of the main source"},
        'line': {
            'rule': 'docs/rules.md R17-R19 (session 5): ruler of prod a scored with the floor, live part — formula A of '
                    'SuMo_Class_Live (4032+144, shown from the full window B3), one ceiling 10 for both colours',
            'ruler': {'kind': 'glob_e', 'knots': [float(x) for x in nc['knots']], 'zk': [float(x) for x in nc['zk']], 'q': q,
                      'formula': 'z = interp(logit, q, zk): a straight line between the knots, beyond the outer knots the '
                                 'edge value (np.interp: knot j with q[j] <= x < q[j+1], z = (zk[j+1] - zk[j]) / '
                                 '(q[j+1] - q[j]) * (x - q[j]) + zk[j])',
                      'source': f"37 quantiles of the logits of the sflip4 surrogates of VAL of all 31 assets of "
                                f"prod_{MODEL_ID}_L64_{seed} scored with the floor (SuMo_Class_Ceil, {ru['n_null']} windows); "
                                'one ruler for every symbol'},
            'live': {'L': linem.L_WIN, 'n0': linem.N0, 'lag': linem.LAG, 's_lo': linem.S_LO, 'm_floor': 0.0,
                     'peak_eps': linem.PEAK_EPS,
                     'formula': 'windows numbered from 0 over the valid windows of the chart (n); the set of the window n — '
                                'the real windows max(0, n - 4095) .. n - 64 (the last 4032 windows ending 64 windows before '
                                'the current one), k = min(max(n - 63, 0), 4032) of them, S1 / S2 — sums of z / z^2 over '
                                'the set; plus 144 virtual random windows (z = 0, spread 1): mh = S1 / max(k, 1); vh = '
                                'max(S2 / max(k, 1) - mh^2, 0); m_win = S1 / (k + 144); s_win = sqrt((k vh + 144) / (k + 144))',
                     'stream': 'a ring of 4096 z (index n mod 4096) and running S1, S2, on a valid window n: if n >= 4096: '
                               'S1 -= z[n - 4096], S2 -= z[n - 4096]^2; z[n mod 4096] = z_n; if n >= 64: S1 += z[n - 64], '
                               'S2 += z[n - 64]^2; then k, m_win, s_win as above',
                     'limits': 'm = max(m_win, 0); s = min(max(s_win, s_lo), 1); zc = min(z, m / max(1 - s^2, peak_eps)) '
                               'if s < 1 else z'},
            'log_m': 'log M = -(zc - m)^2 / (2 s^2) - ln s + zc^2 / 2',
            'show': {'from_window': linem.SHOW_FROM, 'first_line_bar_continuous': FIRST_LINE_BAR,
                     'rule': 'R18: the line from the window number 4096 (= 4032 + 64, the full window); before it — empty, '
                             'with a corner note; an invalid window does not change the state and gets no number (the line '
                             'breaks there); with "Compute only the last N bars" the numbers start at the first computed '
                             'window. L, n0, lag and the start are constants, not settings'},
            'display': {'colour': 'sign of log M: > 0 orange (more often than random), < 0 blue (less often), 0 — floor',
                        'number': 'M = exp(|log M|), floor 1.0',
                        'threshold': 'visual only: input >= 1, default 3 (bright above, pale below, dashed line, optional marks)',
                        'ceiling': 'not above ceiling.value for both colours; above — drawn at the ceiling, no label '
                                   '(decision of 2026-10-18)',
                        'start_mark': 'one mark with a constant text at the first shown window (window 4096)'},
            'ceiling': {'value': float(ce['ceiling']), 'both_colours': True, 'above': 'drawn at the ceiling, no label',
                        'rule': "R19 / SuMo_Class_Ceil (Ц): the first of 10, 8, 5, 3 that passed (Ц1)-(Ц3) on 'stop' C and "
                                "(Ц1)-(Ц2) on 'refit' ALL; above it the line promises nothing more"}},
        'symbols': 'every symbol the same: one floor, one ruler, one ceiling (no constants of the symbol)',
        'timeframe': 'the same computation on any timeframe (bars only); 5M only checked: other timeframes drawn with a note',
        'warmup': {'first_valid_window_end_continuous': FIRST_WINDOW_END, 'history_bars_before_window': NT_HIST,
                   'first_line_bar_continuous': FIRST_LINE_BAR,
                   'rule': 'R13: no window while a bar of it is invalid or its first bar has < 200 bars before it (first '
                           'valid r and steps at bar_index 201, first window 201 .. 264); R18: the line from the window '
                           'number 4096 (bar 4360)'},
    }


GOLDEN_COLS = ['time', 'open', 'high', 'low', 'close', 'norm', 'ok', 'r', 'step_hl2', 'step_hlc3', 'g', 'D', 'x_end',
               'sgn'] + [f'side_{n}' for n in SIDE_NAMES] + ['logit', 'z', 'm_win', 's_win', 'k', 'log_m', 'n', 'shown']


def params_line(ex):
    """(Params, q, zk) from the export."""
    ln = ex['line']['ruler']
    return from_dict(ex['model']), ln['q'], ln['zk']


def golden(ex, bars):
    p, q, zk = params_line(ex)
    res = run(bars, p, ex['floor']['value'], q, zk, eps=ex['tie_rule']['eps'])
    rows = []
    for i, x in enumerate(res):
        side = x.get('side_raw', [np.nan] * len(SIDE_NAMES))
        rows.append([x['time'], bars['open'].iat[i], bars['high'].iat[i], bars['low'].iat[i], bars['close'].iat[i],
                     x['norm'], int(x['ok']), x['r'], x['step_hl2'], x['step_hlc3'], x['g'], x.get('D', np.nan),
                     x.get('x_end', np.nan), x.get('sgn', np.nan)] + list(side)
                    + [x['logit'], x['z'], x['m_win'], x['s_win'], x['k'] if np.isfinite(x['logit']) else -1,
                       x['log_m'], x['n'] if np.isfinite(x['logit']) else -1, int(x['shown'])])
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
        raise SystemExit(f'first window {first_w} / line bar {first} != {FIRST_WINDOW_END} / {FIRST_LINE_BAR} (R13 / R18)')


if __name__ == '__main__':
    main()
