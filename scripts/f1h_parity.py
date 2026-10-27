"""Parity of F1 by timeframe (rules R22, R23 — 1H; R29, R30, R31 — 5M, 15M, 30M; docs/rules.md): the reference f1h (the export
of the timeframe) against research and against the golden of the source, on BTCUSDT of artifacts/bars/ and artifacts/calib/.

    python scripts/f1h_parity.py --tf 1H | 5M | 15M | 30M [--out PATH]     # needs torch (research side)
    # default out: docs/checks/f1h/parity.json (1H), parity_5m.json, parity_15m.json, parity_30m.json

(a) R22 / R29: logit of the reference from the raw candles of smc_bars against research (training code v6.6,
scripts/f1h_research.py) on the same windows of TRAINING and VALIDATION (5M: the compact files — the tail of TRAINING and
VALIDATION). (b): against logit_a of the golden of the source (GPU). R23 / R30: end to end on the candles of VALIDATION
(scripts/f1h_export.py research_candles('va')) — windows, n_live, show, first line bar; the line of the reference from the
recorded logits of the golden; the line from its own logits. R31: the compact golden of the repo against the golden of the
source on its windows (5M — the piece from the start of VALIDATION).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f1h  # noqa: E402
from f1h.line import line  # noqa: E402
from f1h.net import window_inputs  # noqa: E402
from f1h.bars import bar_inputs  # noqa: E402
from f1h_export import TFS, research_candles  # noqa: E402

TOL_LOGIT = 1e-4                  # R22 / R29
TOL_LINE = 1e-9                   # R23 / R30: the line from the recorded logits
TOL_E2E = {'z': 1e-4, 'm': 1e-4, 's': 1e-4, 'log_m': 1e-3}      # R23 / R30, reference: consequences of TOL_LOGIT
L = 64


def rules(tf):
    """Names of the keys of the result: 1H — as docs/checks/f1h/parity.json (R22, R23), other timeframes — R29, R30."""
    if tf == '1H':
        return {'a': 'R22a', 'b': 'R22b', 'exact': 'R23_exact', 'rec': 'R23_from_golden_logit', 'e2e': 'R23_end_to_end',
                'golden': 'golden_tfline21'}
    return {'a': 'R29a', 'b': 'R29b', 'exact': 'R30_exact', 'rec': 'R30_from_golden_logit', 'e2e': 'R30_end_to_end',
            'golden': 'golden_source'}


def stats(d):
    d = np.asarray(d)
    return {'n': int(len(d)), 'max': float(d.max()) if len(d) else 0.0,
            'p99': float(np.percentile(d, 99)) if len(d) else 0.0, 'mean': float(d.mean()) if len(d) else 0.0}


def reference(model, part, tf):
    """Reference over the candles of a research part: compute() and the window info (D, x_end) at window ends."""
    c = research_candles(part, tf)
    out = f1h.compute(model, c['time'], c['open'], c['high'], c['low'], c['close'])
    T = np.where(out['window'])[0]
    f = bar_inputs(c['time'], c['open'], c['high'], c['low'], c['close'], model.bar_ms, model.gap_factor, model.n_norm)
    ohlc = (c['open'], c['high'], c['low'], c['close'])
    D, xe = np.empty(len(T)), np.empty(len(T))
    for a in range(0, len(T), 4096):
        inp = window_inputs(model, f, ohlc, T[a:a + 4096])
        D[a:a + 4096], xe[a:a + 4096] = inp['D'], inp['x_end']
    return c, out, T, D, xe


def part_a(model, res, data, tf):
    """Reference against research, part by part. Research window start st (row of the SeriesData array) -> row of the
       window end in the candles of the part: first + (st - offset of the part) + 63."""
    s = data.series[0]
    rows = {}
    for part, key, code in (('TRAINING', 'tr', 0), ('VALIDATION', 'va', 1)):
        t0 = time.time()
        c, out, T, D, xe = reference(model, key, tf)
        sel = res['part'] == code
        end = c['first'] + (res['st'][sel] - s['off'][key]) + (L - 1)
        mine = set(T.tolist())
        theirs = set(end.tolist())
        only_research = sorted(theirs - mine)
        only_ref = sorted(mine - theirs)
        expect_missing = [c['first'] + L - 1] if key == 'tr' else []     # R22 (a): the window on the first TRAINING bar
        common = np.array(sorted(mine & theirs), dtype=np.int64)
        pos_r = {e: i for i, e in enumerate(end.tolist())}
        ir = np.array([pos_r[e] for e in common.tolist()], dtype=np.int64)
        lr = res['logit'][sel][ir]
        lm = out['logit'][common]
        d = np.abs(lm - lr)
        iT = np.searchsorted(T, common)
        tie = np.abs(xe[iT] / D[iT]) < 1e-3
        flip_mismatch = res['flip'][sel][ir] != (xe[iT] < -model.orient_eps * D[iT])
        fl32 = np.float32(model.floor)
        acts = (res['med'][sel][ir] < fl32) | (res['mabs'][sel][ir] < fl32)
        rows[part] = {
            'candles': int(len(c['time'])), 'windows_research': int(sel.sum()), 'windows_reference': int(len(T)),
            'only_research': only_research, 'only_research_expected': expect_missing,
            'only_reference': only_ref, 'window_set_ok': only_research == expect_missing and not only_ref,
            'dlogit': stats(d), 'pass': bool(d.max() <= TOL_LOGIT),
            'ties': int(tie.sum()), 'dlogit_ties': stats(d[tie]),
            'orientation_mismatch': int(flip_mismatch.sum()), 'dlogit_orientation_mismatch': stats(d[flip_mismatch]),
            'floor_acts': int(acts.sum()), 'dlogit_floor_acts': stats(d[acts]),
            'research_side_vs_shape_flip': int((res['flip'][sel] != res['flip_side'][sel]).sum()),
            'secs_reference': round(time.time() - t0, 1)}
        print(part, json.dumps({k: v for k, v in rows[part].items() if k not in ('only_research', 'only_reference')}))
    return rows


def part_b_end_to_end(model, g, tf):
    """Against the golden of the source (VALIDATION of BTC)."""
    k = rules(tf)
    c, out, T, _, _ = reference(model, 'va', tf)
    t_ns = c['time'][T] * 10 ** 6
    same_windows = bool(np.array_equal(t_ns, g['t_end']))
    res = {'candles': int(len(c['time'])), 'windows': int(len(T)), 'windows_golden': int(len(g['t_end'])),
           'same_windows': same_windows}
    if not same_windows:
        res['pass'] = False
        return res, None
    lg_g = g['logit_a'].astype(np.float64)
    d = np.abs(out['logit'][T] - lg_g)
    res[k['b']] = {'dlogit': stats(d), 'pass': bool(d.max() <= TOL_LOGIT),
                   'logit_eq_logit_a': bool(np.array_equal(g['logit'], lg_g))}
    # R23 / R30: exact
    shown = out['line'][T]
    first_shown = int(np.argmax(shown)) if shown.any() else None
    exact = {'n_live': bool(np.array_equal(out['n_live'][T], g['n_live'])), 'show': bool(np.array_equal(shown, g['show'])),
             'first_shown_window': first_shown, 'first_line_bar': int(T[first_shown]) if first_shown is not None else None,
             'shown': int(shown.sum())}
    exact['pass'] = (exact['n_live'] and exact['show'] and first_shown == model.show_from
                     and exact['first_line_bar'] == model.first_line_bar)
    res[k['exact']] = exact
    # R23 / R30: the line of the reference from the recorded logits
    ln = line(model, lg_g)
    rec = {}
    for key in ('z', 'm', 's', 'log_m'):
        gk = g['logm' if key == 'log_m' else key]
        nan_same = bool(np.array_equal(np.isnan(ln[key]), np.isnan(gk)))
        fin = np.isfinite(gk)
        rec[key] = {'max': float(np.max(np.abs(ln[key][fin] - gk[fin]))),
                    'bit_exact': bool(np.array_equal(ln[key][fin], gk[fin])), 'nan_same': nan_same}
    rec['n_live'] = bool(np.array_equal(ln['n_live'], g['n_live']))
    rec['show'] = bool(np.array_equal(ln['line'], g['show']))
    rec['pass'] = all(v['max'] <= TOL_LINE and v['nan_same'] for v in rec.values() if isinstance(v, dict)) and \
        rec['n_live'] and rec['show']
    res[k['rec']] = rec
    # R23 / R30: end to end (own logits), reference
    res[k['e2e']] = end_to_end(model, {key: out[key][T] for key in ('z', 'm', 's', 'log_m', 'side', 'strength')}, g)
    res['pass'] = res[k['b']]['pass'] and exact['pass'] and rec['pass']
    return res, {'c': c, 'out': out, 'T': T}


def end_to_end(model, mine, g):
    """Own values on the windows of g (dict of per-window arrays z, m, s, log_m, side, strength) against g."""
    e2e = {}
    for key in ('z', 'm', 's', 'log_m'):
        gk = g['logm' if key == 'log_m' else key]
        fin = np.isfinite(gk)
        dd = np.abs(mine[key][fin] - gk[fin])
        e2e[key] = {**stats(dd), 'tol': TOL_E2E[key], 'pass': bool(dd.max() <= TOL_E2E[key])}
    sg_g = np.sign(g['logm'][g['show']])
    sg_m = mine['side'][g['show']]
    flips = sg_g != sg_m
    near = np.abs(g['logm'][g['show']]) <= TOL_E2E['log_m']
    e2e['side_changes'] = int(flips.sum())
    e2e['side_changes_beyond_1e-3'] = int((flips & ~near).sum())
    cap = model.ceiling
    st_g = np.minimum(np.exp(np.abs(g['logm'][g['show']])), cap)
    e2e['strength'] = stats(np.abs(mine['strength'][g['show']] - st_g))
    e2e['share_orange'] = float(np.mean(sg_m > 0))
    e2e['share_blue'] = float(np.mean(sg_m < 0))
    e2e['share_at_ceiling'] = float(np.mean(mine['strength'][g['show']] >= cap))
    e2e['share_at_8'] = float(np.mean(mine['strength'][g['show']] >= 8.0))
    e2e['pass'] = all(e2e[key]['pass'] for key in TOL_E2E) and e2e['side_changes_beyond_1e-3'] == 0
    return e2e


def golden_file(model, g, full, tf):
    """R24 / R31: the compact golden of the repo — its candles are the start of research_candles('va') (all of them on
       1H / 15M / 30M), its values = the reference over all of VALIDATION on those candles (the line is causal), its
       windows are the first windows of the golden of the source; on them — R30 against the source."""
    path = os.path.join(ROOT, TFS[tf]['golden'])
    gf = f1h.read_golden(path)
    n = len(gf['time'])
    c, out = full['c'], full['out']
    res = {'file': TFS[tf]['golden'], 'bytes': os.path.getsize(path), 'candles': n,
           'candles_are_the_start': bool(all(np.array_equal(gf[key], c[key][:n])
                                             for key in ('time', 'open', 'high', 'low', 'close')))}
    same = f1h.golden.compare({key: v[:n] for key, v in out.items()}, gf, tol=0.0)      # bit for bit
    res['values_eq_full_validation'] = {key: v for key, v in same.items() if key != 'ok'}
    res['values_eq_full_validation']['bit_exact'] = same['ok']
    ok_file = f1h.check_golden(model, path)
    res['check_golden'] = ok_file
    w = np.where(gf['window'])[0]
    m = len(w)
    g_m = {key: v[:m] for key, v in g.items() if v.ndim == 1}
    res['windows'] = m
    res['windows_are_the_first_of_source'] = bool(np.array_equal(gf['time'][w] * 10 ** 6, g['t_end'][:m]))
    d = np.abs(gf['logit'][w] - g_m['logit_a'].astype(np.float64))
    res['dlogit_vs_source'] = {**stats(d), 'pass': bool(d.max() <= TOL_LOGIT)}
    res['n_live_same'] = bool(np.array_equal(gf['n_live'][w], g_m['n_live']))
    res['show_same'] = bool(np.array_equal(gf['line'][w], g_m['show']))
    res['first_line_bar'] = int(np.argmax(gf['line']))
    res['bars_with_line'] = int(gf['line'].sum())
    res['end_to_end'] = end_to_end(model, {key: gf[key][w] for key in ('z', 'm', 's', 'log_m', 'side', 'strength')}, g_m)
    res['pass'] = (res['candles_are_the_start'] and same['ok'] and ok_file['ok'] and res['windows_are_the_first_of_source']
                   and res['dlogit_vs_source']['pass'] and res['n_live_same'] and res['show_same']
                   and res['first_line_bar'] == model.first_line_bar and res['end_to_end']['pass'] and res['bytes'] <= 5.5e6)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tf', required=True, choices=sorted(TFS))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    tf = a.tf
    out_path = a.out or os.path.join(ROOT, 'docs', 'checks', 'f1h',
                                     'parity.json' if tf == '1H' else f'parity_{tf.lower()}.json')
    import f1h_research as R
    import torch
    k = rules(tf)
    model = f1h.load_model(tf=tf)
    gpath = os.path.join(ROOT, TFS[tf]['golden_src'])
    g = dict(np.load(gpath))
    summary = {'tf': tf, 'export': os.path.relpath(f1h.model.EXPORTS[tf], ROOT), 'golden': os.path.relpath(gpath, ROOT),
               'model': model.meta, 'live': {'family': model.live_family, 'L': model.live_l, 'n0': model.live_n0,
                                            'k': model.live_k, 'tau': model.live_tau},
               'ceiling': model.ceiling, 'floor': model.floor, 'torch': torch.__version__, 'numpy': np.__version__,
               'threads': torch.get_num_threads()}
    t0 = time.time()
    folder = os.path.dirname(os.path.join(ROOT, TFS[tf]['bars']))
    T, CL, _ = R.modules(R.project(R.bars_files('BTCUSDT', tf, folder)))
    net, run = R.prod_model(T)
    data = R.series_data(T, 'BTCUSDT', tf=tf)
    res = R.logits(T, CL, net, data)
    summary['research'] = {'tag': run['tag'], 'files': os.path.relpath(folder, ROOT), 'windows': int(len(res['st'])),
                           'secs': round(time.time() - t0, 1),
                           'code': 'research/sumo_class_train.py v6.6 (load_pool_model, SeriesData, to_shapes, '
                                   'side_feats), floor_override of research/sumo_class_ceil.py'}
    va = res['part'] == 1
    dg = np.abs(res['logit'][va] - g['logit_a'].astype(np.float64))
    summary['research_cpu_vs_golden_gpu'] = {**stats(dg), 'same_starts': bool(np.array_equal(res['st'][va], g['st']))}
    print('research', json.dumps(summary['research']), '| CPU vs GPU', json.dumps(summary['research_cpu_vs_golden_gpu']))
    summary[k['a']] = part_a(model, res, data, tf)
    summary[k['a']]['pass'] = all(v['pass'] and v['window_set_ok'] for v in summary[k['a']].values())
    gold, full = part_b_end_to_end(model, g, tf)
    summary[k['golden']] = gold
    print('golden', json.dumps(gold, indent=1))
    ok = summary[k['a']]['pass'] and gold['pass']
    if tf != '1H' and full is not None:
        summary['R31_golden_file'] = golden_file(model, g, full, tf)
        print('golden file', json.dumps(summary['R31_golden_file'], indent=1))
        ok = ok and summary['R31_golden_file']['pass']
    summary['pass'] = bool(ok)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    print('pass', summary['pass'], '->', out_path)


if __name__ == '__main__':
    main()
