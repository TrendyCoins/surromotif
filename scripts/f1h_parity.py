"""Parity of F1 on 1H (rules R22, R23, docs/rules.md): the reference f1h (export/f1h_model.json) against research and against
the research golden tfline21, on BTCUSDT 1H of artifacts/bars/ and artifacts/calib/.

    python scripts/f1h_parity.py [--out docs/checks/f1h/parity.json]      # needs torch (research side)

R22 (a): logit of the reference from the raw candles of smc_bars against research (training code v6.6, scripts/
f1h_research.py) on the same windows of TRAINING (26 grid gaps) and VALIDATION. (b): against logit_a of the golden (GPU).
R23: end to end on the 11 480 candles of the compact golden (scripts/f1h_export.py research_candles('va')) — windows, n_live,
show, first line bar; the line of the reference from the recorded logits of the golden; the line from its own logits.
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
from f1h_export import research_candles  # noqa: E402

GOLDEN = os.path.join(ROOT, 'artifacts', 'calib', 'tfline21_golden_20261022_063520.npz')
TOL_LOGIT = 1e-4                  # R22
TOL_LINE = 1e-9                   # R23: the line from the recorded logits
TOL_E2E = {'z': 1e-4, 'm': 1e-4, 's': 1e-4, 'log_m': 1e-3}      # R23, reference: consequences of TOL_LOGIT
L = 64


def stats(d):
    d = np.asarray(d)
    return {'n': int(len(d)), 'max': float(d.max()) if len(d) else 0.0,
            'p99': float(np.percentile(d, 99)) if len(d) else 0.0, 'mean': float(d.mean()) if len(d) else 0.0}


def reference(model, part):
    """Reference over the candles of a research part: compute() and the window info (D, x_end) at window ends."""
    c = research_candles(part)
    out = f1h.compute(model, c['time'], c['open'], c['high'], c['low'], c['close'])
    T = np.where(out['window'])[0]
    f = bar_inputs(c['time'], c['open'], c['high'], c['low'], c['close'], model.bar_ms, model.gap_factor, model.n_norm)
    ohlc = (c['open'], c['high'], c['low'], c['close'])
    D, xe = np.empty(len(T)), np.empty(len(T))
    for a in range(0, len(T), 4096):
        inp = window_inputs(model, f, ohlc, T[a:a + 4096])
        D[a:a + 4096], xe[a:a + 4096] = inp['D'], inp['x_end']
    return c, out, T, D, xe


def r22a(model, res, data):
    """Reference against research, part by part. Research window start st (row of the SeriesData array) -> row of the
       window end in the candles of the part: first + (st - offset of the part) + 63."""
    s = data.series[0]
    rows = {}
    for part, key, code in (('TRAINING', 'tr', 0), ('VALIDATION', 'va', 1)):
        t0 = time.time()
        c, out, T, D, xe = reference(model, key)
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


def r22b_r23(model, g):
    """Against the research golden tfline21 (VALIDATION of BTC)."""
    c, out, T, _, _ = reference(model, 'va')
    t_ns = c['time'][T] * 10 ** 6
    same_windows = bool(np.array_equal(t_ns, g['t_end']))
    res = {'candles': int(len(c['time'])), 'windows': int(len(T)), 'windows_golden': int(len(g['t_end'])),
           'same_windows': same_windows}
    if not same_windows:
        res['pass'] = False
        return res
    lg_g = g['logit_a'].astype(np.float64)
    d = np.abs(out['logit'][T] - lg_g)
    res['R22b'] = {'dlogit': stats(d), 'pass': bool(d.max() <= TOL_LOGIT),
                   'logit_eq_logit_a': bool(np.array_equal(g['logit'], lg_g))}
    # R23: exact
    shown = out['line'][T]
    first_shown = int(np.argmax(shown)) if shown.any() else None
    exact = {'n_live': bool(np.array_equal(out['n_live'][T], g['n_live'])), 'show': bool(np.array_equal(shown, g['show'])),
             'first_shown_window': first_shown, 'first_line_bar': int(T[first_shown]) if first_shown is not None else None,
             'shown': int(shown.sum())}
    exact['pass'] = (exact['n_live'] and exact['show'] and first_shown == model.show_from
                     and exact['first_line_bar'] == model.first_line_bar)
    res['R23_exact'] = exact
    # R23: the line of the reference from the recorded logits
    ln = line(model, lg_g)
    rec = {}
    for k in ('z', 'm', 's', 'log_m'):
        gk = g['logm' if k == 'log_m' else k]
        nan_same = bool(np.array_equal(np.isnan(ln[k]), np.isnan(gk)))
        fin = np.isfinite(gk)
        rec[k] = {'max': float(np.max(np.abs(ln[k][fin] - gk[fin]))), 'bit_exact': bool(np.array_equal(ln[k][fin], gk[fin])),
                  'nan_same': nan_same}
    rec['n_live'] = bool(np.array_equal(ln['n_live'], g['n_live']))
    rec['show'] = bool(np.array_equal(ln['line'], g['show']))
    rec['pass'] = all(v['max'] <= TOL_LINE and v['nan_same'] for k, v in rec.items() if isinstance(v, dict)) and \
        rec['n_live'] and rec['show']
    res['R23_from_golden_logit'] = rec
    # R23: end to end (own logits), reference
    e2e = {}
    for k in ('z', 'm', 's', 'log_m'):
        gk = g['logm' if k == 'log_m' else k]
        fin = np.isfinite(gk)
        dd = np.abs(out[k][T][fin] - gk[fin])
        e2e[k] = {**stats(dd), 'tol': TOL_E2E[k], 'pass': bool(dd.max() <= TOL_E2E[k])}
    sg_g = np.sign(g['logm'][g['show']])
    sg_m = out['side'][T][g['show']]
    flips = sg_g != sg_m
    near = np.abs(g['logm'][g['show']]) <= TOL_E2E['log_m']
    e2e['side_changes'] = int(flips.sum())
    e2e['side_changes_beyond_1e-3'] = int((flips & ~near).sum())
    cap = model.ceiling
    st_g = np.minimum(np.exp(np.abs(g['logm'][g['show']])), cap)
    e2e['strength'] = stats(np.abs(out['strength'][T][g['show']] - st_g))
    e2e['share_orange'] = float(np.mean(sg_m > 0))
    e2e['share_blue'] = float(np.mean(sg_m < 0))
    e2e['share_at_ceiling'] = float(np.mean(out['strength'][T][g['show']] >= cap))
    e2e['pass'] = all(e2e[k]['pass'] for k in TOL_E2E) and e2e['side_changes_beyond_1e-3'] == 0
    res['R23_end_to_end'] = e2e
    res['pass'] = res['R22b']['pass'] and exact['pass'] and rec['pass']
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs', 'checks', 'f1h', 'parity.json'))
    a = ap.parse_args()
    import f1h_research as R
    import torch
    model = f1h.load_model()
    g = dict(np.load(GOLDEN))
    summary = {'export': os.path.relpath(f1h.model.EXPORT, ROOT), 'golden': os.path.relpath(GOLDEN, ROOT),
               'model': model.meta, 'floor': model.floor, 'torch': torch.__version__, 'numpy': np.__version__,
               'threads': torch.get_num_threads()}
    t0 = time.time()
    T, CL, _ = R.modules(R.project(R.bars_files()))
    net, run = R.prod_model(T)
    data = R.series_data(T, 'BTCUSDT')
    res = R.logits(T, CL, net, data)
    summary['research'] = {'tag': run['tag'], 'windows': int(len(res['st'])), 'secs': round(time.time() - t0, 1),
                           'code': 'research/sumo_class_train.py v6.6 (load_pool_model, SeriesData, to_shapes, '
                                   'side_feats), floor_override of research/sumo_class_ceil.py'}
    va = res['part'] == 1
    dg = np.abs(res['logit'][va] - g['logit_a'].astype(np.float64))
    summary['research_cpu_vs_golden_gpu'] = {**stats(dg), 'same_starts': bool(np.array_equal(res['st'][va], g['st']))}
    print('research', json.dumps(summary['research']), '| CPU vs GPU', json.dumps(summary['research_cpu_vs_golden_gpu']))
    summary['R22a'] = r22a(model, res, data)
    summary['R22a']['pass'] = all(v['pass'] and v['window_set_ok'] for v in summary['R22a'].values())
    gold = r22b_r23(model, g)
    summary['golden_tfline21'] = gold
    print('golden', json.dumps(gold, indent=1))
    summary['pass'] = bool(summary['R22a']['pass'] and gold['pass'])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    print('pass', summary['pass'], '->', a.out)


if __name__ == '__main__':
    main()
