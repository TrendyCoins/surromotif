"""Reproduction of the SuMo_Class_Ceil json (artifacts/calib/ceil_20261018_115426.json) on 'refit' ALL a + b — before the
export, obligatory (docs/rules.md, session 5 thresholds; the analogue of R15 of session 4). The main source ('stop' C) is not
in the repo: its numbers are taken from the json as they are.

    'до'    — the pool npz of 'refit' ALL (old floors): ceiling['refit_ALL|до'] and formula_a['refit_ALL, до (npz)']
    'после' — the Colab export with the floor 0.2 (artifacts/calib/ceil_f02_*; the burn-in windows B from the npz, as Ceil
              raw_from): ceiling['refit_ALL|после'] and formula_a['refit_ALL, после']

Two ways:
    (1) research functions as Ceil mode_run: Ceil variant_a, cells_cross (skip FIL, LINK, XLM), rule_source, eval_a; Bell
        check_nocal (the asset order), read_source_raw, build_source; Ceil raw_from for 'после'. choose_c is not called (it
        needs the main source): the choice of C is the json's
    (2) the line of the repo (what is exported: mri.linem ruler, formula A, log M) on the same windows; log M window by
        window = way 1 (<= 1e-12); then the rule — research rule_source / bce_pairs / _nc_summary on these log M
Thresholds: windows per bin (n_real, n_surr) exactly; fact, exp, ratio |d| <= 1e-9; statuses of pairs, over / nodata,
g1 / g2 (g3) and the shares 'C+' (clamp) the same for C = 10, 8, 5, 3, 13, 21; formula A: BCE and cover3 <= 1e-12, ceil and
share the same; the random chart (gp_col) — by way 1, for reference, with the difference. A mismatch — stop and report.

    python scripts/pb5_ceil.py [--out docs/checks/pb5/ceil_repro.json] [--table docs/checks/pb5/ceil_table.md]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mri import linem, research5 as R5  # noqa: E402

KEY = 'refit_ALL'
CS = (10, 8, 5, 3, 13, 21)
TOL_RATIO, TOL_LOGM, TOL_FA = 1e-9, 1e-12, 1e-12
CEIL_JSON = os.path.join(R5.CALIB, R5.CEIL_JSON)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(x) else float(x)
    return x


# ---------------------------------------------------------------------------------------------------------------------
# way 2: the line of the repo
# ---------------------------------------------------------------------------------------------------------------------
def my_cells(T, CE, raw, knots, zk, skip):
    """As Ceil cells_cross, written with mri.linem: cross rulers (np.quantile of the surrogates of the OTHER half of all
       assets of the seed except skip), z by linem.ruler_z, formula A by linem.live_win over the whole series, windows from
       position 4096 of their half; log M by linem.log_m. [(seed, asset, log M real, log M surrogates)]."""
    cells = []
    for seed, D in raw.items():
        F = {a: CE.fold_ids(T, KEY, d) for a, d in D.items()}
        Q = {f: np.quantile(np.concatenate([d['sn'][F[a] != f] for a, d in D.items() if a not in skip]), knots)
             for f in (0, 1)}
        for a, d in D.items():
            n = len(d['sp'])
            if n <= linem.SHOW_FROM:
                continue
            lr_, ln_ = [], []
            for f in (0, 1):
                zr, zs = linem.ruler_z(d['sp'], Q[f], zk), linem.ruler_z(d['sn'], Q[f], zk)
                m, s, _ = linem.live_win(zr)
                i = np.arange(linem.SHOW_FROM, n)
                i = i[F[a][i] == f]
                lr_.append(linem.log_m(zr[i], m[i], s[i]))
                ln_.append(linem.log_m(zs[i], m[i], s[i]))
            cells.append((seed, a, np.concatenate(lr_), np.concatenate(ln_)))
    return cells


def my_formula_a(T, BB, raw, knots, zk):
    """Formula A on the full window as part A of Live (eval_set 'steady'), log M by the line of the repo: ruler — the
       surrogates of the burn-in (П1) of all assets of the seed (Bell build_source), live part over the whole series, windows
       from B. BCE — research bce_pairs, mean over weeks; ceilings / share — research _nc_summary; cover3 — shares of
       log M >= ln 3 / <= -ln 3. Returns (numbers, {(seed, asset): (log M real, log M surrogates)})."""
    recs = []
    for seed, D in raw.items():
        q = np.quantile(np.concatenate([d['sn'][:d['B']] for d in D.values()]), knots)
        for a, d in D.items():
            zr, zs = linem.ruler_z(d['sp'], q, zk), linem.ruler_z(d['sn'], q, zk)
            m, s, _ = linem.live_win(zr)
            i = np.arange(d['B'], len(zr))
            recs.append((seed, a, d['wk'][i], linem.log_m(zr[i], m[i], s[i]), linem.log_m(zs[i], m[i], s[i])))
    w0 = min(int(r[2].min()) for r in recs)
    NW = max(int(r[2].max()) for r in recs) - w0 + 1
    wsum, wcnt = np.zeros(NW), np.zeros(NW)
    cells, o3, b3, n = [], 0, 0, 0
    l3 = math.log(3.0)
    for seed, a, wk, lr, ln in recs:
        b = BB.bce_pairs(lr, ln)
        wsum += np.bincount(wk - w0, weights=b, minlength=NW)
        wcnt += np.bincount(wk - w0, minlength=NW)
        cells.append((seed, a, lr, ln))
        o3 += int((lr >= l3).sum())
        b3 += int((lr <= -l3).sum())
        n += len(lr)
    with np.errstate(invalid='ignore', divide='ignore'):
        bce = float(np.nanmean(wsum / wcnt))
    summ = T._nc_summary(cells)
    return ({'bce': bce, 'ceil': summ['ceil'], 'share': summ['share'], 'cover3': (o3 / n, b3 / n)},
            {(c[0], c[1]): (c[2], c[3]) for c in cells})


def research_eval_logm(T, BB, LV, raw, v):
    """log M of way 1 on the windows of part A (as eval_set 'steady'): Bell ruler of the burn-in, Live chart_ms from B."""
    knots, zk = BB.knots_zk(T)
    out = {}
    for seed, D in raw.items():
        q = BB.ruler_q(np.concatenate([d['sn'][:d['B']] for d in D.values()]), knots)
        for a, d in D.items():
            zr, zs = BB.to_z(d['sp'], q, zk), BB.to_z(d['sn'], q, zk)
            r = LV.chart_ms(T, zr, [(0, len(zr))], v, {}, p_from=d['B'])
            out[(seed, a)] = (T._nc_logm(zr[r['i']], r['m'], r['s']), T._nc_logm(zs[r['i']], r['m'], r['s']))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# comparison with the json
# ---------------------------------------------------------------------------------------------------------------------
class Cmp:
    def __init__(self):
        self.d_ratio = 0.0
        self.bad = []

    def num(self, a, b, where):
        a = None if a is None or (isinstance(a, float) and not math.isfinite(a)) else float(a)
        b = None if b is None else float(b)
        if (a is None) != (b is None):
            self.bad.append(f'{where}: {a} vs {b}')
        elif a is not None:
            self.d_ratio = max(self.d_ratio, abs(a - b))
            if abs(a - b) > TOL_RATIO:
                self.bad.append(f'{where}: |d| {abs(a - b):.2e}')

    def same(self, a, b, where):
        if jsonable(a) != b:
            self.bad.append(f'{where}: {jsonable(a)} vs {b}')

    def row(self, r, j, where):
        for k in ('color', 'lo', 'n_real', 'n_surr', 'sparse'):
            self.same(r[k], j[k], f'{where}.{k}')
        if 'cap' in j:
            self.same(r['cap'], j['cap'], f'{where}.cap')
        for k in ('fact', 'exp', 'ratio'):
            self.num(r.get(k), j.get(k), f'{where}.{k}')


def compare_rule(R, J):
    """rule_source result R against the json ceiling entry J (n_windows, curve, by_c of every C)."""
    c = Cmp()
    c.same(R['n_windows'], J['n_windows'], 'n_windows')
    if len(R['curve']) != len(J['curve']):
        c.bad.append('curve: length')
    for i, (r, j) in enumerate(zip(R['curve'], J['curve'])):
        c.row(r, j, f'curve[{i}]')
    for C in CS:
        r, j = R['by_c'][C], J['by_c'][str(C)]
        for k in ('n_pairs', 'over', 'nodata', 'g1', 'g2', 'g3', 'unseen', 'clamp'):
            c.same(r[k], j[k], f'C{C}.{k}')
        for col in ('orange', 'blue'):
            c.same(r['pool'][col]['bins_bad'], j['pool'][col]['bins_bad'], f'C{C}.pool.{col}.bins_bad')
            c.same(r['pool'][col]['ok'], j['pool'][col]['ok'], f'C{C}.pool.{col}.ok')
            c.row(r['pool'][col]['group'], j['pool'][col]['group'], f'C{C}.pool.{col}.group')
        c.same(r['pool']['ok'], j['pool']['ok'], f'C{C}.pool.ok')
        if sorted(r['pairs']) != sorted(j['pairs']):
            c.bad.append(f'C{C}: pairs differ')
            continue
        for k_, p in r['pairs'].items():
            q = j['pairs'][k_]
            for k in ('status', 'over', 'sparse'):
                c.same(p[k], q[k], f'C{C}.{k_}.{k}')
            c.row(p['group'], q['group'], f'C{C}.{k_}.group')
    return {'ok': not c.bad, 'max_d_fact_exp_ratio': c.d_ratio, 'mismatches': c.bad[:30], 'n_mismatches': len(c.bad)}


def compare_fa(r, j, gp=True):
    out = {'d_bce': abs(r['bce'] - j['bce']),
           'd_cover3': max(abs(a - b) for a, b in zip(r['cover3'], j['cover3'])),
           'ceil_same': jsonable(r['ceil']) == j['ceil'], 'share_same': jsonable(r['share']) == j['share']}
    out['ok'] = bool(out['d_bce'] <= TOL_FA and out['d_cover3'] <= TOL_FA and out['ceil_same'] and out['share_same'])
    if gp:
        out['d_gp_col'] = abs(r['gp_col'] - j['gp_col'])
        out['gp_col'] = [r['gp_col'], j['gp_col']]
    out['bce'] = [r['bce'], j['bce']]
    return out


def max_d_cells(c1, c2):
    if len(c1) != len(c2) or any(x[0] != y[0] or x[1] != y[1] or len(x[2]) != len(y[2]) for x, y in zip(c1, c2)):
        return math.inf
    return max(max(float(np.max(np.abs(x[2] - y[2]), initial=0.0)), float(np.max(np.abs(x[3] - y[3]), initial=0.0)))
               for x, y in zip(c1, c2))


def raw_after(T, CE, BB, raw_b, floors):
    """Scores with the floor 0.2 from the Colab export in the Bell form, by Ceil raw_from itself: the export has med / mabs
       only for BTC (raw_from returns them as its second value, not used here) — zeros elsewhere."""
    from mri.ceil_export import load_export
    ex = load_export()
    rec = {}
    for seed in raw_b:
        tag = T._pool_tag('ALL', seed, BB.BELL['model'], 'refit')
        res = {}
        for a, d in ex[tag].items():
            z0 = np.zeros(len(d['wk']), dtype=np.float32)
            res[(a, 'val')] = {'sp': d['sp'].astype(np.float64), 'sn': d['sn'].astype(np.float64),
                               'wk': d['wk'].astype(np.int64), 'med': d.get('med', z0), 'mabs': d.get('mabs', z0)}
        rec[tag] = {'res': res, 'res_old': {}}
    raw, _ = CE.raw_from(T, KEY, rec, raw_b, floors, 'new')
    return raw


def table_md(cj, rep):
    """Table (Ц) by C: pool, 'C+', over / no data — main source from the json, confirmation reproduced."""
    lines = ['# (Ц) потолок C — SuMo_Class_Ceil v1, пол 0.2 (json `ceil_20261018_115426`; основной источник — из json как есть, '
             "подтверждение 'refit' ALL — воспроизведено в репо, `docs/checks/pb5/ceil_repro.json`)", '',
             'Корзины ниже C — фактическая / обещанная в [0.8; 1.25] (пул); «C+» — фактическая кратность группы силы ≥ C '
             '(оранжевые / синие), проходит при ≥ 0.8·C; пары ассет × цвет (62): завышают / без данных; доля реальных окон '
             'на «C+» (оранжевые / синие).', '']
    for src, title in (('stop_C|после', "основной источник 'stop' C (TUNE + VAL), пол 0.2"),
                       ('refit_ALL|после', "подтверждение 'refit' ALL (VAL), пол 0.2"),
                       ('stop_C|до', "справочно: 'stop' C, старые полы"), ('refit_ALL|до', "справочно: 'refit' ALL, старые полы")):
        J = cj['ceiling'][src]
        lines += [f'## {title}: окон {J["n_windows"]}', '',
                  '| C | (Ц1) корзины ниже C не прошли (о / с) | «C+» факт о / с | (Ц1) | завышают / без данных | (Ц2) | (Ц3) | '
                  '«C+» доля реальных о / с |', '|---|---|---|---|---|---|---|---|']
        for C in CS:
            b = J['by_c'][str(C)]
            p = b['pool']
            bb = ' / '.join(','.join(f'{x:g}' for x in p[c]['bins_bad']) or '—' for c in ('orange', 'blue'))
            gf = ' / '.join((f"{p[c]['group']['fact']:.2f}" if p[c]['group'].get('fact') is not None else '—')
                            + (' (мало)' if p[c]['group']['sparse'] else '') for c in ('orange', 'blue'))
            g3 = '—' if b['g3'] is None else ('✓' if b['g3'] else '✗')
            ref = '' if C in (10, 8, 5, 3) else ' (справочно)'
            lines.append(f"| {C}{ref} | {bb} | {gf} | {'✓' if b['g1'] else '✗'} | {b['over']} / {b['nodata']} из "
                         f"{b['n_pairs']} | {'✓' if b['g2'] else '✗'} | {g3} | {100 * b['clamp'][0]:.2f}% / "
                         f"{100 * b['clamp'][1]:.2f}% |")
        bad = [(k, x) for k, x in J['by_c']['10']['pairs'].items() if x['status'] != 'ок']
        lines += ['', 'При C = 10 не «ок»: ' + ('; '.join(f"{k} — {x['status']} (где: "
                                                         f"{', '.join(str(v) for v in (x['over'] or x['sparse']))})"
                                                         for k, x in sorted(bad)) or 'нет'), '']
    lines += [f"Выбор (json): C = {cj['choice']} (со старыми полами — {cj['choice_old_floors']}). Воспроизведение 'refit' ALL: "
              + ', '.join(f"{k}: {'✅' if v['rule_way1']['ok'] and v['rule_way2']['ok'] else '❌'}"
                          for k, v in rep.items() if 'rule_way1' in v), '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='docs/checks/pb5/ceil_repro.json')
    ap.add_argument('--table', default='docs/checks/pb5/ceil_table.md')
    a = ap.parse_args()
    t0 = time.time()
    T, BB, LV, CE = R5.chain()
    with open(CEIL_JSON, encoding='utf-8') as fh:
        cj = json.load(fh)
    chk, assets = BB.check_nocal(T)
    if not chk['ok']:
        raise SystemExit("research check against the 'nocal' json failed — stop")
    H = tuple(T.POOL['holdout'])
    v = CE.variant_a()
    if H != R5.H or v['key'] != '4032+144' or CE.full_from(v) != linem.SHOW_FROM:
        raise SystemExit(f'holdout {H} / formula {v["key"]} / show from {CE.full_from(v)} — not as the rules')
    knots, zk = BB.knots_zk(T)
    raw_b, tags = BB.read_source_raw(T, KEY, assets)
    print(f"npz {KEY}: " + '; '.join(f'{s}: {len(x)} parts' for s, x in tags.items()))
    sources = {'до': (raw_b, 'refit_ALL|до', 'refit_ALL, до (npz)')}
    from mri.ceil_export import DIR, MANIFEST
    if os.path.isfile(os.path.join(DIR, MANIFEST)):
        sources['после'] = (raw_after(T, CE, BB, raw_b, cj['floors_trained']), 'refit_ALL|после', 'refit_ALL, после')
    else:
        print("no Colab export (artifacts/calib/ceil_f02_manifest.json): 'после' is not checked yet")
    rep = {}
    for cond, (raw, kc, kf) in sources.items():
        t1 = time.time()
        r = {}
        c1 = CE.cells_cross(T, raw, KEY, v, skip=H)
        c2 = my_cells(T, CE, raw, knots, zk, H)
        r['logm_cells_max_d'] = max_d_cells(c1, c2)
        R1 = CE.rule_source(c1, CS, unseen=())
        R2 = CE.rule_source(c2, CS, unseen=())
        r['rule_way1'], r['rule_way2'] = compare_rule(R1, cj['ceiling'][kc]), compare_rule(R2, cj['ceiling'][kc])
        del c1, c2
        fa1 = CE.eval_a(T, BB.build_source(T, KEY, raw, BB.BELL['sources'][KEY]['about']), v)
        fa2, lm2 = my_formula_a(T, BB, raw, knots, zk)
        lm1 = research_eval_logm(T, BB, LV, raw, v)
        r['logm_eval_max_d'] = max(max(float(np.max(np.abs(lm1[k][0] - lm2[k][0]))), float(np.max(np.abs(lm1[k][1] - lm2[k][1]))))
                                   for k in lm1)
        r['fa_way1'], r['fa_way2'] = compare_fa(fa1, cj['formula_a'][kf]), compare_fa(fa2, cj['formula_a'][kf], gp=False)
        r['ok'] = bool(r['rule_way1']['ok'] and r['rule_way2']['ok'] and r['fa_way1']['ok'] and r['fa_way2']['ok']
                       and r['logm_cells_max_d'] <= TOL_LOGM and r['logm_eval_max_d'] <= TOL_LOGM)
        r['minutes'] = round((time.time() - t1) / 60, 1)
        rep[cond] = r
        print(f"{cond}: rule way 1 {r['rule_way1']['ok']} (max |d| {r['rule_way1']['max_d_fact_exp_ratio']:.1e}), way 2 "
              f"{r['rule_way2']['ok']} ({r['rule_way2']['max_d_fact_exp_ratio']:.1e}); log M cells {r['logm_cells_max_d']:.1e}, "
              f"eval {r['logm_eval_max_d']:.1e}; formula A way 1 {r['fa_way1']}, way 2 {r['fa_way2']} | {r['minutes']} min")
    out = {'ceil_json': R5.CEIL_JSON, 'source': KEY, 'check_nocal': {k: chk[k] for k in ('ok', 'bit', 'same_rules')},
           'assets': assets, 'conds': rep, 'choice_json': cj['choice'],
           'reproduced': bool(set(rep) == {'до', 'после'} and all(r['ok'] for r in rep.values())),
           'minutes': round((time.time() - t0) / 60, 1)}
    with open(os.path.join(ROOT, a.out), 'w', encoding='utf-8') as fh:
        json.dump(jsonable(out), fh, ensure_ascii=False, indent=1)
    with open(os.path.join(ROOT, a.table), 'w', encoding='utf-8') as fh:
        fh.write(table_md(cj, rep))
    print('reproduced:', out['reproduced'], '->', a.out, a.table)


if __name__ == '__main__':
    main()
