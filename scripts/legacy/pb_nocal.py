"""Rule R15: reproduction of the 'nocal' json (artifacts/calib/nocal_PB_s64h24_d1_*.json) on the logits of the pool npz —
before the export, two ways:
    research — the research v6.3 functions _nc_rulers / _nc_cells / _nc_summary (identity of the data: the npz the Colab
               run saw); cond_e is not used, so its inputs (zf, zfn) are zeros
    repo     — mri.linem, mri.legacy.linem_s4 / linem_check (the implementation that goes into the export and Pine)
Both against the json: windows per bin (n_real, n_surr) exactly, ratio |d| <= 1e-9, ceilings and the share of pairs >= 5
the same; random chart — covers within 1e-12. Parts: the pool (variants.glob_e: 'refit' on ALL, seeds a + b, rulers of
the other half-year), the unseen symbol ('refit' on C, rulers without H, only H), 'in sample' (one ruler per seed from the
TUNE surrogates of 'refit' on ALL), the random chart (live part on the surrogates).
Then the ceilings table (31 assets + 3 unseen + the unseen symbol) — docs/checks/legacy/pb/ceilings.{json,md}.

    python scripts/legacy/pb_nocal.py [--out docs/checks/legacy/pb/nocal_repro.json]
"""
import argparse
import glob
import json
import math
import os
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mri import linem  # noqa: E402
from mri.legacy import linem_check as LC, linem_s4  # noqa: E402

MODEL = 'PB_s64h24_d1'
NOCAL = sorted(glob.glob(os.path.join(ROOT, f'artifacts/calib/nocal_{MODEL}_*.json')))[-1]
H = ('FILUSDT', 'LINKUSDT', 'XLMUSDT')
TOL_RATIO, TOL_COVER = 1e-9, 1e-12


def load_run(tag, assets):
    arr = {}
    for p in sorted(glob.glob(os.path.join(ROOT, f'artifacts/calib/{tag}_part*.npz'))):
        with np.load(p) as d:
            arr.update({k: d[k] for k in d.files if not k.startswith('sd_')})
    val = {a: {'sp': arr[f'{a}__val__sp'].astype(np.float64), 'sn': arr[f'{a}__val__sn'].astype(np.float64),
               'wk': arr[f'{a}__val__wk']} for a in assets if f'{a}__val__sp' in arr}
    tune = np.concatenate([arr[f'{a}__tune__sn'].astype(np.float64) for a in assets if f'{a}__tune__sn' in arr])
    if 'BTCUSDT__val__st' in arr and not (np.diff(arr['BTCUSDT__val__st']) > 0).all():
        raise SystemExit(f'{tag}: BTCUSDT__val__st is not strictly increasing — order of windows unknown')
    return val, tune


def research_module():
    os.environ.setdefault('SMC_MODE', 'none')
    os.environ.setdefault('SMC_PROJECT_PATH', tempfile.mkdtemp() + '/')
    sys.path.insert(0, os.path.join(ROOT, 'research'))
    import sumo_class_train as T
    return T


def as_research(D_all, n_cond):
    """Research data dicts: + st (time order = npz order), zf / zfn zeros (inputs of cond_e only)."""
    out = {}
    for s, D in D_all.items():
        out[s] = {a: {**d, 'st': np.arange(len(d['sp'])), 'zf': np.zeros((len(d['sp']), n_cond), np.float32),
                      'zfn': np.zeros((len(d['sp']), n_cond), np.float32)} for a, d in D.items()}
    return out


def nan(x):
    return math.nan if x is None else float(x)


def compare_rows(mine, ref):
    """n_real / n_surr exact, max |d ratio| (NaN = NaN), sparse flags equal."""
    if len(mine) != len(ref):
        return {'n_equal': False, 'max_d_ratio': math.inf, 'sparse_equal': False}
    n_eq = all(a['color'] == b['color'] and a['lo'] == b['lo'] and a['n_real'] == b['n_real'] and a['n_surr'] == b['n_surr']
               for a, b in zip(mine, ref))
    d = 0.0
    for a, b in zip(mine, ref):
        ra, rb = nan(a['ratio']), nan(b['ratio'])
        if math.isnan(ra) or math.isnan(rb):
            d = d if (math.isnan(ra) and math.isnan(rb)) else math.inf
        else:
            d = max(d, abs(ra - rb))
    return {'n_equal': bool(n_eq), 'max_d_ratio': d, 'sparse_equal': all(a['sparse'] == b['sparse'] for a, b in zip(mine, ref))}


def ceil_val(v):
    return math.inf if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v)


def compare(sm, ref, part):
    """A summary (research or repo) against the json part."""
    out = {'pool': compare_rows(sm['rows'], ref['rows']),
           'ceil': {c: [ceil_val(sm['ceil'][c]), ceil_val(ref['ceil'][c])] for c in ('orange', 'blue')},
           'share': [sm['share'], ref['share']], 'n': [sm['n'], ref['n']],
           'cover_d': max(abs(sm['cover'][i] - ref['cover'][i]) for i in (0, 1)),
           'cover_null_d': max(abs(sm['cover_null'][i] - ref['cover_null'][i]) for i in (0, 1)), 'assets': {}}
    worst = 0.0
    a_ok = sorted(sm['assets']) == sorted(ref['assets'])
    for a, p in sm['assets'].items():
        q = ref['assets'].get(a)
        if q is None:
            continue
        c = compare_rows(p['rows'], q['rows'])
        same_ceil = ceil_val(p['orange']) == ceil_val(q['orange']) and ceil_val(p['blue']) == ceil_val(q['blue'])
        out['assets'][a] = {**c, 'ceil_equal': same_ceil}
        worst = max(worst, c['max_d_ratio'])
        a_ok &= c['n_equal'] and c['sparse_equal'] and same_ceil
    out['assets_max_d_ratio'] = worst
    out['ok'] = bool(out['pool']['n_equal'] and out['pool']['sparse_equal'] and out['pool']['max_d_ratio'] <= TOL_RATIO
                     and worst <= TOL_RATIO and a_ok and all(x == y for x, y in out['ceil'].values())
                     and sm['share'] == ref['share'] and sm['n'] == ref['n'] and out['cover_d'] <= TOL_COVER)
    print(f"   {part}: windows {sm['n']} | bins n exact {out['pool']['n_equal']} | max |d ratio| pool "
          f"{out['pool']['max_d_ratio']:.1e}, assets {worst:.1e} | ceilings {out['ceil']} | share {sm['share']:.4f} / "
          f"{ref['share']:.4f} | assets ok {a_ok} -> {'OK' if out['ok'] else 'FAIL'}")
    return out


def run_part(name, fn_research, fn_repo, ref, res):
    t0 = time.time()
    sr, sm = fn_research(), fn_repo()
    res[name] = {'research': compare(sr, ref, f'{name} research'), 'repo': compare(sm, ref, f'{name} repo'),
                 'secs': round(time.time() - t0, 1)}
    return sr, sm


def table(S, U, ceil_unseen):
    rows = []
    for a, p in S['assets'].items():
        rows.append({'asset': a, 'orange': p['orange'], 'blue': p['blue'], 'cover_orange': p['cover'][0],
                     'cover_blue': p['cover'][1], 'unseen_in_training': False})
    for a in H:
        p = U['assets'][a]
        rows.append({'asset': a + ' (unseen, refit C)', 'orange': p['orange'], 'blue': p['blue'], 'cover_orange': p['cover'][0],
                     'cover_blue': p['cover'][1], 'unseen_in_training': True})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs/checks/legacy/pb/nocal_repro.json'))
    a = ap.parse_args()
    with open(NOCAL, encoding='utf-8') as fh:
        nc = json.load(fh)
    assets = list(nc['variants']['glob_e']['assets'])          # research pool_assets() order (json keeps it)
    knots, zk = np.asarray(nc['knots'], dtype=np.float64), np.asarray(nc['zk'], dtype=np.float64)
    span, warm, s_lo = nc['rules']['span'], nc['rules']['warm'], nc['rules']['s_lo']
    assert (span, warm, s_lo) == (linem_s4.SPAN, linem_s4.WARM, linem.S_LO)
    D, TU, DU = {}, {}, {}
    for s in nc['seeds']:
        D[s], TU[s] = load_run(f'pool_ALL_{MODEL}_L64_{s}_refit', assets)
        DU[s], _ = load_run(f'pool_C_{MODEL}_L64_{s}_refit', assets)
    print(f"loaded: ALL {[len(D[s]) for s in D]} assets, C {[len(DU[s]) for s in DU]} assets, TUNE surrogates "
          f"{[len(TU[s]) for s in TU]}")
    T = research_module()
    n_cond = len(T.NOCAL['cond_cols'])
    Dr, DUr = as_research(D, n_cond), as_research(DU, n_cond)
    TUr = {s: (TU[s], np.zeros((len(TU[s]), n_cond), np.float32)) for s in TU}
    res = {'nocal_json': os.path.basename(NOCAL), 'tol_ratio': TOL_RATIO, 'tol_cover': TOL_COVER,
           'order': "npz order = time order (research valid_starts / positives; BTC __st strictly increasing)"}
    R_r = T._nc_rulers(Dr, knots)
    R_m = LC.rulers(D, knots)
    res['ruler_max_d'] = max(float(np.max(np.abs(R_r[k]['q'] - R_m[k]))) for k in R_m)
    S_r, S_m = run_part('pool', lambda: T._nc_summary(T._nc_cells(Dr, 'glob_e', R_r, zk, span)),
                        lambda: LC.summary(LC.cells(D, R_m, zk, span, warm, s_lo)), nc['variants']['glob_e'], res)
    nul_r = T._nc_summary(T._nc_cells(Dr, 'glob_e', R_r, zk, span, null_chart=True))['cover_null']
    nul_m = LC.summary(LC.cells(D, R_m, zk, span, warm, s_lo, null_chart=True))['cover_null']
    ref_n = nc['null_chart']['cover']
    res['null_chart'] = {'research': list(nul_r), 'repo': list(nul_m), 'json': ref_n,
                         'max_d': max(abs(x - y) for v in (nul_r, nul_m) for x, y in zip(v, ref_n))}
    res['null_chart']['ok'] = res['null_chart']['max_d'] <= TOL_COVER
    print(f"   random chart: research {nul_r}, repo {nul_m}, json {ref_n} -> {'OK' if res['null_chart']['ok'] else 'FAIL'}")
    U_r, U_m = run_part('unseen', lambda: T._nc_summary(T._nc_cells(DUr, 'glob_e', T._nc_rulers(DUr, knots, skip=H), zk,
                                                                    span, only=H)),
                        lambda: LC.summary(LC.cells(DU, LC.rulers(DU, knots, skip=H), zk, span, warm, s_lo, only=H)),
                        nc['unseen'], res)
    run_part('insample', lambda: T._nc_summary(T._nc_cells(Dr, 'glob_e', T._nc_rulers(Dr, knots, fit_from=TUr), zk, span)),
             lambda: LC.summary(LC.cells(D, LC.rulers(D, knots, fit_from=TU), zk, span, warm, s_lo)), nc['insample'], res)
    # the streaming live part (as the session-4 Pine indicator: linem_s4.LiveM) on real windows: BTC, seed a, both half-year rulers
    d, worst = D['a']['BTCUSDT'], 0.0
    for h in (True, False):
        lv = linem_s4.LiveM(R_m[('a', h)], zk, span, warm, s_lo)
        st = [lv.update(x) for x in d['sp']]
        lm_s = np.array([r['log_m'] for r in st])
        zr = linem.ruler_z(d['sp'], R_m[('a', h)], zk)
        m, sd = linem_s4.live(zr, span)
        worst = max(worst, float(np.max(np.abs(lm_s - linem.log_m(zr, m, sd, s_lo)))))
    res['stream_btc_a'] = {'windows': int(len(d['sp'])), 'max_d_log_m': worst, 'ok': worst <= 1e-10}
    print(f"   stream (LiveM) = vectorized on BTC VAL, seed a: max |d log M| {worst:.1e}")
    res['all_ok'] = bool(all(res[k][w]['ok'] for k in ('pool', 'unseen', 'insample') for w in ('research', 'repo'))
                         and res['null_chart']['ok'] and res['stream_btc_a']['ok'])
    unseen_ceil = {c: min(U_m['assets'][x][c] for x in H) for c in ('orange', 'blue')}
    ceil = {'model': MODEL, 'nocal_json': os.path.basename(NOCAL), 'ruler': 'glob_e',
            'pool': S_m['ceil'], 'share_pairs_ge_5': S_m['share'],
            'by_symbol': {x: {'orange': p['orange'], 'blue': p['blue']} for x, p in S_m['assets'].items()},
            'unseen_pool_H': U_m['ceil'], 'unseen_H': {x: {'orange': U_m['assets'][x]['orange'],
                                                           'blue': U_m['assets'][x]['blue']} for x in H},
            'unseen_symbol': unseen_ceil, 'cover_pool': S_m['cover'], 'null_chart_cover': list(nul_m),
            'table': table(S_m, U_m, unseen_ceil), 'reproduced': res['all_ok']}
    res['ceilings'] = {k: ceil[k] for k in ('pool', 'unseen_pool_H', 'unseen_symbol', 'share_pairs_ge_5')}
    print(json.dumps(res['ceilings']), '| all_ok', res['all_ok'])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    cp = os.path.join(os.path.dirname(a.out), 'ceilings.json')
    with open(cp, 'w', encoding='utf-8') as fh:
        json.dump(ceil, fh, indent=1, ensure_ascii=False)
    lines = ["# Потолки линии M — PB_s64h24_d1, линейка glob_e (R15)", "",
             f"Источник: воспроизведение `{os.path.basename(NOCAL)}` (`scripts/pb_nocal.py`, совпадение с json: "
             f"{'да' if res['all_ok'] else 'НЕТ'}). VAL 'refit' на ALL, сиды a + b, линейка с другого полугодия, живая часть "
             f"по VAL ассета (span {span}, прогрев {warm} окон). Невиданные — 'refit' на C (линейки без H).", "",
             f"Пул (R9, двусторонне): оранжевые {S_m['ceil']['orange']:g}, синие {S_m['ceil']['blue']:g}; пар ассет × цвет "
             f"с потолком ≥ 5 — {100 * S_m['share']:.0f}%. Незнакомый символ (наименьший из FIL, LINK, XLM по 'refit' на C): "
             f"оранжевые {unseen_ceil['orange']:g}, синие {unseen_ceil['blue']:g} (пул H: {U_m['ceil']['orange']:g} / "
             f"{U_m['ceil']['blue']:g}).", "",
             "| ассет | потолок оранж. | потолок син. | отмечено оранж. (M ≥ 1.25) | отмечено син. (M ≤ 0.8) |", "|---|---|---|---|---|"]
    for r in ceil['table']:
        lines.append(f"| {r['asset']} | {r['orange']:g} | {r['blue']:g} | {100 * r['cover_orange']:.0f}% | "
                     f"{100 * r['cover_blue']:.0f}% |")
    lines.append(f"| незнакомый символ | {unseen_ceil['orange']:g} | {unseen_ceil['blue']:g} | — | — |")
    with open(os.path.join(os.path.dirname(a.out), 'ceilings.md'), 'w', encoding='utf-8') as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == '__main__':
    main()
