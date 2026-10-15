"""Rule R12 (docs/rules.md, session 3): per-asset Platt calibration of NT3_PRV_ch48 and its checks.

    python scripts/nt3_calibrate.py [--boot 200] [--out docs/checks/nt3]

Writes into --out:
    calibration.json  per-asset Platt (whole VAL and by half-year) of the 'refit' models on ALL (seeds a + b together) —
                      the export values; transfer 'stop' C -> 'refit' C per asset (v5.2: total / market / generation);
                      unknown symbol (global Platt, 'refit' C); references V0 / V1; cross-checks against the research log
    calib_curve.json  R9 on the pool of 31 (VAL 'refit' ALL, Platt of the asset from the other half) -> pool ceiling;
                      R9 on H for the unknown symbol (global Platt / Platt of H, 'refit' C); V0 / V1 curves (reference)
    ceilings.json     R11 per asset and color (one-sided ratio >= 0.8, >= 1000 windows of each class), at most the new pool
                      ceiling; unknown symbol = the lowest of FIL, LINK, XLM ('refit' C, global Platt of the other half);
                      V0 / V1 ceilings (reference); share of VAL windows at the ceiling with the export Platt
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from calib_curve import EDGES, MIN_N, TOL, ceilings, curve, show  # noqa: E402
from ceilings import MIN_RATIO  # noqa: E402
from mri.calibration import WEEK_NS, calib_ratio, load_part, platt_fit  # noqa: E402

MODEL = 'NT3_PRV_ch48'
SEEDS = ('a', 'b')
H_ASSETS = ('FILUSDT', 'LINKUSDT', 'XLMUSDT')
THR = 1.25
TOL_TRANSFER = 0.10                       # v5.2 calib_tol: |total - 1| <= 0.10 at both thresholds
CUT = pd.Timestamp('2026-01-01').value // WEEK_NS
HALVES = {'H1': lambda wk: wk < CUT, 'H2': lambda wk: wk >= CUT}
OTHER = {'H1': 'H2', 'H2': 'H1'}


def run_name(st, seed, mode):
    return f'pool_{st}_{MODEL}_L64_{seed}' + ('_refit' if mode == 'refit' else '')


def cat(val, assets, half=None):
    """Real / surrogate logits of the assets over both seeds (val: {seed: {asset: (sp, sn, wk)}}), optionally one half."""
    sp, sn = [], []
    for s in SEEDS:
        for a in assets:
            p_, n_, wk = val[s][a]
            m = np.ones(len(wk), bool) if half is None else HALVES[half](wk)
            sp.append(p_[m])
            sn.append(n_[m])
    return np.concatenate(sp), np.concatenate(sn)


def fit_shift(val, assets, half):
    """V1: one logistic regression a * logit + b_i (common slope, own shift of each asset), Newton."""
    xs, ys, gs = [], [], []
    for gi, a in enumerate(assets):
        sp, sn = cat(val, [a], half)
        xs += [sp, sn]
        ys += [np.ones(len(sp)), np.zeros(len(sn))]
        gs += [np.full(len(sp) + len(sn), gi)]
    x, y, g = np.concatenate(xs), np.concatenate(ys), np.concatenate(gs)
    k = len(assets)
    th = np.concatenate([[1.0], np.zeros(k)])
    for _ in range(50):
        z = th[0] * x + th[1:][g]
        p = 1.0 / (1.0 + np.exp(-z))
        r, w = p - y, p * (1 - p)
        grad = np.concatenate([[(r * x).sum()], np.bincount(g, weights=r, minlength=k)])
        H = np.zeros((k + 1, k + 1))
        H[0, 0] = (w * x * x).sum()
        H[0, 1:] = H[1:, 0] = np.bincount(g, weights=w * x, minlength=k)
        H[1:, 1:] = np.diag(np.bincount(g, weights=w, minlength=k))
        step = np.linalg.solve(H + 1e-9 * np.eye(k + 1), grad)
        th -= step
        if np.abs(step).max() < 1e-10:
            break
    return {a: (float(th[0]), float(th[1 + i])) for i, a in enumerate(assets)}


def oof_logs(val, assets, params):
    """Out-of-fold log lift: params[half][asset] = (a, b) fitted on that half; windows of half h get the Platt of the
       other half. Returns lp, ln, bootstrap unit (week x seed) of real windows."""
    lp, ln, wp = [], [], []
    for h in HALVES:
        for s in SEEDS:
            for a_ in assets:
                pa, pb = params[OTHER[h]][a_]
                sp, sn, wk = val[s][a_]
                m = HALVES[h](wk)
                lp.append(pa * sp[m] + pb)
                ln.append(pa * sn[m] + pb)
                wp.append(wk[m] * 10 + (s == 'b'))
    return np.concatenate(lp), np.concatenate(ln), np.concatenate(wp)


def asset_ceiling(rows, color, cap):
    """R11: lower edge of the first bin from below that fails (ratio < 0.8) or has < 1000 windows of a class; <= cap."""
    for r in rows:
        if r['color'] != color:
            continue
        if r['sparse'] or r.get('ratio', 0.0) < MIN_RATIO:
            return min(r['lo'], cap)
    return cap


def ceiling_table(val, assets, params, cap):
    out, bins = {}, {}
    for a in assets:
        lp, ln, _ = oof_logs(val, [a], params)
        rows = curve(lp, ln)
        bins[a] = [{k: r.get(k) for k in ('color', 'lo', 'n_real', 'n_surr', 'factual', 'expected', 'ratio', 'sparse')}
                   for r in rows]
        out[a] = {'more': asset_ceiling(rows, 'orange', cap['orange']), 'less': asset_ceiling(rows, 'blue', cap['blue'])}
    return out, bins


def cap_of(c):
    return {k: (np.inf if v is None else v) for k, v in c.items()}


def transfer(stop, refit, assets):
    """v5.2 per asset (or pooled when assets has several): per seed p11 = Platt on TUNE 'stop', p12 = on VAL 'stop',
       p22 = on VAL 'refit'; ratios at 1.25 and 0.8; mean over seeds."""
    rows = []
    for s in SEEDS:
        sp1t = np.concatenate([stop['tune'][s][a][0] for a in assets])
        sn1t = np.concatenate([stop['tune'][s][a][1] for a in assets])
        sp1v = np.concatenate([stop['val'][s][a][0] for a in assets])
        sn1v = np.concatenate([stop['val'][s][a][1] for a in assets])
        sp2 = np.concatenate([refit[s][a][0] for a in assets])
        sn2 = np.concatenate([refit[s][a][1] for a in assets])
        p11, p12, p22 = platt_fit(sp1t, sn1t), platt_fit(sp1v, sn1v), platt_fit(sp2, sn2)
        rows.append({nm: (calib_ratio(pa, pb, THR), calib_ratio(pa, pb, 1 / THR))
                     for nm, (pa, pb) in (('total', (p11, p22)), ('market', (p11, p12)), ('generation', (p12, p22)))})
    out = {nm: [float(np.mean([r[nm][j] for r in rows])) for j in (0, 1)] for nm in ('total', 'market', 'generation')}
    out['pass'] = bool(all(abs(x - 1) <= TOL_TRANSFER for x in out['total']))
    out['borderline'] = bool(out['pass'] and any(x < 1 / 1.1 for x in out['total']))
    return out


def jdump(path, obj):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False,
                  default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', type=int, default=200)
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs/checks/nt3'))
    a = ap.parse_args()
    root = os.path.join(ROOT, 'artifacts/calib')
    all_ = {s: load_part(root, run_name('ALL', s, 'refit')) for s in SEEDS}
    rc = {s: load_part(root, run_name('C', s, 'refit')) for s in SEEDS}
    stop = {part: {s: load_part(root, run_name('C', s, 'stop'), part) for s in SEEDS} for part in ('tune', 'val')}
    assets = sorted(all_['a'])
    assert len(assets) == 31 and all(sorted(v) == assets for v in (all_['b'], rc['a'], rc['b']))

    # ---- per-asset Platt (V2): whole VAL (export) and by half
    platt = {a_: {'all': platt_fit(*cat(all_, [a_])), **{h: platt_fit(*cat(all_, [a_], h)) for h in HALVES}}
             for a_ in assets}
    v2 = {h: {a_: platt[a_][h] for a_ in assets} for h in HALVES}
    # ---- references: V0 (one Platt for the 31), V1 (common slope, own shift)
    v0_all = platt_fit(*cat(all_, assets))
    v0 = {h: dict.fromkeys(assets, platt_fit(*cat(all_, assets, h))) for h in HALVES}
    v1_all = fit_shift(all_, assets, None)
    v1 = {h: fit_shift(all_, assets, h) for h in HALVES}

    # ---- R9 pool (V2) -> pool ceiling
    lp, ln, wk = oof_logs(all_, assets, v2)
    pool = curve(lp, ln, wk, boot=a.boot)
    show('R9 POOL of 31, Platt of the asset (other half) — rule', pool)
    pool_ceil = ceilings(pool)
    print('pool ceiling (new):', pool_ceil)
    cap = cap_of(pool_ceil)
    # ---- R11 per asset (V2)
    table, bins = ceiling_table(all_, assets, v2, cap)
    # ---- unknown symbol: global Platt on 'refit' C (31 assets, a + b); H ceilings with the global Platt of the other half
    g_all = platt_fit(*cat(rc, assets))
    g_half = {h: dict.fromkeys(assets, platt_fit(*cat(rc, assets, h))) for h in HALVES}
    h_tab, h_bins = ceiling_table(rc, H_ASSETS, g_half, cap)
    unknown = {k: min(h_tab[x][k] for x in H_ASSETS) for k in ('more', 'less')}
    lph, lnh, wkh = oof_logs(rc, H_ASSETS, g_half)
    h_curve_global = curve(lph, lnh, wkh, boot=a.boot)
    show('R9 on H (FIL, LINK, XLM; refit C), global Platt of the other half — report', h_curve_global)
    hh = {h: dict.fromkeys(H_ASSETS, platt_fit(*cat(rc, H_ASSETS, h))) for h in HALVES}
    lph2, lnh2, wkh2 = oof_logs(rc, H_ASSETS, hh)
    h_curve_h = curve(lph2, lnh2, wkh2, boot=a.boot)
    show('R9 on H, Platt of H only (other half) — reference', h_curve_h)
    h_tab_h, _ = ceiling_table(rc, H_ASSETS, hh, cap)
    # ---- V0 / V1 references: curve and ceilings the same way
    ref = {}
    for nm, prm in (('V0', v0), ('V1', v1)):
        lp_, ln_, wk_ = oof_logs(all_, assets, prm)
        rows = curve(lp_, ln_, wk_, boot=0)
        c_ = ceilings(rows)
        t_, _ = ceiling_table(all_, assets, prm, cap_of(c_))
        ref[nm] = {'pool': rows, 'pool_ceiling': c_, 'ceilings': t_}
    # ---- transfer 'stop' C -> 'refit' C per asset (all 31) and pooled (cross-check with the research log, section (5))
    tr = {a_: transfer(stop, rc, [a_]) for a_ in assets}
    tr_pool = transfer(stop, rc, assets)
    fails = [a_ for a_, v in tr.items() if not v['pass']]
    border = [a_ for a_, v in tr.items() if v['borderline']]
    # ---- share of VAL windows at the ceiling with the export Platt of the asset (both seeds)
    at_ceil = {}
    for a_ in assets:
        pa, pb = platt[a_]['all']
        ll = np.concatenate([pa * all_[s][a_][0] + pb for s in SEEDS])
        c = table[a_]
        at_ceil[a_] = {'more': float(np.mean(ll >= np.log(c['more']))), 'less': float(np.mean(-ll >= np.log(c['less'])))}

    os.makedirs(a.out, exist_ok=True)
    cal = {'rule': 'R12', 'model': MODEL, 'source': "VAL of the 'refit' models on ALL, seeds a + b together",
           'wk_cut': int(CUT),
           'platt_asset': {a_: {'all': platt[a_]['all'], 'H1': platt[a_]['H1'], 'H2': platt[a_]['H2'],
                                'windows_val_per_seed': int(len(all_['a'][a_][0]))} for a_ in assets},
           'unknown_symbol': {'platt': g_all, 'platt_by_half': {h: g_half[h][assets[0]] for h in HALVES},
                              'fit_on': "VAL 'refit' C, 31 assets, seeds a + b",
                              'platt_H_only': platt_fit(*cat(rc, H_ASSETS)),
                              'platt_H_only_by_half': {h: hh[h][H_ASSETS[0]] for h in HALVES}},
           'V0': {'all': v0_all, 'by_half': {h: v0[h][assets[0]] for h in HALVES}},
           'V1': {'all': v1_all, 'by_half': v1},
           'transfer': {'tol': TOL_TRANSFER, 'per_asset': tr, 'fails': fails, 'borderline': border, 'pooled_31': tr_pool}}
    jdump(os.path.join(a.out, 'calibration.json'), cal)
    jdump(os.path.join(a.out, 'calib_curve.json'),
          {'rule': 'R12 / R9', 'edges': [e if np.isfinite(e) else None for e in EDGES], 'tol': TOL, 'min_n': MIN_N,
           'pool': pool, 'ceiling_pool': pool_ceil, 'holdout_global_platt': h_curve_global,
           'ceiling_holdout_global_platt': ceilings(h_curve_global), 'holdout_H_platt': h_curve_h,
           'ceiling_holdout_H_platt': ceilings(h_curve_h),
           'V0': {'pool': ref['V0']['pool'], 'ceiling_pool': ref['V0']['pool_ceiling']},
           'V1': {'pool': ref['V1']['pool'], 'ceiling_pool': ref['V1']['pool_ceiling']}})
    jdump(os.path.join(a.out, 'ceilings.json'),
          {'rule': 'R12 / R11', 'min_ratio': MIN_RATIO, 'min_n': MIN_N, 'pool_ceiling': pool_ceil, 'ceilings': table,
           'unknown_symbol': unknown, 'H_ceilings_global_platt': h_tab, 'H_ceilings_H_platt': h_tab_h,
           'share_at_ceiling_val': at_ceil, 'V0': ref['V0']['ceilings'], 'V1': ref['V1']['ceilings'],
           'bins': bins, 'bins_H_global_platt': h_bins})

    # ---- print
    print(f"\n{'asset':<10} {'a_i':>7} {'b_i':>8} | {'a H1':>6} {'b H1':>7} {'a H2':>6} {'b H2':>7} | ceil more/less | at ceil "
          f"VAL | transfer total (market, generation)")
    for a_ in assets:
        pa, pb = platt[a_]['all']
        t_ = tr[a_]
        print(f"{a_:<10} {pa:7.3f} {pb:+8.3f} | {platt[a_]['H1'][0]:6.3f} {platt[a_]['H1'][1]:+7.3f} {platt[a_]['H2'][0]:6.3f} "
              f"{platt[a_]['H2'][1]:+7.3f} | {table[a_]['more']:>5g} / {table[a_]['less']:<5g} | "
              f"{100 * at_ceil[a_]['more']:5.1f}% / {100 * at_ceil[a_]['less']:4.1f}% | x{t_['total'][0]:.3f} / "
              f"x{t_['total'][1]:.3f} (x{t_['market'][0]:.3f} / x{t_['market'][1]:.3f}, x{t_['generation'][0]:.3f} / "
              f"x{t_['generation'][1]:.3f}){'' if t_['pass'] else '  FAIL'}{'  borderline' if t_['borderline'] else ''}"
              + ('  (H)' if a_ in H_ASSETS else ''))
    print('pool ceiling', pool_ceil, '| unknown symbol', unknown, '| H ceilings (global Platt)', h_tab,
          '| H ceilings (Platt of H)', h_tab_h)
    print('global Platt (refit C, 31):', g_all, '| V0:', v0_all, '| transfer fails:', fails, '| borderline:', border)
    print('pooled transfer (research log (5): x0.963 / x0.963, market x0.969 / x0.945, generation x0.995 / x1.018):',
          {k: [round(x, 3) for x in v] for k, v in tr_pool.items() if k in ('total', 'market', 'generation')})
    for nm in ('V0', 'V1'):
        print(nm, 'pool ceiling', ref[nm]['pool_ceiling'])


if __name__ == '__main__':
    main()
