"""Checks of the line M promise (rule R15, as research v6.3 'nocal'; written here from the rule text, the research
functions are only compared with): rulers by half-years, the live part of an asset's VAL in time order, the R9 curve by
M bins of each colour, ceilings (pool two-sided, asset one-sided) and their summary.

Data of one model run: {asset: {'sp': real logits, 'sn': surrogate logits (same windows, sflip4), 'wk': week of the
window end}} in time order (the npz order of the pool runs). Several seeds: {seed: that dict}.
"""
import math

import numpy as np
import pandas as pd

from . import linem

EDGES = (1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, math.inf)   # M bins of each colour (R9)
TOL = (0.8, 1.25)
MIN_N = 1000
HALF_CUT = '2026-01-01'
WEEK_NS = 7 * 86400 * 10 ** 9
CEIL_ASSET = 5.0
EXP_CAP = 700.0


def half1(wk):
    """True — half-year 1 of VAL (the week of the window end before HALF_CUT), False — half-year 2."""
    return np.asarray(wk) < pd.Timestamp(HALF_CUT).value // WEEK_NS


def rulers(D_all, knots, skip=(), fit_from=None):
    """{(seed, half1?): q} — linear np.quantile of the surrogate logits of the OTHER half-year of all assets of the seed
       (except skip), in the knots; fit_from = {seed: surrogate logits} — one ruler per seed for both half-years."""
    R = {}
    for s, D in D_all.items():
        if fit_from is not None:
            R[(s, True)] = R[(s, False)] = np.quantile(np.asarray(fit_from[s], dtype=np.float64), knots)
            continue
        for h in (True, False):
            sn = [np.asarray(d['sn'], dtype=np.float64)[half1(d['wk']) != h] for a, d in D.items() if d and a not in skip]
            R[(s, h)] = np.quantile(np.concatenate(sn), knots)
    return R


def cells(D_all, R, zk, span=linem.SPAN, warm=linem.WARM, s_lo=linem.S_LO, only=None, null_chart=False):
    """[(seed, asset, log M of real windows, log M of surrogates)] after the warm-up: for each half-year h the z of ALL
       windows of the asset by the ruler (seed, h), the live part over the real windows (null_chart: over the
       surrogates), windows of h with the number >= warm from the start of the asset's VAL."""
    out = []
    for s, D in D_all.items():
        for a, d in D.items():
            if not d or (only is not None and a not in only):
                continue
            sp, sn = np.asarray(d['sp'], dtype=np.float64), np.asarray(d['sn'], dtype=np.float64)
            h1 = half1(d['wk'])
            num = np.arange(len(sp))
            for h in (True, False):
                if not (h1 == h).any():
                    continue
                zr, zn = linem.ruler_z(sp, R[(s, h)], zk), linem.ruler_z(sn, R[(s, h)], zk)
                m, sd = linem.live(zn if null_chart else zr, span)
                ok = (h1 == h) & (num >= warm)
                out.append((s, a, linem.log_m(zr[ok], m[ok], sd[ok], s_lo), linem.log_m(zn[ok], m[ok], sd[ok], s_lo)))
    return out


def curve(lr, ln):
    """R9 curve: rows per colour and bin — n_real, n_surr, fact (actual multiple), exp (promised), ratio = fact / exp,
       sparse (fewer than MIN_N windows of a class). Orange: fact = share of the bin among real / among surrogates,
       promised = mean M of the bin's surrogates; blue: fact = share among surrogates / among real, promised = mean 1/M of
       the bin's real windows. exp() is capped at e^700."""
    rows = []
    for color, sg in (('orange', 1.0), ('blue', -1.0)):
        xp, xn = sg * lr, sg * ln
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            l_, h_ = math.log(lo), math.log(hi)
            bp = (xp > 0) & (xp >= l_) & (xp < h_)
            bn = (xn > 0) & (xn >= l_) & (xn < h_)
            n_p, n_n = int(bp.sum()), int(bn.sum())
            row = {'color': color, 'lo': lo, 'n_real': n_p, 'n_surr': n_n, 'ratio': math.nan, 'sparse': min(n_p, n_n) < MIN_N}
            if n_p and n_n:
                if color == 'orange':
                    fact = (n_p / len(lr)) / (n_n / len(ln))
                    exp_ = float(np.exp(np.minimum(xn[bn], EXP_CAP)).mean())
                else:
                    fact = (n_n / len(ln)) / (n_p / len(lr))
                    exp_ = float(np.exp(np.minimum(xp[bp], EXP_CAP)).mean())
                row.update({'fact': fact, 'exp': exp_, 'ratio': fact / exp_})
            rows.append(row)
    return rows


def ceiling(rows, color, two_sided=True, cap=math.inf):
    """Lower edge of the first bin from below that fails (ratio below TOL[0]; two-sided — or above TOL[1]) or is sparse;
       inf when all pass; not above cap."""
    for r in rows:
        if r['color'] != color:
            continue
        ratio = r['ratio']
        if r['sparse'] or not ratio >= TOL[0] or (two_sided and ratio > TOL[1]):
            return min(r['lo'], cap)
    return cap


def cover(lr):
    t = math.log(1.25)
    return (float((lr >= t).mean()), float((lr <= -t).mean())) if len(lr) else (math.nan, math.nan)


def summary(cs):
    """Pool curve and ceilings (two-sided), per-asset ceilings (one-sided, not above the pool), share of asset x colour
       pairs with the ceiling >= CEIL_ASSET, covers of real windows and of surrogates."""
    lr = np.concatenate([c[2] for c in cs])
    ln = np.concatenate([c[3] for c in cs])
    rows = curve(lr, ln)
    ceil = {c: ceiling(rows, c) for c in ('orange', 'blue')}
    per = {}
    for a in dict.fromkeys(c[1] for c in cs):
        la = np.concatenate([c[2] for c in cs if c[1] == a])
        na = np.concatenate([c[3] for c in cs if c[1] == a])
        ra = curve(la, na)
        per[a] = {'orange': ceiling(ra, 'orange', False, ceil['orange']), 'blue': ceiling(ra, 'blue', False, ceil['blue']),
                  'cover': cover(la), 'rows': ra}
    pairs = [p[c] for p in per.values() for c in ('orange', 'blue')]
    return {'rows': rows, 'ceil': ceil, 'assets': per, 'share': float(np.mean([v >= CEIL_ASSET for v in pairs])),
            'cover': cover(lr), 'cover_null': cover(ln), 'n': int(len(lr))}
