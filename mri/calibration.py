"""Platt calibration of the logit into lift and its checks (rule R4, docs/rules.md).

lift = exp(a * logit + b) — with 1:1 real / surrogate balance, the odds of 'real' for the window's score:
how many times windows like this are more frequent in the real market than in the surrogate (sflip4 mirror).
Not a probability and not a direction forecast. M = max(lift, 1 / lift).

platt_fit and lift_ci are copies of research/sumo_class_train.py (same names); tests/test_calibration.py compares
them with the originals.
"""
import glob
import math
import os

import numpy as np

THR = 1.25                       # marks: lift >= 1.25 (orange) or <= 0.8 (blue)
BAND = 1.5                       # band at the threshold (reference): [1.25, 1.5) and (0.667, 0.8]
WEEK_NS = 7 * 86400 * 10 ** 9


def platt_fit(sp, sn, iters=30):
    """Copy of research platt_fit: logistic regression of real (1) / surrogate (0) on the logit."""
    s = np.concatenate([sp, sn])
    y = np.concatenate([np.ones(len(sp)), np.zeros(len(sn))])
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(a * s + b)))
        g = np.array([((p - y) * s).mean(), (p - y).mean()])
        w = p * (1 - p)
        H = np.array([[(w * s * s).mean(), (w * s).mean()], [(w * s).mean(), w.mean()]])
        step = np.linalg.solve(H + 1e-9 * np.eye(2), g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-10:
            break
    return float(a), float(b)


def lift_ci(fp, fn, wp, wn, B=500, seed=0):
    """Copy of research lift_ci: coverage of real / surrogate by marks, realized lift, 95% bootstrap CI by weeks."""
    weeks = np.union1d(wp, wn)
    ip, in_ = np.searchsorted(weeks, wp), np.searchsorted(weeks, wn)
    G = len(weeks)
    a, na = np.bincount(ip, weights=fp, minlength=G), np.bincount(ip, minlength=G)
    c, nc = np.bincount(in_, weights=fn, minlength=G), np.bincount(in_, minlength=G)
    cov_p, cov_n = a.sum() / na.sum(), c.sum() / nc.sum()
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, G, size=(B, G))
    bp = a[pick].sum(1) / np.maximum(na[pick].sum(1), 1)
    bn = c[pick].sum(1) / np.maximum(nc[pick].sum(1), 1)
    lifts = bp / np.maximum(bn, 1e-12)
    lift = cov_p / cov_n if cov_n > 0 else (np.inf if cov_p > 0 else np.nan)
    return (float(cov_p), float(cov_n), float(lift),
            float(np.percentile(lifts, 2.5)), float(np.percentile(lifts, 97.5)))


def load_part(root, run, part='val'):
    """Logits of a pool run (calib npz split into parts '<run>_part*.npz'): {asset: (sp, sn, wk)} of part 'val' | 'tune'."""
    arrs = {}
    for p in sorted(glob.glob(os.path.join(root, f'{run}_part*.npz'))):
        with np.load(p) as d:
            arrs.update({k: d[k] for k in d.files if f'__{part}__' in k})
    if not arrs:
        raise FileNotFoundError(os.path.join(root, f'{run}_part*.npz'))
    assets = sorted({k.split('__')[0] for k in arrs})
    return {a: (arrs[f'{a}__{part}__sp'].astype(np.float64), arrs[f'{a}__{part}__sn'].astype(np.float64),
                arrs[f'{a}__{part}__wk'].astype(np.int64)) for a in assets}


def load_val(root, seed):
    """VAL logits of the F3_g 'refit' model of a seed: {asset: (sp, sn, wk)} (calib npz split into parts)."""
    return load_part(root, f'pool_C_F3_g_L64_{seed}_refit')


def pooled(val, assets=None, mask_fn=None):
    sp, sn = [], []
    for a, (p_, n_, wk) in val.items():
        if assets is not None and a not in assets:
            continue
        m = np.ones(len(p_), bool) if mask_fn is None else mask_fn(wk)
        sp.append(p_[m])
        sn.append(n_[m])
    return np.concatenate(sp), np.concatenate(sn)


def mark_lift(sp, sn, a, b, wk=None, ci=False):
    """Observed lift of orange (est >= 1.25) and blue (est <= 0.8) marks and of the bands at the threshold."""
    lp, ln = a * sp + b, a * sn + b
    t, u = math.log(THR), math.log(BAND)
    out = {}
    for nm, mp, mn in (('orange', lp >= t, ln >= t), ('blue', lp <= -t, ln <= -t),
                       ('band_orange', (lp >= t) & (lp < u), (ln >= t) & (ln < u)),
                       ('band_blue', (lp <= -t) & (lp > -u), (ln <= -t) & (ln > -u))):
        cp, cn = float(mp.mean()), float(mn.mean())
        row = {'share_real': cp, 'share_surr': cn, 'lift': cp / cn if cn > 0 else float('nan')}
        if ci and nm in ('orange', 'blue'):
            _, _, lf, lo, hi = lift_ci(mp.astype(float), mn.astype(float), wk, wk)
            row.update({'lo': lo, 'hi': hi})
        out[nm] = row
    return out


def calib_ratio(pa, pb, thr):
    """Copy of research _calib_ratio: how many times calibration pa estimates the lift of the window that pb puts on thr."""
    s = (math.log(thr) - pb[1]) / pb[0]
    return math.exp(pa[0] * s + pa[1]) / thr
