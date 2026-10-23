"""Line M of the windows in time order (the export of research/sumo_class_tf.py 'line21'), float64:

    z      = interp(logit, q, zk): a straight line between the knots of the ruler, beyond the outer knots the edge value
    live   p — number of the window (from 0 over the existing windows); the set — the windows [lo, hi), hi = p - lag + 1,
           lo = max(0, hi - L); n = hi - lo (not below 0); s1, s2 — sums of z and z^2 over it (differences of cumulative
           sums, as research SuMo_Class_Live chart_ms, kind 'win', family F3):
               mh = s1 / max(n, 1); vh = max(s2 / max(n, 1) - mh^2, 0)
               n > 0: m = max(s1 / n - k sqrt(vh) sqrt(tau / n), 0), s = sqrt(vh);   n = 0: m = 0, s = 1
    log M  (research _nc_logm): m+ = max(m, 0); s+ = min(max(s, s_lo), 1); zc = min(z, m+ / max(1 - s+^2, 1e-12)) if s+ < 1
           else z; log M = -(zc - m+)^2 / (2 s+^2) - ln s+ + zc^2 / 2 — the ratio of the bells N(m+, s+^2) / N(0, 1)
    line   from the window p >= show_from; strength = min(exp |log M|, ceiling) (M for orange, 1 / M for blue); side = sign
           of log M: +1 orange (windows like this are met more often than on a random chart), -1 blue (less often), 0 — M = 1
"""
import numpy as np

PEAK_EPS = 1e-12


def ruler_z(model, logit):
    return np.interp(np.asarray(logit, dtype=np.float64), model.q, model.zk)


def live(model, z):
    """m, s and n of the live part of every window of a series of consecutive windows (chart_ms with one tile from 0)."""
    z = np.asarray(z, dtype=np.float64)
    c1 = np.concatenate([[0.0], np.cumsum(z)])
    c2 = np.concatenate([[0.0], np.cumsum(z * z)])
    p = np.arange(len(z), dtype=np.int64)
    hi = p - model.live_lag + 1
    lo = np.maximum(0, hi - model.live_l)
    n = np.maximum(hi - lo, 0)
    hc = np.maximum(hi, lo)
    s1, s2 = c1[hc] - c1[lo], c2[hc] - c2[lo]
    nn = np.maximum(n, 1)
    mh = s1 / nn
    vh = np.maximum(s2 / nn - mh * mh, 0.0)
    ok = n > 0
    ts = np.where(ok, n, 1)
    m = np.where(ok, s1 / ts, 0.0)
    s2v = np.where(ok, (n * vh + 0.0) / ts, 1.0)          # as chart_ms: (n vh + u) / (n + u), u = 0 virtual windows
    m = np.where(ok, np.maximum(m - model.live_k * np.sqrt(s2v) * np.sqrt(model.live_tau / ts), 0.0), 0.0)
    return m, np.sqrt(s2v), n


def log_m(model, z, m, s):
    m = np.maximum(m, 0.0)
    s = np.clip(s, model.s_lo, 1.0)
    zc = np.where(s < 1.0, np.minimum(z, m / np.maximum(1.0 - s * s, PEAK_EPS)), z)
    return -(zc - m) ** 2 / (2 * s * s) - np.log(s) + zc * zc / 2


def line(model, logit):
    """Line M of consecutive windows: dict of per-window arrays z, m, s, n_live, line (p >= show_from), log_m, strength,
       side (NaN where no line)."""
    z = ruler_z(model, logit)
    m, s, n = live(model, z)
    shown = np.arange(len(z)) >= model.show_from
    lm = np.full(len(z), np.nan)
    lm[shown] = log_m(model, z[shown], m[shown], s[shown])
    with np.errstate(over='ignore'):
        strength = np.where(shown, np.minimum(np.exp(np.abs(lm)), model.ceiling), np.nan)
    return {'z': z, 'm': m, 's': s, 'n_live': n, 'line': shown, 'log_m': lm, 'strength': strength,
            'side': np.where(shown, np.sign(lm), np.nan)}
