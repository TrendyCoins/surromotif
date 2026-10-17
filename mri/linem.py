"""Line M without per-asset calibration (rules R15 / R16; research v6.3 MODE 'nocal', NOCAL v2, ruler glob_e).

Two parts, everything in float64:
    ruler (one per model, for every symbol): z = interp(logit, q, zk) — a straight line between the knots in the z scale,
        the edge value beyond the outer knots (as np.interp); q — 37 quantiles of the surrogate logits, zk = Phi^-1(knots)
    live part (per chart): m_ema, s_ema — normalized EMA of z and z^2 over the valid windows of the chart, the current one
        included (pandas ewm(span, adjust=True)): a = 2 / (span + 1); n1 = (1 - a) n1 + z; n2 = (1 - a) n2 + z^2;
        d = (1 - a) d + 1; m_ema = n1 / d; s_ema = sqrt(max(n2 / d - m_ema^2, 0)); zeros before the first valid window
    log M (research _nc_logm): m = max(m_ema, 0); s = min(max(s_ema, s_lo), 1); zc = min(z, m / max(1 - s^2, 1e-12)) when
        s < 1, else z; log M = -(zc - m)^2 / (2 s^2) - ln s + zc^2 / 2 (ratio of the bells N(m, s^2) / N(0, 1))
    display: colour = sign of log M (orange > 0 — more often than random, blue < 0 — less often, 0 — the floor),
        the number = exp(|log M|) (M of R9), not above the ceiling of its colour (above — drawn at the ceiling, 'N+')
    warm-up (R16): windows are numbered from 0 over the valid windows of the chart; the line is shown from n >= warm.
        An invalid window does not change the state and does not get a number.

Defaults are the checked values (span 8064 bars, warm-up 288 windows, s_lo 0.75); other values are 'not checked'.
"""
import math

import numpy as np
import pandas as pd

SPAN = 8064
WARM = 288
S_LO = 0.75
PEAK_EPS = 1e-12


def ruler_z(logit, q, zk):
    """z of windows by the table ruler (research _nc_z, kind 'glob_e')."""
    return np.interp(np.asarray(logit, dtype=np.float64), np.asarray(q, dtype=np.float64), np.asarray(zk, dtype=np.float64))


def ruler_z1(x, q, zk):
    """The same for one number, written the way the Pine indicator computes it (and as np.interp does: the knot j with
       q[j] <= x < q[j + 1], slope = (zk[j + 1] - zk[j]) / (q[j + 1] - q[j]), z = slope (x - q[j]) + zk[j])."""
    n = len(q)
    if x < q[0]:
        return zk[0]
    if x >= q[n - 1]:
        return zk[n - 1]
    j = 0
    while q[j + 1] <= x:
        j += 1
    if x == q[j]:
        return zk[j]
    return (zk[j + 1] - zk[j]) / (q[j + 1] - q[j]) * (x - q[j]) + zk[j]


def live(z, span=SPAN):
    """m_ema, s_ema over a series of z of consecutive valid windows (research _nc_live: pandas ewm, adjust=True)."""
    z = np.asarray(z, dtype=np.float64)
    m = pd.Series(z).ewm(span=span, adjust=True).mean().to_numpy()
    v = pd.Series(z * z).ewm(span=span, adjust=True).mean().to_numpy()
    return m, np.sqrt(np.maximum(v - m * m, 0.0))


def log_m(z, m_ema, s_ema, s_lo=S_LO):
    """log M of windows by their z and the live m_ema, s_ema (research _nc_logm), vectorized."""
    m = np.maximum(m_ema, 0.0)
    s = np.clip(s_ema, s_lo, 1.0)
    zc = np.where(s < 1.0, np.minimum(z, m / np.maximum(1.0 - s * s, PEAK_EPS)), z)
    return -(zc - m) ** 2 / (2 * s * s) - np.log(s) + zc * zc / 2


def log_m1(z, m_ema, s_ema, s_lo=S_LO):
    """The same for one window, in the order of operations of the Pine indicator."""
    m = max(m_ema, 0.0)
    s = min(max(s_ema, s_lo), 1.0)
    zc = min(z, m / max(1.0 - s * s, PEAK_EPS)) if s < 1.0 else z
    return -(zc - m) ** 2 / (2.0 * s * s) - math.log(s) + zc * zc / 2.0


class LiveM:
    """The live part bar by bar (state of the Pine indicator: n1, n2, d, n). update() — on a valid window only."""

    def __init__(self, q, zk, span=SPAN, warm=WARM, s_lo=S_LO):
        self.q, self.zk = [float(x) for x in q], [float(x) for x in zk]
        self.k = 1.0 - 2.0 / (span + 1.0)
        self.warm, self.s_lo = warm, s_lo
        self.n1 = self.n2 = self.d = 0.0
        self.n = -1                                    # number of the last valid window (from 0)

    def update(self, logit):
        z = ruler_z1(logit, self.q, self.zk)
        self.n += 1
        self.n1 = self.k * self.n1 + z
        self.n2 = self.k * self.n2 + z * z
        self.d = self.k * self.d + 1.0
        m_ema = self.n1 / self.d
        s_ema = math.sqrt(max(self.n2 / self.d - m_ema * m_ema, 0.0))
        lm = log_m1(z, m_ema, s_ema, self.s_lo)
        return {'z': z, 'm_ema': m_ema, 's_ema': s_ema, 'log_m': lm, 'n': self.n, 'shown': self.n >= self.warm}


def line(logit, valid, q, zk, span=SPAN, warm=WARM, s_lo=S_LO):
    """Vectorized line M over the windows of a chart in time order. logit — per window (bar), valid — window valid
       (rule R13); invalid windows get NaN and do not move the state. Returns dict of per-window arrays: z, m_ema, s_ema,
       log_m, n (-1 at invalid windows), shown."""
    logit = np.asarray(logit, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    out = {k: np.full(len(logit), np.nan) for k in ('z', 'm_ema', 's_ema', 'log_m')}
    out['n'] = np.full(len(logit), -1, dtype=np.int64)
    out['shown'] = np.zeros(len(logit), dtype=bool)
    idx = np.where(valid)[0]
    if len(idx):
        z = ruler_z(logit[idx], q, zk)
        m, s = live(z, span)
        out['z'][idx], out['m_ema'][idx], out['s_ema'][idx] = z, m, s
        out['log_m'][idx] = log_m(z, m, s, s_lo)
        out['n'][idx] = np.arange(len(idx))
        out['shown'][idx] = np.arange(len(idx)) >= warm
    return out


def display(lm, ceil_orange, ceil_blue):
    """Number on the line and its colour for log M values: (value, colour, capped). colour +1 orange, -1 blue, 0 floor;
       value = exp(|log M|) not above the ceiling of the colour; capped — above the ceiling ('N+')."""
    lm = np.asarray(lm, dtype=np.float64)
    col = np.sign(lm)
    m = np.exp(np.abs(lm))
    cap = np.where(col > 0, ceil_orange, np.where(col < 0, ceil_blue, np.inf))
    return np.minimum(m, cap), col, m > cap
