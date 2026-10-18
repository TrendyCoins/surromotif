"""Line M without per-asset calibration: the ruler of the model, the live part of the chart, log M, display.

Session 5 (the indicator; rules R17-R19): the live part is a fixed window — formula A of research SuMo_Class_Live
('4032+144': L = 4032 windows, n0 = 144 virtual random windows, lag 64), the line from the full window (n >= 4096), one
ceiling 10 for both colours. Session 4 (rules R15 / R16, research 'nocal'): EMA of z and z^2 from the start of the chart,
span 8064, warm-up 288 — kept below only as the record of that line (live, LiveM, line); it is not exported.

Everything in float64:
    ruler (one per model, for every symbol): z = interp(logit, q, zk) — a straight line between the knots in the z scale,
        the edge value beyond the outer knots (as np.interp); q — 37 quantiles of the surrogate logits, zk = Phi^-1(knots)
    live part (rule R18): windows are numbered from 0 over the valid windows of the chart (n). The calibration set of the
        window n — the real windows max(0, n - 4095) .. n - 64 (the last 4032 windows ending 64 windows before the current
        one: no window of the set shares a bar with it); k = min(max(n - 63, 0), 4032) of them, sums S1 (z) and S2 (z^2);
        plus 144 virtual random windows (z = 0, spread 1):
            mh = S1 / max(k, 1); vh = max(S2 / max(k, 1) - mh^2, 0)
            m_win = S1 / (k + 144); s_win = sqrt((k vh + 144) / (k + 144))
        Vectorized as research chart_ms (differences of cumulative sums); bar by bar as the Pine indicator (LiveWin: a ring
        of 4096 z with the index n mod 4096 and running S1, S2).
    log M (research _nc_logm): m = max(m_win, 0); s = min(max(s_win, s_lo), 1); zc = min(z, m / max(1 - s^2, 1e-12)) when
        s < 1, else z; log M = -(zc - m)^2 / (2 s^2) - ln s + zc^2 / 2 (ratio of the bells N(m, s^2) / N(0, 1))
    display: shown from the window n >= 4096 (R18; first bar 264 + 4096 = 4360 on continuous data); colour = sign of log M
        (orange > 0 — more often than random, blue < 0 — less often, 0 — the floor), the number = exp(|log M|) not above 10
        for both colours (R19; above — drawn at 10, '10+'). An invalid window does not change the state, gets no number.
L, n0, lag and the start of the line are constants of the rule, not settings.
"""
import math

import numpy as np
import pandas as pd

S_LO = 0.75
PEAK_EPS = 1e-12
# session 5 (R18, R19): formula A of SuMo_Class_Live and one ceiling
L_WIN = 4032                # windows in the calibration set
N0 = 144                    # virtual random windows (z = 0, spread 1)
LAG = 64                    # the set ends 64 windows before the current one
SHOW_FROM = L_WIN + LAG     # 4096: the line from the full window (B3)
RING = SHOW_FROM            # ring of z in the stream (indices n - 4096 .. n)
CEIL = 10.0                 # one ceiling for both colours; above — '10+'
# session 4 (R15, R16) — record of the EMA line, not exported
SPAN = 8064
WARM = 288


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


def live_win(z):
    """Formula A over a series of z of consecutive valid windows (rule R18), vectorized as research SuMo_Class_Live
       chart_ms (kind 'win', prior 'null'): the set of the window n is [lo, hi), hi = n - 63, lo = max(0, hi - 4032), its
       sums — differences of cumulative sums. Returns m_win, s_win, k."""
    z = np.asarray(z, dtype=np.float64)
    c1 = np.concatenate([[0.0], np.cumsum(z)])
    c2 = np.concatenate([[0.0], np.cumsum(z * z)])
    i = np.arange(len(z))
    hi = i - LAG + 1
    lo = np.maximum(0, hi - L_WIN)
    k = np.maximum(hi - lo, 0)
    hc = np.maximum(hi, lo)
    s1, s2 = c1[hc] - c1[lo], c2[hc] - c2[lo]
    kk = np.maximum(k, 1)
    mh = s1 / kk
    vh = np.maximum(s2 / kk - mh * mh, 0.0)
    return s1 / (k + N0), np.sqrt((k * vh + N0) / (k + N0)), k


class LiveWin:
    """Formula A window by window, exactly as the Pine indicator (rule R18): a ring of 4096 z (index n mod 4096), running
       S1, S2 and the number n of the last valid window. update() — on a valid window only; the current window gets its
       number but is not in its own set."""

    def __init__(self, q, zk, s_lo=S_LO):
        self.q, self.zk = [float(x) for x in q], [float(x) for x in zk]
        self.s_lo = s_lo
        self.ring = [0.0] * RING
        self.s1 = self.s2 = 0.0
        self.n = -1                                    # number of the last valid window (from 0)

    def step(self, z):
        """Add the window with this z; returns (m_win, s_win, k) of it."""
        self.n += 1
        n = self.n
        if n >= RING:                                  # the window n - 4096 leaves the set
            x = self.ring[n % RING]
            self.s1 -= x
            self.s2 -= x * x
        self.ring[n % RING] = z
        if n >= LAG:                                   # the window n - 64 enters it
            x = self.ring[(n - LAG) % RING]
            self.s1 += x
            self.s2 += x * x
        k = min(max(n - LAG + 1, 0), L_WIN)
        kk = max(k, 1)
        mh = self.s1 / kk
        vh = max(self.s2 / kk - mh * mh, 0.0)
        return self.s1 / (k + N0), math.sqrt((k * vh + N0) / (k + N0)), k

    def update(self, logit):
        z = ruler_z1(logit, self.q, self.zk)
        m_win, s_win, k = self.step(z)
        lm = log_m1(z, m_win, s_win, self.s_lo)
        return {'z': z, 'm_win': m_win, 's_win': s_win, 'k': k, 'log_m': lm, 'n': self.n, 'shown': self.n >= SHOW_FROM}


def line_win(logit, valid, q, zk, s_lo=S_LO):
    """Vectorized line M of session 5 over the windows of a chart in time order (rules R18, R19). logit — per window
       (bar), valid — window valid (rule R13); invalid windows get NaN and do not move the state. Returns dict of
       per-window arrays: z, m_win, s_win, log_m, k and n (-1 at invalid windows), shown (n >= 4096)."""
    logit = np.asarray(logit, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    out = {k: np.full(len(logit), np.nan) for k in ('z', 'm_win', 's_win', 'log_m')}
    out['n'] = np.full(len(logit), -1, dtype=np.int64)
    out['k'] = np.full(len(logit), -1, dtype=np.int64)
    out['shown'] = np.zeros(len(logit), dtype=bool)
    idx = np.where(valid)[0]
    if len(idx):
        z = ruler_z(logit[idx], q, zk)
        m, s, k = live_win(z)
        out['z'][idx], out['m_win'][idx], out['s_win'][idx], out['k'][idx] = z, m, s, k
        out['log_m'][idx] = log_m(z, m, s, s_lo)
        out['n'][idx] = np.arange(len(idx))
        out['shown'][idx] = np.arange(len(idx)) >= SHOW_FROM
    return out


def live(z, span=SPAN):
    """Session 4 (record, not exported). m_ema, s_ema over a series of z of consecutive valid windows (research _nc_live: pandas ewm, adjust=True)."""
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
    """Session 4 (record, not exported). The live part bar by bar (state of the Pine indicator: n1, n2, d, n). update() — on a valid window only."""

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
    """Session 4 (record, not exported). Vectorized line M over the windows of a chart in time order. logit — per window (bar), valid — window valid
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


def display(lm, ceil_orange=CEIL, ceil_blue=CEIL):
    """Number on the line and its colour for log M values: (value, colour, capped). colour +1 orange, -1 blue, 0 floor;
       value = exp(|log M|) not above the ceiling of the colour (session 5: 10 for both, R19); capped — above the ceiling
       ('10+')."""
    lm = np.asarray(lm, dtype=np.float64)
    col = np.sign(lm)
    m = np.exp(np.abs(lm))
    cap = np.where(col > 0, ceil_orange, np.where(col < 0, ceil_blue, np.inf))
    return np.minimum(m, cap), col, m > cap
