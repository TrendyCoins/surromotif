"""The live part of the line M of session 4 (rules R15 / R16, research v6.3 'nocal'), kept as the record of that line: EMA
of z and z^2 over the valid windows of the chart from its start (pandas ewm(span, adjust=True)), span 8064 bars, the line
from the window 288. The indicator uses formula A since session 5 (mri.linem); this is not exported.

    live(z, span)            m_ema, s_ema over a series of z (vectorized)
    LiveM                    the same window by window (state n1, n2, d, n of the session-4 Pine indicator)
    line(logit, valid, ...)  vectorized line M of session 4 over the windows of a chart
The ruler and log M are those of mri.linem.
"""
import math

import numpy as np
import pandas as pd

from ..linem import S_LO, log_m, log_m1, ruler_z, ruler_z1

SPAN = 8064
WARM = 288


def live(z, span=SPAN):
    """Session 4 (record, not exported). m_ema, s_ema over a series of z of consecutive valid windows (research _nc_live: pandas ewm, adjust=True)."""
    z = np.asarray(z, dtype=np.float64)
    m = pd.Series(z).ewm(span=span, adjust=True).mean().to_numpy()
    v = pd.Series(z * z).ewm(span=span, adjust=True).mean().to_numpy()
    return m, np.sqrt(np.maximum(v - m * m, 0.0))


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
