"""PB_s64h24_d1 — the indicator model since session 4 (rule R14): NT3_PRV_ch48 with 64 channels, separable layers 2..4
(research v6.3 cnn_sep) and the head 209 -> 24 -> 1; inputs, side inputs 'nt_PRV', tie rule R1 and the model warm-up R13
are those of NT3 (mri.nt3). Session 5: one divisor floor for every symbol (R17, FLOOR = float32(0.2)); the line M — the
ruler of the model and the live part of the chart by formula A (R18, R19; mri.linem).

    infer(f, hist, p, floor)          vectorized logit of every valid window (mri.nt3.infer: the separable layers come
                                      from Params.pw)
    chart(bars, p, floor, q, zk)      vectorized logit + line M over a chart (windows in time order)
    StreamingPB                       bar by bar, exactly as the Pine indicator: the NT3 stream + the live part (ring of
                                      4096 z, S1, S2, n — change on a valid window only)
"""
import math

import numpy as np

from . import linem, nt3
from .features import bar_features

MODEL_ID = nt3.PB_MODEL_ID
CH, HIDDEN = 64, 24
FLOOR = float(np.float32(0.2))          # rule R17: 0.20000000298023224 for every symbol


def infer(f, hist, p, floor, **kw):
    return nt3.infer(f, hist, p, floor, **kw)


def chart(bars, p, floor, q, zk, s_lo=linem.S_LO, f=None, hist=None):
    """Per-bar arrays over a chart: logit (NaN — no valid window) and the line M (mri.linem.line_win) with its live part
       and window numbers from the first valid window of the bars."""
    f = bar_features(bars) if f is None else f
    hist = nt3.history_features(bars) if hist is None else hist
    n = f['n']
    res = nt3.infer(f, hist, p, floor)
    logit = np.full(n, np.nan)
    logit[res['T']] = res['logit']
    valid = np.zeros(n, dtype=bool)
    valid[res['T']] = True
    out = linem.line_win(logit, valid, q, zk, s_lo)
    out.update({'logit': logit, 'valid': valid, 'res': res, 'f': f, 'hist': hist})
    return out


class StreamingPB(nt3.StreamingNT3):
    """Bar-by-bar PB inference (the NT3 stream; separable layers from the weights) and the line M."""

    def __init__(self, params, floor, q, zk, s_lo=linem.S_LO, ceil=linem.CEIL, **kw):
        if params.pw is None:
            raise ValueError("PB needs separable layers (Params.pw)")
        super().__init__(params, floor, **kw)
        self.live = linem.LiveWin(q, zk, s_lo)
        self.ceil = ceil                                   # one ceiling for both colours (R19) or None

    def update(self, t, o, h, l, c):
        out = super().update(t, o, h, l, c)
        out.update({'z': math.nan, 'm_win': math.nan, 's_win': math.nan, 'k': -1, 'log_m': math.nan, 'n': self.live.n,
                    'shown': False})
        if math.isfinite(out['logit']):
            out.update(self.live.update(out['logit']))
            if self.ceil is not None:
                v, col, cap = linem.display(out['log_m'], self.ceil, self.ceil)
                out.update({'M': float(v), 'color': int(col), 'capped': bool(cap)})
        return out


def run(bars, params, floor, q, zk, **kw):
    """Run the PB stream over a DataFrame of bars (time, open, high, low, close); returns a list of per-bar dicts."""
    m = StreamingPB(params, floor, q, zk, **kw)
    cols = [bars[k].to_numpy() for k in ('time', 'open', 'high', 'low', 'close')]
    return [m.update(int(t), float(o), float(h), float(l), float(c)) for t, o, h, l, c in zip(*cols)]


def random_params(seed=0):
    """Random PB-shaped weights (tests, Pine size prototype)."""
    return nt3.random_params(seed, ch=CH, hidden=HIDDEN, sep=True)
