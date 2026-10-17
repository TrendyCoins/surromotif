"""PB_s64h24_d1 — the indicator model since session 4 (rules R14-R16): NT3_PRV_ch48 with 64 channels, separable layers 2..4
(research v6.3 cnn_sep) and the head 209 -> 24 -> 1; inputs, side inputs 'nt_PRV', tie rule R1, floors R6 and the model
warm-up R13 are those of NT3 (mri.nt3). The line M is not calibrated per asset: the ruler of the model and the live part
of the chart (mri.linem).

    infer(f, hist, p, floor)          vectorized logit of every valid window (mri.nt3.infer: the separable layers come
                                      from Params.pw)
    chart(bars, p, floor, q, zk)      vectorized logit + line M over a chart (windows in time order)
    StreamingPB                       bar by bar, exactly as the Pine indicator: the NT3 stream + the live part (state
                                      n1, n2, d, n changes on a valid window only)
"""
import math

import numpy as np

from . import linem, nt3
from .features import bar_features

MODEL_ID = nt3.PB_MODEL_ID
CH, HIDDEN = 64, 24


def infer(f, hist, p, floor, **kw):
    return nt3.infer(f, hist, p, floor, **kw)


def chart(bars, p, floor, q, zk, span=linem.SPAN, warm=linem.WARM, s_lo=linem.S_LO, f=None, hist=None):
    """Per-bar arrays over a chart: logit (NaN — no valid window) and the line M (mri.linem.line) with its live part
       from the first valid window of the bars."""
    f = bar_features(bars) if f is None else f
    hist = nt3.history_features(bars) if hist is None else hist
    n = f['n']
    res = nt3.infer(f, hist, p, floor)
    logit = np.full(n, np.nan)
    logit[res['T']] = res['logit']
    valid = np.zeros(n, dtype=bool)
    valid[res['T']] = True
    out = linem.line(logit, valid, q, zk, span, warm, s_lo)
    out.update({'logit': logit, 'valid': valid, 'res': res, 'f': f, 'hist': hist})
    return out


class StreamingPB(nt3.StreamingNT3):
    """Bar-by-bar PB inference (the NT3 stream; separable layers from the weights) and the line M."""

    def __init__(self, params, floor, q, zk, span=linem.SPAN, warm=linem.WARM, s_lo=linem.S_LO, ceil=None, **kw):
        if params.pw is None:
            raise ValueError("PB needs separable layers (Params.pw)")
        super().__init__(params, floor, **kw)
        self.live = linem.LiveM(q, zk, span, warm, s_lo)
        self.ceil = ceil                                   # (orange, blue) or None

    def update(self, t, o, h, l, c):
        out = super().update(t, o, h, l, c)
        out.update({'z': math.nan, 'm_ema': math.nan, 's_ema': math.nan, 'log_m': math.nan, 'n': self.live.n,
                    'shown': False})
        if math.isfinite(out['logit']):
            out.update(self.live.update(out['logit']))
            if self.ceil is not None:
                v, col, cap = linem.display(out['log_m'], *self.ceil)
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
