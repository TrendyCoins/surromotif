"""Bar-by-bar inference with ring buffers — the computation the Pine indicator does on each confirmed bar.
Only past and current bars are used; state changes once per bar. Same arithmetic as mri.batch (tests compare them).

State per bar (Pine names in comments of pine/):
    ranges   - last 200 bar ranges (norm of the next bar = their median)
    prev     - previous close, d of HL/2 and HLC/3 of the previous bar (with its own norm)
    win      - last 64 bars: r and ok flag (window divisor, orientation, validity)
    ema7, ema1 - EMA |r| over ok bars (pandas adjust=False from the first value, or normalized - 'adjust')
    cs       - history of the running sum of r (moves before the window), ema1 history (norm of the moves)
    rings    - conv inputs and outputs of layers 1..4 for two streams: depths 3, 5, 9, 17, 33
"""
import math
from collections import deque

import numpy as np

from . import BARS_DAY, CTX_BLOCK, CTX_NBLK, GAP_MS, L, N_NORM, SPAN_CTX, SPAN_SIDE, TIE_EPS
from .params import DILATIONS

DEPTH = (3, 5, 9, 17, 33)     # ring depth of the input of layer 1, outputs of layers 1, 2, 3 and of layer 4 (pooling)
HIST = L + 1 + 7 * BARS_DAY   # bars of cs / ema1 history needed for the moves before the window


class _Ema:
    def __init__(self, span, adjust):
        self.a, self.adjust = 2.0 / (span + 1.0), adjust
        self.s = self.w = None

    def update(self, x):
        if x is None:
            return
        if self.s is None:
            self.s, self.w = x, 1.0
        elif self.adjust:
            self.s, self.w = x + (1.0 - self.a) * self.s, 1.0 + (1.0 - self.a) * self.w
        else:
            self.s = self.a * x + (1.0 - self.a) * self.s

    @property
    def value(self):
        v = math.nan if self.s is None else (self.s / self.w if self.adjust else self.s)
        return v if v > 0 else math.nan      # research: EMA <= 0 -> NaN (no backfill here: that would look ahead)


class StreamingMRI:
    def __init__(self, params, floor, eps=TIE_EPS, ema_adjust=False, platt=None):
        self.p, self.floor, self.eps, self.platt = params, floor, eps, platt
        self.ranges = deque(maxlen=N_NORM)
        self.prev_t = self.prev_c = None
        self.prev_d = (math.nan, math.nan)
        self.win_r = deque([0.0] * L, maxlen=L)
        self.win_ok = deque([False] * L, maxlen=L)
        self.ema7, self.ema1 = _Ema(SPAN_SIDE, ema_adjust), _Ema(SPAN_CTX, ema_adjust)
        self.cs_hist = deque([0.0], maxlen=HIST + 2)     # cs_hist[-1] = sum of r over ok bars up to this bar
        self.e1_hist = deque(maxlen=HIST + 2)
        self.rings = [[deque([np.zeros(4 if j == 0 else 32)] * DEPTH[j], maxlen=DEPTH[j]) for j in range(5)]
                      for _ in range(2)]
        self.bar = -1
        self.first_r = None

    def _norm_inputs(self, t, h, l, c):
        norm = float(np.median(self.ranges)) if len(self.ranges) == N_NORM else math.nan
        good = math.isfinite(norm) and norm > 0
        gap = self.prev_t is not None and (t - self.prev_t) > GAP_MS
        bad = (not good) or gap or self.prev_c is None
        r = s1 = s2 = g = math.nan
        d_now = (math.nan, math.nan)
        if good:
            d_now = (((h + l) / 2.0 - c) / norm, ((h + l + c) / 3.0 - c) / norm)
        if not bad:
            r = (c - self.prev_c) / norm
            s1 = ((h + l) / 2.0 - self.prev_c) / norm - self.prev_d[0]
            s2 = ((h + l + c) / 3.0 - self.prev_c) / norm - self.prev_d[1]
            g = (h - l) / norm
        ok = all(math.isfinite(v) for v in (r, s1, s2, g))
        return norm, ok, r, s1, s2, g, d_now

    def _conv_step(self, s, u):
        rings = self.rings[s]
        rings[0].append(u)
        for j, (w, d) in enumerate(zip(self.p.conv, DILATIONS)):
            src = rings[j]
            # tap k sees the input (2 - k) * d bars back; src[-1] is the current bar
            y = w[:, :, 0] @ src[-1 - 2 * d] + w[:, :, 1] @ src[-1 - d] + w[:, :, 2] @ src[-1]
            rings[j + 1].append(np.maximum(y, 0.0))

    def update(self, t, o, h, l, c):
        """One confirmed bar (open time ms, OHLC). Returns a dict for this bar; 'logit' is NaN if no valid window."""
        self.bar += 1
        norm, ok, r, s1, s2, g, d_now = self._norm_inputs(t, h, l, c)
        self.ranges.append(h - l)
        self.prev_t, self.prev_c, self.prev_d = t, c, d_now
        self.win_r.append(r if ok else 0.0)
        self.win_ok.append(ok)
        if ok and self.first_r is None:
            self.first_r = self.bar
        self.ema7.update(abs(r) if ok else None)
        self.ema1.update(abs(r) if ok else None)
        self.cs_hist.append(self.cs_hist[-1] + (r if ok else 0.0))
        self.e1_hist.append(self.ema1.value)
        p = self.p
        if ok:
            path = np.array([r, s1, s2]) / p.s_sd
            lvl = (g - p.lvl_mu) / p.lvl_sd
        else:
            path, lvl = np.zeros(3), 0.0
        self._conv_step(0, np.append(path, lvl))
        self._conv_step(1, np.append(-path, lvl))
        out = {'bar': self.bar, 'time': t, 'norm': norm, 'ok': ok, 'r': r, 'step_hl2': s1, 'step_hlc3': s2, 'g': g,
               'first_r': self.first_r, 'logit': math.nan}
        if not all(self.win_ok):
            return out
        rw = np.array(self.win_r)
        D = max(float(np.median(np.abs(rw))), self.floor)
        x_end = float(rw @ np.arange(L, dtype=np.float64)) / L
        sgn = -1.0 if x_end < -self.eps * D else 1.0
        ring4 = np.array(self.rings[0 if sgn > 0 else 1][4])          # (33, 32), oldest first
        pooled = np.concatenate([ring4.mean(axis=0), ring4.max(axis=0), ring4[-1]]) / D
        side = self._side(t, D, sgn)
        s_std = (side - p.side_mu) / p.side_sd
        z = np.maximum(p.head0_w @ np.concatenate([pooled, s_std]) + p.head0_b, 0.0)
        logit = float(p.head3_w[0] @ z + p.head3_b[0])
        out.update({'D': D, 'x_end': x_end, 'sgn': sgn, 'side_raw': side, 'logit': logit})
        if self.platt is not None:
            ll = self.platt[0] * logit + self.platt[1]
            out.update({'log_lift': ll, 'lift': math.exp(ll), 'M': math.exp(abs(ll))})
        return out

    def _side(self, t, D, sgn):
        # cs(j) = sum of r over bars < j (j = bar index); bar of the window start s = bar - 63, a1 = s - 1
        cur = self.bar
        a1 = cur - L

        def cs(j):          # exclusive prefix sum up to bar j
            k = cur - j + 1
            return self.cs_hist[-1 - k] if k < len(self.cs_hist) else 0.0
        norm1 = self.e1_hist[-1 - L] if len(self.e1_hist) > L else math.nan

        def move(lo, hi):
            lo, hi = max(lo, 0), max(hi, 0)
            return (cs(hi) - cs(lo)) / (math.sqrt(max(hi - lo, 1)) * norm1) * sgn
        minutes = (t // 60_000) % 1440
        hr = minutes / 60.0 * (2 * math.pi / 24.0)
        dw = (((t // 86_400_000) + 3) % 7) * (2 * math.pi / 7.0)
        e7 = self.ema7.value
        v = [math.log(D) - (math.log(e7) if e7 > 0 else math.nan), math.sin(hr), math.cos(hr), math.sin(dw), math.cos(dw),
             move(a1 - BARS_DAY, a1), move(a1 - 3 * BARS_DAY, a1), move(a1 - 7 * BARS_DAY, a1)]
        for k in range(CTX_NBLK - 1, -1, -1):
            v.append(move(a1 - CTX_BLOCK * (k + 1), a1 - CTX_BLOCK * k))
        return np.array(v)


def run(bars, params, floor, **kw):
    """Run the stream over a DataFrame of bars; returns a list of per-bar dicts."""
    m = StreamingMRI(params, floor, **kw)
    cols = [bars[k].to_numpy() for k in ('time', 'open', 'high', 'low', 'close')]
    return [m.update(int(t), float(o), float(h), float(l), float(c)) for t, o, h, l, c in zip(*cols)]
