"""NT3_PRV_ch48 ('F3 without time'): the streamable CNN of F3_g with 48 channels, head 161 -> 96 -> 1 and 17 side inputs
'nt_PRV' — the window divisor and the history of the 200 bars before the window, in bars (no EMA, no clock).

Window: bars s .. s + 63, t = s + 63 is the current closed bar. Per-bar inputs (norm, r, steps, g), the window divisor D,
the orientation sign (rule R1) and the conv streams are those of F3_g (mri.features, mri.batch, mri.legacy.stream).
Side inputs (research SIDE_COLS['nt_PRV'] = 28, 29, 37-44, 77, 78, 79, 86-89; research side_feats / tok_feats / nt_segment):
     0  log D
     1  log(max(mean |r| over the 64 bars of the window, floor))
        history of the window start s, bars s-200 .. s-1: rmed = median(high - low) (= norm of bar s), rmean = mean(high - low),
        bmed = max(median |close - open|, 0.05 * rmed)
  2-9   P: path before the window in 8 chunks over bars s-200 .. s-2, oldest first: 71, 64, 32, 16, 8, 4, 2, 2 bars;
        chunk of n bars: (close of its last bar - open of its first) / (sqrt(n) * rmed), clipped to +-20, times sign
    10  R1 = 2 * (close[s-2] - lmin) / (hmax - lmin) - 1 (0 if hmax = lmin), times sign; hmax / lmin over bars s-200 .. s-2
    11  R2 = log(max(hmax - lmin, 1e-12) / (sqrt(199) * rmed)), clipped to +-5, no sign
    12  R3 = (close[s-2] - mean close over bars s-200 .. s-2) / (sqrt(199) * rmed), clipped to +-20, times sign
 13-16  V, no sign, each log(max(v, 1e-3)) clipped to +-5: v = mean(high - low) over s-16 .. s-1 / rmean; the same over
        s-64 .. s-1; rmean / rmed; bmed / rmed
NaN -> 0 (research nan_to_num, after the clips). The bar s-1 does not enter the signed inputs (P, R1, R3), only the scales
and V. Nothing deeper than s-200 is read; the history runs over consecutive bars (time and grid gaps are not looked at).

PB_s64h24_d1 (session 4, mri.pb) is this network with 64 channels, separable layers 2..4 (research v6.3 cnn_sep: per
channel kernel 3 with the same dilation, then 1 x 1 channel mixing, then ReLU) and the head 209 -> 24 -> 1; channels, head
and the separable layers are taken from the weights (Params.pw), the inputs are the same.

nt_segment is a copy of research/sumo_class_train.py nt_segment (that file imports torch and is not needed at run time);
tests/test_nt3.py compares the copy with the original. Here it is float64 (research casts to float32 at the end).
"""
import math
from collections import deque

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import GAP_MS, L, N_NORM, TIE_EPS
from .batch import conv_inputs, conv_stream, head
from .params import DILATIONS

MODEL_ID = 'NT3_PRV_ch48'
PB_MODEL_ID = 'PB_s64h24_d1'   # the same network with 64 channels, separable layers 2..4, head 209 -> 24 -> 1 (mri.pb)
NT_HIST = 200
NT_SCALES = ('rmed', 'rmean', 'bmed', 'bmean')
NT_GEO = (71, 64, 32, 16, 8, 4, 2, 2)
NT_UNI = (24, 25, 25, 25, 25, 25, 25, 25)
NT_SHORT = (16, 64)
NT_BODY_FLOOR = 0.05
NT_CLIP_Z, NT_CLIP_LOG = 20.0, 5.0
SIDE_COLS = (28, 29, 37, 38, 39, 40, 41, 42, 43, 44, 77, 78, 79, 86, 87, 88, 89)
SIDE_NAMES = ('log_D', 'log_mean_abs_r', 'path_71', 'path_64', 'path_32', 'path_16', 'path_8', 'path_4', 'path_2a',
              'path_2b', 'range_pos', 'range_width', 'close_dev', 'vol_16', 'vol_64', 'range_mean_med', 'body_med')
SIDE_SIGNED = (False, False) + (True,) * 8 + (True, False, True) + (False,) * 4
HIST_ROWS = tuple(range(8)) + (40, 41, 42) + (49, 50, 51, 52)    # rows of nt_segment used by nt_PRV (P_rmed, R_rmed, V)
N_SIDE = len(SIDE_COLS)
FIRST_START = NT_HIST          # a window needs >= 200 bars before its first bar


def nt_segment(o, h, l, c, n_pre=NT_HIST, with_scales=False):
    """Copy of research nt_segment (v6.0), without the final float32 cast: history features (53, windows) for windows
       starting at bars n_pre .. len - 1 of the arrays (see the module docstring for the 15 rows that nt_PRV uses)."""
    nh, nh2 = NT_HIST, NT_HIST - 1
    m = len(c) - n_pre
    idx = np.arange(n_pre, len(c))
    rng_, body = h - l, np.abs(c - o)

    def roll(x, n, how, lag):
        v = getattr(pd.Series(x).rolling(n, min_periods=n), how)().to_numpy()
        return v[idx - lag]

    with np.errstate(invalid='ignore', divide='ignore'):
        rmed, rmean = roll(rng_, nh, 'median', 1), roll(rng_, nh, 'mean', 1)
        sc = {'rmed': rmed, 'rmean': rmean,
              'bmed': np.maximum(roll(body, nh, 'median', 1), NT_BODY_FLOOR * rmed),
              'bmean': np.maximum(roll(body, nh, 'mean', 1), NT_BODY_FLOOR * rmed)}
        hmax, lmin, cma = roll(h, nh2, 'max', 2), roll(l, nh2, 'min', 2), roll(c, nh2, 'mean', 2)
        c2 = c[idx - 2]

        def blocks(lens):
            res, end = [], idx - 2
            for n in lens[::-1]:
                start = end - n + 1
                res.append((c[end] - o[start]) / math.sqrt(n))
                end = start - 1
            return res[::-1]

        geo, out = blocks(NT_GEO), []
        for x in NT_SCALES:
            out += [np.clip(v / sc[x], -NT_CLIP_Z, NT_CLIP_Z) for v in geo]
        out += [np.clip(v / rmed, -NT_CLIP_Z, NT_CLIP_Z) for v in blocks(NT_UNI)]
        w = hmax - lmin
        out.append(np.where(w > 0, 2.0 * (c2 - lmin) / np.where(w > 0, w, 1.0) - 1.0, 0.0))
        for x in NT_SCALES:
            out.append(np.clip(np.log(np.maximum(w, 1e-12) / (math.sqrt(nh2) * sc[x])), -NT_CLIP_LOG, NT_CLIP_LOG))
            out.append(np.clip((c2 - cma) / (math.sqrt(nh2) * sc[x]), -NT_CLIP_Z, NT_CLIP_Z))
        r16, r64 = roll(rng_, NT_SHORT[0], 'mean', 1), roll(rng_, NT_SHORT[1], 'mean', 1)
        for v in (r16 / rmean, r64 / rmean, rmean / rmed, sc['bmed'] / rmed):
            out.append(np.clip(np.log(np.maximum(v, 1e-3)), -NT_CLIP_LOG, NT_CLIP_LOG))
    f = np.nan_to_num(np.stack(out), nan=0.0, posinf=0.0, neginf=0.0)
    if m <= 0:
        f = np.zeros((len(out), 0))
    return (f, sc) if with_scales else f


def history_features(bars):
    """(n, 15) unsigned history inputs of nt_PRV for every bar as a window start s; NaN where s < 200 (no full history).
       bars: DataFrame with open, high, low, close (consecutive bars)."""
    o, h, l, c = (bars[k].to_numpy(np.float64) for k in ('open', 'high', 'low', 'close'))
    out = np.full((len(c), len(HIST_ROWS)), np.nan)
    if len(c) > NT_HIST:
        out[NT_HIST:] = nt_segment(o, h, l, c)[list(HIST_ROWS)].T
    return out


def side_raw(f, hist, T, D, sgn, floor):
    """17 raw side inputs (after the sign and clips, before standardization) of the windows ending at bars T."""
    s = T - (L - 1)
    if (s < FIRST_START).any():
        raise ValueError("a window without 200 bars of history")
    rw = sliding_window_view(np.nan_to_num(f['r']), L)[s]
    mean_abs = np.abs(rw).mean(axis=1)
    sign = np.where(np.array(SIDE_SIGNED[2:])[None, :], sgn[:, None], 1.0)
    return np.concatenate([np.log(D)[:, None], np.log(np.maximum(mean_abs, floor))[:, None], hist[s] * sign], axis=1)


def valid_windows(f):
    """Window end bars T whose 64 bars are all valid and whose start has >= 200 bars before it (rule R13)."""
    n = f['n']
    okw = np.zeros(n, dtype=bool)
    if n >= L:
        cnt = np.convolve(f['ok'].astype(np.int64), np.ones(L, dtype=np.int64), mode='valid')
        okw[L - 1:] = cnt == L
    okw[:FIRST_START + L - 1] = False
    return okw


def infer(f, hist, p, floor, eps=TIE_EPS, T=None, layers=None):
    """Vectorized logit of every valid window (or of the windows ending at bars T). f = mri.features.bar_features(bars),
       hist = history_features(bars). Same arithmetic as the stream (StreamingNT3)."""
    okw = valid_windows(f)
    if T is None:
        T = np.where(okw)[0]
    T = np.asarray(T, dtype=np.int64)
    if not okw[T].all():
        raise ValueError("some requested windows are not valid")
    rw = sliding_window_view(np.nan_to_num(f['r']), L)[T - (L - 1)]
    D = np.maximum(np.median(np.abs(rw), axis=1), floor)
    x_end = rw @ np.arange(L, dtype=np.float64) / L
    sgn = np.where(x_end < -eps * D, -1.0, 1.0)
    if layers is None:
        up, um = conv_inputs(f, p)
        layers = (conv_stream(up, p), conv_stream(um, p))
    P = L - 1 - sum(DILATIONS) * 2                                   # 33 positions of layer 4
    wp = sliding_window_view(layers[0][3], P, axis=0)[T - (P - 1)]
    wm = sliding_window_view(layers[1][3], P, axis=0)[T - (P - 1)]
    W = np.where((sgn > 0)[:, None, None], wp, wm)
    pooled = np.concatenate([W.mean(axis=2), W.max(axis=2), W[:, :, -1]], axis=1) / D[:, None]
    s_raw = side_raw(f, hist, T, D, sgn, floor)
    s_std = (s_raw - p.side_mu) / p.side_sd
    logit = head(p, np.concatenate([pooled, s_std], axis=1))
    return {'T': T, 'D': D, 'x_end': x_end, 'sgn': sgn, 'side_raw': s_raw, 'logit': logit, 'pooled': pooled,
            'layers': layers}


# ---------------------------------------------------------------------------------------------------------------------
# bar by bar


def _clip(x, lim):
    return min(max(x, -lim), lim)


def _log_clip(v, lim=NT_CLIP_LOG):
    return _clip(math.log(max(v, 1e-3)), lim)


def history_now(o, h, l, c):
    """Unsigned history inputs (15) of a window starting at the current bar s, from the 200 bars before it (oldest
       first: index j is bar s - 200 + j). Pure-Python arithmetic, the way the Pine indicator computes it."""
    rng = h - l
    body = np.abs(c - o)
    rmed = float(np.median(rng))
    rmean = float(np.mean(rng))
    bmed = max(float(np.median(body)), NT_BODY_FLOOR * rmed)
    out = []
    end = NT_HIST - 2                                       # bar s - 2
    chunks = []
    for n in NT_GEO[::-1]:                                  # freshest chunk first
        start = end - n + 1
        chunks.append((c[end] - o[start]) / math.sqrt(n) / rmed)
        end = start - 1
    out += [_clip(v, NT_CLIP_Z) for v in chunks[::-1]]
    hmax, lmin = float(np.max(h[:NT_HIST - 1])), float(np.min(l[:NT_HIST - 1]))
    cma = float(np.mean(c[:NT_HIST - 1]))
    c2 = float(c[NT_HIST - 2])
    w = hmax - lmin
    out.append(2.0 * (c2 - lmin) / w - 1.0 if w > 0 else 0.0)
    sq = math.sqrt(NT_HIST - 1)
    out.append(_clip(math.log(max(w, 1e-12) / (sq * rmed)), NT_CLIP_LOG))
    out.append(_clip((c2 - cma) / (sq * rmed), NT_CLIP_Z))
    r16, r64 = float(np.mean(rng[-NT_SHORT[0]:])), float(np.mean(rng[-NT_SHORT[1]:]))
    out += [_log_clip(r16 / rmean), _log_clip(r64 / rmean), _log_clip(rmean / rmed), _log_clip(bmed / rmed)]
    return np.array([0.0 if not math.isfinite(v) else v for v in out])


class StreamingNT3:
    """Bar-by-bar NT3 inference with ring buffers (the computation of the Pine indicator, see mri.legacy.stream for F3_g)."""

    def __init__(self, params, floor, eps=TIE_EPS, platt=None):
        self.p, self.floor, self.eps, self.platt = params, floor, eps, platt
        self.ch = params.conv[0].shape[0]
        self.ohlc = deque(maxlen=NT_HIST)                   # bars before the current one: (o, h, l, c)
        self.prev_t = self.prev_c = None
        self.prev_d = (math.nan, math.nan)
        self.win_r = deque([0.0] * L, maxlen=L)
        self.win_ok = deque([False] * L, maxlen=L)
        self.hist = deque([None] * L, maxlen=L)            # history inputs of the window starting at each of the last 64 bars
        depth = (3, 5, 9, 17, 33)
        self.rings = [[deque([np.zeros(4 if j == 0 else self.ch)] * depth[j], maxlen=depth[j]) for j in range(5)]
                      for _ in range(2)]
        self.bar = -1
        self.first_r = None

    def _conv_step(self, s, u):
        rings = self.rings[s]
        rings[0].append(u)
        for j, (w, d) in enumerate(zip(self.p.conv, DILATIONS)):
            src = rings[j]
            if self.p.pw is not None and j > 0:        # separable (PB): per-channel taps, then 1 x 1 channel mixing
                y = self.p.pw[j - 1] @ (w[:, 0, 0] * src[-1 - 2 * d] + w[:, 0, 1] * src[-1 - d] + w[:, 0, 2] * src[-1])
            else:
                y = w[:, :, 0] @ src[-1 - 2 * d] + w[:, :, 1] @ src[-1 - d] + w[:, :, 2] @ src[-1]
            rings[j + 1].append(np.maximum(y, 0.0))

    def update(self, t, o, h, l, c):
        """One confirmed bar (open time ms, OHLC). Returns a dict for this bar; 'logit' is NaN if no valid window."""
        self.bar += 1
        if len(self.ohlc) == NT_HIST:
            a = np.array(self.ohlc)
            self.hist.append(history_now(a[:, 0], a[:, 1], a[:, 2], a[:, 3]))
            norm = float(np.median(a[:, 1] - a[:, 2]))
        else:
            self.hist.append(None)
            norm = math.nan
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
        self.ohlc.append((o, h, l, c))
        self.prev_t, self.prev_c, self.prev_d = t, c, d_now
        self.win_r.append(r if ok else 0.0)
        self.win_ok.append(ok)
        if ok and self.first_r is None:
            self.first_r = self.bar
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
        hist = self.hist[0]                                  # history of the window start s = bar - 63
        if not all(self.win_ok) or hist is None:
            return out
        rw = np.array(self.win_r)
        D = max(float(np.median(np.abs(rw))), self.floor)
        x_end = float(rw @ np.arange(L, dtype=np.float64)) / L
        sgn = -1.0 if x_end < -self.eps * D else 1.0
        ring4 = np.array(self.rings[0 if sgn > 0 else 1][4])
        pooled = np.concatenate([ring4.mean(axis=0), ring4.max(axis=0), ring4[-1]]) / D
        side = np.concatenate([[math.log(D), math.log(max(float(np.mean(np.abs(rw))), self.floor))],
                               hist * np.where(SIDE_SIGNED[2:], sgn, 1.0)])
        s_std = (side - p.side_mu) / p.side_sd
        z = np.maximum(p.head0_w @ np.concatenate([pooled, s_std]) + p.head0_b, 0.0)
        logit = float(p.head3_w[0] @ z + p.head3_b[0])
        out.update({'D': D, 'x_end': x_end, 'sgn': sgn, 'side_raw': side, 'logit': logit})
        if self.platt is not None:
            ll = self.platt[0] * logit + self.platt[1]
            out.update({'log_lift': ll, 'lift': math.exp(ll), 'M': math.exp(abs(ll))})
        return out


def run(bars, params, floor, **kw):
    """Run the NT3 stream over a DataFrame of bars (time, open, high, low, close); returns a list of per-bar dicts."""
    m = StreamingNT3(params, floor, **kw)
    cols = [bars[k].to_numpy() for k in ('time', 'open', 'high', 'low', 'close')]
    return [m.update(int(t), float(o), float(h), float(l), float(c)) for t, o, h, l, c in zip(*cols)]


def random_params(seed=0, scale=0.3, ch=48, hidden=96, sep=False):
    """Random NT3-shaped weights (Pine limits prototype, tests); sep — separable layers 2..4 (PB shape)."""
    from .params import Params
    g = np.random.default_rng(seed)
    pw = None
    if sep:
        conv = [g.normal(0, scale, (ch, 4, 3))] + [g.normal(0, 0.6, (ch, 1, 3)) for _ in range(3)]
        pw = [g.normal(0, 1.0 / math.sqrt(ch), (ch, ch)) for _ in range(3)]
    else:
        conv = [g.normal(0, scale, (ch, 4, 3))] + [g.normal(0, scale / 2.5, (ch, ch, 3)) for _ in range(3)]
    return Params(conv=conv, pw=pw, head0_w=g.normal(0, 0.15, (hidden, 3 * ch + N_SIDE)), head0_b=g.normal(0, 0.2, hidden),
                  head3_w=g.normal(0, 0.2, (1, hidden)), head3_b=g.normal(0, 0.1, 1),
                  s_sd=1.6, lvl_mu=0.0, lvl_sd=3.0, side_mu=g.normal(0, 0.1, N_SIDE),
                  side_sd=np.abs(g.normal(0.7, 0.2, N_SIDE)) + 0.1, meta={'random_seed': seed, 'model': PB_MODEL_ID if sep else MODEL_ID})
