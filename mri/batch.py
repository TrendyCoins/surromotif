"""Vectorized inference over a whole bar series — the same arithmetic as the bar-by-bar stream (mri/stream.py),
written over arrays for speed (validation sets, calibration, golden file cross-check).

Why streaming is exact (research CNNBase, streamable): convolutions are 'valid', without bias, with ReLU, so the net
is positively homogeneous: conv(u / D) = conv(u) / D for D > 0. The window divisor D and the orientation sign are
per window, the per-bar inputs are not, so each bar's conv outputs are computed once, for two streams
(u+ = [r, step_hl2, step_hlc3] / s_sd with g / lvl_sd, and u- with the path steps negated, the level kept), and the
window takes the stream of its orientation and divides the pooled features by D.

Conv alignment (torch Conv1d, valid): output at bar t of a layer with dilation d uses inputs at t-2d, t-d, t with
kernel taps 0, 1, 2. Layer 4 outputs at bars T-32..T (33 positions) see exactly the bars T-62..T (the first bar of the
window is lost by the steps of the shape, as in research).
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import BARS_DAY, CTX_BLOCK, CTX_NBLK, L, TIE_EPS
from .params import DILATIONS


def conv_inputs(f, p):
    """Per-bar inputs of the two streams, (n, 4) each; invalid bars -> 0 (windows over them are invalid anyway)."""
    path = np.stack([f['r'], f['step_hl2'], f['step_hlc3']], axis=1) / p.s_sd
    lvl = (f['g'][:, None] - p.lvl_mu) / p.lvl_sd
    up = np.nan_to_num(np.concatenate([path, lvl], axis=1))
    um = np.nan_to_num(np.concatenate([-path, lvl], axis=1))
    return up, um


def conv_stream(u, p):
    """Layer outputs over all bars for one stream: list of (n, 32) arrays, layers 1..4."""
    outs, x = [], u
    for w, d in zip(p.conv, DILATIONS):
        n = len(x)
        y = np.zeros((n, w.shape[0]))
        for k in range(3):                       # tap k sees x[t - (2 - k) * d]
            sh = (2 - k) * d
            y[sh:] += x[:n - sh] @ w[:, :, k].T
        x = np.maximum(y, 0.0)
        outs.append(x)
    return outs


def side_raw(f, T, D, sgn, seg_start=0):
    """16 side inputs ('v7tcp', research side_feats columns 1, 3-6, 8-18) for windows ending at bars T."""
    start = T - (L - 1)
    a1 = np.maximum(start - 1, 0)
    seg = np.full_like(T, seg_start)
    cs0, norm = f['cs0'], f['ema1'][a1]
    out = [np.log(D) - np.log(f['ema7'][T])]
    h = f['hour'][T] * (2 * np.pi / 24.0)
    dw = f['dow'][T] * (2 * np.pi / 7.0)
    out += [np.sin(h), np.cos(h), np.sin(dw), np.cos(dw)]

    def move(lo, hi):
        n_eff = np.maximum(hi - lo, 1)
        return (cs0[hi] - cs0[lo]) / (np.sqrt(n_eff) * norm) * sgn
    a0 = np.minimum(np.maximum(a1 - BARS_DAY, seg), a1)
    out.append(move(a0, a1))
    A1 = np.maximum(a1, seg)
    for n_ in (3 * BARS_DAY, 7 * BARS_DAY):
        out.append(move(np.maximum(A1 - n_, seg), A1))
    for k in range(CTX_NBLK - 1, -1, -1):        # oldest block first
        out.append(move(np.maximum(A1 - CTX_BLOCK * (k + 1), seg), np.maximum(A1 - CTX_BLOCK * k, seg)))
    return np.stack(out, axis=1)


def head(p, feats):
    z = np.maximum(feats @ p.head0_w.T + p.head0_b, 0.0)
    return (z @ p.head3_w.T)[:, 0] + p.head3_b[0]


def infer(f, p, floor, eps=TIE_EPS, seg_start=0, T=None, layers=None):
    """Logit of every valid window (or of the windows ending at bars T). Returns a dict of per-window arrays."""
    n = f['n']
    okw = np.zeros(n, dtype=bool)
    if n >= L:
        cnt = np.convolve(f['ok'].astype(np.int64), np.ones(L, dtype=np.int64), mode='valid')
        okw[L - 1:] = cnt == L
    if T is None:
        T = np.where(okw)[0]
    T = np.asarray(T, dtype=np.int64)
    if not okw[T].all():
        raise ValueError("some requested windows are not valid")
    rw = sliding_window_view(np.nan_to_num(f['r']), L)[T - (L - 1)]           # (m, 64)
    D = np.maximum(np.median(np.abs(rw), axis=1), floor)
    x_end = rw @ np.arange(L, dtype=np.float64) / L
    sgn = np.where(x_end < -eps * D, -1.0, 1.0)
    if layers is None:
        up, um = conv_inputs(f, p)
        layers = (conv_stream(up, p), conv_stream(um, p))
    y4p, y4m = layers[0][3], layers[1][3]
    P = 33
    wp = sliding_window_view(y4p, P, axis=0)[T - (P - 1)]                     # (m, 32, 33)
    wm = sliding_window_view(y4m, P, axis=0)[T - (P - 1)]
    W = np.where((sgn > 0)[:, None, None], wp, wm)
    pooled = np.concatenate([W.mean(axis=2), W.max(axis=2), W[:, :, -1]], axis=1) / D[:, None]
    s_raw = side_raw(f, T, D, sgn, seg_start)
    s_std = (s_raw - p.side_mu) / p.side_sd
    logit = head(p, np.concatenate([pooled, s_std], axis=1))
    return {'T': T, 'D': D, 'x_end': x_end, 'sgn': sgn, 'side_raw': s_raw, 'logit': logit, 'pooled': pooled,
            'layers': layers}
