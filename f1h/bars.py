"""Candles -> per-bar inputs of the model, numpy only. The arithmetic of the research data files (research/sumo_class_tf.py
build_tf_files / r_series = SuMo_Class_Data / Prices / Range / Bars) and of research/sumo_class_train.py nt_segment, over the
rows of the input as they are (missing hours are not filled in):

    norm_t = median(high - low) over the 200 rows before t (pandas rolling 200, closed='left'; even count: mean of the two
             middle values)
    gap_t  = interval to the previous candle > 1.5 bars (research: 1.5 median intervals — on hourly candles the same)
    bad_t  = norm_t not finite or <= 0, or gap_t, or t is the first row
    r_t    = (close_t - close_{t-1}) / norm_t                       (NaN where bad)
    e_p,t  = (p_t - close_{t-1}) / norm_t (NaN where bad), d_p,t = (p_t - close_t) / norm_t, p = HL/2 and HLC/3
    step_p,t = e_p,t - d_p,t-1                                      (d of the previous bar with ITS OWN norm)
    g_t    = (high_t - low_t) / norm_t                              (NaN where bad)
    ok_t   = r, both steps and g finite                             (research SeriesData: the mask of every channel)

History of a window starting at the bar s (rows s-200 .. s-1; nt_segment rows of the 17 side inputs 'nt_PRV'): rmed =
median(high - low) (= norm of s), rmean = mean(high - low), bmed = max(median |close - open|, 0.05 rmed); the path before the
window in 8 chunks over rows s-200 .. s-2 (71, 64, 32, 16, 8, 4, 2, 2 rows, oldest first): (close of the last row - open of
the first) / sqrt(n) / rmed, clipped to +-20; R1 = 2 (close[s-2] - lmin) / (hmax - lmin) - 1 (0 if hmax = lmin), hmax / lmin
over s-200 .. s-2; R2 = log(max(hmax - lmin, 1e-12) / (sqrt(199) rmed)) clipped to +-5; R3 = (close[s-2] - mean close over
s-200 .. s-2) / (sqrt(199) rmed) clipped to +-20; V = log(max(v, 1e-3)) clipped to +-5 for v = mean(high - low) of the last
16 rows / rmean, of the last 64 rows / rmean, rmean / rmed, bmed / rmed. NaN -> 0. Research casts these to float32; here
they stay float64.
"""
import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

NT_GEO = (71, 64, 32, 16, 8, 4, 2, 2)
NT_SHORT = (16, 64)
NT_BODY_FLOOR = 0.05
NT_CLIP_Z, NT_CLIP_LOG = 20.0, 5.0
SIDE_NAMES = ('log_D', 'log_mean_abs_r', 'path_71', 'path_64', 'path_32', 'path_16', 'path_8', 'path_4', 'path_2a',
              'path_2b', 'range_pos', 'range_width', 'close_dev', 'vol_16', 'vol_64', 'range_mean_med', 'body_med')
SIDE_SIGNED = (False, False) + (True,) * 8 + (True, False, True) + (False,) * 4
CHUNK = 20000                    # rows per block of the rolling statistics (memory)


def check_candles(time, open_, high, low, close, bar_ms):
    """Arrays of the input; ValueError if they are not closed candles of one timeframe in time order."""
    t = np.asarray(time)
    if t.dtype.kind not in 'iu':
        raise ValueError(f"time: integer milliseconds expected, got {t.dtype}")
    t = t.astype(np.int64)
    px = [np.asarray(x, dtype=np.float64) for x in (open_, high, low, close)]
    if t.ndim != 1 or any(x.shape != t.shape for x in px):
        raise ValueError("time, open, high, low, close: 1-d arrays of the same length expected")
    o, h, l, c = px
    if len(t) > 1 and not np.all(np.diff(t) > 0):
        raise ValueError("time must increase strictly (oldest candle first, no duplicates)")
    if np.any(t % bar_ms):
        raise ValueError(f"time must be the open time of the candle, a multiple of {bar_ms} ms")
    fin = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    good = fin & (l > 0) & (l <= o) & (o <= h) & (l <= c) & (c <= h)
    if not good.all():
        i = int(np.argmin(good))
        raise ValueError(f"broken candle at row {i} (time {int(t[i])}): o {o[i]}, h {h[i]}, l {l[i]}, c {c[i]}")
    return t, o, h, l, c


def _rolling(x, n, how, starts, lag):
    """Statistic `how` over rows s - lag - n + 1 .. s - lag for each s in starts (all rows must exist)."""
    out = np.empty(len(starts))
    view = sliding_window_view(x, n)                  # row i: rows i .. i + n - 1
    for a in range(0, len(starts), CHUNK):
        rows = starts[a:a + CHUNK] - lag - n + 1
        out[a:a + CHUNK] = how(view[rows], axis=1)
    return out


def bar_inputs(t, o, h, l, c, bar_ms, gap_factor, n_norm):
    """Per-bar inputs: dict of arrays norm, ok, r, step_hl2, step_hlc3, g (NaN where not ok)."""
    n = len(c)
    rng = h - l
    norm = np.full(n, np.nan)
    if n > n_norm:
        norm[n_norm:] = _rolling(rng, n_norm, np.median, np.arange(n_norm, n), 1)
    good = np.isfinite(norm) & (norm > 0)
    gap = np.zeros(n, dtype=bool)
    gap[1:] = np.diff(t) > gap_factor * bar_ms
    bad = ~good | gap
    bad[:1] = True
    c_prev = np.concatenate([[np.nan], c[:-1]])
    safe = np.where(good, norm, 1.0)
    r = np.where(bad, np.nan, (c - c_prev) / safe)
    steps = []
    for p in ((h + l) / 2.0, (h + l + c) / 3.0):
        e = np.where(bad, np.nan, (p - c_prev) / safe)
        d = np.where(good, (p - c) / safe, np.nan)
        steps.append(e - np.concatenate([[np.nan], d[:-1]]))
    g = np.where(bad, np.nan, rng / safe)
    ok = np.isfinite(r) & np.isfinite(steps[0]) & np.isfinite(steps[1]) & np.isfinite(g)
    return {'norm': norm, 'ok': ok, 'r': np.where(ok, r, np.nan), 'step_hl2': np.where(ok, steps[0], np.nan),
            'step_hlc3': np.where(ok, steps[1], np.nan), 'g': np.where(ok, g, np.nan)}


def history(o, h, l, c, starts, n_hist):
    """(len(starts), 15) unsigned history inputs of windows starting at the rows `starts` (each >= n_hist)."""
    starts = np.asarray(starts, dtype=np.int64)
    if len(starts) == 0:
        return np.zeros((0, 15))
    if starts.min() < n_hist:
        raise ValueError(f"a window start with fewer than {n_hist} rows before it")
    nh2 = n_hist - 1
    rng, body = h - l, np.abs(c - o)
    rmed = _rolling(rng, n_hist, np.median, starts, 1)
    rmean = _rolling(rng, n_hist, np.mean, starts, 1)
    bmed = np.maximum(_rolling(body, n_hist, np.median, starts, 1), NT_BODY_FLOOR * rmed)
    hmax = _rolling(h, nh2, np.max, starts, 2)
    lmin = _rolling(l, nh2, np.min, starts, 2)
    cma = _rolling(c, nh2, np.mean, starts, 2)
    c2 = c[starts - 2]
    out = []
    with np.errstate(invalid='ignore', divide='ignore'):
        chunks, end = [], starts - 2                   # freshest chunk first
        for n in NT_GEO[::-1]:
            start = end - n + 1
            chunks.append((c[end] - o[start]) / math.sqrt(n))
            end = start - 1
        out += [np.clip(v / rmed, -NT_CLIP_Z, NT_CLIP_Z) for v in chunks[::-1]]
        w = hmax - lmin
        out.append(np.where(w > 0, 2.0 * (c2 - lmin) / np.where(w > 0, w, 1.0) - 1.0, 0.0))
        out.append(np.clip(np.log(np.maximum(w, 1e-12) / (math.sqrt(nh2) * rmed)), -NT_CLIP_LOG, NT_CLIP_LOG))
        out.append(np.clip((c2 - cma) / (math.sqrt(nh2) * rmed), -NT_CLIP_Z, NT_CLIP_Z))
        r16 = _rolling(rng, NT_SHORT[0], np.mean, starts, 1)
        r64 = _rolling(rng, NT_SHORT[1], np.mean, starts, 1)
        for v in (r16 / rmean, r64 / rmean, rmean / rmed, bmed / rmed):
            out.append(np.clip(np.log(np.maximum(v, 1e-3)), -NT_CLIP_LOG, NT_CLIP_LOG))
    return np.nan_to_num(np.stack(out, axis=1), nan=0.0, posinf=0.0, neginf=0.0)


def window_ends(ok, window, n_hist):
    """Rows T where a window ends: the bars T - window + 1 .. T all ok, the first of them with >= n_hist rows before it."""
    n = len(ok)
    okw = np.zeros(n, dtype=bool)
    if n >= window:
        cnt = np.convolve(ok.astype(np.int64), np.ones(window, dtype=np.int64), mode='valid')
        okw[window - 1:] = cnt == window
    okw[:n_hist + window - 1] = False
    return np.where(okw)[0]
