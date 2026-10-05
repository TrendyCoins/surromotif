"""Per-bar inputs from OHLC, vectorized, with the semantics of research (SMC_Data / SMC_Prices / SMC_Range /
SeriesData of research/sumo_class_train.py):

    norm_t  = median(high - low) over the 200 bars BEFORE t (pandas rolling, closed='left')
    bar t is bad if norm_t is not finite or <= 0, or a grid gap (interval > 1.5 bars), or t is the first bar
    r_t     = (close_t - close_{t-1}) / norm_t
    e_p,t   = (p_t - close_{t-1}) / norm_t,   d_p,t = (p_t - close_t) / norm_t,   p in {HL/2, HLC/3}
    step_p,t = e_p,t - d_p,t-1                (d of the previous bar with ITS OWN norm)
    g_t     = (high_t - low_t) / norm_t       (bar range, a level)
    ok_t    = all of r, step_hl2, step_hlc3, g finite   (research SeriesData 'ok' mask)
    EMA |r| (spans 2016 and 288 bars): pandas ewm(adjust=False, ignore_na=True) over |r| of ok bars
    cs0[j]  = sum of r over ok bars i < j (NaN as 0), for the moves before the window
    hour, dow: from the bar OPEN time (CSV 'timestamp', Pine 'time'), UTC, Monday = 0

Research-only knobs (end-to-end check against the research npz, rules R3):
    ema_exclude - bars the EMA skips (research has no bars of the 200-bar warm-up of VALIDATION)
    seg_start   - first bar index of the research data segment (moves before the window are clipped there)
"""
import numpy as np
import pandas as pd

from . import GAP_MS, N_NORM, SPAN_CTX, SPAN_SIDE


def read_bars(path):
    """Binance CSV (timestamp = bar open time, UTC) -> DataFrame time (int64 ms), open, high, low, close."""
    df = pd.read_csv(path, usecols=['timestamp', 'open', 'high', 'low', 'close'])
    t = pd.to_datetime(df['timestamp']).to_numpy().astype('datetime64[ms]').astype(np.int64)
    return pd.DataFrame({'time': t, 'open': df['open'].to_numpy(np.float64), 'high': df['high'].to_numpy(np.float64),
                         'low': df['low'].to_numpy(np.float64), 'close': df['close'].to_numpy(np.float64)})


def ema(x, span, adjust=False):
    """pandas ewm(span, adjust, ignore_na=True).mean(), then <= 0 -> NaN and back/forward fill (research)."""
    e = pd.Series(np.abs(x)).ewm(span=span, adjust=adjust, ignore_na=True).mean().to_numpy()
    e = np.where(np.isfinite(e) & (e > 0), e, np.nan)
    return pd.Series(e).bfill().ffill().to_numpy()


def bar_features(df, ema_exclude=None, ema_adjust=False):
    t = df['time'].to_numpy(np.int64)
    h, l, c = (df[k].to_numpy(np.float64) for k in ('high', 'low', 'close'))
    n = len(c)
    rng = h - l
    norm = pd.Series(rng).rolling(N_NORM, closed='left').median().to_numpy()
    good = np.isfinite(norm) & (norm > 0)
    gap = np.zeros(n, dtype=bool)
    gap[1:] = np.diff(t) > GAP_MS
    first = np.zeros(n, dtype=bool)
    first[0] = True
    bad = ~good | gap | first
    c_prev = np.concatenate([[np.nan], c[:-1]])
    with np.errstate(invalid='ignore', divide='ignore'):
        safe = np.where(good, norm, 1.0)
        r = np.where(bad, np.nan, (c - c_prev) / safe)
        steps = []
        for p in ((h + l) / 2.0, (h + l + c) / 3.0):
            e = np.where(bad, np.nan, (p - c_prev) / safe)
            d = np.where(good, (p - c) / safe, np.nan)
            steps.append(e - np.concatenate([[np.nan], d[:-1]]))
        g = np.where(bad, np.nan, rng / safe)
    ok = np.isfinite(r) & np.isfinite(steps[0]) & np.isfinite(steps[1]) & np.isfinite(g)
    r_ok = np.where(ok, r, np.nan)
    r_ema = r_ok if ema_exclude is None else np.where(ema_exclude, np.nan, r_ok)
    cs0 = np.concatenate([[0.0], np.cumsum(np.nan_to_num(r_ok))])
    minutes = (t // 60_000) % 1440
    return {'time': t, 'n': n, 'norm': norm, 'ok': ok, 'r': r_ok, 'step_hl2': np.where(ok, steps[0], np.nan),
            'step_hlc3': np.where(ok, steps[1], np.nan), 'g': np.where(ok, g, np.nan),
            'ema7': ema(r_ema, SPAN_SIDE, ema_adjust), 'ema1': ema(r_ema, SPAN_CTX, ema_adjust), 'cs0': cs0,
            'hour': minutes / 60.0, 'dow': ((t // 86_400_000) + 3) % 7, 'first_r': int(np.argmax(ok)) if ok.any() else n}
