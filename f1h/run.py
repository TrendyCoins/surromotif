"""Candles of one symbol -> the values of every bar: the window that ends on the bar, its logit and the line M."""
import numpy as np

from .bars import bar_inputs, check_candles, window_ends
from .line import line
from .net import logits

FIELDS = ('time', 'window', 'n', 'logit', 'z', 'm', 's', 'n_live', 'line', 'log_m', 'strength', 'side')


def compute(model, time, open_, high, low, close):
    """Closed candles of one symbol on the timeframe of the model (model.bar_ms), oldest first: open time (ms UTC, a
       multiple of the bar) and OHLC. Missing bars stay missing. Returns a dict of arrays, one value per candle (the window
       that ends on it):
           time       open time of the candle (int64 ms)
           window     a window ends here (bool); n — its number from the first window of the input (-1: no window)
           logit      logit of the window (NaN: no window)
           z          z of the window by the ruler
           m, s       centre and spread of the live part (before the limits of log M); n_live — windows in it (-1)
           line       the line M is shown here (bool): the window number >= show_from
           log_m      log M (NaN: no line)
           strength   min(exp |log M|, ceiling): M for orange, 1 / M for blue, from 1 to the ceiling (NaN: no line)
           side       +1 orange (more often than random), -1 blue (less often), 0 — M = 1 (NaN: no line)
       A value of a bar depends only on the candles up to it; with enough history before it (bars_needed), not on where
       the input starts."""
    t, o, h, l, c = check_candles(time, open_, high, low, close, model.bar_ms)
    f = bar_inputs(t, o, h, l, c, model.bar_ms, model.gap_factor, model.n_norm)
    T = window_ends(f['ok'], model.window, model.n_hist)
    nb = len(t)
    out = {'time': t, 'window': np.zeros(nb, dtype=bool), 'n': np.full(nb, -1, dtype=np.int64),
           'n_live': np.full(nb, -1, dtype=np.int64), 'line': np.zeros(nb, dtype=bool)}
    for k in ('logit', 'z', 'm', 's', 'log_m', 'strength', 'side'):
        out[k] = np.full(nb, np.nan)
    if len(T) == 0:
        return out
    lg = logits(model, f, (o, h, l, c), T)
    ln = line(model, lg)
    out['window'][T] = True
    out['n'][T] = np.arange(len(T))
    out['logit'][T] = lg
    for k, v in ln.items():
        out[k][T] = v
    return out


def bars_needed(model, k=1):
    """Candles needed for the line on the last k candles when the candles are continuous (no missing bars): the first
       window ends on the candle first_window_bar (from 0), the line starts show_from windows later. A missing bar costs
       more: the candle after it and the windows over that candle do not exist."""
    if k < 1:
        raise ValueError("k >= 1")
    return model.first_line_bar + k
