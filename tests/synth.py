"""Synthetic 5M bars for tests: random walk with fine or coarse tick, optional grid gaps."""
import numpy as np
import pandas as pd

BAR = 300_000


def bars(n=4000, seed=0, tick=None, gaps=(), start='2026-01-05 00:00:00'):
    g = np.random.default_rng(seed)
    if tick is None:
        c = 100.0 * np.exp(np.cumsum(g.normal(0, 0.002, n)))
        o = np.concatenate([[c[0]], c[:-1]]) * np.exp(g.normal(0, 0.0003, n))
        h = np.maximum(o, c) * np.exp(np.abs(g.normal(0, 0.001, n)))
        lo = np.minimum(o, c) * np.exp(-np.abs(g.normal(0, 0.001, n)))
    else:
        k = np.cumsum(g.choice([-1, 0, 0, 0, 1], size=n)) + 3000
        c = k * tick
        o = np.concatenate([[c[0]], c[:-1]])
        h = np.maximum(o, c) + tick * g.choice([0, 1], size=n)
        lo = np.minimum(o, c) - tick * g.choice([0, 1], size=n)
    t = pd.Timestamp(start).value // 10 ** 6 + BAR * np.arange(n, dtype=np.int64)
    for i, k in gaps:                     # gap of k bars before bar i
        t[i:] += BAR * k
    return pd.DataFrame({'time': t, 'open': o, 'high': h, 'low': lo, 'close': c})
