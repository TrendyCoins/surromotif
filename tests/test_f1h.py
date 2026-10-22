"""F1 on 1H (f1h) on synthetic candles and on the golden file (rules R20, R24, R25 of docs/rules.md):
    the golden file end to end (no artifacts/); slow window-by-window reference == vectorized (<= 1e-10); grid gaps (the bar
    after a gap invalid, windows over valid bars only, numbers over existing windows, the line over the sequence of existing
    windows); dead norm; orientation tie and mirrored series; start of the history cut off; little history (first window on
    the candle 264, first line candle 4360); bars_needed; input errors; the line == research formulas written out."""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import f1h  # noqa: E402
from f1h.bars import bar_inputs, window_ends  # noqa: E402
from f1h.line import line  # noqa: E402
from f1h.net import logits, window_inputs  # noqa: E402

H = 3_600_000
T0 = 1_767_225_600_000                      # 2026-01-01 00:00 UTC


@pytest.fixture(scope='module')
def model():
    return f1h.load_model()


def candles(n, seed=0, tick=None, gaps=(), flat=None, start=T0):
    """Hourly random walk; tick — coarse price grid (exact ties); gaps — (row, missing hours before it); flat — (row, rows)
       of candles with high = low = open = close (dead norm)."""
    g = np.random.default_rng(seed)
    if tick is None:
        c = 30000.0 * np.exp(np.cumsum(g.normal(0, 0.006, n)))
        o = np.concatenate([[c[0]], c[:-1]]) * np.exp(g.normal(0, 0.0005, n))
        h = np.maximum(o, c) * np.exp(np.abs(g.normal(0, 0.003, n)))
        lo = np.minimum(o, c) * np.exp(-np.abs(g.normal(0, 0.003, n)))
    else:
        k = np.cumsum(g.choice([-1, 0, 0, 1], size=n)) + 4000
        c = k * tick
        o = np.concatenate([[c[0]], c[:-1]])
        h = np.maximum(o, c) + tick * g.choice([0, 1], size=n)
        lo = np.minimum(o, c) - tick * g.choice([0, 1], size=n)
    if flat is not None:
        a, m = flat
        o[a:a + m] = h[a:a + m] = lo[a:a + m] = c[a:a + m] = c[a - 1]
    t = start + H * np.arange(n, dtype=np.int64)
    for i, k in gaps:
        t[i:] += H * k
    return {'time': t, 'open': o, 'high': h, 'low': lo, 'close': c}


def run(model, cd, a=0, b=None):
    return f1h.compute(model, *(cd[k][a:b] for k in ('time', 'open', 'high', 'low', 'close')))


# --------------------------------------------------------------------------------------------------------------------
# slow reference: one window at a time, the research path written out (to_shapes, conv_maps with padding, mirror_inputs)

def slow_logit(model, cd, T):
    o, h, l, c, t = (np.asarray(cd[k], dtype=np.float64) for k in ('open', 'high', 'low', 'close', 'time'))
    n = len(c)
    norm = np.full(n, np.nan)
    for i in range(200, n):
        norm[i] = np.median(h[i - 200:i] - l[i - 200:i])

    def bar(i):                                   # r, step_hl2, step_hlc3, g of the bar i (NaN if invalid)
        good = np.isfinite(norm[i]) and norm[i] > 0
        good_prev = i >= 1 and np.isfinite(norm[i - 1]) and norm[i - 1] > 0
        if not (good and good_prev and t[i] - t[i - 1] <= 1.5 * H):
            return None
        out = [(c[i] - c[i - 1]) / norm[i]]
        for p in ((h + l) / 2.0, (h + l + c) / 3.0):
            out.append((p[i] - c[i - 1]) / norm[i] - (p[i - 1] - c[i - 1]) / norm[i - 1])
        out.append((h[i] - l[i]) / norm[i])
        return out
    s = T - 63
    rows = [bar(i) for i in range(s, T + 1)]
    assert all(r is not None for r in rows)
    W = np.array(rows).T                                                   # (4, 64)
    D = max(np.median(np.abs(W[0])), model.floor)
    X = np.cumsum(W[:3], axis=1)
    X = X - X.mean(axis=1, keepdims=True)
    flip = X[0, -1] < -model.orient_eps * D
    X = (-X if flip else X) / D
    sgn = -1.0 if flip else 1.0
    lvl = W[3] / D
    # history (research nt_segment for one window)
    rng = h - l
    rmed, rmean = np.median(rng[s - 200:s]), np.mean(rng[s - 200:s])
    bmed = max(np.median(np.abs(c - o)[s - 200:s]), 0.05 * rmed)
    hmax, lmin, cma, c2 = h[s - 200:s - 1].max(), l[s - 200:s - 1].min(), c[s - 200:s - 1].mean(), c[s - 2]
    P, end = [], s - 2
    for m_ in (2, 2, 4, 8, 16, 32, 64, 71):
        P.append(float(np.clip((c[end] - o[end - m_ + 1]) / math.sqrt(m_) / rmed, -20, 20)))
        end -= m_
    w = hmax - lmin
    hist = P[::-1] + [2 * (c2 - lmin) / w - 1 if w > 0 else 0.0,
                      float(np.clip(math.log(max(w, 1e-12) / (math.sqrt(199) * rmed)), -5, 5)),
                      float(np.clip((c2 - cma) / (math.sqrt(199) * rmed), -20, 20))]
    for v in (rng[s - 16:s].mean() / rmean, rng[s - 64:s].mean() / rmean, rmean / rmed, bmed / rmed):
        hist.append(float(np.clip(math.log(max(v, 1e-3)), -5, 5)))
    signed = model.side_signed
    side = np.array([math.log(D), math.log(max(np.abs(W[0]).mean(), model.floor))] + hist)
    side = np.where(signed, side * sgn, side)

    def g(x, sd):                                                          # CNNBase._logit of one window
        z = x[:3, 1:] - x[:3, :-1]
        hh = np.concatenate([z / model.s_sd, (x[3:, 1:] - model.lvl_mu) / model.lvl_sd], axis=0)    # (4, 63)
        for wt, b, d in zip(model.conv_w, model.conv_b, model.dilations):
            pad = 2 * d
            xp = np.concatenate([np.zeros((hh.shape[0], pad)), hh, np.zeros((hh.shape[0], pad))], axis=1)
            y = np.empty((wt.shape[0], 63))
            for p_ in range(63):
                y[:, p_] = b + sum(wt[:, :, j] @ xp[:, p_ + j * d] for j in range(5))
            hh = np.maximum(y, 0.0)
        f = np.concatenate([hh.mean(axis=1), hh.max(axis=1), hh[:, -1], (sd - model.side_mu) / model.side_sd])
        zz = np.maximum(model.head0_w @ f + model.head0_b, 0.0)
        return float(model.head3_w[0] @ zz + model.head3_b[0])
    x = np.concatenate([X, lvl[None, :]], axis=0)
    xm = np.concatenate([-X, lvl[None, :]], axis=0)                       # mirror_inputs: paths negated, level kept
    return 0.5 * (g(x, side) + g(xm, np.where(signed, -side, side)))


def test_slow_equals_vectorized(model):
    for cd in (candles(700, seed=1), candles(700, seed=2, tick=0.5, gaps=((500, 3),))):
        out = run(model, cd)
        T = np.where(out['window'])[0]
        pick = T[np.linspace(0, len(T) - 1, 12).astype(int)]
        d = max(abs(slow_logit(model, cd, int(t)) - out['logit'][t]) for t in pick)
        assert d <= 1e-10


# --------------------------------------------------------------------------------------------------------------------
def test_golden_end_to_end(model):
    res = f1h.check_golden(model)
    assert res['ok'], res
    assert res['first_window_bar'] == 264 and res['first_line_bar'] == 4360 and res['bars_with_line'] == 7120


def test_gaps(model):
    gaps = ((300, 1), (420, 5), (480, 1))
    cd = candles(5200, seed=3, gaps=gaps)
    out = run(model, cd)
    f = bar_inputs(cd['time'], cd['open'], cd['high'], cd['low'], cd['close'], H, 1.5, 200)
    for i, _ in gaps:
        assert not f['ok'][i] and f['ok'][i - 1] and f['ok'][i + 1]
    ok = f['ok']
    want = np.array([i >= 263 + 1 and ok[i - 63:i + 1].all() and i - 63 >= 200 for i in range(len(ok))])
    assert np.array_equal(out['window'], want)
    T = np.where(out['window'])[0]
    assert np.array_equal(out['n'][T], np.arange(len(T))) and (out['n'][~out['window']] == -1).all()
    for i, _ in gaps:                                      # no window covers the bar after a gap
        assert not out['window'][i:i + 64].any()
    ln = line(model, out['logit'][T])                      # the line over the sequence of existing windows
    for k in ('z', 'm', 's', 'log_m', 'strength', 'side'):
        a, b = out[k][T], ln[k]
        assert np.array_equal(np.isnan(a), np.isnan(b)) and np.nanmax(np.abs(a - b)) <= 1e-12
    assert np.array_equal(out['n_live'][T], ln['n_live']) and np.array_equal(out['line'][T], ln['line'])
    first = int(T[model.show_from])
    assert out['line'][first] and not out['line'][:first].any()
    assert first > model.first_line_bar                    # a gap costs windows: the line starts later


def test_dead_norm(model):
    cd = candles(1400, seed=4, flat=(300, 450))
    out = run(model, cd)
    f = bar_inputs(cd['time'], cd['open'], cd['high'], cd['low'], cd['close'], H, 1.5, 200)
    dead = (f['norm'] == 0)
    assert dead.sum() > 100 and not f['ok'][dead].any()
    T = np.where(out['window'])[0]
    assert not any(dead[t - 63:t + 1].any() for t in T)     # no window over a dead norm
    assert T[0] == 264 and (T > np.where(dead)[0][-1]).any()   # windows before the flat stretch and after it
    assert np.isfinite(out['logit'][T]).all()


def test_tie_and_mirror(model):
    cd = candles(2600, seed=5, tick=1.0)                    # coarse grid: exact ties (x_end = 0)
    out = run(model, cd)
    T = np.where(out['window'])[0]
    f = bar_inputs(cd['time'], cd['open'], cd['high'], cd['low'], cd['close'], H, 1.5, 200)
    ohlc = (cd['open'], cd['high'], cd['low'], cd['close'])
    inp = window_inputs(model, f, ohlc, T)
    z = inp['x_end'] / inp['D']
    assert (z == 0).sum() >= 1                             # exact ties
    near = np.argsort(np.abs(z + model.orient_eps))[:5]     # windows at the threshold -eps D
    pick = np.unique(np.concatenate([np.where(z == 0)[0], near]))
    for s in (1.0, -1.0):
        lg = logits(model, f, ohlc, T[pick], orient=s)
        assert np.max(np.abs(lg - out['logit'][T[pick]])) <= 1e-12
    k = 2.0 * float(np.max(cd['high']))                     # mirrored series: K - price, high <-> low
    mir = {'time': cd['time'], 'open': k - cd['open'], 'high': k - cd['low'], 'low': k - cd['high'], 'close': k - cd['close']}
    cf = candles(900, seed=6)
    mir_f = {'time': cf['time'], 'open': 1e5 - cf['open'], 'high': 1e5 - cf['low'], 'low': 1e5 - cf['high'],
             'close': 1e5 - cf['close']}
    for a, b in ((cd, mir), (cf, mir_f)):
        oa, ob = run(model, a), run(model, b)
        assert np.array_equal(oa['window'], ob['window'])
        assert np.nanmax(np.abs(oa['logit'] - ob['logit'])) <= 1e-9


def test_start_cut(model):
    cd = candles(5600, seed=7, gaps=((2500, 2),))
    full = run(model, cd)
    for cut in (1, 333, 700):
        part = run(model, cd, cut)
        sl = slice(cut, None)
        both = full['window'][sl] & part['window']
        assert both.sum() > 1000
        for k in ('logit', 'z'):
            assert np.max(np.abs(full[k][sl][both] - part[k][both])) <= 1e-12
        ln = full['line'][sl] & part['line']
        assert ln.sum() > 100
        for k in ('m', 's', 'log_m', 'strength'):
            assert np.max(np.abs(full[k][sl][ln] - part[k][ln])) <= 1e-9
        near0 = np.abs(full['log_m'][sl][ln]) <= 1e-9
        assert np.array_equal(full['side'][sl][ln][~near0], part['side'][ln][~near0])
        # the line is there on the same bars once both inputs have show_from windows before them
        n_part = part['n'][part['window']]
        assert np.array_equal(part['line'][part['window']], n_part >= model.show_from)


@pytest.mark.parametrize('n', [0, 1, 264, 265, 4360, 4361])
def test_little_history(model, n, long_run):
    cd, full = long_run
    out = run(model, cd, 0, n)
    assert all(len(out[k]) == n for k in f1h.FIELDS)
    assert out['window'].sum() == max(n - 264, 0) and out['line'].sum() == max(n - 4360, 0)
    if n > 264:
        assert np.argmax(out['window']) == 264
        assert np.max(np.abs(out['logit'][264:] - full['logit'][264:n])) <= 1e-12
    if n > 4360:
        assert np.argmax(out['line']) == 4360 and np.isfinite(out['strength'][4360])
    assert np.isnan(out['logit'][:min(n, 264)]).all() and np.isnan(out['strength'][:min(n, 4360)]).all()
    assert np.isnan(out['side'][:min(n, 4360)]).all() and (out['n'][:min(n, 264)] == -1).all()


@pytest.fixture(scope='module')
def long_run(model):
    cd = candles(4361, seed=8)
    return cd, run(model, cd)


def test_bars_needed(model, long_run):
    cd, full = long_run
    assert f1h.bars_needed(model) == 4361 and f1h.bars_needed(model, 24) == 4384
    assert model.first_window_bar == 264 and model.first_line_bar == 4360
    assert len(cd['time']) == f1h.bars_needed(model, 1) and full['line'][-1] and full['line'].sum() == 1
    with pytest.raises(ValueError):
        f1h.bars_needed(model, 0)


def test_strength_and_side(model):
    g = f1h.read_golden()
    lm, st = g['log_m'][g['line']], g['strength'][g['line']]
    assert model.ceiling == 8.0 and np.allclose(st, np.minimum(np.exp(np.abs(lm)), 8.0), rtol=1e-12, atol=0)
    assert st.min() >= 1.0 and st.max() == 8.0 and np.array_equal(g['side'][g['line']], np.sign(lm))
    assert np.isnan(g['strength'][~g['line']]).all() and np.isnan(g['side'][~g['line']]).all()


def test_line_formulas(model):
    """The line == chart_ms (kind 'win', F3) and _nc_logm of research written out window by window (prefix sums aside)."""
    g = np.random.default_rng(9)
    lg = np.concatenate([g.normal(0.5, 1.2, 4500), g.normal(-1.0, 0.3, 200), g.normal(2.0, 1.0, 300)])
    out = line(model, lg)
    z = np.interp(lg, model.q, model.zk)
    for p in (0, 1, 63, 64, 65, 200, 4095, 4096, 4097, 4999):
        hi = p - 63
        lo = max(0, hi - 4032)
        w = z[lo:hi] if hi > lo else np.zeros(0)
        n = len(w)
        if n:
            mh = w.sum() / n
            vh = max((w * w).sum() / n - mh * mh, 0.0)
            m, s = max(mh - 0.5 * math.sqrt(vh) * math.sqrt(64 / n), 0.0), math.sqrt(vh)
        else:
            m, s = 0.0, 1.0
        assert out['n_live'][p] == n and abs(out['m'][p] - m) <= 1e-12 and abs(out['s'][p] - s) <= 1e-12
        if p >= 4096:
            mm, ss = max(m, 0.0), min(max(s, 0.75), 1.0)
            zc = min(z[p], mm / max(1 - ss * ss, 1e-12)) if ss < 1 else z[p]
            want = -(zc - mm) ** 2 / (2 * ss * ss) - math.log(ss) + zc * zc / 2
            assert abs(out['log_m'][p] - want) <= 1e-12
        else:
            assert np.isnan(out['log_m'][p]) and not out['line'][p]


def test_input_errors(model):
    cd = candles(300, seed=10)
    t, o, h, l, c = (cd[k] for k in ('time', 'open', 'high', 'low', 'close'))
    bad = [(t[::-1], o, h, l, c), (np.concatenate([t[:5], t[4:-1]]), o, h, l, c), (t + 60_000, o, h, l, c),
           (t.astype(np.float64), o, h, l, c), (t, o[:-1], h, l, c), (t, o, l, h, c),
           (t, np.where(np.arange(300) == 7, np.nan, o), h, l, c), (t, o, h, l, np.where(np.arange(300) == 9, h * 2, c)),
           (t, o * 0, h, l, c)]
    for args in bad:
        with pytest.raises(ValueError):
            f1h.compute(model, *args)
    assert window_ends(np.ones(10, dtype=bool), 64, 200).size == 0
