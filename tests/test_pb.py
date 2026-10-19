"""PB reference and the line M on synthetic data (thresholds — docs/rules.md, sessions 4 and 5):
    the separable layer by the stream == a direct convolution by its formula; bar-by-bar stream == vectorized (< 1e-10;
    fine / coarse tick, grid gaps), the line of session 5 included (formula A: m_win, s_win, k, log M); ruler and log M ==
    research _nc_z / _nc_logm (<= 1e-12); m = 0, s = 1 -> M = 1; M does not decrease in z for any m >= 0, s; start of the
    line R18 (window numbers over valid windows, invalid windows do not move the state); first line bar 4360 on continuous
    data. The EMA line of session 4 — tests/legacy/test_linem_s4.py."""
import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth  # noqa: E402
from mri import L, linem, nt3, pb  # noqa: E402
from mri.batch import conv_stream  # noqa: E402
from mri.features import bar_features  # noqa: E402
from mri.params import DILATIONS  # noqa: E402

FLOOR = pb.FLOOR
G = np.random.default_rng(11)
Q = np.sort(G.normal(-1.0, 1.4, 37)) + np.arange(37) * 1e-3            # strictly increasing ruler of a fake model
ZK = np.sort(G.normal(0, 1.2, 37))


@pytest.fixture(scope='module')
def research():
    pytest.importorskip('torch')
    os.environ.setdefault('SMC_MODE', 'none')
    os.environ.setdefault('SMC_PROJECT_PATH', tempfile.mkdtemp() + '/')
    sys.path.insert(0, os.path.join(ROOT, 'research'))
    import sumo_class_train as T
    return T


def direct_layers(u, p):
    """Valid convolutions written from the formula (rules R14): layer 1 — sum over input channels and taps; layers 2..4 —
       dw_c(t) = sum_j W[c, 0, j] h_c(t - (2 - j) d), y_c(t) = sum_c' P[c, c'] dw_c'(t), then ReLU."""
    h, outs = u, []
    for i, (w, d) in enumerate(zip(p.conv, DILATIONS)):
        n = len(h)
        y = np.zeros((n, w.shape[0]))
        for t in range(2 * d, n):
            if i == 0:
                y[t] = sum(w[:, :, j] @ h[t - (2 - j) * d] for j in range(3))
            else:
                dw = np.array([sum(w[c, 0, j] * h[t - (2 - j) * d, c] for j in range(3)) for c in range(w.shape[0])])
                y[t] = p.pw[i - 1] @ dw
        h = np.maximum(y, 0.0)
        outs.append(h)
    return outs


def test_separable_layer_stream_equals_direct():
    p = pb.random_params(seed=1)
    assert p.n_params == 18697 and p.conv[1].shape == (64, 1, 3) and p.pw[0].shape == (64, 64)
    u = G.normal(0, 1, (80, 4))
    ref = direct_layers(u, p)
    vec = conv_stream(u, p)
    m = nt3.StreamingNT3(p, FLOOR)
    for t in range(len(u)):
        m._conv_step(0, u[t])
        if t >= 2 * sum(DILATIONS):                         # every layer has full support at this bar
            for j in range(4):
                assert np.abs(np.asarray(m.rings[0][j + 1][-1]) - ref[j][t]).max() < 1e-10
    for j in range(4):
        lo = 2 * sum(DILATIONS[:j + 1])
        assert np.abs(vec[j][lo:] - ref[j][lo:]).max() < 1e-10


@pytest.mark.parametrize('kind', ['fine', 'coarse', 'gaps'])
def test_stream_equals_vectorized(kind):
    b = {'fine': lambda: synth.bars(1500, seed=1),
         'coarse': lambda: synth.bars(1500, seed=2, tick=0.0001),
         'gaps': lambda: synth.bars(1500, seed=3, gaps=[(500, 1), (900, 5), (930, 2)])}[kind]()
    p = pb.random_params(seed=4)
    s = pb.run(b, p, FLOOR, Q, ZK)
    v = pb.chart(b, p, FLOOR, Q, ZK)
    sl = np.array([x['logit'] for x in s])
    assert np.array_equal(np.isfinite(sl), v['valid'])
    T = np.where(v['valid'])[0]
    assert np.abs(sl[T] - v['logit'][T]).max() < 1e-10
    for k in ('z', 'm_win', 's_win', 'log_m'):
        sk = np.array([x[k] for x in s])
        assert np.array_equal(np.isfinite(sk), v['valid']), k
        assert np.abs(sk[T] - v[k][T]).max() < 1e-10, k
    assert np.array_equal(np.array([x['n'] if x['logit'] == x['logit'] else -1 for x in s]), v['n'])
    assert np.array_equal(np.array([x['k'] for x in s]), v['k'])
    assert np.array_equal(np.array([x['shown'] for x in s]), v['shown'])
    if kind != 'gaps':
        assert T[0] == 264


@pytest.fixture(scope='module')
def long_chart():
    b = synth.bars(4700, seed=5, gaps=[(3000, 2)])
    return pb.chart(b, pb.random_params(seed=6), FLOOR, Q, ZK), pb.chart(b.iloc[:2990].reset_index(drop=True),
                                                                          pb.random_params(seed=6), FLOOR, Q, ZK)


def test_first_line_bar_4360(long_chart):
    """Continuous data: first window ends at bar 264 (R13), the line from the window 4096 — bar 4360 (R18); with a gap
       before that the start moves by the windows that lost validity."""
    b = synth.bars(4400, seed=15)
    v = pb.chart(b, pb.random_params(seed=16), FLOOR, Q, ZK)
    assert np.where(v['valid'])[0][0] == 264
    assert np.where(v['shown'])[0][0] == 264 + linem.SHOW_FROM == 4360
    assert v['n'][4360] == 4096 and v['k'][4360] == 4032 and v['k'][4359] == 4032 and v['k'][264 + 63] == 0


def test_line_start_counts_valid_windows_only(long_chart):
    v, head = long_chart
    T = np.where(v['valid'])[0]
    assert np.array_equal(v['n'][T], np.arange(len(T)))
    assert not v['valid'][3000:3000 + L].any()                        # windows over the gap: no number, no state
    assert T[2735] == 2999 and T[2736] == 3000 + L                     # 264..2999 — 2736 windows, then from bar 3064
    assert np.where(v['shown'])[0][0] == T[4096] == 3064 + 4096 - 2736
    # the state after the gap continues from the last valid window (no reset): formula A over valid windows only
    m, s, k = linem.live_win(linem.ruler_z(v['logit'][T], Q, ZK))
    assert np.abs(m - v['m_win'][T]).max() < 1e-12 and np.abs(s - v['s_win'][T]).max() < 1e-12
    assert np.array_equal(k, v['k'][T])
    # the bars before the gap give the same numbers on a shorter chart (the line does not look ahead)
    Th = np.where(head['valid'])[0]
    assert np.array_equal(head['m_win'][Th], v['m_win'][Th]) and np.array_equal(head['log_m'][Th], v['log_m'][Th])


def test_ruler_scalar_equals_np_interp():
    x = np.concatenate([G.normal(-1, 2, 5000), Q, [Q[0] - 1, Q[-1] + 1, -50.0, 50.0]])
    z1 = np.array([linem.ruler_z1(v, list(Q), list(ZK)) for v in x])
    assert np.array_equal(z1, linem.ruler_z(x, Q, ZK))


def test_random_market_gives_one():
    z = np.linspace(-4, 4, 801)
    assert np.abs(linem.log_m(z, np.zeros_like(z), np.ones_like(z))).max() < 1e-15
    assert all(abs(linem.log_m1(x, 0.0, 1.0)) < 1e-15 for x in z)


def test_m_does_not_decrease_in_z():
    z = np.linspace(-3.1, 3.1, 2001)
    for m in (0.0, 0.05, 0.3, 0.8, 1.5, -0.4):                     # m_ema < 0 is floored at 0
        for s in (0.2, 0.75, 0.8, 0.95, 0.999, 1.0, 1.3):           # s_ema clipped to [0.75; 1]
            lm = linem.log_m(z, np.full_like(z, m), np.full_like(z, s))
            assert (np.diff(lm) >= -1e-12).all(), (m, s)
            assert np.abs(lm - np.array([linem.log_m1(x, m, s) for x in z])).max() < 1e-12


def test_display_and_ceiling():
    v, col, cap = linem.display(np.log([0.5, 1.0, 2.0, 50.0, 1 / 50.0]), 34.0, 21.0)
    assert list(col) == [-1, 0, 1, 1, -1] and list(cap) == [False, False, False, True, True]
    assert np.allclose(v, [2.0, 1.0, 2.0, 34.0, 21.0])
    v, col, cap = linem.display(np.log([50.0, 1 / 50.0]))                # session 5: one ceiling 10 (R19)
    assert np.allclose(v, [10.0, 10.0]) and list(cap) == [True, True]


def test_ruler_and_log_m_match_research(research):
    """z by the ruler and log M == research _nc_z / _nc_logm (<= 1e-12), vectorized and window by window."""
    T = research
    x = G.normal(-1, 2, 20000)
    assert np.abs(T._nc_z('glob_e', x, None, {'q': Q}, ZK) - linem.ruler_z(x, Q, ZK)).max() <= 1e-12
    zr = G.normal(0.5, 0.9, 20000)
    for m_, s_ in ((G.normal(0.4, 0.3, 20000), np.abs(G.normal(0.95, 0.1, 20000))),
                   (G.normal(0.2, 0.5, 20000), np.abs(G.normal(0.9, 0.2, 20000)))):
        a = T._nc_logm(zr, m_, s_)
        assert np.abs(a - linem.log_m(zr, m_, s_)).max() <= 1e-12
        assert np.abs(a - np.array([linem.log_m1(*v) for v in zip(zr, m_, s_)])).max() <= 1e-12
    assert T.NOCAL['s_lo'] == linem.S_LO
