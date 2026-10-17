"""PB reference and the line M on synthetic data (thresholds — docs/rules.md, session 4):
    the separable layer by the stream == a direct convolution by its formula; bar-by-bar stream == vectorized (< 1e-10;
    fine / coarse tick, grid gaps); line M == research _nc_z / _nc_live / _nc_logm (<= 1e-12); the streaming EMA == pandas
    ewm(span, adjust=True) (<= 1e-12); m = 0, s = 1 -> M = 1; M does not decrease in z for any m >= 0, s; warm-up R16
    (window numbers over valid windows, invalid windows do not move the state); first line bar 552 on continuous data."""
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

FLOOR = 0.2
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
    s = pb.run(b, p, FLOOR, Q, ZK, warm=50, span=300)
    v = pb.chart(b, p, FLOOR, Q, ZK, warm=50, span=300)
    sl = np.array([x['logit'] for x in s])
    assert np.array_equal(np.isfinite(sl), v['valid'])
    T = np.where(v['valid'])[0]
    assert np.abs(sl[T] - v['logit'][T]).max() < 1e-10
    for k in ('z', 'm_ema', 's_ema', 'log_m'):
        sk = np.array([x[k] for x in s])
        assert np.array_equal(np.isfinite(sk), v['valid']), k
        assert np.abs(sk[T] - v[k][T]).max() < 1e-10, k
    assert np.array_equal(np.array([x['n'] if x['logit'] == x['logit'] else -1 for x in s]), v['n'])
    assert np.array_equal(np.array([x['shown'] for x in s]), v['shown'])
    if kind != 'gaps':
        assert T[0] == 264 and np.where(v['shown'])[0][0] == 264 + 50


def test_first_line_bar_552():
    b = synth.bars(700, seed=5)
    v = pb.chart(b, pb.random_params(seed=6), FLOOR, Q, ZK)
    assert np.where(v['valid'])[0][0] == 264
    assert np.where(v['shown'])[0][0] == 264 + linem.WARM == 552
    assert v['n'][552] == 288


def test_warmup_counts_valid_windows_only():
    b = synth.bars(1300, seed=7, gaps=[(700, 3)])
    v = pb.chart(b, pb.random_params(seed=8), FLOOR, Q, ZK, warm=500)
    T = np.where(v['valid'])[0]
    assert np.array_equal(v['n'][T], np.arange(len(T)))
    assert not v['valid'][700:700 + L].any()                          # windows over the gap: no number, no state
    assert T[435] == 699 and T[436] == 700 + L                         # 264..699 — 436 windows, then from bar 764
    assert np.where(v['shown'])[0][0] == T[500] == 764 + 64
    # the state after the gap continues from the last valid window (no reset): equal to the EMA over valid windows only
    m, s = linem.live(linem.ruler_z(v['logit'][T], Q, ZK))
    assert np.abs(m - v['m_ema'][T]).max() < 1e-12 and np.abs(s - v['s_ema'][T]).max() < 1e-12


def test_ruler_scalar_equals_np_interp():
    x = np.concatenate([G.normal(-1, 2, 5000), Q, [Q[0] - 1, Q[-1] + 1, -50.0, 50.0]])
    z1 = np.array([linem.ruler_z1(v, list(Q), list(ZK)) for v in x])
    assert np.array_equal(z1, linem.ruler_z(x, Q, ZK))


def test_ema_stream_equals_pandas():
    lv = linem.LiveM(Q, ZK, span=8064)
    rows = [lv.update(x) for x in G.normal(-0.5, 1.5, 30000)]
    zz = np.array([r['z'] for r in rows])
    m = pd.Series(zz).ewm(span=8064, adjust=True).mean().to_numpy()
    v = pd.Series(zz * zz).ewm(span=8064, adjust=True).mean().to_numpy()
    assert np.abs(np.array([r['m_ema'] for r in rows]) - m).max() <= 1e-12
    assert np.abs(np.array([r['s_ema'] for r in rows]) - np.sqrt(np.maximum(v - m * m, 0))).max() <= 1e-12


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


def test_line_matches_research(research):
    T = research
    zr = G.normal(0.5, 0.9, 20000)
    m0, s0 = T._nc_live(zr, 8064)
    m1, s1 = linem.live(zr, 8064)
    assert np.abs(m0 - m1).max() <= 1e-12 and np.abs(s0 - s1).max() <= 1e-12
    lv = linem.LiveM(Q, ZK)
    for i, x in enumerate(zr[:3000]):                         # streaming recursion against the research EMA
        lv.n1, lv.n2, lv.d = lv.k * lv.n1 + x, lv.k * lv.n2 + x * x, lv.k * lv.d + 1.0
        mm = lv.n1 / lv.d
        assert abs(mm - m0[i]) <= 1e-12 and abs(math.sqrt(max(lv.n2 / lv.d - mm * mm, 0)) - s0[i]) <= 1e-12
    x = G.normal(-1, 2, 20000)
    c = {'q': Q}
    assert np.abs(T._nc_z('glob_e', x, None, c, ZK) - linem.ruler_z(x, Q, ZK)).max() <= 1e-12
    for m_, s_ in ((m0, s0), (G.normal(0.2, 0.5, 20000), np.abs(G.normal(0.9, 0.2, 20000)))):
        a = T._nc_logm(zr, m_, s_)
        b = linem.log_m(zr, m_, s_)
        assert np.abs(a - b).max() <= 1e-12
        assert np.abs(a - np.array([linem.log_m1(*v) for v in zip(zr, m_, s_)])).max() <= 1e-12
    assert T.NOCAL['span'] == linem.SPAN and T.NOCAL['warm'] == linem.WARM and T.NOCAL['s_lo'] == linem.S_LO


def test_check_functions_match_research(research):
    """mri.linem_check (curve, ceilings, summary, rulers, cells) == research _nc_* on synthetic data of two seeds."""
    from mri import linem_check as LC
    T = research
    g = np.random.default_rng(21)
    wk0 = pd.Timestamp('2025-07-07').value // LC.WEEK_NS
    D = {s: {a: {'sp': g.normal(0.3 + 0.1 * i, 1.3, 3000), 'sn': g.normal(-0.2, 1.0, 3000),
                 'wk': np.sort(g.integers(wk0, wk0 + 50, 3000)).astype(np.int32)} for i, a in enumerate(('X1', 'X2', 'X3'))}
         for s in ('a', 'b')}
    Dr = {s: {a: {**d, 'st': np.arange(3000), 'zf': np.zeros((3000, 8)), 'zfn': np.zeros((3000, 8))} for a, d in D_.items()}
          for s, D_ in D.items()}
    knots = np.asarray(T.NOCAL['knots'])
    from scipy.special import ndtri
    zk = ndtri(knots)
    Rr, Rm = T._nc_rulers(Dr, knots, skip=('X3',)), LC.rulers(D, knots, skip=('X3',))
    assert all(np.array_equal(Rr[k]['q'], Rm[k]) for k in Rm)
    for null in (False, True):
        cr = T._nc_cells(Dr, 'glob_e', Rr, zk, 500, null_chart=null)
        cm = LC.cells(D, Rm, zk, 500, linem.WARM, linem.S_LO, null_chart=null)
        assert all(np.abs(x[2] - y[2]).max() <= 1e-12 and np.abs(x[3] - y[3]).max() <= 1e-12 for x, y in zip(cr, cm))
        sr, sm = T._nc_summary(cr), LC.summary(cm)
        assert sr['ceil'] == sm['ceil'] and sr['share'] == sm['share'] and sr['n'] == sm['n']
        for a, b in zip(sr['rows'], sm['rows']):
            assert (a['n_real'], a['n_surr'], a['sparse']) == (b['n_real'], b['n_surr'], b['sparse'])
            assert (np.isnan(a['ratio']) and np.isnan(b['ratio'])) or abs(a['ratio'] - b['ratio']) <= 1e-12
        assert all(sr['assets'][a][c] == sm['assets'][a][c] for a in sm['assets'] for c in ('orange', 'blue'))
    assert LC.EDGES == T.NC_EDGES and LC.TOL == tuple(T.NOCAL['tol']) and LC.MIN_N == T.NOCAL['min_n']
    assert LC.HALF_CUT == T.NOCAL['half_cut'] and LC.CEIL_ASSET == T.NOCAL['ceil_asset']
