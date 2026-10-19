"""The EMA line M of session 4 (rules R15 / R16; record, not exported — mri/legacy/linem_s4.py) and its 'nocal' checks
(mri/legacy/linem_check.py), thresholds of session 4: the streaming EMA == pandas ewm(span, adjust=True) and research
_nc_live (<= 1e-12); the check functions == research _nc_* on synthetic data. Moved from tests/test_pb.py unchanged."""
import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mri import linem  # noqa: E402
from mri.legacy import linem_check as LC, linem_s4  # noqa: E402

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


def test_ema_stream_equals_pandas():
    lv = linem_s4.LiveM(Q, ZK, span=8064)
    rows = [lv.update(x) for x in G.normal(-0.5, 1.5, 30000)]
    zz = np.array([r['z'] for r in rows])
    m = pd.Series(zz).ewm(span=8064, adjust=True).mean().to_numpy()
    v = pd.Series(zz * zz).ewm(span=8064, adjust=True).mean().to_numpy()
    assert np.abs(np.array([r['m_ema'] for r in rows]) - m).max() <= 1e-12
    assert np.abs(np.array([r['s_ema'] for r in rows]) - np.sqrt(np.maximum(v - m * m, 0))).max() <= 1e-12


def test_ema_matches_research(research):
    T = research
    zr = G.normal(0.5, 0.9, 20000)
    m0, s0 = T._nc_live(zr, 8064)
    m1, s1 = linem_s4.live(zr, 8064)
    assert np.abs(m0 - m1).max() <= 1e-12 and np.abs(s0 - s1).max() <= 1e-12
    lv = linem_s4.LiveM(Q, ZK)
    for i, x in enumerate(zr[:3000]):                         # streaming recursion against the research EMA
        lv.n1, lv.n2, lv.d = lv.k * lv.n1 + x, lv.k * lv.n2 + x * x, lv.k * lv.d + 1.0
        mm = lv.n1 / lv.d
        assert abs(mm - m0[i]) <= 1e-12 and abs(math.sqrt(max(lv.n2 / lv.d - mm * mm, 0)) - s0[i]) <= 1e-12
    a = T._nc_logm(zr, m0, s0)                               # log M on the EMA of session 4
    assert np.abs(a - linem.log_m(zr, m0, s0)).max() <= 1e-12
    assert T.NOCAL['span'] == linem_s4.SPAN and T.NOCAL['warm'] == linem_s4.WARM and T.NOCAL['s_lo'] == linem.S_LO


def test_check_functions_match_research(research):
    """mri.legacy.linem_check (curve, ceilings, summary, rulers, cells) == research _nc_* on synthetic data of two seeds."""
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
        cm = LC.cells(D, Rm, zk, 500, linem_s4.WARM, linem.S_LO, null_chart=null)
        assert all(np.abs(x[2] - y[2]).max() <= 1e-12 and np.abs(x[3] - y[3]).max() <= 1e-12 for x, y in zip(cr, cm))
        sr, sm = T._nc_summary(cr), LC.summary(cm)
        assert sr['ceil'] == sm['ceil'] and sr['share'] == sm['share'] and sr['n'] == sm['n']
        for a, b in zip(sr['rows'], sm['rows']):
            assert (a['n_real'], a['n_surr'], a['sparse']) == (b['n_real'], b['n_surr'], b['sparse'])
            assert (np.isnan(a['ratio']) and np.isnan(b['ratio'])) or abs(a['ratio'] - b['ratio']) <= 1e-12
        assert all(sr['assets'][a][c] == sm['assets'][a][c] for a in sm['assets'] for c in ('orange', 'blue'))
    assert LC.EDGES == T.NC_EDGES and LC.TOL == tuple(T.NOCAL['tol']) and LC.MIN_N == T.NOCAL['min_n']
    assert LC.HALF_CUT == T.NOCAL['half_cut'] and LC.CEIL_ASSET == T.NOCAL['ceil_asset']
