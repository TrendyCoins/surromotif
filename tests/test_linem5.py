"""Formula A of the live part (rule R18) on synthetic z — thresholds of session 5 (docs/rules.md): three computations —
the stream as the Pine indicator (mri.linem.LiveWin: ring and running sums), vectorized (mri.linem.live_win: cumulative
sums) and research SuMo_Class_Live chart_ms (kind 'win', prior 'null', n0 144, L 4032, lag 64) + _nc_logm of the training
code: m_win, s_win <= 1e-10, log M <= 1e-9, k and the start of the line (research options_b B3) exactly. Sets: fine and
coarse z, missing windows, series shorter than 4096 and longer than 20 000 windows."""
import math

import numpy as np
import pytest

from mri import linem, research5

G = np.random.default_rng(5)
Q = np.sort(G.normal(-1.0, 1.4, 37)) + np.arange(37) * 1e-3
ZK = np.sort(G.normal(0, 1.2, 37))


@pytest.fixture(scope='module')
def chain():
    return research5.chain()


def series(kind):
    if kind == 'fine_long':
        return G.normal(0.4, 1.1, 30000) + 0.3 * np.sin(np.arange(30000) / 900.0)
    if kind == 'coarse':                                   # few distinct values (grid of a coarse tick)
        return np.round(G.normal(0.2, 1.0, 12000) * 2) / 2
    if kind == 'short':
        return G.normal(0.8, 0.7, 3000)
    if kind == 'exact_4096':
        return G.normal(0.1, 1.3, 4097)
    raise KeyError(kind)


def stream(z):
    lw = linem.LiveWin(Q, ZK)
    out = np.array([lw.step(x) for x in z])
    return out[:, 0], out[:, 1], out[:, 2].astype(np.int64)


@pytest.mark.parametrize('kind', ['fine_long', 'coarse', 'short', 'exact_4096'])
def test_formula_a_three_ways(chain, kind):
    T, BB, LV, CE = chain
    z = series(kind)
    v = {x['key']: x for x in LV.variants_a()}['4032+144']
    assert (v['kind'], v['prior'], v['n0'], v['L'], v['lag']) == ('win', 'null', linem.N0, linem.L_WIN, linem.LAG)
    show = [o for o in LV.options_b(v) if o['opt'] == 'B3'][0]['show']
    assert show == linem.SHOW_FROM == 4096
    r = LV.chart_ms(T, z, [(0, len(z))], v, {})
    assert np.array_equal(r['i'], np.arange(len(z))) and np.array_equal(r['p'], np.arange(len(z)))
    mv, sv, kv = linem.live_win(z)
    ms, ss, ks = stream(z)
    for m_, s_, k_ in ((mv, sv, kv), (ms, ss, ks)):
        assert np.abs(m_ - r['m']).max() <= 1e-10 and np.abs(s_ - r['s']).max() <= 1e-10
        assert np.array_equal(k_, r['n'])
    lr = T._nc_logm(z, r['m'], r['s'])
    assert np.abs(linem.log_m(z, mv, sv) - lr).max() <= 1e-9
    assert np.abs(np.array([linem.log_m1(*x) for x in zip(z, ms, ss)]) - lr).max() <= 1e-9
    n = np.arange(len(z))
    assert np.array_equal(n >= linem.SHOW_FROM, r['p'] >= show)
    assert np.array_equal(kv, np.minimum(np.maximum(n - 63, 0), 4032))
    print(kind, np.abs(ms - r['m']).max(), np.abs(ss - r['s']).max())


def test_set_is_the_last_4032_windows_ending_64_before():
    """Direct definition of the set: windows max(0, n - 4095) .. n - 64; plus 144 virtual random windows."""
    z = G.normal(0.3, 1.0, 9000)
    ms, ss, ks = stream(z)
    for n in (0, 1, 63, 64, 65, 200, 4095, 4096, 4097, 5000, 8999):
        w = z[max(0, n - 4095):max(n - 63, 0)]
        k = len(w)
        assert ks[n] == k == min(max(n - 63, 0), 4032)
        mh = w.mean() if k else 0.0
        vh = max((w * w).mean() - mh * mh, 0.0) if k else 0.0
        assert abs(ms[n] - w.sum() / (k + 144)) <= 1e-12
        assert abs(ss[n] - math.sqrt((k * vh + 144) / (k + 144))) <= 1e-12
    assert ms[0] == 0.0 and ss[0] == 1.0                    # empty set: only the virtual random windows


def test_missing_windows_do_not_move_the_state():
    """Invalid windows get no number and do not change the state: the line over a chart with holes = formula A over the
       valid windows only; the stream of the indicator (mri.pb) is checked on bars in tests/test_pb.py."""
    n = 6000
    logit = G.normal(-0.5, 1.5, n)
    valid = np.ones(n, dtype=bool)
    valid[[10, 11, 12, 700, 4100, 4101]] = False
    valid[2000:2064] = False
    out = linem.line_win(logit, valid, Q, ZK)
    idx = np.where(valid)[0]
    z = linem.ruler_z(logit[idx], Q, ZK)
    m, s, k = linem.live_win(z)
    assert np.array_equal(out['n'][idx], np.arange(len(idx))) and (out['n'][~valid] == -1).all()
    assert np.array_equal(out['m_win'][idx], m) and np.array_equal(out['k'][idx], k)
    assert np.isnan(out['log_m'][~valid]).all() and not out['shown'][~valid].any()
    assert np.where(out['shown'])[0][0] == idx[4096]
    lw = linem.LiveWin(Q, ZK)
    rows = [lw.update(x) for x in logit[idx]]
    assert np.abs(np.array([r['m_win'] for r in rows]) - m).max() <= 1e-10
    assert np.abs(np.array([r['log_m'] for r in rows]) - out['log_m'][idx]).max() <= 1e-9
    assert [r['shown'] for r in rows] == list(out['shown'][idx])


def test_display_one_ceiling():
    v, col, cap = linem.display(np.log([0.5, 1.0, 2.0, 9.99, 9.9999, 50.0, 1 / 50.0, 1 / 10.5]))
    assert list(col) == [-1, 0, 1, 1, 1, 1, -1, -1]
    assert list(cap) == [False, False, False, False, False, True, True, True]
    assert np.allclose(v, [2.0, 1.0, 2.0, 9.99, 9.9999, 10.0, 10.0, 10.0])
    assert linem.CEIL == 10.0


def test_random_market_gives_one():
    z = np.linspace(-4, 4, 801)
    assert np.abs(linem.log_m(z, np.zeros_like(z), np.ones_like(z))).max() < 1e-15
