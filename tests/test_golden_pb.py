"""PB golden file (tests/golden/btc5m_fresh_pb.csv.gz, written by the bar-by-bar stream) against the vectorized reference
recomputed from the export JSON and the golden OHLC (< 1e-9); export weights against the research npz (exactly the float32
values, separable layers included); the ruler = prod a of the nocal json; ceilings and floors = the checks; first window
264 (R13), first line bar 552 (R16)."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mri import linem, nt3, pb  # noqa: E402
from mri.export import POOL_ALL  # noqa: E402
from mri.export_pb import EXPORT, FIRST_LINE_BAR, FIRST_WINDOW_END, GOLDEN, load_export, params_line  # noqa: E402
from mri.features import bar_features  # noqa: E402
from mri.params import load_npz  # noqa: E402


@pytest.fixture(scope='module')
def ex():
    return load_export(os.path.join(ROOT, EXPORT))


def test_export_weights_are_the_npz_float32(ex):
    npz = os.path.join(ROOT, 'artifacts/weights/prod_PB_s64h24_d1_L64_a_slim.npz')
    if not os.path.isfile(npz):
        pytest.skip('artifacts/ not available (public snapshot)')
    p0, _ = load_npz(npz)
    p1, _, _ = params_line(ex)
    for a, b in zip(p0.conv + p0.pw + [p0.head0_w, p0.head0_b, p0.head3_w, p0.head3_b, p0.side_mu, p0.side_sd],
                    p1.conv + p1.pw + [p1.head0_w, p1.head0_b, p1.head3_w, p1.head3_b, p1.side_mu, p1.side_sd]):
        assert a.shape == b.shape and np.array_equal(a, b)
    assert (p0.s_sd, p0.lvl_sd) == (p1.s_sd, p1.lvl_sd)
    assert p1.n_params == ex['model']['n_params'] == 18697 and p0.lvl_mu == 0.0 == ex['model']['lvl_mu']
    assert [w.shape for w in p1.conv] == [(64, 4, 3)] + [(64, 1, 3)] * 3 and [w.shape for w in p1.pw] == [(64, 64)] * 3
    assert p1.head0_w.shape == (24, 209) and ex['model']['mult_per_bar'] == 32304
    assert ex['model']['side_cols_research'] == list(nt3.SIDE_COLS) and ex['model']['side_names'] == list(nt3.SIDE_NAMES)


def test_export_line_and_tables(ex):
    ln = ex['line']
    q, zk, knots = ln['ruler']['q'], ln['ruler']['zk'], ln['ruler']['knots']
    assert len(q) == len(zk) == len(knots) == 37 and np.all(np.diff(q) > 0) and ln['ruler']['kind'] == 'glob_e'
    nocal = sorted(glob.glob(os.path.join(ROOT, 'artifacts/calib/nocal_PB_s64h24_d1_*.json')))
    if nocal:
        with open(nocal[-1], encoding='utf-8') as fh:
            nc = json.load(fh)
        assert q == nc['prod']['a']['glob_e']['q'] and zk == nc['zk'] and knots == nc['knots']
    assert (ln['live']['span_default'], ln['live']['s_lo'], ln['warmup']['windows_default']) == (8064, 0.75, 288)
    c = ln['ceilings']
    assert c['pool'] == {'orange': 34.0, 'blue': 21.0} and c['unknown_symbol'] == {'orange': 8.0, 'blue': 8.0}
    assert sorted(c['by_symbol']) == sorted(POOL_ALL) == sorted(ex['floors']['floor_L64'])
    for v in c['by_symbol'].values():
        assert v['orange'] <= c['pool']['orange'] and v['blue'] <= c['pool']['blue']
    with open(os.path.join(ROOT, 'docs/checks/pb/ceilings.json'), encoding='utf-8') as fh:
        ce = json.load(fh)
    assert ce['reproduced'] and all(c['by_symbol'][a] == ce['by_symbol'][a] for a in POOL_ALL)
    fl = list(ex['floors']['floor_L64'].values())
    assert ex['floors']['unknown_symbol'] == float(np.median(fl)) and abs(ex['floors']['unknown_symbol'] - 0.2) < 0.02
    assert ex['warmup']['first_line_bar_continuous'] == FIRST_LINE_BAR == 552 and FIRST_WINDOW_END == 264
    assert 'drive' not in json.dumps(ex).lower()


def test_golden_matches_vectorized(ex):
    g = pd.read_csv(os.path.join(ROOT, GOLDEN))
    bars = g[['time', 'open', 'high', 'low', 'close']].copy()
    p, q, zk = params_line(ex)
    v = pb.chart(bars, p, ex['floors']['floor_L64']['BTCUSDT'], q, zk, f=bar_features(bars))
    gl = g['logit'].to_numpy()
    fin = np.isfinite(gl)
    assert np.array_equal(fin, v['valid'])
    tol = lambda x: 1e-9 * max(1.0, np.abs(x).max())
    assert np.abs(v['logit'][fin] - gl[fin]).max() < tol(gl[fin])
    side = g[[f'side_{n}' for n in nt3.SIDE_NAMES]].to_numpy()[fin]
    assert np.abs(v['res']['side_raw'] - side).max() < tol(side)
    for k in ('z', 'm_ema', 's_ema', 'log_m'):
        x = g[k].to_numpy()[fin]
        assert np.abs(v[k][fin] - x).max() < tol(x), k
    assert np.array_equal(g['n'].to_numpy(), v['n'])
    shown = g['shown'].to_numpy() > 0
    assert np.array_equal(shown, v['shown'])
    assert int(np.argmax(fin)) == FIRST_WINDOW_END and int(np.argmax(shown)) == FIRST_LINE_BAR
    assert np.array_equal(shown, fin & (g['n'].to_numpy() >= linem.WARM))
