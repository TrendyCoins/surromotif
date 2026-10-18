"""PB golden file (tests/golden/btc5m_fresh_pb.csv.gz, written by the bar-by-bar stream) against the vectorized reference
recomputed from the export JSON and the golden OHLC (< 1e-9); export weights against the research npz (exactly the float32
values, separable layers included); session 5: one floor FLOOR (R17), the ruler = export.ruler of the Ceil json (R19),
formula A and the start of the line (R18), one ceiling 10, no constants of the symbol; first window 264 (R13), first line
bar 4360 (R18)."""
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


def test_export_line_floor_ceiling(ex):
    ln = ex['line']
    q, zk, knots = ln['ruler']['q'], ln['ruler']['zk'], ln['ruler']['knots']
    assert len(q) == len(zk) == len(knots) == 37 and np.all(np.diff(q) > 0) and ln['ruler']['kind'] == 'glob_e'
    ceil_json = os.path.join(ROOT, 'artifacts/calib/ceil_20261018_115426.json')
    if os.path.isfile(ceil_json):
        with open(ceil_json, encoding='utf-8') as fh:
            cj = json.load(fh)
        assert q == cj['export']['ruler']['glob_e']['q'] and ex['line']['ceiling']['value'] == cj['export']['ceiling'] == 10
        nocal = sorted(glob.glob(os.path.join(ROOT, 'artifacts/calib/nocal_PB_s64h24_d1_*.json')))[-1]
        with open(nocal, encoding='utf-8') as fh:
            nc = json.load(fh)
        assert zk == nc['zk'] and knots == nc['knots']
    with open(os.path.join(ROOT, 'docs/checks/pb5/ceil_repro.json'), encoding='utf-8') as fh:
        rep = json.load(fh)
    assert rep['reproduced'] and rep['ruler']['bit_exact']
    assert ex['floor']['value'] == 0.20000000298023224 == pb.FLOOR and ex['format_version'] == 4
    assert 'floors' not in ex and 'unknown_symbol' not in ex and 'ceilings' not in ln and 'settings' not in ln
    lv = ln['live']
    assert (lv['L'], lv['n0'], lv['lag'], lv['s_lo']) == (4032, 144, 64, 0.75)
    assert ln['show']['from_window'] == 4096 == linem.SHOW_FROM and ln['show']['first_line_bar_continuous'] == 4360
    assert ln['ceiling'] == {**ln['ceiling'], 'value': 10.0, 'both_colours': True} and 'label' not in ln['ceiling']
    assert ex['warmup']['first_line_bar_continuous'] == FIRST_LINE_BAR == 4360 and FIRST_WINDOW_END == 264
    txt = json.dumps(ex)
    assert 'drive' not in txt.lower() and 'span_default' not in txt and 'windows_default' not in txt
    assert 'SMC' not in txt


def test_golden_matches_vectorized(ex):
    g = pd.read_csv(os.path.join(ROOT, GOLDEN))
    bars = g[['time', 'open', 'high', 'low', 'close']].copy()
    p, q, zk = params_line(ex)
    v = pb.chart(bars, p, ex['floor']['value'], q, zk, f=bar_features(bars))
    gl = g['logit'].to_numpy()
    fin = np.isfinite(gl)
    assert np.array_equal(fin, v['valid'])
    tol = lambda x: 1e-9 * max(1.0, np.abs(x).max())
    assert np.abs(v['logit'][fin] - gl[fin]).max() < tol(gl[fin])
    side = g[[f'side_{n}' for n in nt3.SIDE_NAMES]].to_numpy()[fin]
    assert np.abs(v['res']['side_raw'] - side).max() < tol(side)
    for k in ('z', 'm_win', 's_win', 'log_m'):
        x = g[k].to_numpy()[fin]
        assert np.abs(v[k][fin] - x).max() < tol(x), k
    assert np.array_equal(g['n'].to_numpy(), v['n']) and np.array_equal(g['k'].to_numpy(), v['k'])
    shown = g['shown'].to_numpy() > 0
    assert np.array_equal(shown, v['shown'])
    assert int(np.argmax(fin)) == FIRST_WINDOW_END and int(np.argmax(shown)) == FIRST_LINE_BAR
    assert np.array_equal(shown, fin & (g['n'].to_numpy() >= linem.SHOW_FROM))
    D = g['D'].to_numpy()[fin]
    at = np.abs(D - pb.FLOOR) < 1e-10                                       # golden: 11 significant digits
    assert D.min() >= pb.FLOOR - 1e-10 and at.sum() == 273                   # the floor acts on 273 windows (as in R2)
    assert os.path.getsize(os.path.join(ROOT, GOLDEN)) < 5.2e6
