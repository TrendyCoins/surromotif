"""NT3 golden file (tests/golden/btc5m_fresh_nt3.csv.gz, written by the bar-by-bar stream) against the vectorized
reference recomputed from the export JSON and the golden OHLC; export weights against the research npz (exactly the
float32 values); first line bar 264 (rule R13); M with the Platt of BTC."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mri import nt3  # noqa: E402
from mri.export_nt3 import EXPORT, FIRST_LINE_BAR, GOLDEN, load_export  # noqa: E402
from mri.features import bar_features  # noqa: E402
from mri.params import from_dict, load_npz  # noqa: E402


@pytest.fixture(scope='module')
def ex():
    return load_export(os.path.join(ROOT, EXPORT))


def test_export_weights_are_the_npz_float32(ex):
    npz = os.path.join(ROOT, 'artifacts/weights/prod_NT3_PRV_ch48_L64_a_slim.npz')
    if not os.path.isfile(npz):
        pytest.skip('artifacts/ not available (public snapshot)')
    p0, _ = load_npz(npz)
    p1 = from_dict(ex['model'])
    for a, b in zip(p0.conv + [p0.head0_w, p0.head0_b, p0.head3_w, p0.head3_b, p0.side_mu, p0.side_sd],
                    p1.conv + [p1.head0_w, p1.head0_b, p1.head3_w, p1.head3_b, p1.side_mu, p1.side_sd]):
        assert np.array_equal(a, b)
    assert p1.n_params == ex['model']['n_params'] == 36961 and p0.lvl_mu == 0.0 == ex['model']['lvl_mu']
    assert ex['model']['side_cols_research'] == list(nt3.SIDE_COLS) and ex['model']['side_names'] == list(nt3.SIDE_NAMES)


def test_export_tables(ex):
    assert len(ex['calibration']['per_symbol']) == 31 == len(ex['floors']['floor_L64'])
    pool = ex['display']['ceiling']['pool']
    for a, c in ex['display']['ceiling']['by_symbol'].items():
        assert c['more'] <= pool['more'] and c['less'] <= pool['less']
    assert ex['warmup']['first_line_bar_continuous'] == FIRST_LINE_BAR == 264


def test_golden_matches_batch(ex):
    g = pd.read_csv(os.path.join(ROOT, GOLDEN))
    bars = g[['time', 'open', 'high', 'low', 'close']].copy()
    f = bar_features(bars)
    p = from_dict(ex['model'])
    r = nt3.infer(f, nt3.history_features(bars), p, ex['floors']['floor_L64']['BTCUSDT'], eps=ex['tie_rule']['eps'])
    gl = g['logit'].to_numpy()
    fin = np.isfinite(gl)
    assert np.array_equal(np.where(fin)[0], r['T'])
    assert np.abs(r['logit'] - gl[fin]).max() < 1e-9 * max(1.0, np.abs(gl[fin]).max())
    side = g[[f'side_{n}' for n in nt3.SIDE_NAMES]].to_numpy()[fin]
    assert np.abs(r['side_raw'] - side).max() < 1e-9 * max(1.0, np.abs(side).max())
    shown = g['shown'].to_numpy() > 0
    assert np.array_equal(shown, fin) and int(np.argmax(shown)) == FIRST_LINE_BAR
    pl = ex['calibration']['per_symbol']['BTCUSDT']
    assert np.allclose(g['M'][shown], np.exp(np.abs(pl['platt_a'] * gl[shown] + pl['platt_b'])), rtol=1e-10)
