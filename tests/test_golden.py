"""Golden file (tests/golden/btc5m_fresh.csv.gz, written by the bar-by-bar stream) against the vectorized batch reference
recomputed from the export JSON and the golden OHLC; export weights against the research npz (float32 round trip)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mri.batch import infer  # noqa: E402
from mri.features import bar_features  # noqa: E402
from mri.params import from_dict, load_npz  # noqa: E402

EXPORT = os.path.join(ROOT, 'export/mri_model.json')
GOLDEN = os.path.join(ROOT, 'tests/golden/btc5m_fresh.csv.gz')


@pytest.fixture(scope='module')
def ex():
    with open(EXPORT, encoding='utf-8') as fh:
        return json.load(fh)


def test_export_weights_are_the_npz_float32(ex):
    npz = os.path.join(ROOT, 'artifacts/weights/prod_F3_g_L64_a_slim.npz')
    if not os.path.isfile(npz):
        pytest.skip('artifacts/ not available (public snapshot)')
    p0, _ = load_npz(npz)
    p1 = from_dict(ex['model'])
    for a, b in zip(p0.conv + [p0.head0_w, p0.head0_b, p0.head3_w, p0.head3_b, p0.side_mu, p0.side_sd],
                    p1.conv + [p1.head0_w, p1.head0_b, p1.head3_w, p1.head3_b, p1.side_mu, p1.side_sd]):
        assert np.array_equal(a.astype(np.float32), b.astype(np.float32))
    assert p1.n_params == ex['model']['n_params'] == 16897


def test_golden_matches_batch(ex):
    g = pd.read_csv(GOLDEN)
    bars = g[['time', 'open', 'high', 'low', 'close']].copy()
    f = bar_features(bars, ema_adjust=True)
    p = from_dict(ex['model'])
    r = infer(f, p, ex['floors']['floor_L64']['BTCUSDT'], eps=ex['tie_rule']['eps'])
    gl = g['logit'].to_numpy()
    fin = np.isfinite(gl)
    assert set(np.where(fin)[0]) <= set(r['T'])
    T = np.where(fin)[0]
    got = pd.Series(r['logit'], index=r['T'])[T].to_numpy()
    assert np.abs(got - gl[T]).max() < 1e-9 * max(1.0, np.abs(gl[T]).max())
    shown = g['shown'].to_numpy() > 0
    assert np.isfinite(gl[shown]).all() and int(np.argmax(shown)) == 3182
    a, b = ex['calibration']['platt_a'], ex['calibration']['platt_b']
    assert np.allclose(g['M'][shown], np.exp(np.abs(a * gl[shown] + b)), rtol=1e-10)
    assert np.abs(g['norm'].to_numpy()[fin] - f['norm'][fin]).max() < 1e-9 * np.nanmax(f['norm'])
