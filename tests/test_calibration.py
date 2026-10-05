"""Copies in mri.calibration (platt_fit, lift_ci, calib_ratio) against the research originals (research module imports
without side effects with SMC_MODE=none); Platt recovers known parameters on synthetic logits."""
import math
import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mri import calibration as C  # noqa: E402


@pytest.fixture(scope='module')
def research():
    pytest.importorskip('torch')
    os.environ.setdefault('SMC_MODE', 'none')
    os.environ.setdefault('SMC_PROJECT_PATH', tempfile.mkdtemp() + '/')
    sys.path.insert(0, os.path.join(ROOT, 'research'))
    import sumo_class_train as T
    return T


def synth_logits(n=40000, a=0.8, b=-0.2, seed=0):
    g = np.random.default_rng(seed)
    s = g.normal(0.3, 1.5, 2 * n)
    y = g.random(2 * n) < 1 / (1 + np.exp(-(a * s + b)))
    return s[y], s[~y]


def test_platt_recovers_parameters():
    sp, sn = synth_logits()
    # balance-free check: with sampling by label the fitted slope is the generating one
    a, b = C.platt_fit(sp, sn)
    assert abs(a - 0.8) < 0.03


def test_copies_match_research(research):
    T = research
    sp, sn = synth_logits(seed=1)
    assert C.platt_fit(sp, sn) == T.platt_fit(sp, sn)
    g = np.random.default_rng(2)
    fp, fn = (g.random(5000) < 0.3).astype(float), (g.random(5000) < 0.2).astype(float)
    wk = np.sort(g.integers(0, 50, 5000))
    assert C.lift_ci(fp, fn, wk, wk) == T.lift_ci(fp, fn, wk, wk)
    assert math.isclose(C.calib_ratio((0.8, -0.2), (0.82, -0.23), 1.25), T._calib_ratio((0.8, -0.2), (0.82, -0.23), 1.25))
