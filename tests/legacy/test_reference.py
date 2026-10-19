"""Reference on synthetic bars: bar-by-bar stream == vectorized batch == transliteration of the Pine template;
tie rule; parameter shapes."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repo: mri
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # tests/: synth
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pine_sim  # noqa: E402
import synth  # noqa: E402
from mri import L  # noqa: E402
from mri.batch import infer  # noqa: E402
from mri.features import bar_features  # noqa: E402
from mri.params import random_params  # noqa: E402
from mri.legacy.pine_gen import decode_dense, encode_dense, flat_values, flat_weights, fmt  # noqa: E402
from mri.legacy.stream import run  # noqa: E402

FLOOR = 0.2


def stream_logits(b, p, **kw):
    return np.array([x['logit'] for x in run(b, p, FLOOR, **kw)])


@pytest.mark.parametrize('adjust', [False, True])
@pytest.mark.parametrize('kind', ['fine', 'coarse', 'gaps'])
def test_stream_equals_batch(kind, adjust):
    b = {'fine': lambda: synth.bars(3000, seed=1),
         'coarse': lambda: synth.bars(3000, seed=2, tick=0.0001),
         'gaps': lambda: synth.bars(3000, seed=3, gaps=[(900, 1), (1500, 5), (1530, 2)])}[kind]()
    p = random_params(seed=4)
    s = stream_logits(b, p, ema_adjust=adjust)
    f = bar_features(b, ema_adjust=adjust)
    r = infer(f, p, FLOOR)
    # early windows: the stream has no EMA before the first nonzero |r| (research backfills it - a look-ahead);
    # they form a prefix of the valid windows and are hidden by the warm-up anyway
    fin = np.isfinite(s[r['T']])
    k = int(np.argmax(fin))
    assert fin[k:].all() and k <= 0.05 * len(fin)
    assert np.abs(s[r['T'][k:]] - r['logit'][k:]).max() < 1e-10
    assert np.isnan(s[np.setdiff1d(np.arange(len(b)), r['T'])]).all()   # no logit on invalid windows
    if kind == 'gaps':                  # windows over a gap are not valid
        t = b['time'].to_numpy()
        for i in (900, 1500, 1530):
            assert not np.isin(np.arange(i, i + L), r['T']).any()
            assert t[i] - t[i - 1] > 300_000


@pytest.mark.parametrize('calc_bars', [0, 700])
def test_pine_template_logic_equals_stream(calc_bars):
    b = synth.bars(2900, seed=5, gaps=[(2500, 3)])
    p = random_params(seed=6)
    burn = 2100
    s = stream_logits(b, p, ema_adjust=True)
    pz = pine_sim.run(b, p, FLOOR, burn=burn, calc_bars=calc_bars)
    shown = np.isfinite(pz)
    assert shown.sum() > 300
    assert np.abs(pz[shown] - s[shown]).max() < 1e-10
    # hidden while the stream has a value: only the warm-up (BURN) and the start of the heavy part (63 bars)
    first_r = int(np.argmax(np.isfinite(bar_features(b)['r'])))
    heavy0 = 0 if calc_bars == 0 else len(b) - 1 - calc_bars - L
    hidden = np.where(np.isfinite(s) & ~shown)[0]
    assert ((hidden < first_r + burn) | (hidden < heavy0 + L - 1)).all()


def test_tie_rule_coarse_tick():
    b = synth.bars(4000, seed=7, tick=0.0001)
    p = random_params(seed=8)
    r = infer(bar_features(b), p, FLOOR)
    z = r['x_end'] / r['D']
    ties = np.abs(z) < 1e-9
    assert ties.sum() > 0                          # coarse tick: exact ties happen
    assert (r['sgn'][ties] == 1.0).all()           # rule R1: ties -> +1
    assert (r['sgn'][z < -1e-3] == -1.0).all()
    assert (r['sgn'][z > 0] == 1.0).all()


def test_shapes_and_literals():
    p = random_params(seed=9)
    assert p.n_params == 16897
    vals = flat_weights(p)
    assert len(vals) == 16897
    assert all('.' in v and 'e' not in v.lower() for v in vals)
    x = np.float32(np.random.default_rng(0).normal(0, 1, 1000)) * np.float32(1e-3)
    assert all(np.float32(float(fmt(v))) == v for v in x)    # decimal literal round-trips float32


def test_dense_weight_strings_are_exact():
    p = random_params(seed=10)
    v = flat_values(p)
    special = np.array([0.0, -0.0, 1e-40, -3e-39, 1.0, -2.0, 3.4e38, -1.1754944e-38, 0.1], dtype=np.float32)
    for x in (v, special):
        s = encode_dense(x)
        assert len(s) == 3 * len(x) and all(0x4E00 <= ord(c) < 0x4E00 + 2048 for c in s)
        assert np.array_equal(np.array(decode_dense(s)), x.astype(np.float64))
