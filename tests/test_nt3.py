"""NT3 reference on synthetic bars: bar-by-bar stream == vectorized batch; first line bar 264 on continuous data (rule R13);
windows over grid gaps are not valid; the history inputs read nothing deeper than 200 bars before the window and nothing of
the bar right before it in the signed inputs; the copy of nt_segment == research original (when torch is installed)."""
import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth  # noqa: E402
from mri import L, nt3  # noqa: E402
from mri.features import bar_features  # noqa: E402

FLOOR = 0.2


@pytest.mark.parametrize('kind', ['fine', 'coarse', 'gaps'])
def test_stream_equals_batch(kind):
    b = {'fine': lambda: synth.bars(1400, seed=1),
         'coarse': lambda: synth.bars(1400, seed=2, tick=0.0001),
         'gaps': lambda: synth.bars(1400, seed=3, gaps=[(500, 1), (900, 5), (930, 2)])}[kind]()
    p = nt3.random_params(seed=4)
    s = np.array([x['logit'] for x in nt3.run(b, p, FLOOR)])
    r = nt3.infer(bar_features(b), nt3.history_features(b), p, FLOOR)
    assert np.array_equal(np.where(np.isfinite(s))[0], r['T'])
    assert np.abs(s[r['T']] - r['logit']).max() < 1e-10
    assert r['T'][0] == 264 or kind == 'gaps' and r['T'][0] >= 264
    if kind == 'gaps':
        for i in (500, 900, 930):
            assert not np.isin(np.arange(i, i + L), r['T']).any()
    if kind == 'coarse':                          # exact ties of the orientation get +1 (rule R1)
        z = r['x_end'] / r['D']
        assert (np.abs(z) < 1e-12).sum() > 0 and (r['sgn'][np.abs(z) < 1e-12] == 1.0).all()


def test_first_line_bar_is_264():
    b = synth.bars(400, seed=5)
    f = bar_features(b)
    assert int(np.argmax(f['ok'])) == 201
    r = nt3.infer(f, nt3.history_features(b), nt3.random_params(seed=6), FLOOR)
    assert r['T'][0] == 264 and r['T'][0] - (L - 1) == 201


def test_history_depth_and_bar_before_window():
    b = synth.bars(900, seed=7)
    h0 = nt3.history_features(b)
    j = 500
    pert = b.copy()
    pert.loc[j, ['open', 'high', 'low', 'close']] *= [1.01, 1.03, 0.97, 0.99]
    h1 = nt3.history_features(pert)
    changed = np.where(np.any(np.abs(h1 - h0) > 1e-9, axis=1))[0]   # (pandas rolling mean: ~1e-16 drift later)
    assert changed.min() == j + 1 and changed.max() == j + 200          # starts s with s - 200 <= j <= s - 1
    signed = [k - 2 for k, sg in enumerate(nt3.SIDE_SIGNED) if sg]
    s = j + 1                                                            # bar j is the bar right before the window
    inner = b.copy()
    inner.loc[j, 'close'] = 0.5 * (b.loc[j, 'high'] + b.loc[j, 'low'])   # close moves inside the bar: range unchanged
    h3 = nt3.history_features(inner)
    assert np.array_equal(h3[s, signed], h0[s, signed])                 # P, R1, R3 end at s - 2


def test_nt_segment_copy_matches_research():
    pytest.importorskip('torch')
    os.environ.setdefault('SMC_MODE', 'none')
    os.environ.setdefault('SMC_PROJECT_PATH', tempfile.mkdtemp() + '/')
    sys.path.insert(0, os.path.join(ROOT, 'research'))
    import sumo_class_train as T
    for kind, b in (('fine', synth.bars(1200, seed=8)), ('coarse', synth.bars(1200, seed=9, tick=0.01))):
        o, h, l, c = (b[k].to_numpy(np.float64) for k in ('open', 'high', 'low', 'close'))
        orig, sc0 = T.nt_segment(o, h, l, c, with_scales=True)
        copy, sc1 = nt3.nt_segment(o, h, l, c, with_scales=True)
        assert orig.dtype == np.float32 and np.array_equal(orig, copy.astype(np.float32)), kind
        assert all(np.array_equal(sc0[k], sc1[k], equal_nan=True) for k in sc0)
        assert list(T.SIDE_COLS['nt_PRV']) == list(nt3.SIDE_COLS)
        assert [T.NT_COL0 + r for r in nt3.HIST_ROWS] == list(nt3.SIDE_COLS[2:])
        assert [bool(T.NT_ORIENT[r]) for r in nt3.HIST_ROWS] == list(nt3.SIDE_SIGNED[2:])
