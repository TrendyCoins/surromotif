"""Export of F1 on 1H (export/f1h_model.json, rule R21): the constants f1h runs with (floor = float32(0.2), ceiling 8, live
part 4032·k0.5, show from the window 4096, first window / line candle 264 / 4360); with artifacts/: the weights are the
float32 of the slim npz prod a bit for bit, the line is the section 'export' of the tfline21 json field for field, the
sources are those of artifacts/pack/site1h_manifest.json (sha256)."""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import f1h  # noqa: E402

ART = os.path.join(ROOT, 'artifacts')


@pytest.fixture(scope='module')
def ex():
    with open(f1h.model.EXPORT, encoding='utf-8') as fh:
        return json.load(fh)


def test_constants(ex):
    m = f1h.from_export(ex)
    assert m.floor == float(np.float32(0.2)) == 0.20000000298023224 and m.orient_eps == 1e-4
    assert (m.live_l, m.live_lag, m.live_k, m.live_tau, m.s_lo, m.show_from, m.ceiling) == (4032, 64, 0.5, 64, 0.75, 4096, 8.0)
    assert (m.bar_ms, m.gap_factor, m.n_norm, m.n_hist, m.window) == (3_600_000, 1.5, 200, 200, 64)
    assert (m.first_window_bar, m.first_line_bar) == (264, 4360) and m.dilations == (1, 2, 4, 8)
    assert [w.shape for w in m.conv_w] == [(64, 4, 5)] + [(64, 64, 5)] * 3 and m.head0_w.shape == (128, 209)
    assert len(m.q) == 37 and np.all(np.diff(m.q) > 0) and len(m.zk) == 37
    n = sum(w.size for w in m.conv_w + m.conv_b) + m.head0_w.size + m.head0_b.size + m.head3_w.size + m.head3_b.size
    assert n == 89985 == ex['passport']['n_weights'] and m.side_signed.sum() == 10
    for w in m.conv_w + [m.head0_w, m.side_mu]:          # exact float32 values
        assert np.array_equal(w.astype(np.float32).astype(np.float64), w)
    assert 'marks' not in ex['line'] and 'shades' not in ex['line']


def _sha(p):
    with open(p, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def test_against_artifacts(ex):
    npz = os.path.join(ART, 'weights', 'prod_SP_l025_d1_L64_a_slim.npz')
    if not os.path.isfile(npz):
        pytest.skip('artifacts/ not available')
    m = f1h.from_export(ex)
    with np.load(npz) as z:
        sd = {k[3:]: z[k] for k in z.files}
    pairs = [(m.conv_w[i], sd[f'blocks.{i}.0.weight']) for i in range(4)] + \
            [(m.conv_b[i], sd[f'blocks.{i}.0.bias']) for i in range(4)] + \
            [(m.head0_w, sd['head.0.weight']), (m.head0_b, sd['head.0.bias']), (m.head3_w, sd['head.3.weight']),
             (m.head3_b, sd['head.3.bias']), (m.side_mu, sd['side_mu']), (m.side_sd, sd['side_sd']),
             (np.atleast_1d(m.s_sd), np.atleast_1d(sd['s_sd'])), (np.atleast_1d(m.lvl_mu), sd['lvl_mu']),
             (np.atleast_1d(m.lvl_sd), sd['lvl_sd'])]
    assert len(sd) == 17 and all(a.shape == b.shape and np.array_equal(a.astype(np.float32), b) for a, b in pairs)
    with open(os.path.join(ART, 'calib', 'tfline21_20261022_063520.json'), encoding='utf-8') as fh:
        te = json.load(fh)['export']
    ru, lv = ex['line']['ruler'], ex['line']['live']
    assert all(ru[k] == te['ruler'][k] for k in ('kind', 'variant', 'knots', 'zk', 'q', 'n_null'))
    assert all(lv[k] == te['live'][k] for k in ('key', 'family', 'L', 'lag', 'k', 'tau', 's_lo'))
    assert ex['line']['show_from_window'] == te['live']['show_from_window'] and ex['line']['ceiling'] == te['ceiling']
    assert np.float32(ex['window']['floor']) == np.float32(te['floor'])
    assert all(ex['history'][k] == te['history'][k] for k in ('first_window_bar', 'first_line_bar'))
    with open(os.path.join(ART, 'pack', 'site1h_manifest.json'), encoding='utf-8') as fh:
        man = {f['path']: f['sha256'] for f in json.load(fh)['files']}
    for p, sha in ex['passport']['sources'].items():
        assert man[p] == sha == _sha(os.path.join(ROOT, p))
