"""scripts/parity.py on a synthetic 'Export chart data' CSV made from the reference itself (values rounded to 10
decimals, time in UNIX seconds): passes; a logit off by 2e-6, a shifted first bar or a missing bar make it fail."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'legacy'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # tests/: synth
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parity_f3g_nt3 as parity  # noqa: E402
import synth  # noqa: E402


@pytest.fixture(scope='module')
def ex():
    with open(os.path.join(ROOT, 'export/legacy/mri_model.json'), encoding='utf-8') as fh:
        return json.load(fh)


@pytest.fixture(scope='module')
def fake_csv(ex, tmp_path_factory):
    b = synth.bars(3600, seed=21, gaps=[(3400, 2)])
    b[['open', 'high', 'low', 'close']] = b[['open', 'high', 'low', 'close']].round(4)
    ref = parity.reference(b, ex, 'BTCUSDT')
    out = pd.DataFrame({'time': b['time'] // 1000, 'open': b['open'], 'high': b['high'], 'low': b['low'],
                        'close': b['close']})
    for c in parity.INPUTS + ['logit', 'M', 'sgn']:
        out['dbg ' + c] = ref[c]
    out['dbg bar_index'] = np.arange(len(b))
    path = tmp_path_factory.mktemp('tv') / 'chart.csv'
    out.to_csv(path, index=False, float_format='%.10f', na_rep='NaN')
    return path


def run(path, ex):
    tv, dec = parity.load_tv(str(path))
    return parity.compare(tv, dec, parity.reference(tv, ex, 'BTCUSDT'))


def test_parity_passes_on_reference_export(fake_csv, ex):
    res = run(fake_csv, ex)
    assert res['pass_R10'], res
    assert res['logit']['compared'] > 100 and res['first_line_bar']['pine'] == 3182


def test_parity_catches_errors(fake_csv, ex, tmp_path):
    df = pd.read_csv(fake_csv)
    i = int(np.argmax(np.isfinite(df['dbg logit'])) + 50)
    bad = df.copy()
    bad.loc[i, 'dbg logit'] += 2e-6
    bad.to_csv(tmp_path / 'a.csv', index=False, float_format='%.10f', na_rep='NaN')
    assert not run(tmp_path / 'a.csv', ex)['pass_R10']
    df.drop(index=10).to_csv(tmp_path / 'b.csv', index=False, float_format='%.10f', na_rep='NaN')
    assert not run(tmp_path / 'b.csv', ex)['pass_R10']


def test_live_last_bar_is_excluded_only_if_state_did_not_advance(fake_csv, ex, tmp_path):
    df = pd.read_csv(fake_csv)
    live = df.copy()
    live.loc[len(live) - 1, 'dbg logit'] = np.nan                       # forming bar: no value, state not advanced
    for c in ('dbg ema7', 'dbg ema1'):
        live.loc[len(live) - 1, c] = live.loc[len(live) - 2, c]
    live.to_csv(tmp_path / 'c.csv', index=False, float_format='%.10f', na_rep='NaN')
    res = run(tmp_path / 'c.csv', ex)
    assert res['live_last_bar_excluded'] and res['pass_R10']
    gone = df.copy()
    gone.loc[len(gone) - 1, 'dbg logit'] = np.nan                       # value missing, but the state did advance
    gone.to_csv(tmp_path / 'd.csv', index=False, float_format='%.10f', na_rep='NaN')
    res = run(tmp_path / 'd.csv', ex)
    assert not res['live_last_bar_excluded'] and not res['pass_R10']
