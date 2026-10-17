"""scripts/parity.py for PB (R10 addendum 3) on a synthetic 'Export chart data' CSV made from the reference itself (values
rounded to 10 decimals, time in UNIX seconds, the visible 'M' plot and the 'dbg' plots): passes; a logit off by 2e-6, z off
by 2e-6, log M off by 2e-5, a wrong window number, a missing bar make it fail; the unconfirmed last bar is excluded only if
'dbg confirmed' did not advance; 'Compute only the last N bars' (the live part from the first computed window); the
coarse-tick symbol (exact ties) passes too; first line bar 552 (R16)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parity  # noqa: E402
import synth  # noqa: E402
from mri.export_pb import EXPORT, load_export  # noqa: E402


@pytest.fixture(scope='module')
def ex():
    return load_export(os.path.join(ROOT, EXPORT))


def fake(ex, b, symbol, path, calc_bars=0):
    ref = parity.reference_pb(b, ex, symbol, calc_bars)
    out = pd.DataFrame({'time': b['time'] // 1000, 'open': b['open'], 'high': b['high'], 'low': b['low'],
                        'close': b['close'], 'M': ref['M']})
    for c in ['logit', 'z', 'm_ema', 's_ema', 'logM', 'n'] + parity.INPUTS_PB + ['sgn']:
        out['dbg ' + c] = ref[c]
    out['dbg confirmed'] = np.arange(1, len(b) + 1)
    out['dbg bar_index'] = np.arange(len(b))
    out.to_csv(path, index=False, float_format='%.10f', na_rep='NaN')
    return path


@pytest.fixture(scope='module')
def fake_csv(ex, tmp_path_factory):
    b = synth.bars(1500, seed=21, gaps=[(1200, 2)])
    b[['open', 'high', 'low', 'close']] = b[['open', 'high', 'low', 'close']].round(4)
    return fake(ex, b, 'BTCUSDT', tmp_path_factory.mktemp('tv') / 'chart.csv')


def run(path, ex, symbol='BTCUSDT', calc_bars=0):
    tv, dec = parity.load_tv(str(path))
    return parity.compare(tv, dec, parity.reference_pb(tv, ex, symbol, calc_bars), parity.INPUTS_PB, parity.LINE_PB)


def test_parity_passes_on_reference_export(fake_csv, ex):
    res = run(fake_csv, ex)
    assert res['pass_R10'], res
    assert res['logit']['compared'] > 900 and res['first_line_bar'] == {'pine': 552, 'reference': 552}
    assert res['first_window_bar']['pine'] == 264 and res['n']['compared'] == res['logit']['compared']
    assert all(res[c]['over_tol'] == 0 for c in ('z', 'm_ema', 's_ema', 'logM'))


def test_parity_catches_errors(fake_csv, ex, tmp_path):
    df = pd.read_csv(fake_csv)
    i = int(np.argmax(np.isfinite(df['dbg logit'])) + 400)
    for col, eps in (('dbg logit', 2e-6), ('dbg z', 2e-6), ('dbg s_ema', 2e-6), ('dbg logM', 2e-5), ('dbg n', 1.0),
                     ('dbg side_close_dev', 1e-6)):
        bad = df.copy()
        bad.loc[i, col] += eps
        bad.to_csv(tmp_path / 'a.csv', index=False, float_format='%.10f', na_rep='NaN')
        assert not run(tmp_path / 'a.csv', ex)['pass_R10'], col
    df.drop(index=10).to_csv(tmp_path / 'b.csv', index=False, float_format='%.10f', na_rep='NaN')
    assert not run(tmp_path / 'b.csv', ex)['pass_R10']
    late = df.copy()                                                  # the line starts one window late
    late.loc[int(np.argmax(np.isfinite(df['M']))), 'M'] = np.nan
    late.to_csv(tmp_path / 'e.csv', index=False, float_format='%.10f', na_rep='NaN')
    assert not run(tmp_path / 'e.csv', ex)['pass_R10']


def test_live_last_bar_by_confirmed_counter(fake_csv, ex, tmp_path):
    df = pd.read_csv(fake_csv)
    live = df.copy()
    n = len(live) - 1
    keep = ('dbg norm', 'dbg r', 'dbg step_hl2', 'dbg step_hlc3', 'dbg g', 'dbg bar_index')
    for c in [c for c in live.columns if (c.startswith('dbg ') and c not in keep) or c == 'M']:
        live.loc[n, c] = np.nan                                         # forming bar: no window values
    live.loc[n, 'dbg confirmed'] = live.loc[n - 1, 'dbg confirmed']     # the state did not advance
    live.to_csv(tmp_path / 'c.csv', index=False, float_format='%.10f', na_rep='NaN')
    res = run(tmp_path / 'c.csv', ex)
    assert res['live_last_bar_excluded'] and res['pass_R10']
    gone = live.copy()
    gone.loc[n, 'dbg confirmed'] = live.loc[n - 1, 'dbg confirmed'] + 1
    gone.to_csv(tmp_path / 'd.csv', index=False, float_format='%.10f', na_rep='NaN')
    res = run(tmp_path / 'd.csv', ex)
    assert not res['live_last_bar_excluded'] and not res['pass_R10']


def test_calc_bars_live_part_from_first_computed_window(ex, tmp_path):
    b = synth.bars(1500, seed=23)
    path = fake(ex, b, 'BTCUSDT', tmp_path / 'n.csv', calc_bars=800)
    res = run(path, ex, calc_bars=800)
    first_w = 1500 - 1 - 800 - 64 + 63
    assert res['pass_R10'] and res['first_window_bar']['pine'] == first_w and res['first_line_bar']['pine'] == first_w + 288
    assert not run(path, ex, calc_bars=0)['pass_R10']                  # the full-history reference differs there


def test_coarse_tick_symbol(ex, tmp_path):
    b = synth.bars(1000, seed=2, tick=0.0001)
    res = run(fake(ex, b, 'XLMUSDT', tmp_path / 'x.csv'), ex, 'XLMUSDT')
    assert res['pass_R10'] and res['sgn']['ties_abs_z_lt_1e-9'] > 0
