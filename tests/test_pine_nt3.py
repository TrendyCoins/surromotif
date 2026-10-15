"""NT3 Pine template: its line-by-line transliteration (tests/pine_sim_nt3.py) == the reference stream (mri.nt3), with
and without 'compute only the last N bars'; the generated prototype holds the prod a weights bit for bit; project rules on
wording, colors, literals; history offsets of the path chunks."""
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pine_sim_nt3  # noqa: E402
import synth  # noqa: E402
from mri import L, nt3  # noqa: E402
from mri.pine_gen import ALPHA0, decode_dense  # noqa: E402
from mri.pine_nt3 import chunk_offsets, flat_values, make_script  # noqa: E402

FLOOR = 0.2


@pytest.mark.parametrize('calc_bars', [0, 500])
def test_pine_template_logic_equals_stream(calc_bars):
    b = synth.bars(1300, seed=5, gaps=[(1000, 3)])
    p = nt3.random_params(seed=6)
    s = np.array([x['logit'] for x in nt3.run(b, p, FLOOR)])
    pz, side = pine_sim_nt3.run(b, p, FLOOR, calc_bars=calc_bars, signed=nt3.SIDE_SIGNED[2:])
    shown = np.isfinite(pz)
    assert shown.sum() > 300
    assert np.abs(pz[shown] - s[shown]).max() < 1e-10
    ref = {x['bar']: x['side_raw'] for x in nt3.run(b, p, FLOOR) if np.isfinite(x['logit'])}
    assert max(np.abs(side[i] - ref[i]).max() for i in side) < 1e-10
    heavy0 = 0 if calc_bars == 0 else len(b) - 1 - calc_bars - L
    hidden = np.where(np.isfinite(s) & ~shown)[0]
    assert (hidden < heavy0 + L - 1).all()
    if calc_bars == 0:
        assert int(np.argmax(shown)) == 264


def test_chunk_offsets():
    ch = chunk_offsets()
    assert [n for n, _, _ in ch] == list(nt3.NT_GEO)
    assert ch[-1][1] == L + 1 and ch[0][2] == L + 199          # bars s-2 .. s-200
    for (n, e, s) in ch:
        assert s - e + 1 == n
    assert all(ch[i][1] == ch[i + 1][2] + 1 for i in range(len(ch) - 1))


@pytest.fixture(scope='module')
def proto():
    p = nt3.random_params(seed=11)
    table = {'BTCUSDT': {'floor': 0.17, 'platt_a': 0.8, 'platt_b': -0.3, 'ceil_more': 8.0, 'ceil_less': 5.0}}
    unknown = {'floor': 0.2, 'platt_a': 0.81, 'platt_b': -0.33, 'ceil_more': 8.0, 'ceil_less': 2.0}
    passport = {'weights_source': 'random', 'data_end': '-', 'code_version': 'test'}
    return p, make_script(p, table, unknown, passport)


def test_weights_from_strings_are_exact(proto):
    p, script = proto
    strs = re.findall(r'decodeInto\(WT, ALPHA, "([^"]*)"\)', script)
    assert len(strs) >= 28 and all(len(s) <= 3999 for s in strs)
    got = np.array(decode_dense(''.join(strs)))
    assert np.array_equal(got, flat_values(p).astype(np.float64)) and len(got) == 36961
    alpha = re.search(r'str\.split\("([^"]{2048})", ""\)', script).group(1)
    assert [ord(c) - ALPHA0 for c in alpha] == list(range(2048))


def test_wording_colors_literals(proto):
    _, script = proto
    assert script.startswith('//@version=6\n// SPDX-License-Identifier: Apache-2.0\n// © TrendyCoins\n')
    code = re.sub(r'"[一-嗿]+"', '""', script)
    banned = r'\b(buy|sell|long|short|bull\w*|bear\w*|arrow\w*|triangle\w*|label_?up|label_?down)\b'
    assert not re.findall(banned, code, flags=re.I)
    assert not re.findall(r'color\.(red|green|lime|maroon|olive)', code)
    assert not re.findall(r'(?<![A-Za-z_])\d+\.?\d*[eE][-+]?\d+', code)
    assert 'barstate.isconfirmed' in code and 'request.' not in code
    body = re.sub(r'"[^"]*"', '""', re.sub(r'//.*', '', code))           # no clock or EMA in the computation
    assert not re.findall(r'ta\.(ema|rma|wma)|\b(hour|minute|dayofweek|dayofmonth|weekofyear|month|year)\b', body)
    assert [ln.strip() for ln in body.splitlines() if re.search(r'\btime\b', ln)] == [
        'bool  gap    = bar_index > 0 and (time - time[1]) > GAPMS']          # time: grid gaps only
    assert len(re.findall(r'^plot\(', code, flags=re.M)) == 30           # M + 29 debug plots (R10 addition 2)


def test_production_indicator_is_generated_from_export():
    from mri.export_nt3 import EXPORT, load_export
    from mri.params import from_dict
    from mri.pine_nt3 import make_indicator
    ex = load_export(os.path.join(ROOT, EXPORT))
    script = make_indicator(ex)
    with open(os.path.join(ROOT, 'pine/mri.pine'), encoding='utf-8') as fh:
        assert fh.read() == script
    strs = re.findall(r'decodeInto\(WT, ALPHA, "([^"]*)"\)', script)
    assert np.array_equal(np.array(decode_dense(''.join(strs))), flat_values(from_dict(ex['model'])).astype(np.float64))

    def table(fn):
        body = re.search(rf'{fn}\(\) =>\n(.*?)\n    m\n', script, flags=re.S).group(1)
        return {k: float(v) for k, v in re.findall(r'map\.put\(m, "([A-Z]+)", (-?[0-9.]+)\)', body)}
    cal, ceil = ex['calibration'], ex['display']['ceiling']
    assert table('buildFloors') == ex['floors']['floor_L64']
    assert table('buildPlattA') == {a: v['platt_a'] for a, v in cal['per_symbol'].items()}
    assert table('buildPlattB') == {a: v['platt_b'] for a, v in cal['per_symbol'].items()}
    assert table('buildCeilMore') == {a: v['more'] for a, v in ceil['by_symbol'].items()}
    assert table('buildCeilLess') == {a: v['less'] for a, v in ceil['by_symbol'].items()}
    unk = lambda name: float(re.search(rf'var float {name}\s*= known \? map\.get\(\w+, syminfo\.ticker\) : (-?[0-9.]+)',
                                       script).group(1))
    assert unk('FLOOR') == ex['floors']['unknown_symbol']
    assert unk('PLATT_A') == cal['unknown_symbol']['platt_a'] and unk('PLATT_B') == cal['unknown_symbol']['platt_b']
    assert unk('CEIL_O') == ceil['unknown_symbol']['more'] and unk('CEIL_B') == ceil['unknown_symbol']['less']
    assert 'indicator("Market Randomness Index", shorttitle = "MRI", overlay = false' in script
