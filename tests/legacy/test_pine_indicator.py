"""The legacy F3_g indicator (pine/legacy/mri_f3g.pine from export/mri_model.json): weights decoded from its strings are the export
weights bit for bit; floors, Platt, ceiling and warm-up are the export values; project rules on wording and colors;
Pine-safe literals. The arithmetic of the shared core is checked by tests/pine_sim.py (test_reference.py)."""
import json
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mri.params import from_dict  # noqa: E402
from mri.legacy.pine_gen import ALPHA0, decode_dense, flat_values, make_indicator  # noqa: E402


@pytest.fixture(scope='module')
def ex():
    with open(os.path.join(ROOT, 'export/legacy/mri_model.json'), encoding='utf-8') as fh:
        return json.load(fh)


@pytest.fixture(scope='module')
def script(ex):
    return make_indicator(ex)


def test_committed_script_is_generated_from_export(script):
    with open(os.path.join(ROOT, 'pine/legacy/mri_f3g.pine'), encoding='utf-8') as fh:
        assert fh.read() == script


def test_weights_from_strings_are_exact(ex, script):
    strs = re.findall(r'decodeInto\(WT, ALPHA, "([^"]*)"\)', script)
    assert len(strs) >= 10 and all(len(s) <= 3999 for s in strs)
    got = np.array(decode_dense(''.join(strs)))
    want = flat_values(from_dict(ex['model'])).astype(np.float64)
    assert np.array_equal(got, want)
    alpha = re.search(r'str\.split\("([^"]{2048})", ""\)', script).group(1)
    assert [ord(c) - ALPHA0 for c in alpha] == list(range(2048))


def table(script, fn):
    body = re.search(rf'{fn}\(\) =>\n(.*?)\n    m\n', script, flags=re.S).group(1)
    return {k: float(v) for k, v in re.findall(r'map\.put\(m, "([A-Z]+)", ([0-9.]+)\)', body)}


def test_constants_match_export(ex, script):
    assert table(script, 'buildFloors') == ex['floors']['floor_L64']
    ceil = ex['display']['ceiling']
    assert table(script, 'buildCeilMore') == {a: v['more'] for a, v in ceil['by_symbol'].items()}
    assert table(script, 'buildCeilLess') == {a: v['less'] for a, v in ceil['by_symbol'].items()}
    assert len(ceil['by_symbol']) == 31 and all(v['more'] <= 21 and v['less'] <= 21 for v in ceil['by_symbol'].values())
    num = lambda name: float(re.search(rf'const float {name}\s*=\s*(-?[0-9.]+)', script).group(1))
    assert num('PLATT_A') == ex['calibration']['platt_a'] and num('PLATT_B') == ex['calibration']['platt_b']
    unk = lambda name: float(re.search(rf'var float {name} = known \? map\.get\(\w+, syminfo\.ticker\) : ([0-9.]+)',
                                       script).group(1))
    assert unk('CEIL_O') == ceil['unknown_symbol']['more'] and unk('CEIL_B') == ceil['unknown_symbol']['less']
    assert unk('FLOOR') == ex['floors']['unknown_symbol']
    assert int(re.search(r'const int   BURN   = (\d+)', script).group(1)) == ex['warmup']['burn_bars_from_first_valid_r']


def test_header_wording_colors_literals(script):
    assert script.startswith('//@version=6\n// SPDX-License-Identifier: Apache-2.0\n// © TrendyCoins\n')
    assert 'indicator("Market Randomness Index", shorttitle = "MRI", overlay = false' in script
    code = re.sub(r'"[一-嗿]+"', '""', script)                          # drop the weight strings
    banned = r'\b(buy|sell|long|short|bull\w*|bear\w*|arrow\w*|triangle\w*|label_?up|label_?down)\b'
    assert not re.findall(banned, code, flags=re.I)
    assert not re.findall(r'color\.(red|green|lime|maroon|olive)', code)
    assert not re.findall(r'(?<![A-Za-z_])\d+\.?\d*[eE][-+]?\d+', code)          # no exponent literals
    assert 'barstate.isconfirmed' in code and 'request.' not in code
