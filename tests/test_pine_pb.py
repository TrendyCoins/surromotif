"""PB Pine template (mri/pine_pb.py): its line-by-line transliteration (tests/pine_sim_pb.py) == the reference stream
(mri.pb) < 1e-10, with and without 'compute only the last N bars', first line bar 552 (R16); the weights decoded from the
script strings are the export weights bit for bit; strings only inside functions; the script fits the size measured on
TradingView; tables, ruler and ceilings = export; project rules on wording, colors, literals."""
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pine_sim_pb  # noqa: E402
import synth  # noqa: E402
from mri import L, linem, nt3, pb  # noqa: E402
from mri.pine_gen import ALPHA0, decode_dense  # noqa: E402
from mri.pine_pb import (CALC_BARS_DEFAULT, PRODUCTION, PROTO, SIZE_LIMIT_UTF8, flat_values, make_indicator,  # noqa: E402
                          make_script, sizes, weight_decls)

FLOOR = 0.2
G = np.random.default_rng(3)
RULER = (np.sort(G.normal(-1.0, 1.4, 37)) + np.arange(37) * 1e-3, np.sort(G.normal(0, 1.2, 37)))


@pytest.mark.parametrize('calc_bars', [0, 700])
def test_pine_template_logic_equals_stream(calc_bars):
    b = synth.bars(1500, seed=5, gaps=[(1100, 3)])
    p = pb.random_params(seed=6)
    sp = pine_sim_pb.run(b, p, FLOOR, RULER, calc_bars=calc_bars, signed=nt3.SIDE_SIGNED[2:], span=400, warm=100)
    ref = pb.run(b, p, FLOOR, *RULER, span=400, warm=100)
    if calc_bars:            # windows of the computed bars have the logits of the full stream; the live part starts there
        heavy0 = len(b) - 1 - calc_bars - L
        assert np.isfinite([x['logit'] for x in ref[:heavy0 + L - 1]]).any()
        live = linem.LiveM(*RULER, span=400, warm=100)
        for x in ref:
            x.update({'z': np.nan, 'm_ema': np.nan, 's_ema': np.nan, 'log_m': np.nan, 'shown': False})
            if x['bar'] >= heavy0 + L - 1 and np.isfinite(x['logit']):
                x.update(live.update(x['logit']))
            else:
                x['logit'] = np.nan
    sl = np.array([x['logit'] for x in ref])
    fin = np.isfinite(sl)
    assert np.array_equal(fin, np.isfinite(sp['logit'])) and fin.sum() > 300
    assert np.abs(sp['logit'][fin] - sl[fin]).max() < 1e-10
    for k in ('z', 'm_ema', 's_ema', 'log_m'):
        assert np.abs(sp[k][fin] - np.array([x[k] for x in ref])[fin]).max() < 1e-10, k
    assert np.array_equal(sp['shown'], np.array([x['shown'] for x in ref]))
    assert max(np.abs(sp['side'][i] - x['side_raw']).max() for i, x in enumerate(ref) if np.isfinite(x['logit'])) < 1e-10
    first = int(np.argmax(sp['shown']))
    if calc_bars == 0:
        assert int(np.argmax(fin)) == 264 and first == 264 + 100
    else:
        assert int(np.argmax(fin)) == heavy0 + L - 1 and first == heavy0 + L - 1 + 100
    n = sp['n'][fin]
    assert np.array_equal(n, np.arange(fin.sum()))
    m = sp['M'][sp['shown']]
    lm = sp['log_m'][sp['shown']]
    assert np.allclose(m, np.where(lm > 0, np.minimum(np.exp(lm), 34.0), np.where(lm < 0, np.minimum(np.exp(-lm), 21.0), 1.0)))


def test_first_line_bar_552_default_warmup():
    b = synth.bars(800, seed=7)
    p = pb.random_params(seed=8)
    sp = pine_sim_pb.run(b, p, FLOOR, RULER, signed=nt3.SIDE_SIGNED[2:])
    assert int(np.argmax(np.isfinite(sp['logit']))) == 264 and int(np.argmax(sp['shown'])) == 552


TABLE = {'BTCUSDT': {'floor': 0.17, 'ceil_more': 13.0, 'ceil_less': 13.0},
         'ETHUSDT': {'floor': 0.18, 'ceil_more': 3.0, 'ceil_less': 21.0}}
UNKNOWN = {'floor': 0.2, 'ceil_more': 8.0, 'ceil_less': 8.0}
PASSPORT = {'weights_source': 'random', 'data_end': '-', 'code_version': 'test', 'ceil_pool': {'orange': 34.0, 'blue': 21.0}}


@pytest.fixture(scope='module')
def proto():
    p = pb.random_params(seed=11)
    return p, make_script(p, RULER, TABLE, UNKNOWN, PASSPORT)


def script_weights(script):
    funcs = re.findall(r'^w(\d\d)\(\) =>\n    decodeInto\(WT, ALPHA, "([^"]*)"\)$', script, flags=re.M)
    assert [int(j) for j, _ in funcs] == list(range(len(funcs)))
    assert re.findall(r'^var bool WOK(\d\d) = w(\d\d)\(\)$', script, flags=re.M) == [(j, j) for j, _ in funcs]
    return [x for _, x in funcs]


def test_weights_in_functions_are_exact(proto):
    p, script = proto
    strs = script_weights(script)
    assert len(strs) == 15 and all(len(s) <= 3999 for s in strs) and sum(map(len, strs)) == 3 * 18697
    got = np.array(decode_dense(''.join(strs)))
    assert np.array_equal(got, flat_values(p).astype(np.float64)) and len(got) == 18697
    long_str = [ln for ln in script.splitlines() if re.search(r'"[^"]{1000,}"', ln)]
    assert all(ln.startswith('    decodeInto(WT, ALPHA, "') or 'str.split("' in ln for ln in long_str)
    alpha = re.search(r'str\.split\("([^"]{2048})", ""\)', script).group(1)
    assert [ord(c) - ALPHA0 for c in alpha] == list(range(2048))
    assert 'import ' not in script and 'library(' not in script


def test_flat_order_and_declarations(proto):
    p, script = proto
    vals = flat_values(p).astype(np.float64)
    decls, off = weight_decls(64, 24, 209)
    for nm, kind, r, c, o in decls:
        blk = vals[o:o + r * c].reshape(r, c)
        if nm.startswith('K1_'):
            ref = p.conv[0][:, :, int(nm[-1])]
        elif nm.startswith('D'):
            ref = p.conv[int(nm[1]) - 1][:, 0, int(nm[-1])][:, None]
        elif nm.startswith('P'):
            ref = p.pw[int(nm[1]) - 2]
        else:
            ref = p.head0_w if nm == 'H0' else p.head0_b[:, None]
        assert np.array_equal(blk, ref.astype(np.float32).astype(np.float64)), nm
        decl = (f'var matrix<float> {nm:<4} = toMat(WT, {o}, {r}, {c})' if kind == 'mat'
                else f'var array<float>  {nm:<4} = toArr(WT, {o}, {r})')
        assert decl in script
    assert np.array_equal(vals[off:off + 24], p.head3_w[0].astype(np.float32)) and vals[off + 24] == np.float32(p.head3_b[0])
    assert f'H3  = array.copy(array.slice(WT, {off}, {off + 24}))' in script and f'HB3 = array.get(WT, {off + 24})' in script


def test_size_fits_one_script(proto):
    _, script = proto
    s = sizes(script)
    assert s['utf8'] <= SIZE_LIMIT_UTF8
    assert script.count('decodeInto(WT, ALPHA, "') == 15


@pytest.mark.parametrize('which', ['template', 'production'])
def test_wording_colors_literals(proto, which):
    if which == 'template':
        script = proto[1]
    else:
        with open(os.path.join(ROOT, PRODUCTION), encoding='utf-8') as fh:
            script = fh.read()
    assert script.startswith('//@version=6\n// SPDX-License-Identifier: Apache-2.0\n// © TrendyCoins\n')
    code = re.sub(r'"[一-嗿]+"', '""', script)
    banned = r'\b(buy|sell|long|short|bull\w*|bear\w*|arrow\w*|triangle\w*|label_?up|label_?down)\b'
    assert not re.findall(banned, code, flags=re.I)
    assert not re.findall(r'color\.(red|green|lime|maroon|olive)', code)
    assert not re.findall(r'\bSMC\b', code)
    assert 'barstate.isconfirmed' in code and 'request.' not in code
    body = re.sub(r'"[^"]*"', '""', re.sub(r'//.*', '', code))           # no clock or EMA of prices in the computation
    assert not re.findall(r'(?<![A-Za-z_])\d+\.?\d*[eE][-+]?\d+', body)  # no exponent literals (comments: hashes)
    assert not re.findall(r'ta\.(ema|rma|wma)|\b(hour|minute|dayofweek|dayofmonth|weekofyear|month|year)\b', body)
    assert [ln.strip() for ln in body.splitlines() if re.search(r'\btime\b', ln)] == [
        'bool  gap    = bar_index > 0 and (time - time[1]) > GAPMS']          # time: grid gaps only
    dbg = re.findall(r'^plot\([^\n]*"dbg ([^"]+)"', code, flags=re.M)
    assert dbg == ['logit', 'z', 'm_ema', 's_ema', 'logM', 'n', 'norm', 'r', 'step_hl2', 'step_hlc3', 'g', 'D', 'x_end',
                   'sgn'] + [f'side_{n}' for n in nt3.SIDE_NAMES] + ['confirmed', 'bar_index']     # R10 addendum 3
    assert len(re.findall(r'^plot\(', code, flags=re.M)) == 1 + len(dbg)
    # the live part changes only on a confirmed bar with a valid window
    blk = code[code.index('if barstate.isconfirmed and bar_index >= heavyFrom'):code.index('// ------', code.index(
        'if barstate.isconfirmed and bar_index >= heavyFrom'))]
    for v in ('n1', 'n2', 'dd'):
        assert len(re.findall(rf'^\s+{v}\s+:=', code, flags=re.M)) == 1 and re.search(rf'\n        {v}\s+:=', blk)
    assert len(re.findall(r'nW\s+\+= 1', code)) == 1 and re.search(r'\n        nW\s+\+= 1', blk)


@pytest.mark.parametrize('production', [False, True])
def test_script_is_generated_from_export(production):
    """pine/proto_pb.pine (parity R10 was made on it) and the production pine/mri.pine: weights, tables, ruler = export."""
    from mri.export_pb import EXPORT, load_export
    from mri.params import from_dict
    ex = load_export(os.path.join(ROOT, EXPORT))
    script = make_indicator(ex, production=production)
    with open(os.path.join(ROOT, PRODUCTION if production else PROTO), encoding='utf-8') as fh:
        assert fh.read() == script
    assert sizes(script)['utf8'] <= SIZE_LIMIT_UTF8
    assert np.array_equal(np.array(decode_dense(''.join(script_weights(script)))),
                          flat_values(from_dict(ex['model'])).astype(np.float64))

    def table(fn):
        body = re.search(rf'{fn}\(\) =>\n(.*?)\n    m\n', script, flags=re.S).group(1)
        return {k: float(v) for k, v in re.findall(r'map\.put\(m, "([A-Z]+)", (-?[0-9.]+)\)', body)}
    c = ex['line']['ceilings']
    assert table('buildFloors') == ex['floors']['floor_L64']
    assert table('buildCeilMore') == {a: v['orange'] for a, v in c['by_symbol'].items()}
    assert table('buildCeilLess') == {a: v['blue'] for a, v in c['by_symbol'].items()}
    unk = lambda name: float(re.search(rf'var float {name}\s*= known \? map\.get\(\w+, syminfo\.ticker\) : (-?[0-9.]+)',
                                       script).group(1))
    assert unk('FLOOR') == ex['floors']['unknown_symbol']
    assert unk('CEIL_O') == c['unknown_symbol']['orange'] and unk('CEIL_B') == c['unknown_symbol']['blue']
    arr = lambda name: [float(x) for x in re.search(rf'var array<float> {name} = array\.from\(([^)]*)\)', script).group(1)
                        .split(', ')]
    assert arr('RQ') == ex['line']['ruler']['q'] and arr('RZ') == ex['line']['ruler']['zk']
    assert re.search(r'const int   SPAN0  = 8064\b', script) and re.search(r'const int   WARM0  = 288\b', script)
    assert 'input.int(8064, "Live part: span, bars"' in script and 'input.int(288, "Line warm-up, windows"' in script
    assert 'const float S_LO   = 0.75' in script
    if production:
        assert 'indicator("Market Randomness Index", shorttitle = "MRI", overlay = false' in script
        assert f'input.int({CALC_BARS_DEFAULT}, "Compute only the last N bars' in script and CALC_BARS_DEFAULT == 10000
    else:
        assert 'indicator("Market Randomness Index prototype", shorttitle = "MRI proto", overlay = false' in script
        assert 'input.int(0, "Compute only the last N bars' in script


def test_production_differs_from_prototype_only_in_title_and_default_n():
    """The parity of R10 was made on the prototype (N = 0): the production script has the same computation — only the title
       lines and the default of 'Compute only the last N bars' differ."""
    with open(os.path.join(ROOT, PROTO), encoding='utf-8') as fh:
        a = fh.read().splitlines()
    with open(os.path.join(ROOT, PRODUCTION), encoding='utf-8') as fh:
        b = fh.read().splitlines()
    assert len(a) == len(b)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(diff) == 3
    assert diff[0][1].startswith('// Market Randomness Index — generated by mri/pine_pb.py')
    assert diff[1][1].startswith('indicator("Market Randomness Index", shorttitle = "MRI",')
    assert diff[2] == (diff[2][1].replace('input.int(10000, ', 'input.int(0, '), diff[2][1]) and 'calcBars' in diff[2][1]


def test_pine_template_on_real_weights_equals_golden():
    """The transliteration with the export weights, ruler and floor of BTC on the first 1200 bars of the golden file."""
    import pandas as pd
    from mri.export_pb import EXPORT, GOLDEN, load_export, params_line
    ex = load_export(os.path.join(ROOT, EXPORT))
    g = pd.read_csv(os.path.join(ROOT, GOLDEN), nrows=1200)
    p, q, zk = params_line(ex)
    c = ex['line']['ceilings']['by_symbol']['BTCUSDT']
    sp = pine_sim_pb.run(g[['time', 'open', 'high', 'low', 'close']], p, ex['floors']['floor_L64']['BTCUSDT'], (q, zk),
                         signed=nt3.SIDE_SIGNED[2:], ceil=(c['orange'], c['blue']))
    fin = np.isfinite(g['logit'].to_numpy())
    assert np.array_equal(fin, np.isfinite(sp['logit']))
    for k in ('logit', 'z', 'm_ema', 's_ema', 'log_m'):
        x = g[k].to_numpy()[fin]
        assert np.abs(sp[k][fin] - x).max() < 1e-9 * max(1.0, np.abs(x).max()), k
    assert np.array_equal(sp['shown'], g['shown'].to_numpy() > 0) and int(np.argmax(sp['shown'])) == 552
