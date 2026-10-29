"""PB Pine template (mri/pine_pb.py), session 5: its line-by-line transliteration (tests/pine_sim_pb.py) == the reference
stream (mri.pb) < 1e-10, with and without 'compute only the last N bars', first line bar 4360 (R18) with one start mark;
the weights decoded from the script strings are the export weights bit for bit; strings only inside functions; the script
fits the size measured on TradingView; FLOOR, CEIL, ruler, L, n0, lag and the start of the line = export, no symbol tables,
no settings of the live part; project rules on wording, colors (the site palette, 2026-10-27), literals."""
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
from mri.pine_common import ALPHA0, decode_dense  # noqa: E402
from mri.pine_pb import (CALC_BARS_DEFAULT, PRODUCTION, PROTO, SIZE_LIMIT_UTF8, flat_values, make_indicator,  # noqa: E402
                          make_script, sizes, strip_debug, weight_decls)

FLOOR = pb.FLOOR
G = np.random.default_rng(3)
RULER = (np.sort(G.normal(-1.0, 1.4, 37)) + np.arange(37) * 1e-3, np.sort(G.normal(0, 1.2, 37)))
LIVE = {'L': 4032, 'n0': 144, 'lag': 64, 'show_from_window': 4096}


@pytest.mark.parametrize('calc_bars', [0, 4250])
def test_pine_template_logic_equals_stream(calc_bars):
    b = synth.bars(4700, seed=5, gaps=[(1100, 3)])
    p = pb.random_params(seed=6)
    sp = pine_sim_pb.run(b, p, FLOOR, RULER, calc_bars=calc_bars, signed=nt3.SIDE_SIGNED[2:])
    ref = pb.run(b, p, FLOOR, *RULER)
    if calc_bars:            # windows of the computed bars have the logits of the full stream; the live part starts there
        heavy0 = len(b) - 1 - calc_bars - L
        assert np.isfinite([x['logit'] for x in ref[:heavy0 + L - 1]]).any()
        live = linem.LiveWin(*RULER)
        for x in ref:
            x.update({'z': np.nan, 'm_win': np.nan, 's_win': np.nan, 'k': -1, 'log_m': np.nan, 'shown': False})
            if x['bar'] >= heavy0 + L - 1 and np.isfinite(x['logit']):
                x.update(live.update(x['logit']))
            else:
                x['logit'] = np.nan
    sl = np.array([x['logit'] for x in ref])
    fin = np.isfinite(sl)
    assert np.array_equal(fin, np.isfinite(sp['logit'])) and fin.sum() > 4100
    assert np.abs(sp['logit'][fin] - sl[fin]).max() < 1e-10
    for k in ('z', 'm_win', 's_win', 'log_m'):
        assert np.abs(sp[k][fin] - np.array([x[k] for x in ref])[fin]).max() < 1e-10, k
    assert np.array_equal(sp['k'][fin], np.array([x['k'] for x in ref])[fin])
    assert np.array_equal(sp['shown'], np.array([x['shown'] for x in ref]))
    assert max(np.abs(sp['side'][i] - x['side_raw']).max() for i, x in enumerate(ref) if np.isfinite(x['logit'])) < 1e-10
    first = int(np.argmax(sp['shown']))
    w0 = 264 if calc_bars == 0 else heavy0 + L - 1                   # first computed window
    assert int(np.argmax(fin)) == w0
    assert first == w0 + 4096 + 64                                    # 64 windows over the gap at bar 1100 are invalid
    assert np.where(sp['start'])[0].tolist() == [first]               # one mark at the start of the line
    n = sp['n'][fin]
    assert np.array_equal(n, np.arange(fin.sum()))
    m = sp['M'][sp['shown']]
    lm = sp['log_m'][sp['shown']]
    assert np.allclose(m, np.where(lm != 0, np.minimum(np.exp(np.abs(lm)), 10.0), 1.0), rtol=1e-14, atol=0.0)


def test_first_line_bar_4360():
    b = synth.bars(4450, seed=7)
    p = pb.random_params(seed=8)
    sp = pine_sim_pb.run(b, p, FLOOR, RULER, signed=nt3.SIDE_SIGNED[2:])
    assert int(np.argmax(np.isfinite(sp['logit']))) == 264 and int(np.argmax(sp['shown'])) == 4360
    assert np.where(sp['start'])[0].tolist() == [4360] and sp['k'][4359] == 4032 and sp['n'][4360] == 4096


PASSPORT = {'weights_source': 'random', 'data_end': '-', 'code_version': 'test'}


@pytest.fixture(scope='module')
def proto():
    p = pb.random_params(seed=11)
    return p, make_script(p, RULER, FLOOR, 10.0, LIVE, PASSPORT)


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
    # license comments first, then the version (TradingView's script template and the Pine style guide; 2026-10-27)
    assert script.startswith('// SPDX-License-Identifier: Apache-2.0\n'
                             '// This source code is subject to the terms of the Apache License, Version 2.0.\n'
                             '// © TrendyCoins\n\n//@version=6\n')
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
    if which == 'template':                    # the prototype keeps the debug mode (parity R10), the production has none
        assert dbg == ['logit', 'z', 'm_win', 's_win', 'k', 'logM', 'n', 'norm', 'r', 'step_hl2', 'step_hlc3', 'g', 'D',
                       'x_end', 'sgn'] + [f'side_{n}' for n in nt3.SIDE_NAMES] + ['confirmed', 'bar_index']   # R10 add. 4
    else:
        assert dbg == [] and not re.search(r'\b(dbg|DBG|nConf|display_dbg)\b', code) and 'Debug' not in code
    assert len(re.findall(r'^plot\(', code, flags=re.M)) == 1 + len(dbg)
    # the live part changes only on a confirmed bar with a valid window; the ring without array.shift
    blk = code[code.index('if barstate.isconfirmed and bar_index >= heavyFrom'):code.index('// ------', code.index(
        'if barstate.isconfirmed and bar_index >= heavyFrom'))]
    for v in ('S1', 'S2'):
        assert len(re.findall(rf'^\s+{v}\s+[-+]=', code, flags=re.M)) == 2 and len(re.findall(rf'\n\s+{v}\s+[-+]=', blk)) == 2
    assert re.findall(r'array\.\w+\(ZR\b', code) == ['array.get(ZR', 'array.set(ZR', 'array.get(ZR']
    assert all(f'array.{fn}(ZR' in blk for fn in ('get', 'set')) and 'array.shift(ZR' not in code
    assert len(re.findall(r'nW\s+\+= 1', code)) == 1 and re.search(r'\n        nW\s+\+= 1', blk)
    # one floor, one ceiling, no constants of the symbol and no settings of the live part
    for x in ('syminfo.ticker', 'buildFloors', 'buildCeil', 'FLOORS', 'CEILO', 'CEILB', 'known', 'spanIn', 'warmIn',
              'not a Binance spot', 'span / warm', 'KEMA', 'm_ema'):
        assert x not in code, x
    assert len(re.findall(r'^const float FLOOR = 0\.20000000298023224 ', code, flags=re.M)) == 1
    assert len(re.findall(r'^const float CEIL  = 10\.0 ', code, flags=re.M)) == 1
    # the start mark: a plot shape with a constant text (not a label), at the window SHOW only
    st = re.findall(r'^plotshape\(shownM and nW == SHOW, "Line start", shape\.diamond, location\.bottom, '
                    r'color\.new\(colorEqualInput, 100\), text = "([^"]+)", textcolor = chart\.fg_color, size = size\.tiny, '
                    r'editable = false, display = display\.pane\)$', code, flags=re.M)
    assert st == ['MRI starts here:\\non each bar, the 64-bar window\\nis compared with 4032 historical\\n'
                  'windows of this chart']                     # text of 2026-10-27; the shape under it is transparent
    # the faint grid M = 2, 4, 6, 8 next to the line M = 1, no threshold line, no colour of the price bars (2026-10-27)
    assert re.findall(r'^hline\(([0-9.]+), ', code, flags=re.M) == ['1.0', '2.0', '4.0', '6.0', '8.0']
    assert 'barcolor(' not in code and 'thresholdInput' not in code
    assert 'input.float(8.0, "Marks on the price chart at M >=", minval = 0.0' in code   # default 8 (2026-10-27)
    # the floor M = 1 — solid, full silver; the grid — dashed, silver at 70% transparency (2026-10-27) on a dark chart
    # and at 40% on a light one (2026-10-29: over white 70% gives contrast 1.3:1, 40% gives 1.7:1 — as 70% over dark)
    assert re.findall(r'^hline\(1\.0, [^\n]*', code, flags=re.M) == [
        'hline(1.0, "Equally often (M = 1)", color = colorEqualInput, linestyle = hline.style_solid, editable = false)']
    assert '\nbool  lightChart = chart.fg_color == #0F0F0F\n' in code
    assert '\ncolor gridColor  = color.new(colorEqualInput, lightChart ? 40 : 70)\n' in code
    assert len(re.findall(r'^hline\([2468]\.0, "M = [2468]", color = gridColor, '
                          r'linestyle = hline\.style_dashed, editable = false\)$', code, flags=re.M)) == 4
    assert 'label.' not in code and '10+' not in code and 'max_labels_count' not in code   # no "10+" (decision 2026-10-18)
    # the palette of the site (2026-10-25): silver at M = 1 towards orange (real markets) or grey (the random chart),
    # the share of the side colour = ln M / ln CEIL; the old blue is gone
    assert 'input.color(#FF7A1A, "More often in real markets"' in code
    assert 'input.color(#9D9D9D, "Equally often (M = 1)"' in code
    assert 'input.color(#665F5A, "More often on the random chart"' in code
    assert ('color mriColor  = shownM ? color.from_gradient(math.log(Mshow), 0.0, math.log(CEIL), colorEqualInput, '
            'sideColor) : na') in code
    assert 'plot(Mshow, "M", color = mriColor, linewidth = 2)' in code
    assert 'rgb(47, 128, 237)' not in code and not re.search(r'\bblue\b', code, flags=re.I)
    # the comments describe only the indicator: no files, exports, commits or links outside it (2026-10-27)
    for x in ('mri/', 'export/', '.py', '.json', 'generated by', 'passport', 'http', 'github', 'trendycoins.com',
              'prod_PB', 'code_version'):
        assert x not in code, x


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

    const = lambda name: re.search(rf'^const (?:float|int)\s+{name}\s*= (-?[0-9.]+)', script, flags=re.M).group(1)
    assert const('FLOOR') == repr(ex['floor']['value']) == '0.20000000298023224'
    assert float(const('FLOOR')) == ex['floor']['value'] == pb.FLOOR
    assert float(const('CEIL')) == ex['line']['ceiling']['value'] == 10.0
    lv = ex['line']['live']
    assert (int(const('LW')), float(const('N0')), int(const('LAG')), int(const('SHOW'))) == (
        lv['L'], lv['n0'], lv['lag'], ex['line']['show']['from_window']) == (4032, 144, 64, 4096)
    assert float(const('S_LO')) == lv['s_lo'] == 0.75
    arr = lambda name: [float(x) for x in re.search(rf'var array<float> {name} = array\.from\(([^)]*)\)', script).group(1)
                        .split(', ')]
    assert arr('RQ') == ex['line']['ruler']['q'] and arr('RZ') == ex['line']['ruler']['zk']
    assert 'map.put(m, "' not in script and 'syminfo' not in script
    if production:
        assert 'indicator("Market Randomness Index", shorttitle = "MRI", overlay = false' in script
        assert f'input.int({CALC_BARS_DEFAULT}, "Compute only the last N bars' in script and CALC_BARS_DEFAULT == 10000
    else:
        assert 'indicator("Market Randomness Index prototype", shorttitle = "MRI proto", overlay = false' in script
        assert 'input.int(0, "Compute only the last N bars' in script


def test_production_is_the_prototype_without_debug():
    """The parity of R10 was made on the prototype (N = 0, debug mode): the production script is the same text with the debug
       lines cut (decision of 2026-10-18) — after strip_debug only the title lines and the default of 'Compute only the
       last N bars' differ; the computation lines are the same."""
    with open(os.path.join(ROOT, PROTO), encoding='utf-8') as fh:
        proto = fh.read()
    with open(os.path.join(ROOT, PRODUCTION), encoding='utf-8') as fh:
        prod = fh.read()
    a, b = strip_debug(proto).splitlines(), prod.splitlines()
    assert len(a) == len(b) and len(proto.splitlines()) - len(b) == 46     # 1 input, 2 state, 5 writes, 38 debug section
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(diff) == 3
    assert diff[0][1] == '// Market Randomness Index (MRI): shows when the market does not look like a random walk.'
    assert diff[1][1].startswith('indicator("Market Randomness Index", shorttitle = "MRI",')
    assert diff[2] == (diff[2][1].replace('input.int(10000, ', 'input.int(0, '), diff[2][1]) and 'calcBars' in diff[2][1]
    cut = [ln for ln in proto.splitlines() if re.search(r'\b(DBG|nConf|dbg|display_dbg)\b', ln)]
    assert all(ln.lstrip().startswith(('plot(', 'array.set(DBG', 'array.fill(DBG', 'var array<float> DBG', 'var int   nConf',
                                       'nConf  += 1', 'bool  dbg', 'display_dbg')) for ln in cut), cut


def test_pine_template_on_real_weights_equals_golden():
    """The transliteration with the export weights, ruler, floor and ceiling on the first 4600 bars of the golden file."""
    import pandas as pd
    from mri.export_pb import EXPORT, GOLDEN, load_export, params_line
    ex = load_export(os.path.join(ROOT, EXPORT))
    g = pd.read_csv(os.path.join(ROOT, GOLDEN), nrows=4600)
    p, q, zk = params_line(ex)
    sp = pine_sim_pb.run(g[['time', 'open', 'high', 'low', 'close']], p, ex['floor']['value'], (q, zk),
                         signed=nt3.SIDE_SIGNED[2:], ceil=ex['line']['ceiling']['value'])
    fin = np.isfinite(g['logit'].to_numpy())
    assert np.array_equal(fin, np.isfinite(sp['logit']))
    for k in ('logit', 'z', 'm_win', 's_win', 'log_m'):
        x = g[k].to_numpy()[fin]
        assert np.abs(sp[k][fin] - x).max() < 1e-9 * max(1.0, np.abs(x).max()), k
    assert np.array_equal(sp['k'][fin], g['k'].to_numpy()[fin])
    assert np.array_equal(sp['shown'], g['shown'].to_numpy() > 0) and int(np.argmax(sp['shown'])) == 4360
    assert np.where(sp['start'])[0].tolist() == [4360]
