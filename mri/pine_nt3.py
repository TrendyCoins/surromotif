"""Pine Script v6 generator of the NT3_PRV_ch48 indicator.

    python -m mri.pine_nt3                 # pine/mri.pine + its weight libraries pine/lib/MRIWeightsNT3{a,b,c}.pine
    python -m mri.pine_nt3 --ascii-test    # + pine/test/mri_ascii_single.pine: all weights inside, 5 ASCII characters each
                                           #   (a size test: does a single script fit the compilation request?)

TradingView refused the single script with the weights inside (141 919 characters, 368 KB UTF-8): "The size of the script
compilation request is too large", while the F3_g script (75 673 characters, 181 KB) compiles. So the weights go into
3 libraries, each well below the F3_g script by any measure (characters, UTF-8, UTF-16, escaped bytes); every exported
function decodes one string at run time (no constant to fold into the indicator) with the alphabet map passed by the
indicator. The libraries are published by the author (privately for tests) and imported by user name / name / version.
Weights: the dense float32 strings of mri/pine_gen.py (3 CJK characters per float32, bit for bit), now 36 961 numbers.
Flat order: conv layer 1..4, tap 0..2, W[:, :, tap] row-major (out, in); head0 W row-major (96, 161); head0 b (96);
head3 W (96); head3 b (1).

Compute per confirmed bar (the same as mri.nt3.StreamingNT3; tests/pine_sim_nt3.py transliterates the template):
    conv streams + / - in interleaved ring matrices (CH rows), 3 matrix.mult per layer, ReLU loop; layer 4 into two (CH x 33)
    matrices; pooled mean / max / last over 33 positions, / D; head 161 -> 96 -> 1.
History inputs of the window ending at bar t (start s = t - 63), no loops over bars:
    per bar b (global scope, every bar) from the 200 bars before b: nrm = median range (the norm of b), mean range, median
    body (floored at 0.05 * nrm), mean range of the last 16 / 64 bars; ta.highest(high, 199), ta.lowest(low, 199),
    ta.sma(close, 199) over bars b-198 .. b;
    at t: the scales of bar s are those series [63]; hmax / lmin / mean close over s-200 .. s-2 are [65]; close of s-2 is
    close[65]; the 8 path chunks are close / open at fixed offsets 65 .. 263.
"""
import os

import numpy as np

from .nt3 import NT_GEO, SIDE_NAMES, SIDE_SIGNED
from .pine_gen import ALPHA0, ALPHA_N, encode_dense, fmt, weights_block

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L = 64


def flat_values(p):
    out = []
    for w in p.conv:
        for k in range(3):
            out.extend(w[:, :, k].ravel())
    for a in (p.head0_w, p.head0_b, p.head3_w, p.head3_b):
        out.extend(np.ravel(a))
    return np.array(out, dtype=np.float32)


def matrix_offsets(ch, hidden, n_feat):
    """(name, rows, cols, offset) of the matrices in the flat order; then H3 (hidden) and HB3 (1)."""
    out, off = [], 0
    for li, c_in in enumerate((4, ch, ch, ch), 1):
        for k in range(3):
            out.append((f'K{li}_{k}', ch, c_in, off))
            off += ch * c_in
    out.append(('H0', hidden, n_feat, off))
    off += hidden * n_feat
    out.append(('HB0', hidden, 1, off))
    off += hidden
    return out, off


def chunk_offsets():
    """(n, offset of the chunk's last bar, offset of its first bar) from the window's last bar, oldest chunk first.
       Bar s-1 is at offset L, s-2 at L + 1; chunks run over bars s-200 .. s-2."""
    res, end = [], L + 1
    for n in NT_GEO[::-1]:
        start = end + n - 1
        res.append((n, end, start))
        end = start + 1
    assert end == L + 200
    return res[::-1]


def float_list(xs):
    return ', '.join(fmt(x) for x in xs)


CONSTS = r'''// ---------------------------------------------------------------- constants
const int   L      = 64
const int   NN     = 200          // bars of the range median (the norm of a bar) and of the history before a window
const int   CH     = {ch}           // conv channels
const int   HID    = {hidden}           // hidden units of the head
const int   NS     = {n_side}           // side inputs
const float EPS    = 0.0001       // orientation tie rule: flip only if X_end < -EPS * D
const float SQ199  = {sq199}
const float S_SD   = {s_sd}
const float LVL_SD = {lvl_sd}
var array<float> SIDE_MU = array.from({side_mu})
var array<float> SIDE_SD = array.from({side_sd})
var array<float> HSIGNED = array.from({signed})   // history inputs that take the orientation sign

'''

CORE = r'''// ---------------------------------------------------------------- weights: {weights_title}
{weights}

toMat(array<float> w, int off, int nr, int nc) =>
    matrix<float> m = matrix.new<float>(nr, nc, 0.0)
    for i = 0 to nr - 1
        for j = 0 to nc - 1
            matrix.set(m, i, j, array.get(w, off + i * nc + j))
    m

{matrices}
var array<float>  H3  = array.copy(array.slice(WT, {off_h3}, {off_h3_end}))
var float         HB3 = array.get(WT, {off_hb3})

// ---------------------------------------------------------------- state
var array<float> RNG  = array.new<float>(0)        // high - low of the last 200 bars
var array<float> BODY = array.new<float>(0)        // |close - open| of the last 200 bars
var array<float> WR   = array.new<float>(L, 0.0)   // r of the last 64 bars (0 on invalid bars), oldest first
var array<float> WABS = array.new<float>(L, 0.0)   // |r| of the last 64 bars
var float prevC  = na
var float prevD1 = na                              // (HL/2 - close) / norm of the previous bar
var float prevD2 = na                              // (HLC/3 - close) / norm of the previous bar
var int   okRun  = 0                               // consecutive valid bars
var int   firstR = -1                              // bar_index of the first valid r
var int   nConf  = 0                               // confirmed bars processed (debug: the state advanced)

// conv rings: inputs (4 x 2*3), layers 1..3 (CH x 2*5, 2*9, 2*17); layer 4 per stream (CH x 33)
var matrix<float> R0  = matrix.new<float>(4, 6, 0.0)
var matrix<float> R1  = matrix.new<float>(CH, 10, 0.0)
var matrix<float> R2  = matrix.new<float>(CH, 18, 0.0)
var matrix<float> R3  = matrix.new<float>(CH, 34, 0.0)
var matrix<float> R4P = matrix.new<float>(CH, 33, 0.0)
var matrix<float> R4M = matrix.new<float>(CH, 33, 0.0)
var matrix<float> ONES33 = matrix.new<float>(33, 1, 1.0 / 33.0)
var matrix<float> FEAT   = matrix.new<float>(3 * CH + NS, 1, 0.0)
var int hc = -1                                    // heavy-part bar counter (ring position)
var array<float> DBG = array.new<float>(3 + NS, na)   // values of the last window: D, X_end, sign, side inputs (debug)

slot(int cnt, int depth, int back) =>
    (cnt - back + depth * 64) % depth

tap(matrix<float> R, int depth, int cnt, int back) =>
    int s = slot(cnt, depth, back)
    matrix.submatrix(R, 0, matrix.rows(R), 2 * s, 2 * s + 2)

conv3(matrix<float> R, int depth, int cnt, int d, matrix<float> W0, matrix<float> W1, matrix<float> W2) =>
    matrix.sum(matrix.sum(matrix.mult(W0, tap(R, depth, cnt, 2 * d)), matrix.mult(W1, tap(R, depth, cnt, d))), matrix.mult(W2, tap(R, depth, cnt, 0)))

reluInto(matrix<float> y, matrix<float> R, int depth, int cnt) =>
    int s = slot(cnt, depth, 0)
    for i = 0 to CH - 1
        matrix.set(R, i, 2 * s, math.max(matrix.get(y, i, 0), 0.0))
        matrix.set(R, i, 2 * s + 1, math.max(matrix.get(y, i, 1), 0.0))
    true

// one bar of the two conv streams: inputs u (stream +), ring writes for layers 1..4
convStep(int cnt, float u0, float u1, float u2, float u3) =>
    int p0 = slot(cnt, 3, 0)
    matrix.set(R0, 0, 2 * p0, u0)
    matrix.set(R0, 1, 2 * p0, u1)
    matrix.set(R0, 2, 2 * p0, u2)
    matrix.set(R0, 3, 2 * p0, u3)
    matrix.set(R0, 0, 2 * p0 + 1, -u0)
    matrix.set(R0, 1, 2 * p0 + 1, -u1)
    matrix.set(R0, 2, 2 * p0 + 1, -u2)
    matrix.set(R0, 3, 2 * p0 + 1, u3)
    reluInto(conv3(R0, 3, cnt, 1, K1_0, K1_1, K1_2), R1, 5, cnt)
    reluInto(conv3(R1, 5, cnt, 2, K2_0, K2_1, K2_2), R2, 9, cnt)
    reluInto(conv3(R2, 9, cnt, 4, K3_0, K3_1, K3_2), R3, 17, cnt)
    matrix<float> y4 = conv3(R3, 17, cnt, 8, K4_0, K4_1, K4_2)
    int p4 = slot(cnt, 33, 0)
    for i = 0 to CH - 1
        matrix.set(R4P, i, p4, math.max(matrix.get(y4, i, 0), 0.0))
        matrix.set(R4M, i, p4, math.max(matrix.get(y4, i, 1), 0.0))
    true

// logit of the window ending at this bar; hst = 15 history inputs of the window (before the orientation sign)
computeLogit(int cnt, array<float> hst) =>
    float D = math.max(array.median(WABS), FLOOR)
    float xe = 0.0
    for i = 0 to L - 1
        xe += i * array.get(WR, i)
    xe /= L
    float sgn = xe < -EPS * D ? -1.0 : 1.0
    array.set(DBG, 0, D)
    array.set(DBG, 1, xe)
    array.set(DBG, 2, sgn)
    matrix<float> M4 = sgn > 0 ? R4P : R4M
    matrix<float> mean4 = matrix.mult(M4, ONES33)
    int p4 = slot(cnt, 33, 0)
    for i = 0 to CH - 1
        matrix.set(FEAT, i, 0, matrix.get(mean4, i, 0) / D)
        matrix.set(FEAT, CH + i, 0, array.max(matrix.row(M4, i)) / D)
        matrix.set(FEAT, 2 * CH + i, 0, matrix.get(M4, i, p4) / D)
    array<float> side = array.from(math.log(D), math.log(math.max(array.avg(WABS), FLOOR)))
    for j = 0 to NS - 3
        array.push(side, array.get(HSIGNED, j) > 0 ? array.get(hst, j) * sgn : array.get(hst, j))
    for j = 0 to NS - 1
        array.set(DBG, 3 + j, array.get(side, j))
        matrix.set(FEAT, 3 * CH + j, 0, (array.get(side, j) - array.get(SIDE_MU, j)) / array.get(SIDE_SD, j))
    matrix<float> z = matrix.sum(matrix.mult(H0, FEAT), HB0)
    float acc = HB3
    for i = 0 to HID - 1
        acc += math.max(matrix.get(z, i, 0), 0.0) * array.get(H3, i)
    acc

clipZ(float x) =>
    nz(math.max(math.min(x, 20.0), -20.0))

clipLog(float v) =>
    nz(math.max(math.min(math.log(math.max(v, 0.001)), 5.0), -5.0))

// ---------------------------------------------------------------- per-bar inputs (closed='left' norm)
float rngNow = high - low
float nrm    = array.size(RNG) == NN ? array.median(RNG) : na
bool  goodN  = not na(nrm) and nrm > 0
bool  gap    = bar_index > 0 and (time - time[1]) > GAPMS
bool  bad    = not goodN or gap or na(prevC)
float r      = bad ? na : (close - prevC) / nrm
float st1    = bad ? na : (hl2 - prevC) / nrm - prevD1
float st2    = bad ? na : (hlc3 - prevC) / nrm - prevD2
float gg     = bad ? na : rngNow / nrm
bool  ok     = not na(r) and not na(st1) and not na(st2) and not na(gg)

// ---------------------------------------------------------------- history before the window, in bars (no time, no EMA)
// scales of a window starting at this bar, from the 200 bars before it
float rmeanB = array.size(RNG) == NN ? array.avg(RNG) : na
float bmedB  = array.size(BODY) == NN ? math.max(array.median(BODY), 0.05 * nrm) : na
float r16B   = array.size(RNG) == NN ? array.avg(array.slice(RNG, NN - 16, NN)) : na
float r64B   = array.size(RNG) == NN ? array.avg(array.slice(RNG, NN - 64, NN)) : na
float hmaxB  = ta.highest(high, NN - 1)            // bars b-198 .. b
float lminB  = ta.lowest(low, NN - 1)
float cmaB   = ta.sma(close, NN - 1)
// the window ending at this bar starts at s = bar_index - 63: scales of bar s are [63], bars s-200 .. s-2 end at [65]
float rmedW  = nrm[L - 1]
float rmeanW = rmeanB[L - 1]
float bmedW  = bmedB[L - 1]
float hmaxW  = hmaxB[L + 1]
float lminW  = lminB[L + 1]
float wW     = hmaxW - lminW
float c2W    = close[L + 1]
// path before the window in 8 chunks over bars s-200 .. s-2, oldest first: (close of the chunk's last bar - open of its
// first bar) / (sqrt(n) * norm of s), clipped to +-20
{chunks}
float R1h = wW > 0 ? 2.0 * (c2W - lminW) / wW - 1.0 : 0.0          // position of close[s-2] in the 199-bar range
float R2h = nz(math.max(math.min(math.log(math.max(wW, 0.000000000001) / (SQ199 * rmedW)), 5.0), -5.0))   // range width
float R3h = clipZ((c2W - cmaB[L + 1]) / (SQ199 * rmedW))         // close[s-2] from the mean close
float V1h = clipLog(r16B[L - 1] / rmeanW)                        // mean range of the last 16 bars / of 200
float V2h = clipLog(r64B[L - 1] / rmeanW)                        // the same for 64 bars
float V3h = clipLog(rmeanW / rmedW)                              // mean / median range
float V4h = clipLog(bmedW / rmedW)                               // median body / median range
array<float> HST = array.from(P1h, P2h, P3h, P4h, P5h, P6h, P7h, P8h, R1h, R2h, R3h, V1h, V2h, V3h, V4h)

int heavyFrom = calcBars == 0 ? 0 : math.max(last_bar_index - calcBars - L, 0)
float logit = na

if barstate.isconfirmed
    array.push(RNG, rngNow)
    array.push(BODY, math.abs(close - open))
    if array.size(RNG) > NN
        array.shift(RNG)
        array.shift(BODY)
    prevD1 := goodN ? (hl2 - close) / nrm : na
    prevD2 := goodN ? (hlc3 - close) / nrm : na
    prevC  := close
    okRun  := ok ? okRun + 1 : 0
    nConf  += 1
    array.shift(WR)
    array.push(WR, ok ? r : 0.0)
    array.shift(WABS)
    array.push(WABS, ok ? math.abs(r) : 0.0)
    if ok and firstR < 0
        firstR := bar_index

array.fill(DBG, na)
if barstate.isconfirmed and bar_index >= heavyFrom
    hc += 1
    convStep(hc, ok ? r / S_SD : 0.0, ok ? st1 / S_SD : 0.0, ok ? st2 / S_SD : 0.0, ok ? gg / LVL_SD : 0.0)
    if okRun >= L and bar_index - (L - 1) >= NN and hc >= L - 1
        logit := computeLogit(hc, HST)

'''

HEAD = r'''//@version=6
// SPDX-License-Identifier: Apache-2.0
// © TrendyCoins
// {title_line} — generated by mri/pine_nt3.py{source_note}; do not edit by hand.
//
// What the line shows: M — how many times a 64-bar window of this shape is more frequent (orange) or less frequent
// (blue) in real markets than in a random walk with the same volatility (bar-mirror surrogate in blocks of 4 bars).
// M = 1 — indistinguishable from random. The SuMo classifier (a streaming CNN, {n_w} weights{weights_where}) scores
// the window and the 200 bars before it, in bars only (no clock, no calendar); Platt calibration of the symbol turns the
// score into M. M is a frequency ratio of the window's shape, not a forecast.
{calib_note}// No repaint: the state changes only on confirmed bars; a window over a gap in the data is not drawn.
// Passport: weights {source}, data up to {data_end}, export of code {code_version}.
indicator("{title}", shorttitle = "{short}", overlay = false, precision = 2, max_labels_count = 500)

// ---------------------------------------------------------------- inputs
float thr      = input.float(3.0, "Threshold M", minval = 1.0, step = 0.25, tooltip = "Visual only: above the threshold the line is bright, below it is pale; dashed line at the threshold; optional marks on the price chart at M >= threshold.")
int   paleT    = input.int(70, "Transparency below the threshold", minval = 0, maxval = 100)
color colMore  = input.color(color.rgb(242, 140, 40), "More frequent than random", tooltip = "M > 1 on the more frequent side: windows of this shape are M times more frequent in real markets than in a random walk.")
color colLess  = input.color(color.rgb(47, 128, 237), "Less frequent than random", tooltip = "M > 1 on the less frequent side: windows of this shape are M times less frequent in real markets than in a random walk.")
bool  showMarks = input.bool(false, "Marks on the price chart at M >= threshold")
bool  dbg      = input.bool(false, "Debug values (parity check)", group = "Advanced", tooltip = "Plots the internal values of each bar for checking against the Python reference (Export chart data).")
int   calcBars = input.int({calc_bars}, "Compute only the last N bars (0 = all history)", minval = 0, group = "Advanced", tooltip = "The network runs on the last N bars only (TradingView limits the script time on the whole history).")

// ---------------------------------------------------------------- symbol table: floor of the window divisor, Platt of the symbol, ceilings of M
{tables}
var map<string, float> FLOORS = buildFloors()
var map<string, float> PLA    = buildPlattA()
var map<string, float> PLB    = buildPlattB()
var map<string, float> CEILO  = buildCeilMore()
var map<string, float> CEILB  = buildCeilLess()
bool  known = syminfo.prefix == "BINANCE" and not str.endswith(syminfo.ticker, ".P") and map.contains(FLOORS, syminfo.ticker)
var float FLOOR   = known ? map.get(FLOORS, syminfo.ticker) : {floor_unknown}
var float PLATT_A = known ? map.get(PLA, syminfo.ticker) : {platt_a_unknown}
var float PLATT_B = known ? map.get(PLB, syminfo.ticker) : {platt_b_unknown}
var float CEIL_O  = known ? map.get(CEILO, syminfo.ticker) : {ceil_o_unknown}   // checked range of M, more frequent side
var float CEIL_B  = known ? map.get(CEILB, syminfo.ticker) : {ceil_b_unknown}   // checked range of M, less frequent side
bool  tf5   = timeframe.period == "5"
float GAPMS = 1.5 * timeframe.in_seconds() * 1000.0   // interval > 1.5 bars -> grid gap

'''

FOOT = r'''// ---------------------------------------------------------------- display
float ll    = na(logit) ? na : PLATT_A * logit + PLATT_B
float Mraw  = na(ll) ? na : math.exp(math.abs(ll))
bool  more  = not na(ll) and ll > 0
bool  less  = not na(ll) and ll < 0
float Mshow = na(Mraw) ? na : math.min(Mraw, more ? CEIL_O : CEIL_B)
color baseC = more ? colMore : colLess
plot(Mshow, "M", color = Mraw >= thr ? baseC : color.new(baseC, paleT), linewidth = 2)
hline(1.0, "Random (M = 1)", color = color.gray, linestyle = hline.style_solid)
hline(thr, "Threshold", color = color.gray, linestyle = hline.style_dashed)
// at the ceiling: "N+" once at the start of each run
bool atO = more and Mraw >= CEIL_O
bool atB = less and Mraw >= CEIL_B
if barstate.isconfirmed and atO and not atO[1]
    label.new(bar_index, CEIL_O, str.tostring(CEIL_O) + "+", style = label.style_none, textcolor = colMore, size = size.small)
if barstate.isconfirmed and atB and not atB[1]
    label.new(bar_index, CEIL_B, str.tostring(CEIL_B) + "+", style = label.style_none, textcolor = colLess, size = size.small)
plotshape(showMarks and more and Mraw >= thr, "Mark: more frequent than random", shape.circle, location.abovebar, colMore, size = size.tiny, force_overlay = true)
plotshape(showMarks and less and Mraw >= thr, "Mark: less frequent than random", shape.circle, location.abovebar, colLess, size = size.tiny, force_overlay = true)

// ---------------------------------------------------------------- messages
var table MSG = table.new(position.top_right, 1, 3, bgcolor = color.new(color.gray, 85))
if barstate.islast
    table.clear(MSG, 0, 0, 0, 2)
    int row = 0
    if not tf5
        table.cell(MSG, 0, row, "MRI: the model was trained and checked on 5-minute bars only", text_size = size.small, text_color = chart.fg_color)
        row += 1
    if not known
        table.cell(MSG, 0, row, "{unknown_msg}", text_size = size.small, text_color = chart.fg_color)
        row += 1
    int left = (firstR < 0 ? NN + 1 : math.max(firstR, NN + 1)) + L - 1 - bar_index
    if left > 0 and calcBars == 0
        table.cell(MSG, 0, row, "MRI warm-up: " + str.tostring(left) + " bars to the first value", text_size = size.small, text_color = chart.fg_color)

// ---------------------------------------------------------------- debug values (parity with the Python reference, R10 addition 2)
display_dbg = dbg ? display.all : display.none
plot(logit, "dbg logit", display = display_dbg, precision = 10)
plot(Mraw, "dbg M", display = display_dbg, precision = 10)
plot(nrm, "dbg norm", display = display_dbg, precision = 10)
plot(r, "dbg r", display = display_dbg, precision = 10)
plot(st1, "dbg step_hl2", display = display_dbg, precision = 10)
plot(st2, "dbg step_hlc3", display = display_dbg, precision = 10)
plot(gg, "dbg g", display = display_dbg, precision = 10)
plot(array.get(DBG, 0), "dbg D", display = display_dbg, precision = 10)
plot(array.get(DBG, 1), "dbg x_end", display = display_dbg, precision = 10)
plot(array.get(DBG, 2), "dbg sgn", display = display_dbg, precision = 10)
{dbg_side}
plot(nConf, "dbg confirmed", display = display_dbg, precision = 0)
plot(bar_index, "dbg bar_index", display = display_dbg, precision = 0)
'''


def symbol_tables(rows, fields):
    """Pine functions building the symbol maps: rows {symbol: {field: value}}, fields [(function, field)]."""
    out = []
    for fn, key in fields:
        puts = '\n'.join(f'    map.put(m, "{a}", {fmt(v[key])})' for a, v in rows.items())
        out.append(f'{fn}() =>\n    map<string, float> m = map.new<string, float>()\n{puts}\n    m\n')
    return '\n'.join(out)


TABLE_FIELDS = [('buildFloors', 'floor'), ('buildPlattA', 'platt_a'), ('buildPlattB', 'platt_b'),
                ('buildCeilMore', 'ceil_more'), ('buildCeilLess', 'ceil_less')]


LIB_NAMES = ('MRIWeightsNT3a', 'MRIWeightsNT3b', 'MRIWeightsNT3c')
LIB_ALIASES = ('WA', 'WB', 'WC')
TV_USER = 'TrendyCoins'         # TradingView user name the libraries are published under (import path)
STR_MAX = 3999                  # characters per weight string
ASCII_ALPHA = ''.join(chr(c) for c in range(33, 127) if chr(c) not in '"\\/')     # 91 printable characters
ASCII_N = len(ASCII_ALPHA)      # 91 ** 5 > 2 ** 32: 5 characters per float32


def dense_strings(vals, str_max=STR_MAX):
    enc = encode_dense(vals)
    per = (str_max // 3) * 3
    return [enc[i:i + per] for i in range(0, len(enc), per)]


def encode_ascii(vals):
    """float32 -> 5 characters of ASCII_ALPHA each (base 91 of the 32 bits, most significant digit first)."""
    bits = np.asarray(vals, dtype=np.float32).view(np.uint32).astype(np.int64)
    out = []
    for b in bits:
        d = []
        for _ in range(5):
            d.append(int(b % ASCII_N))
            b //= ASCII_N
        out.append(''.join(ASCII_ALPHA[x] for x in reversed(d)))
    return ''.join(out)


def decode_ascii(s):
    """Python mirror of the Pine ASCII decoder: exact float32 values."""
    idx = {c: i for i, c in enumerate(ASCII_ALPHA)}
    out = []
    for k in range(len(s) // 5):
        b = 0.0
        for c in s[5 * k:5 * k + 5]:
            b = b * ASCII_N + idx[c]
        sg = -1.0 if b >= 2147483648.0 else 1.0
        u = b - 2147483648.0 if b >= 2147483648.0 else b
        e = float(np.floor(u / 8388608.0))
        m = u - e * 8388608.0
        out.append(sg * (m * 2.0 ** -149 if e == 0 else (1.0 + m / 8388608.0) * 2.0 ** (e - 127)))
    return out


ALPHA_BLOCK = 'buildAlpha() =>\n    map<string, int> m = map.new<string, int>()\n' \
              '    array<string> cs = str.split("{alpha}", "")\n    for [i, c] in cs\n        map.put(m, c, i)\n    m\n'

DECODE_FLOAT = r"""        float sg = b >= 2147483648.0 ? -1.0 : 1.0
        float u = b >= 2147483648.0 ? b - 2147483648.0 : b
        float e = math.floor(u / 8388608.0)
        float m = u - e * 8388608.0
        float v = e == 0 ? m * math.pow(2.0, -149) : (1.0 + m / 8388608.0) * math.pow(2.0, e - 127)
"""


def weights_ascii(vals, str_max=STR_MAX):
    """Pine code: ASCII alphabet map, 5-character decoder; each string inside its own function wNN() that decodes it into
       WT at run time, so that the global scope holds only the calls ('The main body of the script is too long' when the
       strings stood in the `var` initializers of the global scope)."""
    enc = encode_ascii(vals)
    per = (str_max // 5) * 5
    strs = [enc[i:i + per] for i in range(0, len(enc), per)]
    head = (ALPHA_BLOCK.format(alpha=ASCII_ALPHA) + '\n'
            'decodeInto(array<float> dst, map<string, int> A, string s) =>\n'
            '    array<string> cs = str.split(s, "")\n'
            '    int n = math.floor(array.size(cs) / 5.0)\n'
            '    for k = 0 to n - 1\n'
            '        float b = 0.0\n'
            '        for j = 0 to 4\n'
            f'            b := b * {ASCII_N}.0 + map.get(A, array.get(cs, 5 * k + j))\n'
            + DECODE_FLOAT +
            '        array.push(dst, sg * v)\n'
            '    true\n\n'
            'var map<string, int> ALPHA = buildAlpha()\n'
            'var array<float> WT = array.new<float>(0)\n')
    funcs = '\n'.join(f'w{j:02d}() =>\n    decodeInto(WT, ALPHA, "{x}")' for j, x in enumerate(strs))
    calls = '\n'.join(f'var bool WOK{j:02d} = w{j:02d}()' for j in range(len(strs)))
    return (head + '\n' + funcs + '\n\n' + calls,
            f'{len(vals)} float32, {len(enc)} ASCII characters (5 per weight) in {len(strs)} strings, one function each')


def split_parts(n, k):
    """n strings into k consecutive parts as equal as possible: [(start, end)]."""
    q, r = divmod(n, k)
    out, a = [], 0
    for i in range(k):
        b = a + q + (1 if i < r else 0)
        out.append((a, b))
        a = b
    return out


def weights_libs(vals, user=TV_USER):
    """Pine code of the indicator (imports + alphabet + appends) and the texts of the weight libraries {name: text}."""
    strs = dense_strings(vals)
    per_w = (STR_MAX // 3)
    parts = split_parts(len(strs), len(LIB_NAMES))
    alpha = ''.join(chr(ALPHA0 + i) for i in range(ALPHA_N))
    libs, calls = {}, []
    for i, ((a, b), name, alias) in enumerate(zip(parts, LIB_NAMES, LIB_ALIASES)):
        consts = '\n'.join(f'const string S{j:02d} = "{strs[j]}"' for j in range(a, b))
        funcs = '\n\n'.join(
            f'// @function Weights {j * per_w} .. {min((j + 1) * per_w, len(vals)) - 1} of the model (string {j}).\n'
            '// @param A Alphabet map of the indicator: character -> 0 .. 2047.\n'
            '// @returns Decoded float32 weights in the flat order of the model.\n'
            f'export w{j:02d}(map<string, int> A) =>\n    decodeArr(S{j:02d}, A)' for j in range(a, b))
        libs[name] = (
            '//@version=6\n// SPDX-License-Identifier: Apache-2.0\n// © TrendyCoins\n'
            f'// @description Weights of the Market Randomness Index (SuMo classifier NT3_PRV_ch48), part {i + 1} of '
            f'{len(LIB_NAMES)}: float32, 3 characters each (alphabet of the indicator). Generated by mri/pine_nt3.py; do '
            'not edit by hand.\n'
            f'library("{name}", overlay = false)\n\n{consts}\n\n'
            'decodeArr(string s, map<string, int> A) =>\n'
            '    array<float> dst = array.new<float>(0)\n'
            '    array<string> cs = str.split(s, "")\n'
            '    int n = math.floor(array.size(cs) / 3.0)\n'
            '    for k = 0 to n - 1\n'
            '        float b = map.get(A, array.get(cs, 3 * k)) * 4194304.0 + map.get(A, array.get(cs, 3 * k + 1)) * 2048.0 + '
            'map.get(A, array.get(cs, 3 * k + 2))\n'
            + DECODE_FLOAT +
            '        array.push(dst, sg * v)\n'
            '    dst\n\n' + funcs + '\n')
        calls += [f'var bool WOK{j:02d} = appendW(WT, {alias}.w{j:02d}(ALPHA))' for j in range(a, b)]
    imports = '\n'.join(f'import {user}/{name}/1 as {alias}' for name, alias in zip(LIB_NAMES, LIB_ALIASES))
    block = (ALPHA_BLOCK.format(alpha=alpha) + '\n'
             'appendW(array<float> dst, array<float> src) =>\n    array.concat(dst, src)\n    true\n\n'
             'var map<string, int> ALPHA = buildAlpha()\n'
             'var array<float> WT = array.new<float>(0)\n' + '\n'.join(calls))
    title = (f'{len(vals)} float32 in {len(LIB_NAMES)} libraries ({", ".join(LIB_NAMES)}), 3 characters each, '
             'decoded once')
    return block, title, imports, libs


def make_script(p, table, unknown, passport, title='Market Randomness Index', short='MRI', calib_note='',
                source_note=' from export/mri_model_nt3.json', calc_bars=0,
                unknown_msg='MRI: calibrated on Binance spot; accuracy on this chart is not verified', weights='libs',
                tv_user=TV_USER):
    """The indicator text and its libraries: (script, {library name: text}). p: Params (NT3 shapes); table: {symbol:
       floor, platt_a, platt_b, ceil_more, ceil_less}; unknown: the same fields for a symbol outside the table; passport:
       weights_source, data_end, code_version. weights: 'libs' (3 imported libraries), 'dense' (CJK strings inside the
       script — too large for TradingView), 'ascii' (ASCII strings inside, 5 characters per weight — size test)."""
    vals = flat_values(p)
    ch, hidden, n_side = p.conv[0].shape[0], p.head0_w.shape[0], len(p.side_mu)
    n_feat = 3 * ch + n_side
    assert p.head0_w.shape[1] == n_feat and len(vals) == p.n_params
    mats, off = matrix_offsets(ch, hidden, n_feat)
    mdecl = '\n'.join(f"var matrix<float> {nm:<4} = toMat(WT, {o}, {r}, {c})" for nm, r, c, o in mats)
    libs, imports = {}, ''
    if weights == 'libs':
        wblock, wtitle, imports, libs = weights_libs(vals, tv_user)
        where = f' in {len(libs)} libraries imported by this script'
    elif weights == 'dense':
        wblock, n_str, n_chars = weights_block(vals)
        wtitle, where = f'{len(vals)} float32, {n_chars} characters in {n_str} strings', ' inside this script'
    elif weights == 'ascii':
        wblock, wtitle = weights_ascii(vals)
        where = ' inside this script'
    else:
        raise ValueError(weights)
    chunks = '\n'.join(f'float P{k + 1}h = clipZ((close[{e}] - open[{s}]) / math.sqrt({n}.0) / rmedW)   // {n} bars'
                       for k, (n, e, s) in enumerate(chunk_offsets()))
    dbg_side = '\n'.join(f'plot(array.get(DBG, {3 + j}), "dbg side_{nm}", display = display_dbg, precision = 10)'
                         for j, nm in enumerate(SIDE_NAMES))
    head = HEAD.format(title_line=title if short == 'MRI' else f'{title} ({short})', source_note=source_note,
                       n_w=len(vals), weights_where=where, calib_note=calib_note, source=passport['weights_source'],
                       data_end=passport['data_end'], code_version=passport['code_version'], title=title, short=short,
                       calc_bars=calc_bars, tables=symbol_tables(table, TABLE_FIELDS),
                       floor_unknown=fmt(unknown['floor']), platt_a_unknown=fmt(unknown['platt_a']),
                       platt_b_unknown=fmt(unknown['platt_b']), ceil_o_unknown=fmt(unknown['ceil_more']),
                       ceil_b_unknown=fmt(unknown['ceil_less']))
    if imports:                                    # import statements right after the declaration statement
        decl = head.index('\n', head.index('indicator(')) + 1
        head = (head[:decl] + '// weight libraries (pine/lib/), published under the author\'s TradingView user name\n'
                + imports + '\n' + head[decl:])
    consts = CONSTS.format(ch=ch, hidden=hidden, n_side=n_side, sq199=fmt_d(np.sqrt(199.0)), s_sd=fmt(p.s_sd),
                           lvl_sd=fmt(p.lvl_sd), side_mu=float_list(p.side_mu), side_sd=float_list(p.side_sd),
                           signed=', '.join('1.0' if s else '0.0' for s in SIDE_SIGNED[2:]))
    core = CORE.format(weights_title=wtitle, weights=wblock, matrices=mdecl, off_h3=off, off_h3_end=off + hidden,
                       off_hb3=off + hidden, chunks=chunks)
    foot = FOOT.format(dbg_side=dbg_side, unknown_msg=unknown_msg)
    return head + consts + core + foot, libs


def fmt_d(x):
    """Positional decimal of a float64 (no exponent)."""
    s = np.format_float_positional(np.float64(x), unique=True, trim='-')
    return s if '.' in s else s + '.0'


CALC_BARS_DEFAULT = 0          # 'Compute only the last N bars': set from the Pine Profiler measurement


def make_indicator(ex, calc_bars=CALC_BARS_DEFAULT, weights='libs', tv_user=TV_USER, **kw):
    """pine/mri.pine (and its libraries) from the NT3 export dict (export/mri_model_nt3.json): (script, {name: text})."""
    from .params import from_dict
    p = from_dict(ex['model'])
    if ex['tie_rule']['eps'] != 1e-4 or ex['model']['id'] != 'NT3_PRV_ch48' or len(p.side_mu) != len(SIDE_NAMES):
        raise ValueError('the template has EPS = 0.0001 and the 17 side inputs of NT3_PRV_ch48')
    ceil, cal, fl = ex['display']['ceiling'], ex['calibration'], ex['floors']
    table = {a: {'floor': fl['floor_L64'][a], 'platt_a': cal['per_symbol'][a]['platt_a'],
                 'platt_b': cal['per_symbol'][a]['platt_b'], 'ceil_more': ceil['by_symbol'][a]['more'],
                 'ceil_less': ceil['by_symbol'][a]['less']} for a in fl['floor_L64']}
    unk = {'floor': fl['unknown_symbol'], 'platt_a': cal['unknown_symbol']['platt_a'],
           'platt_b': cal['unknown_symbol']['platt_b'], 'ceil_more': ceil['unknown_symbol']['more'],
           'ceil_less': ceil['unknown_symbol']['less']}
    txt = lambda v: f"{v:g}"
    pool = ceil['pool']
    note = ('// Calibration: Platt of each table symbol, checked out of sample by half-years per symbol; each symbol and side has\n'
            f'// its ceiling (at most {txt(pool["more"])} more frequent / {txt(pool["less"])} less frequent; for symbols outside '
            f'the table {txt(unk["ceil_more"])} / {txt(unk["ceil_less"])}\n'
            '// with the global Platt); above it the line stays at the ceiling, marked "N+".\n')
    return make_script(p, table, unk, ex['passport'], calib_note=note, calc_bars=calc_bars, weights=weights,
                       tv_user=tv_user, **kw)


def main():
    import argparse
    from .export_nt3 import load_export
    ap = argparse.ArgumentParser()
    ap.add_argument('--ascii-test', action='store_true', help='also pine/test/mri_ascii_single.pine')
    ap.add_argument('--tv-user', default=TV_USER, help='TradingView user name of the weight libraries')
    a = ap.parse_args()
    ex = load_export(os.path.join(ROOT, 'export/mri_model_nt3.json'))
    script, libs = make_indicator(ex, tv_user=a.tv_user)
    outs = {'pine/mri.pine': script, **{f'pine/lib/{name}.pine': t for name, t in libs.items()}}
    if a.ascii_test:
        outs['pine/test/mri_ascii_single.pine'] = make_indicator(
            ex, weights='ascii', title='Market Randomness Index size test', short='MRI test',
            source_note=' (single script, ASCII weights: size test only)')[0]
    for path, s in outs.items():
        full = os.path.join(ROOT, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, 'w', encoding='utf-8') as fh:
            fh.write(s)
        print(f'{path}: {len(s)} chars, {len(s.encode())} bytes UTF-8, {len(s.splitlines())} lines')


if __name__ == '__main__':
    main()
