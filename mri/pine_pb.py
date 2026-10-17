"""Pine Script v6 generator of the PB_s64h24_d1 indicator — one script, no libraries (session 4).

    python -m mri.pine_pb               # pine/proto_pb.pine: the production template on the real weights, ruler and
                                        # ceilings, named "prototype" until the parity check (R10 addendum 3)
    python -m mri.pine_pb --production  # pine/mri.pine (after the TradingView measurements: default N of
                                        # "Compute only the last N bars")

Weights: the dense float32 strings of mri/pine_gen.py (3 CJK characters per float32, bit for bit), 18 697 numbers =
56 091 characters in strings of <= 3 999; every string sits in its own function wNN() that decodes it into WT at run time,
the global scope holds only the calls (session 3 probe: this passes the request size and the global-scope limit; tokens
~1 per string character). Flat order: layer 1 taps 0..2 W[:, :, tap] row-major (64, 4); layers 2..4: depthwise taps 0..2
(64 each), then the pointwise W row-major (64, 64); head0 W row-major (24, 209); head0 b (24); head3 W (24); head3 b (1).

Compute per confirmed bar (the same as mri.pb.StreamingPB; tests/pine_sim_pb.py transliterates the template):
    conv streams + / - in interleaved ring matrices (CH rows); layer 1 — 3 matrix.mult (64 x 4) by (4 x 2); layers 2..4 —
    per channel 3 taps with the dilation from the ring of the previous layer (elementwise), then one matrix.mult (64 x 64)
    by (64 x 2), then ReLU; layer 4 into two (CH x 33) matrices; pooled mean / max / last over 33 positions, / D; head
    209 -> 24 -> 1. History inputs — as NT3 (mri/pine_nt3.py: per-bar scale series read back with [63] / [65],
    ta.highest / ta.lowest / ta.sma over 199 bars, path chunks at fixed offsets). Line M: z by the ruler (37 knots), the
    live part — `var` state n1, n2, d, n — changes only on barstate.isconfirmed and only on a valid window.
"""
import os

import numpy as np

from .nt3 import SIDE_NAMES, SIDE_SIGNED
from .pine_gen import ALPHA0, ALPHA_N, encode_dense, fmt
from .pine_nt3 import ALPHA_BLOCK, DECODE_FLOAT, STR_MAX, chunk_offsets, float_list, fmt_d, symbol_tables

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L = 64
PROTO = 'pine/proto_pb.pine'
PRODUCTION = 'pine/mri.pine'
SIZE_LIMIT_UTF8 = 215 * 1024           # largest single script that passed the request size on TradingView (session 3)


def flat_values(p):
    """Weights in the flat order of the template (float32)."""
    out = []
    for k in range(3):
        out.extend(p.conv[0][:, :, k].ravel())
    for i in (1, 2, 3):
        for k in range(3):
            out.extend(p.conv[i][:, 0, k])
        out.extend(p.pw[i - 1].ravel())
    for a in (p.head0_w, p.head0_b, p.head3_w, p.head3_b):
        out.extend(np.ravel(a))
    return np.array(out, dtype=np.float32)


def weight_decls(ch, hidden, n_feat):
    """Pine declarations of the weight matrices / arrays (name, kind, rows, cols, offset) in the flat order; then H3, HB3."""
    out, off = [], 0
    for k in range(3):
        out.append((f'K1_{k}', 'mat', ch, 4, off))
        off += ch * 4
    for li in (2, 3, 4):
        for k in range(3):
            out.append((f'D{li}_{k}', 'arr', ch, 1, off))
            off += ch
        out.append((f'P{li}', 'mat', ch, ch, off))
        off += ch * ch
    out.append(('H0', 'mat', hidden, n_feat, off))
    off += hidden * n_feat
    out.append(('HB0', 'mat', hidden, 1, off))
    off += hidden
    return out, off


def dense_strings(vals, str_max=STR_MAX):
    enc = encode_dense(vals)
    per = (str_max // 3) * 3
    return [enc[i:i + per] for i in range(0, len(enc), per)]


def weights_in_functions(vals):
    """Pine code: CJK alphabet map, 3-character decoder, one function wNN() per weight string, the calls at the global
       scope (var initializers, run once)."""
    strs = dense_strings(vals)
    alpha = ''.join(chr(ALPHA0 + i) for i in range(ALPHA_N))
    head = (ALPHA_BLOCK.format(alpha=alpha) + '\n'
            'decodeInto(array<float> dst, map<string, int> A, string s) =>\n'
            '    array<string> cs = str.split(s, "")\n'
            '    int n = math.floor(array.size(cs) / 3.0)\n'
            '    for k = 0 to n - 1\n'
            '        float b = map.get(A, array.get(cs, 3 * k)) * 4194304.0 + map.get(A, array.get(cs, 3 * k + 1)) * 2048.0 + '
            'map.get(A, array.get(cs, 3 * k + 2))\n'
            + DECODE_FLOAT +
            '        array.push(dst, sg * v)\n'
            '    true\n\n'
            'var map<string, int> ALPHA = buildAlpha()\n'
            'var array<float> WT = array.new<float>(0)\n')
    funcs = '\n'.join(f'w{j:02d}() =>\n    decodeInto(WT, ALPHA, "{x}")' for j, x in enumerate(strs))
    calls = '\n'.join(f'var bool WOK{j:02d} = w{j:02d}()' for j in range(len(strs)))
    title = f'{len(vals)} float32, {sum(map(len, strs))} characters (3 per weight) in {len(strs)} strings, one function each'
    return head + '\n' + funcs + '\n\n' + calls, title


HEAD = r'''//@version=6
// SPDX-License-Identifier: Apache-2.0
// © TrendyCoins
// {title_line} — generated by mri/pine_pb.py from export/mri_model_pb.json; do not edit by hand.
//
// What the line shows: M — how many times a 64-bar window of this shape is more frequent (orange) or less frequent
// (blue) in real markets than in a random walk with the same volatility (bar-mirror surrogate in blocks of 4 bars).
// M = 1 — indistinguishable from random. The SuMo classifier (a streaming CNN, {n_w} weights inside this script) scores
// the window and the 200 bars before it, in bars only (no clock, no calendar). The score becomes M without constants of
// the symbol: a ruler of the model (z — the place of the score among random windows, one table for every symbol) and a
// live part of this chart (how the scores of this chart's windows are spread: EMA of z and z^2 over {span} bars);
// M is the ratio of the two bells "windows of this chart" / "random windows". On a random market M stays near 1.
// M is a frequency ratio of the window's shape, not a forecast.
// Checked range: each table symbol and side has its ceiling (at most {pool_o} more frequent / {pool_b} less frequent; for
// symbols outside the table {unk_o} / {unk_b}); above it the line stays at the ceiling, marked "N+".
// No repaint: the state changes only on confirmed bars; a window over a gap in the data is not drawn.
// Passport: weights {source}, data up to {data_end}, export of code {code_version}.
indicator("{title}", shorttitle = "{short}", overlay = false, precision = 2, max_labels_count = 500)

// ---------------------------------------------------------------- inputs
float thr      = input.float(3.0, "Threshold M", minval = 1.0, step = 0.25, tooltip = "Visual only: above the threshold the line is bright, below it is pale; dashed line at the threshold; optional marks on the price chart at M >= threshold.")
int   paleT    = input.int(70, "Transparency below the threshold", minval = 0, maxval = 100)
color colMore  = input.color(color.rgb(242, 140, 40), "More frequent than random", tooltip = "M > 1 on the more frequent side: windows of this shape are M times more frequent in real markets than in a random walk.")
color colLess  = input.color(color.rgb(47, 128, 237), "Less frequent than random", tooltip = "M > 1 on the less frequent side: windows of this shape are M times less frequent in real markets than in a random walk.")
bool  showMarks = input.bool(false, "Marks on the price chart at M >= threshold")
int   spanIn   = input.int({span}, "Live part: span, bars", minval = 2, group = "Advanced", tooltip = "EMA span of the live part (spread of the scores of this chart). The promise of the line is checked at {span} only.")
int   warmIn   = input.int({warm}, "Line warm-up, windows", minval = 0, group = "Advanced", tooltip = "The line is shown from the window number {warm} of the chart (windows counted from 0). The promise is checked at {warm} only.")
bool  dbg      = input.bool(false, "Debug values (parity check)", group = "Advanced", tooltip = "Plots the internal values of each bar for checking against the Python reference (Export chart data).")
int   calcBars = input.int({calc_bars}, "Compute only the last N bars (0 = all history)", minval = 0, group = "Advanced", tooltip = "The network and the live part run on the last N bars only (TradingView limits the script time on the whole history); the line appears after the warm-up.")

// ---------------------------------------------------------------- symbol table: floor of the window divisor, ceilings of M
{tables}
var map<string, float> FLOORS = buildFloors()
var map<string, float> CEILO  = buildCeilMore()
var map<string, float> CEILB  = buildCeilLess()
bool  known = syminfo.prefix == "BINANCE" and not str.endswith(syminfo.ticker, ".P") and map.contains(FLOORS, syminfo.ticker)
var float FLOOR  = known ? map.get(FLOORS, syminfo.ticker) : {floor_unknown}
var float CEIL_O = known ? map.get(CEILO, syminfo.ticker) : {ceil_o_unknown}   // checked range of M, more frequent side
var float CEIL_B = known ? map.get(CEILB, syminfo.ticker) : {ceil_b_unknown}   // checked range of M, less frequent side
bool  tf5   = timeframe.period == "5"
float GAPMS = 1.5 * timeframe.in_seconds() * 1000.0   // interval > 1.5 bars -> grid gap

'''

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
const int   SPAN0  = {span}         // live part: checked span, bars
const int   WARM0  = {warm}          // line warm-up: checked number of windows
const float S_LO   = {s_lo}
const float PEAK_EPS = 0.000000000001
var array<float> SIDE_MU = array.from({side_mu})
var array<float> SIDE_SD = array.from({side_sd})
var array<float> HSIGNED = array.from({signed})   // history inputs that take the orientation sign
// ruler of the model: 37 quantiles of the scores of random (surrogate) windows and their places on the bell, Phi^-1(knot)
var array<float> RQ = array.from({ruler_q})
var array<float> RZ = array.from({ruler_z})

'''

CORE = r'''// ---------------------------------------------------------------- weights: {weights_title}
{weights}

toMat(array<float> w, int off, int nr, int nc) =>
    matrix<float> m = matrix.new<float>(nr, nc, 0.0)
    for i = 0 to nr - 1
        for j = 0 to nc - 1
            matrix.set(m, i, j, array.get(w, off + i * nc + j))
    m

toArr(array<float> w, int off, int n) =>
    array.copy(array.slice(w, off, off + n))

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
// live part of the line M: EMA of z and z^2 over the valid windows of this chart (n1, n2 / d), window number nW
var float n1 = 0.0
var float n2 = 0.0
var float dd = 0.0
var int   nW = -1
var float KEMA = 1.0 - 2.0 / (spanIn + 1.0)

// conv rings: inputs (4 x 2*3), layers 1..3 (CH x 2*5, 2*9, 2*17); layer 4 per stream (CH x 33)
var matrix<float> R0  = matrix.new<float>(4, 6, 0.0)
var matrix<float> R1  = matrix.new<float>(CH, 10, 0.0)
var matrix<float> R2  = matrix.new<float>(CH, 18, 0.0)
var matrix<float> R3  = matrix.new<float>(CH, 34, 0.0)
var matrix<float> R4P = matrix.new<float>(CH, 33, 0.0)
var matrix<float> R4M = matrix.new<float>(CH, 33, 0.0)
var matrix<float> DWM = matrix.new<float>(CH, 2, 0.0)   // depthwise outputs of a separable layer (2 streams)
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

// separable layer: per channel 3 taps with dilation d (elementwise), then the 1 x 1 channel mixing (one matrix.mult)
sep3(matrix<float> R, int depth, int cnt, int d, array<float> A0, array<float> A1, array<float> A2, matrix<float> PW) =>
    int c0 = 2 * slot(cnt, depth, 2 * d)
    int c1 = 2 * slot(cnt, depth, d)
    int c2 = 2 * slot(cnt, depth, 0)
    for i = 0 to CH - 1
        float a0 = array.get(A0, i)
        float a1 = array.get(A1, i)
        float a2 = array.get(A2, i)
        matrix.set(DWM, i, 0, a0 * matrix.get(R, i, c0) + a1 * matrix.get(R, i, c1) + a2 * matrix.get(R, i, c2))
        matrix.set(DWM, i, 1, a0 * matrix.get(R, i, c0 + 1) + a1 * matrix.get(R, i, c1 + 1) + a2 * matrix.get(R, i, c2 + 1))
    matrix.mult(PW, DWM)

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
    reluInto(sep3(R1, 5, cnt, 2, D2_0, D2_1, D2_2, P2), R2, 9, cnt)
    reluInto(sep3(R2, 9, cnt, 4, D3_0, D3_1, D3_2, P3), R3, 17, cnt)
    matrix<float> y4 = sep3(R3, 17, cnt, 8, D4_0, D4_1, D4_2, P4)
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

// z of a score by the ruler: a straight line between the knots, the edge value beyond the outer knots
rulerZ(float x) =>
    int n = array.size(RQ)
    float z = na
    if x < array.get(RQ, 0)
        z := array.get(RZ, 0)
    else if x >= array.get(RQ, n - 1)
        z := array.get(RZ, n - 1)
    else
        int j = 0
        while array.get(RQ, j + 1) <= x
            j += 1
        float q0 = array.get(RQ, j)
        z := x == q0 ? array.get(RZ, j) : (array.get(RZ, j + 1) - array.get(RZ, j)) / (array.get(RQ, j + 1) - q0) * (x - q0) + array.get(RZ, j)
    z

// log M of a window: ratio of the bells N(m, s^2) of this chart and N(0, 1) of random windows; m >= 0, s in [S_LO; 1],
// z not above the peak m / (1 - s^2)
logMOf(float z, float mE, float sE) =>
    float m = math.max(mE, 0.0)
    float s = math.min(math.max(sE, S_LO), 1.0)
    float zc = s < 1.0 ? math.min(z, m / math.max(1.0 - s * s, PEAK_EPS)) : z
    float lm = -(zc - m) * (zc - m) / (2.0 * s * s) - math.log(s) + zc * zc / 2.0
    lm

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
float zz    = na
float mE    = na
float sE    = na
float logM  = na

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
        zz    := rulerZ(logit)
        nW    += 1
        n1    := KEMA * n1 + zz
        n2    := KEMA * n2 + zz * zz
        dd    := KEMA * dd + 1.0
        mE    := n1 / dd
        sE    := math.sqrt(math.max(n2 / dd - mE * mE, 0.0))
        logM  := logMOf(zz, mE, sE)

'''

FOOT = r'''// ---------------------------------------------------------------- display
bool  shownM = not na(logM) and nW >= warmIn
float Mraw   = shownM ? math.exp(math.abs(logM)) : na
bool  more   = shownM and logM > 0
bool  less   = shownM and logM < 0
float Mshow  = na(Mraw) ? na : more ? math.min(Mraw, CEIL_O) : less ? math.min(Mraw, CEIL_B) : 1.0
color baseC  = more ? colMore : less ? colLess : color.gray
plot(Mshow, "M", color = Mraw >= thr ? baseC : color.new(baseC, paleT), linewidth = 2)
hline(1.0, "Random (M = 1)", color = color.gray, linestyle = hline.style_solid)
hline(thr, "Threshold", color = color.gray, linestyle = hline.style_dashed)
// above the ceiling: "N+" once at the start of each run
bool atO = more and Mraw > CEIL_O
bool atB = less and Mraw > CEIL_B
if barstate.isconfirmed and atO and not atO[1]
    label.new(bar_index, CEIL_O, str.tostring(CEIL_O) + "+", style = label.style_none, textcolor = colMore, size = size.small)
if barstate.isconfirmed and atB and not atB[1]
    label.new(bar_index, CEIL_B, str.tostring(CEIL_B) + "+", style = label.style_none, textcolor = colLess, size = size.small)
plotshape(showMarks and more and Mraw >= thr, "Mark: more frequent than random", shape.circle, location.abovebar, colMore, size = size.tiny, force_overlay = true)
plotshape(showMarks and less and Mraw >= thr, "Mark: less frequent than random", shape.circle, location.abovebar, colLess, size = size.tiny, force_overlay = true)

// ---------------------------------------------------------------- messages
var table MSG = table.new(position.top_right, 1, 4, bgcolor = color.new(color.gray, 85))
if barstate.islast
    table.clear(MSG, 0, 0, 0, 3)
    int row = 0
    if not tf5
        table.cell(MSG, 0, row, "MRI: the model was trained and checked on 5-minute bars only", text_size = size.small, text_color = chart.fg_color)
        row += 1
    if not known
        table.cell(MSG, 0, row, "{unknown_msg}", text_size = size.small, text_color = chart.fg_color)
        row += 1
    if spanIn != SPAN0 or warmIn != WARM0
        table.cell(MSG, 0, row, "MRI: span / warm-up differ from the checked " + str.tostring(SPAN0) + " / " + str.tostring(WARM0) + ": not checked", text_size = size.small, text_color = chart.fg_color)
        row += 1
    if nW < warmIn
        int left = nW >= 0 ? warmIn - nW : (firstR < 0 ? NN + 1 : math.max(firstR, NN + 1)) + L - 1 - bar_index + warmIn
        table.cell(MSG, 0, row, "MRI warm-up: " + str.tostring(left) + " bars to the first value", text_size = size.small, text_color = chart.fg_color)

// ---------------------------------------------------------------- debug values (parity with the Python reference, R10 addendum 3)
display_dbg = dbg ? display.all : display.none
plot(logit, "dbg logit", display = display_dbg, precision = 10)
plot(zz, "dbg z", display = display_dbg, precision = 10)
plot(mE, "dbg m_ema", display = display_dbg, precision = 10)
plot(sE, "dbg s_ema", display = display_dbg, precision = 10)
plot(logM, "dbg logM", display = display_dbg, precision = 10)
plot(na(logit) ? na : nW, "dbg n", display = display_dbg, precision = 0)
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

TABLE_FIELDS = [('buildFloors', 'floor'), ('buildCeilMore', 'ceil_more'), ('buildCeilLess', 'ceil_less')]
UNKNOWN_MSG = 'MRI: not a Binance spot symbol of the table: floor and ceilings of an unknown symbol, accuracy not verified'


def make_script(p, ruler, table, unknown, passport, title='Market Randomness Index', short='MRI', calc_bars=0,
                span=8064, warm=288, s_lo=0.75, unknown_msg=UNKNOWN_MSG):
    """The indicator text. p: Params (PB shapes, separable); ruler: (q, zk) — 37 numbers each; table: {symbol: floor,
       ceil_more, ceil_less}; unknown: the same fields for a symbol outside the table; passport: weights_source,
       data_end, code_version."""
    if p.pw is None:
        raise ValueError('the PB template needs separable layers 2..4')
    q, zk = (list(map(float, x)) for x in ruler)
    if len(q) != len(zk) or not np.all(np.diff(q) > 0):
        raise ValueError('ruler: strictly increasing quantiles and their z')
    vals = flat_values(p)
    ch, hidden, n_side = p.conv[0].shape[0], p.head0_w.shape[0], len(p.side_mu)
    n_feat = 3 * ch + n_side
    assert p.head0_w.shape[1] == n_feat and len(vals) == p.n_params
    decls, off = weight_decls(ch, hidden, n_feat)
    mdecl = '\n'.join(f"var matrix<float> {nm:<4} = toMat(WT, {o}, {r}, {c})" if kind == 'mat' else
                      f"var array<float>  {nm:<4} = toArr(WT, {o}, {r})" for nm, kind, r, c, o in decls)
    wblock, wtitle = weights_in_functions(vals)
    chunks = '\n'.join(f'float P{k + 1}h = clipZ((close[{e}] - open[{s}]) / math.sqrt({n}.0) / rmedW)   // {n} bars'
                       for k, (n, e, s) in enumerate(chunk_offsets()))
    dbg_side = '\n'.join(f'plot(array.get(DBG, {3 + j}), "dbg side_{nm}", display = display_dbg, precision = 10)'
                         for j, nm in enumerate(SIDE_NAMES))
    g = lambda v: f'{v:g}'
    pool = passport.get('ceil_pool', {'orange': max(v['ceil_more'] for v in table.values()),
                                      'blue': max(v['ceil_less'] for v in table.values())})
    head = HEAD.format(title_line=title if short == 'MRI' else f'{title} ({short})', n_w=len(vals), span=span, warm=warm,
                       pool_o=g(pool['orange']), pool_b=g(pool['blue']), unk_o=g(unknown['ceil_more']),
                       unk_b=g(unknown['ceil_less']), source=passport['weights_source'], data_end=passport['data_end'],
                       code_version=passport['code_version'], title=title, short=short, calc_bars=calc_bars,
                       tables=symbol_tables(table, TABLE_FIELDS), floor_unknown=fmt(unknown['floor']),
                       ceil_o_unknown=fmt(unknown['ceil_more']), ceil_b_unknown=fmt(unknown['ceil_less']))
    consts = CONSTS.format(ch=ch, hidden=hidden, n_side=n_side, sq199=fmt_d(np.sqrt(199.0)), s_sd=fmt(p.s_sd),
                           lvl_sd=fmt(p.lvl_sd), side_mu=float_list(p.side_mu), side_sd=float_list(p.side_sd),
                           signed=', '.join('1.0' if s else '0.0' for s in SIDE_SIGNED[2:]), span=span, warm=warm,
                           s_lo=fmt_d(s_lo), ruler_q=', '.join(fmt_d(x) for x in q), ruler_z=', '.join(fmt_d(x) for x in zk))
    core = CORE.format(weights_title=wtitle, weights=wblock, matrices=mdecl, off_h3=off, off_h3_end=off + hidden,
                       off_hb3=off + hidden, chunks=chunks)
    foot = FOOT.format(dbg_side=dbg_side, unknown_msg=unknown_msg)
    return head + consts + core + foot


# 'Compute only the last N bars' of the production script, from the Pine Profiler (BINANCE:BTCUSDT 5M, 21 832 bars):
# N = 0 — 23 s (heavy part ~1.0 ms per bar), N = 10000 — 12 s; a 40 000-bar chart with N = 10000 — ~13 s (limit 40 s)
CALC_BARS_DEFAULT = 10000


def make_indicator(ex, production=False, calc_bars=None):
    """The PB indicator from the export dict (export/mri_model_pb.json): production — pine/mri.pine (default N of the
       measurement); prototype (pine/proto_pb.pine, the parity check was made on it) — N = 0 and the prototype title."""
    from .params import from_dict
    p = from_dict(ex['model'])
    if (ex['tie_rule']['eps'] != 1e-4 or ex['model']['id'] != 'PB_s64h24_d1' or len(p.side_mu) != len(SIDE_NAMES)
            or ex['line']['ruler']['kind'] != 'glob_e'):
        raise ValueError('the template has EPS = 0.0001, the 17 side inputs and the glob_e ruler of PB_s64h24_d1')
    ln, fl = ex['line'], ex['floors']
    ceil = ln['ceilings']
    table = {a: {'floor': fl['floor_L64'][a], 'ceil_more': ceil['by_symbol'][a]['orange'],
                 'ceil_less': ceil['by_symbol'][a]['blue']} for a in fl['floor_L64']}
    unk = {'floor': fl['unknown_symbol'], 'ceil_more': ceil['unknown_symbol']['orange'],
           'ceil_less': ceil['unknown_symbol']['blue']}
    passport = {**ex['passport'], 'ceil_pool': ceil['pool']}
    kw = {} if production else {'title': 'Market Randomness Index prototype', 'short': 'MRI proto'}
    if calc_bars is None:
        calc_bars = CALC_BARS_DEFAULT if production else 0
    return make_script(p, (ln['ruler']['q'], ln['ruler']['zk']), table, unk, passport,
                       calc_bars=calc_bars, span=ln['live']['span_default'],
                       warm=ln['warmup']['windows_default'], s_lo=ln['live']['s_lo'], **kw)


def sizes(text):
    """Four size measures of a script: characters, UTF-8 bytes, UTF-16 bytes, bytes with non-ASCII as \\uXXXX."""
    n_wide = sum(ord(c) > 127 for c in text)
    return {'chars': len(text), 'utf8': len(text.encode()), 'utf16': len(text.encode('utf-16-le')),
            'escaped': len(text) + 5 * n_wide}


def main():
    import argparse
    from .export_pb import EXPORT, load_export
    ap = argparse.ArgumentParser()
    ap.add_argument('--production', action='store_true', help=f'write {PRODUCTION} instead of the prototype')
    a = ap.parse_args()
    ex = load_export(os.path.join(ROOT, EXPORT))
    path = PRODUCTION if a.production else PROTO
    s = make_indicator(ex, production=a.production)
    with open(os.path.join(ROOT, path), 'w', encoding='utf-8') as fh:
        fh.write(s)
    print(f'{path}: {sizes(s)}, {len(s.splitlines())} lines')
    if sizes(s)['utf8'] > SIZE_LIMIT_UTF8:
        raise SystemExit(f'{path}: larger than {SIZE_LIMIT_UTF8} bytes UTF-8 — shrink the code first (rules, session 4)')


if __name__ == '__main__':
    main()
