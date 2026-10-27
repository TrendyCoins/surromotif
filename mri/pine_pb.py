"""Pine Script v6 generator of the PB_s64h24_d1 indicator — one script, no libraries (sessions 4-5).

    python -m mri.pine_pb               # pine/proto_pb.pine: the production template on the real weights, ruler, floor
                                        # and ceiling, named "prototype" until the parity check (R10 addendum 4)
    python -m mri.pine_pb --production  # pine/mri.pine (after the TradingView measurements: default N of
                                        # "Compute only the last N bars"; no debug mode — decision of 2026-10-18)

Weights: the dense float32 strings of mri/pine_common.py (3 CJK characters per float32, bit for bit), 18 697 numbers =
56 091 characters in strings of <= 3 999; every string sits in its own function wNN() that decodes it into WT at run time,
the global scope holds only the calls (session 3 probe: this passes the request size and the global-scope limit; tokens
~1 per string character). Flat order: layer 1 taps 0..2 W[:, :, tap] row-major (64, 4); layers 2..4: depthwise taps 0..2
(64 each), then the pointwise W row-major (64, 64); head0 W row-major (24, 209); head0 b (24); head3 W (24); head3 b (1).

Compute per confirmed bar (the same as mri.pb.StreamingPB; tests/pine_sim_pb.py transliterates the template):
    conv streams + / - in interleaved ring matrices (CH rows); layer 1 — 3 matrix.mult (64 x 4) by (4 x 2); layers 2..4 —
    per channel 3 taps with the dilation from the ring of the previous layer (elementwise), then one matrix.mult (64 x 64)
    by (64 x 2), then ReLU; layer 4 into two (CH x 33) matrices; pooled mean / max / last over 33 positions, / D; head
    209 -> 24 -> 1. History inputs — as NT3 (mri/legacy/pine_nt3.py: per-bar scale series read back with [63] / [65],
    ta.highest / ta.lowest / ta.sma over 199 bars, path chunks at fixed offsets). Line M (session 5, rules R17-R19): one
    floor FLOOR for every symbol, z by the ruler (37 knots), the live part — formula A: `var` ring of 4096 z (index
    n % 4096), running S1, S2 and the window number nW — changes only on barstate.isconfirmed and only on a valid window
    (no array.shift on the ring); the line from nW >= 4096 with one mark at its start; one ceiling 10 for both colours.

Script layout (2026-10-27, open source): license, version, the name with a one-line description, what the indicator
shows and how it works (for Pine programmers of a beginner to middle level; nothing that points outside the script —
no files of this repository, exports, passports or commits), the declaration, then constants, inputs, weights, state,
functions, calculations, display and messages, with `// @function` / `// @param` / `// @returns` and `//@variable`
comments as in the Pine style guide. Display: the palette of trendycoins.com — silver #9D9D9D at M = 1 towards orange
#FF7A1A (more often in real markets) or grey #665F5A (more often on the random chart), the share of the side colour =
ln M / ln CEIL (the site mixes in OKLab, Pine's color.from_gradient in RGB: the same ends, a hardly visible difference
in between); a faint grid at M = 2, 4, 6, 8; marks on the price chart at M >= the level of one input (0 = no marks); only
the line M is listed in the Style tab (the rest has editable = false). Edit this file, not the generated scripts:
tests/test_pine_pb.py checks that pine/*.pine equal its output.
"""
import os
import re

import numpy as np

from .nt3 import SIDE_NAMES, SIDE_SIGNED
from .pine_common import (ALPHA0, ALPHA_N, ALPHA_BLOCK, DECODE_FLOAT, STR_MAX, chunk_offsets, encode_dense, float_list,
                          fmt, fmt_d)

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
    head = ('// @function Builds the alphabet of the weight strings: each of its 2048 characters -> its number, '
            '0..2047.\n'
            '// @returns (map<string, int>) The table that decodeInto() reads.\n'
            + ALPHA_BLOCK.format(alpha=alpha) + '\n'
            '// @function Decodes one weight string and appends its numbers to an array, in order.\n'
            '// @param dst (array<float>) The array that receives the numbers.\n'
            '// @param A (map<string, int>) The alphabet table from buildAlpha().\n'
            '// @param s (string) The weight string: 3 characters per number.\n'
            '// @returns (bool) Always true: a `var` declaration calls the function once, on the first bar.\n'
            'decodeInto(array<float> dst, map<string, int> A, string s) =>\n'
            '    array<string> cs = str.split(s, "")\n'
            '    int n = math.floor(array.size(cs) / 3.0)\n'
            '    for k = 0 to n - 1\n'
            '        // 3 characters = 3 digits in base 2048 = the 32 bits of one float32 number\n'
            '        float b = map.get(A, array.get(cs, 3 * k)) * 4194304.0 + map.get(A, array.get(cs, 3 * k + 1)) * 2048.0 + '
            'map.get(A, array.get(cs, 3 * k + 2))\n'
            '        // the float32 format: 1 sign bit, 8 exponent bits, 23 mantissa bits -> the exact value\n'
            + DECODE_FLOAT +
            '        array.push(dst, sg * v)\n'
            '    true\n\n'
            '//@variable The alphabet table, built once.\n'
            'var map<string, int> ALPHA = buildAlpha()\n'
            '//@variable All weights of the network in one flat list; the calls below fill it on the first bar.\n'
            'var array<float> WT = array.new<float>(0)\n')
    funcs = '\n'.join(f'w{j:02d}() =>\n    decodeInto(WT, ALPHA, "{x}")' for j, x in enumerate(strs))
    calls = '\n'.join(f'var bool WOK{j:02d} = w{j:02d}()' for j in range(len(strs)))
    title = f'{len(vals):,} float32 numbers, {sum(map(len, strs)):,} characters in {len(strs)} strings, one function each'
    return (head + '\n// One small function per weight string keeps every block of code within the size limits of Pine.\n'
            + funcs + '\n\n// Each call runs once, on the first bar ("var"), and appends its string to WT in the right order.\n'
            + calls), title


HEAD = r'''// SPDX-License-Identifier: Apache-2.0
// This source code is subject to the terms of the Apache License, Version 2.0.
// © TrendyCoins

//@version=6

// {title_line}: shows when the market does not look like a random walk.
//
// WHAT THE LINE SHOWS
// On every closed bar the indicator takes the last 64 bars (a "window") and answers one question: where does a
// situation like this occur more often — in real markets or on a random chart made of the same bars? The random chart
// keeps every bar, but a coin toss decides whether each stretch of 4 bars is flipped upside down: the moves keep their
// sizes and lose their order. The answer is the number M on the line and its color:
//   orange — windows like this occur M times more often in real markets: the market is not random right now, there is
//            something to look for;
//   grey   — M times more often on the random chart: right now the market behaves like a coin toss, so a pattern found
//            in it is likely noise;
//   silver — equally often (M = 1): MRI is silent.
// The deeper the color, the larger M; full color at M = {ceil}, the end of the checked range, where the line stops.
// MRI does not see direction: to the model a window and its upside-down copy are the same thing. It is not a forecast.
//
// HOW IT WORKS
// It uses only the bars of this chart: not the time of day, not the date, not other symbols.
//  1. Inputs. Each bar becomes 4 numbers: the steps of close, HL/2 and HLC/3 from the bar before and the range of the
//     bar, all in units of the median bar range of the 200 bars before it.
//  2. Network. A small convolutional neural network ({n_w} weights, stored in this script) scores the window together
//     with the 200 bars before it. It runs as a stream: each new bar costs one step of the network, the window is never
//     recomputed from scratch.
//  3. Ruler. The score becomes z: its place among the scores of random windows (one table for every symbol).
//  4. Line M. z is compared with the z of the last {lw} windows of this chart, plus {n0} virtual random windows for a
//     careful start. M is the ratio of two bell curves at z: "windows of this chart" over "random windows". On a random
//     market the two bells coincide and M stays near 1.
//
// LIMITS
//  - Trained and checked on 5-minute bars of 31 crypto markets (Binance spot); other timeframes give an estimate.
//  - The line starts after {show} windows of continuous data (bar {first_bar}); until then a message counts the bars left.
//  - Checked range: M up to {ceil} on both sides; above it the line stays at {ceil}.
//  - One floor of the window divisor for every symbol ({floor_short} of the bar norm): very quiet windows are not blown up.
//  - No repaint: the state changes only on confirmed (closed) bars; a window over a gap in the data is skipped.
//  - The network is heavy for Pine: by default only the last N bars are computed ("Compute only the last N bars").

indicator("{title}", shorttitle = "{short}", overlay = false, precision = 2)

'''

CONSTS = r'''// ---------------------------------------------------------------- 1. Constants
// One floor and one ceiling for every symbol: the script has no tables of symbols. FLOOR is 0.2 as the network stores
// it (a 32-bit float), written in full so that the script matches the model exactly.
const float FLOOR = {floor}   // floor of the window divisor D, in bar norms
const float CEIL  = {ceil_f}                  // end of the checked range of M on both sides; above it the line stays at CEIL
const int   L      = 64           // bars in a window
const int   NN     = 200          // bars of the bar norm (median range) and of the history before a window
const int   CH     = {ch}           // channels of each layer of the network
const int   HID    = {hidden}           // hidden units of the head
const int   NS     = {n_side}           // side inputs of the head: 2 scales of the window + 15 numbers of its history
const float EPS    = 0.0001       // orientation tie rule: the window is flipped only if X_end < -EPS * D
const float SQ199  = {sq199}   // square root of 199
const float S_SD   = {s_sd}   // standard deviation of the price steps in training: inputs are divided by it
const float LVL_SD = {lvl_sd}   // standard deviation of the bar range in training
const int   LW     = {lw}         // live part: windows of this chart in the calibration set
const float N0     = {n0_f}        // virtual random windows (z = 0, spread 1) added to the set for a careful start
const int   LAG    = {lag}           // the set ends LAG windows before the current one, so the two share no bars
const int   SHOW   = {show}         // the line starts at the window number SHOW (= LW + LAG, a full set); ring size of z
const float S_LO   = {s_lo}         // lowest allowed spread of the bell of this chart
const float PEAK_EPS = 0.000000000001   // a tiny number that keeps a division away from zero
var array<float> SIDE_MU = array.from({side_mu})   // mean of each side input in training
var array<float> SIDE_SD = array.from({side_sd})   // standard deviation of each side input in training
var array<float> HSIGNED = array.from({signed})   // 1 = a history input that changes sign with the orientation of the window
// Ruler of the model: 37 quantiles of the scores of random (surrogate) windows (RQ) and their places on the bell (RZ)
var array<float> RQ = array.from({ruler_q})
var array<float> RZ = array.from({ruler_z})

'''

INPUTS = r'''// ---------------------------------------------------------------- 2. Inputs (the "Inputs" tab of the settings)
// Colors as on the scale grey — silver — orange: silver is the middle (M = 1); as M grows the line turns towards orange
// or grey and reaches the full color at M = CEIL. Grey stays close to the chart background, orange stands out.
color colorRealInput   = input.color(#FF7A1A, "More often in real markets", group = "Colors", tooltip = "Windows like the current one occur M times more often in real markets than on a random chart made of the same bars: the market is not random right now, there is something to look for.")
color colorEqualInput  = input.color(#9D9D9D, "Equally often (M = 1)", group = "Colors", tooltip = "M = 1: windows like the current one occur equally often in real markets and on the random chart, MRI is silent. The line starts from this color and turns orange or grey as M grows.")
color colorRandomInput = input.color(#665F5A, "More often on the random chart", group = "Colors", tooltip = "Windows like the current one occur M times more often on a random chart made of the same bars: right now the market behaves like a coin toss, so a pattern found in it is likely noise.")
float markLevelInput   = input.float(8.0, "Marks on the price chart at M >=", minval = 0.0, step = 0.5, group = "Marks", tooltip = "A small mark above the price bar where M reaches this value, in the color of its side. 0 = no marks.")
bool  dbg      = input.bool(false, "Debug values (parity check)", group = "Advanced", tooltip = "Plots the internal values of each bar for checking against the Python reference (Export chart data).")
int   calcBarsInput    = input.int({calc_bars}, "Compute only the last N bars (0 = all history)", minval = 0, group = "Advanced", tooltip = "The network runs on the last N bars only: TradingView limits the run time of a script on the whole history. The line starts {show} windows after the first computed window.")

'''

CORE = r'''// ---------------------------------------------------------------- 3. Weights of the network
// The {n_w} numbers of the network are kept in this script as text, 3 characters per number: written as decimal numbers
// they would not fit into the limits of one Pine script. This is plain data, not hidden code: each character stands for
// a number 0..2047 (11 bits), 3 characters hold the 32 bits of one float32 number, and decodeInto() below turns them back
// into exactly the same number. The characters are CJK ideographs only because each of them is a single character; their
// meaning plays no role. The text is decoded once, on the first bar, into the flat list WT.
// Here: {weights_title}.

{weights}

// @function Copies a block of the flat weight list into a matrix, row by row.
// @param w (array<float>) The flat weight list WT.
// @param off (int) Where the block starts in w.
// @param nr (int) Rows of the matrix.
// @param nc (int) Columns of the matrix.
// @returns (matrix<float>) The matrix nr x nc.
toMat(array<float> w, int off, int nr, int nc) =>
    matrix<float> m = matrix.new<float>(nr, nc, 0.0)
    for i = 0 to nr - 1
        for j = 0 to nc - 1
            matrix.set(m, i, j, array.get(w, off + i * nc + j))
    m

// @function Copies a block of the flat weight list into a new array.
// @param w (array<float>) The flat weight list WT.
// @param off (int) Where the block starts in w.
// @param n (int) Number of values.
// @returns (array<float>) The values w[off] .. w[off + n - 1].
toArr(array<float> w, int off, int n) =>
    array.copy(array.slice(w, off, off + n))

// The weights of each layer, cut out of WT once. Layer 1: K1_0..K1_2 — one CH x 4 matrix per kernel tap (an ordinary
// convolution over the 4 inputs). Layers 2..4: Dk_0..Dk_2 — one weight per channel and tap (each channel alone), then
// Pk — a CH x CH matrix that mixes the channels. Head: H0 and HB0 (209 -> HID), then H3 and HB3 (HID -> 1).
{matrices}
var array<float>  H3  = array.copy(array.slice(WT, {off_h3}, {off_h3_end}))
var float         HB3 = array.get(WT, {off_hb3})

// ---------------------------------------------------------------- 4. State kept from bar to bar
// A "var" variable is set once, on the first bar, and then keeps its value from bar to bar until it is changed.
var array<float> RNG  = array.new<float>(0)        // high - low of the last 200 bars
var array<float> BODY = array.new<float>(0)        // |close - open| of the last 200 bars
var array<float> WR   = array.new<float>(L, 0.0)   // r of the last 64 bars (0 on invalid bars), oldest first
var array<float> WABS = array.new<float>(L, 0.0)   // |r| of the last 64 bars
var float prevC  = na                              // close of the previous bar
var float prevD1 = na                              // (HL/2 - close) / norm of the previous bar
var float prevD2 = na                              // (HLC/3 - close) / norm of the previous bar
var int   okRun  = 0                               // valid bars in a row
var int   firstR = -1                              // bar_index of the first valid r
var int   nConf  = 0                               // confirmed bars processed (debug: the state advanced)
// Live part of the line M: z of the last SHOW windows in a ring (index nW % SHOW), the sums S1, S2 of z and z^2 over the
// calibration set (windows nW - SHOW + 1 .. nW - LAG) and the number nW of the current window
var array<float> ZR = array.new<float>(SHOW, 0.0)
var float S1 = 0.0
var float S2 = 0.0
var int   nW = -1

// The network as a stream. Its convolutions have no bias, so scaling the inputs scales the outputs:
// conv(u / D) = conv(u) / D. That is why every bar goes through the network only once, and each window divides its
// pooled values by its own divisor D at the very end. Each layer keeps its recent outputs in a matrix used as a ring
// buffer: a new value overwrites the oldest one, nothing is shifted. Columns come in pairs: 2*s for the bars as they are
// (stream +) and 2*s+1 for the same bars upside down (stream -); a window takes one of them by its orientation in
// computeLogit(). Ring sizes, in positions: inputs 3, layers 1..3 — 5, 9, 17 (enough for the dilations 2, 4, 8),
// layer 4 — the 33 positions that see only bars of the window.
var matrix<float> R0  = matrix.new<float>(4, 6, 0.0)
var matrix<float> R1  = matrix.new<float>(CH, 10, 0.0)
var matrix<float> R2  = matrix.new<float>(CH, 18, 0.0)
var matrix<float> R3  = matrix.new<float>(CH, 34, 0.0)
var matrix<float> R4P = matrix.new<float>(CH, 33, 0.0)
var matrix<float> R4M = matrix.new<float>(CH, 33, 0.0)
var matrix<float> DWM = matrix.new<float>(CH, 2, 0.0)   // per-channel outputs of a separable layer (both streams)
var matrix<float> ONES33 = matrix.new<float>(33, 1, 1.0 / 33.0)   // multiplying by it averages the 33 positions
var matrix<float> FEAT   = matrix.new<float>(3 * CH + NS, 1, 0.0)   // inputs of the head
var int hc = -1                                    // step counter of the network (the position in the rings)
var array<float> DBG = array.new<float>(3 + NS, na)   // values of the last window: D, X_end, sign, side inputs (debug)

// @function Position in a ring buffer.
// @param cnt (int) Step counter of the network (hc).
// @param depth (int) Size of the ring, in positions.
// @param back (int) How many steps back: 0 = the newest value.
// @returns (int) The position, 0 .. depth - 1.
slot(int cnt, int depth, int back) =>
    (cnt - back + depth * 64) % depth

// @function Both streams of a ring `back` steps ago.
// @returns (matrix<float>) CH x 2: column 0 — stream +, column 1 — stream -.
tap(matrix<float> R, int depth, int cnt, int back) =>
    int s = slot(cnt, depth, back)
    matrix.submatrix(R, 0, matrix.rows(R), 2 * s, 2 * s + 2)

// @function One step of an ordinary convolution with kernel 3 and dilation d, for both streams:
//   W0 * x[t - 2d] + W1 * x[t - d] + W2 * x[t], where x is the ring R of the layer below.
// @returns (matrix<float>) CH x 2: the new outputs before ReLU.
conv3(matrix<float> R, int depth, int cnt, int d, matrix<float> W0, matrix<float> W1, matrix<float> W2) =>
    matrix.sum(matrix.sum(matrix.mult(W0, tap(R, depth, cnt, 2 * d)), matrix.mult(W1, tap(R, depth, cnt, d))), matrix.mult(W2, tap(R, depth, cnt, 0)))

// @function One step of a separable convolution, for both streams: first each channel alone with its own 3 taps
//   (dilation d), then one matrix product PW mixes the channels. It needs far fewer weights than an ordinary convolution.
// @returns (matrix<float>) CH x 2: the new outputs before ReLU.
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

// @function ReLU (negative values become 0), written into a ring at its newest position.
// @returns (bool) Always true.
reluInto(matrix<float> y, matrix<float> R, int depth, int cnt) =>
    int s = slot(cnt, depth, 0)
    for i = 0 to CH - 1
        matrix.set(R, i, 2 * s, math.max(matrix.get(y, i, 0), 0.0))
        matrix.set(R, i, 2 * s + 1, math.max(matrix.get(y, i, 1), 0.0))
    true

// @function One bar of the network: puts the 4 inputs of the bar into the input ring — stream + as they are, stream -
//   with the 3 price steps negated (the range of the bar stays) — and runs layers 1..4 for both streams.
// @param cnt (int) Step counter of the network (hc).
// @param u0 (float) Step of close.
// @param u1 (float) Step of HL/2.
// @param u2 (float) Step of HLC/3.
// @param u3 (float) Range of the bar.
// @returns (bool) Always true.
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

// @function Score (logit) of the window that ends at this bar, in 5 steps:
//   1. D, the window divisor: the median |r| of its 64 bars, at least FLOOR.
//   2. Orientation: the window is turned "end up". If its path ends below its own average level (X_end < 0), the
//      upside-down stream is used and the history inputs that depend on direction change sign. So a window and its
//      upside-down copy get the same score: the model does not see direction.
//   3. Pooling: for each channel of layer 4 — the mean, the maximum and the last value over the 33 positions, / D.
//   4. Side inputs: log D, the log of the mean |r| and the 15 history numbers, each standardized by its training mean
//      and standard deviation.
//   5. Head: 3 * CH + NS inputs -> HID (ReLU) -> 1.
// @param cnt (int) Step counter of the network (hc).
// @param hst (array<float>) The 15 history inputs of the window, before the orientation sign.
// @returns (float) The logit: the higher, the more typical of real markets.
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

// @function z of a score by the ruler: its place on the bell of random windows. Between two knots of the ruler — a
//   straight line; beyond the outer knots — the edge value.
// @param x (float) The logit of a window.
// @returns (float) z: about N(0, 1) on random windows, higher on windows typical of real markets.
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

// @function log M of a window: the log of the ratio of two bell curves at z — N(m, s^2), the bell of this chart, over
//   N(0, 1), the bell of random windows. m is kept >= 0 and s in [S_LO; 1]; z is capped at the peak of the ratio,
//   m / (1 - s^2), so that a higher z never gives a lower M.
// @param z (float) z of the window.
// @param mE (float) Mean of z on this chart (live part).
// @param sE (float) Spread of z on this chart (live part).
// @returns (float) log M: above 0 — more often in real markets, below 0 — more often on the random chart.
logMOf(float z, float mE, float sE) =>
    float m = math.max(mE, 0.0)
    float s = math.min(math.max(sE, S_LO), 1.0)
    float zc = s < 1.0 ? math.min(z, m / math.max(1.0 - s * s, PEAK_EPS)) : z
    float lm = -(zc - m) * (zc - m) / (2.0 * s * s) - math.log(s) + zc * zc / 2.0
    lm

// @function Clips a value to [-20; 20]; na becomes 0.
clipZ(float x) =>
    nz(math.max(math.min(x, 20.0), -20.0))

// @function The log of a positive ratio, clipped to [-5; 5]; na becomes 0.
clipLog(float v) =>
    nz(math.max(math.min(math.log(math.max(v, 0.001)), 5.0), -5.0))

// ---------------------------------------------------------------- 5. Inputs of this bar
// The bar norm nrm is the median range (high - low) of the 200 bars before this bar: the bar itself is not included, so
// the norm never looks at the bar it measures. In units of nrm: r — the step of close from the previous bar; st1, st2 —
// the steps of HL/2 and HLC/3 (built from the offsets of each bar to the previous close, so that a bar flipped upside
// down flips its steps exactly); gg — the range of this bar. After a gap in the data or while the norm is not ready, the
// bar is not valid (na).
bool  tf5   = timeframe.period == "5"                     // the chart is on 5-minute bars
float GAPMS = 1.5 * timeframe.in_seconds() * 1000.0   // more than 1.5 bars between two bars = a gap in the data
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

// ---------------------------------------------------------------- 6. History before the window (15 numbers, in bars)
// The window that ends at this bar starts at the bar s = bar_index - 63; its history is the 200 bars before s. The bar
// just before the window (s - 1) is left out of the path and of the high-low range: the model was trained that way. The
// scales below are computed on every bar for a window that would start there and are read back later with the history
// operator.
// scales of a window starting at this bar, from the 200 bars before it
float rmeanB = array.size(RNG) == NN ? array.avg(RNG) : na
float bmedB  = array.size(BODY) == NN ? math.max(array.median(BODY), 0.05 * nrm) : na
float r16B   = array.size(RNG) == NN ? array.avg(array.slice(RNG, NN - 16, NN)) : na
float r64B   = array.size(RNG) == NN ? array.avg(array.slice(RNG, NN - 64, NN)) : na
float hmaxB  = ta.highest(high, NN - 1)            // bars b-198 .. b
float lminB  = ta.lowest(low, NN - 1)
float cmaB   = ta.sma(close, NN - 1)
// read back for the window that ends at this bar: [63] is its first bar s, [65] is the bar s - 2
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

// ---------------------------------------------------------------- 7. The network and the line M, on closed bars only
// heavyFrom: the first bar where the network runs ("Compute only the last N bars"); before it only the lists above move.
int heavyFrom = calcBarsInput == 0 ? 0 : math.max(last_bar_index - calcBarsInput - L, 0)
float logit = na
float zz    = na
float mW    = na
float sW    = na
int   kW    = na
float logM  = na

// once per closed bar: the 200-bar lists, the values of the previous bar and the 64-bar window move one bar forward
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
    // a window gets a score when its 64 bars are valid in a row, its 200 bars of history exist and the stream has run
    // through a whole window
    if okRun >= L and bar_index - (L - 1) >= NN and hc >= L - 1
        logit := computeLogit(hc, HST)
        zz    := rulerZ(logit)
        nW    += 1
        // the ring keeps z of the last SHOW windows; the running sums S1, S2 cover the calibration set: one window enters
        // it and one leaves it, so no loop over 4032 values is needed
        int slotZ = nW % SHOW
        if nW >= SHOW                              // the window nW - SHOW leaves the set
            float zo = array.get(ZR, slotZ)
            S1 -= zo
            S2 -= zo * zo
        array.set(ZR, slotZ, zz)
        if nW >= LAG                               // the window nW - LAG enters it
            float zi = array.get(ZR, (nW - LAG) % SHOW)
            S1 += zi
            S2 += zi * zi
        // mean and spread of z on this chart, with N0 virtual random windows (z = 0, spread 1) mixed in
        kW    := math.min(math.max(nW - LAG + 1, 0), LW)
        float kk = math.max(kW, 1)
        float mh = S1 / kk
        float vh = math.max(S2 / kk - mh * mh, 0.0)
        mW    := S1 / (kW + N0)
        sW    := math.sqrt((kW * vh + N0) / (kW + N0))
        logM  := logMOf(zz, mW, sW)

'''

FOOT = r'''// ---------------------------------------------------------------- 8. Display
//@variable The line is drawn from the window SHOW on (a full calibration set), on every bar with a valid window.
bool  shownM = not na(logM) and nW >= SHOW
float Mraw   = shownM ? math.exp(math.abs(logM)) : na   // M: how many times more often, on its side
bool  more   = shownM and logM > 0                      // more often in real markets (orange side)
bool  less   = shownM and logM < 0                      // more often on the random chart (grey side)
float Mshow  = na(Mraw) ? na : more or less ? math.min(Mraw, CEIL) : 1.0
// The color: silver at M = 1, the color of its side at M = CEIL; in between the share of the side color grows with
// log M, so that 2x -> 4x and 4x -> 8x are equal steps (M is a ratio)
color sideColor = more ? colorRealInput : less ? colorRandomInput : colorEqualInput
color mriColor  = shownM ? color.from_gradient(math.log(Mshow), 0.0, math.log(CEIL), colorEqualInput, sideColor) : na
plot(Mshow, "M", color = mriColor, linewidth = 2)
// Dashed horizontal lines: M = 1 (equally often) and a faint grid at 2, 4, 6, 8. They are fixed: "editable = false" keeps
// them out of the Style tab of the settings; their color follows the silver of the Inputs tab.
hline(1.0, "Equally often (M = 1)", color = colorEqualInput, linestyle = hline.style_dashed, editable = false)
hline(2.0, "M = 2", color = color.new(colorEqualInput, 70), linestyle = hline.style_dashed, editable = false)
hline(4.0, "M = 4", color = color.new(colorEqualInput, 70), linestyle = hline.style_dashed, editable = false)
hline(6.0, "M = 6", color = color.new(colorEqualInput, 70), linestyle = hline.style_dashed, editable = false)
hline(8.0, "M = 8", color = color.new(colorEqualInput, 70), linestyle = hline.style_dashed, editable = false)
// the start of the line: a constant text at the window SHOW; the shape under the text is fully transparent
plotshape(shownM and nW == SHOW, "Line start", shape.diamond, location.bottom, color.new(colorEqualInput, 100), text = "{start_text}", textcolor = chart.fg_color, size = size.tiny, editable = false, display = display.pane)
// marks on the price chart where M reaches the level set in the Inputs tab (0 = no marks)
plotshape(markLevelInput > 0 and more and Mraw >= markLevelInput, "Mark: more often in real markets", shape.circle, location.abovebar, colorRealInput, size = size.tiny, force_overlay = true, editable = false, display = display.pane)
plotshape(markLevelInput > 0 and less and Mraw >= markLevelInput, "Mark: more often on the random chart", shape.circle, location.abovebar, colorRandomInput, size = size.tiny, force_overlay = true, editable = false, display = display.pane)

// ---------------------------------------------------------------- 9. Messages (top right corner of the pane)
var table MSG = table.new(position.top_right, 1, 2, bgcolor = color.new(colorEqualInput, 85))
if barstate.islast
    table.clear(MSG, 0, 0, 0, 1)
    int row = 0
    if not tf5
        table.cell(MSG, 0, row, "MRI is trained and checked on 5-minute bars; on this timeframe the line is an estimate", text_size = size.small, text_color = chart.fg_color)
        row += 1
    if nW < SHOW
        // the bar where the first window ends, then SHOW windows more until the line
        int firstEnd = math.max((firstR < 0 ? NN + 1 : math.max(firstR, NN + 1)) + L - 1, heavyFrom + L - 1)
        int left = nW >= 0 ? SHOW - nW : firstEnd - bar_index + SHOW
        table.cell(MSG, 0, row, "MRI warm-up: " + str.tostring(left) + " more bars until the line starts", text_size = size.small, text_color = chart.fg_color)

// ---------------------------------------------------------------- debug values (parity with the Python reference, R10 addendum 4)
display_dbg = dbg ? display.all : display.none
plot(logit, "dbg logit", display = display_dbg, precision = 10)
plot(zz, "dbg z", display = display_dbg, precision = 10)
plot(mW, "dbg m_win", display = display_dbg, precision = 10)
plot(sW, "dbg s_win", display = display_dbg, precision = 10)
plot(kW, "dbg k", display = display_dbg, precision = 0)
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

DEBUG_TOKENS = re.compile(r'\b(DBG|nConf|dbg|display_dbg)\b')
DEBUG_HEADER = '// ---------------------------------------------------------------- debug values'


def strip_debug(text):
    """The production script without the debug mode (decision of 2026-10-18): the debug section at the end and every line
       with a debug name (the input dbg, the array DBG, the counter nConf) are cut; nothing else changes."""
    body = text[:text.index(DEBUG_HEADER)].rstrip('\n') + '\n'
    return ''.join(ln for ln in body.splitlines(keepends=True) if not DEBUG_TOKENS.search(ln))


# constant text of the mark at the line start (2026-10-27)
START_TEXT = ('MRI starts here:\\non each bar, the 64-bar window\\nis compared with {lw} historical\\n'
              'windows of this chart')


def make_script(p, ruler, floor, ceil, live, passport, title='Market Randomness Index', short='MRI', calc_bars=0,
                s_lo=0.75, start_text=START_TEXT):
    """The indicator text. p: Params (PB shapes, separable); ruler: (q, zk) — 37 numbers each; floor — one floor of the
       window divisor (R17); ceil — one ceiling for both colours (R19); live: {L, n0, lag, show_from_window} (R18);
       passport: weights_source, data_end, code_version — kept in the call, not printed since 2026-10-27 (the comments of
       the script describe only the indicator itself: no files, exports or commits outside it)."""
    if p.pw is None:
        raise ValueError('the PB template needs separable layers 2..4')
    q, zk = (list(map(float, x)) for x in ruler)
    if len(q) != len(zk) or not np.all(np.diff(q) > 0):
        raise ValueError('ruler: strictly increasing quantiles and their z')
    lw, n0, lag, show = (int(live[k]) for k in ('L', 'n0', 'lag', 'show_from_window'))
    if show != lw + lag:
        raise ValueError('the line starts at the full set: show_from_window = L + lag')
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
    cg = f'{float(ceil):g}'
    head = HEAD.format(title_line=f'{title} ({short})', n_w=f'{len(vals):,}', lw=lw, n0=n0, show=show,
                       first_bar=264 + show, ceil=cg, floor_short=f'{float(floor):.1f}', title=title, short=short)
    consts = CONSTS.format(floor=repr(float(floor)), ceil_f=fmt_d(float(ceil)), ch=ch, hidden=hidden, n_side=n_side,
                           sq199=fmt_d(np.sqrt(199.0)), s_sd=fmt(p.s_sd), lvl_sd=fmt(p.lvl_sd),
                           side_mu=float_list(p.side_mu), side_sd=float_list(p.side_sd),
                           signed=', '.join('1.0' if s else '0.0' for s in SIDE_SIGNED[2:]), lw=lw, n0_f=f'{n0}.0',
                           lag=lag, show=show, s_lo=fmt_d(s_lo), ruler_q=', '.join(fmt_d(x) for x in q),
                           ruler_z=', '.join(fmt_d(x) for x in zk))
    inputs = INPUTS.format(calc_bars=calc_bars, show=show)
    core = CORE.format(weights_title=wtitle, weights=wblock, n_w=f'{len(vals):,}', matrices=mdecl, off_h3=off,
                       off_h3_end=off + hidden, off_hb3=off + hidden, chunks=chunks)
    foot = FOOT.format(dbg_side=dbg_side, start_text=start_text.format(lw=lw))
    return head + consts + inputs + core + foot


# 'Compute only the last N bars' of the production script, from the Pine Profiler of session 4 (BINANCE:BTCUSDT 5M,
# 21 832 bars): N = 0 — 23 s (heavy part ~1.0 ms per bar), N = 10000 — 12 s; a 40 000-bar chart with N = 10000 — ~13 s
# (limit 40 s). Session 5 keeps N = 10000 (the line on the last ~5 900 bars: 4096 windows of the computed part first)
CALC_BARS_DEFAULT = 10000


def make_indicator(ex, production=False, calc_bars=None):
    """The PB indicator from the export dict (export/mri_model_pb.json): production — pine/mri.pine (default N of the
       measurement, no debug mode); prototype (pine/proto_pb.pine, the parity check is made on it) — N = 0, the debug
       mode and the prototype title."""
    from .params import from_dict
    p = from_dict(ex['model'])
    if (ex['tie_rule']['eps'] != 1e-4 or ex['model']['id'] != 'PB_s64h24_d1' or len(p.side_mu) != len(SIDE_NAMES)
            or ex['line']['ruler']['kind'] != 'glob_e' or ex.get('format_version') != 4):
        raise ValueError('the template has EPS = 0.0001, the 17 side inputs, the glob_e ruler of PB_s64h24_d1 and the '
                         'session-5 line (export format 4)')
    ln = ex['line']
    live = {'L': ln['live']['L'], 'n0': ln['live']['n0'], 'lag': ln['live']['lag'],
            'show_from_window': ln['show']['from_window']}
    kw = {} if production else {'title': 'Market Randomness Index prototype', 'short': 'MRI proto'}
    if calc_bars is None:
        calc_bars = CALC_BARS_DEFAULT if production else 0
    s = make_script(p, (ln['ruler']['q'], ln['ruler']['zk']), ex['floor']['value'], ln['ceiling']['value'], live,
                    ex['passport'], calc_bars=calc_bars, s_lo=ln['live']['s_lo'], **kw)
    return strip_debug(s) if production else s


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
        raise SystemExit(f'{path}: larger than {SIZE_LIMIT_UTF8} bytes UTF-8 — shrink the code first (rules, sessions 4-5)')


if __name__ == '__main__':
    main()
