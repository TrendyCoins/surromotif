"""Pine Script v6 generator: the streaming F3_g computation with weights inside the script.

Limits met so far (TradingView compiler):
    - a local block (if / function body) has a size limit: 'The if statement is too long', 'The body of the ... function
      is too long'; constant results of functions are folded into the call site, so a function returning a long string
      constant counts in the caller's body;
    - 100 256 compiled tokens per script; every literal is embedded into the IL: 16 897 decimal literals cost 225 018
      tokens (~13 per number, about their text length).
So the weights are packed densely: each float32 — all 32 bits, exact — as 3 characters of a 2048-character alphabet
(CJK ideographs U+4E00..U+55FF, 11 bits per character): 50 691 characters in strings of <= `str_max` characters.
Each string is decoded by a `var` initializer at the global scope (runs once, no local block holds the constants);
the alphabet becomes a map<string, int> once. Decoding is exact (powers of two), so Pine holds the float32 weights
bit for bit; the export JSON writes the same values as exact decimals.

Flat weight order (16897 for F3_g): conv layer 1..4, tap 0..2, matrix W[:, :, tap] row-major (out, in);
head0 W row-major (64, 112); head0 b (64); head3 W (64); head3 b (1).

The compute: per confirmed bar, inputs of two streams (+ / -) go into interleaved ring matrices (column 2s = stream +,
2s + 1 = stream -, s = ring slot); each conv layer = 3 matrix.mult of (32 x c_in) by (c_in x 2) + ReLU loop; layer 4
outputs go to two matrices (32 x 33) for pooling; the heavy part runs in functions (convStep, computeLogit).
tests/pine_sim.py transliterates this template line by line; tests/test_reference.py checks it against mri/stream.py.
"""
import numpy as np


ALPHA0, ALPHA_N = 0x4E00, 2048               # alphabet of the dense weight strings: 2048 CJK ideographs


def fmt(x):
    """Exact positional decimal of the float32 value (as float64): no exponent, safe as a Pine literal, parses back to
       exactly the float32 value."""
    s = np.format_float_positional(np.float64(np.float32(x)), unique=True, trim='-')
    return s if '.' in s else s + '.0'          # always a float literal


def flat_values(p):
    out = []
    for w in p.conv:
        for k in range(3):
            out.extend(w[:, :, k].ravel())
    for a in (p.head0_w, p.head0_b, p.head3_w, p.head3_b):
        out.extend(np.ravel(a))
    return np.array(out, dtype=np.float32)


def encode_dense(vals):
    """float32 -> 3 characters each: bits = d0 * 2^22 + d1 * 2^11 + d2, character = chr(ALPHA0 + d)."""
    bits = np.asarray(vals, dtype=np.float32).view(np.uint32).astype(np.int64)
    d = np.stack([bits >> 22, (bits >> 11) & 2047, bits & 2047], axis=1)
    return ''.join(chr(ALPHA0 + int(x)) for x in d.ravel())


def decode_dense(s):
    """Python mirror of the Pine decoder (arithmetic on doubles, powers of two): exact float32 values."""
    out = []
    for k in range(len(s) // 3):
        b = (ord(s[3 * k]) - ALPHA0) * 4194304.0 + (ord(s[3 * k + 1]) - ALPHA0) * 2048.0 + (ord(s[3 * k + 2]) - ALPHA0)
        sg = -1.0 if b >= 2147483648.0 else 1.0
        u = b - 2147483648.0 if b >= 2147483648.0 else b
        e = float(np.floor(u / 8388608.0))
        m = u - e * 8388608.0
        v = m * 2.0 ** -149 if e == 0 else (1.0 + m / 8388608.0) * 2.0 ** (e - 127)
        out.append(sg * v)
    return out


def flat_weights(p):
    return [fmt(v) for v in flat_values(p)]


def matrix_offsets():
    """(name, rows, cols, offset) of the matrices in the flat order; then H3 (64) and HB3 (1)."""
    out, off = [], 0
    for li, c_in in enumerate((4, 32, 32, 32), 1):
        for k in range(3):
            out.append((f'K{li}_{k}', 32, c_in, off))
            off += 32 * c_in
    out.append(('H0', 64, 112, off))
    off += 64 * 112
    out.append(('HB0', 64, 1, off))
    off += 64
    return out, off


def weights_block(vals32, str_max=3999):
    """Pine code: alphabet map, decoder, one `var` initializer per dense string (global scope)."""
    enc = encode_dense(vals32)
    per = (str_max // 3) * 3
    strs = [enc[i:i + per] for i in range(0, len(enc), per)]
    alpha = ''.join(chr(ALPHA0 + i) for i in range(ALPHA_N))
    head = (f'buildAlpha() =>\n    map<string, int> m = map.new<string, int>()\n'
            f'    array<string> cs = str.split("{alpha}", "")\n'
            f'    for [i, c] in cs\n        map.put(m, c, i)\n    m\n\n'
            'decodeInto(array<float> dst, map<string, int> A, string s) =>\n'
            '    array<string> cs = str.split(s, "")\n'
            '    int n = math.floor(array.size(cs) / 3.0)\n'
            '    for k = 0 to n - 1\n'
            '        float b = map.get(A, array.get(cs, 3 * k)) * 4194304.0 + map.get(A, array.get(cs, 3 * k + 1)) * 2048.0 + '
            'map.get(A, array.get(cs, 3 * k + 2))\n'
            '        float sg = b >= 2147483648.0 ? -1.0 : 1.0\n'
            '        float u = b >= 2147483648.0 ? b - 2147483648.0 : b\n'
            '        float e = math.floor(u / 8388608.0)\n'
            '        float m = u - e * 8388608.0\n'
            '        float v = e == 0 ? m * math.pow(2.0, -149) : (1.0 + m / 8388608.0) * math.pow(2.0, e - 127)\n'
            '        array.push(dst, sg * v)\n'
            '    true\n\n'
            'var map<string, int> ALPHA = buildAlpha()\n'
            'var array<float> WT = array.new<float>(0)\n')
    body = '\n'.join(f'var bool WOK{j:02d} = decodeInto(WT, ALPHA, "{x}")' for j, x in enumerate(strs))
    return head + body, len(strs), len(enc)


def float_list(xs):
    return ', '.join(fmt(x) for x in xs)


CONSTS = r'''// ---------------------------------------------------------------- constants
const int   L      = 64
const int   NN     = 200          // bars of the range median (before the bar)
const float EPS    = 0.0001       // orientation tie rule: flip only if X_end < -EPS * D
const float A7     = 2.0 / 2017.0 // EMA |r|, 7 days of 5M
const float A1     = 2.0 / 289.0  // EMA |r|, 1 day
const int   DAY    = 288
const int   BURN   = {burn}         // bars from the first valid r before the line is shown (EMA burn-in)
const float S_SD   = {s_sd}
const float LVL_SD = {lvl_sd}
var array<float> SIDE_MU = array.from({side_mu})
var array<float> SIDE_SD = array.from({side_sd})

'''

CORE = r'''// ---------------------------------------------------------------- weights: {n_w} float32, {n_chars} characters in {n_str} strings
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
var array<float> RNG  = array.new<float>(0)        // ranges of the last 200 bars
var array<float> WR   = array.new<float>(L, 0.0)   // r of the last 64 bars (0 on invalid bars), oldest first
var array<float> WABS = array.new<float>(L, 0.0)   // |r| of the last 64 bars
var float prevC  = na
var float prevD1 = na                              // (HL/2 - close) / norm of the previous bar
var float prevD2 = na                              // (HLC/3 - close) / norm of the previous bar
var int   okRun  = 0                               // consecutive valid bars
var int   firstR = -1                              // bar_index of the first valid r
var float s7 = 0.0
var float w7 = 0.0
var float s1 = 0.0
var float w1 = 0.0
var float csum = 0.0                               // running sum of r over valid bars

// conv rings: inputs (4 x 2*3), layers 1..3 (32 x 2*5, 2*9, 2*17); layer 4 per stream (32 x 33)
var matrix<float> R0  = matrix.new<float>(4, 6, 0.0)
var matrix<float> R1  = matrix.new<float>(32, 10, 0.0)
var matrix<float> R2  = matrix.new<float>(32, 18, 0.0)
var matrix<float> R3  = matrix.new<float>(32, 34, 0.0)
var matrix<float> R4P = matrix.new<float>(32, 33, 0.0)
var matrix<float> R4M = matrix.new<float>(32, 33, 0.0)
var matrix<float> ONES33 = matrix.new<float>(33, 1, 1.0 / 33.0)
var matrix<float> FEAT   = matrix.new<float>(112, 1, 0.0)
var int hc = -1                                    // heavy-part bar counter (ring position)
var array<float> DBG = array.new<float>(19, na)    // values of the last window: D, X_end, sign, 16 side inputs (debug)

slot(int cnt, int depth, int back) =>
    (cnt - back + depth * 64) % depth

tap(matrix<float> R, int depth, int cnt, int back) =>
    int s = slot(cnt, depth, back)
    matrix.submatrix(R, 0, matrix.rows(R), 2 * s, 2 * s + 2)

conv3(matrix<float> R, int depth, int cnt, int d, matrix<float> W0, matrix<float> W1, matrix<float> W2) =>
    matrix.sum(matrix.sum(matrix.mult(W0, tap(R, depth, cnt, 2 * d)), matrix.mult(W1, tap(R, depth, cnt, d))), matrix.mult(W2, tap(R, depth, cnt, 0)))

reluInto(matrix<float> y, matrix<float> R, int depth, int cnt) =>
    int s = slot(cnt, depth, 0)
    for i = 0 to 31
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
    for i = 0 to 31
        matrix.set(R4P, i, p4, math.max(matrix.get(y4, i, 0), 0.0))
        matrix.set(R4M, i, p4, math.max(matrix.get(y4, i, 1), 0.0))
    true

// logit of the window ending at this bar; mv = 11 unsigned moves before the window (1d, 3d, 7d, 8 blocks), nrm1 =
// EMA |r| (1 day) at the bar before the window; tm = bar open time
computeLogit(int cnt, array<float> mv, float nrm1, int tm) =>
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
    for i = 0 to 31
        matrix.set(FEAT, i, 0, matrix.get(mean4, i, 0) / D)
        matrix.set(FEAT, 32 + i, 0, array.max(matrix.row(M4, i)) / D)
        matrix.set(FEAT, 64 + i, 0, matrix.get(M4, i, p4) / D)
    float mins = math.floor(tm / 60000.0) % 1440
    float hr = mins / 60.0 * (2.0 * math.pi / 24.0)
    float dw = ((math.floor(tm / 86400000.0) + 3) % 7) * (2.0 * math.pi / 7.0)
    array<float> side = array.from(math.log(D) - math.log(s7 / w7), math.sin(hr), math.cos(hr), math.sin(dw), math.cos(dw))
    for j = 0 to 10
        array.push(side, array.get(mv, j) / nrm1 * sgn)
    for j = 0 to 15
        array.set(DBG, 3 + j, array.get(side, j))
        matrix.set(FEAT, 96 + j, 0, (array.get(side, j) - array.get(SIDE_MU, j)) / array.get(SIDE_SD, j))
    matrix<float> z = matrix.sum(matrix.mult(H0, FEAT), HB0)
    float acc = HB3
    for i = 0 to 63
        acc += math.max(matrix.get(z, i, 0), 0.0) * array.get(H3, i)
    acc

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

int heavyFrom = calcBars == 0 ? 0 : math.max(last_bar_index - calcBars - L, 0)
float logit = na

if barstate.isconfirmed
    array.push(RNG, rngNow)
    if array.size(RNG) > NN
        array.shift(RNG)
    prevD1 := goodN ? (hl2 - close) / nrm : na
    prevD2 := goodN ? (hlc3 - close) / nrm : na
    prevC  := close
    okRun  := ok ? okRun + 1 : 0
    array.shift(WR)
    array.push(WR, ok ? r : 0.0)
    array.shift(WABS)
    array.push(WABS, ok ? math.abs(r) : 0.0)
    if ok
        if firstR < 0
            firstR := bar_index
        s7 := math.abs(r) + (1.0 - A7) * s7
        w7 := 1.0 + (1.0 - A7) * w7
        s1 := math.abs(r) + (1.0 - A1) * s1
        w1 := 1.0 + (1.0 - A1) * w1
        csum += r

// series with history (moves before the window): csS[m] = sum of r over bars <= bar_index - m
float csS = csum
float e1S = w1 > 0 ? s1 / w1 : na
max_bars_back(csS, 2100)
max_bars_back(e1S, 80)
float csA = csS[L + 1]                             // sum up to the bar before the window's previous bar
array<float> MV = array.from((csA - csS[L + 1 + DAY]) / math.sqrt(DAY), (csA - csS[L + 1 + 3 * DAY]) / math.sqrt(3 * DAY), (csA - csS[L + 1 + 7 * DAY]) / math.sqrt(7 * DAY), (csS[L + 113] - csS[L + 129]) / 4.0, (csS[L + 97] - csS[L + 113]) / 4.0, (csS[L + 81] - csS[L + 97]) / 4.0, (csS[L + 65] - csS[L + 81]) / 4.0, (csS[L + 49] - csS[L + 65]) / 4.0, (csS[L + 33] - csS[L + 49]) / 4.0, (csS[L + 17] - csS[L + 33]) / 4.0, (csA - csS[L + 17]) / 4.0)
float nrm1 = e1S[L]

array.fill(DBG, na)
if barstate.isconfirmed and bar_index >= heavyFrom
    hc += 1
    convStep(hc, ok ? r / S_SD : 0.0, ok ? st1 / S_SD : 0.0, ok ? st2 / S_SD : 0.0, ok ? gg / LVL_SD : 0.0)
    if okRun >= L and firstR >= 0 and bar_index - firstR >= BURN and hc >= L - 1
        logit := computeLogit(hc, MV, nrm1, time)

'''

# ---------------------------------------------------------------------------------------------------------------------
# Production indicator (pine/mri.pine): the same CONSTS + CORE, with the symbol table, display, messages and debug plots.

HEAD_PROD = r'''//@version=6
// SPDX-License-Identifier: Apache-2.0
// © TrendyCoins
// Market Randomness Index (MRI) — generated by mri/pine_gen.py from export/mri_model.json; do not edit by hand.
//
// What the line shows: M — how many times a 64-bar window of this shape is more frequent (orange) or less frequent
// (blue) in real markets than in a random walk with the same volatility (bar-mirror surrogate in blocks of 4 bars).
// M = 1 — indistinguishable from random. The SuMo classifier (a streaming CNN, {n_w} weights inside this script) scores
// the window; Platt calibration turns the score into M. M is a frequency ratio of the window's shape, not a forecast.
// Calibration was checked out of sample per symbol: each symbol and side has its ceiling (at most {ceil_txt}; for symbols
// outside the table {ceil_unknown}); above it the line stays at the ceiling, marked "N+".
// No repaint: the state changes only on confirmed bars; a window over a gap in the data is not drawn.
// Passport: weights {source}, data up to {data_end}, export of code {code_version}.
indicator("Market Randomness Index", shorttitle = "MRI", overlay = false, precision = 2, max_labels_count = 500)

// ---------------------------------------------------------------- inputs
float thr      = input.float(3.0, "Threshold M", minval = 1.0, step = 0.25, tooltip = "Visual only: above the threshold the line is bright, below it is pale; dashed line at the threshold; optional marks on the price chart at M >= threshold.")
int   paleT    = input.int(70, "Transparency below the threshold", minval = 0, maxval = 100)
color colMore  = input.color(color.rgb(242, 140, 40), "More frequent than random", tooltip = "M > 1 on the more frequent side: windows of this shape are M times more frequent in real markets than in a random walk.")
color colLess  = input.color(color.rgb(47, 128, 237), "Less frequent than random", tooltip = "M > 1 on the less frequent side: windows of this shape are M times less frequent in real markets than in a random walk.")
bool  showMarks = input.bool(false, "Marks on the price chart at M >= threshold")
bool  dbg      = input.bool(false, "Debug values (parity check)", group = "Advanced", tooltip = "Plots the internal values of each bar for checking against the Python reference (Export chart data).")
int   calcBars = input.int(0, "Compute only the last N bars (0 = all history)", minval = 0, group = "Advanced")

// ---------------------------------------------------------------- symbol, timeframe, floor of the window divisor
buildFloors() =>
    map<string, float> m = map.new<string, float>()
{floor_puts}
    m

// ceilings of M by symbol: more frequent / less frequent side (rule R11)
buildCeilMore() =>
    map<string, float> m = map.new<string, float>()
{ceil_more_puts}
    m

buildCeilLess() =>
    map<string, float> m = map.new<string, float>()
{ceil_less_puts}
    m

var map<string, float> FLOORS = buildFloors()
var map<string, float> CEILO  = buildCeilMore()
var map<string, float> CEILB  = buildCeilLess()
bool  known = syminfo.prefix == "BINANCE" and not str.endswith(syminfo.ticker, ".P") and map.contains(FLOORS, syminfo.ticker)
var float FLOOR = known ? map.get(FLOORS, syminfo.ticker) : {floor_unknown}
bool  tf5   = timeframe.period == "5"
float GAPMS = 1.5 * timeframe.in_seconds() * 1000.0   // interval > 1.5 bars -> grid gap
const float PLATT_A = {platt_a}
const float PLATT_B = {platt_b}
var float CEIL_O = known ? map.get(CEILO, syminfo.ticker) : {ceil_o_unknown}   // checked range of M, more frequent side
var float CEIL_B = known ? map.get(CEILB, syminfo.ticker) : {ceil_b_unknown}   // checked range of M, less frequent side

'''

FOOT_PROD = r'''// ---------------------------------------------------------------- display
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
        table.cell(MSG, 0, row, "MRI: calibrated on Binance spot; accuracy on this chart is not verified", text_size = size.small, text_color = chart.fg_color)
        row += 1
    int left = firstR < 0 ? NN + 1 + BURN - bar_index : firstR + BURN - bar_index
    if left > 0
        table.cell(MSG, 0, row, "MRI warm-up: " + str.tostring(left) + " bars to the first value", text_size = size.small, text_color = chart.fg_color)

// ---------------------------------------------------------------- debug values (parity with the Python reference)
display_dbg = dbg ? display.all : display.none
plot(logit, "dbg logit", display = display_dbg, precision = 10)
plot(Mraw, "dbg M", display = display_dbg, precision = 10)
plot(nrm, "dbg norm", display = display_dbg, precision = 10)
plot(r, "dbg r", display = display_dbg, precision = 10)
plot(st1, "dbg step_hl2", display = display_dbg, precision = 10)
plot(st2, "dbg step_hlc3", display = display_dbg, precision = 10)
plot(gg, "dbg g", display = display_dbg, precision = 10)
plot(w7 > 0 ? s7 / w7 : na, "dbg ema7", display = display_dbg, precision = 10)
plot(w1 > 0 ? s1 / w1 : na, "dbg ema1", display = display_dbg, precision = 10)
plot(array.get(DBG, 0), "dbg D", display = display_dbg, precision = 10)
plot(array.get(DBG, 1), "dbg x_end", display = display_dbg, precision = 10)
plot(array.get(DBG, 2), "dbg sgn", display = display_dbg, precision = 10)
{dbg_side}
plot(bar_index, "dbg bar_index", display = display_dbg, precision = 0)
'''


def make_indicator(ex):
    """pine/mri.pine from the export dict (export/mri_model.json)."""
    from .params import SIDE_NAMES, from_dict
    p = from_dict(ex['model'])
    vals = flat_values(p)
    assert len(vals) == p.n_params == ex['model']['n_params']
    mats, off = matrix_offsets()
    mdecl = '\n'.join(f"var matrix<float> {nm:<4} = toMat(WT, {o}, {r}, {c})" for nm, r, c, o in mats)
    weights, n_str, n_chars = weights_block(vals)
    ceil = ex['display']['ceiling']
    txt = lambda v: f"{v:g}"
    floor_puts = '\n'.join(f'    map.put(m, "{a}", {fmt(v)})' for a, v in ex['floors']['floor_L64'].items())
    ceil_puts = {k: '\n'.join(f'    map.put(m, "{a}", {fmt(v[k])})' for a, v in ceil['by_symbol'].items())
                 for k in ('more', 'less')}
    dbg_side = '\n'.join(f'plot(array.get(DBG, {3 + j}), "dbg side_{nm}", display = display_dbg, precision = 10)'
                         for j, nm in enumerate(SIDE_NAMES))
    pa = ex['passport']
    pool, unk = ceil['pool'], ceil['unknown_symbol']
    head = HEAD_PROD.format(n_w=len(vals), ceil_txt=txt(max(pool['more'], pool['less'])),
                            ceil_unknown=f"{txt(unk['more'])} / {txt(unk['less'])}",
                            ceil_more_puts=ceil_puts['more'], ceil_less_puts=ceil_puts['less'],
                            ceil_o_unknown=fmt(unk['more']), ceil_b_unknown=fmt(unk['less']),
                            source=pa['weights_source'], data_end=pa['data_end'], code_version=pa['code_version'],
                            floor_puts=floor_puts, floor_unknown=fmt(ex['floors']['unknown_symbol']),
                            platt_a=fmt(ex['calibration']['platt_a']), platt_b=fmt(ex['calibration']['platt_b']))
    if ex['tie_rule']['eps'] != 1e-4:
        raise ValueError("CONSTS has EPS = 0.0001")
    consts = CONSTS.format(burn=ex['warmup']['burn_bars_from_first_valid_r'], s_sd=fmt(p.s_sd), lvl_sd=fmt(p.lvl_sd),
                           side_mu=float_list(p.side_mu), side_sd=float_list(p.side_sd))
    core = CORE.format(n_w=len(vals), n_str=n_str, n_chars=n_chars, weights=weights, matrices=mdecl, off_h3=off,
                       off_h3_end=off + 64, off_hb3=off + 64)
    foot = FOOT_PROD.format(dbg_side=dbg_side)
    return head + consts + core + foot


def main():
    import json
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, 'export/mri_model.json'), encoding='utf-8') as fh:
        ex = json.load(fh)
    s = make_indicator(ex)
    path = os.path.join(root, 'pine/mri.pine')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(s)
    print(path, len(s), 'chars', len(s.splitlines()), 'lines')


if __name__ == '__main__':
    main()
