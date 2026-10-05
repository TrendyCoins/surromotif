"""Pine Script v6 generator: the streaming F3_g computation with weights inside the script.

Pine limits a local block (the body of an `if`, a function) by its compiled size — 'The if statement is too long'
(the threshold is not documented). So nothing big lives in one block:
    - weights: many small functions, each returns one chunk; `loadWeights()` concatenates them; `var` initializers
      call it once (no `if barstate.isfirst`)
    - weight storage ('store'): 'literals' — `array.from(...)` of `chunk` decimal literals per function;
      'strings' — a function per string constant of <= `str_max` chars, parsed with str.split + str.tonumber
    - per-bar heavy part: convStep() and computeLogit() functions; the `if` bodies only call them
Weights are written as the shortest decimal that round-trips the float32 value; the export JSON (mri/export.py)
uses the same decimals, so Pine and the Python reference parse the same numbers.

Flat weight order (16897 for F3_g): conv layer 1..4, tap 0..2, matrix W[:, :, tap] row-major (out, in);
head0 W row-major (64, 112); head0 b (64); head3 W (64); head3 b (1).

The compute: per confirmed bar, inputs of two streams (+ / -) go into interleaved ring matrices (column 2s = stream +,
2s + 1 = stream -, s = ring slot); each conv layer = 3 matrix.mult of (32 x c_in) by (c_in x 2) + ReLU loop; layer 4
outputs go to two matrices (32 x 33) for pooling. tests/pine_sim.py transliterates this template line by line;
tests/test_reference.py checks it against mri/stream.py.
"""
import numpy as np


def fmt(x):
    """Shortest positional decimal that round-trips the float32 value (no exponent: safe as a Pine literal)."""
    s = np.format_float_positional(np.float32(x), unique=True, trim='-')
    return s if '.' in s else s + '.0'          # always a float literal (array.from infers the type)


def flat_weights(p):
    out = []
    for w in p.conv:
        for k in range(3):
            out.extend(w[:, :, k].ravel())
    out.extend(p.head0_w.ravel())
    out.extend(p.head0_b.ravel())
    out.extend(p.head3_w.ravel())
    out.extend(p.head3_b.ravel())
    return [fmt(v) for v in out]


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


def _wrap_args(vals, per_line=12, indent=6):
    # Pine: a wrapped line must be indented by a number of spaces that is NOT a multiple of 4
    lines = []
    for i in range(0, len(vals), per_line):
        lines.append(' ' * indent + ', '.join(vals[i:i + per_line]))
    return ',\n'.join(lines)


def weights_block(vals, store, chunk=250, str_max=4000):
    """Pine code that defines `loadWeights()` -> array<float> of all weights in the flat order."""
    fns, calls = [], []
    if store == 'literals':
        for j, i in enumerate(range(0, len(vals), chunk)):
            fns.append(f"wc{j}() =>\n    array.from(\n{_wrap_args(vals[i:i + chunk])})")
            calls.append(f"    array.concat(w, wc{j}())")
    elif store == 'strings':
        strs, cur, n = [], [], 0
        for v in vals:
            if cur and n + len(v) + 1 > str_max:
                strs.append(' '.join(cur))
                cur, n = [], 0
            cur.append(v)
            n += len(v) + 1
        strs.append(' '.join(cur))
        fns.append("decodeInto(array<float> dst, string s) =>\n"
                   "    for tok in str.split(s, \" \")\n"
                   "        array.push(dst, str.tonumber(tok))\n"
                   "    true")
        for j, s in enumerate(strs):
            fns.append(f'ws{j}() =>\n    "{s}"')
            calls.append(f"    decodeInto(w, ws{j}())")
    else:
        raise ValueError(store)
    load = "loadWeights() =>\n    array<float> w = array.new<float>(0)\n" + '\n'.join(calls) + "\n    w"
    return '\n\n'.join(fns + [load]), len(fns)


def float_list(xs):
    return ', '.join(fmt(x) for x in xs)


TEMPLATE = r'''//@version=6
// SPDX-License-Identifier: Apache-2.0
// © TrendyCoins
// {title}
// {note}
indicator("{name}", shorttitle = "{short}", overlay = false)

// ---------------------------------------------------------------- inputs
int calcBars = input.int(0, "Heavy part: last N bars (0 = all history)", minval = 0)

// ---------------------------------------------------------------- constants
const int   L      = 64
const int   NN     = 200          // bars of the range median (before the bar)
const float GAPMS  = 450000.0     // interval > 1.5 bars of 5M -> grid gap
const float EPS    = 0.0001       // orientation tie rule: flip only if X_end < -EPS * D
const float A7     = 2.0 / 2017.0 // EMA |r|, 7 days of 5M
const float A1     = 2.0 / 289.0  // EMA |r|, 1 day
const int   DAY    = 288
const int   BURN   = {burn}         // bars from the first valid r before the line is shown (EMA burn-in)
const float S_SD   = {s_sd}
const float LVL_SD = {lvl_sd}
const float FLOOR  = {floor}
var array<float> SIDE_MU = array.from({side_mu})
var array<float> SIDE_SD = array.from({side_sd})

// ---------------------------------------------------------------- weights ({n_w} numbers, {n_fn} functions)
{weights}

toMat(array<float> w, int off, int nr, int nc) =>
    matrix<float> m = matrix.new<float>(nr, nc, 0.0)
    for i = 0 to nr - 1
        for j = 0 to nc - 1
            matrix.set(m, i, j, array.get(w, off + i * nc + j))
    m

var array<float> WT = loadWeights()
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

if barstate.isconfirmed and bar_index >= heavyFrom
    hc += 1
    convStep(hc, ok ? r / S_SD : 0.0, ok ? st1 / S_SD : 0.0, ok ? st2 / S_SD : 0.0, ok ? gg / LVL_SD : 0.0)
    if okRun >= L and firstR >= 0 and bar_index - firstR >= BURN and hc >= L - 1
        logit := computeLogit(hc, MV, nrm1, time)

plot(logit, "logit", color = color.new(color.orange, 0), linewidth = 1)
'''


def make_script(p, store='literals', floor=0.2, burn=2981, chunk=250, str_max=4000, title='', note='',
                name='MRI limits prototype', short='MRI-proto'):
    vals = flat_weights(p)
    assert len(vals) == p.n_params
    mats, off = matrix_offsets()
    mdecl = '\n'.join(f"var matrix<float> {nm:<4} = toMat(WT, {o}, {r}, {c})" for nm, r, c, o in mats)
    weights, n_fn = weights_block(vals, store, chunk, str_max)
    return TEMPLATE.format(title=title, note=note, name=name, short=short, burn=burn, s_sd=fmt(p.s_sd),
                           lvl_sd=fmt(p.lvl_sd), floor=fmt(floor), side_mu=float_list(p.side_mu),
                           side_sd=float_list(p.side_sd), n_w=len(vals), n_fn=n_fn, weights=weights,
                           matrices=mdecl, off_h3=off, off_h3_end=off + 64, off_hb3=off + 64)
