"""Line-by-line Python transliteration of the PB Pine template (mri/pine_pb.py CORE / FOOT): the same state variables,
interleaved ring matrices, slot arithmetic, separable layers (per channel taps into DWM, one matrix product), per-bar scale
series read back with [63] / [65], ta.highest / ta.lowest / ta.sma over 199 bars, path chunks at fixed offsets, the ruler
search, the `var` live part (n1, n2, dd, nW) and the display. Checks the Pine algorithm (not its syntax) against mri.pb
(tests/test_pine_pb.py). Keep in sync with the template."""
import math

import numpy as np

from mri.pine_nt3 import chunk_offsets

L, NN, GAPMS, EPS = 64, 200, 450000.0, 0.0001
SQ199 = math.sqrt(199.0)
PEAK_EPS = 0.000000000001


def slot(cnt, depth, back):
    return (cnt - back + depth * 64) % depth


def tap(R, depth, cnt, back):
    s = slot(cnt, depth, back)
    return R[:, 2 * s:2 * s + 2]


def conv3(R, depth, cnt, d, W):
    return W[0] @ tap(R, depth, cnt, 2 * d) + W[1] @ tap(R, depth, cnt, d) + W[2] @ tap(R, depth, cnt, 0)


def sep3(R, depth, cnt, d, A, PW, DWM):
    c0, c1, c2 = 2 * slot(cnt, depth, 2 * d), 2 * slot(cnt, depth, d), 2 * slot(cnt, depth, 0)
    for i in range(len(A[0])):
        a0, a1, a2 = A[0][i], A[1][i], A[2][i]
        DWM[i, 0] = a0 * R[i, c0] + a1 * R[i, c1] + a2 * R[i, c2]
        DWM[i, 1] = a0 * R[i, c0 + 1] + a1 * R[i, c1 + 1] + a2 * R[i, c2 + 1]
    return PW @ DWM


def relu_into(y, R, depth, cnt):
    s = slot(cnt, depth, 0)
    R[:, 2 * s] = np.maximum(y[:, 0], 0.0)
    R[:, 2 * s + 1] = np.maximum(y[:, 1], 0.0)


def nz(x):
    return 0.0 if x is None or math.isnan(x) else x


def clip_z(x):
    return nz(max(min(x, 20.0), -20.0))


def clip_log(v):
    return nz(max(min(math.log(max(v, 0.001)), 5.0), -5.0))


def ruler_z(x, RQ, RZ):
    n = len(RQ)
    if x < RQ[0]:
        return RZ[0]
    if x >= RQ[n - 1]:
        return RZ[n - 1]
    j = 0
    while RQ[j + 1] <= x:
        j += 1
    q0 = RQ[j]
    return RZ[j] if x == q0 else (RZ[j + 1] - RZ[j]) / (RQ[j + 1] - q0) * (x - q0) + RZ[j]


def log_m_of(z, mE, sE, S_LO):
    m = max(mE, 0.0)
    s = min(max(sE, S_LO), 1.0)
    zc = min(z, m / max(1.0 - s * s, PEAK_EPS)) if s < 1.0 else z
    return -(zc - m) * (zc - m) / (2.0 * s * s) - math.log(s) + zc * zc / 2.0


def run(bars, p, floor, ruler, calc_bars=0, signed=None, span=8064, warm=288, s_lo=0.75, ceil=(34.0, 21.0)):
    """Per-bar dict of arrays: logit, z, m_ema, s_ema, log_m, n (debug plot: NaN where no window), shown, M (plotted
       value), side (bar -> 17 raw side inputs)."""
    K1 = [p.conv[0][:, :, k] for k in range(3)]
    DW = [[p.conv[i][:, 0, k] for k in range(3)] for i in (1, 2, 3)]
    PW = p.pw
    RQ, RZ = list(ruler[0]), list(ruler[1])
    CH, NS = p.conv[0].shape[0], len(p.side_mu)
    o_, h_, l_, c_, t_ = (bars[k].to_numpy() for k in ('open', 'high', 'low', 'close', 'time'))
    n = len(c_)
    last_bar_index = n - 1
    RNG, BODY, WR, WABS = [], [], [0.0] * L, [0.0] * L
    prevC = prevD1 = prevD2 = math.nan
    okRun, firstR, nConf = 0, -1, 0
    n1 = n2 = dd = 0.0
    nW = -1
    KEMA = 1.0 - 2.0 / (span + 1.0)
    R0, R1, R2, R3 = np.zeros((4, 6)), np.zeros((CH, 10)), np.zeros((CH, 18)), np.zeros((CH, 34))
    R4P, R4M = np.zeros((CH, 33)), np.zeros((CH, 33))
    DWM = np.zeros((CH, 2))
    hc = -1
    ser = {k: [] for k in ('nrm', 'rmeanB', 'bmedB', 'r16B', 'r64B', 'hmaxB', 'lminB', 'cmaB')}
    out = {k: np.full(n, np.nan) for k in ('logit', 'z', 'm_ema', 's_ema', 'log_m', 'n', 'M')}
    out['shown'] = np.zeros(n, dtype=bool)
    side_out = {}
    na = math.nan

    def hist(name, bi, m):
        return ser[name][bi - m] if bi - m >= 0 else na

    for bi in range(n):
        time, op, high, low, close = t_[bi], o_[bi], h_[bi], l_[bi], c_[bi]
        hl2, hlc3 = (high + low) / 2.0, (high + low + close) / 3.0
        rngNow = high - low
        nrm = float(np.median(RNG)) if len(RNG) == NN else na
        goodN = not math.isnan(nrm) and nrm > 0
        gap = bi > 0 and (time - t_[bi - 1]) > GAPMS
        bad = (not goodN) or gap or math.isnan(prevC)
        r = na if bad else (close - prevC) / nrm
        st1 = na if bad else (hl2 - prevC) / nrm - prevD1
        st2 = na if bad else (hlc3 - prevC) / nrm - prevD2
        gg = na if bad else rngNow / nrm
        ok = not any(math.isnan(v) for v in (r, st1, st2, gg))
        full = len(RNG) == NN
        ser['nrm'].append(nrm)
        ser['rmeanB'].append(sum(RNG) / NN if full else na)
        ser['bmedB'].append(max(float(np.median(BODY)), 0.05 * nrm) if full else na)
        ser['r16B'].append(sum(RNG[NN - 16:]) / 16 if full else na)
        ser['r64B'].append(sum(RNG[NN - 64:]) / 64 if full else na)
        lo = bi - (NN - 2)
        ser['hmaxB'].append(float(np.max(h_[lo:bi + 1])) if lo >= 0 else na)
        ser['lminB'].append(float(np.min(l_[lo:bi + 1])) if lo >= 0 else na)
        ser['cmaB'].append(sum(c_[lo:bi + 1]) / (NN - 1) if lo >= 0 else na)
        rmedW, rmeanW, bmedW = hist('nrm', bi, L - 1), hist('rmeanB', bi, L - 1), hist('bmedB', bi, L - 1)
        hmaxW, lminW = hist('hmaxB', bi, L + 1), hist('lminB', bi, L + 1)
        wW = hmaxW - lminW
        c2W = c_[bi - L - 1] if bi - L - 1 >= 0 else na
        HST = None
        if bi >= L + NN - 1:
            HST = [clip_z((c_[bi - e] - o_[bi - s]) / math.sqrt(float(m)) / rmedW) for m, e, s in chunk_offsets()]
            HST += [2.0 * (c2W - lminW) / wW - 1.0 if wW > 0 else 0.0,
                    nz(max(min(math.log(max(wW, 1e-12) / (SQ199 * rmedW)), 5.0), -5.0)),
                    clip_z((c2W - hist('cmaB', bi, L + 1)) / (SQ199 * rmedW)),
                    clip_log(hist('r16B', bi, L - 1) / rmeanW), clip_log(hist('r64B', bi, L - 1) / rmeanW),
                    clip_log(rmeanW / rmedW), clip_log(bmedW / rmedW)]
        heavyFrom = 0 if calc_bars == 0 else max(last_bar_index - calc_bars - L, 0)
        logit = zz = mE = sE = logM = na
        # confirmed bar
        RNG.append(rngNow)
        BODY.append(abs(close - op))
        if len(RNG) > NN:
            RNG.pop(0)
            BODY.pop(0)
        prevD1 = (hl2 - close) / nrm if goodN else na
        prevD2 = (hlc3 - close) / nrm if goodN else na
        prevC = close
        okRun = okRun + 1 if ok else 0
        nConf += 1
        WR.pop(0)
        WR.append(r if ok else 0.0)
        WABS.pop(0)
        WABS.append(abs(r) if ok else 0.0)
        if ok and firstR < 0:
            firstR = bi
        if bi >= heavyFrom:
            hc += 1
            p0 = slot(hc, 3, 0)
            u = [r / p.s_sd, st1 / p.s_sd, st2 / p.s_sd, gg / p.lvl_sd] if ok else [0.0] * 4
            R0[:, 2 * p0] = u
            R0[:, 2 * p0 + 1] = [-u[0], -u[1], -u[2], u[3]]
            relu_into(conv3(R0, 3, hc, 1, K1), R1, 5, hc)
            relu_into(sep3(R1, 5, hc, 2, DW[0], PW[0], DWM), R2, 9, hc)
            relu_into(sep3(R2, 9, hc, 4, DW[1], PW[1], DWM), R3, 17, hc)
            y4 = sep3(R3, 17, hc, 8, DW[2], PW[2], DWM)
            p4 = slot(hc, 33, 0)
            R4P[:, p4] = np.maximum(y4[:, 0], 0.0)
            R4M[:, p4] = np.maximum(y4[:, 1], 0.0)
            if okRun >= L and bi - (L - 1) >= NN and hc >= L - 1:
                D = max(float(np.median(WABS)), floor)
                xe = 0.0
                for i in range(L):
                    xe += i * WR[i]
                xe /= L
                sgn = -1.0 if xe < -EPS * D else 1.0
                M4 = R4P if sgn > 0 else R4M
                feat = np.zeros(3 * CH + NS)
                feat[:CH] = (M4 @ np.full(33, 1.0 / 33.0)) / D
                feat[CH:2 * CH] = M4.max(axis=1) / D
                feat[2 * CH:3 * CH] = M4[:, p4] / D
                side = [math.log(D), math.log(max(sum(WABS) / L, floor))]
                side += [HST[j] * sgn if signed[j] else HST[j] for j in range(NS - 2)]
                feat[3 * CH:] = (np.array(side) - p.side_mu) / p.side_sd
                z = p.head0_w @ feat + p.head0_b
                acc = p.head3_b[0]
                for i in range(len(z)):
                    acc += max(z[i], 0.0) * p.head3_w[0][i]
                logit = acc
                side_out[bi] = np.array(side)
                zz = ruler_z(logit, RQ, RZ)
                nW += 1
                n1 = KEMA * n1 + zz
                n2 = KEMA * n2 + zz * zz
                dd = KEMA * dd + 1.0
                mE = n1 / dd
                sE = math.sqrt(max(n2 / dd - mE * mE, 0.0))
                logM = log_m_of(zz, mE, sE, s_lo)
        # display
        shownM = not math.isnan(logM) and nW >= warm
        Mraw = math.exp(abs(logM)) if shownM else na
        more, less = shownM and logM > 0, shownM and logM < 0
        out['M'][bi] = na if math.isnan(Mraw) else (min(Mraw, ceil[0]) if more else min(Mraw, ceil[1]) if less else 1.0)
        out['shown'][bi] = shownM
        for k, v in (('logit', logit), ('z', zz), ('m_ema', mE), ('s_ema', sE), ('log_m', logM)):
            out[k][bi] = v
        out['n'][bi] = na if math.isnan(logit) else nW
    out['side'] = side_out
    return out
