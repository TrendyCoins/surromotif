"""Line-by-line Python transliteration of the NT3 Pine template (mri/pine_nt3.py CORE): the same state variables,
interleaved ring matrices, slot arithmetic, per-bar scale series read back with [63] / [65], ta.highest / ta.lowest /
ta.sma over 199 bars and the path chunks at fixed offsets. Checks the Pine algorithm (not its syntax) against
mri/nt3.py (tests/test_pine_nt3.py). Keep in sync with the template."""
import math

import numpy as np

from mri.pine_nt3 import chunk_offsets

L, NN, GAPMS, EPS = 64, 200, 450000.0, 0.0001
SQ199 = math.sqrt(199.0)


def slot(cnt, depth, back):
    return (cnt - back + depth * 64) % depth


def tap(R, depth, cnt, back):
    s = slot(cnt, depth, back)
    return R[:, 2 * s:2 * s + 2]


def conv3(R, depth, cnt, d, W):
    return W[0] @ tap(R, depth, cnt, 2 * d) + W[1] @ tap(R, depth, cnt, d) + W[2] @ tap(R, depth, cnt, 0)


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


def run(bars, p, floor, calc_bars=0, signed=None):
    K = [[w[:, :, k] for k in range(3)] for w in p.conv]
    CH, HID, NS = p.conv[0].shape[0], p.head0_w.shape[0], len(p.side_mu)
    o_, h_, l_, c_, t_ = (bars[k].to_numpy() for k in ('open', 'high', 'low', 'close', 'time'))
    n = len(c_)
    last_bar_index = n - 1
    RNG, BODY, WR, WABS = [], [], [0.0] * L, [0.0] * L
    prevC = prevD1 = prevD2 = math.nan
    okRun, firstR, nConf = 0, -1, 0
    R0, R1, R2, R3 = np.zeros((4, 6)), np.zeros((CH, 10)), np.zeros((CH, 18)), np.zeros((CH, 34))
    R4P, R4M = np.zeros((CH, 33)), np.zeros((CH, 33))
    hc = -1
    ser = {k: [] for k in ('nrm', 'rmeanB', 'bmedB', 'r16B', 'r64B', 'hmaxB', 'lminB', 'cmaB')}
    out = np.full(n, np.nan)
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
            relu_into(conv3(R0, 3, hc, 1, K[0]), R1, 5, hc)
            relu_into(conv3(R1, 5, hc, 2, K[1]), R2, 9, hc)
            relu_into(conv3(R2, 9, hc, 4, K[2]), R3, 17, hc)
            y4 = conv3(R3, 17, hc, 8, K[3])
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
                out[bi] = p.head3_b[0] + float(np.maximum(z, 0.0) @ p.head3_w[0])
                side_out[bi] = np.array(side)
    return out, side_out
