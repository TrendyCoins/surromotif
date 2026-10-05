"""Line-by-line Python transliteration of the Pine template (mri/pine_gen.py TEMPLATE): the same state variables,
interleaved ring matrices, slot arithmetic, normalized EMA and csS history. Checks the Pine algorithm (not its
syntax) against mri/stream.py. Keep in sync with the template."""
import math

import numpy as np

L, NN, GAPMS, EPS, DAY = 64, 200, 450000.0, 0.0001, 288
A7, A1 = 2.0 / 2017.0, 2.0 / 289.0


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


def run(bars, p, floor, burn, calc_bars=0):
    K = [[w[:, :, k] for k in range(3)] for w in p.conv]
    t_, h_, l_, c_ = (bars[k].to_numpy() for k in ('time', 'high', 'low', 'close'))
    n = len(c_)
    last_bar_index = n - 1
    RNG, WR, WABS = [], [0.0] * L, [0.0] * L
    prevC = prevD1 = prevD2 = math.nan
    okRun, firstR = 0, -1
    s7 = w7 = s1 = w1 = csum = 0.0
    R0, R1, R2, R3 = np.zeros((4, 6)), np.zeros((32, 10)), np.zeros((32, 18)), np.zeros((32, 34))
    R4P, R4M = np.zeros((32, 33)), np.zeros((32, 33))
    hc = -1
    csS, e1S = [], []
    out = np.full(n, np.nan)
    for bi in range(n):
        time, high, low, close = t_[bi], h_[bi], l_[bi], c_[bi]
        hl2, hlc3 = (high + low) / 2.0, (high + low + close) / 3.0
        rngNow = high - low
        nrm = float(np.median(RNG)) if len(RNG) == NN else math.nan
        goodN = not math.isnan(nrm) and nrm > 0
        gap = bi > 0 and (time - t_[bi - 1]) > GAPMS
        bad = (not goodN) or gap or math.isnan(prevC)
        r = math.nan if bad else (close - prevC) / nrm
        st1 = math.nan if bad else (hl2 - prevC) / nrm - prevD1
        st2 = math.nan if bad else (hlc3 - prevC) / nrm - prevD2
        gg = math.nan if bad else rngNow / nrm
        ok = not any(math.isnan(v) for v in (r, st1, st2, gg))
        heavyFrom = 0 if calc_bars == 0 else max(last_bar_index - calc_bars - L, 0)
        # confirmed bar
        RNG.append(rngNow)
        if len(RNG) > NN:
            RNG.pop(0)
        prevD1 = (hl2 - close) / nrm if goodN else math.nan
        prevD2 = (hlc3 - close) / nrm if goodN else math.nan
        prevC = close
        okRun = okRun + 1 if ok else 0
        WR.pop(0)
        WR.append(r if ok else 0.0)
        WABS.pop(0)
        WABS.append(abs(r) if ok else 0.0)
        if ok:
            if firstR < 0:
                firstR = bi
            s7 = abs(r) + (1.0 - A7) * s7
            w7 = 1.0 + (1.0 - A7) * w7
            s1 = abs(r) + (1.0 - A1) * s1
            w1 = 1.0 + (1.0 - A1) * w1
            csum += r
        csS.append(csum)
        e1S.append(s1 / w1 if w1 > 0 else math.nan)

        def hist(a, m):
            return a[bi - m] if bi - m >= 0 else math.nan
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
            ready = okRun >= L and firstR >= 0 and bi - firstR >= burn and hc >= L - 1
            if ready:
                D = max(float(np.median(WABS)), floor)
                xe = sum(i * WR[i] for i in range(L)) / L
                sgn = -1.0 if xe < -EPS * D else 1.0
                M4 = R4P if sgn > 0 else R4M
                feat = np.zeros(112)
                feat[:32] = (M4 @ np.full(33, 1.0 / 33.0)) / D
                feat[32:64] = M4.max(axis=1) / D
                feat[64:96] = M4[:, p4] / D
                nrm1 = hist(e1S, L)
                mins = math.floor(time / 60000.0) % 1440
                hr = mins / 60.0 * (2.0 * math.pi / 24.0)
                dw = ((math.floor(time / 86400000.0) + 3) % 7) * (2.0 * math.pi / 7.0)
                side = [math.log(D) - math.log(s7 / w7), math.sin(hr), math.cos(hr), math.sin(dw), math.cos(dw)]
                csA = hist(csS, L + 1)
                mv = [(csA - hist(csS, L + 1 + nd * DAY)) / math.sqrt(nd * DAY) for nd in (1, 3, 7)]
                mv += [(hist(csS, L + 1 + 16 * k) - hist(csS, L + 1 + 16 * (k + 1))) / 4.0 for k in range(7, -1, -1)]
                side += [v / nrm1 * sgn for v in mv]
                feat[96:] = (np.array(side) - p.side_mu) / p.side_sd
                z = p.head0_w @ feat + p.head0_b
                out[bi] = p.head3_b[0] + float(np.maximum(z, 0.0) @ p.head3_w[0])
    return out
