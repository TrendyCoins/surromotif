"""Investigation of the R3 failure of NT3 (session 3): where does the difference between the reference and the stored VAL
logits of the NT3 'refit' models come from?

    1) research code itself (SeriesData / to_shapes / side_feats / CNNBase on CPU, float32) on the research data files
       written from the same bars (mri.research_bridge, with smc_bars) — against the reference and against the stored logits;
    2) the same research model with TF32 convolutions emulated (inputs and weights of every conv rounded to TF32, 10-bit
       mantissa, round to nearest or truncation) — the size and the structure (lag correlation over consecutive windows)
       of the TF32 noise, compared with the observed difference.

    python scripts/legacy/nt3_r3_probe.py [--npz artifacts/calib/pool_ALL_NT3_PRV_ch48_L64_a_refit_part*.npz]
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from check_torch import research_model, research_module  # noqa: E402
from mri import L, nt3  # noqa: E402
from mri.features import bar_features, read_bars  # noqa: E402
from mri.params import load_npz  # noqa: E402
from mri.research_bridge import write_research_files  # noqa: E402

VAL_START = '2025-07-01 00:00:00'


def tf32(x, mode):
    import torch
    i = x.contiguous().view(torch.int32)
    if mode == 'trunc':
        i = i & ~0x1FFF
    else:                                   # round to nearest even on the 13 dropped mantissa bits
        i = (i + 0x0FFF + ((i >> 13) & 1)) & ~0x1FFF
    return i.view(torch.float32)


def stats(e):
    a = np.abs(e)
    lag = lambda k: float(np.corrcoef(e[:-k], e[k:])[0, 1])
    return {'max': float(a.max()), 'p99': float(np.percentile(a, 99)), 'p50': float(np.median(a)),
            'mean': float(e.mean()), 'lag1_corr': lag(1), 'lag32_corr': lag(32), 'lag64_corr': lag(64),
            'lag200_corr': lag(200), 'n_gt_1e-4': int((a > 1e-4).sum())}


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=os.path.join(ROOT, 'artifacts/calib/pool_ALL_NT3_PRV_ch48_L64_a_refit_part*.npz'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs/checks/legacy/nt3/r3_probe.json'))
    a = ap.parse_args()
    with open(os.path.join(ROOT, 'artifacts/pack/floors.json')) as fh:
        floor = json.load(fh)['smc_series_BTCUSDT_5M_N200.npz']['floor_L64']
    bars = read_bars(os.path.join(ROOT, 'artifacts/bars/btc_old_tail.csv.gz'))
    tmp = tempfile.mkdtemp() + '/'
    rows = write_research_files(bars, tmp, 'BTCUSDT', str(np.datetime64(int(bars['time'].iloc[0]), 'ms')), VAL_START,
                                floor, with_bars=True)
    T = research_module()
    T.DATA_DIR, T.DATA_OVERLAY = tmp, None
    data = T.SeriesData(['BTCUSDT'], '5M', 200, '2100-01-01', 'multi_g')
    w = data.windows(L)
    va = w['st'][data.part[w['st']] == 1]
    p, arr = load_npz(a.npz)
    st, sp = arr['BTCUSDT__val__st'], arr['BTCUSDT__val__sp'].astype(np.float64)
    assert np.array_equal(va - va.min(), st - st.min()), 'window sets differ'

    def run(mode):
        net = research_model(T, nt3.MODEL_ID, a.npz)
        if mode:
            for blk in net.blocks:
                conv = blk[0]
                conv.weight.data = tf32(conv.weight.data, mode)
                conv.register_forward_pre_hook(lambda m, inp: (tf32(inp[0], mode),))
        out = []
        with torch.no_grad():
            for i in range(0, len(va), 32768):
                s_ = torch.from_numpy(va[i:i + 32768])
                W = data.gw(s_, L)
                fl = w['floor_t'][torch.zeros(len(s_), dtype=torch.long)]
                out.append(net(T.to_shapes(W, fl), T.side_feats(data, s_, L, W, fl))[0].double().numpy())
        return np.concatenate(out)

    base = run(None)
    Tbar = va - va.min() + rows['va'][0] + L - 1
    ref = nt3.infer(bar_features(bars), nt3.history_features(bars), p, floor, T=Tbar)
    res = {'npz': os.path.basename(a.npz), 'windows': int(len(va)),
           'research_cpu_vs_reference': stats(base - ref['logit']),
           'research_cpu_vs_stored': stats(base - sp),
           'tf32_emulated_vs_research_cpu': {m: stats(run(m) - base) for m in ('rne', 'trunc')}}
    print(json.dumps(res, indent=1))
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
