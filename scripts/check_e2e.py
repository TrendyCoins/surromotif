"""Rule R3 (docs/rules.md): reference with the 'refit' weights on the VAL windows of BTC against the logits saved by
research (calib npz, key BTCUSDT__val__sp). Checks data preparation end to end, not only the net.

Mapping: 'st' in the npz is the window start in the research pool array; BTC is the first asset, its VALIDATION
part starts at offset min(st) (first VAL window starts at the first VAL bar — checked by count and end time).
Research semantics reproduced: VALIDATION starts 200 bars after 2025-07-01 00:00 (warm-up dropped), the EMA does not
see those 200 bars, moves before the window are clipped at the VAL part start.

    python scripts/check_e2e.py [--seed a]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mri import L  # noqa: E402
from mri.batch import infer  # noqa: E402
from mri.features import bar_features, read_bars  # noqa: E402
from mri.params import load_npz  # noqa: E402

VAL_START = '2025-07-01 00:00:00'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', default='a')
    ap.add_argument('--floor', type=float, default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    floor = a.floor
    if floor is None:
        with open(os.path.join(ROOT, 'artifacts/pack/floors.json')) as fh:
            floor = json.load(fh)['smc_series_BTCUSDT_5M_N200.npz']['floor_L64']
    bars = read_bars(os.path.join(ROOT, 'artifacts/bars/btc_old_tail.csv.gz'))
    p, arr = load_npz(os.path.join(ROOT, f'artifacts/calib/pool_C_F3_g_L64_{a.seed}_refit_part*.npz'))
    st, sp = arr['BTCUSDT__val__st'], arr['BTCUSDT__val__sp'].astype(np.float64)
    t = bars['time'].to_numpy()
    i0 = int(np.searchsorted(t, pd.Timestamp(VAL_START).value // 10 ** 6))
    seg = i0 + 200                                # first bar of the research VALIDATION part
    excl = np.zeros(len(bars), dtype=bool)
    excl[i0:seg] = True
    f = bar_features(bars, ema_exclude=excl)
    off = int(st.min())
    Tbar = st - off + seg + (L - 1)
    okw = np.convolve(f['ok'].astype(int), np.ones(L, dtype=int), 'valid') == L
    starts_ok = np.where(okw[seg - 0:])[0]                         # valid window starts from the VAL part start
    same_set = len(starts_ok) == len(st) and np.array_equal(starts_ok, st - off)
    ref = infer(f, p, floor, seg_start=seg, T=Tbar)
    d = np.abs(ref['logit'] - sp)
    z = ref['x_end'] / ref['D']
    tie = np.abs(z) < 1e-3
    med = np.median(np.abs(np.lib.stride_tricks.sliding_window_view(np.nan_to_num(f['r']), L)[Tbar - (L - 1)]), axis=1)
    floor_bind = med < floor
    res = {'seed': a.seed, 'windows': int(len(st)), 'same_window_set': bool(same_set),
           'last_end': str(pd.Timestamp(int(t[Tbar[-1]]), unit='ms')), 'floor': floor,
           'ties': int(tie.sum()), 'floor_binding': int(floor_bind.sum()),
           'max_dlogit_non_tie': float(d[~tie].max()), 'p99': float(np.percentile(d, 99)),
           'max_dlogit_tie': float(d[tie].max()) if tie.any() else 0.0,
           'max_dlogit_floor_binding': float(d[floor_bind].max()) if floor_bind.any() else 0.0,
           'n_dlogit_gt_1e-4': int((d > 1e-4).sum())}
    res['pass_R3'] = bool(res['same_window_set'] and res['max_dlogit_non_tie'] <= 1e-4)
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, 'w') as fh:
            json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
