"""Rule R3 (docs/rules.md): reference with the 'refit' weights on the VAL windows of BTC against the logits saved by
research (calib npz, key BTCUSDT__val__sp). Checks data preparation end to end, not only the net.

Mapping: 'st' in the npz is the window start in the research pool array; BTC is the first asset, its VALIDATION
part starts at offset min(st) (first VAL window starts at the first VAL bar — checked by count and end time).
Research semantics reproduced: VALIDATION starts 200 bars after 2025-07-01 00:00 (warm-up dropped), the EMA does not
see those 200 bars, moves before the window are clipped at the VAL part start.

NT3 (rules R3 amendment, session 3): the history inputs run over consecutive CSV bars (the part + 200 CSV bars before it,
research nt_segment / SeriesData._nt_arrays), not clipped at the part start, and there is no EMA: the reference computes
on the continuous bars of the tail without exclusions.

    python scripts/check_e2e.py [--seed a] [--model NT3_PRV_ch48 --set ALL] [--logits artifacts/calib/nt3_btc_val_logits.npz]

--logits: the research logits of the same windows recomputed in Colab with TF32 off / on (scripts/colab_nt3_val_logits.py,
after the R3 failure of session 3): the reference is compared with the TF32-off logits as well (the rule's threshold), and
the TF32-on logits with the stored ones.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mri import L, nt3  # noqa: E402
from mri.batch import infer  # noqa: E402
from mri.features import bar_features, read_bars  # noqa: E402
from mri.params import load_npz  # noqa: E402

VAL_START = '2025-07-01 00:00:00'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', default='a')
    ap.add_argument('--model', default='F3_g', choices=['F3_g', nt3.MODEL_ID])
    ap.add_argument('--set', default='C', choices=['C', 'ALL'], help="training set of the 'refit' model")
    ap.add_argument('--logits', default=None, help='npz of scripts/colab_nt3_val_logits.py')
    ap.add_argument('--floor', type=float, default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    floor = a.floor
    if floor is None:
        with open(os.path.join(ROOT, 'artifacts/pack/floors.json')) as fh:
            floor = json.load(fh)['smc_series_BTCUSDT_5M_N200.npz']['floor_L64']
    bars = read_bars(os.path.join(ROOT, 'artifacts/bars/btc_old_tail.csv.gz'))
    is_nt = a.model == nt3.MODEL_ID
    p, arr = load_npz(os.path.join(ROOT, f'artifacts/calib/pool_{a.set}_{a.model}_L64_{a.seed}_refit_part*.npz'))
    st, sp = arr['BTCUSDT__val__st'], arr['BTCUSDT__val__sp'].astype(np.float64)
    t = bars['time'].to_numpy()
    i0 = int(np.searchsorted(t, pd.Timestamp(VAL_START).value // 10 ** 6))
    seg = i0 + 200                                # first bar of the research VALIDATION part
    if is_nt:
        f = bar_features(bars)
    else:
        excl = np.zeros(len(bars), dtype=bool)
        excl[i0:seg] = True
        f = bar_features(bars, ema_exclude=excl)
    off = int(st.min())
    Tbar = st - off + seg + (L - 1)
    okw = np.convolve(f['ok'].astype(int), np.ones(L, dtype=int), 'valid') == L
    starts_ok = np.where(okw[seg - 0:])[0]                         # valid window starts from the VAL part start
    same_set = len(starts_ok) == len(st) and np.array_equal(starts_ok, st - off)
    if is_nt:
        ref = nt3.infer(f, nt3.history_features(bars), p, floor, T=Tbar)
    else:
        ref = infer(f, p, floor, seg_start=seg, T=Tbar)
    d = np.abs(ref['logit'] - sp)
    z = ref['x_end'] / ref['D']
    tie = np.abs(z) < 1e-3
    med = np.median(np.abs(np.lib.stride_tricks.sliding_window_view(np.nan_to_num(f['r']), L)[Tbar - (L - 1)]), axis=1)
    floor_bind = med < floor
    res = {'model': a.model, 'set': a.set, 'seed': a.seed, 'windows': int(len(st)), 'same_window_set': bool(same_set),
           'last_end': str(pd.Timestamp(int(t[Tbar[-1]]), unit='ms')), 'floor': floor,
           'ties': int(tie.sum()), 'floor_binding': int(floor_bind.sum()),
           'max_dlogit_non_tie': float(d[~tie].max()), 'p99': float(np.percentile(d, 99)),
           'max_dlogit_tie': float(d[tie].max()) if tie.any() else 0.0,
           'max_dlogit_floor_binding': float(d[floor_bind].max()) if floor_bind.any() else 0.0,
           'n_dlogit_gt_1e-4': int((d > 1e-4).sum())}
    res['pass_R3'] = bool(res['same_window_set'] and res['max_dlogit_non_tie'] <= 1e-4)
    if a.logits:
        run = f'pool_{a.set}_{a.model}_L64_{a.seed}_refit'
        with np.load(a.logits) as z:
            off, on = z[f'{run}__tf32_off'].astype(np.float64), z[f'{run}__tf32_on'].astype(np.float64)
            res['colab_device'] = str(z['device'])
            d2 = np.abs(ref['logit'] - off)
            res['tf32_off'] = {'max_dlogit_non_tie': float(d2[~tie].max()), 'p99': float(np.percentile(d2, 99)),
                               'max_dlogit_tie': float(d2[tie].max()) if tie.any() else 0.0,
                               'max_dlogit_floor_binding': float(d2[floor_bind].max()) if floor_bind.any() else 0.0,
                               'n_dlogit_gt_1e-4': int((d2 > 1e-4).sum())}
            if 'feats_side17' in z.files:          # research side inputs of every 10th window (float32) vs the reference
                rel = z['feats_st'] - z['feats_st'][0]                      # every 10th window from the first
                k = np.searchsorted(st - st[0], rel)
                assert np.array_equal(st[k] - st[0], rel)
                dside = np.abs(ref['side_raw'][k] - z['feats_side17'].astype(np.float64))
                res['side17_max_abs_diff'] = dict(zip(nt3.SIDE_NAMES, np.round(dside.max(axis=0), 9).tolist()))
                res['side17_windows'] = int(len(k))
        res['tf32_on_vs_stored_max'] = float(np.abs(on - sp).max())
        res['pass_R3_tf32_off'] = bool(res['same_window_set'] and res['tf32_off']['max_dlogit_non_tie'] <= 1e-4)
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, 'w') as fh:
            json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
