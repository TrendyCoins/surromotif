"""Rule R3 with the floor 0.2 (docs/rules.md, session 5, R17 and the R3 amendment): the reference with the floor FLOOR =
float32(0.2) on the VAL windows of BTC (window starts BTCUSDT__val__st of the npz 'refit' ALL, bars btc_old_tail) against
the scores of the Colab export of the SuMo_Class_Ceil cache (TF32 off, research data for every asset): 'refit' ALL a, b
(weights from the pool npz) and prod a (artifacts/weights). Threshold: max |dlogit| <= 1e-4 outside ties (R1); separately
— the windows where the floor acts (Ceil affected: median or mean |r| below the floor, against the trained floor of BTC), 834
by gate (д). The window inputs of the export (median and mean |r| of the window, float32) are compared with the reference.

    python scripts/pb5_r3.py [--out docs/checks/pb5/e2e_floor02.json]
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
from mri.ceil_export import TAGS, load_export  # noqa: E402
from mri.features import bar_features, read_bars  # noqa: E402
from mri.params import load_npz  # noqa: E402
from mri.pb import FLOOR, MODEL_ID  # noqa: E402
from mri.research_bridge import _ceil_module  # noqa: E402

VAL_START = '2025-07-01 00:00:00'
CEIL_JSON = os.path.join(ROOT, 'artifacts/calib/ceil_20261018_115426.json')
N_AFFECTED = 834


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='docs/checks/pb5/e2e_floor02.json')
    a = ap.parse_args()
    with open(CEIL_JSON, encoding='utf-8') as fh:
        fl_btc = json.load(fh)['floors_trained']['BTCUSDT']
    ce = _ceil_module()
    ex = load_export()
    bars = read_bars(os.path.join(ROOT, 'artifacts/bars/btc_old_tail.csv.gz'))
    f = bar_features(bars)
    hist = nt3.history_features(bars)
    t = bars['time'].to_numpy()
    seg = int(np.searchsorted(t, pd.Timestamp(VAL_START).value // 10 ** 6)) + 200   # first bar of the research VAL part
    weights = {'refit_a': f'artifacts/calib/pool_ALL_{MODEL_ID}_L64_a_refit_part*.npz',
               'refit_b': f'artifacts/calib/pool_ALL_{MODEL_ID}_L64_b_refit_part*.npz',
               'prod_a': f'artifacts/weights/prod_{MODEL_ID}_L64_a_slim.npz'}
    _, arr = load_npz(os.path.join(ROOT, weights['refit_a']))
    st = arr['BTCUSDT__val__st']
    off = int(st.min())
    Tbar = st - off + seg + (L - 1)
    okw = np.convolve(f['ok'].astype(int), np.ones(L, dtype=int), 'valid') == L
    starts_ok = np.where(okw[seg:])[0]
    same_set = bool(len(starts_ok) == len(st) and np.array_equal(starts_ok, st - off))
    rw = np.lib.stride_tricks.sliding_window_view(np.nan_to_num(f['r']), L)[Tbar - (L - 1)]
    med_ref, mabs_ref = np.median(np.abs(rw), axis=1), np.mean(np.abs(rw), axis=1)
    out = {'model': MODEL_ID, 'floor': FLOOR, 'windows': int(len(st)), 'same_window_set': same_set,
           'last_end': str(pd.Timestamp(int(t[Tbar[-1]]), unit='ms')), 'floor_trained_btc': fl_btc, 'models': {}}
    for key, wpath in weights.items():
        tag = TAGS[key]
        e = ex[tag]['BTCUSDT']
        p, arr_ = load_npz(os.path.join(ROOT, wpath))
        if key != 'prod_a' and not np.array_equal(arr_['BTCUSDT__val__st'], st):
            raise SystemExit(f'{tag}: other window starts than refit a')
        ref = nt3.infer(f, hist, p, FLOOR, T=Tbar)
        sp = e['sp'].astype(np.float64)
        if len(sp) != len(st):
            raise SystemExit(f'{tag}: {len(sp)} exported windows, {len(st)} in the npz')
        d = np.abs(ref['logit'] - sp)
        tie = np.abs(ref['x_end'] / ref['D']) < 1e-3
        aff = ce.affected(e['med'], e['mabs'], fl_btc, ce.CEIL['floor'])
        acts = (np.float32(med_ref) < np.float32(FLOOR)) | (np.float32(mabs_ref) < np.float32(FLOOR))
        row = {'tag': tag, 'ties': int(tie.sum()), 'max_dlogit_non_tie': float(d[~tie].max()),
               'p99': float(np.percentile(d, 99)), 'max_dlogit_tie': float(d[tie].max()) if tie.any() else 0.0,
               'n_dlogit_gt_1e-4': int((d > 1e-4).sum()), 'n_affected': int(aff.sum()),
               'max_dlogit_affected': float(d[aff].max()) if aff.any() else 0.0,
               'n_floor_acts_reference': int(acts.sum()), 'affected_eq_floor_acts': bool(np.array_equal(aff, acts)),
               'med_max_abs_diff': float(np.abs(med_ref - e['med'].astype(np.float64)).max()),
               'mabs_max_abs_diff': float(np.abs(mabs_ref - e['mabs'].astype(np.float64)).max())}
        row['pass_R3'] = bool(same_set and row['max_dlogit_non_tie'] <= 1e-4 and row['n_affected'] == N_AFFECTED)
        out['models'][key] = row
        print(key, json.dumps(row))
    out['pass_R3'] = all(r['pass_R3'] for r in out['models'].values())
    with open(os.path.join(ROOT, a.out), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    print('R3 with the floor 0.2:', out['pass_R3'])


if __name__ == '__main__':
    main()
