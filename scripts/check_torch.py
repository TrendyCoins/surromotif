"""Rule R2 (docs/rules.md): reference (numpy, mri.batch / mri.nt3) against research torch code on the same bars.

Research side: data files written from the bars by mri.research_bridge (for NT3 also smc_bars), then research SeriesData
('multi_g'), to_shapes, side_feats and CNNBase (weights from the npz). One continuous segment: TRAINING from the first bar
of the file, a short VALIDATION tail (its windows are not compared).

    python scripts/check_torch.py [--model F3_g | NT3_PRV_ch48 | PB_s64h24_d1] [--bars artifacts/bars/btc_fresh_tail.csv.gz]
        [--npz ...]
PB_s64h24_d1 (session 4): the NT3 inputs, research CNNBase with separable layers (make_cfg cnn_sep, research v6.3).
--floor-override 0.2 (session 5, rule R17): the reference with the floor float32(0.2); the research side keeps the floor of
the data file and replaces it for every series by research/sumo_class_ceil.py floor_override (mri.research_bridge). Windows
where the floor acts (median or mean |r| of the window below the floor, float32 as research) are counted with their
max |dlogit|.
"""
import argparse
import glob
import json
import os
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mri import L, TIE_EPS  # noqa: E402
from mri.batch import infer  # noqa: E402
from mri.features import bar_features, read_bars  # noqa: E402
from mri import nt3  # noqa: E402
from mri.params import load_npz  # noqa: E402
from mri.research_bridge import floor_override, write_research_files  # noqa: E402


def research_module():
    os.environ.setdefault('SMC_MODE', 'none')
    os.environ.setdefault('SMC_PROJECT_PATH', tempfile.mkdtemp() + '/')
    sys.path.insert(0, os.path.join(ROOT, 'research'))
    import sumo_class_train as T
    return T


def research_model(T, model, npz_path):
    import torch
    cfg = T.make_cfg(model, {'L': L}, 'a')
    net = T.CNNBase(cfg['L'], cfg['cnn_ch'], cfg['cnn_k'], cfg['cnn_dil'], cfg['head_hidden'], cfg['dropout'], 1.0,
                      cfg['cnn_bn'], cfg['cnn_res'], cfg['cnn_cdrop'], cfg['cnn_pool'], cfg['cnn_in'], cfg['cnn_sdrop'],
                      cfg['cnn_valid'], cfg['cnn_bias'], cfg['cnn_ma'], T.SIDE_COLS[cfg['side']], 4, None, 'rw', None,
                      T.N_LEVEL[cfg['price']], False, cfg.get('cnn_sep', False))
    sd = {}
    for p in sorted(glob.glob(npz_path)):
        with np.load(p) as d:
            sd.update({k[3:]: torch.from_numpy(d[k]) for k in d.files if k.startswith('sd_')})
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net


def research_logits(T, data_dir, npz_path, model='F3_g', asset='BTCUSDT', bs=32768, data=None, starts=None):
    """Research logits of the TRAINING windows of the asset (or of the windows starting at positions `starts`)."""
    import torch
    T.DATA_DIR, T.DATA_OVERLAY = data_dir, None
    if data is None:
        data = T.SeriesData([asset], '5M', 200, '2100-01-01', 'multi_g')
    w = data.windows(L)
    tr = w['st'][(w['sid'] == 0) & (data.part[w['st']] == 0)] if starts is None else np.asarray(starts)
    net = research_model(T, model, npz_path)
    out = {'st': tr, 'logit': [], 'flip_shape': [], 'flip_side': [], 'med': [], 'mabs': []}
    fl_all = w['floor_t']
    with torch.no_grad():
        for i in range(0, len(tr), bs):
            st = torch.from_numpy(tr[i:i + bs])
            W = data.gw(st, L)
            fl = fl_all[torch.zeros(len(st), dtype=torch.long)]
            X = T.to_shapes(W, fl)
            S = T.side_feats(data, st, L, W, fl)
            out['logit'].append(net(X, S)[0].double().numpy())
            # orientation of research (rule R1 since research v6.0: flip only below -ORIENT_EPS * divisor)
            div = torch.maximum(T.median_abs(W[:, 0]), fl)
            Xc = torch.cumsum(W[:, :3], dim=2)
            Xc = Xc - Xc.mean(dim=2, keepdim=True)
            out['flip_shape'].append((Xc[:, 0, -1] < -T.ORIENT_EPS * div).numpy())
            cs = torch.cumsum(W[:, 0], dim=1)
            out['flip_side'].append((cs[:, -1] - cs.mean(dim=1) < -T.ORIENT_EPS * div).numpy())
            out['med'].append(T.median_abs(W[:, 0]).float().numpy())      # as research Ceil collect (before the floor)
            out['mabs'].append(W[:, 0].abs().mean(dim=1).float().numpy())
    return {k: (np.concatenate(v) if isinstance(v, list) else v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='F3_g', choices=['F3_g', nt3.MODEL_ID, nt3.PB_MODEL_ID])
    ap.add_argument('--bars', default=os.path.join(ROOT, 'artifacts/bars/btc_fresh_tail.csv.gz'))
    ap.add_argument('--npz', nargs='+', default=None, help='default: prod a / b / c of the model')
    ap.add_argument('--floor', type=float, default=None, help='floor_L64; default: BTCUSDT from artifacts/pack/floors.json')
    ap.add_argument('--floor-override', type=float, default=None,
                    help='session 5 (R17): one floor for the reference and, by floor_override, for the research side')
    ap.add_argument('--out', default=None, help='json with the summary')
    a = ap.parse_args()
    if a.npz is None:
        a.npz = [os.path.join(ROOT, f'artifacts/weights/prod_{a.model}_L64_{s}_slim.npz') for s in 'abc']
    is_nt = a.model in (nt3.MODEL_ID, nt3.PB_MODEL_ID)        # PB: NT3 inputs, separable layers from the weights
    floor = a.floor
    if floor is None:
        with open(os.path.join(ROOT, 'artifacts/pack/floors.json')) as fh:
            floor = json.load(fh)['smc_series_BTCUSDT_5M_N200.npz']['floor_L64']
    bars = read_bars(a.bars)
    n = len(bars)
    tmp = tempfile.mkdtemp() + '/'
    t_start = str(np.datetime64(int(bars['time'].iloc[0]), 'ms'))
    t_valid = str(np.datetime64(int(bars['time'].iloc[n - 300]), 'ms'))
    rows = write_research_files(bars, tmp, 'BTCUSDT', t_start, t_valid, floor, with_bars=is_nt)
    T = research_module()
    f = bar_features(bars)
    hist = nt3.history_features(bars) if is_nt else None
    seg = rows['tr'][0]                           # research segment start (CSV row of the first TRAINING bar)
    fo = None if a.floor_override is None else float(np.float32(a.floor_override))
    summary = {'model': a.model, 'bars': os.path.basename(a.bars), 'n_bars': n, 'floor': floor, 'eps': TIE_EPS,
               'models': {}}
    if fo is not None:
        summary.update({'floor_file': floor, 'floor': fo, 'floor_override': True})
        floor = fo
        T.DATA_DIR, T.DATA_OVERLAY = tmp, None
        data = T.SeriesData(['BTCUSDT'], '5M', 200, '2100-01-01', 'multi_g')
    for path in a.npz:
        t0 = time.time()
        if fo is None:
            res = research_logits(T, tmp, path, a.model)
        else:
            with floor_override(data, L, fo):
                res = research_logits(T, tmp, path, a.model, data=data)
        Tbar = res['st'] + seg + (L - 1)           # CSV row of the window's last bar
        p, _ = load_npz(path)
        if is_nt:
            ref = nt3.infer(f, hist, p, floor, T=Tbar)
            summary['first_window_end'] = int(Tbar.min())
        else:
            ref = infer(f, p, floor, seg_start=seg, T=Tbar)
        d = np.abs(ref['logit'] - res['logit'])
        z = ref['x_end'] / ref['D']
        tie = ((np.abs(z) < 1e-3) | (res['flip_shape'] != (ref['sgn'] < 0)) | (res['flip_side'] != res['flip_shape']))
        row = {'windows': int(len(d)), 'ties': int(tie.sum()), 'tie_share': float(tie.mean()),
               'max_dlogit_non_tie': float(d[~tie].max()), 'p99_dlogit_non_tie': float(np.percentile(d[~tie], 99)),
               'max_dlogit_tie': float(d[tie].max()) if tie.any() else 0.0,
               'orient_mismatch_shape': int((res['flip_shape'] != (ref['sgn'] < 0)).sum()),
               'side_vs_shape_mismatch': int((res['flip_side'] != res['flip_shape']).sum()),
               'exact_zero_x_end': int((ref['x_end'] == 0).sum()),
               'pass_R2': bool(d[~tie].max() <= 1e-4), 'secs': round(time.time() - t0, 1)}
        if fo is not None:
            acts = (res['med'] < np.float32(fo)) | (res['mabs'] < np.float32(fo))
            row.update({'floor_acts': int(acts.sum()), 'floor_acts_median': int((res['med'] < np.float32(fo)).sum()),
                        'floor_acts_mean_only': int((acts & ~(res['med'] < np.float32(fo))).sum()),
                        'max_dlogit_floor_acts': float(d[acts].max()) if acts.any() else 0.0,
                        'max_dlogit_floor_acts_non_tie': float(d[acts & ~tie].max()) if (acts & ~tie).any() else 0.0})
        summary['models'][os.path.basename(path)] = row
        print(os.path.basename(path), json.dumps(row))
    if a.out:
        with open(a.out, 'w') as fh:
            json.dump(summary, fh, indent=1)


if __name__ == '__main__':
    main()
