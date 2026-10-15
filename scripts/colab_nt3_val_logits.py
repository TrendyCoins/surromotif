"""Colab: logits of the BTC VAL windows of the NT3 'refit' models, recomputed by research code with TF32 off and on.

Why (rule R3, session 3): the reference (and research code on CPU, float32) reproduces the stored VAL logits of the NT3
'refit' models (calib npz, BTCUSDT__val__sp) only to max |dlogit| ~0.012 (p50 ~9e-4) instead of 1e-4; an emulation of TF32
convolutions (cuDNN default on Ampere GPUs) gives the same size and structure of the difference. This run settles it on the
same data the models saw (SMC/data/): with TF32 off the logits must match the repo reference (then R3 is checked against
them), with TF32 on they should reproduce the stored ones.

Run in Colab with a GPU, after mounting Drive:
    1) put research/sumo_class_train.py of this repo into the runtime (Files panel: /content), or set MRI_RESEARCH_DIR;
    2) set MRI_PROJECT to the folder on your Drive that contains SMC/ (with SMC/data/ and SMC/runs/pool/);
    3) run this file. Output: nt3_btc_val_logits.npz (~2.5 MB) — put it into artifacts/calib/.
"""
import os
import sys
import time

import numpy as np

PROJECT = os.environ.get('MRI_PROJECT', '')
CODE = os.environ.get('MRI_RESEARCH_DIR', '/content')
OUT = 'nt3_btc_val_logits.npz'
MODEL, L = 'NT3_PRV_ch48', 64
RUNS = {'pool_ALL_NT3_PRV_ch48_L64_a_refit': 'a', 'pool_ALL_NT3_PRV_ch48_L64_b_refit': 'b',
        'pool_C_NT3_PRV_ch48_L64_a_refit': 'a'}

if not PROJECT or not os.path.isdir(os.path.join(PROJECT, 'SMC')):
    raise SystemExit("set MRI_PROJECT to the folder that contains SMC/")
os.environ['SMC_MODE'] = 'none'                     # import without running anything
os.environ['SMC_PROJECT_PATH'] = os.path.join(PROJECT, '')
sys.path.insert(0, CODE)
import torch  # noqa: E402
import sumo_class_train as T  # noqa: E402

print('device', T.DEVICE, torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
T.DATA_OVERLAY = None                               # 'refit' runs: research data folder, no overlay
data = T.SeriesData(['BTCUSDT'], '5M', 200, T.DATA['tune_start'], 'multi_g')
win = data.windows(L)
out = {'device': np.array(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')}
for run, seed in RUNS.items():
    t0 = time.time()
    z = np.load(os.path.join(T.RUN_DIR, 'pool', run + '.npz'))
    st, sp = z['BTCUSDT__val__st'].astype(np.int64), z['BTCUSDT__val__sp'].astype(np.float64)
    cfg = T.make_cfg(MODEL, {'L': L}, seed)
    net = T.CNNBase(cfg['L'], cfg['cnn_ch'], cfg['cnn_k'], cfg['cnn_dil'], cfg['head_hidden'], cfg['dropout'], 1.0,
                    cfg['cnn_bn'], cfg['cnn_res'], cfg['cnn_cdrop'], cfg['cnn_pool'], cfg['cnn_in'], cfg['cnn_sdrop'],
                    cfg['cnn_valid'], cfg['cnn_bias'], cfg['cnn_ma'], T.SIDE_COLS[cfg['side']], 4, None, 'rw', None,
                    T.N_LEVEL[cfg['price']], False)
    net.load_state_dict({k[3:]: torch.from_numpy(z[k]) for k in z.files if k.startswith('sd_')})
    net = net.to(T.DEVICE).eval()
    for tf32 in (False, True):
        torch.backends.cudnn.allow_tf32 = tf32      # cuDNN default: True
        torch.backends.cuda.matmul.allow_tf32 = False   # PyTorch default for matmul
        res = []
        with torch.no_grad():
            for i in range(0, len(st), 32768):
                s_ = torch.from_numpy(st[i:i + 32768]).to(T.DEVICE)
                W = data.gw(s_, L)
                fl = win['floor_t'][torch.zeros(len(s_), dtype=torch.long, device=T.DEVICE)]
                X = T.to_shapes(W, fl)
                S = T.side_feats(data, s_, L, W, fl)
                res.append(T.score(net, X, S=S))
        lg = np.concatenate(res)
        key = 'tf32_on' if tf32 else 'tf32_off'
        out[f'{run}__{key}'] = lg.astype(np.float32)
        d = np.abs(lg - sp)
        print(f"{run} {key}: windows {len(st)} | vs stored: max {d.max():.3g}, p50 {np.median(d):.3g}")
    print(f"   {time.time() - t0:.0f} s")
torch.backends.cudnn.allow_tf32 = True
np.savez_compressed(OUT, **out)
print('saved', OUT, os.path.getsize(OUT), 'bytes')
