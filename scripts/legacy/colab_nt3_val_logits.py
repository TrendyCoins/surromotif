"""Colab: logits of the BTC VAL windows of the NT3 (or PB) 'refit' models, recomputed by research code with TF32 off and on.

Model: MRI_MODEL = 'NT3_PRV_ch48' (default, session 3 -> nt3_btc_val_logits.npz) or 'PB_s64h24_d1' (session 4, R3 amendment:
the gate is the TF32-off logits -> pb_btc_val_logits.npz). Runs: 'refit' on ALL a, b and 'refit' on C a of the model.

Why (rule R3, session 3): the reference (and research code on CPU, float32) reproduces the stored VAL logits of the NT3
'refit' models (calib npz, BTCUSDT__val__sp) only to max |dlogit| ~0.012 instead of 1e-4; an emulation of TF32 convolutions
(cuDNN default on Ampere GPUs) gives the same size and structure of the difference. This run settles it on the data the
models saw (SMC/data/): with TF32 off the logits must match the repo reference (R3 is then checked against them), with
TF32 on they should reproduce the stored ones. It also saves the 17 side inputs of every 10th window (research side_feats),
so that a data difference, if any, can be located.

HOW TO RUN (in Colab, not locally):
    1) Runtime -> Change runtime type -> GPU, preferably A100 or L4 (TF32 exists only on such GPUs; on a T4 or CPU the
       TF32-off logits are still valid for R3, but the stored ones cannot be reproduced);
    2) Files panel -> upload two files of this repo into /content: research/sumo_class_train.py and this script;
    3) in a cell (MRI_PROJECT = the folder on your Drive that contains SMC/, as SMC_PROJECT_PATH of the Train notebook):
           import os
           os.environ['MRI_PROJECT'] = '<the folder that contains SMC/>'
           os.environ['MRI_MODEL'] = 'PB_s64h24_d1'          # session 4; omit for NT3_PRV_ch48
           %run /content/colab_nt3_val_logits.py
       Drive is mounted by the script (a prompt may appear the first time).
    4) the result nt3_btc_val_logits.npz / pb_btc_val_logits.npz (~3 MB) is downloaded by the browser (and stays in /content):
       put it into artifacts/calib/ of the repo.
Runs a few minutes (the BTC data with its smc_bars file is loaded once).
"""
import os
import sys
import time

import numpy as np

PROJECT = os.environ.get('MRI_PROJECT', '')
CODE = os.environ.get('MRI_RESEARCH_DIR', '/content')
MODEL, L = os.environ.get('MRI_MODEL', 'NT3_PRV_ch48'), 64
if MODEL not in ('NT3_PRV_ch48', 'PB_s64h24_d1'):
    raise SystemExit(f"MRI_MODEL = '{MODEL}': 'NT3_PRV_ch48' or 'PB_s64h24_d1'")
OUT = os.environ.get('MRI_OUT', 'nt3_btc_val_logits.npz' if MODEL == 'NT3_PRV_ch48' else 'pb_btc_val_logits.npz')
RUNS = {f'pool_ALL_{MODEL}_L64_a_refit': 'a', f'pool_ALL_{MODEL}_L64_b_refit': 'b', f'pool_C_{MODEL}_L64_a_refit': 'a'}
FEATS_EVERY = 10                                    # side inputs of every 10th window (data check)

try:                                                # Colab: mount Drive here (in the notebook kernel); research/
    from google.colab import drive                  # sumo_class_train.py mounts it again on import — made a no-op
    if not os.path.isdir('/content/drive/MyDrive'):
        drive.mount('/content/drive')
    drive.mount = lambda *a, **k: None
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if not PROJECT or not os.path.isdir(os.path.join(PROJECT, 'SMC')):
    raise SystemExit(f"MRI_PROJECT = '{PROJECT}': set it to the folder that contains SMC/ (see the docstring)")
if not os.path.isfile(os.path.join(CODE, 'sumo_class_train.py')):
    raise SystemExit(f"no sumo_class_train.py in {CODE}: upload research/sumo_class_train.py of the repo there")
for run in RUNS:
    p_ = os.path.join(PROJECT, 'SMC', 'runs', 'pool', run + '.npz')
    if not os.path.isfile(p_):
        raise SystemExit(f"no {run}.npz in SMC/runs/pool/")
os.environ['SMC_MODE'] = 'none'                     # import without running anything
os.environ['SMC_PROJECT_PATH'] = os.path.join(PROJECT, '')
sys.path.insert(0, CODE)
import torch  # noqa: E402
import sumo_class_train as T  # noqa: E402

gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
print(f"device {gpu} (compute capability {cap[0]}.{cap[1]}; TF32 exists from 8.0) | torch {torch.__version__}")
T.DATA_OVERLAY = None                               # 'refit' runs: research data folder (SMC/data/), no overlay
t0 = time.time()
data = T.SeriesData(['BTCUSDT'], '5M', 200, T.DATA['tune_start'], 'multi_g')
win = data.windows(L)
print(f"BTC data loaded: {time.time() - t0:.0f} s | history inputs: {'yes' if data.nt_ok else 'NO (smc_bars missing)'}")
if not data.nt_ok:
    raise SystemExit("smc_bars_BTCUSDT_5M_N200.npz is missing in SMC/data/")
out = {'device': np.array(gpu), 'torch': np.array(torch.__version__), 'model': np.array(MODEL)}
side_cols = list(T.SIDE_COLS['nt_PRV'])
first_run = next(iter(RUNS))
for run, seed in RUNS.items():
    t0 = time.time()
    z = np.load(os.path.join(T.RUN_DIR, 'pool', run + '.npz'))
    st, sp = z['BTCUSDT__val__st'].astype(np.int64), z['BTCUSDT__val__sp'].astype(np.float64)
    # BTC is the first asset of the pool: its window positions are those of a BTC-only data set (research pool_assets)
    if st.max() + L > len(data.r) or not (np.isin(st, win['st']).all() and (data.part[st] == 1).all()):
        raise SystemExit(f"{run}: the stored BTC VAL windows are not valid VAL windows of SMC/data/ BTC — data changed?")
    cfg = T.make_cfg(MODEL, {'L': L}, seed)
    net = T.CNNBase(cfg['L'], cfg['cnn_ch'], cfg['cnn_k'], cfg['cnn_dil'], cfg['head_hidden'], cfg['dropout'], 1.0,
                    cfg['cnn_bn'], cfg['cnn_res'], cfg['cnn_cdrop'], cfg['cnn_pool'], cfg['cnn_in'], cfg['cnn_sdrop'],
                    cfg['cnn_valid'], cfg['cnn_bias'], cfg['cnn_ma'], T.SIDE_COLS[cfg['side']], 4, None, 'rw', None,
                    T.N_LEVEL[cfg['price']], False, cfg.get('cnn_sep', False))   # PB: separable layers (v6.3)
    net.load_state_dict({k[3:]: torch.from_numpy(z[k]) for k in z.files if k.startswith('sd_')}, strict=True)
    net = net.to(T.DEVICE).eval()
    feats = []
    for tf32 in (False, True):
        torch.backends.cudnn.allow_tf32 = tf32      # cuDNN convolutions; PyTorch default: True
        torch.backends.cuda.matmul.allow_tf32 = False   # matmul (head); PyTorch default: False
        res = []
        with torch.no_grad():
            for i in range(0, len(st), 32768):
                s_ = torch.from_numpy(st[i:i + 32768]).to(T.DEVICE)
                W = data.gw(s_, L)
                fl = win['floor_t'][torch.zeros(len(s_), dtype=torch.long, device=T.DEVICE)]
                X = T.to_shapes(W, fl)
                S = T.side_feats(data, s_, L, W, fl)
                res.append(T.score(net, X, S=S))
                if not tf32 and run == first_run:            # side inputs do not depend on the model
                    k = np.arange(i, i + len(s_)) % FEATS_EVERY == 0
                    feats.append(S[torch.from_numpy(k).to(S.device)][:, side_cols].float().cpu().numpy())
        lg = np.concatenate(res)
        key = 'tf32_on' if tf32 else 'tf32_off'
        out[f'{run}__{key}'] = lg.astype(np.float32)
        d = np.abs(lg - sp)
        print(f"{run} {key}: windows {len(st)} | against the stored logits: max {d.max():.3g}, median {np.median(d):.3g}")
    if feats:
        out['feats_st'] = st[::FEATS_EVERY]
        out['feats_side17'] = np.concatenate(feats)
    print(f"   TF32 on - off: max {np.abs(out[f'{run}__tf32_on'] - out[f'{run}__tf32_off']).max():.3g} "
          f"(0 means TF32 had no effect on this GPU) | {time.time() - t0:.0f} s")
torch.backends.cudnn.allow_tf32 = True
np.savez_compressed(OUT, **out)
print('saved', os.path.abspath(OUT), os.path.getsize(OUT), 'bytes -> put it into artifacts/calib/ of the repo')
if IN_COLAB:
    try:
        from google.colab import files
        files.download(OUT)
    except Exception as e:                          # the file stays in /content: download it from the Files panel
        print(f"browser download did not start ({type(e).__name__}); take the file from the Files panel")
