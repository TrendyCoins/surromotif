"""The research side of the F1 checks (rules R22, R25; by timeframe — R29): the training code v6.6
(research/sumo_class_train.py) imported as a module, the prod a model loaded by its own load_pool_model (build_model + the
weights of the slim npz), SeriesData 'multi_g' over research files of a timeframe (1H, 5M, 15M, 30M), to_shapes /
side_feats inside floor_override 0.2 (research/sumo_class_ceil.py). CPU, float32. Needs torch; f1h does not.

Layout of the temporary project folder (SMC_PROJECT_PATH; nothing is written to the repository):
    SMC/data/        research files (links to artifacts/bars/ or files written by build_tf_files)
    SMC/runs/prod/   prod_SP_l025_d1_L64_a.json and .npz (links to artifacts/weights/: json and slim npz)
"""
import glob
import importlib.util
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESEARCH = os.path.join(ROOT, 'research')
PROD = os.path.join(ROOT, 'artifacts', 'weights', 'prod_SP_l025_d1_L64_a')
FLOOR = float(np.float32(0.2))
L = 64
_MOD = {}


def project(data_files=()):
    """Temporary project folder with the prod a model and links to data_files in SMC/data/. Returns its path (with /)."""
    d = tempfile.mkdtemp(prefix='f1h_research_') + '/'
    os.makedirs(d + 'SMC/data/')
    os.makedirs(d + 'SMC/runs/prod/')
    os.symlink(PROD + '.json', d + 'SMC/runs/prod/prod_SP_l025_d1_L64_a.json')
    os.symlink(PROD + '_slim.npz', d + 'SMC/runs/prod/prod_SP_l025_d1_L64_a.npz')
    for p in data_files:
        os.symlink(os.path.abspath(p), d + 'SMC/data/' + os.path.basename(p))
    return d


def _import(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def modules(proj):
    """(training code, Ceil, TF) imported once per process; the training code points at proj."""
    if not _MOD:
        os.environ.update({'SMC_MODE': 'none', 'SMC_PROJECT_PATH': proj, 'CEIL_MODE': 'none', 'SMCTF_MODE': 'none'})
        T = _import(os.path.join(RESEARCH, 'sumo_class_train.py'), 'sumo_class_train')
        _MOD.update({'T': T, 'CL': _import(os.path.join(RESEARCH, 'sumo_class_ceil.py'), 'sumo_class_ceil'),
                     'TF': _import(os.path.join(RESEARCH, 'sumo_class_tf.py'), 'sumo_class_tf')})
    T = _MOD['T']
    T.project_path = proj
    T.DATA_DIR, T.DATA_OVERLAY, T.RUN_DIR = proj + 'SMC/data/', None, proj + 'SMC/runs/'
    return T, _MOD['CL'], _MOD['TF']


def prod_model(T):
    """prod a by research load_pool_model (mode 'prod'): build_model with the cfg of the json, weights sd_*, eval."""
    model, run = T.load_pool_model('ALL', 'a', 'SP_l025_d1', mode='prod')
    return model, run


def logits(T, CL, model, data, bs=4096):
    """Research logit of every window of the SeriesData (with the floor 0.2) and per-window info: start st, part (0 / 1),
       median and mean |r| of the window (float32, before the floor), orientation of to_shapes and of side_feats."""
    import torch
    w = data.windows(L)
    st_all = w['st']
    out = {'st': st_all, 'part': data.part[st_all].astype(np.int64), 'logit': [], 'med': [], 'mabs': [], 'flip': [],
           'flip_side': []}
    with CL.floor_override(data, L, FLOOR) as wf, torch.no_grad():
        for i in range(0, len(st_all), bs):
            st = torch.from_numpy(st_all[i:i + bs])
            W = data.gw(st, L)
            fl = wf['floor_t'][torch.zeros(len(st), dtype=torch.long)]
            X = T.to_shapes(W, fl)
            S = T.side_feats(data, st, L, W, fl)
            out['logit'].append(T.score(model, X, S=S))
            div = torch.maximum(T.median_abs(W[:, 0]), fl)
            Xc = torch.cumsum(W[:, :3], dim=2)
            Xc = Xc - Xc.mean(dim=2, keepdim=True)
            out['flip'].append((Xc[:, 0, -1] < -T.ORIENT_EPS * div).numpy())
            cs = torch.cumsum(W[:, 0], dim=1)
            out['flip_side'].append((cs[:, -1] - cs.mean(dim=1) < -T.ORIENT_EPS * div).numpy())
            out['med'].append(T.median_abs(W[:, 0]).float().numpy())
            out['mabs'].append(W[:, 0].abs().mean(dim=1).float().numpy())
    return {k: (np.concatenate(v) if isinstance(v, list) else v) for k, v in out.items()}


def series_data(T, asset, tune_start='2100-01-01', tf='1H'):
    """SeriesData of one asset on tf (three prices + range + history in bars), as SuMo_Class_TF series_data."""
    return T.SeriesData([asset], tf, 200, tune_start, 'multi_g')


def bars_files(asset='BTCUSDT', tf='1H', folder=os.path.join(ROOT, 'artifacts', 'bars')):
    """Research files of an asset on tf in folder (5M: artifacts/bars/btc5m_compact/)."""
    return sorted(glob.glob(os.path.join(folder, f'smc_*_{asset}_{tf}_N200.npz')))
