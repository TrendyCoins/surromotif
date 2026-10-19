"""Colab: floors of the shape divisor from the smc_series files of BOTH data folders, separately.

Why: artifacts/pack/floors.json has one value per file name, and smc_series_BTCUSDT / ETHUSDT exist in both
SMC/data/ (research) and SMC/data_fresh/ (prod training of BTC / ETH). The end-to-end check showed that the BTC value
there is the research one (SMC/data/); the prod model used the fresh folder for BTC and ETH.

Run in Colab after mounting Drive; set PROJECT to the project folder on your Drive (the one that holds SMC/).
Output: one small JSON next to this run — put it into artifacts/pack/ (floors_by_folder.json).
"""
import glob
import json
import os
import zlib

import numpy as np

PROJECT = os.environ.get('MRI_PROJECT', '')          # e.g. the Drive folder that contains SMC/ — set before running
FOLDERS = ('SMC/data/', 'SMC/data_fresh/')
OUT = 'floors_by_folder.json'

if not PROJECT or not os.path.isdir(os.path.join(PROJECT, 'SMC')):
    raise SystemExit("set MRI_PROJECT (or PROJECT) to the folder that contains SMC/")

res = {}
for folder in FOLDERS:
    rows = {}
    for path in sorted(glob.glob(os.path.join(PROJECT, folder, 'smc_series_*_5M_N200.npz'))):
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d['meta']))
        asset = os.path.basename(path)[len('smc_series_'):-len('_5M_N200.npz')]
        rows[asset] = {
            'floor_L64': float(d['floor_L64']), 'floor_L32': float(d['floor_L32']) if 'floor_L32' in d.files else None,
            'training_start': meta.get('training_start'), 'csv_last': meta.get('csv_last'),
            'n_tr': int(len(d['r_tr'])), 'n_va': int(len(d['r_va'])),
            'crc_tr': zlib.crc32(d['r_tr'].astype(np.float32).tobytes()),
            'crc_va': zlib.crc32(d['r_va'].astype(np.float32).tobytes())}
    res[folder] = rows
    print(folder, len(rows), 'files', {a: round(v['floor_L64'], 6) for a, v in rows.items() if a in ('BTCUSDT', 'ETHUSDT')})

with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=1)
print('saved', OUT, os.path.getsize(OUT), 'bytes')
