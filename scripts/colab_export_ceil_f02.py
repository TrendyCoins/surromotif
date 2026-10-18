"""Colab: export of the window scores with the floor 0.2 from the SuMo_Class_Ceil cache (session 5, docs/rules.md R17 / R19).

SuMo_Class_Ceil v1 ('run' 2026-10-18) rescored the windows with the divisor floor 0.2 (TF32 off, deterministic cuDNN) and
kept the scores in its cache SMC/runs/ceil/cache/ceil_<tag>_f0.2.npz (research/sumo_class_ceil.py save_cache: meta — json
{tag, floor, ver, run, floors_trained}; keys '<ASSET>__<part>__{sp,sn,med,mabs,wk}' — scores with the floor 0.2; 'old__…' —
the rescoring with the old floor, not exported). The repo needs these scores for R3 with the floor 0.2, the reproduction of
the Ceil json and the ruler of the indicator; this script only copies them out, nothing is recomputed.

What is exported (only VAL; arrays exactly as in the cache: float32, wk int32):
    'refit' on ALL a and b (pool_ALL_PB_s64h24_d1_L64_{a,b}_refit): every one of the 31 assets — sp, sn, wk; BTC also med, mabs
    prod a (prod_PB_s64h24_d1_L64_a): every one of the 31 assets — sn, wk (the ruler and the window check); BTC also sp, med, mabs
Keys '<tag>__<ASSET>__val__<field>'. Output: ceil_f02_partK.npz (K = 0, 1, …; each part <= 20 MB, cut by keys — GitHub takes
files up to 25 MB through the browser) and ceil_f02_manifest.json (parts, size, sha256, keys, meta of each tag without Drive
paths). The prefix ceil_f02_ keeps them apart from the pool npz parts (<tag>_partK.npz, the glob of Bell).

The cache must be complete: a missing file, another floor / version in meta, a missing asset or field — stop with a message.
Rebuilding it means a new 'run' of SuMo_Class_Ceil on a GPU (the cache builds itself).

HOW TO RUN (in Colab, CPU is enough, ~1 min):
    1) Files panel -> upload this script into /content;
    2) in a cell (MRI_PROJECT = the folder on your Drive that contains SMC/, as for the other colab scripts):
           import os
           os.environ['MRI_PROJECT'] = '<the folder that contains SMC/>'
           %run /content/colab_export_ceil_f02.py
       Drive is mounted by the script (a prompt may appear the first time).
    3) the files are written to /content/ceil_f02/ (MRI_OUT_DIR) and offered for download by the browser: put all of them
       (ceil_f02_part*.npz and ceil_f02_manifest.json) into artifacts/calib/ of the repo.
"""
import hashlib
import json
import os
import sys

import numpy as np

PROJECT = os.environ.get('MRI_PROJECT', '')
OUT_DIR = os.environ.get('MRI_OUT_DIR', '/content/ceil_f02')
MODEL = 'PB_s64h24_d1'
FLOOR, CEIL_VER = 0.2, 1
PART_MAX = 20_000_000                     # bytes per part (file size checked after writing)
ASSETS = ['BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'SOLUSDT', 'ADAUSDT', 'BNBUSDT', 'BCHUSDT', 'DOTUSDT', 'UNIUSDT', 'XRPUSDT',
          'LTCUSDT', 'DASHUSDT', 'ZECUSDT', 'SPELLUSDT', 'ICPUSDT', 'NEARUSDT', 'ZENUSDT', 'AVAXUSDT', 'FETUSDT', 'RAYUSDT',
          'CAKEUSDT', 'INJUSDT', 'RUNEUSDT', 'PHAUSDT', 'QNTUSDT', 'HBARUSDT', 'AAVEUSDT', 'VTHOUSDT', 'FILUSDT', 'LINKUSDT',
          'XLMUSDT']
REF = 'BTCUSDT'
WHAT = {   # tag -> (fields of every asset, extra fields of BTC)
    f'pool_ALL_{MODEL}_L64_a_refit': (('sp', 'sn', 'wk'), ('med', 'mabs')),
    f'pool_ALL_{MODEL}_L64_b_refit': (('sp', 'sn', 'wk'), ('med', 'mabs')),
    f'prod_{MODEL}_L64_a': (('sn', 'wk'), ('sp', 'med', 'mabs')),
}
DTYPE = {'sp': np.float32, 'sn': np.float32, 'med': np.float32, 'mabs': np.float32, 'wk': np.int32}


def stop(msg):
    raise SystemExit(f"STOP: {msg}")


def folder(p):
    """A Drive path reduced to its last folder name (no Drive paths in the output)."""
    return None if not p else os.path.basename(str(p).rstrip('/'))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_tag(tag):
    path = os.path.join(PROJECT, 'SMC', 'runs', 'ceil', 'cache', f'ceil_{tag}_f{FLOOR:g}.npz')
    if not os.path.isfile(path):
        stop(f"no cache file SMC/runs/ceil/cache/{os.path.basename(path)} — the cache of SuMo_Class_Ceil is incomplete; "
             "rebuild it with a new 'run' of SuMo_Class_Ceil on a GPU (the cache builds itself), then run this again")
    out = {}
    with np.load(path) as d:
        meta = json.loads(str(d['meta']))
        if meta.get('tag') != tag or meta.get('floor') != FLOOR or meta.get('ver') != CEIL_VER:
            stop(f"{os.path.basename(path)}: meta tag {meta.get('tag')}, floor {meta.get('floor')}, ver {meta.get('ver')} — "
                 f"expected {tag}, {FLOOR}, {CEIL_VER}")
        fields, extra = WHAT[tag]
        for a in ASSETS:
            need = fields + (extra if a == REF else ())
            n = None
            for f in need:
                k = f'{a}__val__{f}'
                if k not in d.files:
                    stop(f"{os.path.basename(path)}: no key {k}")
                x = d[k]
                if x.dtype != DTYPE[f]:
                    stop(f"{os.path.basename(path)}: {k} is {x.dtype}, expected {np.dtype(DTYPE[f])}")
                if n is not None and len(x) != n:
                    stop(f"{os.path.basename(path)}: {a} — fields of different length")
                n = len(x)
                out[f'{tag}__{a}__val__{f}'] = x
            if (np.diff(out[f'{tag}__{a}__val__wk'].astype(np.int64)) < 0).any():
                stop(f"{os.path.basename(path)}: {a} — weeks of the windows decrease (order of windows unknown)")
    run = meta.get('run') or {}
    fl = meta.get('floors_trained') or {}
    if set(fl) != set(ASSETS):
        stop(f"{os.path.basename(path)}: floors_trained has {len(fl)} assets, expected the 31")
    m = {'tag': meta['tag'], 'floor': meta['floor'], 'ver': meta['ver'],
         'run': {'tag': run.get('tag'), 't_cut': run.get('t_cut'), 'data_overlay_folder': folder(run.get('data_overlay'))},
         'floors_trained': {a: float(fl[a]) for a in ASSETS},
         'n_windows': {a: int(len(out[f'{tag}__{a}__val__wk'])) for a in ASSETS}}
    print(f"{tag}: {len(ASSETS)} assets, {sum(m['n_windows'].values())} VAL windows, keys {len(out)}")
    return out, m


def main():
    try:
        from google.colab import drive
        if not os.path.isdir('/content/drive/MyDrive'):
            drive.mount('/content/drive')
        in_colab = True
    except ImportError:
        in_colab = False
    if not PROJECT or not os.path.isdir(os.path.join(PROJECT, 'SMC')):
        stop(f"MRI_PROJECT = '{PROJECT}': set it to the folder that contains SMC/ (see the docstring)")
    arrays, meta = {}, {}
    for tag in WHAT:
        a, m = read_tag(tag)
        arrays.update(a)
        meta[tag] = m
    os.makedirs(OUT_DIR, exist_ok=True)
    parts, cur, size = [], [], 0
    for k, x in arrays.items():                   # cut by keys: raw bytes + 1 KB of zip headers per key <= 0.97 of the limit
        if cur and size + x.nbytes + 1024 > 0.97 * PART_MAX:
            parts.append(cur)
            cur, size = [], 0
        cur.append(k)
        size += x.nbytes + 1024
    parts.append(cur)
    man = {'what': 'window scores with the floor 0.2 from the SuMo_Class_Ceil v1 cache (VAL)', 'model': MODEL,
           'floor': FLOOR, 'ceil_ver': CEIL_VER, 'meta': meta, 'parts': []}
    files = []
    for i, keys in enumerate(parts):
        name = f'ceil_f02_part{i}.npz'
        path = os.path.join(OUT_DIR, name)
        np.savez_compressed(path, **{k: arrays[k] for k in keys})
        sz = os.path.getsize(path)
        if sz > PART_MAX:
            stop(f"{name}: {sz} bytes > {PART_MAX}")
        man['parts'].append({'file': name, 'size': sz, 'sha256': sha256(path), 'keys': keys})
        files.append(path)
        print(f"{name}: {len(keys)} keys, {sz / 1e6:.1f} MB")
    mp = os.path.join(OUT_DIR, 'ceil_f02_manifest.json')
    with open(mp, 'w', encoding='utf-8') as fh:
        json.dump(man, fh, ensure_ascii=False, indent=1)
    files.append(mp)
    print(f"done: {len(parts)} parts + manifest in {OUT_DIR} — put them all into artifacts/calib/ of the repo")
    if in_colab:
        from google.colab import files as colab_files
        for p in files:
            colab_files.download(p)


if __name__ == '__main__' or 'ipykernel' in sys.modules:
    main()
