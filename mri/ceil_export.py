"""Reader of the Colab export of the SuMo_Class_Ceil cache (scripts/colab_export_ceil_f02.py): window scores of VAL with the
divisor floor 0.2 (rule R17) — 'refit' on ALL a / b and prod a of PB_s64h24_d1. For checks only (gate (д), R3 with the floor
0.2, the reproduction of the Ceil json, the ruler of the export); the indicator never needs it.

Files in artifacts/calib/: ceil_f02_partK.npz (keys '<tag>__<ASSET>__val__<field>', fields sp, sn, med, mabs float32, wk
int32) and ceil_f02_manifest.json (parts with size, sha256 and keys; meta of each tag).
"""
import hashlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, 'artifacts', 'calib')
MANIFEST = 'ceil_f02_manifest.json'
MODEL = 'PB_s64h24_d1'
TAGS = {'refit_a': f'pool_ALL_{MODEL}_L64_a_refit', 'refit_b': f'pool_ALL_{MODEL}_L64_b_refit',
        'prod_a': f'prod_{MODEL}_L64_a'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def manifest(d=DIR):
    with open(os.path.join(d, MANIFEST), encoding='utf-8') as fh:
        return json.load(fh)


def load_export(d=DIR, verify=True):
    """{tag: {asset: {field: array}}} as exported (float32 / int32); verify — size and sha256 of every part by the manifest."""
    man = manifest(d)
    out = {}
    for part in man['parts']:
        p = os.path.join(d, part['file'])
        if verify and (os.path.getsize(p) != part['size'] or sha256(p) != part['sha256']):
            raise ValueError(f"{part['file']}: size / sha256 differ from {MANIFEST}")
        with np.load(p) as z:
            if sorted(z.files) != sorted(part['keys']):
                raise ValueError(f"{part['file']}: keys differ from {MANIFEST}")
            for k in z.files:
                tag, asset, part_, field = k.split('__')
                if part_ != 'val':
                    raise ValueError(f'{k}: only VAL is exported')
                out.setdefault(tag, {}).setdefault(asset, {})[field] = z[k]
    return out
