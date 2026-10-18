"""The Colab export of the SuMo_Class_Ceil cache (scripts/colab_export_ceil_f02.py) on a synthetic cache written by the
research save_cache itself: the exported arrays = the cache bit for bit, parts within the size limit, the manifest checks,
mri.ceil_export reads it back; a missing cache file or another floor stops with a message."""
import importlib.util
import os

import numpy as np
import pytest

from mri import ceil_export, research5

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def colab_module():
    spec = importlib.util.spec_from_file_location('colab_export_ceil_f02',
                                                  os.path.join(ROOT, 'scripts', 'colab_export_ceil_f02.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fake_cache(CE, project, assets, floor_meta=0.2, skip=None):
    """Cache files of the three tags in project/SMC/runs/ceil/cache/ by research save_cache (with 'old__' keys too)."""
    rng = np.random.default_rng(3)
    keep = CE.OUT_DIR
    CE.OUT_DIR = os.path.join(project, 'SMC', 'runs', 'ceil') + '/'
    floors = {a: 0.2 if i % 2 else 0.17 + 0.001 * i for i, a in enumerate(assets)}
    truth = {}
    try:
        for tag in ceil_export.TAGS.values():
            if tag == skip:
                continue
            res, old = {}, {}
            for i, a in enumerate(assets):
                n = 40 + i
                d = {'sp': rng.standard_normal(n), 'sn': rng.standard_normal(n), 'med': rng.random(n).astype(np.float32),
                     'mabs': rng.random(n).astype(np.float32), 'wk': np.sort(rng.integers(2900, 2960, n))}
                res[(a, 'val')] = d
                old[(a, 'val')] = {k: v + 1 if k in ('sp', 'sn') else v for k, v in d.items()}
            CE.save_cache({'tag': tag}, res, old, {'tag': tag, 't_cut': '2026-09-26 09:15:00',
                                                   'data_overlay': '/mnt/train_machine/data_fresh/'}, floors)
            truth[tag] = res
    finally:
        CE.OUT_DIR = keep
    if floor_meta != 0.2:                                  # rewrite one meta with another floor
        tag = ceil_export.TAGS['prod_a']
        p = os.path.join(project, 'SMC', 'runs', 'ceil', 'cache', f'ceil_{tag}_f0.2.npz')
        with np.load(p) as z:
            arr = {k: z[k] for k in z.files}
        import json
        meta = json.loads(str(arr['meta']))
        meta['floor'] = floor_meta
        arr['meta'] = np.array(json.dumps(meta))
        np.savez(p, **arr)
    return truth, floors


def test_export_roundtrip(tmp_path, monkeypatch):
    T, BB, LV, CE = research5.chain()
    m = colab_module()
    truth, floors = fake_cache(CE, str(tmp_path / 'proj'), m.ASSETS)
    os.makedirs(tmp_path / 'proj' / 'SMC', exist_ok=True)
    monkeypatch.setattr(m, 'PROJECT', str(tmp_path / 'proj'))
    monkeypatch.setattr(m, 'OUT_DIR', str(tmp_path / 'out'))
    monkeypatch.setattr(m, 'PART_MAX', 20_000)              # force several parts
    m.main()
    man = ceil_export.manifest(str(tmp_path / 'out'))
    assert len(man['parts']) > 1 and all(p['size'] <= 20_000 for p in man['parts'])
    assert 'train_machine' not in open(tmp_path / 'out' / ceil_export.MANIFEST, encoding='utf-8').read()
    ex = ceil_export.load_export(str(tmp_path / 'out'))
    for tag, res in truth.items():
        fields, extra = m.WHAT[tag]
        meta = man['meta'][tag]
        assert meta['tag'] == tag and meta['floor'] == 0.2 and meta['ver'] == 1
        assert meta['floors_trained'] == {a: float(floors[a]) for a in m.ASSETS}
        assert meta['run']['data_overlay_folder'] == 'data_fresh'
        for a in m.ASSETS:
            want = fields + (extra if a == 'BTCUSDT' else ())
            assert sorted(ex[tag][a]) == sorted(want)
            for f in want:
                x = ex[tag][a][f]
                ref = res[(a, 'val')][f]
                ref = ref.astype(np.float32) if f in ('sp', 'sn', 'med', 'mabs') else ref.astype(np.int32)
                assert x.dtype == ref.dtype and np.array_equal(x, ref)       # bit for bit with the cache


def test_export_stops(tmp_path, monkeypatch):
    T, BB, LV, CE = research5.chain()
    m = colab_module()
    os.makedirs(tmp_path / 'p1' / 'SMC', exist_ok=True)
    fake_cache(CE, str(tmp_path / 'p1'), m.ASSETS, skip=ceil_export.TAGS['refit_b'])
    monkeypatch.setattr(m, 'PROJECT', str(tmp_path / 'p1'))
    monkeypatch.setattr(m, 'OUT_DIR', str(tmp_path / 'o1'))
    with pytest.raises(SystemExit, match='no cache file'):
        m.main()
    os.makedirs(tmp_path / 'p2' / 'SMC', exist_ok=True)
    fake_cache(CE, str(tmp_path / 'p2'), m.ASSETS, floor_meta=0.25)
    monkeypatch.setattr(m, 'PROJECT', str(tmp_path / 'p2'))
    with pytest.raises(SystemExit, match='floor 0.25'):
        m.main()
