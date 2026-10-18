"""mri.research_bridge.make_parts is a copy of research/sumo_class_data.py make_parts (that file runs main() on import):
the original is taken from the source with ast (only that function) and compared bit for bit."""
import ast
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth  # noqa: E402
from mri.research_bridge import make_parts  # noqa: E402


def original(name, path):
    tree = ast.parse(open(path, encoding='utf-8').read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {'np': np, 'pd': pd}
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), ns)
    return ns[name]


def test_make_parts_copy_matches_original():
    orig = original('make_parts', os.path.join(ROOT, 'research', 'sumo_class_data.py'))
    b = synth.bars(5000, seed=11, gaps=[(1200, 2), (3100, 7)])
    df = pd.DataFrame({'timestamp': pd.to_datetime(b['time'], unit='ms'), 'open': b['open'], 'high': b['high'],
                       'low': b['low'], 'close': b['close']})
    t_valid = str(df['timestamp'].iloc[3000])
    a, sa = orig(df, 200, str(df['timestamp'].iloc[0]), t_valid)
    c, sc = make_parts(df, 200, str(df['timestamp'].iloc[0]), t_valid)
    assert sa == sc
    for name in ('TRAINING', 'VALIDATION'):
        ra, rc = a[name]['r'], c[name]['r']
        assert np.array_equal(np.isnan(ra), np.isnan(rc)) and np.array_equal(ra[~np.isnan(ra)], rc[~np.isnan(rc)])
        assert (a[name]['i0'], a[name]['i1'], a[name]['n_gap']) == (c[name]['i0'], c[name]['i1'], c[name]['n_gap'])


def test_bars_copy_matches_original_script(tmp_path):
    """research/sumo_class_bars.py runs as is (subprocess, env SMC_PROJECT_PATH / SMC_DATA_DIR / SMC_ASSETS) on a synthetic
       CSV with smc_series written beforehand; mri.research_bridge.bars_arrays must give the same arrays bit for bit and the
       same meta except 'created'."""
    import json
    import subprocess
    from mri.research_bridge import bars_arrays, write_research_files
    b = synth.bars(3000, seed=12, gaps=[(1700, 3)])
    proj, data = tmp_path / 'proj', tmp_path / 'data'
    (proj / 'Raw Datasets').mkdir(parents=True)
    data.mkdir()
    df = pd.DataFrame({'timestamp': pd.to_datetime(b['time'], unit='ms'), 'open': b['open'], 'high': b['high'],
                       'low': b['low'], 'close': b['close'], 'volume': 1.0})
    csv = proj / 'Raw Datasets' / 'SYNUSDT-5M.csv'
    df.to_csv(csv, index=False)
    back = pd.read_csv(csv, parse_dates=['timestamp'])
    bars = pd.DataFrame({'time': back['timestamp'].to_numpy().astype('datetime64[ms]').astype(np.int64),
                         'open': back['open'], 'high': back['high'], 'low': back['low'], 'close': back['close']})
    write_research_files(bars, str(data), 'SYNUSDT', str(back['timestamp'].iloc[0]), str(back['timestamp'].iloc[2200]), 0.2)
    env = dict(os.environ, SMC_PROJECT_PATH=str(proj) + '/', SMC_DATA_DIR=str(data) + '/', SMC_ASSETS='SYNUSDT')
    run = subprocess.run([sys.executable, os.path.join(ROOT, 'research', 'sumo_class_bars.py')], env=env,
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0 and 'ГОТОВО: 1 из 1' in run.stdout, run.stdout + run.stderr
    with np.load(data / 'smc_bars_SYNUSDT_5M_N200.npz') as o, np.load(data / 'smc_series_SYNUSDT_5M_N200.npz') as d:
        orig = {k: o[k] for k in o.files}
        copy = bars_arrays(back, d, 'SYNUSDT', 'smc_series_SYNUSDT_5M_N200.npz')
    assert set(orig) == set(copy)
    for k in orig:
        if k == 'meta':
            mo, mc = json.loads(str(orig[k])), json.loads(str(copy[k]))
            mo.pop('created'), mc.pop('created')
            assert mo == mc
        else:
            assert orig[k].dtype == copy[k].dtype and np.array_equal(orig[k], copy[k]), k


def test_floor_override(tmp_path):
    """Rule R17 on the research side: inside the block every series has the floor 0.2 (float32, also the torch tensor),
       afterwards the same objects as before; on a real research SeriesData built from synthetic bars and on the stand
       object of research Ceil."""
    import tempfile
    from mri.research_bridge import FLOOR, _ceil_module, floor_override, write_research_files
    assert FLOOR == 0.20000000298023224
    ce = _ceil_module()
    dd = ce._DummyData([0.1756, 0.2, 0.2025])
    w0 = dd.windows(64)
    old, old_t = w0['floor'], w0['floor_t']
    with floor_override(dd, 64) as w:
        assert np.all(w['floor'] == np.float32(0.2)) and np.all(w['floor_t'].numpy() == np.float32(0.2))
        assert w['floor'].dtype == np.float32
    assert dd.windows(64)['floor'] is old and dd.windows(64)['floor_t'] is old_t
    assert np.array_equal(old, np.float32([0.1756, 0.2, 0.2025]))
    # a real research data set
    os.environ.setdefault('SMC_MODE', 'none')
    os.environ.setdefault('SMC_PROJECT_PATH', tempfile.mkdtemp() + '/')
    sys.path.insert(0, os.path.join(ROOT, 'research'))
    import sumo_class_train as T
    b = synth.bars(1500, seed=21)
    t0 = str(np.datetime64(int(b['time'].iloc[0]), 'ms'))
    t1 = str(np.datetime64(int(b['time'].iloc[1200]), 'ms'))
    write_research_files(b, str(tmp_path) + '/', 'BTCUSDT', t0, t1, 0.1753023)
    T.DATA_DIR, T.DATA_OVERLAY = str(tmp_path) + '/', None
    data = T.SeriesData(['BTCUSDT'], '5M', 200, '2100-01-01', 'multi_g')
    w = data.windows(64)
    old, old_t = w['floor'], w['floor_t']
    assert np.all(old == np.float32(0.1753023))
    with floor_override(data, 64) as w2:
        assert np.all(data.windows(64)['floor'] == np.float32(FLOOR)) and np.all(w2['floor_t'].numpy() == np.float32(FLOOR))
    assert data.windows(64)['floor'] is old and data.windows(64)['floor_t'] is old_t
