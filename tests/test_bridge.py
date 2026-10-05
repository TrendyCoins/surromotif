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
