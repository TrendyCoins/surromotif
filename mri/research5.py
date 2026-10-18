"""The research chain of session 5 — SuMo_Class_Bell (research/sumo_class_bell.py), SuMo_Class_Live
(research/sumo_class_live.py) and SuMo_Class_Ceil (research/sumo_class_ceil.py) with the training code v6.3 — loaded as
modules for checks only (gates, reproduction of the Ceil json, R2 / R3 with the floor 0.2); the indicator never needs it.

The research files are used as they are. Their paths are module constants overridden by environment variables (BELL_*,
LIVE_*, CEIL_*); RUN_DIR is a temporary folder with links to artifacts/calib/ in the layout the research code expects:
    pool/   the pool npz parts (<tag>_partK.npz)
    nocal/  the 'nocal' json
    live/   the json of the SuMo_Class_Live 'run'
Nothing is added to the repository.
"""
import glob
import importlib.util
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESEARCH = os.path.join(ROOT, 'research')
CALIB = os.path.join(ROOT, 'artifacts', 'calib')
MODEL = 'PB_s64h24_d1'
LIVE_JSON = 'live_20261018_093025.json'
CEIL_JSON = 'ceil_20261018_115426.json'
H = ('FILUSDT', 'LINKUSDT', 'XLMUSDT')

_CHAIN = {}


def run_dir(base=None):
    """A temporary RUN_DIR with links to the pool npz parts, the 'nocal' json and the Live json (trailing slash)."""
    d = tempfile.mkdtemp(prefix='mri_research5_', dir=base)
    for sub, pattern in (('pool', f'pool_*_{MODEL}_*_part*.npz'), ('nocal', f'nocal_{MODEL}_*.json'),
                         ('live', LIVE_JSON)):
        os.makedirs(os.path.join(d, sub))
        for p in sorted(glob.glob(os.path.join(CALIB, pattern))):
            os.symlink(p, os.path.join(d, sub, os.path.basename(p)))
    return d + '/'


def env(rd, mode='none'):
    """Environment of the three research files and of the training code for a RUN_DIR rd."""
    e = {'SMC_MODE': 'none', 'SMC_PROJECT_PATH': rd + 'project/'}
    for pre, name in (('BELL', 'bell'), ('LIVE', 'live'), ('CEIL', 'ceil')):
        e.update({f'{pre}_MODE': mode, f'{pre}_TRAIN_PY': os.path.join(RESEARCH, 'sumo_class_train.py'),
                  f'{pre}_RUN_DIR': rd, f'{pre}_OUT_DIR': rd + f'out_{name}/'})
    e.update({'LIVE_BELL_PY': os.path.join(RESEARCH, 'sumo_class_bell.py'),
              'CEIL_BELL_PY': os.path.join(RESEARCH, 'sumo_class_bell.py'),
              'CEIL_LIVE_PY': os.path.join(RESEARCH, 'sumo_class_live.py')})
    return e


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def chain():
    """(T, BB, LV, CE): training code, Bell, Live, Ceil — loaded once per process by Ceil load_modules (Live from LIVE_PY,
       Bell from BELL_PY, the training code through Bell), with a temporary RUN_DIR."""
    if not _CHAIN:
        rd = run_dir()
        os.makedirs(rd + 'project/', exist_ok=True)
        os.environ.update(env(rd))
        ce = import_file(os.path.join(RESEARCH, 'sumo_class_ceil.py'), 'sumo_class_ceil')
        T = ce.load_modules()
        _CHAIN.update({'T': T, 'BB': ce.BB, 'LV': ce.LV, 'CE': ce, 'run_dir': rd})
    return _CHAIN['T'], _CHAIN['BB'], _CHAIN['LV'], _CHAIN['CE']
