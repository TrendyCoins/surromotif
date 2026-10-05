"""Weights, buffers and calibration of the F3_g model (streamable CNN, research CNNBase).

Layout (research names in brackets):
    conv[0..3]  (blocks.i.0.weight)  shape (32, c_in, 3), c_in = 4 for the first layer
                (channels: close, HL/2, HLC/3 steps; bar range level), dilations 1, 2, 4, 8, no bias
    head0_w, head0_b (head.0)        (64, 112), (64,)   input: [mean 32, max 32, last 32, side 16]
    head3_w, head3_b (head.3)        (1, 64), (1,)
    s_sd, lvl_mu, lvl_sd             input scales of path steps and of the range level (lvl_mu = 0)
    side_mu, side_sd                 standardization of the 16 side inputs ('v7tcp')
"""
import glob
import json
from dataclasses import dataclass, field

import numpy as np

DILATIONS = (1, 2, 4, 8)
SIDE_COLS = (1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)   # research SIDE_COLS['v7tcp']
SIDE_NAMES = ('vol_to_7d', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'move_1d', 'move_3d', 'move_7d',
              'path_128_112', 'path_112_96', 'path_96_80', 'path_80_64', 'path_64_48', 'path_48_32',
              'path_32_16', 'path_16_0')


@dataclass
class Params:
    conv: list                      # 4 arrays (32, c_in, 3), float64
    head0_w: np.ndarray
    head0_b: np.ndarray
    head3_w: np.ndarray
    head3_b: np.ndarray
    s_sd: float
    lvl_mu: float
    lvl_sd: float
    side_mu: np.ndarray
    side_sd: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_params(self):
        return int(sum(w.size for w in self.conv) + self.head0_w.size + self.head0_b.size
                   + self.head3_w.size + self.head3_b.size)

    def to_dict(self):
        return {'conv': [w.tolist() for w in self.conv], 'dilations': list(DILATIONS),
                'head0_w': self.head0_w.tolist(), 'head0_b': self.head0_b.tolist(),
                'head3_w': self.head3_w.tolist(), 'head3_b': self.head3_b.tolist(),
                's_sd': self.s_sd, 'lvl_mu': self.lvl_mu, 'lvl_sd': self.lvl_sd,
                'side_mu': self.side_mu.tolist(), 'side_sd': self.side_sd.tolist(),
                'side_names': list(SIDE_NAMES), 'side_cols_research': list(SIDE_COLS)}


def _f64(x):
    # float32 values of research -> exact float64 (no rounding)
    return np.asarray(x, dtype=np.float32).astype(np.float64)


def from_state(sd, meta=None):
    """sd: dict research state_dict names WITHOUT the 'sd_' prefix -> Params."""
    lvl_mu = float(_f64(sd['lvl_mu'])[0])
    if lvl_mu != 0.0:
        raise ValueError(f"lvl_mu = {lvl_mu}: streaming needs a level without shift (research v5.7, streamable)")
    return Params(conv=[_f64(sd[f'blocks.{i}.0.weight']) for i in range(4)],
                  head0_w=_f64(sd['head.0.weight']), head0_b=_f64(sd['head.0.bias']),
                  head3_w=_f64(sd['head.3.weight']), head3_b=_f64(sd['head.3.bias']),
                  s_sd=float(_f64(sd['s_sd'])), lvl_mu=lvl_mu, lvl_sd=float(_f64(sd['lvl_sd'])[0]),
                  side_mu=_f64(sd['side_mu']), side_sd=_f64(sd['side_sd']), meta=dict(meta or {}))


def load_npz(path_or_glob):
    """Research npz (whole or split into parts: '..._part*.npz') -> (Params, other arrays dict)."""
    paths = sorted(glob.glob(path_or_glob)) if any(c in path_or_glob for c in '*?[') else [path_or_glob]
    if not paths:
        raise FileNotFoundError(path_or_glob)
    arrs = {}
    for p in paths:
        with np.load(p) as d:
            arrs.update({k: d[k] for k in d.files})
    sd = {k[3:]: v for k, v in arrs.items() if k.startswith('sd_')}
    rest = {k: v for k, v in arrs.items() if not k.startswith('sd_')}
    return from_state(sd, {'source': [p.split('/')[-1] for p in paths]}), rest


def from_dict(d):
    return Params(conv=[np.asarray(w, dtype=np.float64) for w in d['conv']],
                  head0_w=np.asarray(d['head0_w'], dtype=np.float64),
                  head0_b=np.asarray(d['head0_b'], dtype=np.float64),
                  head3_w=np.asarray(d['head3_w'], dtype=np.float64),
                  head3_b=np.asarray(d['head3_b'], dtype=np.float64),
                  s_sd=float(d['s_sd']), lvl_mu=float(d['lvl_mu']), lvl_sd=float(d['lvl_sd']),
                  side_mu=np.asarray(d['side_mu'], dtype=np.float64),
                  side_sd=np.asarray(d['side_sd'], dtype=np.float64))


def load_export(path):
    """Export JSON (mri/export.py) -> (Params, full export dict)."""
    with open(path, encoding='utf-8') as fh:
        ex = json.load(fh)
    return from_dict(ex['model']), ex


def random_params(seed=0, scale=0.3):
    """Random weights of the same shapes and scales (Pine limits prototype, tests)."""
    g = np.random.default_rng(seed)
    conv = [g.normal(0, scale, (32, 4, 3))] + [g.normal(0, scale / 2, (32, 32, 3)) for _ in range(3)]
    return Params(conv=conv, head0_w=g.normal(0, 0.2, (64, 112)), head0_b=g.normal(0, 0.2, 64),
                  head3_w=g.normal(0, 0.2, (1, 64)), head3_b=g.normal(0, 0.1, 1),
                  s_sd=1.6, lvl_mu=0.0, lvl_sd=3.0, side_mu=g.normal(0, 0.1, 16),
                  side_sd=np.abs(g.normal(0.7, 0.2, 16)) + 0.1, meta={'random_seed': seed})
