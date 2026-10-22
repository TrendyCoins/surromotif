"""The model of the export (export/f1h_model.json): the weights of SP_l025_d1 prod a as float64 copies of their exact
float32 values, and the constants of the bars, of the window and of the line M on 1H."""
import json
import os
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT = os.path.join(ROOT, 'export', 'f1h_model.json')
FORMAT, FORMAT_VERSION = 'f1h-export', 1


@dataclass
class Model:
    conv_w: list                    # 4 arrays (64, c_in, 5): torch Conv1d weights (out, in, tap); c_in = 4, then 64
    conv_b: list                    # 4 arrays (64,)
    dilations: tuple
    head0_w: np.ndarray             # (128, 209)
    head0_b: np.ndarray
    head3_w: np.ndarray             # (1, 128)
    head3_b: np.ndarray
    s_sd: float                     # scale of the path steps
    lvl_mu: float                   # shift and scale of the bar range level g / D
    lvl_sd: float
    side_mu: np.ndarray             # standardization of the 17 side inputs
    side_sd: np.ndarray
    side_signed: np.ndarray         # side inputs that take the orientation sign (and flip in the mirror)
    floor: float                    # floor of the window divisor D
    orient_eps: float               # tie rule: flip the shape only if x_end < -orient_eps * D
    q: np.ndarray                   # ruler: 37 quantiles of surrogate logits
    zk: np.ndarray                  # ruler: z of the knots
    live_l: int                     # live part: windows in the set
    live_lag: int                   # the set ends lag windows before the current one
    live_k: float                   # centre minus k standard errors of the mean
    live_tau: int                   # one independent window per tau
    s_lo: float                     # spread of the live part not below s_lo (and not above 1)
    show_from: int                  # the line from the window number show_from
    ceiling: float                  # strength on the line not above the ceiling (both colours)
    bar_ms: int = 3_600_000         # 1H
    gap_factor: float = 1.5         # interval > gap_factor bars -> grid gap
    n_norm: int = 200               # rows in the median of (high - low) before a bar
    n_hist: int = 200               # rows of history before the first bar of a window
    window: int = 64                # bars in a window
    meta: dict = field(default_factory=dict)

    @property
    def first_window_bar(self):
        """Bar (from 0) where the first window ends on continuous candles: the norm needs n_norm rows, the steps of HL/2
           and HLC/3 need the norm of the previous bar as well, then window - 1 more bars."""
        return self.n_norm + 1 + self.window - 1

    @property
    def first_line_bar(self):
        return self.first_window_bar + self.show_from


def _f64(x):
    # exact float32 values of the export -> float64 (no rounding)
    return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float64)


def from_export(ex):
    """Export dict -> Model."""
    if ex.get('format') != FORMAT or ex.get('format_version') != FORMAT_VERSION:
        raise ValueError(f"not an {FORMAT} v{FORMAT_VERSION} export: {ex.get('format')} v{ex.get('format_version')}")
    mo, ba, wi, li = ex['model'], ex['bars'], ex['window'], ex['line']
    lv = li['live']
    m = Model(conv_w=[_f64(w) for w in mo['conv_w']], conv_b=[_f64(b) for b in mo['conv_b']],
              dilations=tuple(int(d) for d in mo['dilations']),
              head0_w=_f64(mo['head0_w']), head0_b=_f64(mo['head0_b']),
              head3_w=_f64(mo['head3_w']), head3_b=_f64(mo['head3_b']),
              s_sd=float(_f64(mo['s_sd'])), lvl_mu=float(_f64(mo['lvl_mu'])), lvl_sd=float(_f64(mo['lvl_sd'])),
              side_mu=_f64(mo['side_mu']), side_sd=_f64(mo['side_sd']),
              side_signed=np.asarray(mo['side_signed'], dtype=bool),
              floor=float(wi['floor']), orient_eps=float(wi['orient_eps']),
              q=np.asarray(li['ruler']['q'], dtype=np.float64), zk=np.asarray(li['ruler']['zk'], dtype=np.float64),
              live_l=int(lv['L']), live_lag=int(lv['lag']), live_k=float(lv['k']), live_tau=int(lv['tau']),
              s_lo=float(lv['s_lo']), show_from=int(li['show_from_window']), ceiling=float(li['ceiling']),
              bar_ms=int(ba['bar_ms']), gap_factor=float(ba['gap_factor']), n_norm=int(ba['n_norm']),
              n_hist=int(ba['n_hist']), window=int(wi['L']),
              meta={'model': mo['id'], 'source': mo['source'], 'code_version': ex['passport'].get('code_version')})
    h = ex['history']
    if (m.first_window_bar, m.first_line_bar) != (h['first_window_bar'], h['first_line_bar']):
        raise ValueError(f"export history {h['first_window_bar']} / {h['first_line_bar']} != "
                         f"{m.first_window_bar} / {m.first_line_bar} from the constants")
    return m


def load_model(path=None):
    """export/f1h_model.json of this repository (or the export at path) -> Model."""
    with open(path or EXPORT, encoding='utf-8') as fh:
        return from_export(json.load(fh))
