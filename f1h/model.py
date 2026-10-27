"""The model of an export (export/f1h_model.json — 1H; f1h_model_5m.json, _15m.json, _30m.json — 5M, 15M, 30M): the weights
of SP_l025_d1 prod a as float64 copies of their exact float32 values, and the constants of the bars, of the window and of
the line M on the timeframe of the export. The network is the same on every timeframe; the timeframe sets the grid of the
candles (bar_ms) and the line (ruler, live part, ceiling)."""
import json
import os
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT = os.path.join(ROOT, 'export', 'f1h_model.json')
EXPORTS = {'1H': EXPORT, **{tf: os.path.join(ROOT, 'export', f'f1h_model_{tf.lower()}.json') for tf in ('5M', '15M', '30M')}}
FORMAT, FORMAT_VERSION = 'f1h-export', 1
FAMILIES = ('F3', 'F2')         # live part: F3 — centre minus k errors of the mean; F2 — plus n0 virtual random windows


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
    live_k: float                   # F3: centre minus k standard errors of the mean (F2: None)
    live_tau: int                   # F3: one independent window per tau (F2: None)
    s_lo: float                     # spread of the live part not below s_lo (and not above 1)
    show_from: int                  # the line from the window number show_from
    ceiling: float                  # strength on the line not above the ceiling (both colours)
    live_family: str = 'F3'         # F3 or F2
    live_n0: int = 0                # F2: virtual random windows (z ~ N(0, 1)) added to the set (F3: 0)
    timeframe: str = '1H'
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


def tf_ms(tf):
    """'5M' -> 300 000, '1H' -> 3 600 000 (ms in a bar of the timeframe)."""
    n, unit = int(tf[:-1]), tf[-1].upper()
    return n * {'M': 60_000, 'H': 3_600_000}[unit]


def _live(lv):
    """Family, n0, k, tau of the live part; ValueError if they do not make one of FAMILIES."""
    fam, n0, k, tau = lv['family'], lv.get('n0', 0), lv['k'], lv['tau']
    if fam == 'F3' and n0 == 0 and k is not None and tau is not None:
        return fam, 0, float(k), int(tau)
    if fam == 'F2' and int(n0) > 0 and k is None and tau is None:
        return fam, int(n0), None, None
    raise ValueError(f"live part {lv.get('key')}: family {fam}, n0 {n0}, k {k}, tau {tau} — not one of {FAMILIES} "
                     f"(F3: n0 0, k and tau set; F2: n0 > 0, k and tau null)")


def from_export(ex):
    """Export dict -> Model."""
    if ex.get('format') != FORMAT or ex.get('format_version') != FORMAT_VERSION:
        raise ValueError(f"not an {FORMAT} v{FORMAT_VERSION} export: {ex.get('format')} v{ex.get('format_version')}")
    mo, ba, wi, li = ex['model'], ex['bars'], ex['window'], ex['line']
    lv = li['live']
    fam, n0, k, tau = _live(lv)
    if tf_ms(ba['timeframe']) != int(ba['bar_ms']):
        raise ValueError(f"bars: timeframe {ba['timeframe']} but bar_ms {ba['bar_ms']}")
    m = Model(conv_w=[_f64(w) for w in mo['conv_w']], conv_b=[_f64(b) for b in mo['conv_b']],
              dilations=tuple(int(d) for d in mo['dilations']),
              head0_w=_f64(mo['head0_w']), head0_b=_f64(mo['head0_b']),
              head3_w=_f64(mo['head3_w']), head3_b=_f64(mo['head3_b']),
              s_sd=float(_f64(mo['s_sd'])), lvl_mu=float(_f64(mo['lvl_mu'])), lvl_sd=float(_f64(mo['lvl_sd'])),
              side_mu=_f64(mo['side_mu']), side_sd=_f64(mo['side_sd']),
              side_signed=np.asarray(mo['side_signed'], dtype=bool),
              floor=float(wi['floor']), orient_eps=float(wi['orient_eps']),
              q=np.asarray(li['ruler']['q'], dtype=np.float64), zk=np.asarray(li['ruler']['zk'], dtype=np.float64),
              live_l=int(lv['L']), live_lag=int(lv['lag']), live_k=k, live_tau=tau,
              s_lo=float(lv['s_lo']), show_from=int(li['show_from_window']), ceiling=float(li['ceiling']),
              live_family=fam, live_n0=n0, timeframe=ba['timeframe'], bar_ms=int(ba['bar_ms']),
              gap_factor=float(ba['gap_factor']), n_norm=int(ba['n_norm']), n_hist=int(ba['n_hist']), window=int(wi['L']),
              meta={'model': mo['id'], 'source': mo['source'], 'code_version': ex['passport'].get('code_version')})
    h = ex['history']
    if (m.first_window_bar, m.first_line_bar) != (h['first_window_bar'], h['first_line_bar']):
        raise ValueError(f"export history {h['first_window_bar']} / {h['first_line_bar']} != "
                         f"{m.first_window_bar} / {m.first_line_bar} from the constants")
    return m


def load_model(path=None, tf=None):
    """The export at path -> Model; without a path — the export of the timeframe tf of this repository (EXPORTS; without
       tf — 1H, export/f1h_model.json)."""
    if path is None:
        path = EXPORTS[(tf or '1H').upper()]
    with open(path, encoding='utf-8') as fh:
        m = from_export(json.load(fh))
    if tf is not None and m.timeframe != tf.upper():
        raise ValueError(f"{path}: timeframe {m.timeframe}, not {tf}")
    return m
