"""f1h — the SuMo classifier F1 (SP_l025_d1 prod a) and its line M on 5M, 15M, 30M and 1H candles: numpy reference for
the CPU. One network on every timeframe; the export of a timeframe sets the grid of the candles and the line M.

    import f1h
    model = f1h.load_model(tf='15M')                  # export/f1h_model_15m.json (1H: export/f1h_model.json; or a path)
    out = f1h.compute(model, time_ms, o, h, l, c)     # dict of per-candle arrays, see f1h.run.compute
    f1h.bars_needed(model, k)                         # candles for the line on the last k candles (continuous)
    f1h.check_golden(model)                           # end-to-end check on tests/golden/btc15m_f1h.csv.gz

Modules: model (the export), bars (candles -> per-bar inputs), net (the network over windows), line (ruler, live part,
log M), run (compute, bars_needed), golden (read and check the golden file). Runtime dependency: numpy only.
"""
from .golden import check_golden, read_golden
from .model import Model, from_export, load_model
from .run import FIELDS, bars_needed, compute

__all__ = ['Model', 'load_model', 'from_export', 'compute', 'bars_needed', 'FIELDS', 'read_golden', 'check_golden']
