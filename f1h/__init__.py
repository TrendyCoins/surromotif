"""f1h — the SuMo classifier F1 (SP_l025_d1 prod a) and its line M on 1H candles: numpy reference for the CPU.

    import f1h
    model = f1h.load_model()                          # export/f1h_model.json (or load_model(path))
    out = f1h.compute(model, time_ms, o, h, l, c)     # dict of per-candle arrays, see f1h.run.compute
    f1h.bars_needed(model, k)                         # candles for the line on the last k candles (continuous)
    f1h.check_golden(model)                           # end-to-end check on tests/golden/btc1h_f1h.csv.gz

Modules: model (the export), bars (candles -> per-bar inputs), net (the network over windows), line (ruler, live part,
log M), run (compute, bars_needed), golden (read and check the golden file). Runtime dependency: numpy only.
"""
from .golden import check_golden, read_golden
from .model import Model, from_export, load_model
from .run import FIELDS, bars_needed, compute

__all__ = ['Model', 'load_model', 'from_export', 'compute', 'bars_needed', 'FIELDS', 'read_golden', 'check_golden']
