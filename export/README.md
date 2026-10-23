# export — the model as JSON

`mri_model_pb.json` — the indicator model, self-contained: weights (exact float32), inputs and side inputs, tie rule, floor,
the line M (ruler, live part, start of the line, ceiling), passport. Written by `python -m mri.export_pb`; `pine/` is
generated from it, `tests/golden/` checks it bar by bar.

`f1h_model.json` — the SuMo classifier F1 (SP_l025_d1 prod a) and its line M on 1H candles, for the package `f1h/`:
weights (exact float32), bars, window, line M (ruler, live part 4032·k0.5, floor, ceiling 8, history), passport with the
sha256 of its sources. Written by `python scripts/f1h_export.py`.

`legacy/` — exports of the earlier models F3_g and NT3_PRV_ch48.
