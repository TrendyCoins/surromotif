# export — the model as JSON

`mri_model_pb.json` — the indicator model, self-contained: weights (exact float32), inputs and side inputs, tie rule, floor,
the line M (ruler, live part, start of the line, ceiling), passport. Written by `python -m mri.export_pb`; `pine/` is
generated from it, `tests/golden/` checks it bar by bar.

`legacy/` — exports of the earlier models F3_g and NT3_PRV_ch48.
