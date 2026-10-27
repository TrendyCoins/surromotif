# export — the model as JSON

`mri_model_pb.json` — the indicator model, self-contained: weights (exact float32), inputs and side inputs, tie rule, floor,
the line M (ruler, live part, start of the line, ceiling), passport. Written by `python -m mri.export_pb`; `pine/` is
generated from it, `tests/golden/` checks it bar by bar.

`f1h_model.json` (1H), `f1h_model_5m.json`, `f1h_model_15m.json`, `f1h_model_30m.json` — the SuMo classifier F1
(SP_l025_d1 prod a) and its line M on candles of one timeframe, for the package `f1h/`: weights (exact float32; the same
network in all four), bars (timeframe, bar length, grid gap), window, line M of the timeframe (own ruler, live part, floor,
ceiling, history), passport with the sha256 of its sources. Written by `python scripts/f1h_export.py --tf TF` from the
manifests `artifacts/pack/site*_manifest.json`; `f1h_model.json` is held by the lock of the site generator and is not
rewritten (`--tf 1H` writes only into another folder, a test rebuilds it field for field).

| timeframe | line M from | live part | ceiling |
|---|---|---|---|
| 5M | `research/sumo_class_site.py` v1.1 'run' (20261021_084916) | 4032·k0.5 (F3) | 10 |
| 15M | `research/sumo_class_tf.py` v2.1 'line21' via `scripts/f1h_tf_line21.py` (20261027_084749) | 4032+288 (F2: `n0` 288, `k` and `tau` null) | 10 |
| 30M | the same (20261027_090105) | 4032·k0.5 (F3) | 8 |
| 1H | `research/sumo_class_tf.py` v2.1 'line21' (20261022_063520) | 4032·k0.5 (F3) | 8 |

The ceiling is the checked one; the site shows the strength up to 8 on every timeframe (its generator cuts).

`legacy/` — exports of the earlier models F3_g and NT3_PRV_ch48.
