# scripts — checks, parity, Colab exports

| script | what |
|---|---|
| `check_torch.py` | reference against the research torch model on the same bars (rule R2) |
| `pb5_gates.py` | gates of session 5: research files, Live / Ceil json, the Colab export |
| `pb5_r3.py` | reference against the scores of the Colab export on VAL BTC (rule R3, floor 0.2) |
| `pb5_ceil.py` | reproduction of the SuMo_Class_Ceil json in the repo (two ways); required before the export |
| `parity.py` | parity of the Pine prototype (TradingView debug export) with the reference (rule R10) |
| `display_metrics.py` | how jumpy the line M is (rule R5, reference) |
| `colab_export_ceil_f02.py` | Colab: export of the scores with the floor 0.2 from the SuMo_Class_Ceil cache |
| `f1h_export.py` | F1 by timeframe (`--tf 5M / 15M / 30M`): `export/f1h_model_<tf>.json` and `tests/golden/btc<tf>_f1h.csv.gz` (rules R28, R31); 1H (`export/f1h_model.json`, rules R21, R24) — only into another folder (`--out`), the committed files are held by the site generator |
| `f1h_parity.py` | F1 by timeframe (`--tf`) against research and the golden of the source (rules R22, R23 — 1H; R29, R30, R31 — 5M, 15M, 30M; torch) |
| `f1h_research.py` | F1: the research side (training code imported, prod a, floor 0.2, research files of a timeframe) for the parity and tests |
| `f1h_bench.py` | F1: time and memory of the full recompute (rule R26; `--tf` — the export and golden of a timeframe) |
| `f1h_tf_line21.py` (`.ipynb`) | Colab notebook: line M of F1 on 15M / 30M — modes 'line_stand' and 'line21' of `research/sumo_class_tf.py` v2.1 with the timeframe and folders swapped; the stand copied with the TF as a parameter (rule R27) |
| `colab_notebooks.py` | builds the Colab notebooks `f1h_tf_line21.ipynb`, `f1h_tf_pack.ipynb` from their .py (a text cell with the run order and one code cell = the .py as is; the .py is the source, a test checks the sync) |
| `f1h_tf_stand.py` | the stand of `f1h_tf_line21.py` here (CPU): regression with the research file on 1H bit for bit, 15M / 30M (rule R27) |
| `f1h_tf_pack.py` (`.ipynb`) | Colab notebook: one zip with the inputs of f1h on 5M / 15M / 30M (source json and golden, research files BTC, logs, manifests; rules R28–R31) |
| `futures_klines.py` | run locally, not in Colab: 5M candles of Binance USDT-M perpetual futures, with the taker volume split by side, for `research/sumo_class_flow.py` and `research/sumo_class_eff.py` (CSV → Drive `Raw Datasets/futures/`, not in git) |

Results go to `docs/checks/pb5/` (F1 on 1H — `docs/checks/f1h/`; the wrapper 15M / 30M — `docs/checks/f1h_tf/`). `legacy/` — scripts of the earlier models and of session 4.
