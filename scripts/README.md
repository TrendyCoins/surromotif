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
| `f1h_export.py` | F1 on 1H: `export/f1h_model.json` and `tests/golden/btc1h_f1h.csv.gz` (rules R21, R24) |
| `f1h_parity.py` | F1 on 1H against research and the research golden tfline21 (rules R22, R23; torch) |
| `f1h_research.py` | F1 on 1H: the research side (training code imported, prod a, floor 0.2) for the parity and tests |
| `f1h_bench.py` | F1 on 1H: time and memory of the full recompute (rule R26) |
| `futures_klines.py` | run locally, not in Colab: 5M candles of Binance USDT-M perpetual futures, with the taker volume split by side, for `research/sumo_class_flow.py` (CSV → Drive `Raw Datasets/futures/`, not in git) |

Results go to `docs/checks/pb5/` (F1 on 1H — `docs/checks/f1h/`). `legacy/` — scripts of the earlier models and of session 4.
