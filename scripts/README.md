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

Results go to `docs/checks/pb5/`. `legacy/` — scripts of the earlier models and of session 4.
