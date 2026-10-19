# mri — Python reference of the indicator

Everything `pine/mri.pine` computes, in Python: bar by bar (as Pine) and vectorized, plus the export and the Pine generator.
Model: SuMo classifier PB_s64h24_d1; line M of session 5 (one floor, formula A live part, ceiling 10).

| module | what |
|---|---|
| `params.py` | weights and buffers (from the research npz or the export JSON) |
| `features.py` | per-bar inputs from OHLC |
| `batch.py`, `nt3.py` | the network: conv streams, head, the 200 bars of history before a window |
| `pb.py` | the model with the floor: vectorized chart and the bar-by-bar stream with the line M |
| `linem.py` | line M: ruler, live part (formula A), log M, display |
| `export_pb.py` | `python -m mri.export_pb` → `export/mri_model_pb.json` and `tests/golden/btc5m_fresh_pb.csv.gz` |
| `pine_pb.py` | `python -m mri.pine_pb [--production]` → `pine/proto_pb.pine` / `pine/mri.pine` |
| `pine_common.py`, `export_common.py` | pieces shared with the legacy generators and exports |
| `research_bridge.py`, `research5.py`, `ceil_export.py` | checks only: research data files, the research chain, the Colab export |

`legacy/` — earlier models F3_g and NT3_PRV_ch48 and the EMA line of session 4; the current code does not import it.
