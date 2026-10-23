# tests — pytest

`python -m pytest -q` from the repo root. Synthetic data (`synth.py`) and the golden file check the reference, the line M,
the export and the Pine template (transliterated line by line in `pine_sim_pb.py`); checks against research code need
torch.

- `golden/btc5m_fresh_pb.csv.gz` — bar-by-bar values of the indicator on 26 000 bars of BTCUSDT 5M.
- `golden/btc1h_f1h.csv.gz` — F1 on 1H (package `f1h/`): 11 480 continuous BTCUSDT 1H candles (Binance spot 5M
  resampled to hours) and the values of every candle; `test_f1h*.py` — f1h on synthetic candles, the golden file, the
  export, research on synthetic candles (torch).
- `legacy/`, `golden/legacy/` — tests and golden files of the earlier models and of the EMA line of session 4.
