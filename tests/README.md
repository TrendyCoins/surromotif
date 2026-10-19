# tests — pytest

`python -m pytest -q` from the repo root. Synthetic data (`synth.py`) and the golden file check the reference, the line M,
the export and the Pine template (transliterated line by line in `pine_sim_pb.py`); checks against research code need
torch.

- `golden/btc5m_fresh_pb.csv.gz` — bar-by-bar values of the indicator on 26 000 bars of BTCUSDT 5M.
- `legacy/`, `golden/legacy/` — tests and golden files of the earlier models and of the EMA line of session 4.
