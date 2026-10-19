# pine — the TradingView indicator (Pine Script v6)

- `mri.pine` — the indicator: one script, weights inside, "Compute only the last N bars" = 10000 by default.
- `proto_pb.pine` — its test prototype: the same computation, N = 0 and a debug mode (values for the parity check with the
  Python reference, `scripts/parity.py`).

Both are generated from `export/mri_model_pb.json` by `python -m mri.pine_pb [--production]`; do not edit by hand.
`legacy/` — the indicators of the earlier models F3_g and NT3_PRV_ch48.
