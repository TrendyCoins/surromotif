# Market Randomness Index (MRI)

A TradingView indicator (Pine Script v6) for 5-minute charts. It shows how many times a 64-bar window of this shape is
more frequent (orange) or less frequent (blue) in real markets than in a random walk with the same volatility; M = 1 —
indistinguishable from random. The window is scored by the SuMo classifier, a small streaming CNN, and the score becomes
the line M without constants of the symbol. M is a frequency ratio of the window's shape, not a forecast.

| folder | what is there |
|---|---|
| `pine/` | the indicator `mri.pine` and its test prototype |
| `mri/` | Python reference of the indicator: model, line M, export, Pine generator |
| `export/` | the model as JSON: weights, rules, constants |
| `tests/` | pytest and the golden file (bar-by-bar values on BTCUSDT 5M) |
| `scripts/` | checks against research code, parity with TradingView, Colab exports |
| `docs/` | progress, rules of the checks, Pine limits, check results |
| `research/` | research scripts (Colab): data, training, surrogate tests |
| `artifacts/` | weights, scores and data slices for the checks (the indicator does not need them) |

Earlier models are kept in `legacy/` subfolders with their tests.

License: Apache-2.0.
