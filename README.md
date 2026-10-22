# Market Randomness Index (MRI)

The open code behind MRI, a TradingView indicator (Pine Script v6) for 5-minute charts: the research that trained its
classifier, the Python reference of the indicator, the Pine script generated from it, and the checks that tie them
together bar by bar.

## What MRI shows

How many times a 64-bar window of this shape is more frequent (orange) or less frequent (blue) in real markets than in a
random walk with the same volatility; M = 1 — indistinguishable from random. The window is scored by the SuMo classifier,
a small streaming CNN trained on Binance spot 5M candles of 31 assets, and the score becomes the line M without constants
of the symbol. M is a frequency ratio of the window's shape, not a forecast and not financial advice.

## Why this repository is open

For transparency: the development history, the rules of every check written before its run (`docs/rules.md`) and
the results with numbers. It is a record of the work, not a product: provided as is under Apache-2.0, without support or
promises.

## Map

| folder | what is there |
|---|---|
| `pine/` | the indicator `mri.pine` and its test prototype |
| `mri/` | Python reference of the indicator: model, line M, export, Pine generator |
| `export/` | the model as JSON: weights, rules, constants |
| `tests/` | pytest and the golden file (bar-by-bar values on BTCUSDT 5M) |
| `scripts/` | checks against research code, parity with TradingView, Colab exports |
| `docs/` | progress, rules of the checks, Pine limits, check results |
| `research/` | research scripts as they ran in Colab (their Drive paths and Colab links are private): data, training, surrogate tests |
| `artifacts/` | weights, scores and data slices for the checks (the indicator does not need them) |

Earlier models are kept in `legacy/` subfolders with their tests.

## Notes

- **Data.** `tests/golden/` and `artifacts/bars/` hold Binance spot candles of BTCUSDT 5M (`artifacts/bars/` also has
  BTCUSDT 1H: the 5M candles resampled to hours, in the research file format); `artifacts/parity/` holds TradingView
  chart exports (BINANCE:BTCUSDT and XLMUSDT 5M) with the indicator's debug values. The market data belongs to its
  sources and is here only for the tests and the parity checks.
- **SMC** in `research/` and the logs means SuMo classifier, not Smart Money Concepts.
- **Language.** Code and READMEs are in English; `docs/` and the comments in `research/` are in Russian.
- **Tests:** `python -m pytest -q`.

License: Apache-2.0.
