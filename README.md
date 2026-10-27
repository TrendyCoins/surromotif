# SurroMotif (SuMo) — the open code behind the Market Randomness Index (MRI) indicator

MRI is a machine-learning indicator: a neural network trained to tell a real market from a random one. This repository
holds all of it — the research that trained the models, the models, the TradingView indicator and the checks that tie
them together. The name is short for *surrogate motifs*: price patterns tested against random surrogates.

## What MRI shows

On every closed bar MRI looks at the latest 64 bars and answers one question: where does a situation like this occur
more often — in real markets or in a random twin of the same bars, where a coin toss decides whether each 4-bar
stretch is flipped upside down? The answer is a number M and a side: M = 3 means three times more often on that side;
M ≈ 1 means equally often, and MRI is silent.

**What it is for.** MRI shows when there is something to look for in the market: on a random market, any pattern you
find is noise.

**What it is not.** A forecast. MRI is blind to direction by design — to the model, a window and its upside-down copy
are the same thing — so it never says where the price will go. Not financial advice.

MRI comes as a TradingView indicator for 5-minute charts and as the colors of the price bars on
[trendycoins.com](https://trendycoins.com).

## How it is built

- **Self-supervised learning.** A convolutional neural network (PyTorch) learns to tell real 64-bar windows from their
  random twins. The labels come for free — real or twin — so there is no human labeling and nothing taken from the
  future. Trained this way, a classifier learns the odds that a window comes from a real market rather than from
  chance, and those odds are the ratio behind M
  ([noise-contrastive estimation](https://proceedings.mlr.press/v9/gutmann10a.html)). The language model
  [ELECTRA](https://arxiv.org/abs/2003.10555) is pretrained the same way: it learns by spotting replaced words in real
  text.
- **Data.** 5-minute candles of 31 crypto markets (Binance spot) since 2018. Quality is measured only out of sample, on
  later periods and on symbols kept out of training: AUC around 0.8 against the random twin, where 0.5 is chance.
- **Two models.** A compact streaming CNN (~19 thousand weights) that runs bar by bar inside the Pine Script indicator
  itself, with no server behind it; and a larger mirror-symmetric CNN (~90 thousand weights) for
  [trendycoins.com](https://trendycoins.com), which runs on a plain CPU with numpy alone.
- **Rigor.** The rule of every check is written down before its run; anything decided after seeing the results is
  marked as such. On a pure random walk every model must come out at 0.5. The Pine indicator matches its Python
  reference bar by bar to ~1e-13 on full TradingView exports.
- **What was checked.** On days of market stress MRI sees a less random market, while classical efficiency measures
  (autocorrelation, variance ratios) see nothing; at price levels, at trend reversals and in divergences with price it
  gives no edge.

## Map

| folder | what is there |
|---|---|
| `pine/` | the indicator `mri.pine` and its test prototype |
| `mri/` | Python reference of the indicator: model, line M, export, Pine generator |
| `f1h/` | the website model (F1) on 1H candles: numpy reference of the logit and the line M (CPU, numpy only) |
| `export/` | the models as JSON: weights, rules, constants |
| `tests/` | pytest and the golden files (bar-by-bar values on BTCUSDT 5M and 1H) |
| `scripts/` | checks against research code, parity with TradingView, Colab exports |
| `docs/` | progress, rules of the checks, Pine limits, check results (in Russian) |
| `research/` | research scripts as they ran in Colab (their Drive paths and Colab links are private): data, training, surrogate tests; SMC there is the SuMo classifier, not Smart Money Concepts |
| `artifacts/` | weights, scores and data slices for the checks (the indicator does not need them) |

Earlier models are kept in `legacy/` subfolders with their tests.

**Market data.** Slices of Binance spot candles and TradingView chart exports are kept here only for the tests and the
parity checks; the data belongs to its sources.

License: Apache-2.0, provided as is.
