# f1h — the SuMo classifier F1 and its line M on 5M, 15M, 30M and 1H candles

Numpy reference of the model SP_l025_d1 (prod a, "F1") on candles of one timeframe: for every closed candle, the 64-candle
window that ends on it, its logit and the line M — how many times windows like this one are met more often (orange) or
less often (blue) in real markets than on a random chart, M = 1 — indistinguishable. CPU only; at run time it needs numpy
and the export of the timeframe, nothing else (no torch, no pandas, no `research/`, no `artifacts/`).

The network is one on every timeframe. The export of a timeframe sets the grid of the candles (bar length, grid gap) and
the line M — its own ruler, live part and ceiling, each checked on that timeframe:

| timeframe | export | golden | live part | ceiling |
|---|---|---|---|---|
| 5M | `export/f1h_model_5m.json` | `tests/golden/btc5m_f1h.csv.gz` | 4032·k0.5 (F3) | 10 |
| 15M | `export/f1h_model_15m.json` | `tests/golden/btc15m_f1h.csv.gz` | 4032+288 (F2) | 10 |
| 30M | `export/f1h_model_30m.json` | `tests/golden/btc30m_f1h.csv.gz` | 4032·k0.5 (F3) | 8 |
| 1H | `export/f1h_model.json` | `tests/golden/btc1h_f1h.csv.gz` | 4032·k0.5 (F3) | 8 |

The ceiling in the export is the checked one; the site shows the strength up to 8 on every timeframe (its generator cuts).

The rules it follows and the checks with numbers: `docs/rules.md` R20–R26 (1H) and R27–R32 (5M, 15M, 30M),
`docs/progress.md`, `docs/checks/f1h/`.

## API

```python
import f1h
model = f1h.load_model(tf='15M')              # the export of the timeframe in this repo, or load_model('path/....json')
out = f1h.compute(model, time_ms, open_, high, low, close)
f1h.bars_needed(model, k)                     # candles for the line on the last k candles: 4360 + k (continuous)
f1h.check_golden(model, 'path/btc15m_f1h.csv.gz')  # end-to-end check, dict with 'ok'
```

`load_model()` without arguments is the 1H export. `python -m f1h [--tf TF] [--model PATH] [--golden PATH]` runs the same
check and exits with 0 when it passes (defaults: the export and the golden of the timeframe in this repo, 1H without
`--tf`). `model.timeframe` and `model.bar_ms` say what the export is for.

### Input

Closed candles of one symbol on the timeframe of the export, oldest first, as an exchange returns them: open time (integer
ms UTC, a multiple of the bar: 5, 15, 30 or 60 minutes) and open, high, low, close. Missing bars stay missing: they are not
filled in, the windows over them do not exist; an interval longer than 1.5 bars is a grid gap. `ValueError` if the time does
not increase, is not on the grid of the timeframe, the arrays differ in length or a candle is broken (a price not finite or
<= 0, high < low, open or close outside [low, high]). The last, still open candle must not be passed: a value of a candle
depends on it being final.

### Output

A dict of arrays, one value per input candle (the window that ends on it). No value = NaN (numbers: -1, flags: false).

| key | type | what |
|---|---|---|
| `time` | int64 | open time of the candle, ms UTC |
| `window` | bool | a window ends on this candle |
| `n` | int64 | number of the window from the first window of the input (-1: no window) |
| `logit` | float64 | logit of the window |
| `z` | float64 | z of the window by the ruler of the timeframe (37 quantiles of the logits of surrogates) |
| `m`, `s` | float64 | centre and spread of the live part (the 4032 windows ending 64 windows earlier), before the limits of log M |
| `n_live` | int64 | windows in the live part (-1: no window) |
| `line` | bool | the line M is there: window number >= 4096 |
| `log_m` | float64 | log M (NaN: no line) |
| `strength` | float64 | min(exp \|log M\|, ceiling): M for orange, 1 / M for blue, from 1 to the ceiling of the export (NaN: no line) |
| `side` | float64 | +1 orange (more often than random), -1 blue (less often), 0 — M = 1 (NaN: no line) |

### Live part

Over the z of the windows [p − 4095, p − 63] (fewer at the start of the input), n of them, sums s1 = Σ z, s2 = Σ z²:

- F3 `4032·k0.5` (5M, 30M, 1H): m = max(s1 / n − 0.5 · s · sqrt(64 / n), 0), s² = max(s2 / n − (s1 / n)², 0) — the centre
  minus half an error of the mean;
- F2 `4032+288` (15M): m = s1 / (n + 288), s² = (max(s2 − s1² / n, 0) + 288) / (n + 288) — 288 virtual random windows
  (z ~ N(0, 1)) added to the set, no k correction;
- n = 0: m = 0, s = 1. Then log M — the ratio of the bells N(m, s²) / N(0, 1) at z (s in [0.75; 1], m ≥ 0).

### History

On continuous candles the first window ends on the candle 264 (counted from 0: 200 rows for the norm of the bar range, one
more for the steps of HL/2 and HLC/3, then 63 bars of the window), the line starts 4096 windows later, on the candle
4360 — on every timeframe. `bars_needed(model, k)` = 4360 + k candles give the line on the last k candles (160 bars of
the chart with the line: 4520 candles; 4520 bars are ~15.7 days of 5M, ~47 days of 15M, ~94 days of 30M, ~188 days of 1H).
Each missing bar removes the candle after it from every window (64 windows fewer), so with gaps fetch more and check
`out['line']`.

With enough history the value of a candle does not depend on where the input starts (the live part is the last 4032
windows, the network sees 264 candles): cutting the start changes the values by ~1e-14 (tests/test_f1h.py).

### Speed and memory

Full batch recompute of 4 600 candles (4 336 windows, line on the last 240): 2.2–2.4 s at 2 BLAS threads with numpy 2.5,
6.0 s with numpy 1.26.4 (its OpenBLAS is slower on this CPU), peak RSS ~100 MB (Intel Xeon @ 2.10 GHz; `scripts/
f1h_bench.py`, `docs/checks/f1h/bench.json`). The cost per candle does not depend on the timeframe. numpy needs a real BLAS
(the pip wheels bundle OpenBLAS); with a reference BLAS it is many times slower.

## Checked with

- Python 3.12.3 + numpy 1.26.4 and 2.5.4; Python 3.13 + numpy 2.5.3 (tests and the golden check; the golden values are
  reproduced to <= 4e-13 on numpy 1.26.4, bit for bit on numpy 2.5).
- Against research (training code v6.6 with torch 2.14, CPU) on BTCUSDT: max |dlogit| 9.1e-6 on 1H (74 917 windows),
  1.22e-5 on 5M (113 175), 1.14e-5 on 15M (305 782), 9.7e-6 on 30M (151 880); against the logits of the golden of the
  source (GPU) <= 1.85e-5; the line from those logits bit for bit on every timeframe (`scripts/f1h_parity.py --tf TF`,
  `docs/checks/f1h/parity*.json`).

## Taking a copy

At a pinned commit of this repository copy `f1h/*.py` (the package), the exports of the timeframes you need and, for the
end-to-end check, their golden files (table above); pass their paths to `load_model` and `check_golden` (the defaults are
the paths of this repository). The package does not import anything from the rest of the repository.

## Files

| file | what |
|---|---|
| `model.py` | the export -> `Model` (weights as exact float32 values, constants of the bars, window and line; exports by timeframe) |
| `bars.py` | candles -> per-bar inputs and the history of a window (research data arithmetic, numpy only) |
| `net.py` | the network over windows: input shape, 4 convolutions with zero padding, pooling, head, mirror symmetry |
| `line.py` | ruler z, live part (F3 4032·k0.5 or F2 4032+288), log M, strength and side |
| `run.py` | `compute`, `bars_needed` |
| `golden.py` | the golden file: write, read, compare (`check_golden`) |
