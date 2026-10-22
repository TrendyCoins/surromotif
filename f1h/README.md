# f1h — the SuMo classifier F1 and its line M on 1H candles

Numpy reference of the model SP_l025_d1 (prod a, "F1") on hourly candles: for every closed candle, the 64-candle window
that ends on it, its logit and the line M — how many times windows like this one are met more often (orange) or less
often (blue) in real markets than on a random chart, M = 1 — indistinguishable. CPU only; at run time it needs numpy and
the export `export/f1h_model.json`, nothing else (no torch, no pandas, no `research/`, no `artifacts/`).

The rules it follows and the checks with numbers: `docs/rules.md` R20–R26, `docs/progress.md`, `docs/checks/f1h/`.

## API

```python
import f1h
model = f1h.load_model()                      # export/f1h_model.json of the repo, or load_model('path/f1h_model.json')
out = f1h.compute(model, time_ms, open_, high, low, close)
f1h.bars_needed(model, k)                     # candles for the line on the last k candles: 4360 + k (continuous)
f1h.check_golden(model, 'path/btc1h_f1h.csv.gz')   # end-to-end check, dict with 'ok'
```

`python -m f1h [--model PATH] [--golden PATH]` runs the same check and exits with 0 when it passes.

### Input

Closed 1H candles of one symbol, oldest first, as an exchange returns them: open time (integer ms UTC, a multiple of the
hour) and open, high, low, close. Missing hours stay missing: they are not filled in, the windows over them do not exist.
`ValueError` if the time does not increase, is not on the hour, the arrays differ in length or a candle is broken (a
price not finite or <= 0, high < low, open or close outside [low, high]). The last, still open candle must not be passed:
a value of a candle depends on it being final.

### Output

A dict of arrays, one value per input candle (the window that ends on it). No value = NaN (numbers: -1, flags: false).

| key | type | what |
|---|---|---|
| `time` | int64 | open time of the candle, ms UTC |
| `window` | bool | a window ends on this candle |
| `n` | int64 | number of the window from the first window of the input (-1: no window) |
| `logit` | float64 | logit of the window |
| `z` | float64 | z of the window by the ruler (37 quantiles of the logits of surrogates) |
| `m`, `s` | float64 | centre and spread of the live part (the 4032 windows ending 64 windows earlier), before the limits of log M |
| `n_live` | int64 | windows in the live part (-1: no window) |
| `line` | bool | the line M is there: window number >= 4096 |
| `log_m` | float64 | log M (NaN: no line) |
| `strength` | float64 | min(exp \|log M\|, 8): M for orange, 1 / M for blue, from 1 to the ceiling 8 (NaN: no line) |
| `side` | float64 | +1 orange (more often than random), -1 blue (less often), 0 — M = 1 (NaN: no line) |

### History

On continuous candles the first window ends on the candle 264 (counted from 0: 200 rows for the norm of the bar range, one
more for the steps of HL/2 and HLC/3, then 63 bars of the window), the line starts 4096 windows later, on the candle
4360. `bars_needed(model, k)` = 4360 + k candles give the line on the last k candles. Each missing hour removes the candle
after it from every window (64 windows fewer), so with gaps fetch more and check `out['line']`.

With enough history the value of a candle does not depend on where the input starts (the live part is the last 4032
windows, the network sees 264 candles): cutting the start changes the values by ~1e-14 (tests/test_f1h.py).

### Speed and memory

Full batch recompute of 4 600 candles (4 336 windows, line on the last 240): 2.2–2.4 s at 2 BLAS threads with numpy 2.5,
6.0 s with numpy 1.26.4 (its OpenBLAS is slower on this CPU), peak RSS ~100 MB (Intel Xeon @ 2.10 GHz; `scripts/
f1h_bench.py`, `docs/checks/f1h/bench.json`). numpy needs a real BLAS (the pip wheels bundle OpenBLAS); with a reference BLAS
it is many times slower.

## Checked with

- Python 3.12.3 + numpy 1.26.4 and 2.5.4; Python 3.13 + numpy 2.5.3 (tests and the golden check; the golden values are
  reproduced to <= 4e-13 on numpy 1.26.4, bit for bit on numpy 2.5).
- Against research (training code v6.6 with torch 2.14, CPU): |dlogit| <= 9.1e-6 on 74 917 windows of BTCUSDT 1H; the
  line from the logits of the research golden bit for bit (`scripts/f1h_parity.py`).

## Taking a copy

At a pinned commit of this repository copy `f1h/*.py` (the package), `export/f1h_model.json` (the model) and, for the
end-to-end check, `tests/golden/btc1h_f1h.csv.gz`; pass their paths to `load_model` and `check_golden` (the defaults are
the paths of this repository). The package does not import anything from the rest of the repository.

## Files

| file | what |
|---|---|
| `model.py` | the export -> `Model` (weights as exact float32 values, constants of the bars, window and line) |
| `bars.py` | candles -> per-bar inputs and the history of a window (research data arithmetic, numpy only) |
| `net.py` | the network over windows: input shape, 4 convolutions with zero padding, pooling, head, mirror symmetry |
| `line.py` | ruler z, live part (4032·k0.5), log M, strength and side |
| `run.py` | `compute`, `bars_needed` |
| `golden.py` | the golden file: write, read, compare (`check_golden`) |
