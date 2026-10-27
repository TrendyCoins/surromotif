"""Benchmark of F1 (rules R26, R32, reference): full batch recompute of the last N candles of the golden file of a timeframe
(one symbol, continuous BTCUSDT; 1H by default) with its export on this CPU, at 1, 2 and 4 BLAS threads, each in a fresh
process (the thread count is fixed when numpy loads). Time — the best of R runs; memory — peak RSS of the process and the
peak of numpy allocations (tracemalloc) during one compute.

    python scripts/f1h_bench.py [--tf 1H] [--n 4600] [--runs 5] [--threads 1 2 4] [--python python3]
                                [--out docs/checks/f1h/bench.json]
"""
import argparse
import json
import os
import platform
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD = r"""
import json, resource, sys, time, tracemalloc
sys.path.insert(0, ROOT)
import numpy as np
import f1h
model = f1h.load_model(tf=TF)
g = f1h.read_golden(f1h.golden.GOLDENS[model.timeframe])
args = [g[k][-N:] for k in ('time', 'open', 'high', 'low', 'close')]
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
tracemalloc.start()
out = f1h.compute(model, *args)
peak = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()
times = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    f1h.compute(model, *args)
    times.append(time.perf_counter() - t0)
print(json.dumps({'tf': model.timeframe, 'numpy': np.__version__, 'python': sys.version.split()[0], 'candles': N,
                  'windows': int(out['window'].sum()), 'with_line': int(out['line'].sum()),
                  'secs_best': min(times), 'secs_all': times, 'rss_before_mb': rss0 / 1024,
                  'rss_peak_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  'numpy_peak_mb': peak / 2 ** 20}))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tf', default='1H')
    ap.add_argument('--n', type=int, default=4600)
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4])
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    code = CHILD.replace('ROOT', repr(ROOT)).replace('RUNS', str(a.runs)).replace('tf=TF', f'tf={a.tf!r}').replace('N:', f'{a.n}:').replace(
        "'candles': N", f"'candles': {a.n}")
    rows = []
    for th in a.threads:
        env = dict(os.environ, OPENBLAS_NUM_THREADS=str(th), OMP_NUM_THREADS=str(th), MKL_NUM_THREADS=str(th))
        r = subprocess.run([a.python, '-c', code], env=env, capture_output=True, text=True, check=True)
        row = {'threads': th, **json.loads(r.stdout.strip().splitlines()[-1])}
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != 'secs_all'}))
    res = {'tf': a.tf, 'cpu': platform.processor() or platform.machine(), 'cpus': os.cpu_count(), 'rows': rows}
    try:
        with open('/proc/cpuinfo') as fh:
            res['cpu'] = next(ln.split(':', 1)[1].strip() for ln in fh if ln.startswith('model name'))
    except (OSError, StopIteration):
        pass
    if a.out:
        old = []
        if os.path.isfile(a.out):
            with open(a.out) as fh:
                old = json.load(fh).get('runs', [])
        with open(a.out, 'w') as fh:
            json.dump({'runs': old + [res]}, fh, indent=1)


if __name__ == '__main__':
    main()
