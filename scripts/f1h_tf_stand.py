"""The stand of the wrapper scripts/f1h_tf_line21.py here, on CPU (docs/rules.md R27), and its regression with the research file.

    python scripts/f1h_tf_stand.py regress   # TF 1H: research/sumo_class_tf.py 'line_stand' and the wrapper — bit for bit
    python scripts/f1h_tf_stand.py 15M       # the stand of the wrapper on 15M (or 30M): all checks must pass

Every stand runs in its own process (fresh modules) with the research .py files of research/ and a temporary project folder;
the stand folder is kept (SMCTF_KEEP) for the comparison and removed afterwards. Results — docs/checks/f1h_tf/: stand logs
(the temporary paths and the repo replaced by <PROJECT> / <STAND> / <REPO>) and stand_regress_1h.json, stand_15m.json, stand_30m.json.
Needs torch, scikit-learn, matplotlib, pandas.

Regression (R27): the list of checks with their numbers and the stand log are the same after replacing the stand folder, the
log path, minutes / seconds and time stamps of file names (the header line of the wrapper is dropped); the files of the kept
stand folders: npz — the same keys and arrays bit for bit (in the json 'meta' — except 'created' and 'sec'), json — equal
after the same replacements and without 'minutes' / 'sec'; png — for reference.
"""
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESEARCH = os.path.join(ROOT, 'research') + '/'
OUT = os.path.join(ROOT, 'docs', 'checks', 'f1h_tf')
WRAPPER = os.path.join(ROOT, 'scripts', 'f1h_tf_line21.py')
TIMEOUT = 3 * 3600
META_SKIP = ('created', 'sec')
JSON_SKIP = ('minutes', 'sec')
PEAK = {}


def env_for(proj):
    """Environment of one stand process: the paths of v2.1 under proj, the research .py of the repo, the folder kept."""
    e = dict(os.environ)
    e.update({'SMCTF_TRAIN_PY': RESEARCH + 'sumo_class_train.py', 'SMCTF_BELL_PY': RESEARCH + 'sumo_class_bell.py',
              'SMCTF_LIVE_PY': RESEARCH + 'sumo_class_live.py', 'SMCTF_CEIL_PY': RESEARCH + 'sumo_class_ceil.py',
              'SMCTF_SITE_PY': RESEARCH + 'sumo_class_site.py', 'SMCTF_RUN_DIR': proj + 'SMC/runs/',
              'SMCTF_OUT_DIR': proj + 'SMC/runs/tf/', 'SMCTF_DATA_5M': proj + 'SMC/data/', 'SMCTF_DATA_TF': proj + 'SMC/data_tf/',
              'SMCTF_CSV_DIR': proj + 'Raw Datasets/', 'SMCTF_DATA_1H': proj + 'SMC/data_1h/', 'SMCTF_KEEP': '1',
              'SMC_PROJECT_PATH': proj, 'MPLBACKEND': 'Agg', 'PYTHONHASHSEED': '0',
              'F1HTF_PROJECT': proj, 'F1HTF_CODE_DIR': RESEARCH})
    return e


def run(kind, tf, proj):
    """One stand: kind 'research' (the file itself, MODE 'line_stand') or 'wrapper' (TF tf). Returns (log text, stand folder,
       minutes)."""
    os.makedirs(proj + 'tmp/', exist_ok=True)
    e = env_for(proj)
    e['TMPDIR'] = proj + 'tmp/'                    # the stand folder (mkdtemp) — inside proj, removed with it
    if kind == 'research':
        e['SMCTF_MODE'] = 'line_stand'
        cmd = [sys.executable, RESEARCH + 'sumo_class_tf.py']
        logs = proj + 'SMC/runs/tf/logs/'
    else:
        e.update({'F1HTF_MODE': 'line_stand', 'F1HTF_TF': tf})
        cmd = [sys.executable, WRAPPER]
        logs = proj + {'1H': 'SMC/runs/tf/', '15M': 'SMC/runs/tf15m/', '30M': 'SMC/runs/tf30m/'}[tf] + 'logs/'
    t0 = time.time()
    p = subprocess.run(cmd, env=e, cwd=proj, capture_output=True, text=True, timeout=TIMEOUT)
    minutes = (time.time() - t0) / 60
    PEAK[kind, tf] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024      # MB, max over children so far
    if p.returncode != 0:
        raise SystemExit(f"{kind} {tf}: exit {p.returncode}\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    paths = sorted(f for f in os.listdir(logs) if f.startswith('tf_line_stand_'))
    if len(paths) != 1:
        raise SystemExit(f"{kind} {tf}: logs {paths}")
    with open(logs + paths[0], encoding='utf-8') as fh:
        text = fh.read()
    m = re.search(r"папка стенда оставлена: (\S+)", text)
    if not m:
        raise SystemExit(f"{kind} {tf}: the stand folder is not in the log (SMCTF_KEEP?)")
    return text, m.group(1), minutes


def norm(text, stand, proj):
    """Log / json text without what differs between two runs: folders, the log path, minutes and seconds, stamps."""
    text = text.replace(stand, '<STAND>/').replace(proj, '<PROJECT>/')
    text = re.sub(r"🗒️ лог: .*", "🗒️ лог: <LOG>", text)
    text = re.sub(r"\d{8}_\d{6}", "<STAMP>", text)
    text = re.sub(r"\d+(?:\.\d+)? мин", "<t> мин", text)
    text = re.sub(r"\b\d+ c\b", "<t> c", text)
    return text


def log_lines(text, stand, proj):
    return [ln for ln in norm(text, stand, proj).splitlines() if not ln.startswith('🧭 f1h_tf_line21')]


def checks_of(text):
    """Check lines of a stand log (✅ / ❌ and the text) and the final count."""
    rows = [ln.strip() for ln in text.splitlines() if ln.startswith(('   ✅ ', '   ❌ '))]
    m = re.search(r"СТЕНД v2: (\d+) / (\d+) проверок пройдено", text)
    return rows, (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _json_norm(x, stand, proj):
    if isinstance(x, dict):
        return {k: _json_norm(v, stand, proj) for k, v in x.items() if k not in JSON_SKIP}
    if isinstance(x, list):
        return [_json_norm(v, stand, proj) for v in x]
    if isinstance(x, str):
        return norm(x, stand, proj)
    return x


def _same_arr(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def same_file(pa, pb, sa, sb, ja, jb):
    """(same, why) for two files of the stand folders."""
    if pa.endswith('.npz'):
        with np.load(pa, allow_pickle=False) as a, np.load(pb, allow_pickle=False) as b:
            if sorted(a.files) != sorted(b.files):
                return False, 'keys'
            for k in a.files:
                x, y = a[k], b[k]
                if k == 'meta' and x.dtype.kind == 'U':
                    mx, my = json.loads(str(x)), json.loads(str(y))
                    for d in (mx, my):
                        for s in META_SKIP:
                            d.pop(s, None)
                    if _json_norm(mx, sa, ja) != _json_norm(my, sb, jb):
                        return False, 'meta'
                elif not _same_arr(x, y):
                    return False, k
        return True, ''
    if pa.endswith('.json'):
        with open(pa, encoding='utf-8') as fa, open(pb, encoding='utf-8') as fb:
            x, y = json.load(fa), json.load(fb)
        return _json_norm(x, sa, ja) == _json_norm(y, sb, jb), 'json'
    with open(pa, 'rb') as fa, open(pb, 'rb') as fb:
        return fa.read() == fb.read(), 'bytes'


def files_of(stand):
    """{relative path with stamps replaced (and a number when several share it, in time order): path}."""
    got = {}
    for d, _, fs in os.walk(stand):
        for f in fs:
            p = os.path.join(d, f)
            got.setdefault(re.sub(r"\d{8}_\d{6}", "<STAMP>", os.path.relpath(p, stand)), []).append(p)
    out = {}
    for k, ps in got.items():
        for i, p in enumerate(sorted(ps)):
            out[k if len(ps) == 1 else f"{k}#{i}"] = p
    return out


def save_log(text, stand, proj, name):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), 'w', encoding='utf-8') as fh:
        fh.write(text.replace(stand, '<STAND>/').replace(proj, '<PROJECT>/').replace(ROOT + '/', '<REPO>/'))


def env_info():
    import torch
    import pandas
    return {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pandas.__version__,
            'torch': torch.__version__, 'threads': torch.get_num_threads(), 'cpu': platform.processor() or platform.machine()}


def regress():
    base = tempfile.mkdtemp(prefix='f1h_tf_regress_') + '/'
    res = {'what': "R27: the wrapper with TF '1H' and the paths of v2.1 against research/sumo_class_tf.py on the stand",
           'env': env_info(), 'date': time.strftime('%Y-%m-%d %H:%M')}
    try:
        got = {}
        for kind in ('research', 'wrapper'):
            proj = base + kind + '/'
            text, stand, minutes = run(kind, '1H', proj)
            got[kind] = (text, stand, proj)
            rows, (n_ok, n) = checks_of(text)
            res[kind] = {'minutes': round(minutes, 1), 'n_ok': n_ok, 'n': n, 'checks': rows}
            save_log(text, stand, proj, f"stand_1h_{kind}.log")
        (ta, sa, ja), (tb, sb, jb) = got['research'], got['wrapper']
        la, lb = log_lines(ta, sa, ja), log_lines(tb, sb, jb)
        diff = [(i, x, y) for i, (x, y) in enumerate(zip(la, lb)) if x != y]
        res['log'] = {'lines': [len(la), len(lb)], 'same': la == lb, 'diff': diff[:20]}
        res['checks_same'] = [norm(r, sa, ja) for r in res['research']['checks']] == \
                             [norm(r, sb, jb) for r in res['wrapper']['checks']]
        fa, fb = files_of(sa), files_of(sb)
        rows, png = [], []
        for k in sorted(set(fa) | set(fb)):
            if k not in fa or k not in fb:
                rows.append({'file': k, 'same': False, 'why': 'only one side'})
                continue
            same, why = same_file(fa[k], fb[k], sa, sb, ja, jb)
            (png if k.endswith('.png') else rows).append({'file': k, 'same': same, 'why': '' if same else why})
        res['files'] = {'n': len(rows), 'same': all(r['same'] for r in rows), 'diff': [r for r in rows if not r['same']],
                        'list': [r['file'] for r in rows], 'png_reference': png}
        res['ok'] = bool(res['log']['same'] and res['checks_same'] and res['files']['same']
                         and res['research']['n_ok'] == res['research']['n'] == res['wrapper']['n_ok'])
    finally:
        shutil.rmtree(base, ignore_errors=True)       # stand folders are inside (TMPDIR)
    write(res, 'stand_regress_1h.json')
    print(f"regress 1H: log same {res['log']['same']}, checks same {res['checks_same']}, files {res['files']['n']} same "
          f"{res['files']['same']}, research {res['research']['n_ok']} / {res['research']['n']}, wrapper "
          f"{res['wrapper']['n_ok']} / {res['wrapper']['n']} -> {'OK' if res['ok'] else 'FAIL'}")
    return res['ok']


def stand(tf):
    base = tempfile.mkdtemp(prefix=f"f1h_tf_stand_{tf.lower()}_") + '/'
    res = {'what': f"R27: the stand of the wrapper on {tf}", 'env': env_info(), 'date': time.strftime('%Y-%m-%d %H:%M')}
    try:
        text, stand_dir, minutes = run('wrapper', tf, base + 'p/')
        rows, (n_ok, n) = checks_of(text)
        res.update({'tf': tf, 'minutes': round(minutes, 1), 'peak_rss_mb': round(PEAK['wrapper', tf]), 'n_ok': n_ok, 'n': n,
                    'ok': bool(n and n_ok == n),
                    'failed': [r for r in rows if r.startswith('❌')],
                    'checks': [r.replace(stand_dir, '<STAND>/').replace(base + 'p/', '<PROJECT>/') for r in rows]})
        save_log(text, stand_dir, base + 'p/', f"stand_{tf.lower()}.log")
    finally:
        shutil.rmtree(base, ignore_errors=True)       # the stand folder is inside (TMPDIR)
    write(res, f"stand_{tf.lower()}.json")
    print(f"stand {tf}: {res['n_ok']} / {res['n']} ({res['minutes']} min) -> {'OK' if res['ok'] else 'FAIL'}")
    return res['ok']


def write(res, name):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'regress'
    ok = regress() if what == 'regress' else stand(what)
    sys.exit(0 if ok else 1)
