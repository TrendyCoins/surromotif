"""The wrapper scripts/f1h_tf_line21.py and the pack scripts/f1h_tf_pack.py (docs/rules.md R27, R29).

- The stand copy mode_line_stand_tf is research/sumo_class_tf.py mode_line_stand with exactly the edits STAND_EDITS.
- The swap: with TF '1H' and the paths of v2.1 nothing changes; with 15M / 30M — only TFL['tf'], TFL['csv_ref'], DATA_1H and
  OUT_DIR. Stand parameters per TF.
- Compact research files (pack): values are slices of the whole files, the training code reads them, window inputs equal
  (needs torch). The pack end to end on a synthetic project: 5M (Site, compact), 15M (wrapper), 30M (a stopped line — not
  packed) -> zip with the manifests (needs torch and artifacts/weights).
The behaviour of the stand copy (1H bit for bit with the research file; 15M / 30M all checks) — scripts/f1h_tf_stand.py,
docs/checks/f1h_tf/.
"""
import ast
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
import zipfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
pd = pytest.importorskip('pandas')
pytest.importorskip('matplotlib')
RESEARCH_TF = os.path.join(ROOT, 'research', 'sumo_class_tf.py')
TF_ENV = ('SMCTF_MODE', 'SMCTF_TRAIN_PY', 'SMCTF_BELL_PY', 'SMCTF_LIVE_PY', 'SMCTF_CEIL_PY', 'SMCTF_SITE_PY',
          'SMCTF_RUN_DIR', 'SMCTF_OUT_DIR', 'SMCTF_DATA_5M', 'SMCTF_DATA_TF', 'SMCTF_CSV_DIR', 'SMCTF_DATA_1H')


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def W():
    return _load(os.path.join(ROOT, 'scripts', 'f1h_tf_line21.py'), 'f1h_tf_line21_test')


@pytest.fixture()
def M(monkeypatch):
    """A fresh research/sumo_class_tf.py module with its defaults (no SMCTF_* overrides)."""
    for k in TF_ENV:
        monkeypatch.delenv(k, raising=False)
    return _load(RESEARCH_TF, 'sumo_class_tf_test')


def test_stand_copy_is_the_original_with_the_edits(W):
    src = open(RESEARCH_TF, encoding='utf-8').read()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == 'mode_line_stand')
    orig = ''.join(src.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
    want = orig
    for old, new in W.STAND_EDITS:
        assert want.count(old) == 1, old
        want = want.replace(old, new)
    assert inspect.getsource(W.mode_line_stand_tf) == want
    assert want != orig
    assert not re.search(r"resample\([^)]*, 60\)|per_bar_counts\([^)]*, 60\)|ohlc_check\([^)]*, 12\)|hours=1", want)


def test_swap_1h_is_the_research_file(W, M):
    c = {'MODE': 'line_stand', 'TF': '1H', 'project_path': M.project_path, 'code_dir': 'SMC/'}
    keep = {k: getattr(M, k) for k in ('TRAIN_PY', 'BELL_PY', 'LIVE_PY', 'CEIL_PY', 'SITE_PY', 'RUN_DIR', 'DATA_5M', 'DATA_TF',
                                       'CSV_DIR', 'DATA_1H', 'OUT_DIR')}
    tfl = dict(M.TFL)
    P = W.paths(c, '1H')
    assert P['TF_PY'] == M.project_path + 'SMC/sumo_class_tf.py'
    swap = W.apply(M, '1H', P)
    assert all(a == b for a, b in swap.values()), {k: v for k, v in swap.items() if v[0] != v[1]}
    assert M.TFL == tfl and {k: getattr(M, k) for k in keep} == keep


@pytest.mark.parametrize('tf,data,out,csv', [('15M', 'SMC/data_15m/', 'SMC/runs/tf15m/', 'BTCUSDT-15M.csv'),
                                             ('30M', 'SMC/data_30m/', 'SMC/runs/tf30m/', 'BTCUSDT-30M.csv')])
def test_swap_tf_changes_only_the_tf_and_its_folders(W, M, tf, data, out, csv):
    c = {'MODE': 'line21', 'TF': tf, 'project_path': M.project_path, 'code_dir': 'SMC/'}
    tfl = dict(M.TFL)
    swap = W.apply(M, tf, W.paths(c, tf))
    changed = {k: v[1] for k, v in swap.items() if v[0] != v[1]}
    assert changed == {"TFL['tf']": tf, "TFL['csv_ref']": csv, 'DATA_1H': M.project_path + data,
                       'OUT_DIR': M.project_path + out}
    assert {k: v for k, v in M.TFL.items() if k not in ('tf', 'csv_ref')} == \
           {k: v for k, v in tfl.items() if k not in ('tf', 'csv_ref')}
    assert M.cache_1h_path('x').startswith(M.project_path + out)       # the cache of the TF — in its folder


def test_stand_params(W, M):
    p1, p15, p30 = (W.stand_params(M, tf) for tf in ('1H', '15M', '30M'))
    assert (p1['mins'], p1['k'], p1['gap3'], p1['long_conf']) == (60, 12, 3, 'part_a')
    assert p1['fil_gap'] == tuple(M.LSTAND['fil_gap']) and M._span_txt(p1['fil_min']) == '4 ч'
    assert (p15['mins'], p15['k'], p15['gap3'], p15['long_conf']) == (15, 3, 12, 'all')
    assert (p30['mins'], p30['k'], p30['gap3'], p30['long_conf']) == (30, 6, 6, 'all')
    assert M._span_txt(p15['fil_min']) == '153 сут' and M._span_txt(p30['fil_min']) == '111 сут'


def test_main_refuses_line21_on_1h(W, monkeypatch, capsys):
    monkeypatch.setenv('F1HTF_MODE', 'line21')
    monkeypatch.setenv('F1HTF_TF', '1H')
    assert W.main() is None
    assert '1H проверен' in capsys.readouterr().out


# --------------------------------------------------------------------------------------------------------------------
# pack (torch)

def _candles(n, mins, seed, start, gaps=()):
    g = np.random.default_rng(seed)
    c = 30000.0 * np.exp(np.cumsum(g.normal(0, 0.003, n)))
    o = np.concatenate([[c[0]], c[:-1]]) * np.exp(g.normal(0, 0.0003, n))
    h = np.maximum(o, c) * np.exp(np.abs(g.normal(0, 0.002, n)))
    lo = np.minimum(o, c) * np.exp(-np.abs(g.normal(0, 0.002, n)))
    t = pd.Timestamp(start) + pd.to_timedelta(mins * np.arange(n), unit='min')
    df = pd.DataFrame({'timestamp': t, 'open': o, 'high': h, 'low': lo, 'close': c})
    for i, k in gaps:
        df = df.drop(index=range(i, i + k))
    return df.reset_index(drop=True)


def _research(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('sklearn')
    if not os.path.isfile(os.path.join(ROOT, 'artifacts', 'weights', 'prod_SP_l025_d1_L64_a_slim.npz')):
        pytest.skip('artifacts/ not available')
    import f1h_research as R
    proj = R.project()
    T, CL, TF = R.modules(proj)
    pk = _load(os.path.join(ROOT, 'scripts', 'f1h_tf_pack.py'), 'f1h_tf_pack_test')
    return T, CL, TF, pk, R


def _files(TF, d, tf, df, t_valid):
    os.makedirs(d, exist_ok=True)
    return TF.build_tf_files(df, d, 'BTCUSDT', tf, 200, str(df['timestamp'].iloc[0]), t_valid, 0.2)


def test_compact_files(tmp_path):
    T, CL, TF, pk, _ = _research(tmp_path)
    df = _candles(5000, 5, 31, '2025-06-01', gaps=((1800, 3), (3900, 2)))
    whole, comp = str(tmp_path / 'whole') + '/', str(tmp_path / 'comp') + '/'
    _files(TF, whole, '5M', df, str(df['timestamp'].iloc[2600]))
    lo = pk.compact(whole, comp, '5M', 1000)
    F = {k: pk.load_npz(p) for k, p in pk.files_of(whole, '5M').items()}
    C = {k: pk.load_npz(p) for k, p in pk.files_of(comp, '5M').items()}
    n_tr = len(F['series']['r_tr'])
    assert lo == n_tr - 1000 and len(C['series']['r_tr']) == 1000
    for kind, keys in (('series', ('r', 't', 'close')), ('range', ('g',)), ('bars', ('o', 'h', 'l', 'c')),
                       ('prices', ('e_hl2', 'd_hl2', 'e_hlc3', 'd_hlc3'))):
        for k in keys:
            assert C[kind][f'{k}_tr'].tobytes() == F[kind][f'{k}_tr'][lo:].tobytes(), (kind, k)
            assert C[kind][f'{k}_va'].tobytes() == F[kind][f'{k}_va'].tobytes(), (kind, k)
    for k in 'ohlc':
        ext = np.concatenate([F['bars'][f'pre_{k}_tr'], F['bars'][f'{k}_tr']])
        assert np.array_equal(C['bars'][f'pre_{k}_tr'], ext[lo:lo + 200])
        assert np.array_equal(C['bars'][f'pre_{k}_va'], F['bars'][f'pre_{k}_va'])
    assert C['prices']['d0_hl2_tr'] == np.float32(F['prices']['d_hl2_tr'][lo - 1])
    meta = json.loads(str(C['series']['meta']))
    assert meta['parts']['TRAINING']['n'] == 1000 and meta['compact']['cut_row'] == lo
    data = pk.series(T, whole, '5M')
    chk, Cs = pk.check_compact(T, data, comp, '5M', lo)
    assert chk['ok'] and chk['windows'] > 1000, chk


def _write_json(p, x):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(x, fh)


def _golden(T, CL, pk, model, d, tf, path, meta):
    """A golden of VAL windows: st (positions of the series), t_end, logit_a — the research logit (float32), show."""
    data = pk.series(T, d, tf)
    w = data.windows(64)
    st = w['st'][data.part[w['st']] == 1].astype(np.int64)
    lg = pk.research_logits(T, CL, model, data, st).astype(np.float32)
    show = np.zeros(len(st), bool)
    show[len(st) // 2:] = True
    np.savez_compressed(path, logit_a=lg, logit=lg.astype(np.float64), st=st, t_end=data.t[st + 63].astype(np.int64),
                        show=show, meta=np.array(json.dumps(meta)))
    return len(st)


def test_pack_end_to_end(tmp_path):
    T, CL, TF, pk, R = _research(tmp_path)
    proj = str(tmp_path / 'proj') + '/'
    code = proj + 'code/'
    os.makedirs(code)
    for f, src in (('sumo_class_train.py', 'research'), ('sumo_class_ceil.py', 'research'), ('f1h_tf_line21.py', 'scripts')):
        os.symlink(os.path.join(ROOT, src, f), code + f)
    os.makedirs(proj + 'SMC/runs/prod/')
    for ext, src in (('.json', '.json'), ('.npz', '_slim.npz')):
        os.symlink(R.PROD + src, proj + 'SMC/runs/prod/prod_SP_l025_d1_L64_a' + ext)
    model, _ = R.prod_model(T)
    # 15M: the wrapper's json and golden, research files of the TF
    d15 = proj + 'SMC/data_15m/'
    df15 = _candles(3200, 15, 41, '2025-05-01')
    m15 = _files(TF, d15, '15M', df15, str(df15['timestamp'].iloc[1700]))
    out15 = proj + 'SMC/runs/tf15m/'
    os.makedirs(out15 + 'logs/')
    n15 = _golden(T, CL, pk, model, d15, '15M', out15 + 'tfline21_golden_20261026_101010.npz', {'tf': '15M'})
    _write_json(out15 + 'tfline21_20261026_101010.json',
                {'ver': '2.1', 'mode': 'line21', 'tf': '15M', 'data': {'BTCUSDT': {'crc': m15['crc']}},
                 'export': {'tf': '15M', 'rules': 'v2.1', 'floor': 0.2, 'ceiling': 8, 'marks': {'orange': 1.25, 'blue': 0.8},
                            'live': {'key': '4032·k0.5'}},
                 'golden': {'path': 'tfline21_golden_20261026_101010.npz', 'n': n15, 'same': True}})
    open(out15 + 'logs/tf_line21_20261026_100000.log', 'w').write('log')
    # 30M: a stopped line — not packed
    _write_json(proj + 'SMC/runs/tf30m/tfline21_20261026_111111.json',
                {'ver': '2.1', 'mode': 'line21', 'tf': '30M', 'stop': '(Ж): формула живой части не выбрана'})
    # 5M: Site 'run', its live json, logs, research files 5M (compact)
    d5 = proj + 'SMC/data/'
    df5 = _candles(4200, 5, 42, '2025-06-01', gaps=((900, 2),))
    _files(TF, d5, '5M', df5, str(df5['timestamp'].iloc[2400]))
    site = proj + 'SMC/runs/site/'
    os.makedirs(site + 'logs/')
    n5 = _golden(T, CL, pk, model, d5, '5M', site + 'site_golden_20261021_120000.npz', {'asset': 'BTCUSDT'})
    _write_json(site + 'site_20261021_120000.json',
                {'ver': '1.1', 'mode': 'run', 'formula': {'key': '4032·k0.5', 'from': 'site_live_20261021_100000.json'},
                 'export': {'config': 'single', 'seeds': ['a'], 'floor': 0.2, 'ceiling': 10, 'live': {'key': '4032·k0.5'},
                            'marks': {'orange': 1.25, 'blue': 0.8}},
                 'golden': {'path': 'site_golden_20261021_120000.npz', 'n': n5, 'same': True, 't_end': True}})
    _write_json(site + 'site_live_20261021_100000.json', {'pick': '4032·k0.5'})
    _write_json(site + 'site_20261020_090000.json', {'ver': '1.1', 'mode': 'run'})          # a candidate without export
    for f in ('site_run_20261021_115500.log', 'site_live_20261021_095500.log', 'site_run_20261021_130000.log'):
        open(site + 'logs/' + f, 'w').write(f)
    env = dict(os.environ, F1HTF_PACK_PROJECT=proj, F1HTF_PACK_CODE_DIR=code, F1HTF_PACK_OUT_DIR=proj + 'out/',
               F1HTF_PACK_TAIL_ROWS='800', F1HTF_PACK_N_SAMPLE='300', MPLBACKEND='Agg')
    env.pop('F1HTF_PACK_SITE_STAMP', None)
    p = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'f1h_tf_pack.py')], env=env, capture_output=True,
                       text=True, timeout=1800)
    assert 'СТОП: 5M: впиши' in p.stdout and not os.path.isdir(proj + 'out/'), p.stdout[-2000:] + p.stderr[-2000:]
    env['F1HTF_PACK_SITE_STAMP'] = '20261021_120000'
    p = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'f1h_tf_pack.py')], env=env, capture_output=True,
                       text=True, timeout=1800)
    zs = [f for f in os.listdir(proj + 'out/')] if os.path.isdir(proj + 'out/') else []
    assert len(zs) == 1, p.stdout[-3000:] + p.stderr[-3000:]
    with zipfile.ZipFile(proj + 'out/' + zs[0]) as z:
        names = set(z.namelist())
        assert 'artifacts/pack/site30m_manifest.json' not in names
        for tf in ('5m', '15m'):
            man = json.loads(z.read(f'artifacts/pack/site{tf}_manifest.json'))
            for r in man['files']:
                assert r['path'] in names and r['bytes'] == len(z.read(r['path']))
            assert man['checks']['research_eq_golden']['ok']
        m5 = json.loads(z.read('artifacts/pack/site5m_manifest.json'))
        m15 = json.loads(z.read('artifacts/pack/site15m_manifest.json'))
    assert m5['research']['kind'] == 'compact' and m5['research']['checks']['ok'] and m5['research']['tail_rows'] == 800
    assert m15['research']['kind'] == 'whole' and m15['line21'] == '20261026_101010'
    want = {'artifacts/calib/site_20261021_120000.json', 'artifacts/calib/site_golden_20261021_120000.npz',
            'artifacts/calib/site_live_20261021_100000.json', 'artifacts/logs/site/site_run_20261021_115500.log',
            'artifacts/logs/site/site_live_20261021_095500.log', 'artifacts/calib/tfline21_15m_20261026_101010.json',
            'artifacts/calib/tfline21_15m_golden_20261026_101010.npz', 'artifacts/logs/tf15m/tf_line21_20261026_100000.log'}
    want |= {f'artifacts/bars/btc5m_compact/smc_{k}_BTCUSDT_5M_N200.npz' for k in pk.KINDS}
    want |= {f'artifacts/bars/smc_{k}_BTCUSDT_15M_N200.npz' for k in pk.KINDS}
    assert want <= names, sorted(want - names)
    assert 'не пакуется' in p.stdout                                      # 30M
    print(p.stdout)
