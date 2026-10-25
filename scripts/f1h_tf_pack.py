# -*- coding: utf-8 -*-
"""Colab: one zip with the inputs of f1h on 5M, 15M and 30M for part 2 (export, golden, parity; docs/rules.md R28–R31) — to
be unpacked into the root of the repo, as commit e035bc2 did for 1H.

Per TF:
  15M / 30M — the result of scripts/f1h_tf_line21.py 'line21' (SMC/runs/tf15m/, tf30m/): the tfline21 json (section export)
      and its golden npz, the research files BTC of the TF on which the golden was computed (SMC/data_15m/, data_30m/), all
      logs of the folder. A TF whose json has no export (the line did not pass, 'stop' in the json) is not packed.
  5M — the json and the golden of a 'run' of SuMo_Class_Site v1.1 with the section export (CONFIG['site_stamp']; without it
      the script lists the candidates and stops), the json 'live' it took the formula from, the logs of both, the research
      files BTC 5M of the golden (looked up in CONFIG['research_5m']).
  artifacts/pack/site<tf>_manifest.json — path, bytes, sha256, from (as site1h_manifest.json).
The prod a weights are in the repo already (artifacts/weights/, sha256 in site1h_manifest.json): not packed; the Drive json of
prod a must have their sha256 (the same model).

Research files: whole or compact (CONFIG['compact']; default — compact for 5M: whole they are ~40 MB). Compact (R29): VAL
whole and the last CONFIG['tail_rows'] rows of TRAINING with the 200 rows before them; the values are slices of the whole
files bit for bit, VAL — all arrays as they are; meta gets the new part sizes and checksums and a field 'compact'.

Checks (any fails — stop, no zip):
  (1) the prod a json on Drive = the one in the repo (sha256);
  (2) the source json: 15M / 30M — mode 'line21', tf and export['tf'] = TF, golden written and read back bit for bit; 5M —
      SuMo_Class_Site v1.1 'run', one seed prod a, floor 0.2, golden with the window times;
  (3) the research files are those of the golden: 15M / 30M — checksums of r = the json; every window of the golden ends
      at the time t_end on these files (positions of BTC in the pool = in its own series);
  (4) research on these files = the golden: logit of prod a (training code v6.6, floor 0.2, CPU) on CONFIG['n_sample']
      windows of the golden evenly spread — |Δlogit| <= 1e-4 to logit_a (as R22 (б));
  (5) compact files (R29): read by the training code; window inputs = those of the whole files (r and channels bit for bit,
      history features: VAL bit for bit, the TRAINING tail within 1 ulp float32); windows of VAL the same, of the tail — the
      windows of the whole files that start in it; (4) is repeated on the compact files.

HOW TO RUN (Colab; CPU is enough, ~10 min with 5M): put this file into SMC/ next to f1h_tf_line21.py, sumo_class_train.py
(v6.6) and sumo_class_ceil.py; first run with site_stamp None — it prints the candidate 'run' json of Site; put the stamp
into CONFIG and run again:  %run /content/drive/MyDrive/TrendyCoins/SMC/f1h_tf_pack.py
The zip — SMC/runs/pack/f1h_tf_pack_<stamp>.zip (and a download in the browser); unpack it into the repo root.
"""
import glob
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import zipfile
import zlib

import numpy as np

CONFIG = {
    'project_path': '/content/drive/MyDrive/TrendyCoins/',
    'code_dir': 'SMC/',                     # f1h_tf_line21.py, sumo_class_train.py (v6.6), sumo_class_ceil.py
    'tfs': ('5M', '15M', '30M'),
    'site_stamp': None,                     # 5M: stamp of SMC/runs/site/site_<stamp>.json ('run' Site v1.1 with export)
    'line21_stamp': {'15M': None, '30M': None},   # None — the last tfline21_*.json of the folder
    'research_5m': ('SMC/data/', 'SMC/data_fresh/'),   # where the research files BTC 5M of the golden Site may be
    'compact': {'5M': True, '15M': False, '30M': False},
    'tail_rows': 8640,                      # compact: rows of TRAINING kept (30 days of 5M)
    'n_sample': 2048,                       # check (4): windows of the golden
    'out_dir': 'SMC/runs/pack/',
}
ASSET, N, L, MODEL, SEED = 'BTCUSDT', 200, 64, 'SP_l025_d1', 'a'
PROD_TAG = 'prod_SP_l025_d1_L64_a'
PROD_JSON_SHA256 = '7245459310e0c5fd9a3ae56097f9187a1307fea7443dfc4084bcada165935d9a'   # artifacts/pack/site1h_manifest.json
FLOOR = 0.2
TOL_LOGIT = 1e-4
KINDS = ('series', 'prices', 'range', 'bars')
STAMP = re.compile(r"(\d{8}_\d{6})")
_ENV = 'F1HTF_PACK_'

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


class Stop(Exception):
    pass


def stop(msg):
    raise Stop(msg)


def config():
    """CONFIG with env overrides for runs out of Colab (F1HTF_PACK_PROJECT, _CODE_DIR, _SITE_STAMP, _OUT_DIR, _TFS, _TAIL_ROWS,
       _N_SAMPLE)."""
    c = dict(CONFIG)
    for k, e in (('project_path', 'PROJECT'), ('code_dir', 'CODE_DIR'), ('site_stamp', 'SITE_STAMP'), ('out_dir', 'OUT_DIR')):
        c[k] = os.environ.get(_ENV + e, c[k])
    if os.environ.get(_ENV + 'TFS'):
        c['tfs'] = tuple(os.environ[_ENV + 'TFS'].split(','))
    for k, e in (('tail_rows', 'TAIL_ROWS'), ('n_sample', 'N_SAMPLE')):
        if os.environ.get(_ENV + e):
            c[k] = int(os.environ[_ENV + e])
    return c


def absp(c, p):
    return p if os.path.isabs(p) else c['project_path'] + p


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _import(path, name):
    if not os.path.isfile(path):
        stop(f"нет {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def modules(c):
    """(training code v6.6, Ceil, wrapper) — the training code reads data from the folders set per call."""
    code = absp(c, c['code_dir'])
    os.environ.setdefault('SMC_MODE', 'none')
    os.environ.setdefault('CEIL_MODE', 'none')
    os.environ.setdefault('SMC_PROJECT_PATH', c['project_path'])
    T = _import(code + 'sumo_class_train.py', 'sumo_class_train')
    CL = _import(code + 'sumo_class_ceil.py', 'sumo_class_ceil')
    W = _import(code + 'f1h_tf_line21.py', 'f1h_tf_line21')
    T.RUN_DIR = c['project_path'] + 'SMC/runs/'
    return T, CL, W


# =====================================================================
# research files: read, window times, compact copy
# =====================================================================
def files_of(d, tf):
    return {k: f"{d}smc_{k}_{ASSET}_{tf}_N{N}.npz" for k in KINDS}


def load_npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def window_end_times(T, ser, st):
    """Time of the last bar of windows starting at positions st of the series of one asset (as SeriesData: TRAINING, PAD,
       VALIDATION). Positions outside the parts -> -1."""
    n_tr = len(ser['r_tr'])
    t = np.concatenate([ser['t_tr'], np.full(T.SeriesData.PAD, -1, np.int64), ser['t_va']]).astype(np.int64)
    e = np.asarray(st, np.int64) + L - 1
    out = np.full(len(e), -1, np.int64)
    ok = (e >= 0) & (e < len(t))
    out[ok] = t[e[ok]]
    return out, n_tr


def crc_of(ser):
    return {k: zlib.crc32(ser[f'r_{k}'].astype(np.float32).tobytes()) for k in ('tr', 'va')}


def compact(src, dst, tf, tail):
    """Compact research files of one asset: VAL whole, TRAINING — its last `tail` rows (with the N rows before them for the
       bars). Values — slices of the whole files; meta — new sizes, checksums, 'compact'. Returns the row of the cut."""
    F = {k: load_npz(p) for k, p in files_of(src, tf).items()}
    n_tr = len(F['series']['r_tr'])
    lo = max(n_tr - tail, 0)
    ser = {}
    for k, v in F['series'].items():
        if k.startswith(('probe_', 'bank_')):                  # SMC_Data extras of TRAINING positions (not F1 inputs)
            continue
        ser[k] = v[lo:] if k in ('r_tr', 't_tr', 'close_tr') else v
    ser['fin_tr'] = np.where(np.isfinite(ser['r_tr']))[0].astype(F['series']['fin_tr'].dtype)
    crc = crc_of(ser)
    meta = json.loads(str(F['series']['meta']))
    tr = dict(meta.get('parts', {}).get('TRAINING', {}))
    tr.update({'n': int(n_tr - lo), 'from': str(np.datetime64(int(ser['t_tr'][0]), 'ns')),
               'n_nan': int((~np.isfinite(ser['r_tr'])).sum())})
    meta.setdefault('parts', {})['TRAINING'] = tr
    info = {'from': f"smc_*_{ASSET}_{tf}_N{N}.npz (whole)", 'tail_rows': int(n_tr - lo), 'cut_row': int(lo),
            'n_tr_whole': int(n_tr), 'crc_whole': meta.get('crc'),
            'what': 'VALIDATION whole; TRAINING — the last rows of the whole file (values are its slices bit for bit, '
                    'smc_bars: the N rows before them); probe_* / bank_* dropped (not inputs of F1)'}
    meta['crc'], meta['compact'] = crc, info
    ser['meta'] = np.array(json.dumps(meta, ensure_ascii=False))
    out = {'series': ser}
    pri = dict(F['prices'])
    for k in list(pri):
        if k.startswith(('e_', 'd_')) and k.endswith('_tr'):
            pri[k] = F['prices'][k][lo:]
    for k in [k for k in pri if k.startswith('d0_') and k.endswith('_tr')]:
        if lo > 0:
            nm = k[3:-3]
            pri[k] = np.array(np.float32(F['prices'][f'd_{nm}_tr'][lo - 1]))
    out['prices'] = pri
    rng = dict(F['range'])
    rng['g_tr'] = F['range']['g_tr'][lo:]
    out['range'] = rng
    bars = dict(F['bars'])
    for k in 'ohlc':
        ext = np.concatenate([F['bars'][f'pre_{k}_tr'], F['bars'][f'{k}_tr']])
        n_pre = len(F['bars'][f'pre_{k}_tr'])
        bars[f'{k}_tr'] = F['bars'][f'{k}_tr'][lo:]
        bars[f'pre_{k}_tr'] = ext[lo:lo + n_pre]
    out['bars'] = bars
    for kind in ('prices', 'range', 'bars'):
        m = json.loads(str(F[kind]['meta']))
        m['crc'], m['compact'] = crc, {'tail_rows': int(n_tr - lo), 'cut_row': int(lo), 'crc_whole': m.get('crc')}
        out[kind]['meta'] = np.array(json.dumps(m, ensure_ascii=False))
    os.makedirs(dst, exist_ok=True)
    for kind, p in files_of(dst, tf).items():
        np.savez_compressed(p, **out[kind])
    return lo


def series(T, d, tf):
    """SeriesData of BTC from folder d with the history columns of F1 (as SuMo_Class_TF series_data)."""
    T.DATA_DIR, T.DATA_OVERLAY = d, None
    T.NT_KEEP = tuple(sorted(set(T._pool_nt_cols(MODEL))))
    try:
        return T.SeriesData([ASSET], tf, N, '2100-01-01', 'multi_g')
    finally:
        T._reset_nt_keep()


def check_compact(T, F, comp, tf, lo):
    """(5) of the docstring: positions of the compact series -> positions of the whole one F; inputs and windows equal."""
    C = series(T, comp, tf)
    sf, sc = F.series[0], C.series[0]
    pos_c = np.concatenate([sc['off']['tr'] + np.arange(sc['n']['tr']), sc['off']['va'] + np.arange(sc['n']['va'])])
    pos_f = np.concatenate([sf['off']['tr'] + lo + np.arange(sc['n']['tr']), sf['off']['va'] + np.arange(sc['n']['va'])])
    if sc['n']['va'] != sf['n']['va'] or sf['n']['tr'] - lo != sc['n']['tr']:
        stop(f"{tf}: компактные части не той длины")
    out = {'positions': int(len(pos_c)), 'time_same': bool(np.array_equal(C.t[pos_c], F.t[pos_f]))}
    arrs = [('r', C.r, F.r), ('rc', C.rc, F.rc), ('g', C.chan_g, F.chan_g)]
    arrs += [(f'e{j}', a, b) for j, (a, b) in enumerate(zip(C.chan_e, F.chan_e))]
    arrs += [(f'd{j}', a, b) for j, (a, b) in enumerate(zip(C.chan_d, F.chan_d))]
    out['inputs_bit'] = all(np.asarray(a)[pos_c].tobytes() == np.asarray(b)[pos_f].tobytes() for _, a, b in arrs)
    nc, nf = C.nt_t.cpu().numpy(), F.nt_t.cpu().numpy()
    va_c, va_f = pos_c[sc['n']['tr']:], pos_f[sc['n']['tr']:]
    tr_c, tr_f = pos_c[:sc['n']['tr']], pos_f[:sc['n']['tr']]
    out['nt_val_bit'] = bool(nc[:, va_c].tobytes() == nf[:, va_f].tobytes())
    a, b = nc[:, tr_c].astype(np.float64), nf[:, tr_f].astype(np.float64)
    ulp = np.spacing(np.abs(nf[:, tr_f])).astype(np.float64)
    out['nt_tail_max_ulp'] = float(np.max(np.abs(a - b) / ulp)) if a.size else 0.0
    out['nt_tail_n_diff'] = int((a != b).sum())
    wc, wf = C.windows(L)['st'], F.windows(L)['st']
    m = np.full(len(C.t), -1, np.int64)
    m[pos_c] = pos_f
    wc_f = m[wc]
    wf_keep = wf[((wf >= sf['off']['tr'] + lo) & (wf < sf['off']['tr'] + sf['n']['tr'])) | (wf >= sf['off']['va'])]
    out['windows'] = int(len(wc))
    out['windows_same'] = bool(np.array_equal(np.sort(wc_f), np.sort(wf_keep)))
    out['ok'] = bool(out['time_same'] and out['inputs_bit'] and out['nt_val_bit'] and out['nt_tail_max_ulp'] <= 1.0
                     and out['windows_same'])
    return out, C


def research_logits(T, CL, model, data, st):
    """Logit of prod a with the floor 0.2 on windows starting at st (as scripts/f1h_research.logits)."""
    import torch
    out = []
    with CL.floor_override(data, L, FLOOR) as wf, torch.no_grad():
        for i in range(0, len(st), 1024):
            s = torch.from_numpy(np.asarray(st[i:i + 1024], np.int64)).to(T.DEVICE)
            W = data.gw(s, L)
            fl = wf['floor_t'][torch.zeros(len(s), dtype=torch.long, device=wf['floor_t'].device)]
            out.append(T.score(model, T.to_shapes(W, fl), S=T.side_feats(data, s, L, W, fl)))
    return np.concatenate(out)


def sample_check(T, CL, model, data, st, logit_a, n):
    """(4): research logit on n windows of the golden evenly spread against logit_a (float32 of the golden)."""
    idx = np.unique(np.linspace(0, len(st) - 1, min(n, len(st))).astype(np.int64))
    lg = research_logits(T, CL, model, data, st[idx])
    d = np.abs(lg - np.asarray(logit_a, np.float64)[idx])
    return {'n': int(len(idx)), 'max': float(d.max()), 'p99': float(np.percentile(d, 99)), 'ok': bool(d.max() <= TOL_LOGIT)}


# =====================================================================
# sources per TF
# =====================================================================
def stamp_of(p):
    m = STAMP.search(os.path.basename(p))
    return m.group(1) if m else ''


def line21_source(c, W, tf):
    """15M / 30M: (json path, json) of the wrapper folder — CONFIG stamp or the last tfline21_*.json; (None, why) — not
       packed."""
    out = c['project_path'] + W.TFS[tf]['out']
    cands = sorted(p for p in glob.glob(out + 'tfline21_*.json') if re.fullmatch(r"tfline21_\d{8}_\d{6}\.json",
                                                                                   os.path.basename(p)))
    print(f"\n📂 {tf}: {out} — json 'line21': {len(cands)}")
    for p in cands:
        with open(p, encoding='utf-8') as fh:
            j = json.load(fh)
        ex = j.get('export') or {}
        print(f"   {stamp_of(p)}: tf {j.get('tf')}, " + (f"стоп — {j['stop']}" if j.get('stop') else
              f"экспорт: живая часть {(ex.get('live') or {}).get('key')}, потолок {ex.get('ceiling')}, golden "
              f"{(j.get('golden') or {}).get('n')} окон"))
    want = c['line21_stamp'].get(tf)
    pick = [p for p in cands if stamp_of(p) == want] if want else cands[-1:]
    if not pick:
        return None, f"нет json 'line21'{' со штампом ' + want if want else ''}"
    with open(pick[0], encoding='utf-8') as fh:
        j = json.load(fh)
    if j.get('stop') or not j.get('export'):
        return None, f"линия не прошла ({j.get('stop') or 'нет раздела export'}) — ТФ не пакуется (R27)"
    return pick[0], j


def site_candidates(c):
    d = c['project_path'] + 'SMC/runs/site/'
    rows = []
    for p in sorted(glob.glob(d + 'site_*.json')):
        if not re.fullmatch(r"site_\d{8}_\d{6}\.json", os.path.basename(p)):
            continue
        with open(p, encoding='utf-8') as fh:
            j = json.load(fh)
        rows.append((p, j))
    print(f"\n📂 5M: {d} — json SuMo_Class_Site: {len(rows)}")
    for p, j in rows:
        ex = j.get('export') or {}
        g = j.get('golden') or {}
        print(f"   {stamp_of(p)}: v{j.get('ver')} '{j.get('mode')}', формула {(j.get('formula') or {}).get('key')} "
              f"({(j.get('formula') or {}).get('from')}), " + (f"экспорт: {ex.get('config')}, пол {ex.get('floor')}, потолок "
                                                              f"{ex.get('ceiling')}, golden {g.get('n')} окон, время окон "
                                                              f"{'есть' if g.get('t_end') else 'нет'}" if ex else
                                                              "без раздела export"))
    return rows


# =====================================================================
# pack
# =====================================================================
class Pack:
    def __init__(self, c, stage):
        self.c, self.stage, self.by_tf = c, stage, {}

    def add(self, tf, src, rel, how=None):
        """Copy (or take an already written) file into stage/rel; the manifest row of the TF."""
        dst = self.stage + rel
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if src != dst:
            shutil.copyfile(src, dst)
        frm = how or os.path.relpath(src, self.c['project_path'])
        self.by_tf.setdefault(tf, []).append({'path': rel, 'bytes': os.path.getsize(dst), 'sha256': sha256(dst),
                                              'from': frm})

    def manifest(self, tf, head):
        rel = f"artifacts/pack/site{tf.lower()}_manifest.json"
        m = {**head, 'files': self.by_tf[tf], 'created': time.strftime('%Y-%m-%d %H:%M:%S')}
        os.makedirs(os.path.dirname(self.stage + rel), exist_ok=True)
        with open(self.stage + rel, 'w', encoding='utf-8') as fh:
            json.dump(m, fh, ensure_ascii=False, indent=1)
        return rel


def golden_of(p):
    with np.load(p, allow_pickle=False) as z:
        g = {k: z[k] for k in z.files}
    g['meta_d'] = json.loads(str(g['meta']))
    return g


def research_dir_check(T, d, tf, g, n_va_want=None):
    """(3): every window of the golden g ends at t_end on the research files of folder d. (ok, info)."""
    ps = files_of(d, tf)
    if not all(os.path.isfile(p) for p in ps.values()):
        return False, {'dir': d, 'why': 'нет файлов'}
    ser = load_npz(ps['series'])
    te, n_tr = window_end_times(T, ser, g['st'])
    ok = bool(np.array_equal(te, np.asarray(g['t_end'], np.int64)))
    return ok, {'dir': d, 'n_tr': int(n_tr), 'n_va': int(len(ser['r_va'])), 'windows': int(len(te)),
                'same_t_end': ok, 'crc': crc_of(ser)}


def pack_tf(c, T, CL, W, model, P, tf, tmp):
    """One TF into the pack. Returns the manifest head or None (not packed: why printed)."""
    tl = tf.lower()
    research = {}
    if tf == '5M':
        rows = site_candidates(c)
        want = c['site_stamp']
        pick = [(p, j) for p, j in rows if stamp_of(p) == want]
        if not want or not pick:
            stop("5M: впиши в CONFIG['site_stamp'] штамп json Site v1.1 'run' с разделом export (список выше) и запусти снова")
        jp, j = pick[0]
        ex = j.get('export') or {}
        gi = j.get('golden') or {}
        bad = [k for k, ok in (('ver 1.1', j.get('ver') == '1.1'), ("mode 'run'", j.get('mode') == 'run'),
                               ('export', bool(ex)), ('один сид prod a', ex.get('config') == 'single' and
                                                     ex.get('seeds') == [SEED]), ('пол 0.2', ex.get('floor') == FLOOR),
                               ('golden', bool(gi.get('same')) and bool(gi.get('t_end')))) if not ok]
        if bad:
            stop(f"5M: json {os.path.basename(jp)} — не то: {', '.join(bad)}")
        gp = os.path.dirname(jp) + '/' + gi['path']
        g = golden_of(gp)
        if 'st' not in g or 't_end' not in g or f'logit_{SEED}' not in g:
            stop(f"5M: в golden {gi['path']} нет st / t_end / logit_{SEED}")
        found = []
        for d in c['research_5m']:
            ok, info = research_dir_check(T, absp(c, d), '5M', g)
            print(f"   research 5M {d}: " + ('окна golden — на этих файлах ✅' if ok else f"не те ({info.get('why') or 'время окон'})"))
            if ok:
                found.append(info)
        if not found:
            stop("5M: файлов research BTC 5M golden Site нет ни в одной папке CONFIG['research_5m']")
        research = found[0]
        src_dir = research['dir']
        P.add(tf, jp, f"artifacts/calib/{os.path.basename(jp)}")
        P.add(tf, gp, f"artifacts/calib/{gi['path']}")
        lf = (j.get('formula') or {}).get('from') or ''
        lp = os.path.dirname(jp) + '/' + lf
        live_stamp = None
        if lf.endswith('.json') and os.path.isfile(lp):
            P.add(tf, lp, f"artifacts/calib/{lf}")
            live_stamp = stamp_of(lf)
        logs = os.path.dirname(jp) + '/logs/'
        for kind, s in (('run', stamp_of(jp)), ('live', live_stamp)):
            ls = sorted(p for p in glob.glob(logs + f"site_{kind}_*.log") if s and stamp_of(p) <= s)
            if ls:
                P.add(tf, ls[-1], f"artifacts/logs/site/{os.path.basename(ls[-1])}")
            else:
                print(f"   ❗ лог Site '{kind}' к {s} не найден в {logs}")
        head = {'what': "вход бэкенда сайта на 5M: итог SuMo_Class_Site v1.1 'run' — линия M F1 SP_l025_d1 prod a "
                        "(экспорт: потолок как проверено; до 8 на сайте режет генератор)",
                'tf': '5M', 'site': stamp_of(jp), 'prod': PROD_TAG,
                'export': {'tf': '5M', 'rules': 'SuMo_Class_Site v1.1', 'floor': ex.get('floor'), 'ceiling': ex.get('ceiling'),
                           'marks': ex.get('marks'), 'live': (ex.get('live') or {}).get('key')},
                'live': (ex.get('live') or {}).get('key')}
    else:
        jp, j = line21_source(c, W, tf)
        if jp is None:
            print(f"   ⏭️ {tf}: {j}")
            return None
        ex, gi = j['export'], j.get('golden') or {}
        bad = [k for k, ok in (("mode 'line21'", j.get('mode') == 'line21'), ('ver 2.1', j.get('ver') == '2.1'),
                               (f'tf {tf}', j.get('tf') == tf and ex.get('tf') == tf), ('пол 0.2', ex.get('floor') == FLOOR),
                               ('golden (4)', bool(gi.get('same')))) if not ok]
        if bad:
            stop(f"{tf}: json {os.path.basename(jp)} — не то: {', '.join(bad)}")
        gp = os.path.dirname(jp) + '/' + gi['path']
        g = golden_of(gp)
        if g['meta_d'].get('tf') != tf:
            stop(f"{tf}: golden {gi['path']} — ТФ {g['meta_d'].get('tf')}")
        src_dir = c['project_path'] + W.TFS[tf]['data']
        ok, research = research_dir_check(T, src_dir, tf, g)
        crc_json = {k: int(v) for k, v in ((j.get('data') or {}).get(ASSET) or {}).get('crc', {}).items()}
        if not ok or research.get('crc') != crc_json:
            stop(f"{tf}: файлы research BTC в {src_dir} — не те, на которых считался golden (контрольные суммы "
                 f"{research.get('crc')} против json {crc_json}; время окон {'да' if ok else 'нет'}) — 'line21' запускался "
                 f"позже? пакуй по последнему json")
        print(f"   research {tf}: {src_dir} — контрольные суммы = json, окна golden на этих файлах ✅")
        s = stamp_of(jp)
        P.add(tf, jp, f"artifacts/calib/tfline21_{tl}_{s}.json")
        P.add(tf, gp, f"artifacts/calib/tfline21_{tl}_golden_{s}.npz")
        for lp in sorted(glob.glob(os.path.dirname(jp) + '/logs/*.log')):
            P.add(tf, lp, f"artifacts/logs/tf{tl}/{os.path.basename(lp)}")
        head = {'what': f"вход бэкенда сайта на {tf}: итог 'line21' обёртки scripts/f1h_tf_line21.py (SuMo_Class_TF v2.1) — "
                        f"линия M F1 SP_l025_d1 prod a",
                'tf': tf, 'line21': s, 'prod': PROD_TAG,
                'export': {'tf': tf, 'rules': ex.get('rules'), 'floor': ex.get('floor'), 'ceiling': ex.get('ceiling'),
                           'marks': ex.get('marks')},
                'live': (ex.get('live') or {}).get('key')}
    t_end = np.asarray(g['t_end'], np.int64)
    head['golden'] = {'asset': ASSET, 'tf': tf, 'n': int(len(t_end)), 'shown': int(np.asarray(g['show']).sum()),
                      'from': str(np.datetime64(int(t_end[0]), 'ns')), 'to': str(np.datetime64(int(t_end[-1]), 'ns'))}
    # (4) research on these files = golden
    data = series(T, src_dir, tf)
    sc = sample_check(T, CL, model, data, np.asarray(g['st'], np.int64), g[f'logit_{SEED}'], c['n_sample'])
    print(f"   (4) research на этих файлах = golden: {sc['n']} окон, max |Δlogit| {sc['max']:.1e} (p99 {sc['p99']:.1e}) "
          f"{'✅' if sc['ok'] else '❌'}")
    if not sc['ok']:
        stop(f"{tf}: research на файлах не воспроизводит logit_{SEED} golden")
    head['checks'] = {'research_files': {k: v for k, v in research.items() if k != 'dir'}, 'research_eq_golden': sc}
    from_rel = os.path.relpath(src_dir, c['project_path']) + '/'
    if c['compact'].get(tf):
        cdir = tmp + f"compact_{tl}/"
        lo = compact(src_dir, cdir, tf, c['tail_rows'])
        chk, C = check_compact(T, data, cdir, tf, lo)
        n_tr_c = C.series[0]['n']['tr']
        st_c = np.asarray(g['st'], np.int64) - (research['n_tr'] - n_tr_c)
        sc2 = sample_check(T, CL, model, C, st_c, g[f'logit_{SEED}'], c['n_sample'])
        chk['research_eq_golden'] = sc2
        print(f"   (5) компактно: строк TRAINING {n_tr_c} (с {lo}), окон {chk['windows']}: время {chk['time_same']}, входы "
              f"бит в бит {chk['inputs_bit']}, история VAL бит в бит {chk['nt_val_bit']}, хвост TRAINING — max "
              f"{chk['nt_tail_max_ulp']:.0f} ulp ({chk['nt_tail_n_diff']} отличий), окна те же {chk['windows_same']}; "
              f"research = golden {sc2['max']:.1e} -> {'✅' if chk['ok'] and sc2['ok'] else '❌'}")
        if not (chk['ok'] and sc2['ok']):
            stop(f"{tf}: компактные файлы не дают research тех же входов — CONFIG['compact']['{tf}'] = False (целиком)")
        del C, data
        for kind, p in files_of(cdir, tf).items():
            P.add(tf, p, f"artifacts/bars/btc{tl}_compact/{os.path.basename(p)}",
                  f"{from_rel}{os.path.basename(p)} (компактно: VAL целиком + последние {n_tr_c} строк TRAINING)")
        head['research'] = {'kind': 'compact', 'tail_rows': int(n_tr_c), 'cut_row': int(lo), 'checks': chk}
    else:
        del data
        for kind, p in files_of(src_dir, tf).items():
            P.add(tf, p, f"artifacts/bars/{os.path.basename(p)}")
        head['research'] = {'kind': 'whole'}
    head['weights'] = {'manifest': 'artifacts/pack/site1h_manifest.json', 'prod_json_sha256': PROD_JSON_SHA256,
                       'note': 'веса prod a — в репо (artifacts/weights/); json prod a на Drive — тот же (sha256)'}
    return head


def main():
    c = config()
    if IN_COLAB:
        drive.mount('/content/drive')
    t0 = time.time()
    stamp = time.strftime('%Y%m%d_%H%M%S')
    tmp = f"/tmp/f1h_tf_pack_{stamp}/"
    stage = tmp + 'stage/'
    out_dir = absp(c, c['out_dir'])
    print(f"📦 f1h_tf_pack: ТФ {', '.join(c['tfs'])} -> {out_dir}f1h_tf_pack_{stamp}.zip")
    try:
        prod = c['project_path'] + f"SMC/runs/prod/{PROD_TAG}.json"
        if not os.path.isfile(prod) or sha256(prod) != PROD_JSON_SHA256:
            stop(f"(1) {prod} — не тот prod a, что в репо (sha256 site1h_manifest.json)")
        print(f"   (1) prod a на Drive = в репо (sha256) ✅")
        T, CL, W = modules(c)
        model, _ = T.load_pool_model('ALL', SEED, MODEL, mode='prod')
        P = Pack(c, stage)
        heads = {}
        for tf in c['tfs']:
            h = pack_tf(c, T, CL, W, model, P, tf, tmp)
            if h is not None:
                heads[tf] = P.manifest(tf, h)
        if not heads:
            stop("нечего паковать")
        os.makedirs(out_dir, exist_ok=True)
        zp = out_dir + f"f1h_tf_pack_{stamp}.zip"
        with zipfile.ZipFile(zp, 'w') as z:
            for d, _, fs in os.walk(stage):
                for f in sorted(fs):
                    p = os.path.join(d, f)
                    rel = os.path.relpath(p, stage)
                    z.write(p, rel, compress_type=zipfile.ZIP_STORED if f.endswith('.npz') else zipfile.ZIP_DEFLATED)
        print(f"\n🏁 упаковано: {', '.join(heads)} | {os.path.getsize(zp) / 2 ** 20:.1f} MB | {(time.time() - t0) / 60:.1f} мин")
        for tf, rows in P.by_tf.items():
            print(f"   {tf}: " + ', '.join(f"{r['path']} ({r['bytes'] / 2 ** 20:.2f} MB)" for r in rows))
        print(f"💾 {zp} — распакуй в корень репо")
        if IN_COLAB:
            try:
                from google.colab import files
                files.download(zp)
            except Exception as e:
                print(f"⚠️ скачивание не началось ({type(e).__name__}: {e}) — возьми zip с Drive")
            drive.flush_and_unmount()
        return zp
    except Stop as e:
        print(f"\n🍆 СТОП: {e}")
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
