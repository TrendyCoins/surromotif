# -*- coding: utf-8 -*-
"""F1h_TF_Pack — один zip со входом f1h на 5M, 15M и 30M для части 2 (экспорт, golden, паритет; docs/rules.md R28–R31):
распаковать в корень репо, как коммит e035bc2 у 1H (2026-10-25)

ЧТО КЛАДЁТ. По каждому ТФ:
  15M / 30M — итог 'line21' блокнота f1h_tf_line21 (SMC/runs/tf15m/, tf30m/): json tfline21 (раздел export) и его golden npz,
      файлы research BTC этого ТФ, на которых считался golden (SMC/data_15m/, data_30m/), все логи папки. ТФ, у которого в json
      нет раздела export (линия не прошла, в json — 'stop'), не пакуется.
  5M — json и golden 'run' SuMo_Class_Site v1.1 с разделом export (штамп — SITE_STAMP; без него блокнот показывает кандидатов
      и останавливается), json 'live', из которого взята формула, логи обоих, файлы research BTC 5M golden (ищутся в
      RESEARCH_5M).
  artifacts/pack/site<tf>_manifest.json — path, bytes, sha256, from (как site1h_manifest.json).
Веса prod a уже в репо (artifacts/weights/, sha256 — в site1h_manifest.json): не пакуются; json prod a на Drive обязан иметь их
sha256 (та же модель).

ФАЙЛЫ RESEARCH — целиком или компактно (COMPACT; по умолчанию компактно только 5M: целиком у него ~40 MB). Компактно (R29):
VAL целиком и последние TAIL_ROWS строк TRAINING с 200 строками перед ними; значения — срезы полных файлов бит в бит, у VAL —
все массивы как есть; в meta — новые длины частей, контрольные суммы и поле 'compact'.

ПРОВЕРКИ (не сошлось — стоп, zip не пишется):
  (1) json prod a на Drive = тот, что в репо (sha256);
  (2) json источника: 15M / 30M — mode 'line21', tf и export['tf'] = ТФ, golden записан и прочитан бит в бит; 5M — Site v1.1
      'run', один сид prod a, пол 0.2, golden со временем окон;
  (3) файлы research — те, на которых считался golden: 15M / 30M — контрольные суммы r = json; каждое окно golden кончается в
      момент t_end на этих файлах (позиции BTC в пуле = в его ряде);
  (4) research на этих файлах = golden: logit prod a (код обучения v6.6, пол 0.2) на N_SAMPLE окнах golden, равномерно —
      |Δlogit| <= 1e-4 к logit_a (как R22 (б));
  (5) компактные файлы (R29 и поправка к нему): код обучения их читает; входы окон = у полных файлов (r и каналы бит в бит,
      признаки истории: VAL бит в бит, хвост TRAINING — не больше 1 ulp float32); окна VAL те же, хвоста — окна полных файлов,
      начинающиеся в нём; (4) повторяется на компактных файлах.

ЗАПУСК (Colab). Этот файл — блокнот: открой scripts/f1h_tf_pack.ipynb (Файл -> Загрузить блокнот) или вставь файл целиком в
одну ячейку нового блокнота. Нужны SMC/sumo_class_train.py (v6.6) и SMC/sumo_class_ceil.py (они уже на Drive), прогоны
f1h_tf_line21 (15M, 30M) и Site v1.1 'run'. Среда — CPU (~5–10 мин). Первый запуск с SITE_STAMP = None печатает json Site —
впиши штамп последнего v1.1 'run' с разделом export и запусти ячейку снова. Итог — SMC/runs/pack/f1h_tf_pack_<штамп>.zip (и
скачивание в браузере), лог — SMC/runs/pack/logs/; zip распаковать в корень репо.
"""

import os
import re
import sys
import glob
import json
import time
import zlib
import shutil
import hashlib
import zipfile
import contextlib
import importlib.util
import numpy as np

try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


# =====================================================================
# НАСТРОЙКИ
# =====================================================================
PACK_TFS = ('5M', '15M', '30M')                        # что пакуем
SITE_STAMP = None                                      # 5M: штамп SMC/runs/site/site_<штамп>.json ('run' Site v1.1 с export)
LINE21_STAMP = {'15M': None, '30M': None}              # None — последний tfline21_*.json папки ТФ
COMPACT = {'5M': True, '15M': False, '30M': False}     # файлы research: компактно (VAL + хвост TRAINING) или целиком
TAIL_ROWS = 8640                                       # компактно: строк TRAINING (30 суток 5M)
N_SAMPLE = 2048                                        # проверка (4): окон golden

project_path = '/content/drive/MyDrive/TrendyCoins/'
CODE_DIR = project_path + 'SMC/'                       # sumo_class_train.py (v6.6), sumo_class_ceil.py
RESEARCH_5M = ('SMC/data/', 'SMC/data_fresh/')         # где могут лежать файлы research BTC 5M golden Site
OUT_DIR = project_path + 'SMC/runs/pack/'
# Папки 15M / 30M — как TFS в блокноте f1h_tf_line21 ('data' — файлы research, 'out' — итоги 'line21'; тест сверяет).
FOLDERS = {'15M': {'data': 'SMC/data_15m/', 'out': 'SMC/runs/tf15m/'},
           '30M': {'data': 'SMC/data_30m/', 'out': 'SMC/runs/tf30m/'}}

ASSET, N, L, MODEL, SEED = 'BTCUSDT', 200, 64, 'SP_l025_d1', 'a'
PROD_TAG = 'prod_SP_l025_d1_L64_a'
PROD_JSON_SHA256 = '7245459310e0c5fd9a3ae56097f9187a1307fea7443dfc4084bcada165935d9a'   # artifacts/pack/site1h_manifest.json
FLOOR = 0.2
TOL_LOGIT = 1e-4
KINDS = ('series', 'prices', 'range', 'bars')
STAMP = re.compile(r"(\d{8}_\d{6})")
_ENV = 'F1HTF_PACK_'                                   # переменные окружения — для теста вне Colab
if os.environ.get(_ENV + 'TFS'):
    PACK_TFS = tuple(os.environ[_ENV + 'TFS'].split(','))
SITE_STAMP = os.environ.get(_ENV + 'SITE_STAMP', SITE_STAMP)
TAIL_ROWS = int(os.environ.get(_ENV + 'TAIL_ROWS', TAIL_ROWS))
N_SAMPLE = int(os.environ.get(_ENV + 'N_SAMPLE', N_SAMPLE))
project_path = os.environ.get(_ENV + 'PROJECT', project_path)
CODE_DIR = os.environ.get(_ENV + 'CODE_DIR', project_path + 'SMC/')
OUT_DIR = os.environ.get(_ENV + 'OUT_DIR', project_path + 'SMC/runs/pack/')


class Stop(Exception):
    pass


def stop(msg):
    raise Stop(msg)


def config():
    """Настройки одним словарём (их читают функции ниже)."""
    return {'project_path': project_path, 'code_dir': CODE_DIR, 'tfs': tuple(PACK_TFS), 'site_stamp': SITE_STAMP,
            'line21_stamp': dict(LINE21_STAMP), 'research_5m': tuple(RESEARCH_5M), 'compact': dict(COMPACT),
            'tail_rows': int(TAIL_ROWS), 'n_sample': int(N_SAMPLE), 'out_dir': OUT_DIR}


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
    """Код обучения v6.6 и Ceil (floor_override) из CODE_DIR; данные код обучения читает из папок, заданных при вызове."""
    os.environ.setdefault('SMC_MODE', 'none')
    os.environ.setdefault('CEIL_MODE', 'none')
    os.environ.setdefault('SMC_PROJECT_PATH', c['project_path'])
    T = _import(c['code_dir'] + 'sumo_class_train.py', 'sumo_class_train')
    CL = _import(c['code_dir'] + 'sumo_class_ceil.py', 'sumo_class_ceil')
    T.RUN_DIR = c['project_path'] + 'SMC/runs/'
    return T, CL


@contextlib.contextmanager
def _tee(path):
    """Лог в файл и на экран (как у research)."""
    class Tee:
        def __init__(self, fh, keep):
            self.fh, self.keep = fh, keep

        def write(self, s):
            self.keep.write(s)
            self.fh.write(s)

        def flush(self):
            self.keep.flush()
            self.fh.flush()

        def __getattr__(self, name):
            return getattr(self.keep, name)
    keep = sys.stdout
    with open(path, 'w', encoding='utf-8') as fh:
        sys.stdout = Tee(fh, keep)
        try:
            yield
        finally:
            sys.stdout = keep


# =====================================================================
# ФАЙЛЫ RESEARCH: чтение, время окон, компактная копия
# =====================================================================
def files_of(d, tf):
    return {k: f"{d}smc_{k}_{ASSET}_{tf}_N{N}.npz" for k in KINDS}


def load_npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def window_end_times(T, ser, st):
    """Время последнего бара окон, начинающихся на позициях st ряда одного ассета (как SeriesData: TRAINING, PAD,
       VALIDATION). Позиции вне частей -> −1."""
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
    """Компактные файлы research одного ассета: VAL целиком, TRAINING — последние tail строк (у smc_bars — с N строками перед
       ними). Значения — срезы полных файлов; в meta — новые длины, контрольные суммы, 'compact'. Возвращает строку среза."""
    F = {k: load_npz(p) for k, p in files_of(src, tf).items()}
    n_tr = len(F['series']['r_tr'])
    lo = max(n_tr - tail, 0)
    ser = {}
    for k, v in F['series'].items():
        if k.startswith(('probe_', 'bank_')):                  # дополнения SMC_Data по позициям TRAINING (не входы F1)
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
    """SeriesData BTC из папки d со столбцами истории F1 (как series_data SuMo_Class_TF)."""
    T.DATA_DIR, T.DATA_OVERLAY = d, None
    T.NT_KEEP = tuple(sorted(set(T._pool_nt_cols(MODEL))))
    try:
        return T.SeriesData([ASSET], tf, N, '2100-01-01', 'multi_g')
    finally:
        T._reset_nt_keep()


def check_compact(T, F, comp, tf, lo):
    """Проверка (5): позиции компактного ряда -> позиции полного F; входы и окна равны."""
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
    """Logit prod a с полом 0.2 на окнах с началами st (как scripts/f1h_research.logits в репо)."""
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
    """Проверка (4): logit research на n окнах golden, равномерно, против logit_a (float32 golden)."""
    idx = np.unique(np.linspace(0, len(st) - 1, min(n, len(st))).astype(np.int64))
    lg = research_logits(T, CL, model, data, st[idx])
    d = np.abs(lg - np.asarray(logit_a, np.float64)[idx])
    return {'n': int(len(idx)), 'max': float(d.max()), 'p99': float(np.percentile(d, 99)), 'ok': bool(d.max() <= TOL_LOGIT)}


# =====================================================================
# ИСТОЧНИКИ ПО ТФ
# =====================================================================
def stamp_of(p):
    m = STAMP.search(os.path.basename(p))
    return m.group(1) if m else ''


def line21_source(c, tf):
    """15M / 30M: (путь json, json) из папки 'line21' ТФ — штамп LINE21_STAMP или последний tfline21_*.json; (None, почему)
       — не пакуется."""
    out = c['project_path'] + FOLDERS[tf]['out']
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
# УПАКОВКА
# =====================================================================
class Pack:
    def __init__(self, c, stage):
        self.c, self.stage, self.by_tf = c, stage, {}

    def add(self, tf, src, rel, how=None):
        """Файл — в stage/rel (копией или уже записанный); строка описи ТФ."""
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
    """Проверка (3): каждое окно golden g кончается в момент t_end на файлах research папки d. (ok, сводка)."""
    ps = files_of(d, tf)
    if not all(os.path.isfile(p) for p in ps.values()):
        return False, {'dir': d, 'why': 'нет файлов'}
    ser = load_npz(ps['series'])
    te, n_tr = window_end_times(T, ser, g['st'])
    ok = bool(np.array_equal(te, np.asarray(g['t_end'], np.int64)))
    return ok, {'dir': d, 'n_tr': int(n_tr), 'n_va': int(len(ser['r_va'])), 'windows': int(len(te)),
                'same_t_end': ok, 'crc': crc_of(ser)}


def pack_tf(c, T, CL, model, P, tf, tmp):
    """Один ТФ в упаковку. Возвращает шапку описи или None (не пакуется — причина в логе)."""
    tl = tf.lower()
    research = {}
    if tf == '5M':
        rows = site_candidates(c)
        want = c['site_stamp']
        pick = [(p, j) for p, j in rows if stamp_of(p) == want]
        if not want or not pick:
            stop("5M: впиши в НАСТРОЙКАХ SITE_STAMP = '<штамп>' — json Site v1.1 'run' с разделом export (список выше) — и запусти ячейку снова")
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
            stop("5M: файлов research BTC 5M golden Site нет ни в одной папке RESEARCH_5M")
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
        jp, j = line21_source(c, tf)
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
        src_dir = c['project_path'] + FOLDERS[tf]['data']
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
            stop(f"{tf}: компактные файлы не дают research тех же входов — в НАСТРОЙКАХ COMPACT['{tf}'] = False (целиком)")
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


# =====================================================================
# ЗАПУСК
# =====================================================================
def run(c, stamp):
    """Проверки и zip. Путь к zip или None (стоп — причина в логе)."""
    t0 = time.time()
    tmp = f"/tmp/f1h_tf_pack_{stamp}/"
    stage = tmp + 'stage/'
    out_dir = absp(c, c['out_dir'])
    print(f"📦 f1h_tf_pack: ТФ {', '.join(c['tfs'])} -> {out_dir}f1h_tf_pack_{stamp}.zip")
    try:
        prod = c['project_path'] + f"SMC/runs/prod/{PROD_TAG}.json"
        if not os.path.isfile(prod) or sha256(prod) != PROD_JSON_SHA256:
            stop(f"(1) {prod} — не тот prod a, что в репо (sha256 site1h_manifest.json)")
        print("   (1) prod a на Drive = в репо (sha256) ✅")
        T, CL = modules(c)
        model, _ = T.load_pool_model('ALL', SEED, MODEL, mode='prod')
        P = Pack(c, stage)
        heads = {}
        for tf in c['tfs']:
            h = pack_tf(c, T, CL, model, P, tf, tmp)
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
        return zp
    except Stop as e:
        print(f"\n🍆 СТОП: {e}")
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    c = config()
    stamp = time.strftime('%Y%m%d_%H%M%S')
    os.makedirs(c['out_dir'] + 'logs/', exist_ok=True)
    log_path = c['out_dir'] + f"logs/pack_{stamp}.log"
    with _tee(log_path):
        print(f"🗒️ лог: {log_path}")
        zp = run(c, stamp)
    if IN_COLAB:
        if zp:
            try:
                from google.colab import files
                files.download(zp)
            except Exception as e:
                print(f"⚠️ скачивание не началось ({type(e).__name__}: {e}) — возьми zip с Drive")
        try:                                       # всё записанное этой консолью — на Drive до закрытия runtime
            drive.flush_and_unmount()
            print("☁️ Drive: записи выгружены; следующий запуск — запусти ячейку заново (Drive подключится сам)")
        except Exception as e:
            print(f"⚠️ Drive: flush_and_unmount не прошёл ({type(e).__name__}: {e}) — проверь файлы на Drive")
    return zp


if __name__ == '__main__':
    main()
