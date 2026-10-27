# -*- coding: utf-8 -*-
"""F1h_TF_Line21 — линия M модели сайта F1 (SP_l025_d1) на 15M и 30M: режимы 'line_stand' и 'line21' SuMo_Class_TF v2.1
(research/sumo_class_tf.py) с подменой ТФ и папок (2026-10-25; правила — docs/rules.md R27, записаны до прогона)

ЗАЧЕМ. На графике сайта — выбор ТФ 5M, 15M, 30M, 1H и Auto (решение Фёдора 2026-10-25). На сайт идут только ТФ с проверенной
линией M. 1H проверен 'line21' v2.1 (2026-10-22), 15M и 30M — нет: на них нужна своя линейка и пройденные ворота. Правила —
v2.1 из шапки sumo_class_tf.py без изменений; вердикт у каждого ТФ свой; не прошедший ТФ не экспортируется и на сайт не идёт.

ЧТО ДЕЛАЕТ. sumo_class_tf.py v2.1 подгружается модулем (его main() — под __main__, сам он не запускается) и не правится.
Подменяется только:
    TFL['tf']        '15M' / '30M'
    TFL['csv_ref']   BTCUSDT-15M.csv / BTCUSDT-30M.csv — свечи Binance ТФ для сверки (2) ресемпла
    DATA_1H          SMC/data_15m/ / SMC/data_30m/      — файлы research ТФ (пересобираются при каждом 'line21')
    OUT_DIR          SMC/runs/tf15m/ / SMC/runs/tf30m/  — кэш, json, golden, рисунок, логи; папки 1H (SMC/runs/tf/) не трогаются
Всё остальное — правила v2.1, модели, источники, даты, кандидаты, пороги — из файла как есть. Первая строка лога после пути —
настоящий ТФ; «1H» в текстах research ниже по логу — это он.

1H В ЛОГИКЕ (прочитано 2026-10-25, R27). Цепочка 'line21' берёт ТФ из TFL['tf'] и папки из DATA_1H / OUT_DIR в момент вызова:
минуты ТФ и баров 5M в баре (build_1h, ohlc_check), кэш (cache_1h_path — под OUT_DIR, ТФ сверяется по meta), golden, дни
истории, рисунок; живая часть, линейка, недели и П1 / П2 — в окнах и во времени, не в часах; значений по умолчанию, связанных
с 1H при определении функций, нет. «1H» остаётся только в текстах лога и в текстовом поле export['bars'] json (ТФ — в полях
'tf' и export['tf']). Стенд 'line_stand' 1H в логике имеет (ресемпл на 60 мин, 12 баров 5M в часе, дыра 3 ч = 3 бара, разгон
FIL) — он скопирован ниже (mode_line_stand_tf; источник — mode_line_stand sumo_class_tf.py v2.1, правки — STAND_EDITS) и
исполняется в пространстве имён модуля research, как оригинал. С TF '1H' копия даёт то же, что исходный файл, бит в бит
(проверено в репо: scripts/f1h_tf_stand.py regress); копия = оригинал ровно с этими правками (tests/test_f1h_tf.py).

ЗАПУСК (Colab). Этот файл — блокнот: открой scripts/f1h_tf_line21.ipynb (Файл -> Загрузить блокнот) или вставь файл целиком
в одну ячейку нового блокнота. Рядом в SMC/ должны лежать файлы research, с которыми считался 1H: sumo_class_tf.py (v2.1),
sumo_class_train.py (v6.6), sumo_class_bell.py, sumo_class_live.py (v1.1), sumo_class_ceil.py, sumo_class_site.py (v1.1);
в Raw Datasets/ — CSV 5M 31 ассета (как у 'line21' 1H) и BTCUSDT-15M.csv / BTCUSDT-30M.csv. Сам этот файл на Drive не нужен.
Порядок: MODE = 'line_stand', TF = '15M' (CPU, ~5–8 мин; итог — «🏁 СТЕНД v2: 47 / 47») -> MODE = 'line21', TF = '15M'
(A100 High-RAM, ~15 мин) -> TF = '30M' (~10 мин). Каждый запуск — заново вся ячейка (Drive подключится сам). Итог — SMC/runs/tf15m/,
SMC/runs/tf30m/ (tfline21_*.json, golden npz, рисунок, logs/); дальше — блокнот f1h_tf_pack.
"""

import os
import sys
import time
import types
import importlib.util
import pandas as pd

try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


# =====================================================================
# НАСТРОЙКИ
# =====================================================================
MODE = 'line_stand'     # 'line_stand' (синтетика, CPU, ~5–8 мин) -> 'line21' (данные Drive; A100 High-RAM: 15M ~15 мин, 30M ~10)
TF = '15M'              # '15M' | '30M'; '1H' — только 'line_stand' (регрессия с исходным файлом)

project_path = '/content/drive/MyDrive/TrendyCoins/'
CODE_DIR = project_path + 'SMC/'                       # файлы research: sumo_class_tf.py v2.1 и всё, что он подгружает

# По ТФ: свечи Binance для сверки (2), папка файлов research, папка итогов; для стенда — дыра FIL в П1 (R27: разгон FIL короче
# 4032 + 64 окон, как от дыры 4 ч на 1H, а VAL у FIL длиннее 5000 окон) и ряды проверки «длинные окна на подтверждении — стоп».
TFS = {
    '15M': {'csv_ref': 'BTCUSDT-15M.csv', 'data': 'SMC/data_15m/', 'out': 'SMC/runs/tf15m/',
            'fil_gap': ('2025-07-15 00:00', '2025-12-15 00:00'), 'long_conf': 'all'},
    '30M': {'csv_ref': 'BTCUSDT-30M.csv', 'data': 'SMC/data_30m/', 'out': 'SMC/runs/tf30m/',
            'fil_gap': ('2025-08-01 00:00', '2025-11-20 00:00'), 'long_conf': 'all'},
    '1H': {'csv_ref': 'BTCUSDT-1H.csv', 'data': 'SMC/data_1h/', 'out': 'SMC/runs/tf/', 'fil_gap': None,
           'long_conf': 'part_a'},    # пути v2.1; fil_gap None — LSTAND['fil_gap'] файла (4 ч)
}
MODES = ('line_stand', 'line21')
_ENV = 'F1HTF_'                                        # переменные окружения — для стенда вне Colab (scripts/f1h_tf_stand.py)
MODE = os.environ.get(_ENV + 'MODE', MODE)
TF = os.environ.get(_ENV + 'TF', TF)
project_path = os.environ.get(_ENV + 'PROJECT', project_path)
CODE_DIR = os.environ.get(_ENV + 'CODE_DIR', project_path + 'SMC/')


# =====================================================================
# ПОДМЕНА
# =====================================================================
def paths(tf, project=None, code=None):
    """Пути v2.1 для project_path и папки кода; DATA_1H / OUT_DIR — папки ТФ."""
    p = project or project_path
    c = code or (CODE_DIR if project is None else p + 'SMC/')
    return {'TF_PY': c + 'sumo_class_tf.py', 'TRAIN_PY': c + 'sumo_class_train.py', 'BELL_PY': c + 'sumo_class_bell.py',
            'LIVE_PY': c + 'sumo_class_live.py', 'CEIL_PY': c + 'sumo_class_ceil.py', 'SITE_PY': c + 'sumo_class_site.py',
            'RUN_DIR': p + 'SMC/runs/', 'DATA_5M': p + 'SMC/data/', 'DATA_TF': p + 'SMC/data_tf/', 'CSV_DIR': p + 'Raw Datasets/',
            'DATA_1H': p + TFS[tf]['data'], 'OUT_DIR': p + TFS[tf]['out']}


def load_tf(path):
    """sumo_class_tf.py модулем 'sumo_class_tf' (v2.1: режимы 'line_stand' и 'line21', LINE21)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"🍆 АЙ, ОШИБКА! нет {path} — положи туда sumo_class_tf.py (SuMo_Class_TF v2.1)")
    spec = importlib.util.spec_from_file_location('sumo_class_tf', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['sumo_class_tf'] = mod
    spec.loader.exec_module(mod)
    need = ('mode_line21', 'mode_line_stand', 'mode_line_diag', 'LINE21', 'LSTAND', 'TFL', 'build_1h', '_tee', '_span_txt')
    miss = [k for k in need if not hasattr(mod, k)]
    if miss or mod.TF_VER != 2:
        raise ImportError(f"🍆 АЙ, ОШИБКА! {path}: нужен SuMo_Class_TF v2.1 (нет {', '.join(miss) or '—'}, TF_VER {mod.TF_VER})")
    return mod


def apply(M, tf, P):
    """Подмена R27 в модуле M: пути, TFL['tf'], TFL['csv_ref']. Возвращает {имя: (было, стало)}."""
    out = {}
    for k in ('TRAIN_PY', 'BELL_PY', 'LIVE_PY', 'CEIL_PY', 'SITE_PY', 'RUN_DIR', 'DATA_5M', 'DATA_TF', 'CSV_DIR', 'DATA_1H',
              'OUT_DIR'):
        out[k] = (getattr(M, k), P[k])
        setattr(M, k, P[k])
    for k, v in (('tf', tf), ('csv_ref', TFS[tf]['csv_ref'])):
        out[f"TFL['{k}']"] = (M.TFL[k], v)
        M.TFL[k] = v
    return out


def stand_params(M, tf):
    """Параметры копии стенда на ТФ (R27): минуты, баров 5M в баре, баров дыры BTC 3 ч, дыра FIL в П1 и её длина в минутах,
       ряды проверки «длинные окна на подтверждении — стоп»."""
    mins = int(M.T_MIN[tf])
    gap = TFS[tf]['fil_gap'] or M.LSTAND['fil_gap']
    if 180 % mins or mins % 5:
        raise ValueError(f"🍆 АЙ, ОШИБКА! ТФ {tf}: стенд считается на ТФ, кратных 5M и делящих 3 ч")
    return {'tf': tf, 'mins': mins, 'k': mins // 5, 'gap3': 180 // mins, 'fil_gap': tuple(gap),
            'fil_min': int((pd.Timestamp(gap[1]) - pd.Timestamp(gap[0])).total_seconds() // 60),
            'long_conf': TFS[tf]['long_conf']}


def header(M, tf, swap):
    print(f"🧭 f1h_tf_line21 (обёртка, docs/rules.md R27): ТФ {tf} — SuMo_Class_TF v{M.TF_VER}.1 модулем; подмена: "
          + ', '.join(f"{k} = {a!r}" for k, (_, a) in swap.items() if k in ("TFL['tf']", "TFL['csv_ref']", 'DATA_1H', 'OUT_DIR'))
          + f". «1H» в текстах research ниже — это {tf} (логика берёт ТФ из TFL['tf'])")


def run_stand(M, tf):
    """Копия стенда (mode_line_stand_tf) в пространстве имён модуля research — как исполняется оригинал."""
    f = types.FunctionType(mode_line_stand_tf.__code__, M.__dict__, 'mode_line_stand_tf')
    return f(stand_params(M, tf))


# =====================================================================
# ЗАПУСК
# =====================================================================
def main():
    if MODE not in MODES or TF not in TFS:
        print(f"🍆 АЙ, ОШИБКА! MODE '{MODE}', TF '{TF}' — есть MODE {', '.join(MODES)}, TF {', '.join(TFS)}")
        return None
    if MODE == 'line21' and TF == '1H':
        print("🍆 СТОП: 1H проверен 'line21' 2026-10-22 исходным файлом; обёртка на 1H — только стенд (регрессия), папки 1H не "
              "трогаю")
        return None
    P = paths(TF)
    M = load_tf(P['TF_PY'])
    swap = apply(M, TF, P)
    os.makedirs(M.OUT_DIR + 'logs/', exist_ok=True)
    log_path = M.OUT_DIR + f"logs/tf_{MODE}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    with M._tee(log_path):
        print(f"🗒️ лог: {log_path}")
        header(M, TF, swap)
        res = M.mode_line21() if MODE == 'line21' else run_stand(M, TF)
    if IN_COLAB:
        try:                                       # всё записанное этой консолью — на Drive до закрытия runtime
            drive.flush_and_unmount()
            print("☁️ Drive: записи выгружены; следующий режим — запусти ячейку заново (Drive подключится сам)")
        except Exception as e:
            print(f"⚠️ Drive: flush_and_unmount не прошёл ({type(e).__name__}: {e}) — проверь файлы на Drive")
    return res


# =====================================================================
# КОПИЯ СТЕНДА. Источник — sumo_class_tf.py v2.1, mode_line_stand, как есть, с правками STAND_EDITS (было -> стало, у каждой —
# ровно одно место): ТФ — из параметра tfp (stand_params) вместо литералов 1H. Исполняется с глобальными именами модуля research
# (run_stand): имена ниже — имена того файла.
# =====================================================================
STAND_EDITS = (
    ("def mode_line_stand():", "def mode_line_stand_tf(tfp):"),
    ("'FILUSDT': (LSTAND['fil_gap'],)}.get(a, ())", "'FILUSDT': (tfp['fil_gap'],)}.get(a, ())"),
    ("ref1 = resample(full['BTCUSDT'], 60)", "ref1 = resample(full['BTCUSDT'], tfp['mins'])"),
    ("df1 = resample(df5, 60)", "df1 = resample(df5, tfp['mins'])"),
    ("n5 = per_bar_counts(df5, 60)", "n5 = per_bar_counts(df5, tfp['mins'])"),
    ("pd.Timedelta(hours=1)", "pd.Timedelta(minutes=tfp['mins'])"),
    ('check("сверка (2) на точной копии: полные часы равны у 100%, дыра 3 ч — только в CSV 1H,',
     'check(f"сверка (2) на точной копии: полные часы равны у 100%, дыра 3 ч — только в CSV {TFL[\'tf\']},'),
    ("ohlc['only_ref'] == 3 and", "ohlc['only_ref'] == tfp['gap3'] and"),
    ("только в CSV 1H {ohlc['only_ref']}", "только в CSV {TFL['tf']} {ohlc['only_ref']}"),
    ("o1 = ohlc_check(df1, n5, r_bad, 12)", "o1 = ohlc_check(df1, n5, r_bad, tfp['k'])"),
    ("o2 = ohlc_check(df1, n5, r_many, 12)", "o2 = ohlc_check(df1, n5, r_many, tfp['k'])"),
    ('check("пул 1H кодом обучения:', 'check(f"пул {TFL[\'tf\']} кодом обучения:'),
    ('check("разгон короче 4032 + 64: FIL (дыра 4 ч в П1)',
     'check(f"разгон короче 4032 + 64: FIL (дыра {_span_txt(tfp[\'fil_min\'])} в П1)'),
    ('check("линейка «своя» части A = как построил бы Bell по рядам 1H"',
     'check(f"линейка «своя» части A = как построил бы Bell по рядам {TFL[\'tf\']}"'),
    ('check("(Л) считается на рядах 1H', 'check(f"(Л) считается на рядах {TFL[\'tf\']}'),
    ('check("(Ц) на рядах 1H:', 'check(f"(Ц) на рядах {TFL[\'tf\']}:'),
    ('check("ρ по ассетам считается (у 5M и 1H', 'check(f"ρ по ассетам считается (у 5M и {TFL[\'tf\']}'),
    ('print("\\n🧪 (сл7) линейки экспорта, golden 1H, рисунок, json")',
     'print(f"\\n🧪 (сл7) линейки экспорта, golden {TFL[\'tf\']}, рисунок, json")'),
    ('check("«своя» линейка экспорта = квантили суррогатов VAL 1H боевого сида"',
     'check(f"«своя» линейка экспорта = квантили суррогатов VAL {TFL[\'tf\']} боевого сида"'),
    ('check("golden 1H:', 'check(f"golden {TFL[\'tf\']}:'),
    ("src_c = build_source_q(T, conf, raw1f[conf], BB.BELL['sources'][conf]['about'], rl['own'][conf]['burn'])",
     "src_c = build_source_q(T, conf, raw1f[conf] if tfp['long_conf'] == 'part_a' else raw1[conf],\n"
     "                               BB.BELL['sources'][conf]['about'], rl['own'][conf]['burn'])"),
    ('"оценка — свежие; FIL (разгон 4085) и ряды', 'f"оценка — свежие; FIL (разгон {sc[\'a\'][\'FILUSDT\'][\'B\']}) и ряды'),
)


def mode_line_stand_tf(tfp):
    t_all = time.time()
    tmp = tempfile.mkdtemp(prefix='smctf_line_') + '/'
    global RUN_DIR, OUT_DIR, DATA_5M, DATA_TF, DATA_1H, CSV_DIR
    keep = (RUN_DIR, OUT_DIR, DATA_5M, DATA_TF, DATA_1H, CSV_DIR)
    RUN_DIR, OUT_DIR, DATA_5M, DATA_TF, DATA_1H, CSV_DIR = (tmp + 'runs/', tmp + 'runs/tf/', tmp + 'data/', tmp + 'data_tf/',
                                                             tmp + 'data_1h/', tmp + 'raw/')
    checks = []

    def check(name, ok, detail=''):
        checks.append((name, bool(ok)))
        print(f"   {'✅' if ok else '❌'} {name}" + (f" — {detail}" if detail else ""))

    try:
        T = load_modules()
        S.SITE['formula'] = '4032·k0.5'
        f5, _ = S.pick_formula()
        VA = LV.variants_a()
        vm = {v['key']: v for v in VA}
        v5 = vm[f5]
        main, conf = keys_src()
        tf32_off(T)
        A_ = LSTAND['assets']
        T.POOL['panel'] = list(A_['panel'])
        T.POOL['sets'].update({'A': list(A_['panel']), 'B': list(A_['panel']), 'C': list(A_['C']),
                               'ALL': list(A_['C']) + list(A_['holdout'])})
        T.POOL['holdout'] = list(A_['holdout'])
        H = tuple(T.POOL['holdout'])
        print(f"\n{'#' * 110}\n# SuMo_Class_TF v{TF_VER} | СТЕНД v2 (синтетика, {tmp})\n{'#' * 110}")
        print(rules_text_line(T, f5))

        # (сл1) данные 1H из CSV 5M
        print(f"\n🧪 (сл1) бары {TFL['tf']} из CSV 5M: ресемпл, начало TRAINING, сверка (2), пул кодом обучения")
        assets = assets_1h(T)
        os.makedirs(CSV_DIR, exist_ok=True)
        full = {}
        for i, a in enumerate(assets):
            t_from = LSTAND['eth_from'] if a == 'ETHUSDT' else LSTAND['from']
            gaps = {'BTCUSDT': (LSTAND['btc_gap'], LSTAND['btc_gap_short']), 'FILUSDT': (tfp['fil_gap'],)}.get(a, ())
            full[a] = synth_bars(t_from, LSTAND['end'], 30 + i)
            df = full[a]
            for g0, g1 in gaps:
                df = df[(df['timestamp'] < pd.Timestamp(g0)) | (df['timestamp'] >= pd.Timestamp(g1))]
            df.to_csv(CSV_DIR + TFL['csv5'].format(a=a), index=False, date_format='%Y-%m-%d %H:%M:%S')
        ref1 = resample(full['BTCUSDT'], tfp['mins'])                              # «свечи Binance» — из полных 5M, без дыр
        ref1.to_csv(CSV_DIR + TFL['csv_ref'], index=False, date_format='%Y-%m-%d %H:%M:%S')
        os.makedirs(DATA_5M, exist_ok=True)
        np.savez(DATA_5M + f"smc_series_SOLUSDT_5M_N{TFC['N']}.npz",
                 meta=np.array(json.dumps({'training_start': LSTAND['sol_meta_start']})))
        infos, ohlc = build_1h(T, assets)
        show_data_1h(infos, ohlc)
        tr = {a: infos[a]['training_start'] for a in assets}
        exp_tr = {'BTCUSDT': (pd.Timestamp(LSTAND['from']) + pd.Timedelta(days=90)).strftime('%Y-%m-%d %H:%M:%S'),
                  'ETHUSDT': (pd.Timestamp(LSTAND['eth_from']) + pd.Timedelta(days=90)).strftime('%Y-%m-%d %H:%M:%S'),
                  'SOLUSDT': LSTAND['sol_meta_start'],
                  'FILUSDT': (pd.Timestamp(LSTAND['from']) + pd.Timedelta(days=90)).strftime('%Y-%m-%d %H:%M:%S')}
        check("начало TRAINING: из meta файла 5M (SOL), иначе листинг + 90 дн. (BTC, ETH, FIL)", tr == exp_tr,
              ', '.join(f"{a[:-4]} {t[:10]} ({infos[a]['train_from']})" for a, t in tr.items()))
        df5 = read_csv(CSV_DIR + TFL['csv5'].format(a='BTCUSDT'))
        df1 = resample(df5, tfp['mins'])
        n5 = per_bar_counts(df5, tfp['mins'])
        rng_ = np.random.default_rng(5)
        bad = 0
        for j in rng_.choice(len(df1), size=300, replace=False):
            t_ = df1['timestamp'].iloc[j]
            sub = df5[(df5['timestamp'] >= t_) & (df5['timestamp'] < t_ + pd.Timedelta(minutes=tfp['mins']))]
            bad += int(not (df1['open'].iloc[j] == sub['open'].iloc[0] and df1['high'].iloc[j] == sub['high'].max()
                            and df1['low'].iloc[j] == sub['low'].min() and df1['close'].iloc[j] == sub['close'].iloc[-1]
                            and n5[j] == len(sub)))
        check("ресемпл: open / high / low / close часа и число баров 5M внутри — как руками (300 случайных часов)", bad == 0,
              f"расхождений {bad}")
        check(f"сверка (2) на точной копии: полные часы равны у 100%, дыра 3 ч — только в CSV {TFL['tf']}, неполный час (дыра 20 мин) — "
              "в неполных", ohlc['ok'] and ohlc['share_full'] == 1.0 and ohlc['only_ref'] == tfp['gap3'] and ohlc['partial'] >= 1
              and ohlc['n_diff_full'] == 0, f"полных {ohlc['full']}, неполных {ohlc['partial']}, только в CSV {TFL['tf']} {ohlc['only_ref']}")
        ref_csv = read_csv(CSV_DIR + TFL['csv_ref'])                    # как в сверке: CSV -> float (без ошибки записи)
        r_bad = ref_csv.copy()
        r_bad.loc[5000, 'high'] = r_bad.loc[5000, 'high'] * 1.001
        o1 = ohlc_check(df1, n5, r_bad, tfp['k'])
        r_many = ref_csv.copy()
        pick = np.random.default_rng(6).choice(np.arange(10, len(r_many) - 10), size=len(r_many) // 50, replace=False)
        r_many.loc[pick, 'close'] = r_many.loc[pick, 'close'] * 1.0001
        o2 = ohlc_check(df1, n5, r_many, tfp['k'])
        check("сверка (2): один чужой бар найден (первый с разницей — он, порог 99.9% держит); 2% чужих — не прошла",
              o1['n_diff_full'] == 1 and o1['first_diff'] == str(ref_csv.loc[5000, 'timestamp']) and o1['ok']
              and not o2['ok'], f"{o1['n_diff_full']} / {o1['first_diff']} | {100 * o2['share_full']:.2f}%")
        cfg_m = {**T.SMC_BASE, **T.RUNS[model_name()]}
        price, nt_cols = cfg_m['price'], T._pool_nt_cols(model_name())
        keep_d = (T.DATA['timeframe'], T.DATA_DIR, T.DATA_OVERLAY)
        pdata = pool_1h(T, price, nt_cols)
        same_d = (T.DATA['timeframe'], T.DATA_DIR, T.DATA_OVERLAY) == keep_d
        ne = float(pdata['data'].nt_stats['norm_err'])
        w = pdata['data'].windows(T.POOL['L'])
        n_by = {a: {p: int((w['sid'][w[p]] == pdata['sid'][a]).sum()) for p in ('tune', 'val')} for a in assets}
        check(f"пул {TFL['tf']} кодом обучения: файлы читаются (нормировщик < 1e-5), окна TUNE / VAL у всех, настройки данных после — как "
              "были", ne < 1e-5 and all(x['tune'] > 4096 and x['val'] > 5000 for x in n_by.values()) and same_d
              and pdata['data'].tf == TFL['tf'], ' | '.join(f"{a[:-4]} {x['tune']} / {x['val']}" for a, x in n_by.items()))

        # (сл2) оценки 1H: случайные сети F1 как прогоны пула, кэш, ряды источников
        print(f"\n🧪 (сл2) оценки {TFL['tf']}: случайные сети {model_name()} вместо прогонов пула, кэш, ряды источников")
        with CL.floor_override(pdata['data'], T.POOL['L'], TFC['floor']):
            st_ = T.pool_asset_store(pdata['data'], T.POOL['L'], 0, pdata['t_cut'])
            P_ = st_.positives('tune')
            Xs, Ss = P_['X'][:4096], P_['S'][:4096]
        items = items_1h(T)
        for i, it in enumerate(items):
            stand_pool_run(T, cfg_m, Xs, Ss, it['set'], it['seed'], it['mode'], i)
        del st_, P_, Xs, Ss
        fp_ = data_fp(infos)
        t1 = time.time()
        rec = score_all_1h(T, pdata, assets, items, fp_, None)
        t_sc = time.time() - t1
        rec2 = score_all_1h(T, pdata, assets, items, fp_, None)
        same_c = all(rec2[t]['cached'] and set(rec2[t]['res']) == set(rec[t]['res'])
                     and all(np.array_equal(rec2[t]['res'][k][f], rec[t]['res'][k][f]) for k in rec[t]['res']
                             for f in ('sp', 'sn', 'wk', 'st', 't', 'med', 'mabs')) for t in rec)
        check("кэш оценок = пересчёт бит в бит (sp, sn, недели, начала, время, |r| окна); другой отпечаток файлов — не читается",
              same_c and load_cache_1h(items[0]['tag'], fp_ + 'x') is None, f"счёт {t_sc / 60:.1f} мин")
        x0 = rec[items[0]['tag']]['res'][('BTCUSDT', 'val')]
        check("время окна = время последнего бара окна в пуле; неделя = неделя этого времени",
              np.array_equal(x0['t'] // T.WEEK_NS, x0['wk']) and np.array_equal(x0['t'], pdata['data'].t[x0['st'] + 63]))
        aff_in = {}
        with CL.floor_override(pdata['data'], T.POOL['L'], TFC['floor']):
            aff_in['floor'] = float(pdata['data'].windows(T.POOL['L'])['floor'][0])
        check("пол внутри floor_override — 0.2 у всех рядов", aff_in['floor'] == np.float32(TFC['floor']))
        raw1, raw1f, drops = {}, {}, {}
        for key in (main, conf):
            raw1[key], miss = raw_1h(T, rec, key, assets)
            raw1f[key], drops[key] = burn_ok(raw1[key], max(LV.LIVE['L']) + LV.LIVE['lag'])
        d_m, d_c = raw1[main]['a']['ETHUSDT'], raw1[conf]['a']['ETHUSDT']
        R_m = rec[items[0]['tag']]['res']
        check("ряды как у Bell: основной — TUNE + VAL, разгон = окон TUNE; подтверждение — VAL, разгон = окон П1",
              d_m['B'] == len(R_m[('ETHUSDT', 'tune')]['sp']) and len(d_m['sp']) == d_m['B'] + len(R_m[('ETHUSDT', 'val')]['sp'])
              and d_c['B'] == int(T._nc_half(d_c['wk']).sum()) and d_c['B'] > 0, f"{d_m['B']} / {d_c['B']}")
        check(f"разгон короче 4032 + 64: FIL (дыра {_span_txt(tfp['fil_min'])} в П1) — вне части A подтверждения, у основного — все",
              drops[conf] == ['FILUSDT'] and drops[main] == [] and 'FILUSDT' in raw1[conf]['a'],
              f"{conf}: {drops[conf]} (у FIL разгон {raw1[conf]['a']['FILUSDT']['B']})")
        fb = floor_binds(T, rec, items)
        check("доля окон, где сработал пол, — в [0; 1]", 0.0 <= fb['share'] <= 1.0, f"{100 * fb['share']:.3f}%")

        # (сл3) копии Bell / Ceil = оригиналы
        print("\n🧪 (сл3) копии Bell / Ceil с подменой линейки — при «как было» бит в бит с оригиналами")
        Wm = S.synth_world(T, assets, 6000, 6000, 0.5, 21)
        src_b = BB.build_source(T, main, Wm, 'стенд')
        src_q = build_source_q(T, main, Wm, 'стенд')
        q_own = {s: src_b['ruler'][s] for s in Wm}
        src_q2 = build_source_q(T, main, Wm, 'стенд', q_own)

        def same_src(x, y):
            return all(np.array_equal(x['ruler'][s], y['ruler'][s]) for s in x['ruler']) and all(
                np.array_equal(x['series'][s][a][f], y['series'][s][a][f]) for s in x['series'] for a in x['series'][s]
                for f in ('zr', 'zs', 'zg', 'zw')) and all(
                np.array_equal(x['series'][s][a]['acf']['rho_null'], y['series'][s][a]['acf']['rho_null'])
                for s in x['series'] for a in x['series'][s])
        check("build_source: копия без линейки и с той же линейкой = Bell (z, случайный график, автокорреляция)",
              same_src(src_b, src_q) and same_src(src_b, src_q2))
        c_b = CL.cells_cross(T, Wm, main, v5, skip=H)
        c_q = cells_cross_q(T, Wm, main, v5, skip=H)
        c_q2 = cells_cross_q(T, Wm, main, v5, skip=H, Q=CL.fold_rulers(T, Wm, main, H))
        check("cells_cross: копия без линеек и с теми же линейками = Ceil",
              len(c_b) == len(c_q) == len(c_q2) and all(x[:2] == y[:2] == z[:2] and np.array_equal(x[2], y[2])
                                                        and np.array_equal(x[3], y[3]) and np.array_equal(x[2], z[2])
                                                        and np.array_equal(x[3], z[3]) for x, y, z in zip(c_b, c_q, c_q2)))
        c_p = cells_cross_q(T, Wm, main, vm['2016·k0.5'], skip=H, P=4096)
        c_d = cells_cross_q(T, Wm, main, vm['2016·k0.5'], skip=H)
        n_p, n_d = sum(len(c[2]) for c in c_p), sum(len(c[2]) for c in c_d)
        check("первое окно счёта P: у формулы L 2016 с P = 4096 окон меньше на (4096 − 2080) на ряд",
              n_d - n_p == (4096 - 2080) * len(c_d), f"{n_d} − {n_p}")
        UN = ('SYN28USDT', 'SYN29USDT', 'SYN30USDT')                   # «незнакомые» мира synth_cells Ceil
        cs = CL.synth_cells(31, 30000, 3, over=('SYN1USDT', 'SYN2USDT'))
        Cs = tuple(TFL['cands_c']) + tuple(CL.CEIL['refs'])
        r_c = CL.rule_source(cs, Cs, unseen=UN)
        r_t = rule_source_tf(cs, Cs, unseen=UN)
        check("rule_source: копия с корзинами Ceil = Ceil (все поля, все кандидаты)",
              json.dumps(_jsonable(r_c), sort_keys=True) == json.dumps(_jsonable(r_t), sort_keys=True))

        # (сл4) правило по ассету на 1H
        print("\n🧪 (сл4) правило по ассету на 1H: (б)-1H, (Ц2)-1H, (Ц3) на мирах")
        honest = CL.synth_cells(31, 40000, 4)
        b_h = b_gate_tf(honest)
        r_h = rule_source_tf(honest, Cs, unseen=UN, pair_edges=TFL['pair_edges'], fine_max=TFL['pair_fine_max'],
                             unseen_sparse_ok=True)
        check("честная линия: (б)-1H проходит, (Ц2)-1H при C = 10 проходит, у пар — корзины 1–1.5, 1.5–2, 2–3 и группа 3+",
              b_h['ok'] and r_h['by_c'][10]['g2'] and r_h['by_c'][10]['pairs']['SYN0USDT|orange']['group']['lo'] == 3.0,
              f"(б) {b_h['over']} / {b_h['nodata']}, (Ц2) {r_h['by_c'][10]['over']} / {r_h['by_c'][10]['nodata']}")
        over_a = tuple(f"SYN{i}USDT" for i in range(1, 8))
        ov = CL.synth_cells(31, 40000, 5, over=over_a)
        b_o = b_gate_tf(ov)
        flag = sorted({k.split('|')[0] for k, s_ in b_o['bad'].items() if s_ == 'завышает'})
        check("7 ассетов завышают: (б)-1H не проходит, завышение найдено ровно у них", not b_o['ok'] and flag == sorted(over_a),
              f"завышают {b_o['over']} пар: {', '.join(a[:-4] for a in flag)}")
        sp_ = CL.synth_cells(31, 1500, 6)
        b_s = b_gate_tf(sp_)
        check("мало данных: (б)-1H не проходит из-за «без данных», не из-за завышения", not b_s['ok'] and b_s['nodata'] > 6,
              f"без данных {b_s['nodata']} из {b_s['n']}")
        mix = [c for c in honest if c[1] != UN[2]] + [(c[0], c[1], c[2][:900], c[3][:900]) for c in honest if c[1] == UN[2]]
        r_mx = rule_source_tf(mix, (10,), unseen=UN, pair_edges=TFL['pair_edges'], fine_max=TFL['pair_fine_max'],
                              unseen_sparse_ok=True)
        r_mx5 = rule_source_tf(mix, (10,), unseen=UN)
        ovu = [c for c in honest if c[1] != UN[2]] + [c for c in CL.synth_cells(31, 40000, 7, over=(UN[2],)) if c[1] == UN[2]]
        r_ou = rule_source_tf(ovu, (10,), unseen=UN, pair_edges=TFL['pair_edges'], fine_max=TFL['pair_fine_max'],
                              unseen_sparse_ok=True)
        check("(Ц3) на 1H: у незнакомого «мало данных» — пометка, не провал (на 5M — провал); завышение у незнакомого — провал",
              r_mx['by_c'][10]['g3'] is True and r_mx['by_c'][10]['unseen_sparse'] and r_mx5['by_c'][10]['g3'] is False
              and r_ou['by_c'][10]['g3'] is False, f"мало данных: {r_mx['by_c'][10]['unseen_sparse']}")
        check("(Ц) на 1H: C проходит только при (Ц1)–(Ц3) на основном и (Ц1) на подтверждении",
              passes_tf(r_h, r_h, 10) and not passes_tf(r_ou, r_h, 10) and not passes_tf(r_h, {'by_c': {10: {'g1': False}}}, 10))

        # (сл5) (Л) и (Ж) на готовых числах
        print("\n🧪 (сл5) правила (Л) и (Ж) на готовых числах")

        def noise(seed, n=60):
            x = np.random.default_rng(seed).normal(0, 1e-4, n)
            return x - x.mean()                                         # среднее ровно 0: разница — ровно заданная

        def fake(wk_, ok=True):
            return {'wk_bce': wk_, 'bce': float(wk_.mean()), 'n_shown': 1000, 'ceil': {'orange': 13 if ok else 3,
                    'blue': 13}, 'share': 1.0, 'gp_col': 0.02}

        def fake_ev(d_m, d_c, ok_one=True):
            bm, bc = 0.6 + noise(1), 0.6 + noise(2)
            one_c = fake(bc + d_c + noise(4), ok_one)
            if not ok_one:
                one_c['ceil'] = {'orange': 3, 'blue': 13}
            return {'own': {main: {f5: fake(bm)}, conf: {f5: fake(bc)}},
                    'one': {main: {f5: fake(bm + d_m + noise(3))}, conf: {f5: one_c}}}
        cases = [('«одна» лучше', fake_ev(-2e-4, -2e-4), True, 'one'),
                 ('«одна» не хуже (Δ = 0)', fake_ev(0.0, 0.0), True, 'one'),
                 ('«одна» хуже на основном больше 1 SE', fake_ev(3e-4, 0.0), True, 'own'),
                 ('«одна» хуже на подтверждении больше 2 SE', fake_ev(0.0, 3e-4), True, 'own'),
                 ('ворота «одной» на подтверждении ✗', fake_ev(-2e-4, -2e-4, ok_one=False), True, 'own'),
                 ('нечего экспортировать', fake_ev(-2e-4, -2e-4), False, 'own')]
        got = [(nm, ruler_rule(T, e_, f5, ex)['pick'], want) for nm, e_, ex, want in cases]
        miss_ = [f"{nm}: {g_} вместо {w_}" for nm, g_, w_ in got if g_ != w_]
        check(f"(Л): {len(cases)} случаев — линейка по правилу", not miss_, '; '.join(miss_))
        base = 0.6 + noise(10)
        res_f = {'4032': fake(base - 1e-3 + noise(11), ok=False),           # F1: BCE лучше, но не прошёл (а)
                 '4032+144': fake(base + 1e-6 + noise(12)),                  # F2: в полосе, по порядку Live — раньше F3
                 f5: fake(base),                                             # формула 5M: лучшая из прошедших
                 '3024·k0.5': fake(base + 5e-3 + noise(13))}                 # вне полосы
        ch_ = LV.choose(T, res_f, list(res_f), lambda k: LV.order_a(vm[k]))
        band_ok = f5 in ch_['band'] and ch_['pick'] == '4032+144' and '3024·k0.5' not in ch_['band']
        res_g = dict(res_f)
        res_g[f5] = fake(base + 5e-3 + noise(14))

        def pick_tf(res_):
            c = LV.choose(T, res_, list(res_), lambda k: LV.order_a(vm[k]))
            return f5 if (c['pick'] and f5 in c['band']) else c['pick']
        check("(Ж): формула 5M в полосе — берётся она (по порядку Live был бы F2); вне полосы — порядок Live",
              band_ok and pick_tf(res_f) == f5 and pick_tf(res_g) == '4032+144',
              f"полоса {ch_['band']}, Live — {ch_['pick']}")

        # (сл6) часть A на 1H целиком: случайные сети (сигнала нет) — путь до вердикта
        print(f"\n🧪 (сл6) части (Л), (Ж), (Ц) на рядах {TFL['tf']} случайных сетей (сигнала нет — путь, не числа)")
        os.makedirs(RUN_DIR + 'pool/', exist_ok=True)
        w_val = pd.Timestamp(TFC['t_valid']).value // T.WEEK_NS
        w_half = pd.Timestamp(T.NOCAL['half_cut']).value // T.WEEK_NS
        for key, (set_, mode_, parts_) in ((main, ('C', 'stop', ('tune', 'val'))), (conf, ('ALL', 'refit', ('val',)))):
            W5 = S.synth_world(T, assets, 9000 if len(parts_) == 2 else 0, 9000, 0.5, 31 + len(parts_))
            for s_ in W5:
                for a, d in W5[s_].items():
                    if len(parts_) == 2:                                    # VAL — с недели t_valid
                        d['wk'] = d['wk'] - d['wk'][d['B']] + w_val
                    else:                                                   # П1 — 2 недели, дальше П2
                        d['wk'] = d['wk'] - d['wk'][0] + w_half - 2
                        d['B'] = int(T._nc_half(d['wk']).sum())
            stand_scores_npz(T, W5, set_, mode_, parts_)
        raw5, src5 = {}, {}
        for key in (main, conf):
            raw5[key], src5[key] = raw_5m(T, key, assets)
        check("оценки 5M без кэша Site — из npz пула (с пометкой)", all(raw5[k] is not None and src5[k].startswith('npz')
                                                                         for k in (main, conf)), ' | '.join(src5.values()))
        tg = T._pool_tag('C', 'a', model_name(), 'stop')
        res_c = {(a, p): {'sp': raw5[main]['a'][a]['sp'][sl] + 0.01, 'sn': raw5[main]['a'][a]['sn'][sl],
                          'med': np.ones(n_, np.float32), 'mabs': np.ones(n_, np.float32), 'wk': raw5[main]['a'][a]['wk'][sl]}
                 for a in assets for p, sl, n_ in (('tune', slice(0, raw5[main]['a'][a]['B']), raw5[main]['a'][a]['B']),
                                                   ('val', slice(raw5[main]['a'][a]['B'], None),
                                                    len(raw5[main]['a'][a]['sp']) - raw5[main]['a'][a]['B']))}
        for s_ in ('a', 'b'):
            CL.save_cache({'tag': T._pool_tag('C', s_, model_name(), 'stop')}, res_c, {}, {}, {a: 0.2 for a in assets})
        r5c, s5c = raw_5m(T, main, assets)
        want_sp = (raw5[main]['a']['BTCUSDT']['sp'] + 0.01).astype(np.float32).astype(np.float64)
        check("оценки 5M с кэшем Site — из кэша (пол 0.2)", s5c.startswith('кэш') and np.array_equal(
            r5c['a']['BTCUSDT']['sp'], want_sp) and r5c['a']['BTCUSDT']['B'] == raw5[main]['a']['BTCUSDT']['B'],
              f"{s5c}; тег {tg}")
        P_b = max(LV.LIVE['L']) + LV.LIVE['lag']
        rl = {'own': {k: rulers_from(T, raw1[k], k, H) for k in (main, conf)},
              'one': {k: rulers_from(T, raw5[k], k, H) for k in (main, conf)}}
        knots, zk = BB.knots_zk(T)
        src_chk = build_source_q(T, main, raw1f[main], 'стенд')
        check(f"линейка «своя» части A = как построил бы Bell по рядам {TFL['tf']}", all(
            np.array_equal(src_chk['ruler'][s_], rl['own'][main]['burn'][s_]) for s_ in src_chk['ruler']))
        ev = eval_rulers(T, raw1, raw1f, rl, v5, H, P_b)
        r_m, r_c2 = ev['own'][main][f5], ev['own'][conf][f5]
        b_ind = b_gate_tf(cells_cross_q(T, raw1[main], main, v5, skip=H, Q=rl['own'][main]['fold'], P=P_b))
        check("(б)-1H в воротах Live: на основном — по своим ячейкам (= независимый пересчёт), доля пар 1 / 0 по нему; на "
              "подтверждении не считается", r_m['b1h'] == b_ind and r_m['share'] == (1.0 if b_ind['ok'] else 0.0)
              and r_c2['b1h'] is None and r_c2['share'] == 1.0 and 'share_5m' in r_m,
              f"основной: {b_ind['over']} / {b_ind['nodata']} из {b_ind['n']}")
        RL = ruler_rule(T, ev, f5, True)
        check(f"(Л) считается на рядах {TFL['tf']} (обе линейки, оба источника)", RL['pick'] in ('one', 'own') and
              all(np.isfinite(x) for x in RL['d_main']), f"выбор {RL['pick']}, Δ BCE основной {1e4 * RL['d_main'][0]:+.2f} ×1e4")
        t1 = time.time()
        VA2 = [vm[k] for k in ('4032', '4032+144', f5, '2016·k2', 'EMA 8064')]
        res_a, ch, cf, ok_f = part_a_tf(T, raw1, raw1f, rl, 'own', VA2, f5, H, P_b)
        show_part_a(T, res_a, ch, cf, f5, VA2)
        check("(Ж): все варианты на обоих источниках, у кандидатов основного — (б)-1H, итог — выбор или граница",
              set(res_a[main]) == {v['key'] for v in VA2} and all(res_a[main][v['key']]['b1h'] is not None
                                                                  for v in VA2 if v['cand'])
              and (ok_f or ch['pick'] is None or (cf is not None and not cf['ok'])),
              f"выбор {ch['pick']}, подтверждено {ok_f} ({time.time() - t1:.0f} c)")
        Rc, C = ceiling_tf(T, raw1, {k: rl['own'][k]['fold'] for k in (main, conf)}, v5, H)
        check(f"(Ц) на рядах {TFL['tf']}: оба источника, пары — с корзинами 1H, незнакомые — FIL",
              set(Rc) == {main, conf} and Rc[main]['by_c'][10]['unseen'] == ['FILUSDT']
              and Rc[main]['by_c'][10]['pairs']['BTCUSDT|orange']['group']['lo'] == 3.0
              and Rc[conf]['by_c'][10]['g3'] is None, f"выбор C — {C}")
        rho = rho_assets(T, raw1[main], raw5[main])
        check(f"ρ по ассетам считается (у 5M и {TFL['tf']} разные календари — по общим неделям)", set(rho) <= set(assets))

        # (сл7) экспорт, golden, рисунок, json
        print(f"\n🧪 (сл7) линейки экспорта, golden {TFL['tf']}, рисунок, json")
        q_own = ruler_own_q(T, rec, assets)
        sn_all = np.concatenate([rec[prod_tag(T)]['res'][(a, 'val')]['sn'] for a in assets])
        check(f"«своя» линейка экспорта = квантили суррогатов VAL {TFL['tf']} боевого сида", np.array_equal(
            q_own['q'], BB.ruler_q(sn_all, knots)) and q_own['n_null'] == len(sn_all))
        check("«одна» без json Site и без кэша prod — нет", ruler_one_q(T, assets) is None)
        q_fake = np.sort(np.random.default_rng(8).normal(0, 1, len(knots)))
        os.makedirs(RUN_DIR + 'site/', exist_ok=True)
        save_json(RUN_DIR + 'site/site_20261021_000000.json', {'export': {'ruler': {'q': list(q_fake), 'n_null': 123},
                                                                          'live': {'key': f5}, 'ceiling': 10}})
        q1 = ruler_one_q(T, assets)
        check("«одна» — из раздела export json Site", q1 is not None and np.array_equal(q1['q'], q_fake) and
              q1['site_live'] == f5 and q1['site_ceiling'] == 10)
        sh = ruler_shift(T, q_own['q'], q_own['q'], sn_all)
        check("сдвиг линейки сама с собой — 0", sh['dz_max'] < 1e-12 and abs(sh['z_sd'] - sh['z_sd_own']) < 1e-12)
        gp = OUT_DIR + 'stand_golden.npz'
        os.makedirs(OUT_DIR, exist_ok=True)
        same_g, gi = golden_1h(T, rec, q_own, v5, gp)
        with np.load(gp) as d:
            first = int(np.argmax(d['show']))
            lm_ok = bool(np.isnan(d['logm'][:first]).all() and np.isfinite(d['logm'][first:]).all())
            meta = json.loads(str(d['meta']))
            t_ok = np.array_equal(d['t_end'], rec[prod_tag(T)]['res'][('BTCUSDT', 'val')]['t'])
        check(f"golden {TFL['tf']}: линия из записанных logit бит в бит, с окна 4096, до него пусто; ТФ и время окон в файле",
              same_g and first == CL.full_from(v5) and lm_ok and meta['tf'] == TFL['tf'] and t_ok,
              f"окон {gi['n']}, линия на {gi['shown']}, {gi['from'][:16]} … {gi['to'][:16]}")
        pp = OUT_DIR + 'stand_line.png'
        plot_line(pp, Rc, 10, gp, 'СТЕНД v2: случайные сети на синтетике (числа — не про рынок)')
        save_json(OUT_DIR + 'stand_line.json', {'checks': checks, 'ceiling': pack_ceil(Rc), 'eval': pack_eval(res_a[main])})
        check("рисунок и json записаны", os.path.getsize(pp) > 10_000 and os.path.getsize(OUT_DIR + 'stand_line.json') > 1000)

        # (сл8) v2.1: разведка 'line_diag'
        print("\n🧪 (сл8) v2.1, разведка 'line_diag': копия eval_set, детали (б)-1H, длинные окна, отсечка свежих данных, путь "
              "целиком")
        src_d = build_source_q(T, main, raw1f[main], BB.BELL['sources'][main]['about'], rl['own'][main]['burn'])
        recs_d = LV.prep_source(T, src_d)
        w0_d, NW_d = LV.week_span(recs_d)

        def same_num(x, y):
            return bool(x == y or (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y)))
        bad_e = []
        for k in ('4032', f5, '2016+144', 'EMA 8064'):
            e1 = LV.eval_set(T, recs_d, vm[k], w0_d, NW_d, 'steady')
            e2 = diag_eval(T, recs_d, vm[k], w0_d, NW_d, LDIAG['marks'])
            ok_ = (same_num(e1['bce'], e2['bce']) and np.array_equal(e1['wk_bce'], e2['wk_bce'], equal_nan=True)
                   and same_num(e1['gp_col'], e2['gp_col']) and same_num(e1['gp_se'], e2['gp_se'])
                   and all(same_num(e1['ceil'][c_], e2['ceil'][c_]) for c_ in COLS) and e1['n_shown'] == e2['n_shown']
                   and all(same_num(x, y) for x, y in zip(e1['cover'], e2['cover']['1.25'])))
            g_ = [e2['gp'][f"{t:g}"][0] for t in LDIAG['marks']]
            c_ = [sum(e2['cover'][f"{t:g}"]) for t in LDIAG['marks']]
            if not (ok_ and all(a >= b for a, b in zip(g_, g_[1:])) and all(a >= b for a, b in zip(c_, c_[1:]))):
                bad_e.append(k)
        check("diag_eval: при отметке 1.25 — BCE, R9, cover и случайный график бит в бит с eval_set Live; при 1.5 и 2 доли не "
              "больше", not bad_e, f"не сошлось: {', '.join(bad_e)}" if bad_e else "4 варианта")
        bad_b = []
        for nm_, cs_ in (('честная', honest), ('7 завышают', ov), ('мало данных', sp_)):
            b1, b2 = b_gate_tf(cs_), diag_b(cs_)
            st_ok = all(s_['status'] == ('завышает' if any(x['over'] for x in s_['checks']) else
                                         ('без данных' if any(x['sparse'] for x in s_['checks']) else 'ок'))
                        for s_ in b2['bad'].values())
            if not ((b1['ok'], b1['over'], b1['nodata'], b1['n']) == (b2['ok'], b2['over'], b2['nodata'], b2['n'])
                    and b1['bad'] == {k_: s_['status'] for k_, s_ in b2['bad'].items()} and st_ok):
                bad_b.append(nm_)
        check("diag_b: числа (б)-1H = b_gate_tf, у пар не «ок» — корзины с отношением и окнами, статус по ним тот же",
              not bad_b, f"не сошлось: {', '.join(bad_b)}" if bad_b else "честная / завышают / мало данных")
        VL_ = variants_long()
        e8 = diag_eval(T, recs_d, VL_[3], w0_d, NW_d, LDIAG['marks'])
        src_c = build_source_q(T, conf, raw1f[conf] if tfp['long_conf'] == 'part_a' else raw1[conf],
                               BB.BELL['sources'][conf]['about'], rl['own'][conf]['burn'])
        recs_c = LV.prep_source(T, src_c)
        w0_c, NW_c = LV.week_span(recs_c)
        try:
            diag_eval(T, recs_c, VL_[3], w0_c, NW_c, LDIAG['marks'])
            raised = False
        except ValueError:
            raised = True
        check("длинные окна: F1 / F2 / F3 у 6048 и 8064; на основном (разгон TUNE) считаются, на подтверждении (разгон П1) — "
              "стоп, а не тихий неполный счёт", [v['key'] for v in VL_] == ['6048', '6048+144', '6048·k0.5', '8064',
                                                                             '8064+144', '8064·k0.5']
              and e8['n_shown'] > 0 and raised, f"8064 на основном: показано {e8['n_shown']} окон")
        del src_d, recs_d, src_c, recs_c
        keep_h = LDIAG['holdout_from']
        try:
            LDIAG['holdout_from'] = '2026-02-01 00:00:00'
            cut1, end1 = diag_cut_end()
            LDIAG['holdout_from'] = '2026-07-01 00:00:00'
            cut2, end2 = diag_cut_end()
        finally:
            LDIAG['holdout_from'] = keep_h
        tmax = pd.Timestamp(fresh_max(rec))
        tman = pd.Timestamp(max(int(d['t'].max()) for r_ in rec.values() for d in r_['res'].values()))
        check("отсечка свежих данных: CSV BTC дальше даты — режем все CSV по ней, раньше — данные как есть; последнее окно "
              "оценок — максимум по всем", cut1 == '2026-02-01 00:00:00' and cut2 is None and end1 == end2 and tmax == tman,
              f"конец CSV BTC {end1}, последнее окно {tmax}")
        t1 = time.time()
        od = mode_line_diag(T)
        ev_m, ev_c = od.get('eval', {}).get(main, {}), od.get('eval', {}).get(conf, {})
        want_m = list(LDIAG['keys']) + [v['key'] for v in VL_]
        check("путь 'line_diag' целиком: данные как у 'line' (кэш), основной — все варианты с (б)-1H, подтверждение — без "
              "длинных окон, json", 'stop' not in od and set(ev_m) == set(want_m) and set(ev_c) == set(LDIAG['keys'])
              and all(ev_m[k]['b1h'] is not None for k in ev_m) and all(ev_c[k]['b1h'] is None for k in ev_c)
              and pd.Timestamp(od['data_end']) < pd.Timestamp(LDIAG['holdout_from']),
              f"{time.time() - t1:.0f} c, окна до {od.get('data_end')}")
        try:
            LDIAG['holdout_from'] = '2026-02-01 00:00:00'
            od2 = mode_line_diag(T)
        finally:
            LDIAG['holdout_from'] = keep_h
        check("данные режутся — кэша 'line' нет: на CPU — стоп с просьбой включить GPU, а не счёт часами",
              str(od2.get('stop', '')).startswith('нет кэша'), od2.get('stop', ''))

        # (сл9) v2.1: правила 'line21'
        print("\n🧪 (сл9) v2.1, правила 'line21': допуски и правило по ассету (подменой настроек), ряды до / после свежей даты, "
              "потолок")

        def snap():
            return json.dumps(_jsonable(T.NOCAL), sort_keys=True), json.dumps(_jsonable(CL.CEIL), sort_keys=True)
        s0 = snap()
        with rules21(T):
            inside = (tuple(T.NOCAL['tol']), T.NOCAL['ceil_pool'], tuple(CL.tol()), CL.CEIL['share_over'],
                      CL.CEIL['share_nodata'])
        check("rules21: на время блока — допуски 0.8 / 1.5, (а) до 5, по ассету завышают <= 10% и без данных <= 50%; после — "
              "как было", inside == ((0.8, 1.5), 5, (0.8, 1.5), 0.1, 0.5) and snap() == s0, str(inside))
        E_ = list(T.NC_EDGES)

        def fake_rows(ratios):
            rows_ = []
            for col in COLS:
                for i, lo in enumerate(E_[:-1]):
                    r_ = ratios[i] if i < len(ratios) else None
                    rows_.append({'color': col, 'lo': lo, 'n_real': 5000 if r_ else 0, 'n_surr': 5000 if r_ else 0,
                                  'ratio': r_ if r_ else math.nan, 'sparse': r_ is None})
            return rows_
        f_rows = (fake_rows([1, 1, 1, 1, 1.4, 1, 1]), fake_rows([1, 1, 1, 1, 0.79, 1]), fake_rows([1, 1, 1, 1, 1, 1.6]))
        c2 = tuple(T._nc_ceil(x, 'orange') for x in f_rows)
        with rules21(T):
            c21 = tuple(T._nc_ceil(x, 'orange') for x in f_rows)
        check("(а) v2.1: занижение в 1.4 раза — не провал (в v2 — провал); завышение (< 0.8) и занижение больше 1.5 — провал",
              c2 == (3.0, 3.0, 5.0) and c21 == (13.0, 3.0, 5.0), f"v2 {c2}, v2.1 {c21}")
        nd_a = tuple(f"SYN{i}USDT" for i in range(10, 20))
        mix_nd = ([c for c in honest if c[1] not in nd_a]
                  + [(c[0], c[1], c[2][:1500], c[3][:1500]) for c in honest if c[1] in nd_a])
        b_v2 = b_gate_tf(mix_nd)
        with rules21(T):
            b_v21 = [b_gate_tf(x) for x in (honest, ov, sp_, mix_nd)]
        check("(б) v2.1: «нет данных» у трети пар — пометка (в v2 — провал); 7 завышающих (11%) — провал; без данных у всех — "
              "провал", not b_v2['ok'] and b_v2['nodata'] > 6 and [x['ok'] for x in b_v21] == [True, False, False, True],
              f"v2: без данных {b_v2['nodata']} из {b_v2['n']}; v2.1 (завышают, без данных): "
              f"{[(x['over'], x['nodata']) for x in b_v21]}")
        hold_t = pd.Timestamp('2026-01-01 00:00:00')
        r_m, _ = raw_1h(T, rec, main, assets)
        r_c, _ = raw_1h(T, rec, conf, assets)
        sm, sc = rows_split(r_m, int(hold_t.value), False), rows_split(r_c, int(hold_t.value), True)
        ok_r = True
        for s_ in r_m:
            for a in r_m[s_]:
                d0, d1, e0, e1 = r_m[s_][a], sm[s_][a], r_c[s_][a], sc[s_][a]
                k = int((d0['t'] < hold_t.value).sum())
                ok_r = ok_r and (len(d1['sp']) == len(d1['sn']) == len(d1['wk']) == len(d1['st']) == k and d1['B'] == d0['B']
                                 and np.array_equal(d1['t'], d0['t'][:k]) and int(d1['t'][-1]) < hold_t.value
                                 and e1['B'] == int((e0['t'] < hold_t.value).sum()) and np.array_equal(e1['sp'], e0['sp']))
        fr, drop_f = fresh_ok(sc, P_b, LINE21['fresh_min'])
        _, drop_all = fresh_ok(sc, P_b, 10 ** 9)
        check("ряды v2.1: основной — только окна раньше свежей даты (разгон TUNE как был); подтверждение — разгон до даты, "
              f"оценка — свежие; FIL (разгон {sc['a']['FILUSDT']['B']}) и ряды без свежих окон — не берутся", ok_r and drop_f == ['FILUSDT']
              and sorted(drop_all) == sorted(assets), f"не взяты: {drop_f}")
        rl21 = {main: rulers_from(T, sm, main, H), conf: burn_rulers(T, sc)}
        with rules21(T):
            Rc21, C21 = ceiling_21(T, sm, rl21[main]['fold'], fr, rl21[conf], v5, H)
        n_fresh = sum(len(d['sp']) - d['B'] for D in fr.values() for d in D.values())
        check("(Ц) v2.1: основной — как v2 (пары с корзинами 1H, незнакомые), подтверждение — только свежие окна с линейкой "
              "разгона, кандидаты 10 / 8 / 5", Rc21[conf]['n_windows'] == n_fresh and Rc21[main]['by_c'][10]['unseen'] ==
              ['FILUSDT'] and Rc21[conf]['by_c'][10]['g3'] is None and set(LINE21['cands_c']) <= set(Rc21[main]['by_c'])
              and (C21 is None or C21 in LINE21['cands_c']), f"свежих окон {n_fresh}, выбор C — {C21}")
        del pdata, rec
    finally:
        RUN_DIR, OUT_DIR, DATA_5M, DATA_TF, DATA_1H, CSV_DIR = keep
    n_ok = sum(ok for _, ok in checks)
    print(f"\n{'=' * 110}\n🏁 СТЕНД v2: {n_ok} / {len(checks)} проверок пройдено | {(time.time() - t_all) / 60:.1f} мин"
          + ("" if n_ok == len(checks) else " — ❌ ЕСТЬ ПРОВАЛЫ, 'line' не запускать до разбора") + f"\n{'=' * 110}")
    if os.environ.get(_ENV + 'KEEP'):
        print(f"   папка стенда оставлена: {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return n_ok == len(checks)


if __name__ == '__main__':
    main()
