"""TrendyCoins_SMC_Range — диапазон бара для моделей SMC (v5.6; дополнение к SMC_Data и SMC_Prices)

ЗАЧЕМ. Канал «диапазон бара» на каждом шаге окна: сеть видит у бара не только его цены (close, HL/2, HLC/3), но и его
ширину. В тех же единицах, что r_t у SMC_Data (норм. = rolling_median(high − low, N=200, closed='left')):

    g_t = (high_t − low_t) / норм._t        (≈ 1 у типичного бара последних 200)

«Зеркало бара» (нулевая моделей на трёх ценах) переворачивает бар вокруг прошлого close: high и low меняются местами
зеркально, ширина бара та же. У суррогата канал тот же, что у реала (как |r| у sflip), — нулевая честная без доработок.

КАК. Бары берутся по меткам времени smc_series (t_tr / t_va); r в этих барах пересчитывается из CSV ровно как в SMC_Data и
сверяется с сохранённым бит в бит. CSV может быть длиннее smc_series (дописаны свежие бары, как у BTC / ETH для боевого
обучения) — берутся только бары smc_series, остальное не нужно. g = NaN там же, где r = NaN (пропуск сетки, мёртвый
нормировщик); где r конечен, g конечен (бар без движения — честный 0). smc_series и smc_prices не трогаются: исследовательский
набор SMC/data/ остаётся бит в бит, добавляется только новый файл. Контрольная сумма r — в meta (тренировка сверяет её при
загрузке: файл диапазона обязан относиться к тому же ряду).

ФАЙЛ: <DATA_DIR>/smc_range_<ASSET>_<TF>_N<N>.npz — g_tr, g_va (float32) и meta.
ЗАПУСК: исследовательский набор — DATA_DIR = SMC/data/, ASSETS = все 31 ассет пула (для этапа 1 хватит BTCUSDT); боевое
обучение — DATA_DIR = SMC/data_fresh/, ASSETS = ассеты накладки (там у ассета — тройка smc_series, smc_prices, smc_range).
"""

import os

os.environ['SMC_DATA_DIR'] = '/content/drive/MyDrive/TrendyCoins/SMC/data_fresh/'
os.environ['SMC_ASSETS'] = 'BTCUSDT,ETHUSDT'

import json
import time
import zlib
import numpy as np
import pandas as pd

try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass

project_path = os.environ.get('SMC_PROJECT_PATH', '/content/drive/MyDrive/TrendyCoins/')
DATA_DIR = os.environ.get('SMC_DATA_DIR', project_path + 'SMC/data/')     # исследовательский набор (не data_fresh!)
POOL_ALL = ['BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'SOLUSDT', 'ADAUSDT', 'BNBUSDT', 'BCHUSDT', 'DOTUSDT', 'UNIUSDT',
            'XRPUSDT', 'LTCUSDT', 'DASHUSDT', 'ZECUSDT', 'SPELLUSDT', 'ICPUSDT', 'NEARUSDT',
            'ZENUSDT', 'AVAXUSDT', 'FETUSDT', 'RAYUSDT', 'CAKEUSDT', 'INJUSDT', 'RUNEUSDT', 'PHAUSDT', 'QNTUSDT',
            'HBARUSDT', 'AAVEUSDT', 'VTHOUSDT', 'FILUSDT', 'LINKUSDT', 'XLMUSDT']      # боевой пул (C + H), BTC первым
ASSETS = os.environ.get('SMC_ASSETS', ','.join(POOL_ALL)).split(',')
TF, N = '5M', 200
MAX_R_MISMATCH = 0      # сколько баров с r, не совпавшим с smc_series, терпеть (0 — ни одного: CSV тот же, что у SMC_Data)


def same_array(a, b):
    """Совпадение float32-рядов бит в бит, NaN на тех же местах."""
    if len(a) != len(b):
        return False
    na, nb = np.isnan(a), np.isnan(b)
    return bool(np.array_equal(na, nb) and np.array_equal(a[~na], b[~nb]))


def prepare(asset):
    t0 = time.time()
    series = DATA_DIR + f"smc_series_{asset}_{TF}_N{N}.npz"
    if not os.path.isfile(series):
        raise FileNotFoundError(f"🍆 АЙ, ОШИБКА! нет {series} — сначала TrendyCoins_SMC_Data")
    d = np.load(series, allow_pickle=False)
    csv = project_path + "Raw Datasets/" + f"{asset}-{TF}.csv"
    df = pd.read_csv(csv, parse_dates=['timestamp'])
    high, low, close = (df[k].to_numpy(dtype=np.float64) for k in ('high', 'low', 'close'))
    date = df['timestamp']
    ts = date.to_numpy().astype('datetime64[ns]').astype(np.int64)
    if not np.all(np.diff(ts) > 0):
        raise ValueError(f"🍆 АЙ, ОШИБКА! {asset}: метки времени CSV не по возрастанию")
    print(f"\n{'=' * 78}\n📈 {asset} {TF}: {os.path.basename(csv)} | {len(df)} баров | {date.iloc[0]} → "
          f"{date.iloc[-1]}\n{'=' * 78}")

    # ровно как make_parts в SMC_Data: нормировщик по всему файлу, пропуски сетки и мёртвый нормировщик -> NaN
    roll = pd.Series(high - low).rolling(N, closed='left').median().to_numpy()
    dt = date.diff().dt.total_seconds().to_numpy()
    step = float(np.nanmedian(dt[1:]))
    gap = np.zeros(len(df), dtype=bool)
    gap[1:] = dt[1:] > step * 1.5
    d_close = np.diff(close, prepend=close[0])
    d_close[0] = 0.0
    r = np.zeros_like(d_close)
    np.divide(d_close, roll, out=r, where=roll != 0)
    bad = ~np.isfinite(roll) | (roll <= 0) | ~np.isfinite(r) | gap
    r[bad] = np.nan
    r[0] = np.nan
    g = np.full(len(close), np.nan)
    np.divide(high - low, roll, out=g, where=~bad)           # диапазон бара в единицах того же делителя, что у r

    out, crc = {}, {}
    for name, key in (('TRAINING', 'tr'), ('VALIDATION', 'va')):
        t_part, r_ser = d[f't_{key}'], d[f'r_{key}']
        idx = np.searchsorted(ts, t_part)
        idx_c = np.minimum(idx, len(ts) - 1)
        found = (idx < len(ts)) & (ts[idx_c] == t_part)
        if not found.all():
            raise ValueError(f"🍆 АЙ, ОШИБКА! {asset} {name}: {int((~found).sum())} баров smc_series нет в CSV (первый — "
                             f"{pd.Timestamp(int(t_part[~found][0]))}) — CSV не тот, на котором считался smc_series")
        r_part = r[idx_c].astype(np.float32)
        if not same_array(r_ser, r_part):
            fin = np.isfinite(r_ser) | np.isfinite(r_part)
            mism = (np.isfinite(r_ser) != np.isfinite(r_part)) | (fin & (r_ser != r_part))
            n_m = int(mism.sum())
            first = pd.Timestamp(int(t_part[np.argmax(mism)]))
            if n_m > MAX_R_MISMATCH:
                raise ValueError(f"🍆 АЙ, ОШИБКА! {asset} {name}: r из CSV не совпадает с smc_series на {n_m} барах (первый — "
                                 f"{first}) — CSV переписан другим (перекачка с правками истории?). Верни CSV, на котором "
                                 f"считался smc_series, или осознанно подними MAX_R_MISMATCH")
            print(f"   ⚠️ {name}: r не совпадает на {n_m} барах (первый — {first}), терпимо по MAX_R_MISMATCH = {MAX_R_MISMATCH}")
        g_part = g[idx_c].astype(np.float32)
        g_part[~np.isfinite(r_ser)] = np.nan                  # g там же, где r: окна у моделей с диапазоном — те же
        if not np.array_equal(np.isfinite(g_part), np.isfinite(r_ser)):
            raise ValueError(f"🍆 АЙ, ОШИБКА! {asset} {name}: диапазон не конечен там, где конечен r — разобраться с данными")
        crc[key] = zlib.crc32(r_ser.astype(np.float32).tobytes())
        out[f'g_{key}'] = g_part
        v = g_part[np.isfinite(g_part)].astype(np.float64)
        ar = np.abs(r_ser[np.isfinite(r_ser)].astype(np.float64))
        tail = len(ts) - 1 - int(idx_c[-1])
        print(f"   {name:<10} баров {len(t_part)} | r совпадает с smc_series бит в бит "
              f"{'✅' if same_array(r_ser, r_part) else '⚠️'} | диапазон / норм.: медиана {np.median(v):.3f}, p99 "
              f"{np.percentile(v, 99):.2f}, max {v.max():.1f} | бар без движения {100 * (v == 0).mean():.3f}% | "
              f"корр. |r| и диапазона {np.corrcoef(ar, v)[0, 1]:.3f}"
              + (f" | в CSV после конца части ещё {tail} баров (не нужны)" if key == 'va' and tail else ""))

    out['meta'] = np.array(json.dumps({'asset': asset, 'tf': TF, 'N': N, 'ver': 1, 'what': 'g = (high - low) / norm',
                                       'series': os.path.basename(series), 'crc': crc, 'csv_last': str(date.iloc[-1]),
                                       'created': time.strftime('%Y-%m-%d %H:%M:%S')}, ensure_ascii=False))
    fn = DATA_DIR + f"smc_range_{asset}_{TF}_N{N}.npz"
    np.savez_compressed(fn, **out)
    print(f"💾 {fn} ({os.path.getsize(fn) / 1e6:.1f} МБ) | {time.time() - t0:.0f} c")


def main():
    t0 = time.time()
    print(f"\n{'#' * 78}\n# SMC: ДИАПАЗОН БАРА | {DATA_DIR} | {len(ASSETS)} ассетов | {TF} | N={N}\n{'#' * 78}")
    failed = []
    for a in ASSETS:
        try:
            prepare(a)
        except Exception as e:
            print(f"\n{e}" if str(e).startswith('🍆') else f"\n🍆 АЙ, ОШИБКА! {a}: {type(e).__name__}: {e}")
            failed.append(a)
    print(f"\n👑 ГОТОВО: {len(ASSETS) - len(failed)} из {len(ASSETS)}, ошибок {len(failed)}"
          f"{' (' + ', '.join(failed) + ')' if failed else ''} | {time.time() - t0:.0f} c")
    print("   Дальше — TrendyCoins_SMC_Train: MODE 'check' (сверка канала диапазона), затем задания 'RGa' / 'RGb' / 'RGc'.")


main()
