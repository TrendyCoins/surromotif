"""futures_klines — свечи бессрочных фьючерсов Binance USDT-M (5M) для SuMo_Class_Flow.

ЗАПУСК — ЛОКАЛЬНО (python-binance, как спотовый загрузчик): API Binance с американских IP Colab не отвечает.
Ключи не нужны — свечи публичные.

Что пишет. По CSV на ассет в OUT_DIR, имя «<ассет>-5M.csv» (имя спотового ассета, даже если у фьючерса префикс 1000),
колонки — как у спотовых CSV проекта: timestamp (время открытия свечи, UTC), open, high, low, close, volume, close_time,
quote_av, trades, tb_base_av, tb_quote_av, ignore. Для проверки нужны volume (объём в монетах), trades (число сделок) и
tb_base_av — объём тейкер-покупок (рыночных покупок) в монетах: продажи = volume − tb_base_av.

Дозагрузка. Если файл ассета уже есть, качается только хвост после его последней свечи. Незакрытая (текущая) свеча не пишется.

Постранично — своим циклом по client.futures_klines (по LIMIT свечей за запрос, пауза PAUSE между запросами), не через
futures_historical_klines: у той в новых python-binance (1.0.37) аргумент limit — потолок ВСЕГО ответа, а не размер страницы.
Время всей загрузки (31 ассет с 2023-12-01, ~6 тыс. запросов) — порядка 40–60 минут.

Потом. Папку OUT_DIR целиком положить на Google Drive в TrendyCoins/Raw Datasets/futures/ (путь FUT_DIR в sumo_class_flow.py).
"""
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from binance.client import Client

# =====================================================================
# НАСТРОЙКИ
# =====================================================================
ASSETS = [                                    # пул 31 ассета (POOL['sets']['ALL'] кода обучения)
    'BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'SOLUSDT', 'ADAUSDT', 'BNBUSDT', 'BCHUSDT', 'DOTUSDT', 'UNIUSDT',
    'XRPUSDT', 'LTCUSDT', 'DASHUSDT', 'ZECUSDT', 'SPELLUSDT', 'ICPUSDT', 'NEARUSDT',
    'ZENUSDT', 'AVAXUSDT', 'FETUSDT', 'RAYUSDT', 'CAKEUSDT', 'INJUSDT', 'RUNEUSDT', 'PHAUSDT', 'QNTUSDT',
    'HBARUSDT', 'AAVEUSDT', 'VTHOUSDT',
    'FILUSDT', 'LINKUSDT', 'XLMUSDT',
]
ALIASES = {}                                  # ассет -> символ фьючерса, если он не <ассет> и не 1000<ассет> (по умолчанию пусто;
                                              # чужой рынок ноутбук всё равно остановит сверкой (0б))
INTERVAL = '5m'
START = '01/12/2023 00:00:00'                 # дд/мм/гггг чч:мм:сс UTC — с запасом на разгон (поиск начинается 2024-07-01)
END = None                                    # None — до последней закрытой свечи; иначе 'дд/мм/гггг чч:мм:сс'
OUT_DIR = 'futures/'
STEP_MS = 5 * 60 * 1000
LIMIT = 1500                                  # свечей за запрос — максимум API (вес запроса 10)
PAUSE = 0.3                                   # с между запросами: не больше 200 запросов (2000 веса) в минуту при лимите 2400
RETRIES = 6                                   # повторов при ошибке сети / API, пауза 5, 10, 20… с
COLS = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av',
        'tb_quote_av', 'ignore']


def to_ms(s):
    """'дд/мм/гггг чч:мм:сс' (UTC) -> миллисекунды."""
    return int(datetime.strptime(s, '%d/%m/%Y %H:%M:%S').replace(tzinfo=timezone.utc).timestamp() * 1000)


def perp_symbols(client):
    """Бессрочные фьючерсы USDT-M, которые биржа знает сейчас: {символ: статус}."""
    info = client.futures_exchange_info()
    return {s['symbol']: s.get('status', '?') for s in info['symbols']
            if s.get('contractType') == 'PERPETUAL' and s.get('quoteAsset') == 'USDT'}


def pick_symbol(asset, syms):
    """Символ фьючерса ассета: из ALIASES, тот же, что у спота, или с префиксом 1000 (у дешёвых монет). None — фьючерса нет."""
    for s in (ALIASES.get(asset), asset, '1000' + asset):
        if s and s in syms:
            return s
    return None


def klines_page(client, symbol, start_ms, end_ms):
    """Одна страница свечей (не больше LIMIT, время открытия в [start_ms, end_ms]) с повторами при ошибке."""
    for k in range(RETRIES):
        try:
            return client.futures_klines(symbol=symbol, interval=INTERVAL, startTime=start_ms, endTime=end_ms, limit=LIMIT)
        except Exception as e:                                   # сеть, 429 / 418 (лимит веса), 5xx
            if k == RETRIES - 1:
                raise
            wait = 5 * 2 ** k
            print(f"    … {symbol}: {type(e).__name__}: {str(e)[:120]} — пауза {wait} с, повтор {k + 1} из {RETRIES - 1}")
            time.sleep(wait)


def fetch(client, symbol, start_ms, end_ms):
    """Свечи с start_ms по end_ms постранично — DataFrame в формате спотовых CSV (timestamp — время открытия, naive UTC)."""
    rows, t = [], start_ms
    while t <= end_ms:
        page = klines_page(client, symbol, t, end_ms)
        if not page:
            break
        rows += page
        t = int(page[-1][0]) + STEP_MS
        if len(page) < LIMIT:
            break
        time.sleep(PAUSE)
    df = pd.DataFrame(rows, columns=COLS)
    if df.empty:
        return df
    now_ms = int(time.time() * 1000)
    df = df[df['close_time'].astype(np.int64) < now_ms].copy()     # незакрытую свечу не пишем
    for c in ('open', 'high', 'low', 'close', 'volume', 'quote_av', 'tb_base_av', 'tb_quote_av'):
        df[c] = df[c].astype(float)
    df['trades'] = df['trades'].astype(np.int64)
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
    return df


def check(df):
    """Проверки файла: время по возрастанию без повторов, бары OHLC целые, тейкер-покупки не больше объёма. Возвращает
       сводку (пропусков сетки — недостающих свечей 5M внутри файла)."""
    t = df['timestamp'].to_numpy().astype('datetime64[ms]').astype(np.int64)
    o, h, l, c = (df[k].to_numpy() for k in ('open', 'high', 'low', 'close'))
    v, b = df['volume'].to_numpy(), df['tb_base_av'].to_numpy()
    dt = np.diff(t)
    return {'rows': len(df), 'first': str(df['timestamp'].iloc[0]), 'last': str(df['timestamp'].iloc[-1]),
            'back': int((dt <= 0).sum()), 'missing': int(((dt // STEP_MS) - 1)[dt > STEP_MS].sum()),
            'bad_ohlc': int(((h < l) | (o > h) | (o < l) | (c > h) | (c < l) | (l <= 0)).sum()),
            'bad_taker': int((b > v * (1 + 1e-9) + 1e-12).sum()), 'zero_vol': int((v <= 0).sum())}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    client = Client(api_key='', api_secret='')
    syms = perp_symbols(client)
    start0 = to_ms(START)
    end = to_ms(END) if END else int(time.time() * 1000)
    span = f"с {pd.Timestamp(start0, unit='ms'):%Y-%m-%d} по {pd.Timestamp(end, unit='ms'):%Y-%m-%d %H:%M}"
    print('=' * 100)
    print(f"Фьючерсы USDT-M {INTERVAL}: {len(ASSETS)} ассетов, {span} UTC -> {OUT_DIR}")
    print('=' * 100)
    rows = []
    for asset in ASSETS:
        sym = pick_symbol(asset, syms)
        if sym is None:
            print(f"[-] {asset:<10} | бессрочного фьючерса USDT-M нет (или он делистнут) — пропуск")
            rows.append((asset, None, None))
            continue
        path = os.path.join(OUT_DIR, f"{asset}-5M.csv")
        old = None
        start = start0
        if os.path.isfile(path):
            old = pd.read_csv(path, parse_dates=['timestamp'])
            start = int(old['timestamp'].iloc[-1].value // 10 ** 6) + STEP_MS
        t1 = time.time()
        try:
            new = fetch(client, sym, start, end)
        except Exception as e:
            print(f"[!] {asset:<10} | {sym} | ошибка API: {e}")
            rows.append((asset, sym, None))
            continue
        if old is None:
            df = new
        elif new.empty:
            df = old
        else:
            df = pd.concat([old, new], ignore_index=True)
        if df.empty:
            print(f"[-] {asset:<10} | {sym} | свечей нет")
            rows.append((asset, sym, None))
            continue
        df = df.drop_duplicates('timestamp', keep='last').sort_values('timestamp').reset_index(drop=True)
        df[COLS].set_index('timestamp').to_csv(path)
        x = check(df)
        flag = 'OK' if not (x['back'] or x['bad_ohlc'] or x['bad_taker']) else 'ПРОВЕРЬ'
        print(f"[+] {asset:<10} | {sym:<14} | +{len(new):>7} свечей | всего {x['rows']:>7}: {x['first'][:16]} … "
              f"{x['last'][:16]} | пропусков {x['missing']:>5} | нулевой объём {x['zero_vol']:>4} | {time.time() - t1:5.0f} с "
              f"| {flag}")
        if flag != 'OK':
            print(f"    назад во времени {x['back']}, битых OHLC {x['bad_ohlc']}, тейкер-покупок больше объёма {x['bad_taker']}")
        rows.append((asset, sym, x))
    print('=' * 100)
    got = [r for r in rows if r[2]]
    print(f"Итого: файлов {len(got)} из {len(ASSETS)}. Без фьючерса или с ошибкой: "
          + (', '.join(a for a, s, x in rows if not x) or 'нет'))
    print(f"Дальше: папку {OUT_DIR} — на Google Drive в TrendyCoins/Raw Datasets/futures/")


if __name__ == '__main__':
    main()
