# -*- coding: utf-8 -*-
"""SuMo_Class_Eff v2 — видит ли MRI неслучайный рынок: мишени — крупные рыночные события (2026-10-24)

ЗАЧЕМ. Доказательная проверка, что MRI распознаёт неслучайный рынок. Судья — не суррогаты (на них MRI учили), а ситуации, где
теория ждёт менее эффективный, то есть менее случайный рынок; главные мишени — крупные рыночные события: дни стресса и
макрорелизы США. Эффективный рынок — тот, где ход цены не предсказать по прошлым ходам: направления шагов — как монетка при тех
размерах свечей, что были; ровно это — нулевая MRI (суррогаты: те же свечи, случайные направления). Гипотеза адаптивного рынка
(Lo, 2004): эффективность меняется с условиями. Точка сравнения — классические меры эффективности (ими её меряют экономисты).
Вывод для трейдеров, если подтвердится: MRI — мера доверия к анализу рынка любыми способами: он говорит, когда рынок
неслучаен, то есть когда искать в нём узоры есть смысл. Прогноза цены и направления здесь нет.

СИТУАЦИИ И ОЖИДАНИЯ (записаны до прогона v1, 2026-10-24; в v2 те же): в ситуации A рынок менее эффективен, чем в B, поэтому
неслучайность e (ниже) в A больше.
  (Н1) Тонкие часы против ликвидных. Час недели (UTC) — по объёму BTC-фьючерса USDT-M: у каждого часа — его объём к медиане
       часа своей недели, медиана по неделям до 2024-07-01 (до обоих периодов); тонкие — треть часов недели с наименьшей
       долей (56 из 168), ликвидные — треть с наибольшей. Одна разметка на все монеты. Ожидание: в тонкие часы
       неслучайность больше (меньше ликвидности — цена медленнее переваривает заявки).
  (Н2) Дни стресса против остальных. Стресс — день (UTC), у которого реализованная дисперсия BTC (сумма квадратов log-ходов
       баров дня, тот же ТФ) к медиане 30 прошлых дней — в верхних 10% своего периода. Ожидание: в дни стресса
       неслучайность больше (каскады и выносы; у Noda эффективность крипты падает на обвалах).
  (Н3) 4 часа после макрорелиза США против тех же часов (ET) в будни без релизов. Релизы — CPI и NFP (08:30 ET), решение
       FOMC (14:00 ET); даты — официальные календари BLS и ФРС с переносами шатдауна 2025 (список EVENTS). Бар — в окне
       релиза, если закрылся в (τ; τ + 4 ч], τ — время релиза: на 1H первый бар окна — тот, внутри которого релиз.
       Сравнение — внутри слота (08:30 и 14:00 отдельно), итог — среднее по слотам с весами числа баров релиза; дни с любым
       релизом в пару не идут. Ожидание: после релиза неслучайность больше (цена ищет новую цену — снос или перехлёст).
  Справочно, без вердикта: (Н4) монеты — средняя неслучайность монеты против log медианы суточного объёма её фьючерса
  (Spearman; литература спорит — Plastun et al. 2023 зависимости от размера не нашли); карта неслучайности по часам недели.

ЦВЕТ MRI. Модель ставит окну из 64 баров оценку; линейка переводит её в z — место среди оценок случайных окон (суррогатов
sflip4 источника): z = 0 — как у типичного случайного окна, больше — похоже на реальное. Линия M сравнивает z окна с двумя
колоколами: «как обычно у этой монеты» (центр m — среднее z её прошлых окон с поправкой вниз на разброс, не меньше 0; окон
не больше 4032, кроме 64 последних — они перекрываются с текущим; ширина s) и «как у случайного рынка» (центр 0, ширина
1); log M — логарифм отношения колоколов в точке z. Оранжевый (log M > 0) — окно ближе к обычному для монеты, то есть к
неслучайному; синий (< 0) — ближе к случайному. Цвет бара c — log M под потолком (5M — 10, 1H — 8), делённый на его
логарифм: от −1 до +1 (формула сайта 4032·k0.5, как в Legs).

МЕРЫ. На каждом баре t — окно из 64 баров, кончающееся на t. Только бары, где есть цвет MRI и годны все меры; у всех мер — одни
и те же бары.
  MRI   — цвет реала c минус цвет близнеца на той же линии: база m, s — от реала, а z — у того же окна, у которого направление
          каждой четвёрки баров задано монеткой (sflip4 — случайный рынок самой линейки; z — из кэшей Flow). Свечи и база —
          общие, поэтому мера показывает только вклад порядка ходов: плюс — фактический порядок делает цвет оранжевее, чем
          случайные направления на тех же свечах.
  AC1   — автокорреляция соседних ходов (связь хода бара с ходом прошлого бара), мера |AC1|.
  VR16  — variance ratio: средний квадрат хода за 16 баров к 16 средним квадратам хода за бар; у случайного рынка 1, у
          трендового больше, у пилы меньше; мера |ln VR16|. Показатель Хёрста — через него: H = 0.5 + ln VR16 / (2 ln 16)
          (у случайного 0.5) — справочно.
  VR4x4 — то же на четвёрках баров (ход за 16 баров к четырём ходам за 4 бара) — на масштабе блоков нулевой MRI (sflip4).
  Ходы — log close; суммы без вычитания среднего (у ходов 5M и 1H оно ~0); |ln VR| обрезается на ln 4. У классики e =
  мера(реал) − мера(близнец): у AC1 и VR16 случаен знак каждого бара, у VR4x4 — знак каждой четвёрки (размеры те же). Близнец
  убирает то, что дают одни размеры свечей (в дни стресса и на релизах свечи крупные и рваные).
  Справочно: MRI без линии — z реала минус z того же окна sflip4 (оценка модели, без базы); MRI с близнецом sflip1 (знак
  каждого бара — монетка) на линии реала; мера v1 — |c| реала минус |c| близнеца sflip1 на его собственной линии; MRI только
  BTC; цвет реала и близнеца, доли оранжевых и z суррогата в A и B (двигают ли оценку модели одни размеры свечей); 4 ч до
  релиза против тех же часов без релиза (началось ли до релиза); доля синих баров у реала и у близнеца на линии реала.

ПОЧЕМУ v2 (2026-10-24: после прогона v1, до прогона v2). В мере MRI v1 (|c| реала − |c| близнеца sflip1 на его собственной
линии) два дефекта дизайна. (1) Близнец не контроль: его линия строится на его же случайной истории — база m = 0 (формула не
даёт m < 0), колокола совпадают, и log M ≈ 0 при любых свечах (|c| близнеца на данных ≈ 0.02); размеры свечей он не вычитает.
(2) |c| складывает синий с оранжевым, а синий — «ближе к случайному»: окно с z = 0 на монете с m = 1 даёт синий M ≈ 1.65.
Стенд v1 этого не поймал: в его мирах MRI заменял |ln VR16|. Игрушка на формуле линии (z — гладкий шум, общая часть «размеров
свечей» у реала и близнеца): «крупные свечи сдвигают z всех окон на +0.5» — v1 даёт Δ +0.103, «порядок ходов добавляет +0.5 к z
реала» — +0.105, не различает; мера v2 — −0.018 и +0.208, а на «рынок как монетка» (z реала −1) — −0.303. В v2 ситуации,
данные, классика и правила те же; заменена только мера MRI (мера v1 — справочно), добавлены справки и ярлыки ([v2] ниже).

СТАТИСТИКА. Δ = среднее e в A − в B (бары всех монет вместе); 95% интервал — бутстрепом по неделям (все монеты недели вместе,
500 повторов, одни и те же повторы у всех мер периода); t = Δ / ошибка бутстрепа — «резкость» меры. Периоды — как в Legs:
поиск — 'stop' на C, окна TUNE (2024-07-01 … 2025-06-30), подтверждение — 'refit' на ALL, окна VAL (с 2025-07-01); оба вне
выборки. ТФ — 5M и 1H.

ПРАВИЛА (записаны до прогона v1; в v2 — те же для новой меры MRI; дополнения [v2] — до прогона v2):
  «мера видит Нk на ТФ» — Δ > 0 и нижний край интервала > 0 на обоих периодах; [v2] «обратный знак» — Δ < 0 и верхний край
  интервала < 0 на обоих;
  «MRI резче классики» — MRI видит Нk, и его t больше t каждой классической меры (AC1, VR16, VR4x4) на обоих периодах;
  «классика резче» — какая-то классическая мера видит Нk, и её t больше t MRI на обоих периодах; иначе — «наравне».
  [v2] Ярлык Нk: «MRI видит, классика — нет» / «видят MRI и классика» / «видит только классика» / «MRI — обратный знак» /
  «не видит никто». Если MRI видит Нk, а у MRI без линии Δ ≤ 0 хотя бы на одном периоде, — «через базу линии» (цвет видит
  ситуацию за счёт истории монеты, а не текущего окна), иначе — «текущим окном».
  [v2] Мишени — крупные события, Н2 и Н3. Вывод для трейдеров («MRI говорит, когда рынок неслучаен») подтверждён на ТФ, если
  MRI видит хотя бы одну мишень на этом ТФ текущим окном; «и классика этого не видит» — если ни одна классическая мера ни
  одну из этих мишеней не видит. Справочные меры, Н4, карта и 4 ч до релиза — без вердикта.
  [v2] Справочно, к описанию цветов: синий читается как «похоже на случайный», если у тех же окон со случайными
  направлениями четвёрок на линии реала синих больше половины и больше, чем у реала, на обоих периодах обоих ТФ.

СТЕНД (MODE 'stand', синтетика, до данных): (ст1) меры окна классики = прямой счёт окно за окном; близнецы сохраняют размеры
(бар / четвёрку); на случайном ряде e ~ 0; на ряде с памятью знака (AR(1) ±0.2) |AC1| и |ln VR16| реала больше, чем у
близнеца, а знаки AC1 и ln VR16 — как у памяти; на ряде с памятью четвёрок VR4x4 > 1; мера на баре t — только из баров <= t.
(ст2) Час недели, профиль объёма, ET -> UTC через переходы на летнее время, календарь, окна релизов и [v2] «4 ч до релиза»,
их пары, дни стресса. (ст3) Миры цен, классика (4 монеты × 2 × 26 недель 5M): с подсаженной неэффективностью в ситуациях
(знак хода держится за состоянием, которое меняется в среднем раз в 16 баров) каждая мера видит каждую Нk хотя бы в 9 сидах
из 10 (VR4x4 на Н3 — справочно); без неё, но с крупными и рваными свечами в ситуациях (рывки в дни стресса, крупный бар
релиза), — ложных «видит» не больше 2 на все меры и гипотезы 20 сидов, z = Δ / ошибка у каждой меры с разбросом от 0.7 до 1.6.
[v2] (ст4) Линия M: копия формулы = FL.line_m бит в бит; база окна — только из окон до него; log M близнеца = формула в точке
его z при базе реала. Миры z окон (общая часть z — «размеры свечей окна» — у реала и близнецов одна, z реала в среднем 1):
«порядок» (z реала +0.5 в ситуациях) — MRI и MRI без линии видят каждую Нk хотя бы в 9 сидах из 10; «размеры» (z всех +0.5
в ситуациях) — их ложных «видит» не больше 2 на все Нk 20 сидов, z с разбросом от 0.7 до 1.6, а мера v1 «видит» Н2 хотя бы в
15 сидах (дефект v1 записан стендом); «монетка» (z реала −0.5 в ситуациях) — у MRI обратный знак на каждой Нk хотя бы в 9
сидах из 10 и ни одного «видит». (ст5) Сквозной путь на синтетических файлах research, кэшах Flow и CSV фьючерса; цвета реала
и близнеца v1 — бит в бит как у Legs.
[Правка v1 2026-10-24, до данных, по стенду: в мире «плюс» память знака была слабее (kappa 0.5), релиз — раз-два в неделю, а
крупный бар релиза и рывки стояли в обоих мирах. Крупный бар, попав в окно из 64 баров, сам решает его AC1 и VR, и память за
ним не видна ни одной мере; окно релиза — 4 ч, то есть окно MRI всегда наполовину из баров до релиза, и при ~40 релизах за
период эффект тонет. Теперь: в «плюсе» — только память знака (kappa 0.8) и крупные свечи дней стресса (по ним RV находит дни),
три релиза в неделю; размеры — в мире без связи, где их снимает близнец. VR4x4 окна релизов почти не видит и при kappa 0.8
(4 сида из 10: на четвёрках за 4 ч мало точек) — на Н3 у него справочно. Для данных это значит: в Н3 ~35 релизов на период —
меньше всего мощности, а бар релиза глушит AC1 и VR своего окна.]

РЕЖИМЫ (MODE): 'stand' — синтетика (CPU, пара минут); 'run' — кэши Flow, файлы research, CSV фьючерсов (CPU; на 5M — сессия
с большой памятью, как у Legs; ~5–15 мин). GPU не нужен.
ЗАПУСК (Colab): файл — в SMC/ рядом с sumo_class_flow.py и sumo_class_legs.py (v2 заменяет v1, путь тот же); нужны кэши Flow
(SMC/runs/flow/cache/flow_{5M,1H}_{tune,val}_f0.2.npz), файлы research (SMC/data/, SMC/data_1h/) и CSV фьючерсов
(Raw Datasets/futures/, обязателен BTCUSDT-5M.csv). Итог — SMC/runs/eff/ (лог, json, рисунок).

ИТОГ v1 (прогон 2026-10-24, Colab, 4.4 мин; стенд 18 / 18). По правилам: на обоих ТФ Н2 — MRI «видит» и резче классики (t:
5M +3.0 / +4.5, 1H +1.7 / +4.0; классика — от −3.3 до +0.6), Н1 и Н3 не видит никто. Но мера MRI v1 вывода не даёт (см.
«ПОЧЕМУ v2»): её Н2, отрицательная Н3 на 5M (t −2.7 / −2.3) и Н4 не трактуются. Классика (близнец честный): на 5M и 1H лишней
неслучайности ни в одной ситуации; её неслучайность здесь — пила соседних баров (AC1 ≈ −0.02 … −0.03, Хёрст ≈ 0.44), в дни
стресса пила тонет в крупных ходах (5M: AC1 −0.004 против −0.020). Н4 на 5M: по классике неслучайнее неликвидные монеты
(Spearman с log объёма: AC1 −0.81, VR16 −0.70), |c| MRI — наоборот (+0.68); на 1H связей нет. Лог, json и рисунок —
SMC/runs/eff/ (eff_20261024_120939).
ИТОГ v2 (прогон 2026-10-24, Colab, 2.7 мин; стенд 24 / 24). По правилам вывод для трейдеров («MRI говорит, когда рынок
неслучаен») подтверждён на 5M и 1H, и классика этого не видит. Н2 (дни стресса) — «MRI видит, классика — нет (текущим окном),
MRI резче классики» на 5M (t +2.7 / +2.5) и 1H (t +2.3 / +4.1); Н1 (тонкие часы) — MRI видит на 5M (t +3.0 / +3.8; только BTC
+5.8 / +4.7), на 1H — нет; Н3 (после релизов) — не видит никто. Механика Н2: окна реала в дни стресса ближе к реальному рынку
(z +0.12 на 5M, +0.42 / +0.55 на 1H), а те же свечи со случайными направлениями четвёрок — ближе к суррогату (z −0.04 … −0.13):
размеры свечей тянут в другую сторону. Справочно: после релизов цвет на 5M сереет (поиск +0.06 против +0.29, t −3.7;
подтверждение t −1.3 — правило не прошло), за 4 ч до релиза разницы нет; Н4 на 5M — вклад порядка ходов выше у ликвидных монет
(Spearman +0.60, без линии +0.74), доля оранжевых от ликвидности не зависит (−0.01), у классики наоборот (AC1 −0.81).
Цвета (смысл от 2026-10-24: оранжевый — ситуация чаще на реальном рынке, серебро, бывший синий, — на суррогате, серый —
одинаково): на линии реальной монеты те же окна со случайными направлениями четвёрок (суррогат) — серебро в 68–72% баров, реал —
в 24–32%; оранжевый у реала — 68–76%, у суррогата — 28–32%. «Синий» в строках лога v2 — это серебро. Лог, json и рисунок —
SMC/runs/eff/ (eff_20261024_151406). Что рынок делает после бара каждого цвета — SuMo_Class_Anat.
"""

import os
import sys
import json
import math
import time
import tempfile
import contextlib
import importlib.util
import numpy as np
import pandas as pd

try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

import matplotlib
if not IN_COLAB:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =====================================================================
# НАСТРОЙКИ
# =====================================================================
MODE = 'stand'          # 'stand' (синтетика) -> 'run'

project_path = '/content/drive/MyDrive/TrendyCoins/'
LEGS_PY = project_path + 'SMC/sumo_class_legs.py'      # SuMo_Class_Legs v1: бары research, цвет сайта по кэшам Flow
FLOW_PY = project_path + 'SMC/sumo_class_flow.py'      # SuMo_Class_Flow v1: кэши оценок, линейка, линия M, CSV фьючерсов
RUN_DIR = project_path + 'SMC/runs/'
DATA_5M = project_path + 'SMC/data/'                   # research 5M
DATA_1H = project_path + 'SMC/data_1h/'                # research 1H (SuMo_Class_TF v2.1)
FUT_DIR = project_path + 'Raw Datasets/futures/'       # CSV фьючерсов USDT-M 5M (research/futures_klines.py)

EFF = {
    'tfs': ('5M', '1H'),
    'periods': {'search': 'tune', 'confirm': 'val'},   # как в Legs: источник кэша Flow -> часть
    'tune_start': '2024-07-01 00:00:00',
    'ceiling': {'5M': 10.0, '1H': 8.0},     # потолок цвета — как в Legs (сверяется при загрузке)
    'L': 64,                                # окно мер — как окно MRI
    'vr_cap': math.log(4.0),                # |ln VR| обрезается на ln 4
    'hours_n': 56,                          # тонких и ликвидных часов недели — по трети (56 из 168)
    'profile_asset': 'BTCUSDT',
    'profile_before': '2024-07-01 00:00:00',   # профиль объёма — по неделям до обоих периодов
    'profile_min_weeks': 20,
    'profile_min_hours': 160,               # неделя годна, если в ней >= 160 полных часов из 168
    'stress_asset': 'BTCUSDT',
    'stress_q': 0.9,                        # стресс — верхние 10% отношения RV дня к медиане прошлых дней (в периоде)
    'stress_lookback': 30,                  # прошлых календарных дней
    'stress_min_days': 20,                  # годных среди них — не меньше
    'day_min_share': 0.7,                   # день годен, если в нём >= 70% баров ТФ
    'news_h': 4,                            # окно релиза — 4 ч после него
    'slots': {'08:30': (8 * 60 + 30, 12 * 60 + 30), '14:00': (14 * 60, 18 * 60)},   # ET: (начало; конец] по концу бара, мин
    'n_boot': 500,
    'seed': 20261024,
}
# Макрорелизы США, время — ET (America/New_York). Источники: BLS — https://www.bls.gov/schedule/2024/home.htm,
# https://www.bls.gov/schedule/2025, https://www.bls.gov/schedule/2026/home.htm; переносы шатдауна 2025 —
# https://www.bls.gov/bls/2025-lapse-revised-release-dates.htm (CPI и NFP за октябрь 2025 отменены); ФРС —
# https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm (заявление — 14:00 ET в последний день заседания).
# Сверено 2026-10-24. Релизы после конца данных просто не попадут в периоды.
EVENTS = (
    # CPI, 08:30 ET
    ('2024-07-11', '08:30', 'CPI'), ('2024-08-14', '08:30', 'CPI'), ('2024-09-11', '08:30', 'CPI'),
    ('2024-10-10', '08:30', 'CPI'), ('2024-11-13', '08:30', 'CPI'), ('2024-12-11', '08:30', 'CPI'),
    ('2025-01-15', '08:30', 'CPI'), ('2025-02-12', '08:30', 'CPI'), ('2025-03-12', '08:30', 'CPI'),
    ('2025-04-10', '08:30', 'CPI'), ('2025-05-13', '08:30', 'CPI'), ('2025-06-11', '08:30', 'CPI'),
    ('2025-07-15', '08:30', 'CPI'), ('2025-08-12', '08:30', 'CPI'), ('2025-09-11', '08:30', 'CPI'),
    ('2025-10-24', '08:30', 'CPI'),     # сентябрь 2025 — перенос (шатдаун); октябрь 2025 — отменён
    ('2025-12-18', '08:30', 'CPI'),     # ноябрь 2025 — перенос с 12-10
    ('2026-01-13', '08:30', 'CPI'), ('2026-02-13', '08:30', 'CPI'), ('2026-03-11', '08:30', 'CPI'),
    ('2026-04-10', '08:30', 'CPI'), ('2026-05-12', '08:30', 'CPI'), ('2026-06-10', '08:30', 'CPI'),
    ('2026-07-14', '08:30', 'CPI'), ('2026-08-12', '08:30', 'CPI'), ('2026-09-11', '08:30', 'CPI'),
    ('2026-10-14', '08:30', 'CPI'),
    # NFP (Employment Situation), 08:30 ET
    ('2024-07-05', '08:30', 'NFP'), ('2024-08-02', '08:30', 'NFP'), ('2024-09-06', '08:30', 'NFP'),
    ('2024-10-04', '08:30', 'NFP'), ('2024-11-01', '08:30', 'NFP'), ('2024-12-06', '08:30', 'NFP'),
    ('2025-01-10', '08:30', 'NFP'), ('2025-02-07', '08:30', 'NFP'), ('2025-03-07', '08:30', 'NFP'),
    ('2025-04-04', '08:30', 'NFP'), ('2025-05-02', '08:30', 'NFP'), ('2025-06-06', '08:30', 'NFP'),
    ('2025-07-03', '08:30', 'NFP'), ('2025-08-01', '08:30', 'NFP'), ('2025-09-05', '08:30', 'NFP'),
    ('2025-11-20', '08:30', 'NFP'),     # сентябрь 2025 — перенос (шатдаун); октябрь 2025 — отменён
    ('2025-12-16', '08:30', 'NFP'),     # ноябрь 2025 (с данными за октябрь)
    ('2026-01-09', '08:30', 'NFP'), ('2026-02-11', '08:30', 'NFP'), ('2026-03-06', '08:30', 'NFP'),
    ('2026-04-03', '08:30', 'NFP'), ('2026-05-08', '08:30', 'NFP'), ('2026-06-05', '08:30', 'NFP'),
    ('2026-07-02', '08:30', 'NFP'), ('2026-08-07', '08:30', 'NFP'), ('2026-09-04', '08:30', 'NFP'),
    ('2026-10-02', '08:30', 'NFP'),
    # FOMC — заявление, 14:00 ET
    ('2024-07-31', '14:00', 'FOMC'), ('2024-09-18', '14:00', 'FOMC'), ('2024-11-07', '14:00', 'FOMC'),
    ('2024-12-18', '14:00', 'FOMC'),
    ('2025-01-29', '14:00', 'FOMC'), ('2025-03-19', '14:00', 'FOMC'), ('2025-05-07', '14:00', 'FOMC'),
    ('2025-06-18', '14:00', 'FOMC'), ('2025-07-30', '14:00', 'FOMC'), ('2025-09-17', '14:00', 'FOMC'),
    ('2025-10-29', '14:00', 'FOMC'), ('2025-12-10', '14:00', 'FOMC'),
    ('2026-01-28', '14:00', 'FOMC'), ('2026-03-18', '14:00', 'FOMC'), ('2026-04-29', '14:00', 'FOMC'),
    ('2026-06-17', '14:00', 'FOMC'), ('2026-07-29', '14:00', 'FOMC'), ('2026-09-16', '14:00', 'FOMC'),
    ('2026-10-28', '14:00', 'FOMC'),
)
STAND = {
    'assets': 4, 'weeks': 26,               # мир стенда: 4 монеты 5M, два периода по 26 недель
    't0': '2024-01-01 00:00:00',            # понедельник; оба перехода ET на летнее / зимнее время — внутри
    'seeds_plus': 10, 'seeds_null': 20,
    'kappa': 0.8,                           # «плюс»: в ситуациях P(знак хода = состояние) = (1 + kappa) / 2
    'flip': 1 / 16,                         # состояние меняется в среднем раз в 16 баров
    'stress_share': 0.12, 'stress_vol': 2.5,   # дни стресса — 12% дней после первого месяца (верхние 10% — все из них),
                                            # свечи в 2.5 раза крупнее
    'jump_p': 0.02, 'jump': 6.0,            # в дни стресса — рывки: 2% баров в 6 раз крупнее
    'news_bar': 6.0,                        # первый бар после релиза — в 6 раз крупнее
    'need_plus': 9, 'false_max': 2, 'calib': (0.7, 1.6), 'need_leak': 15,
    # (ст4) миры линии M: z окон по барам (окно кончается на баре); общая часть z — «размеры свечей окна» — у реала и
    # близнецов одна, своя — у каждого; z гладкие (окна подряд перекрываются)
    'z_rho': 1 - 1 / 32,                    # память гладкого шума — ~32 окна
    'z_level': 1.0,                         # z реала в среднем 1: рынок неслучаен, база линии m > 0
    'z_common': 0.6, 'z_own': 0.8,          # общая и своя части (дисперсия z — 1)
    'z_s1': -0.2,                           # сдвиг z близнеца sflip1 (как в сквозном пути)
    'd_order': 0.5,                         # «порядок»: в ситуациях z реала выше
    'd_size': 0.5,                          # «размеры»: в ситуациях выше z всех трёх
    'd_coin': 0.5,                          # «монетка»: в ситуациях z реала ниже
}
EFF_VER = 2
_ENV = 'SMCEFF_'                                       # переменные окружения — для стенда вне Colab
MODE = os.environ.get(_ENV + 'MODE', MODE)
LEGS_PY = os.environ.get(_ENV + 'LEGS_PY', LEGS_PY)
FLOW_PY = os.environ.get(_ENV + 'FLOW_PY', FLOW_PY)
RUN_DIR = os.environ.get(_ENV + 'RUN_DIR', RUN_DIR)
OUT_DIR = os.environ.get(_ENV + 'OUT_DIR', RUN_DIR + 'eff/')
DATA_5M = os.environ.get(_ENV + 'DATA_5M', DATA_5M)
DATA_1H = os.environ.get(_ENV + 'DATA_1H', DATA_1H)
FUT_DIR = os.environ.get(_ENV + 'FUT_DIR', FUT_DIR)

HOUR_NS, DAY_NS = 3600 * 10 ** 9, 86400 * 10 ** 9
WEEK_NS = 7 * DAY_NS
BAR_NS = {'5M': 300 * 10 ** 9, '1H': 3600 * 10 ** 9}
ET = 'America/New_York'
MEASURES = ('MRI', 'AC1', 'VR16', 'VR4x4')            # меры с вердиктом
CLASSIC = ('AC1', 'VR16', 'VR4x4')
MRI_REF = ('MRIz', 'MRI1', 'MRIv1')                    # справочные меры MRI (без линии, близнец sflip1, мера v1)
MRI_COLS = ('c', 'ct', 'cx4', 'cx1')                   # цвета, без которых бара в кадре нет
HYPS = {'H1': ('тонкие часы против ликвидных', 'больше в тонкие часы'),
        'H2': ('дни стресса против остальных', 'больше в дни стресса'),
        'H3': ('4 ч после релиза против тех же часов без релиза', 'больше после релиза')}
PRE = 'H3pre'                                          # справочно: 4 ч до релиза против тех же часов без релиза
TARGETS = ('H2', 'H3')                                 # мишени — крупные рыночные события
LABEL = {'MRI': 'MRI', 'MRIz': 'MRI без линии', 'MRI1': 'MRI, близнец sflip1', 'MRIv1': 'мера v1 (|c|)',
         'MRI_btc': 'MRI, только BTC', 'AC1': 'AC1', 'VR16': 'VR16', 'VR4x4': 'VR4x4'}
C_INK, C_INK2, C_GRID, C_BG = '#0b0b0b', '#52514e', '#e1e0d9', '#fcfcfb'
C_MEAS = {'MRI': '#eb6834', 'MRIz': '#a8431b', 'AC1': '#2a78d6', 'VR16': '#e87ba4', 'VR4x4': '#6250d6'}   # цвета на рисунке
LG = None                                              # модуль SuMo_Class_Legs (load_mods)
FL = None                                              # модуль SuMo_Class_Flow (через Legs)


# =====================================================================
# МЕЛОЧИ
# =====================================================================
def _f(x, fmt, width=None, dash='—'):
    try:
        ok = x is not None and np.isfinite(x)
    except TypeError:
        ok = False
    s = f"{x:{fmt}}" if ok else dash
    return f"{s:>{width}}" if width else s


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    return x


def save_json(path, out):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(_jsonable(out), fh, ensure_ascii=False, indent=1)


def load_mods():
    """SuMo_Class_Legs модулем (он подгружает SuMo_Class_Flow): бары research (load_bars), цвет сайта (color_of,
       windows_to_bars), кэши Flow (load_cache, src_logits), линейка и линия M, CSV фьючерсов. Пути — этого файла."""
    global LG, FL
    if LG is not None:
        return LG, FL
    for p, what in ((LEGS_PY, 'sumo_class_legs.py (SuMo_Class_Legs v1)'), (FLOW_PY, 'sumo_class_flow.py (SuMo_Class_Flow v1)')):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"🍆 АЙ, ОШИБКА! нет {p} — положи туда {what}")
    spec = importlib.util.spec_from_file_location('sumo_class_legs', LEGS_PY)
    lg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lg)
    need = ('load_flow', 'load_bars', 'color_of', 'windows_to_bars', 'source_colors', 'write_synth_research', 'LEGS',
            'LEGS_VER')
    miss = [k for k in need if not hasattr(lg, k)]
    if miss or getattr(lg, 'LEGS_VER', None) != 1:
        raise ImportError(f"🍆 АЙ, ОШИБКА! {LEGS_PY}: нужен SuMo_Class_Legs v1" + (f" (нет {', '.join(miss)})" if miss else ''))
    for k in ('periods', 'tune_start', 'ceiling'):
        if lg.LEGS[k] != EFF[k]:
            raise ValueError(f"🍆 АЙ, ОШИБКА! LEGS['{k}'] = {lg.LEGS[k]} не совпал с EFF['{k}'] = {EFF[k]}")
    lg.FLOW_PY, lg.RUN_DIR, lg.DATA_5M, lg.DATA_1H = FLOW_PY, RUN_DIR, DATA_5M, DATA_1H
    fl = lg.load_flow()
    need_f = ('read_fut_csv', 'fut_dir_diag', 'save_cache', 'load_cache', 'src_logits', 'ruler', 'to_z', 'line_m', 'FLOW')
    miss = [k for k in need_f if not hasattr(fl, k)]
    if miss:
        raise ImportError(f"🍆 АЙ, ОШИБКА! {FLOW_PY}: нет {', '.join(miss)} — нужен SuMo_Class_Flow v1")
    fl.FUT_DIR = FUT_DIR
    LG, FL = lg, fl
    return LG, FL


# =====================================================================
# ЛИНИЯ M НА ДВОИХ: база — от реала, точка — z близнеца
# =====================================================================
def line_parts(z, z_cur=None):
    """Линия M ряда окон подряд — копия FL.line_m (research/sumo_class_flow.py, формула сайта 4032·k0.5; стенд сверяет бит в
       бит). База окна p — m (сжатое среднее z, не меньше 0) и s (разброс z, от s_lo до 1) — из истории z: окна до p − lag,
       не больше L штук. log M — отношение колоколов N(m, s²) и N(0, 1) в точке z_cur[p]; без z_cur — в точке z[p] (ровно
       FL.line_m). Возвращает (log M, m, s); NaN до окна show_from."""
    P = FL.FLOW['line']
    z = np.asarray(z, dtype=np.float64)
    zx = z if z_cur is None else np.asarray(z_cur, dtype=np.float64)
    c1 = np.concatenate([[0.0], np.cumsum(z)])
    c2 = np.concatenate([[0.0], np.cumsum(z * z)])
    p = np.arange(len(z), dtype=np.int64)
    hi = p - P['lag'] + 1
    lo = np.maximum(0, hi - P['L'])
    n = np.maximum(hi - lo, 0)
    hc = np.maximum(hi, lo)
    s1, s2 = c1[hc] - c1[lo], c2[hc] - c2[lo]
    nn = np.maximum(n, 1)
    mh = s1 / nn
    vh = np.maximum(s2 / nn - mh * mh, 0.0)
    okn = n > 0
    ts = np.where(okn, n, 1)
    m = np.where(okn, s1 / ts, 0.0)
    s2v = np.where(okn, (n * vh + 0.0) / ts, 1.0)
    m = np.where(okn, np.maximum(m - P['k'] * np.sqrt(s2v) * np.sqrt(P['tau'] / ts), 0.0), 0.0)
    s = np.clip(np.sqrt(s2v), P['s_lo'], 1.0)
    zc = np.where(s < 1.0, np.minimum(zx, m / np.maximum(1.0 - s * s, 1e-12)), zx)
    lm = -(zc - m) ** 2 / (2 * s * s) - np.log(s) + zc * zc / 2
    show = p >= P['show_from']
    return np.where(show, lm, np.nan), np.where(show, m, np.nan), np.where(show, s, np.nan)


def line_colors(z, z4, z1, ceil):
    """Цвета и сырьё MRI ряда окон подряд: c — цвет реала (как у сайта); cx4, cx1 — цвет на линии реала (его база) в точке
       z близнеца sflip4 / sflip1 того же окна; ct — цвет близнеца sflip1 на его собственной линии (мера v1); z, z4, z1;
       m — база линии реала."""
    lm, m, _ = line_parts(z)
    return {'c': LG.color_of(lm, ceil), 'cx4': LG.color_of(line_parts(z, z4)[0], ceil),
            'cx1': LG.color_of(line_parts(z, z1)[0], ceil), 'ct': LG.color_of(FL.line_m(z1), ceil),
            'z': np.asarray(z, dtype=np.float64), 'z4': np.asarray(z4, dtype=np.float64),
            'z1': np.asarray(z1, dtype=np.float64), 'm': m}


# =====================================================================
# МЕРЫ ОКНА: AC1, VR16, VR4x4 у реала и близнецов (векторно, только бары <= t)
# =====================================================================
def _roll(x, w):
    """Сумма w элементов, кончающихся на i (NaN, пока их меньше w). NaN во входе считается нулём."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    cs = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1:] = cs[w:] - cs[:-w]
    return out


def _back(x, s):
    """x[i − s] (нули до начала ряда)."""
    out = np.zeros(len(x))
    if s < len(x):
        out[s:] = x[:len(x) - s]
    return out


def _stride_sum(x, n, step=4):
    """Σ x[i − step·k], k = 0 … n − 1 (свои суммы у каждой фазы i mod step)."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    out = np.full(len(x), np.nan)
    for p in range(step):
        v = x[p::step]
        cs = np.concatenate([[0.0], np.cumsum(v)])
        o = np.full(len(v), np.nan)
        if len(v) >= n:
            o[n - 1:] = cs[n:] - cs[:-n]
        out[p::step] = o
    return out


def twin_signs(n, rng):
    """Знаки близнецов классических мер: s1 — у каждого бара (AC1, VR16), sb — у четвёрки, кончающейся на баре (VR4x4)."""
    return np.where(rng.random(n) < 0.5, -1.0, 1.0), np.where(rng.random(n) < 0.5, -1.0, 1.0)


def classics(c, signs):
    """AC1, ln VR16, ln VR4x4 окна из L ходов, кончающегося на баре t (ход бара — log close к прошлому бару), у реала
       (суффикс _r) и близнеца (_t). NaN — у первых L баров ряда и у окон без движения."""
    L = EFF['L']
    s1, sb = signs
    n = len(c)
    r = np.zeros(n)
    r[1:] = np.diff(np.log(np.asarray(c, dtype=np.float64)))
    out = {}
    with np.errstate(divide='ignore', invalid='ignore'):
        for tag, rr in (('r', r), ('t', r * s1)):
            sq = _roll(rr * rr, L)                                  # Σ r² окна
            pr = _roll(np.r_[0.0, rr[1:] * rr[:-1]], L - 1)         # Σ r_i·r_(i−1): L − 1 пар внутри окна
            R16 = np.nan_to_num(_roll(rr, 16))                      # ход за 16 баров, кончающийся на баре
            num = _roll(R16 * R16, L - 15) / (L - 15)               # среднее по L − 15 таким ходам внутри окна
            out['ac_' + tag] = pr / sq
            out['lv16_' + tag] = np.log(num / (16.0 * sq / L))
        R4 = np.nan_to_num(_roll(r, 4))                             # четвёрка, кончающаяся на баре
        nb = L // 4                                                 # четвёрок в окне
        den = _stride_sum(R4 * R4, nb) * 4.0 / nb                   # 4 · средний квадрат четвёрки
        for tag, R4x in (('r', R4), ('t', R4 * sb)):
            R16b = R4x + _back(R4x, 4) + _back(R4x, 8) + _back(R4x, 12)     # 16 баров — четыре четвёрки подряд
            num = _stride_sum(R16b * R16b, nb - 3) / (nb - 3)
            out['lv44_' + tag] = np.log(num / den)
    for k in out:
        v = out[k]
        v[:L] = np.nan
        v[~np.isfinite(v)] = np.nan
    return out


def classics_direct(c, t, signs):
    """Те же меры одного окна, кончающегося на баре t, — прямым счётом (стенд)."""
    L = EFF['L']
    s1, sb = signs
    r = np.diff(np.log(np.asarray(c[t - L:t + 1], dtype=np.float64)))     # L ходов: бары t − L + 1 … t
    out = {}
    for tag, rr in (('r', r), ('t', r * s1[t - L + 1:t + 1])):
        ac = float(np.sum(rr[1:] * rr[:-1]) / np.sum(rr * rr))
        R16 = np.array([rr[j - 15:j + 1].sum() for j in range(15, L)])
        out['ac_' + tag] = ac
        out['lv16_' + tag] = math.log(np.mean(R16 ** 2) / (16.0 * np.mean(rr * rr)))
    blocks = np.array([r[L - 4 * k - 4:L - 4 * k].sum() for k in range(L // 4)])         # k = 0 — четвёрка у конца окна
    for tag, sg in (('r', np.ones(L // 4)), ('t', np.array([sb[t - 4 * k] for k in range(L // 4)]))):
        b = blocks * sg
        R16 = np.array([b[m:m + 4].sum() for m in range(L // 4 - 3)])
        out['lv44_' + tag] = math.log(np.mean(R16 ** 2) / (4.0 * np.mean(blocks ** 2)))
    return out


# =====================================================================
# СИТУАЦИИ: час недели и профиль объёма, дни стресса, окна релизов
# =====================================================================
def hour_of_week(T):
    """Час недели UTC: понедельник 00:00 — 0, воскресенье 23:00 — 167."""
    T = np.asarray(T, dtype=np.int64)
    return (((T // DAY_NS) + 3) % 7) * 24 + (T // HOUR_NS) % 24


def hour_profile(t_ms, qv):
    """Профиль часа недели по объёму фьючерса (бары 5M: время открытия, мс; объём в USDT): объём полных часов UTC к медиане
       часа своей недели (неделя — с понедельника), медиана по годным неделям до profile_before. Тонкие — hours_n часов
       с наименьшим профилем, ликвидные — с наибольшим."""
    t = np.asarray(t_ms, dtype=np.int64) * 10 ** 6
    keep = t < pd.Timestamp(EFF['profile_before']).value
    t, qv = t[keep], np.asarray(qv, dtype=np.float64)[keep]
    hour = t // HOUR_NS
    hu, inv, cnt = np.unique(hour, return_inverse=True, return_counts=True)
    hv = np.bincount(inv, weights=qv)
    full = cnt == HOUR_NS // BAR_NS['5M']
    hu, hv = hu[full], hv[full]
    how = hour_of_week(hu * HOUR_NS)
    week = (hu * HOUR_NS // DAY_NS + 3) // 7
    rows = []
    for w in np.unique(week):
        m = week == w
        if m.sum() >= EFF['profile_min_hours'] and np.median(hv[m]) > 0:
            row = np.full(168, np.nan)
            row[how[m]] = hv[m] / np.median(hv[m])
            rows.append(row)
    n_w = len(rows)
    if n_w < EFF['profile_min_weeks']:
        return {'ok': False, 'weeks': n_w}
    prof = np.nanmedian(np.array(rows), axis=0)
    order = np.argsort(prof, kind='stable')
    k = EFF['hours_n']
    return {'ok': True, 'weeks': n_w, 'profile': prof, 'thin': np.sort(order[:k]), 'liquid': np.sort(order[-k:])}


def stress_days(segs, bar_ns, bounds):
    """Дни стресса по ряду BTC ТФ. segs — части (без pre-баров), склеенные по времени; RV дня — сумма квадратов log-ходов
       баров дня; отношение — к медиане RV годных дней из stress_lookback прошлых календарных (их >= stress_min_days).
       bounds — {период: (начало, конец) в нс}; стресс — отношение >= квантиля stress_q годных дней периода.
       Возвращает {период: {'stress': дни, 'valid': дни, 'thr': порог}}."""
    T = np.concatenate([B['T'][B['n_pre']:] for B in segs]).astype(np.int64)
    c = np.concatenate([B['c'][B['n_pre']:] for B in segs]).astype(np.float64)
    o = np.argsort(T, kind='stable')
    T, c = T[o], c[o]
    r2 = np.r_[np.nan, np.diff(np.log(c))] ** 2
    day = T // DAY_NS
    du, inv = np.unique(day, return_inverse=True)
    ok = np.isfinite(r2)
    rv = np.bincount(inv[ok], weights=r2[ok], minlength=len(du))
    nb = np.bincount(inv, minlength=len(du))
    valid = nb >= EFF['day_min_share'] * (DAY_NS // bar_ns)
    pos = {d: i for i, d in enumerate(du)}
    ratio = np.full(len(du), np.nan)
    for i, d in enumerate(du):
        if not valid[i]:
            continue
        prev = [pos[x] for x in range(d - EFF['stress_lookback'], d) if x in pos and valid[pos[x]]]
        if len(prev) >= EFF['stress_min_days']:
            med = np.median(rv[prev])
            if med > 0:
                ratio[i] = rv[i] / med
    out = {}
    for name, (a, b) in bounds.items():
        m = np.isfinite(ratio) & (du * DAY_NS >= a) & (du * DAY_NS < b)
        if m.sum() < 10:
            out[name] = {'stress': np.zeros(0, np.int64), 'valid': du[m], 'thr': math.nan}
            continue
        thr = float(np.quantile(ratio[m], EFF['stress_q']))
        out[name] = {'stress': du[m & (ratio >= thr)], 'valid': du[m], 'thr': thr}
    return out


def et_of(t_ns):
    """UTC (нс) -> ET: минуты суток, день недели (пн — 0), номер даты ET."""
    x = pd.to_datetime(np.asarray(t_ns, dtype=np.int64), utc=True).tz_convert(ET)
    mins = np.asarray(x.hour, dtype=np.int64) * 60 + np.asarray(x.minute, dtype=np.int64)
    wd = np.asarray(x.dayofweek, dtype=np.int64)
    day = x.tz_localize(None).to_numpy().astype('datetime64[D]').astype(np.int64)
    return mins, wd, day


def event_utc(date, hm):
    """Время релиза ET -> UTC, нс."""
    return pd.Timestamp(f"{date} {hm}", tz=ET).tz_convert('UTC').value


def news_masks(T, bar_ns, events=None, pre=False):
    """Окна релизов и их пары по концу бара в ET. Слот s = (начало; конец] минут суток: бар релиза — в слоте s дня с релизом
       этого слота; пара — в слоте s будня без единого релиза. pre — те же окна, сдвинутые на news_h раньше: news_h часов до
       релиза, последний бар кончается не позже релиза (справочно). Возвращает {слот: (релиз, пара)}."""
    events = EVENTS if events is None else events
    mins, wd, day = et_of(np.asarray(T, dtype=np.int64) + bar_ns)
    by_slot = {s: [] for s in EFF['slots']}
    for d, hm, _ in events:
        by_slot[hm].append(np.datetime64(d, 'D').astype(np.int64))
    any_ev = np.isin(day, np.array([x for v in by_slot.values() for x in v], dtype=np.int64))
    sh = EFF['news_h'] * 60 if pre else 0
    out = {}
    for s, (a, b) in EFF['slots'].items():
        in_s = (mins > a - sh) & (mins <= b - sh)
        out[s] = (in_s & np.isin(day, np.array(by_slot[s], dtype=np.int64)), in_s & (wd < 5) & ~any_ev)
    return out


# =====================================================================
# ЦВЕТ И БАРЫ ЧАСТИ -> e мер на барах периода
# =====================================================================
def eff_colors(C, src, bars, tf):
    """На барах части у каждого ассета — line_colors по окнам кэша Flow: цвет реала c и цвет близнеца sflip1 на его линии ct
       (оба — как source_colors Legs, стенд сверяет бит в бит), цвета близнецов на линии реала cx4 / cx1, z и база m.
       Линейка — суррогаты sflip4 источника, как у продукта."""
    lgt, tags = FL.src_logits(C, src)
    if not lgt:
        return {}, []
    part = EFF['periods'][src]
    code = 0 if part == 'tune' else 1
    q, zk = FL.ruler(np.concatenate([x['s4'] for x in lgt.values()]))
    out = {}
    for a, x in lgt.items():
        if a not in bars:
            continue
        B = bars[a][code]
        idx = LG.windows_to_bars(B['T'], B['n_pre'], C['res'][a]['t'].astype(np.int64), a, f"{tf} {part}")
        vals = line_colors(*(FL.to_z(x[k], q, zk) for k in ('sp', 's4', 's1')), EFF['ceiling'][tf])
        cols = {}
        for k, v in vals.items():
            col = np.full(len(B['T']), np.nan, dtype=np.float32)
            col[idx] = v
            cols[k] = col
        out[a] = cols
    return out, tags


def period_mask(B, code):
    """Бары периода: TUNE — бары TRAINING с tune_start, VAL — бары VALIDATION; pre-бары — никогда."""
    m = np.arange(len(B['T'])) >= B['n_pre']
    if code == 0:
        m &= B['T'] >= pd.Timestamp(EFF['tune_start']).value
    return m


def frame_part(T, c, cols, in_per, rng, parts=('mri', 'classic')):
    """e мер на барах периода: MRI — где есть цвета реала и близнецов (MRI_COLS), классика — где годны все её меры (они
       считаются с L + 1 баров до первого бара с цветом). У всех мер кадра — одни и те же бары; плюс сырьё для справок.
       parts — что считать ('mri', 'classic'); у стенда бывает одно из двух."""
    ok = np.array(in_per, dtype=bool)
    if 'mri' in parts:
        for k in MRI_COLS:
            ok &= np.isfinite(cols[k])
    if not ok.any():
        return None
    F = {}
    if 'classic' in parts:
        lo = max(int(np.flatnonzero(ok)[0]) - EFF['L'] - 1, 0)
        K = classics(c[lo:], twin_signs(len(c) - lo, rng))
        K = {k: np.r_[np.full(lo, np.nan), v] for k, v in K.items()}
        for k in K:
            ok &= np.isfinite(K[k])
        cap = EFF['vr_cap']
        F.update({'AC1': np.abs(K['ac_r'][ok]) - np.abs(K['ac_t'][ok]),
                  'VR16': np.minimum(np.abs(K['lv16_r'][ok]), cap) - np.minimum(np.abs(K['lv16_t'][ok]), cap),
                  'VR4x4': np.minimum(np.abs(K['lv44_r'][ok]), cap) - np.minimum(np.abs(K['lv44_t'][ok]), cap),
                  'ac_r': K['ac_r'][ok], 'ac_t': K['ac_t'][ok], 'lv16_r': K['lv16_r'][ok], 'lv16_t': K['lv16_t'][ok]})
    if 'mri' in parts:
        g = {k: np.asarray(cols[k])[ok].astype(np.float64) for k in MRI_COLS + ('z', 'z4', 'm')}
        F.update({'MRI': g['c'] - g['cx4'],                 # главная: вклад порядка ходов в цвет
                  'MRIz': g['z'] - g['z4'],                 # без линии: оценка модели реала − того же окна sflip4
                  'MRI1': g['c'] - g['cx1'],                # близнец sflip1 на линии реала
                  'MRIv1': np.abs(g['c']) - np.abs(g['ct']),   # мера v1
                  'c_r': g['c'], 'c_x': g['cx4'], 'z_r': g['z'], 'z_4': g['z4'], 'm_r': g['m']})
    F['T'] = np.asarray(T, dtype=np.int64)[ok]
    return {k: (v if k == 'T' else v.astype(np.float32)) for k, v in F.items()}


def stack_frames(parts):
    """Части (ассет, кадр) -> один кадр периода с номером ассета."""
    parts = [(ia, F) for ia, F in parts if F is not None and len(F['T'])]
    if not parts:
        return None
    out = {k: np.concatenate([F[k] for _, F in parts]) for k in parts[0][1]}
    out['asset'] = np.concatenate([np.full(len(F['T']), ia, dtype=np.int16) for ia, F in parts])
    return out


def hyp_strata(F, bar_ns, prof, stress, events=None):
    """Пары (A, B) по слоям для Н1–Н3 на барах кадра: Н1 — тонкие / ликвидные часы недели; Н2 — дни стресса / прочие годные
       дни периода; Н3 — слоты 08:30 и 14:00: релиз / та же пора будня без релизов; PRE — то же за news_h часов до релиза."""
    T = F['T']
    out = {}
    if prof is not None and prof.get('ok'):
        how = hour_of_week(T)
        out['H1'] = [(np.isin(how, prof['thin']), np.isin(how, prof['liquid']))]
    if stress is not None and len(stress['stress']):
        day = T // DAY_NS
        st = np.isin(day, stress['stress'])
        out['H2'] = [(st, np.isin(day, stress['valid']) & ~st)]
    nm = news_masks(T, bar_ns, events)
    out['H3'] = [nm[s] for s in EFF['slots']]
    npre = news_masks(T, bar_ns, events, pre=True)
    out[PRE] = [npre[s] for s in EFF['slots']]
    return out


# =====================================================================
# СТАТИСТИКА: Δ по слоям, бутстреп по неделям
# =====================================================================
def boot_weeks(T, rng):
    """Номер недели бара и матрица бутстрепа: сколько раз каждая неделя попала в каждый повтор."""
    wk = np.asarray(T, dtype=np.int64) // WEEK_NS
    wu, wi = np.unique(wk, return_inverse=True)
    nb, nw = EFF['n_boot'], len(wu)
    M = np.zeros((nb, nw))
    np.add.at(M, (np.repeat(np.arange(nb), nw), rng.integers(0, nw, size=nb * nw)), 1.0)
    return wi, M


def hyp_stat(e, wi, M, strata):
    """Δ = Σ_s w_s (ē_A,s − ē_B,s), w_s ∝ n_A,s (бары с годным e); интервал и ошибка — по повторам M."""
    nw = M.shape[1]
    e = np.asarray(e, dtype=np.float64)
    fin = np.isfinite(e)
    pts, boots, nA, nB = [], [], 0, 0
    for mA, mB in strata:
        sums = []
        for m in (mA, mB):
            mm = m & fin
            s = np.bincount(wi[mm], weights=e[mm], minlength=nw)
            n = np.bincount(wi[mm], minlength=nw).astype(np.float64)
            sums.append((s, n))
        (sA, cA), (sB, cB) = sums
        if cA.sum() == 0 or cB.sum() == 0:
            continue
        nA += int(cA.sum())
        nB += int(cB.sum())
        pts.append((sA.sum(), cA.sum(), sB.sum(), cB.sum()))
        boots.append((M @ sA, M @ cA, M @ sB, M @ cB))
    if not pts:
        return {'d': math.nan, 'ci': (math.nan, math.nan), 'se': math.nan, 't': math.nan, 'nA': nA, 'nB': nB}

    def delta(rows):
        num = 0.0
        den = 0.0
        for SA, NA, SB, NB in rows:
            with np.errstate(divide='ignore', invalid='ignore'):
                num = num + (SA - NA * SB / NB)
            den = den + NA
        with np.errstate(divide='ignore', invalid='ignore'):
            return num / den
    d = float(delta(pts))
    db = np.asarray(delta(boots), dtype=np.float64)
    db = db[np.isfinite(db)]
    se = float(db.std(ddof=1)) if len(db) > 1 else math.nan
    ci = (float(np.quantile(db, 0.025)), float(np.quantile(db, 0.975))) if len(db) > 1 else (math.nan, math.nan)
    return {'d': d, 'ci': ci, 'se': se, 't': d / se if se and se > 0 else math.nan, 'nA': nA, 'nB': nB}


def sees(st):
    """Мера видит Нk: Δ > 0 и нижний край интервала > 0 на обоих периодах."""
    return all(st.get(src) is not None and np.isfinite(st[src]['d']) and st[src]['d'] > 0 and st[src]['ci'][0] > 0
               for src in EFF['periods'])


def inverse(st):
    """Обратный знак: Δ < 0 и верхний край интервала < 0 на обоих периодах."""
    return all(st.get(src) is not None and np.isfinite(st[src]['d']) and st[src]['d'] < 0 and st[src]['ci'][1] < 0
               for src in EFF['periods'])


def verdict(stats):
    """Вердикт Нk на ТФ по правилам: видит ли каждая мера (или обратный знак); резче ли MRI классики (или наоборот);
       ярлык Нk; если MRI видит — текущим окном (MRI без линии: Δ > 0 на обоих периодах) или через базу линии."""
    seen = {m: sees(stats[m]) for m in MEASURES}
    inv = {m: inverse(stats[m]) for m in MEASURES}

    def t_of(m, src):
        x = stats[m].get(src)
        return x['t'] if x is not None and np.isfinite(x['t']) else -math.inf
    sharper = seen['MRI'] and all(t_of('MRI', s) > t_of(m, s) for s in EFF['periods'] for m in CLASSIC)
    beat = [m for m in CLASSIC if seen[m] and all(t_of(m, s) > t_of('MRI', s) for s in EFF['periods'])]
    cmp_ = 'MRI резче классики' if sharper else (f"классика резче ({', '.join(beat)})" if beat else 'наравне')
    cls = [m for m in CLASSIC if seen[m]]
    label = ('MRI видит, классика — нет' if seen['MRI'] and not cls else 'видят MRI и классика' if seen['MRI'] else
             'видит только классика' if cls else 'MRI — обратный знак' if inv['MRI'] else 'не видит никто')
    zst = stats.get('MRIz', {})
    zpos = all(zst.get(s) is not None and np.isfinite(zst[s]['d']) and zst[s]['d'] > 0 for s in EFF['periods'])
    how = None if not seen['MRI'] else ('текущим окном' if zpos else 'через базу линии')
    return {'seen': seen, 'inverse': inv, 'compare': cmp_, 'mri_sharper': sharper, 'classic_sharper': beat,
            'classic_seen': cls, 'label': label, 'how': how}


def _means(x, mA, mB):
    return (float(np.mean(x[mA])) if mA.any() else math.nan, float(np.mean(x[mB])) if mB.any() else math.nan)


def period_stats(Fs, bar_ns, prof, stress, events=None, seed=None, measures=None):
    """Статистика Н1–Н3 и PRE по кадрам периодов: Δ мер (по умолчанию — MEASURES и справочные MRI_REF; с MRI — ещё MRI
       только BTC), справки по A / B (цвет реала и близнеца на линии реала, доли оранжевых, z реала и sflip4, AC1, Хёрст).
       Одни повторы бутстрепа у всех мер периода."""
    seed = EFF['seed'] if seed is None else seed
    ms = MEASURES + MRI_REF if measures is None else tuple(measures)
    hs = tuple(HYPS) + (PRE,)
    res = {h: {'stats': {m: {} for m in ms + (('MRI_btc',) if 'MRI' in ms else ())}, 'ref': {}} for h in hs}
    for code, (src, F) in enumerate(Fs.items()):
        if F is None:
            continue
        wi, M = boot_weeks(F['T'], np.random.default_rng([seed, 7, code]))
        strata = hyp_strata(F, bar_ns, prof, stress.get(src) if stress else None, events)
        btc = F['asset'] == 0
        for h in hs:
            if h not in strata:
                continue
            S = strata[h]
            for m in ms:
                res[h]['stats'][m][src] = hyp_stat(F[m], wi, M, S)
            if 'MRI' in ms:
                res[h]['stats']['MRI_btc'][src] = hyp_stat(F['MRI'], wi, M, [(a & btc, b & btc) for a, b in S])
            mA = np.zeros(len(F['T']), bool)
            mB = np.zeros(len(F['T']), bool)
            for a, b in S:
                mA |= a
                mB |= b
            ref = {k: _means(F[k], mA, mB) for k in ('c_r', 'c_x', 'z_r', 'z_4', 'ac_r', 'ac_t', 'lv16_r', 'lv16_t')
                   if k in F}
            if 'c_r' in F:
                ref['orange_r'] = _means(F['c_r'] > 0, mA, mB)
                ref['orange_x'] = _means(F['c_x'] > 0, mA, mB)
            if 'lv16_r' in F:
                ref['hurst_r'] = tuple(0.5 + x / (2 * math.log(16)) for x in ref['lv16_r'])
            res[h]['ref'][src] = ref
    for h in HYPS:
        res[h]['verdict'] = verdict(res[h]['stats']) if all(res[h]['stats'].get(m) for m in MEASURES) else None
    return res


# =====================================================================
# ПЕЧАТЬ И РИСУНОК
# =====================================================================
def _st_txt(x):
    if not x or not np.isfinite(x.get('d', math.nan)):
        return f"{'—':^34}"
    return f"{x['d']:+.4f} [{x['ci'][0]:+.4f}; {x['ci'][1]:+.4f}] ({_f(x['t'], '+.1f')})"


def _pair(x, fmt, pct=False):
    if pct:
        return f"{_f(100 * x[0], '.0f')}% / {_f(100 * x[1], '.0f')}%"
    return f"{_f(x[0], fmt)} / {_f(x[1], fmt)}"


def _ref_txt(src, ref):
    return (f"справочно {src}: A / B — цвет реала {_pair(ref['c_r'], '+.3f')} (оранжевых {_pair(ref['orange_r'], '', True)}), "
            f"тех же окон со случайными направлениями четвёрок {_pair(ref['c_x'], '+.3f')} (оранжевых "
            f"{_pair(ref['orange_x'], '', True)}); z реала {_pair(ref['z_r'], '+.2f')}, z sflip4 {_pair(ref['z_4'], '+.2f')}; "
            f"AC1 реала {_pair(ref['ac_r'], '+.4f')}, близнеца {_pair(ref['ac_t'], '+.4f')}; Хёрст (VR16) реала "
            f"{_pair(ref['hurst_r'], '.3f')}")


def show_hyps(tf, res):
    for h, (title, expect) in HYPS.items():
        r = res[h]
        print(f"\n   ▶ ({h.replace('H', 'Н')}) {title} — ожидание: неслучайность {expect}")
        if r['verdict'] is None:
            print("     нет данных для этой гипотезы на этом ТФ")
            continue
        v = r['verdict']
        print(f"     {'мера':<7} {'поиск: Δ [95%] (t)':<38} | {'подтверждение: Δ [95%] (t)':<38} | вердикт")
        for m in MEASURES:
            st = r['stats'][m]
            res_m = 'видит ✅' if v['seen'][m] else ('обратный знак' if v['inverse'][m] else 'не видит')
            print(f"     {m:<7} {_st_txt(st.get('search')):<38} | {_st_txt(st.get('confirm')):<38} | {res_m}")
        for m in MRI_REF + ('MRI_btc',):
            st = r['stats'][m]
            print(f"     справочно {LABEL[m]}: поиск {_st_txt(st.get('search')).strip()} | подтверждение "
                  f"{_st_txt(st.get('confirm')).strip()}")
        for src in EFF['periods']:
            if r['ref'].get(src):
                print('     ' + _ref_txt(src, r['ref'][src]))
        print(f"     -> {v['label']}" + (f" ({v['how']})" if v['how'] else '') + f"; {v['compare']}")
    r = res[PRE]
    if r['stats']['MRI']:
        print(f"\n   ▶ справочно: {EFF['news_h']} ч до релиза против тех же часов без релиза (началось ли до самого релиза?)")
        for m in ('MRI', 'MRIz') + CLASSIC:
            st = r['stats'][m]
            print(f"     {LABEL[m]:<14} поиск {_st_txt(st.get('search')).strip()} | подтверждение "
                  f"{_st_txt(st.get('confirm')).strip()}")


PLOT_M = ('MRI', 'MRIz', 'AC1', 'VR16', 'VR4x4')       # меры на рисунке (подписи строк — у каждой точки)


def plot(path, res):
    """Сверху — резкость мер (t = Δ / ошибка) по гипотезам, поиск и подтверждение; снизу — неслучайность MRI по часам недели."""
    tfs = [tf for tf in EFF['tfs'] if tf in res]
    if not tfs:
        return False
    fig = plt.figure(figsize=(7.4 * len(tfs), 12.5))
    fig.patch.set_facecolor(C_BG)
    gs = fig.add_gridspec(2, len(tfs), height_ratios=[2.0, 1.0])
    rows = [(h, m) for h in HYPS for m in PLOT_M]
    names = {'MRI': 'MRI', 'MRIz': 'MRI без линии', 'AC1': 'AC1', 'VR16': 'VR16', 'VR4x4': 'VR4x4'}
    for j, tf in enumerate(tfs):
        ax = fig.add_subplot(gs[0, j])
        ax.set_facecolor(C_BG)
        y = np.arange(len(rows))[::-1].astype(float)
        for (h, m), yy in zip(rows, y):
            st = res[tf]['hyps'][h]['stats'][m]
            for src, mk, off in (('search', 'o', 0.15), ('confirm', 's', -0.15)):
                x = st.get(src)
                if x is not None and np.isfinite(x.get('t', math.nan)):
                    ax.plot(x['t'], yy + off, mk, color=C_MEAS[m], ms=6,
                            mfc=C_MEAS[m] if src == 'search' else 'none', mew=1.4)
        for xv, ls in ((0.0, '-'), (1.96, ':'), (-1.96, ':')):
            ax.axvline(xv, color=C_INK2, lw=0.8, ls=ls)
        for k in range(1, len(HYPS)):
            ax.axhline(len(PLOT_M) * k - 0.5, color=C_GRID, lw=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{h.replace('H', 'Н')} {names[m]}" for h, m in rows], fontsize=8, color=C_INK)
        ax.set_xlabel('t = Δ / ошибка бутстрепа (● поиск, □ подтверждение)', color=C_INK2)
        ax.set_title(f"{tf}: насколько резко мера видит ситуацию", fontsize=10, color=C_INK)
        ax.grid(True, axis='x', color=C_GRID, lw=0.6)
        ax2 = fig.add_subplot(gs[1, j])
        mp = np.asarray(res[tf]['map'], dtype=float).reshape(7, 24)
        im = ax2.imshow(mp, aspect='auto', cmap='Oranges')
        ax2.set_yticks(range(7))
        ax2.set_yticklabels(['пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'вс'], fontsize=8)
        ax2.set_xticks(range(0, 24, 3))
        ax2.set_xlabel('час UTC', color=C_INK2)
        ax2.set_title(f"{tf}: неслучайность MRI по часам недели, оба периода\n(цвет реала − цвет тех же окон со случайными "
                      f"направлениями четвёрок)", fontsize=9, color=C_INK)
        fig.colorbar(im, ax=ax2, fraction=0.025)
    fig.suptitle(f"SuMo_Class_Eff v{EFF_VER}: видит ли MRI неслучайный рынок — ситуации, где теория ждёт меньше эффективности",
                 color=C_INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=C_BG)
    plt.close(fig)
    return True


# =====================================================================
# 'run'
# =====================================================================
def fut_daily_volume(assets):
    """Суточный объём фьючерса в USDT по CSV (Н4, справочно): {ассет: (дни, объём)}; нет файла — ассета нет."""
    out = {}
    for a in assets:
        B5, info = FL.read_fut_csv(FUT_DIR + FL.FLOW['fut_csv'].format(a=a))
        if B5 is None or not info['ok']:
            continue
        day = B5['t'].astype(np.int64) * 10 ** 6 // DAY_NS
        du, inv = np.unique(day, return_inverse=True)
        out[a] = (du, np.bincount(inv, weights=B5['v'] * B5['c']))
    return out


H4_COLS = ('MRI', 'MRIz', 'MRIv1', 'orange', 'base_m') + CLASSIC   # столбцы таблицы монет (Н4)
H4_LAB = {'MRI': 'MRI', 'MRIz': 'MRI без линии', 'MRIv1': 'мера v1', 'orange': 'доля оранжевых', 'base_m': 'база m',
          'AC1': 'AC1', 'VR16': 'VR16', 'VR4x4': 'VR4x4'}


def h4_table(Fs, assets, fvol):
    """Н4 (справочно): у каждой монеты — среднее e мер, доля оранжевых баров, средняя база m линии (оба периода) и log
       медианы суточного объёма фьючерса за дни её баров; Spearman по монетам."""
    from scipy.stats import spearmanr
    src_of = {'MRI': 'MRI', 'MRIz': 'MRIz', 'MRIv1': 'MRIv1', 'orange': 'c_r', 'base_m': 'm_r',
              **{m: m for m in CLASSIC}}
    rows = []
    for ia, a in enumerate(assets):
        if a not in fvol:
            continue
        es = {k: [] for k in H4_COLS}
        days = []
        for F in Fs.values():
            if F is None:
                continue
            m_ = F['asset'] == ia
            if not m_.any():
                continue
            for k in H4_COLS:
                x = F[src_of[k]][m_]
                es[k].append((x > 0).astype(np.float64) if k == 'orange' else x)
            days.append(np.unique(F['T'][m_] // DAY_NS))
        if not days:
            continue
        du, v = fvol[a]
        sel = np.isin(du, np.concatenate(days)) & (v > 0)
        if sel.sum() < 30:
            continue
        rows.append({'asset': a, 'log_vol': float(np.log(np.median(v[sel]))),
                     **{k: float(np.mean(np.concatenate(es[k]))) for k in H4_COLS}})
    out = {'n': len(rows), 'rows': rows, 'rho': {}}
    if len(rows) >= 5:
        for k in H4_COLS:
            rho, p = spearmanr([r_['log_vol'] for r_ in rows], [r_[k] for r_ in rows])
            out['rho'][k] = {'rho': float(rho), 'p': float(p)}
    return out


def summary(res):
    """Итог по правилам: ярлык каждой Нk на ТФ; мишени (Н2, Н3) и вывод для трейдеров; справка к описанию цветов."""
    print(f"\n{'=' * 110}\n🏁 ИТОГ (по правилам, записанным до прогона)\n{'=' * 110}")
    out = {'tfs': {}}
    for tf, r in res.items():
        parts = []
        for h in HYPS:
            v = r['hyps'][h]['verdict']
            parts.append(f"{h.replace('H', 'Н')} " + ('—' if v is None else
                                                    v['label'] + (f" ({v['how']})" if v['how'] else '') + f", {v['compare']}"))
        print(f"   {tf}: " + ' | '.join(parts))
    print("\n   Мишени — крупные рыночные события (Н2 — дни стресса, Н3 — макрорелизы США):")
    nm = lambda hs: ', '.join(h.replace('H', 'Н') for h in hs)
    for tf, r in res.items():
        vs = {h: r['hyps'][h]['verdict'] for h in TARGETS}
        seen = [h for h in TARGETS if vs[h] is not None and vs[h]['seen']['MRI']]
        hit = [h for h in seen if vs[h]['how'] == 'текущим окном']
        base = [h for h in seen if h not in hit]
        blind = [h for h in hit if not vs[h]['classic_seen']]
        out['tfs'][tf] = {'targets_seen': hit, 'targets_seen_base': base, 'classic_blind': blind}
        print(f"   {tf}: MRI видит текущим окном — {nm(hit) or 'ни одной'}" +
              (f"; из них классика не видит — {nm(blind) or 'ни одной'}" if hit else '') +
              (f"; через базу линии — {nm(base)}" if base else ''))
    ok_tfs = [tf for tf, x in out['tfs'].items() if x['targets_seen']]
    blind_tfs = [tf for tf in ok_tfs if out['tfs'][tf]['classic_blind'] == out['tfs'][tf]['targets_seen']]
    out['claim_tfs'], out['claim_classic_blind_tfs'] = ok_tfs, blind_tfs
    tail = f"; и классика этого не видит — на {', '.join(blind_tfs)}" if blind_tfs else ''
    print("   Вывод для трейдеров («MRI говорит, когда рынок неслучаен»): " + (
        f"подтверждён на {', '.join(ok_tfs)}{tail}" if ok_tfs else 'не подтверждён'))
    bl = [(tf, src, x) for tf, r in res.items() for src, x in r.get('blue', {}).items()]
    blue_ok = bool(bl) and all(x['twin'] > 0.5 and x['twin'] > x['real'] for _, _, x in bl)
    out['blue_reads_random'] = blue_ok
    print("   Справочно, к описанию цветов: синий на линии реальной монеты — «похоже на случайный»: " +
          ('да' if blue_ok else 'нет') + ' (доля синих: реал / те же окна со случайными направлениями — ' +
          ', '.join(f"{tf} {src} {100 * x['real']:.0f}% / {100 * x['twin']:.0f}%" for tf, src, x in bl) + ')')
    return out


def mode_run(tfs=None):
    t0 = time.time()
    load_mods()
    tfs = tuple(tfs or EFF['tfs'])
    print(f"\n{'#' * 110}\n# SuMo_Class_Eff v{EFF_VER} | 'run': видит ли MRI неслучайный рынок — ситуации, где теория ждёт "
          f"меньше эффективности ({', '.join(tfs)})\n{'#' * 110}")
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    jp, pp = OUT_DIR + f"eff_{stamp}.json", OUT_DIR + f"eff_{stamp}.png"
    out = {'ver': EFF_VER, 'mode': 'run', 'design': {'EFF': EFF, 'events': EVENTS}, 'tfs': {}}
    # (Н1) профиль часа недели по объёму BTC-фьючерса
    pa = EFF['profile_asset']
    B5, info = FL.read_fut_csv(FUT_DIR + FL.FLOW['fut_csv'].format(a=pa))
    prof = None
    if B5 is None or not info['ok']:
        print(f"🍆 (Н1) нет годного CSV {pa} ({info.get('why', 'сверка не прошла')}) — Н1 не считается")
        print(FL.fut_dir_diag([pa]))
    else:
        prof = hour_profile(B5['t'], B5['v'] * B5['c'])
        if not prof['ok']:
            print(f"🍆 (Н1) годных недель профиля {prof['weeks']} < {EFF['profile_min_weeks']} — Н1 не считается")
        else:
            order = np.argsort(prof['profile'], kind='stable')
            cells = lambda s: ', '.join(f"{['пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'вс'][x // 24]} {x % 24:02d}" for x in s)
            print(f"🕐 (Н1) профиль часа недели по объёму {pa}-фьючерса: недель {prof['weeks']} (до "
                  f"{EFF['profile_before'][:10]}); самые тонкие — {cells(order[:6])} …, самые ликвидные — "
                  f"{cells(order[::-1][:6])} … (UTC)")
    out['profile'] = None if prof is None else {k: v for k, v in prof.items()}
    fvol = None
    res = {}
    for tf in tfs:
        print(f"\n{'=' * 110}\n🧾 {tf}\n{'=' * 110}")
        caches = {src: FL.load_cache(tf, part) for src, part in EFF['periods'].items()}
        miss = [src for src, C in caches.items() if C is None]
        if miss:
            print(f"🍆 {tf}: нет кэша Flow для {', '.join(miss)} ({FL.OUT_DIR}cache/) — ТФ пропущен")
            out['tfs'][tf] = {'stop': f"нет кэша Flow: {', '.join(miss)}"}
            continue
        assets = sorted(set.intersection(*[set(C['res']) for C in caches.values()]), key=lambda a: (a != 'BTCUSDT', a))
        if not assets or assets[0] != 'BTCUSDT':
            print(f"🍆 {tf}: в кэшах нет BTCUSDT — ТФ пропущен (Н2 и «только BTC» считаются по нему)")
            out['tfs'][tf] = {'stop': 'нет BTCUSDT в кэшах'}
            continue
        t1 = time.time()
        bars = {a: LG.load_bars(a, tf) for a in assets}
        print(f"📦 бары {len(bars)} ассетов: {sum(len(B['c']) for P_ in bars.values() for B in P_.values()) / 1e6:.1f} млн "
              f"| {time.time() - t1:.0f} c")
        bounds = {'search': (pd.Timestamp(EFF['tune_start']).value, int(bars['BTCUSDT'][1]['T'][bars['BTCUSDT'][1]['n_pre']])),
                  'confirm': (int(bars['BTCUSDT'][1]['T'][bars['BTCUSDT'][1]['n_pre']]), 2 ** 62)}
        stress = stress_days([bars['BTCUSDT'][0], bars['BTCUSDT'][1]], BAR_NS[tf], bounds)
        print(f"🌪️ (Н2) дни стресса по RV {EFF['stress_asset']} {tf}: " + ' | '.join(
            f"{src} — {len(x['stress'])} из {len(x['valid'])} годных (порог отношения {_f(x['thr'], '.2f')})"
            for src, x in stress.items()))
        Fs, tags = {}, {}
        for code, (src, C) in enumerate(caches.items()):
            t1 = time.time()
            colors, tags[src] = eff_colors(C, src, bars, tf)
            parts = []
            for ia, a in enumerate(assets):
                if a not in colors:
                    continue
                B = bars[a][code]
                parts.append((ia, frame_part(B['T'], B['c'], colors[a], period_mask(B, code),
                                             np.random.default_rng([EFF['seed'], ia, code]))))
            Fs[src] = stack_frames(parts)
            n_ = 0 if Fs[src] is None else len(Fs[src]['T'])
            print(f"🎨 {tf} {src}: модели {', '.join(tags[src])} | ассетов с цветом {len(colors)} | баров в мерах "
                  f"{n_ / 1e6:.2f} млн | {time.time() - t1:.0f} c", flush=True)
            del colors
        del caches
        ev_n = {}
        for src, F in Fs.items():
            if F is None:
                continue
            nm = news_masks(F['T'][F['asset'] == 0], BAR_NS[tf])
            days = {s: set((F['T'][F['asset'] == 0][nm[s][0]] // DAY_NS).tolist()) for s in nm}
            ev_n[src] = {s: len(v) for s, v in days.items()}
        print("📰 (Н3) релизов с барами в периоде (по BTC): " + ' | '.join(
            f"{src} — 08:30 {x['08:30']}, 14:00 {x['14:00']}" for src, x in ev_n.items()))
        blue = {src: {'real': float(np.mean(F['c_r'] < 0)), 'twin': float(np.mean(F['c_x'] < 0)),
                      'c_real': float(np.mean(F['c_r'])), 'c_twin': float(np.mean(F['c_x']))}
                for src, F in Fs.items() if F is not None}
        print("🔵 доля синих баров: " + ' | '.join(
            f"{src} — реал {100 * x['real']:.0f}%, те же окна со случайными направлениями четвёрок на линии реала "
            f"{100 * x['twin']:.0f}% (средний цвет {x['c_real']:+.3f} / {x['c_twin']:+.3f})" for src, x in blue.items()))
        t1 = time.time()
        hy = period_stats(Fs, BAR_NS[tf], prof, stress)
        show_hyps(tf, hy)
        # карта по часам недели (оба периода) и Н4
        allF = [F for F in Fs.values() if F is not None]
        how = np.concatenate([hour_of_week(F['T']) for F in allF])
        em = np.concatenate([F['MRI'] for F in allF]).astype(np.float64)
        cnt = np.bincount(how, minlength=168)
        mp = np.where(cnt > 0, np.bincount(how, weights=em, minlength=168) / np.maximum(cnt, 1), np.nan)
        if fvol is None:
            fvol = fut_daily_volume(assets)
        h4 = h4_table(Fs, assets, fvol)
        print(f"\n   ▶ (Н4) справочно — монеты с CSV фьючерса: {h4['n']}" + (
            '; Spearman с log суточного объёма монеты: ' + ', '.join(
                f"{H4_LAB[k]} {x['rho']:+.2f} (p {x['p']:.2f})" for k, x in h4['rho'].items()) if h4['rho'] else
            ' — мало монет'))
        print(f"   | {time.time() - t1:.0f} c", flush=True)
        res[tf] = {'hyps': hy, 'map': mp, 'blue': blue}
        out['tfs'][tf] = {'assets': assets, 'tags': tags,
                          'n_bars': {s: (0 if F is None else len(F['T'])) for s, F in Fs.items()},
                          'stress': {s: {'days': [str(np.datetime64(int(d), 'D')) for d in x['stress']],
                                         'valid': len(x['valid']), 'thr': x['thr']} for s, x in stress.items()},
                          'events_with_bars': ev_n, 'blue': blue, 'hyps': hy, 'map_mri': mp, 'h4': h4}
        del Fs, bars
    out['summary'] = summary(res)
    out['figure'] = os.path.basename(pp) if plot(pp, res) else None
    out['minutes'] = (time.time() - t0) / 60
    save_json(jp, out)
    print(f"\n💾 {jp}" + (f"\n🖼️ {pp}" if out['figure'] else '') + f" | {out['minutes']:.1f} мин")
    return out, res


# =====================================================================
# 'stand' (синтетика)
# =====================================================================
def stand_events(t0, weeks):
    """Календарь стенда (ET): понедельник и среда 08:30, вторник 14:00; четверг и пятница — без релизов (для пар)."""
    ev = []
    d0 = pd.Timestamp(t0).normalize()                         # t0 — понедельник
    for w in range(weeks):
        mon = d0 + pd.Timedelta(days=7 * w)
        ev.append((mon.strftime('%Y-%m-%d'), '08:30', 'CPI'))
        ev.append(((mon + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), '14:00', 'FOMC'))
        ev.append(((mon + pd.Timedelta(days=2)).strftime('%Y-%m-%d'), '08:30', 'NFP'))
    return tuple(ev)


def stand_signs(rng, n, situ, kind):
    """Знаки ходов мира стенда: 'plus' — в ситуациях знак следует состоянию (±1, меняется в среднем раз в 1 / flip баров) с
       вероятностью (1 + kappa) / 2, вне ситуаций — монетка; 'null' — везде монетка."""
    P = STAND
    coin = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    if kind != 'plus':
        return coin
    sw = rng.random(n) < P['flip']
    state = np.where(np.cumsum(sw) % 2 == 0, 1.0, -1.0) * (1.0 if rng.random() < 0.5 else -1.0)
    agree = rng.random(n) < (1 + P['kappa']) / 2
    return np.where(situ, np.where(agree, state, -state), coin)


def stand_skeleton(seed):
    """Каркас мира стенда: STAND['assets'] монет 5M, два периода по STAND['weeks'] недель; календарь stand_events; дни
       стресса — 12% дней после первого месяца (общие для монет); профиль часов: тонкие — 00–07 UTC, ликвидные — 12–19 UTC;
       ситуации — тонкие часы, дни стресса, окна релизов."""
    P = STAND
    rng = np.random.default_rng(seed)
    nb = P['weeks'] * 7 * 288
    n = 2 * nb
    T = pd.Timestamp(P['t0']).value + BAR_NS['5M'] * np.arange(n, dtype=np.int64)
    events = stand_events(P['t0'], 2 * P['weeks'])
    hr = (T // HOUR_NS) % 24
    day = T // DAY_NS
    du = np.unique(day)
    cand = du[EFF['stress_lookback']:]                        # у дней первого месяца нет прошлых дней для отношения
    st_days = rng.choice(cand, int(round(len(cand) * P['stress_share'])), replace=False)
    stress_bar = np.isin(day, st_days)
    nm = news_masks(T, BAR_NS['5M'], events)
    tau = np.array([event_utc(d, hm) for d, hm, _ in events], dtype=np.int64)
    how = np.arange(168)
    prof = {'ok': True, 'weeks': 0, 'thin': how[how % 24 < 8], 'liquid': how[(how % 24 >= 12) & (how % 24 < 20)]}
    return {'n': n, 'nb': nb, 'T': T, 'events': events, 'st_days': st_days, 'stress_bar': stress_bar,
            'first_news': np.isin(T, tau),                    # бар 5M, открывшийся в момент релиза
            'situ': (hr < 8) | stress_bar | nm['08:30'][0] | nm['14:00'][0], 'prof': prof}


def stand_world(kind, seed):
    """Мир цен (ст3, классика): 'plus' — в ситуациях знак хода с памятью; 'null' — везде монетка, но в ситуациях крупные и
       рваные свечи (рывки в дни стресса, крупный бар релиза). Крупные свечи дней стресса — в обоих мирах (по ним RV
       находит дни). Возвращает (кадры периодов, профиль, дни стресса, календарь)."""
    P = STAND
    S = stand_skeleton(seed)
    n, nb, T = S['n'], S['nb'], S['T']
    parts = {0: [], 1: []}
    btc = None
    for ia in range(P['assets']):
        r_ = np.random.default_rng([seed, ia])
        eps = np.abs(r_.standard_normal(n))
        eps[S['stress_bar']] *= P['stress_vol']
        if kind != 'plus':                                    # рывки и бар релиза — только в мире без связи: их снимает близнец
            eps[S['stress_bar'] & (r_.random(n) < P['jump_p'])] *= P['jump']
            eps[S['first_news']] *= P['news_bar']
        c = 100.0 * np.exp(np.cumsum(0.001 * eps * stand_signs(r_, n, S['situ'], kind)))
        if ia == 0:
            btc = {'T': T, 'c': c, 'n_pre': 0}
        for code in (0, 1):
            in_per = np.zeros(n, bool)
            in_per[code * nb:(code + 1) * nb] = True
            parts[code].append((ia, frame_part(T, c, None, in_per, np.random.default_rng([seed, ia, code, 5]),
                                               parts=('classic',))))
    Fs = {src: stack_frames(parts[code]) for code, src in enumerate(EFF['periods'])}
    bounds = {'search': (int(T[0]), int(T[nb])), 'confirm': (int(T[nb]), 2 ** 62)}
    return Fs, S['prof'], stress_days([btc], BAR_NS['5M'], bounds), S['events']


def _smooth(rng, n):
    """Гладкий шум с дисперсией 1: AR(1) с памятью ~1 / (1 − z_rho) окон (окна подряд перекрываются)."""
    from scipy.signal import lfilter
    rho = STAND['z_rho']
    return lfilter([math.sqrt(1.0 - rho * rho)], [1.0, -rho], rng.standard_normal(n))


def stand_zworld(kind, seed):
    """Мир линии M (ст4): у каждой монеты по барам — z реала, суррогата sflip4 и близнеца sflip1 (окно кончается на баре).
       Общая часть z (размеры свечей окна) у всех трёх одна, своя — у каждого; z реала в среднем z_level (рынок неслучаен,
       база линии m > 0). В ситуациях: 'order' — z реала выше на d_order (порядок ходов неслучайнее); 'sizes' — z всех трёх
       выше на d_size (оценку модели двигают одни размеры свечей); 'coin' — z реала ниже на d_coin (рынок ближе к монетке).
       Цвета — формулой линии M сайта. Возвращает (кадры периодов, профиль, дни стресса, календарь)."""
    P = STAND
    S = stand_skeleton(seed)
    n, nb, T, situ = S['n'], S['nb'], S['T'], S['situ'].astype(np.float64)
    parts = {0: [], 1: []}
    for ia in range(P['assets']):
        r_ = np.random.default_rng([seed, ia, 11])
        com = P['z_common'] * _smooth(r_, n)
        zr = P['z_level'] + com + P['z_own'] * _smooth(r_, n)
        z4 = com + P['z_own'] * _smooth(r_, n)
        z1 = P['z_s1'] + com + P['z_own'] * _smooth(r_, n)
        if kind == 'order':
            zr = zr + P['d_order'] * situ
        elif kind == 'sizes':
            zr, z4, z1 = (v + P['d_size'] * situ for v in (zr, z4, z1))
        elif kind == 'coin':
            zr = zr - P['d_coin'] * situ
        cols = line_colors(zr, z4, z1, EFF['ceiling']['5M'])
        for code in (0, 1):
            in_per = np.zeros(n, bool)
            in_per[code * nb:(code + 1) * nb] = True
            parts[code].append((ia, frame_part(T, None, cols, in_per, None, parts=('mri',))))
    Fs = {src: stack_frames(parts[code]) for code, src in enumerate(EFF['periods'])}
    day = T // DAY_NS
    first = np.unique(day)[EFF['stress_lookback']]            # дни с историей — как у дней стресса цен
    stress = {}
    for code, src in enumerate(EFF['periods']):
        dd = np.unique(day[code * nb:(code + 1) * nb])
        dd = dd[dd >= first]
        stress[src] = {'stress': np.intersect1d(S['st_days'], dd), 'valid': dd, 'thr': math.nan}
    return Fs, S['prof'], stress, S['events']


def stand_measures(check):
    """(ст1) меры окна: прямой счёт, близнецы, случайный ряд, ряды с памятью, обрезка ряда."""
    print("\n🧪 (ст1) меры окна: AC1, VR16, VR4x4 у реала и близнецов")
    rng = np.random.default_rng(1)
    n = 30_000
    r = 0.001 * rng.standard_normal(n) * np.exp(0.5 * np.sin(np.arange(n) / 500))
    c = 100 * np.exp(np.cumsum(r))
    sg = twin_signs(n, np.random.default_rng(2))
    K = classics(c, sg)
    worst = 0.0
    for t in np.random.default_rng(3).choice(np.arange(EFF['L'] + 1, n), 300, replace=False):
        d = classics_direct(c, int(t), sg)
        for k, v in d.items():
            worst = max(worst, abs(K[k][t] - v))
    check("меры окна векторно = прямой счёт окно за окном (300 окон, реал и близнецы)", worst < 1e-9,
          f"наибольшее расхождение {worst:.1e} (AC1 и ln VR — порядка 0.01–1)")
    s1, sb = sg
    R4 = _roll(np.r_[0.0, np.diff(np.log(c))], 4)
    ok_sizes = np.array_equal(np.abs(r * s1), np.abs(r)) and np.allclose(np.abs(R4[3:] * sb[3:]), np.abs(R4[3:]))
    big = twin_signs(1_000_000, np.random.default_rng(4))
    check("близнецы сохраняют размеры (бар — у AC1 / VR16, четвёрку — у VR4x4), знаки — монетка",
          ok_sizes and abs(big[0].mean()) < 0.005 and abs(big[1].mean()) < 0.005,
          f"доля «+» {100 * (big[0] > 0).mean():.2f}% / {100 * (big[1] > 0).mean():.2f}%")

    def e_of(c_, seed):
        K_ = classics(c_, twin_signs(len(c_), np.random.default_rng(seed)))
        cap = EFF['vr_cap']
        return {'AC1': np.abs(K_['ac_r']) - np.abs(K_['ac_t']),
                'VR16': np.minimum(np.abs(K_['lv16_r']), cap) - np.minimum(np.abs(K_['lv16_t']), cap),
                'VR4x4': np.minimum(np.abs(K_['lv44_r']), cap) - np.minimum(np.abs(K_['lv44_t']), cap)}, K_

    def blk_se(x, k=200):
        x = x[np.isfinite(x)]
        b = np.array([v.mean() for v in np.array_split(x, k)])
        return x.mean(), b.std(ddof=1) / math.sqrt(k)
    n2 = 1_000_000
    rw = 100 * np.exp(np.cumsum(0.001 * np.random.default_rng(5).standard_normal(n2)))
    E, _ = e_of(rw, 6)
    z = {m: blk_se(E[m]) for m in E}
    check("случайный ряд: e ~ 0 у всех мер (|среднее| < 3 ошибок по блокам)", all(abs(mu) < 3 * se for mu, se in z.values()),
          ', '.join(f"{m} {mu:+.5f} ± {se:.5f}" for m, (mu, se) in z.items()))
    msg, ok = [], True
    for rho in (0.2, -0.2):
        g = np.random.default_rng(7 if rho > 0 else 8)
        x = np.zeros(400_000)
        w = g.standard_normal(len(x))
        for i in range(1, len(x)):
            x[i] = rho * x[i - 1] + w[i]
        E, K_ = e_of(100 * np.exp(np.cumsum(0.001 * x)), 9)
        ac, lv = np.nanmean(K_['ac_r']), np.nanmean(K_['lv16_r'])
        e1, e2 = np.nanmean(E['AC1']), np.nanmean(E['VR16'])
        ok &= abs(ac - rho) < 0.04 and np.sign(lv) == np.sign(rho) and e1 > 0 and e2 > 0
        msg.append(f"ρ {rho:+.1f}: AC1 {ac:+.3f}, ln VR16 {lv:+.3f}, e AC1 {e1:+.3f}, e VR16 {e2:+.3f}")
    check("ряд с памятью знака (AR(1) ±0.2): AC1 ~ ρ, знак ln VR16 = знак ρ, e AC1 и e VR16 > 0", ok, '; '.join(msg))
    g = np.random.default_rng(10)
    m_ = 400_000
    sw = g.random(m_) < STAND['flip']
    state = np.where(np.cumsum(sw) % 2 == 0, 1.0, -1.0)
    sgn = np.where(g.random(m_) < 0.75, state, -state)
    E, K_ = e_of(100 * np.exp(np.cumsum(0.001 * np.abs(g.standard_normal(m_)) * sgn)), 11)
    lv44, e44 = np.nanmean(K_['lv44_r']), np.nanmean(E['VR4x4'])
    check("ряд с памятью четвёрок (знак держится ~16 баров): ln VR4x4 > 0.05, e VR4x4 > 0", lv44 > 0.05 and e44 > 0,
          f"ln VR4x4 {lv44:+.3f}, e {e44:+.3f}")
    cut = 17_777
    Kc = classics(c[:cut], (s1[:cut], sb[:cut]))
    same = all(np.array_equal(Kc[k], K[k][:cut], equal_nan=True) for k in K)
    check("обрезка ряда: меры на барах до обреза — те же бит в бит (мера бара t — только из баров <= t)", same,
          f"обрез на баре {cut}")


def stand_labels(check):
    """(ст2) ситуации: час недели, профиль объёма, ET -> UTC, календарь, окна релизов и пары, дни стресса."""
    print("\n🧪 (ст2) ситуации: часы недели, профиль, ET, календарь, релизы, стресс")
    ts = [('2024-01-01 00:00', 0), ('2024-01-07 23:00', 167), ('2024-01-04 13:00', 85), ('2025-03-09 07:00', 151)]
    ok = all(int(hour_of_week([pd.Timestamp(s, tz='UTC').value])[0]) == v for s, v in ts)
    check("час недели UTC: пн 00:00 — 0, вс 23:00 — 167", ok)
    rng = np.random.default_rng(12)
    true = np.exp(np.linspace(0.0, 4.0, 168))[rng.permutation(168)]   # «настоящий» профиль: все часы разные
    t0 = pd.Timestamp('2023-12-04 00:00', tz='UTC').value         # понедельник
    t = t0 + BAR_NS['5M'] * np.arange(40 * 7 * 288, dtype=np.int64)
    how = hour_of_week(t)
    wk_level = np.exp(rng.normal(0, 0.3, 40))[(t - t0) // WEEK_NS]
    qv = true[how] * wk_level * np.exp(rng.normal(0, 0.05, len(t)))
    late = t >= pd.Timestamp(EFF['profile_before']).value
    qv[late] = (200.0 - true[how[late]]) * 5                      # после границы — чужой профиль: не должен влиять
    pr = hour_profile(t // 10 ** 6, qv)
    k = EFF['hours_n']
    want_thin, want_liq = np.sort(np.argsort(true)[:k]), np.sort(np.argsort(true)[-k:])
    check("профиль объёма: тонкие и ликвидные часы — ровно подсаженные, данные после границы не влияют",
          pr['ok'] and np.array_equal(pr['thin'], want_thin) and np.array_equal(pr['liquid'], want_liq),
          f"годных недель {pr['weeks']}")
    conv = [('2025-03-07', '08:30', '2025-03-07 13:30'), ('2025-03-12', '08:30', '2025-03-12 12:30'),
            ('2024-11-07', '14:00', '2024-11-07 19:00'), ('2024-10-10', '08:30', '2024-10-10 12:30'),
            ('2025-07-30', '14:00', '2025-07-30 18:00'), ('2025-12-10', '14:00', '2025-12-10 19:00')]
    ok = all(event_utc(d, hm) == pd.Timestamp(u, tz='UTC').value for d, hm, u in conv)
    check("ET -> UTC через переходы на летнее / зимнее время (EST +5 ч, EDT +4 ч)", ok,
          ', '.join(f"{d} {hm} ET -> {u[11:]}" for d, hm, u in conv[:3]))
    kinds = {'CPI': '08:30', 'NFP': '08:30', 'FOMC': '14:00'}
    keys = [(d, hm) for d, hm, _ in EVENTS]
    wd_ok = all(pd.Timestamp(d).dayofweek < 5 for d, _, _ in EVENTS)
    by_year = {}
    for d, _, k_ in EVENTS:
        by_year[(k_, d[:4])] = by_year.get((k_, d[:4]), 0) + 1
    want = {('CPI', '2024'): 6, ('CPI', '2025'): 11, ('CPI', '2026'): 10, ('NFP', '2024'): 6, ('NFP', '2025'): 11,
            ('NFP', '2026'): 10, ('FOMC', '2024'): 4, ('FOMC', '2025'): 8, ('FOMC', '2026'): 7}
    check("календарь: без повторов, только будни, время по виду релиза, число релизов по годам как в источниках",
          len(set(keys)) == len(keys) and wd_ok and all(kinds[k_] == hm for _, hm, k_ in EVENTS) and by_year == want,
          ', '.join(f"{k_} {y}: {v}" for (k_, y), v in sorted(by_year.items())))
    ev = (('2025-03-12', '08:30', 'CPI'), ('2025-03-19', '14:00', 'FOMC'))
    evd = np.array([np.datetime64(x, 'D').astype(np.int64) for x, _, _ in ev])
    good, msg, good_pre = True, [], True
    for tf in ('5M', '1H'):
        b = BAR_NS[tf]
        T = pd.Timestamp('2025-03-10 00:00', tz='UTC').value + b * np.arange(14 * DAY_NS // b, dtype=np.int64)
        nm = news_masks(T, b, ev)
        npre = news_masks(T, b, ev, pre=True)
        for s, (d, hm, _) in zip(EFF['slots'], ev):
            news, ref = nm[s]
            tau = event_utc(d, hm)
            fo = int(T[news][0]) if news.any() else -1
            good &= int(news.sum()) == EFF['news_h'] * HOUR_NS // b and fo <= tau < fo + b    # первый бар — с релизом
            good &= bool(ref.any()) and not bool((news & ref).any())
            _, wd_, dd = et_of(T[ref] + b)
            good &= bool((wd_ < 5).all()) and not bool(np.isin(dd, evd).any())
            msg.append(f"{tf} {s}: баров {int(news.sum())}, первый открыт "
                       f"{pd.Timestamp(fo, tz='UTC').tz_convert(ET).strftime('%H:%M')} ET")
            pre, pref = npre[s]                                    # до релиза: вплотную к окну релиза, не позже релиза
            lp = int(T[pre][-1]) if pre.any() else -1
            good_pre &= int(pre.sum()) == EFF['news_h'] * HOUR_NS // b and lp + b == fo and lp + b <= tau
            good_pre &= not bool((pre & news).any()) and bool(pref.any()) and not bool((pre & pref).any())
            _, wd_, dd = et_of(T[pref] + b)
            good_pre &= bool((wd_ < 5).all()) and not bool(np.isin(dd, evd).any())
    check("окна релизов: 4 ч после релиза, первый бар — тот, внутри которого релиз; пары — та же пора будней без релизов, "
          "без пересечений", good, '; '.join(msg))
    check("окна «4 ч до релиза»: 4 ч вплотную к окну релиза, последний бар кончается не позже релиза; пары — та же пора "
          "будней без релизов, без пересечений", good_pre)
    rng = np.random.default_rng(13)
    n = 120 * 288
    T = pd.Timestamp('2024-01-01', tz='UTC').value + BAR_NS['5M'] * np.arange(n, dtype=np.int64)
    day = T // DAY_NS
    du = np.unique(day)
    planted = rng.choice(du[40:], 8, replace=False)
    vol = np.where(np.isin(day, planted), 3.0, 1.0)
    c = 100 * np.exp(np.cumsum(0.001 * vol * rng.standard_normal(n)))
    sd = stress_days([{'T': T, 'c': c, 'n_pre': 0}], BAR_NS['5M'], {'p': (int(du[40] * DAY_NS), 2 ** 62)})['p']
    hit = len(set(sd['stress'].tolist()) & set(planted.tolist()))
    check("дни стресса: найдены подсаженные дни с крупными свечами (верхние 10% отношения RV в периоде)", hit >= 7,
          f"подсажено 8, найдено {hit}, отмечено {len(sd['stress'])} из {len(sd['valid'])} годных дней")


def _tally(kind, worlds, ms):
    """Сводка миров: сколько сидов каждая мера видит каждую Нk (и обратный знак), z = Δ / ошибка."""
    rows = []
    for s, (Fs, prof, stress, events) in worlds:
        hy = period_stats(Fs, BAR_NS['5M'], prof, stress, events, seed=s, measures=ms)
        rows.append({'seen': {(h, m): sees(hy[h]['stats'][m]) for h in HYPS for m in ms},
                     'inv': {(h, m): inverse(hy[h]['stats'][m]) for h in HYPS for m in ms},
                     'z': {(h, m, src): hy[h]['stats'][m][src]['t'] for h in HYPS for m in ms for src in EFF['periods']}})
    seen_n = {(h, m): sum(r_['seen'][(h, m)] for r_ in rows) for h in HYPS for m in ms}
    inv_n = {(h, m): sum(r_['inv'][(h, m)] for r_ in rows) for h in HYPS for m in ms}
    print(f"   {kind:<6}: видит / обратный знак (из {len(rows)}) — " + ' | '.join(
        f"{h.replace('H', 'Н')} " + ' '.join(f"{LABEL[m]} {seen_n[(h, m)]} / {inv_n[(h, m)]}" for m in ms) for h in HYPS))
    return rows


def _calib(rows, ms):
    """Разброс и среднее z = Δ / ошибка по всем Нk, периодам и сидам."""
    out = {}
    for m in ms:
        z = np.array([v for r_ in rows for (h, m_, src), v in r_['z'].items() if m_ == m and np.isfinite(v)])
        out[m] = (float(z.std(ddof=1)), float(z.mean()))
    return out


def stand_worlds(check):
    """(ст3) миры цен, классика: подсаженная неэффективность в ситуациях и мир без неё (с крупными и рваными свечами)."""
    P = STAND
    print(f"\n🧪 (ст3) миры цен, классика: «плюс» — {P['seeds_plus']} сидов, «без связи» — {P['seeds_null']} ({P['assets']} "
          f"монеты × 2 × {P['weeks']} недель 5M; в ситуациях знак хода с памятью ~{1 / P['flip']:.0f} баров, kappa "
          f"{P['kappa']:g})")
    tally = {}
    for kind, n_s, base in (('plus', P['seeds_plus'], 0), ('null', P['seeds_null'], 100)):
        tally[kind] = _tally(kind, ((s, stand_world(kind, s)) for s in range(base, base + n_s)), CLASSIC)
    pairs = [(h, m) for h in HYPS for m in CLASSIC if not (h == 'H3' and m == 'VR4x4')]
    worst = min(sum(r_['seen'][p_] for r_ in tally['plus']) for p_ in pairs)
    v44 = sum(r_['seen'][('H3', 'VR4x4')] for r_ in tally['plus'])
    check(f"мир с подсаженной неэффективностью: каждая классическая мера видит каждую Нk хотя бы в {P['need_plus']} сидах "
          f"из {P['seeds_plus']} (VR4x4 на Н3 — справочно)", worst >= P['need_plus'],
          f"худшая пара мера × Нk — {worst}; VR4x4 на Н3 — {v44}")
    false_ = sum(sum(r_['seen'].values()) for r_ in tally['null'])
    check(f"мир без связи (крупные и рваные свечи в ситуациях): ложных «видит» не больше {P['false_max']} на все меры и Нk "
          f"{P['seeds_null']} сидов", false_ <= P['false_max'], f"ложных {false_}")
    lo, hi = P['calib']
    sds = _calib(tally['null'], CLASSIC)
    check(f"мир без связи: z = Δ / ошибка у каждой меры — разброс от {lo:g} до {hi:g}, среднее по модулю < 0.5",
          all(lo <= sd <= hi and abs(mu) < 0.5 for sd, mu in sds.values()),
          ', '.join(f"{m} sd {sd:.2f}, среднее {mu:+.2f}" for m, (sd, mu) in sds.items()))
    return tally


def stand_line(check):
    """(ст4) линия M на двоих: копия = FL.line_m бит в бит; база окна — только из окон до него; log M близнеца = формула в
       точке его z при базе реала; миры z окон «порядок», «размеры», «монетка» — что видит мера MRI v2 и мера v1."""
    P = STAND
    print("\n🧪 (ст4) линия M: цвет близнеца на линии реала; миры z окон («порядок», «размеры», «монетка»)")
    try:
        load_mods()
    except (FileNotFoundError, ImportError, AttributeError, ValueError) as e:
        check("модули SuMo_Class_Legs и SuMo_Class_Flow доступны (линия M — их формулой)", False, str(e))
        return None
    L_ = FL.FLOW['line']
    rng = np.random.default_rng(21)
    n = 30_000
    same = all(np.array_equal(FL.line_m(z), line_parts(z)[0], equal_nan=True)
               for z in (rng.standard_normal(n), 0.8 + _smooth(rng, n), -0.3 + 0.5 * rng.standard_normal(n)))
    check("копия линии M (line_parts) = FL.line_m бит в бит (ряды z: случайный, «реальный», «синий»)", same)
    z, zt = 0.8 + _smooth(rng, n), _smooth(rng, n)
    _, m, s = line_parts(z)
    lx = line_parts(z, zt)[0]
    base_ok, worst = True, 0.0
    for p in rng.choice(np.arange(L_['show_from'], n), 200, replace=False):
        z2 = z[:p + 1].copy()
        z2[p - L_['lag'] + 1:] += 3.0                           # последние lag окон (с текущим) — другие
        _, m2, s2 = line_parts(z2)
        base_ok &= bool(m2[p] == m[p] and s2[p] == s[p])
        zc = min(zt[p], m[p] / max(1.0 - s[p] * s[p], 1e-12)) if s[p] < 1.0 else zt[p]
        want = -(zc - m[p]) ** 2 / (2 * s[p] * s[p]) - math.log(s[p]) + zc * zc / 2
        worst = max(worst, abs(lx[p] - want))
    check("база линии в окне — только из окон до него (последние lag окон её не меняют); log M близнеца = формула в точке его "
          "z при базе реала (200 окон)", base_ok and worst < 1e-12, f"наибольшее расхождение {worst:.1e}")
    ms = ('MRI', 'MRIz', 'MRIv1')
    print(f"   миры: {P['assets']} монеты × 2 × {P['weeks']} недель 5M; z реала в среднем {P['z_level']:g}, общая часть z "
          f"(размеры свечей окна) у реала и близнецов одна; сдвиг в ситуациях — {P['d_order']:g}")
    tally = {}
    for kind, n_s, base in (('order', P['seeds_plus'], 200), ('sizes', P['seeds_null'], 300), ('coin', P['seeds_plus'], 400)):
        tally[kind] = _tally(kind, ((s, stand_zworld(kind, s)) for s in range(base, base + n_s)), ms)
    cnt = lambda kind, key, m: {h: sum(r_[key][(h, m)] for r_ in tally[kind]) for h in HYPS}
    so, sz = cnt('order', 'seen', 'MRI'), cnt('order', 'seen', 'MRIz')
    check(f"«порядок» (z реала выше в ситуациях): MRI и MRI без линии видят каждую Нk хотя бы в {P['need_plus']} сидах из "
          f"{P['seeds_plus']}", min(so.values()) >= P['need_plus'] and min(sz.values()) >= P['need_plus'],
          f"MRI {so}, без линии {sz}".replace("'", ''))
    false_ = sum(r_['seen'][(h, m)] for r_ in tally['sizes'] for h in HYPS for m in ('MRI', 'MRIz'))
    v1 = cnt('sizes', 'seen', 'MRIv1')
    check(f"«размеры» (z всех выше в ситуациях): ложных «видит» MRI и MRI без линии не больше {P['false_max']} на все Нk "
          f"{P['seeds_null']} сидов, а мера v1 «видит» Н2 хотя бы в {P['need_leak']} — дефект v1 записан стендом",
          false_ <= P['false_max'] and v1['H2'] >= P['need_leak'], f"ложных {false_}; мера v1 видит {v1}".replace("'", ''))
    lo, hi = P['calib']
    sds = _calib(tally['sizes'], ('MRI', 'MRIz'))
    check(f"«размеры»: z = Δ / ошибка у MRI и MRI без линии — разброс от {lo:g} до {hi:g}, среднее по модулю < 0.5",
          all(lo <= sd <= hi and abs(mu) < 0.5 for sd, mu in sds.values()),
          ', '.join(f"{LABEL[m]} sd {sd:.2f}, среднее {mu:+.2f}" for m, (sd, mu) in sds.items()))
    ci, cs = cnt('coin', 'inv', 'MRI'), cnt('coin', 'seen', 'MRI')
    check(f"«монетка» (z реала ниже в ситуациях): у MRI обратный знак на каждой Нk хотя бы в {P['need_plus']} сидах из "
          f"{P['seeds_plus']}, «видит» — ни разу", min(ci.values()) >= P['need_plus'] and sum(cs.values()) == 0,
          f"обратный знак {ci}".replace("'", ''))
    return tally


def write_fut_csv(path, t0, t1, seed):
    """Синтетический CSV фьючерса 5M в формате research/futures_klines.py: объём по часу недели с рисунком."""
    rng = np.random.default_rng(seed)
    t = np.arange(pd.Timestamp(t0).value, pd.Timestamp(t1).value, BAR_NS['5M'], dtype=np.int64)
    how = hour_of_week(t)
    pat = 1.0 + 0.8 * np.sin(2 * np.pi * (how % 24) / 24) + 0.3 * (how // 24 >= 5)
    c = 40_000 * np.exp(np.cumsum(0.001 * rng.standard_normal(len(t))))
    o = np.r_[c[0], c[:-1]]
    h, l = np.maximum(o, c) * 1.0005, np.minimum(o, c) * 0.9995
    v = pat * np.exp(rng.normal(0, 0.2, len(t))) * 10
    df = pd.DataFrame({'timestamp': pd.to_datetime(t, utc=True).tz_localize(None), 'open': o, 'high': h, 'low': l, 'close': c,
                       'volume': v, 'close_time': t // 10 ** 6 + 299_999, 'quote_av': v * c, 'trades': 100,
                       'tb_base_av': v / 2, 'tb_quote_av': v * c / 2, 'ignore': 0})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def stand_e2e(check):
    """(ст5) сквозной путь: синтетические файлы research, кэши Flow и CSV фьючерса -> 'run' (оба ТФ); цвета = Legs."""
    global RUN_DIR, OUT_DIR, DATA_5M, DATA_1H, FUT_DIR
    print("\n🧪 (ст5) сквозной путь: синтетические файлы research, кэши Flow, CSV фьючерса -> 'run'")
    try:
        load_mods()
    except (FileNotFoundError, ImportError, AttributeError, ValueError) as e:
        check("модули SuMo_Class_Legs и SuMo_Class_Flow доступны (без них 'run' не пройти)", False, str(e))
        return
    tmp = tempfile.mkdtemp(prefix='smceff_') + '/'
    keep = (RUN_DIR, OUT_DIR, DATA_5M, DATA_1H, FUT_DIR, LG.RUN_DIR, LG.DATA_5M, LG.DATA_1H, FL.OUT_DIR, FL.FUT_DIR)
    RUN_DIR, OUT_DIR, DATA_5M, DATA_1H = tmp + 'runs/', tmp + 'runs/eff/', tmp + 'data/', tmp + 'data_1h/'
    FUT_DIR = tmp + 'fut/'
    LG.RUN_DIR, LG.DATA_5M, LG.DATA_1H = RUN_DIR, DATA_5M, DATA_1H
    FL.OUT_DIR, FL.FUT_DIR = RUN_DIR + 'flow/', FUT_DIR
    try:
        assets = ['BTCUSDT', 'ETHUSDT', 'FILUSDT']
        rng = np.random.default_rng(9)
        spans = {'5M': ('2023-12-15 00:00:00', '2024-09-01 00:00:00', '2024-11-01 00:00:00'),
                 '1H': ('2023-01-01 00:00:00', '2025-07-01 00:00:00', '2026-06-01 00:00:00')}
        items = [{'tag': 'syn_stop_a', 'src': 'search'}, {'tag': 'syn_refit_a', 'src': 'confirm'}]
        for tf in ('5M', '1H'):
            tr0, va0, end = spans[tf]
            caches = {}
            for i, a in enumerate(assets):
                ser = LG.write_synth_research(DATA_5M if tf == '5M' else DATA_1H, a, tf, tr0, va0, end, 50 + 7 * i)
                for part, key, tag in (('tune', 'tr', 'syn_stop_a'), ('val', 'va', 'syn_refit_a')):
                    t = ser[f't_{key}']
                    t = t[t >= pd.Timestamp(EFF['tune_start']).value] if part == 'tune' else t[LG.LEGS['N']:]
                    caches.setdefault(part, {})[a] = {
                        't': t.astype(np.int64), 'wk': t // WEEK_NS,
                        f'{tag}__sp': rng.normal(0.5, 1.0, len(t)).astype(np.float32),
                        f'{tag}__s4': rng.normal(0.0, 1.0, len(t)).astype(np.float32),
                        f'{tag}__s1': rng.normal(-0.2, 1.0, len(t)).astype(np.float32)}
            for part in ('tune', 'val'):
                its = [it for it in items if (it['src'] == 'search') == (part == 'tune')]
                FL.save_cache(tf, part, caches[part], its, {'ok': True}, 'stand')
        write_fut_csv(FUT_DIR + 'BTCUSDT-5M.csv', '2023-12-01 00:00:00', '2024-07-15 00:00:00', 3)
        t1 = time.time()
        out, res = mode_run(tfs=('5M', '1H'))
        ok = out.get('figure') and all(tf in res for tf in ('5M', '1H'))
        full = ok and all(res[tf]['hyps'][h]['stats'][m].get(src) and res[tf]['hyps'][h]['stats'][m][src]['nA'] > 0
                          and res[tf]['hyps'][h]['stats'][m][src]['nB'] > 0
                          for tf in ('5M', '1H') for h in HYPS for m in MEASURES for src in EFF['periods'])
        full = full and all(res[tf]['hyps'][h]['stats'][m].get(src) for tf in ('5M', '1H') for h in tuple(HYPS) + (PRE,)
                            for m in MRI_REF + ('MRI_btc',) for src in EFF['periods']) and 'summary' in out
        check("'run' на синтетике: оба ТФ, Н1–Н3 × 4 меры × 2 периода с барами в A и B, справочные меры и «до релиза», итог, "
              "json и рисунок", bool(full), f"{(time.time() - t1) / 60:.1f} мин")
        same = True
        for tf in ('5M', '1H'):
            bars = {a: LG.load_bars(a, tf) for a in assets}
            for src, part in EFF['periods'].items():
                C = FL.load_cache(tf, part)
                mine, _ = eff_colors(C, src, bars, tf)
                legs, _ = LG.source_colors(C, src, bars, tf)
                same &= set(mine) == set(legs) and all(
                    np.array_equal(mine[a][k], legs[a][k], equal_nan=True) for a in legs for k in ('c', 'ct'))
        check("цвета реала и близнеца sflip1 = source_colors Legs бит в бит (оба ТФ, оба периода)", same)
    finally:
        (RUN_DIR, OUT_DIR, DATA_5M, DATA_1H, FUT_DIR, LG.RUN_DIR, LG.DATA_5M, LG.DATA_1H, FL.OUT_DIR, FL.FUT_DIR) = keep


def mode_stand():
    t0 = time.time()
    checks = []

    def check(name, ok, detail=''):
        checks.append((name, bool(ok)))
        print(f"   {'✅' if ok else '❌'} {name}" + (f" — {detail}" if detail else ""), flush=True)

    print(f"\n{'#' * 110}\n# SuMo_Class_Eff v{EFF_VER} | СТЕНД (синтетика)\n{'#' * 110}")
    stand_measures(check)
    stand_labels(check)
    stand_worlds(check)
    stand_line(check)
    stand_e2e(check)
    n_ok = sum(ok for _, ok in checks)
    print(f"\n🏁 СТЕНД: {n_ok} / {len(checks)} ✅ | {(time.time() - t0) / 60:.1f} мин")
    return checks


@contextlib.contextmanager
def _tee(path):
    """Лог в файл и на экран."""
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


def main():
    modes = {'stand': mode_stand, 'run': mode_run}
    if MODE not in modes:
        print(f"🍆 АЙ, ОШИБКА! MODE '{MODE}' — есть {', '.join(modes)}")
        return
    log_dir = (tempfile.gettempdir() + '/') if MODE == 'stand' and not IN_COLAB else OUT_DIR + 'logs/'
    os.makedirs(log_dir, exist_ok=True)
    log_path = log_dir + f"eff_{MODE}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    with _tee(log_path):
        print(f"🗒️ лог: {log_path}")
        modes[MODE]()
    if IN_COLAB:
        try:                                       # всё записанное этой консолью — на Drive до закрытия runtime
            drive.flush_and_unmount()
            print("☁️ Drive: записи выгружены; следующий режим — запусти файл заново (Drive подключится сам)")
        except Exception as e:
            print(f"⚠️ Drive: flush_and_unmount не прошёл ({type(e).__name__}: {e}) — проверь файлы на Drive")


if __name__ == '__main__':
    main()
