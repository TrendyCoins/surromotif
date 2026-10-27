"""F1 on 5M, 15M, 30M (f1h with export/f1h_model_5m.json, _15m.json, _30m.json; rules R28, R30, R31, R32 of
docs/rules.md): the constants of the exports (the network is the one of 1H, the line — of the timeframe: 15M — family F2
4032+288, 5M / 30M — F3 4032·k0.5; ceilings 10 / 10 / 8); the loader (F2 with k and tau null, wrong families and grids
refused, the export of a timeframe by name); the compact golden files end to end (no artifacts/); the live part F2 written
out window by window == vectorized (<= 1e-12; research chart_ms, prior 'null'); on the grid of each timeframe: open time a
multiple of the bar, a grid gap — interval > 1.5 bars, first window on the candle 264 and the line from 264 +
show_from_window, bars_needed; the same candles stamped 5M / 15M / 30M / 1H give the same values (|d| = 0); research on
synthetic 15M candles (torch); with artifacts/: every export is rebuilt from its sources field for field (1H — the committed
export/f1h_model.json)."""
import copy
import dataclasses
import json
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f1h  # noqa: E402
from f1h.bars import bar_inputs  # noqa: E402
from f1h.line import line  # noqa: E402
from f1h.model import EXPORTS, tf_ms  # noqa: E402
from test_f1h import T0, candles  # noqa: E402

H = 3_600_000
TFS = ('5M', '15M', '30M')
WANT = {   # tf: (family, L, n0, k, tau, ceiling, candles of the golden, bars with line)
    '5M': ('F3', 4032, 0, 0.5, 64, 10.0, 24520, 20160),
    '15M': ('F2', 4032, 288, None, None, 10.0, 45914, 41554),
    '30M': ('F3', 4032, 0, 0.5, 64, 8.0, 22958, 18598),
    '1H': ('F3', 4032, 0, 0.5, 64, 8.0, 11480, 7120)}
ART = os.path.join(ROOT, 'artifacts')


@pytest.fixture(scope='module')
def models():
    return {tf: f1h.load_model(tf=tf) for tf in TFS + ('1H',)}


def regrid(cd, bar_ms):
    """The candles of test_f1h.candles (hourly, gaps in hours) stamped on the grid of bar_ms: the same rows and gaps."""
    return {**cd, 'time': T0 + (cd['time'] - T0) // H * bar_ms}


def run(model, cd, a=0, b=None):
    return f1h.compute(model, *(cd[k][a:b] for k in ('time', 'open', 'high', 'low', 'close')))


def same(a, b, keys=f1h.FIELDS[1:]):
    return all(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == 'f') for k in keys)


# --------------------------------------------------------------------------------------------------------------------
def test_constants(models):
    with open(EXPORTS['1H'], encoding='utf-8') as fh:
        net_1h = json.load(fh)['model']
    for tf in TFS + ('1H',):
        m = models[tf]
        fam, L, n0, k, tau, cap, _, _ = WANT[tf]
        assert (m.timeframe, m.bar_ms, m.gap_factor, m.n_norm, m.n_hist, m.window) == (tf, tf_ms(tf), 1.5, 200, 200, 64)
        live = (m.live_family, m.live_l, m.live_n0, m.live_k, m.live_tau, m.live_lag, m.s_lo)
        assert live == (fam, L, n0, k, tau, 64, 0.75)
        assert (m.show_from, m.ceiling, m.first_window_bar, m.first_line_bar) == (4096, cap, 264, 4360)
        assert m.floor == float(np.float32(0.2)) and m.orient_eps == 1e-4
        assert len(m.q) == 37 and np.all(np.diff(m.q) > 0) and len(m.zk) == 37
        with open(EXPORTS[tf], encoding='utf-8') as fh:
            ex = json.load(fh)
        assert ex['model'] == net_1h                       # one network on every timeframe
        assert ex['bars']['timeframe'] == ex['passport']['timeframe'] == tf and tf in ex['what']
        assert 'marks' not in ex['line'] and 'shades' not in ex['line']
        assert ('n0' in ex['line']['live']) == (fam == 'F2')
        assert f1h.bars_needed(m, 160) == 4520
    assert len({tuple(models[tf].q) for tf in models}) == 4    # own ruler on every timeframe


def test_loader():
    with open(EXPORTS['15M'], encoding='utf-8') as fh:
        ex = json.load(fh)
    assert ex['line']['live']['k'] is None and ex['line']['live']['tau'] is None and ex['line']['live']['n0'] == 288
    bad = []
    for path, v in ((('line', 'live', 'k'), 0.5), (('line', 'live', 'tau'), 64), (('line', 'live', 'n0'), 0),
                    (('line', 'live', 'family'), 'F1'), (('line', 'live', 'family'), 'F3'), (('bars', 'bar_ms'), H),
                    (('bars', 'timeframe'), '1H')):
        e = copy.deepcopy(ex)
        d = e
        for key in path[:-1]:
            d = d[key]
        d[path[-1]] = v
        bad.append(e)
    for e in bad:
        with pytest.raises(ValueError):
            f1h.from_export(e)
    with open(EXPORTS['1H'], encoding='utf-8') as fh:
        e1 = json.load(fh)
    assert 'n0' not in e1['line']['live'] and f1h.from_export(e1).live_n0 == 0        # the 1H export as it is
    e1['line']['live']['k'] = None
    with pytest.raises(ValueError):
        f1h.from_export(e1)
    with pytest.raises(ValueError):
        f1h.load_model(EXPORTS['15M'], tf='1H')
    assert f1h.load_model().timeframe == '1H' and f1h.load_model(tf='30m').timeframe == '30M'


@pytest.mark.parametrize('tf', TFS)
def test_golden_end_to_end(models, tf):
    path = os.path.join(ROOT, 'tests', 'golden', f'btc{tf.lower()}_f1h.csv.gz')
    assert os.path.getsize(path) <= 5 * 2 ** 20
    res = f1h.check_golden(models[tf], path)
    assert res['ok'], res
    _, _, _, _, _, cap, n, with_line = WANT[tf]
    assert (res['n_bars'], res['first_window_bar'], res['first_line_bar'], res['bars_with_line']) == (n, 264, 4360, with_line)
    g = f1h.read_golden(path)
    assert np.all(np.diff(g['time']) % tf_ms(tf) == 0) and g['time'][0] % tf_ms(tf) == 0
    st = g['strength'][g['line']]
    assert st.min() >= 1.0 and st.max() <= cap and np.isnan(g['strength'][~g['line']]).all()
    assert f1h.check_golden(models[tf])['ok']                # the golden of the timeframe by default


def test_live_f2_formulas(models):
    """The live part F2 == research chart_ms (kind 'win', prior 'null', n0 virtual windows) written out window by window."""
    m = models['15M']
    g = np.random.default_rng(31)
    lg = np.concatenate([g.normal(0.5, 1.2, 4500), g.normal(-1.0, 0.3, 200), g.normal(2.0, 1.0, 300)])
    out = line(m, lg)
    z = np.interp(lg, m.q, m.zk)
    for p in (0, 1, 63, 64, 65, 200, 4095, 4096, 4097, 4999):
        hi = p - 63
        lo = max(0, hi - 4032)
        w = z[lo:hi] if hi > lo else np.zeros(0)
        n = len(w)
        if n:
            mh = w.sum() / n
            vh = max((w * w).sum() / n - mh * mh, 0.0)
            mm, ss = w.sum() / (n + 288), math.sqrt((n * vh + 288) / (n + 288))
        else:
            mm, ss = 0.0, 1.0
        assert out['n_live'][p] == n and abs(out['m'][p] - mm) <= 1e-12 and abs(out['s'][p] - ss) <= 1e-12
        if n == 0:
            assert out['m'][p] == 0.0 and out['s'][p] == 1.0
        if p >= 4096:
            m_, s_ = max(mm, 0.0), min(max(ss, 0.75), 1.0)
            zc = min(z[p], m_ / max(1 - s_ * s_, 1e-12)) if s_ < 1 else z[p]
            want = -(zc - m_) ** 2 / (2 * s_ * s_) - math.log(s_) + zc * zc / 2
            assert abs(out['log_m'][p] - want) <= 1e-12
            assert abs(out['strength'][p] - min(math.exp(abs(out['log_m'][p])), 10.0)) <= 1e-12
        else:
            assert np.isnan(out['log_m'][p]) and not out['line'][p]
    m_f3 = dataclasses.replace(m, live_family='F3', live_n0=0, live_k=0.5, live_tau=64)    # F3 with the same ruler
    assert np.max(np.abs(line(m_f3, lg)['m'] - out['m'])) > 1e-3                          # the families differ


@pytest.fixture(scope='module')
def series():
    """Continuous candles for the line on the last one (bars_needed(1) = 4361) and candles with grid gaps."""
    return candles(4361, seed=41), candles(4700, seed=42, gaps=((1200, 1), (3000, 3)))


@pytest.mark.parametrize('tf', TFS)
def test_grid_of_timeframe(models, series, tf):
    m, bar = models[tf], tf_ms(tf)
    cont, gapped = regrid(series[0], bar), regrid(series[1], bar)
    out = run(m, cont)
    assert len(cont['time']) == f1h.bars_needed(m, 1) and out['line'][-1] and out['line'].sum() == 1
    assert np.argmax(out['window']) == 264 == m.first_window_bar and np.argmax(out['line']) == 4360 == m.first_line_bar
    for n in (0, 1, 264, 265):                                              # little history
        o = run(m, cont, 0, n)
        assert o['window'].sum() == max(n - 264, 0) and not o['line'].any()
    f = bar_inputs(gapped['time'], gapped['open'], gapped['high'], gapped['low'], gapped['close'], bar, 1.5, 200)
    for i in (1200, 3000):                                                  # missing bars: interval >= 2 bars > 1.5
        assert not f['ok'][i] and f['ok'][i - 1] and f['ok'][i + 1]
    og = run(m, gapped)
    assert not og['window'][1200:1264].any() and not og['window'][3000:3064].any() and og['line'].sum() > 100
    t_off = cont['time'].copy()
    t_off[300:] += bar // 2                                                 # off the grid of the timeframe
    with pytest.raises(ValueError):
        f1h.compute(m, t_off, cont['open'], cont['high'], cont['low'], cont['close'])
    if bar < H:
        with pytest.raises(ValueError):                                     # candles of the timeframe into the 1H model
            f1h.compute(models['1H'], *(cont[k][:300] for k in ('time', 'open', 'high', 'low', 'close')))


def test_same_candles_any_grid(models, series):
    """The network and the line do not depend on the timeframe, only the grid does: the same rows stamped 5M / 15M / 30M /
       1H give the same values, |d| = 0, with every export."""
    cd = series[1]
    for tf_model in TFS + ('1H',):
        m = models[tf_model]
        ref = None
        for tf_grid in ('5M', '15M', '30M', '1H'):
            mg = dataclasses.replace(m, bar_ms=tf_ms(tf_grid), timeframe=tf_grid)
            out = run(mg, regrid(cd, tf_ms(tf_grid)))
            if ref is None:
                ref = out
                assert out['line'].sum() > 100
            else:
                assert same(out, ref), (tf_model, tf_grid)
            if tf_model != '15M':
                break                       # all four grids with the F2 export, the own and 5M grids with the others
        if tf_model != '15M':
            out = run(dataclasses.replace(m, bar_ms=tf_ms('5M'), timeframe='5M'), regrid(cd, tf_ms('5M')))
            own = run(m, regrid(cd, m.bar_ms))
            assert same(out, own)


def test_research_synthetic_15m(models):
    """F1 on 15M against research on synthetic candles (as tests/test_f1h_research.py on 1H): research 15M files by
       build_tf_files (grid step from the data) -> training code -> logit; f1h with the 15M export: the same windows,
       |dlogit| <= 1e-4."""
    pytest.importorskip('torch')
    pytest.importorskip('sklearn')
    pd = pytest.importorskip('pandas')
    if not os.path.isfile(os.path.join(ART, 'weights', 'prod_SP_l025_d1_L64_a_slim.npz')):
        pytest.skip('artifacts/ not available')
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import f1h_research as R
    cd = regrid(candles(2600, seed=23, gaps=((700, 1), (1500, 4), (2300, 1)), flat=(1000, 330)), tf_ms('15M'))
    model = models['15M']
    out = run(model, cd)
    proj = R.project()
    T, CL, TF = R.modules(proj)
    df = pd.DataFrame({'timestamp': pd.to_datetime(cd['time'], unit='ms'), 'open': cd['open'], 'high': cd['high'],
                       'low': cd['low'], 'close': cd['close']})
    t_start, t_valid = str(df['timestamp'].iloc[0]), str(df['timestamp'].iloc[1300])
    TF.build_tf_files(df, proj + 'SMC/data/', 'SYNUSDT', '15M', 200, t_start, t_valid, 0.2)
    net, _ = R.prod_model(T)
    data = R.series_data(T, 'SYNUSDT', tf='15M')
    res = R.logits(T, CL, net, data)
    s = data.series[0]
    ends, lr = [], []
    inside = np.zeros(len(cd['time']), dtype=bool)
    for key, code, a, b in (('tr', 0, t_start, t_valid), ('va', 1, t_valid, None)):
        lo, hi = TF.part_rows(df['timestamp'], 200, a, b, key)
        sel = res['part'] == code
        ends.append(lo + (res['st'][sel] - s['off'][key]) + 63)
        lr.append(res['logit'][sel])
        inside[lo + 63:hi] = True
    ends, lr = np.concatenate(ends), np.concatenate(lr)
    assert np.array_equal(np.sort(ends), np.where(out['window'] & inside)[0])
    d = np.abs(out['logit'][ends] - lr)
    assert len(d) > 1000 and d.max() <= 1e-4, d.max()


@pytest.mark.parametrize('tf', TFS + ('1H',))
def test_export_rebuilt(tf):
    """With artifacts/: scripts/f1h_export.py rebuilds the export of the timeframe from its sources (manifests, sha256)
       field for field — the committed file, apart from the time of export and the code version (1H: rule R21 held)."""
    if not os.path.isfile(os.path.join(ART, 'pack', 'site5m_manifest.json')):
        pytest.skip('artifacts/ not available')
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import f1h_export as E
    with open(EXPORTS[tf], encoding='utf-8') as fh:
        have = json.load(fh)
    new = json.loads(json.dumps(E.build_export(tf), ensure_ascii=False))
    for d in (have, new):
        d['passport'].pop('exported')
        d['passport'].pop('code_version')
    assert new == have
    src = E.line_source(tf)[1]
    li = have['line']
    assert li['ruler']['q'] == src['ruler']['q'] and li['ruler']['zk'] == src['ruler']['zk']
    assert li['ceiling'] == src['ceiling'] and li['show_from_window'] == src['live']['show_from_window']
    assert all(li['live'][k] == src['live'][k] for k in ('key', 'family', 'L', 'lag', 'k', 'tau', 's_lo'))
