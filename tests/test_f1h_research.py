"""F1 on 1H against research on synthetic candles (rule R25; needs torch, skipped without it): candles with grid gaps, a
dead norm and a coarse price grid (exact ties) -> research 1H files (build_tf_files of research/sumo_class_tf.py) ->
SeriesData 'multi_g' / to_shapes / side_feats / prod a (training code v6.6, scripts/f1h_research.py) with the floor 0.2
-> logit against f1h on the same windows: the windows of research = the windows of f1h inside the research parts,
|dlogit| <= 1e-4. Needs artifacts/weights (the prod a weights for the research side)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f1h  # noqa: E402
from test_f1h import candles  # noqa: E402

pytest.importorskip('torch')
pytest.importorskip('sklearn')
pd = pytest.importorskip('pandas')
if not os.path.isfile(os.path.join(ROOT, 'artifacts', 'weights', 'prod_SP_l025_d1_L64_a_slim.npz')):
    pytest.skip('artifacts/ not available', allow_module_level=True)
import f1h_research as R  # noqa: E402

CASES = {'gaps_dead_norm': dict(n=2600, seed=21, gaps=((700, 1), (1500, 4), (2300, 1)), flat=(1000, 330)),
         'coarse_tick': dict(n=2600, seed=22, tick=1.0, gaps=((1900, 2),))}


@pytest.mark.parametrize('case', sorted(CASES))
def test_research_synthetic(case):
    cd = candles(**CASES[case])
    model = f1h.load_model()
    out = f1h.compute(model, cd['time'], cd['open'], cd['high'], cd['low'], cd['close'])
    proj = R.project()
    T, CL, TF = R.modules(proj)
    df = pd.DataFrame({'timestamp': pd.to_datetime(cd['time'], unit='ms'), 'open': cd['open'], 'high': cd['high'],
                       'low': cd['low'], 'close': cd['close']})
    t_start, t_valid = str(df['timestamp'].iloc[0]), str(df['timestamp'].iloc[1300])
    TF.build_tf_files(df, proj + 'SMC/data/', 'SYNUSDT', '1H', 200, t_start, t_valid, 0.2)
    net, _ = R.prod_model(T)
    data = R.series_data(T, 'SYNUSDT')
    res = R.logits(T, CL, net, data)
    s = data.series[0]
    ends, lr = [], []
    inside = np.zeros(len(cd['time']), dtype=bool)
    for key, code, a, b in (('tr', 0, t_start, t_valid), ('va', 1, t_valid, None)):
        lo, hi = TF.part_rows(df['timestamp'], 200, a, b, key)
        sel = res['part'] == code
        ends.append(lo + (res['st'][sel] - s['off'][key]) + 63)
        lr.append(res['logit'][sel])
        inside[lo + 63:hi] = True                      # windows of f1h fully inside the part
    ends, lr = np.concatenate(ends), np.concatenate(lr)
    mine = np.where(out['window'] & inside)[0]
    assert np.array_equal(np.sort(ends), mine), case
    d = np.abs(out['logit'][ends] - lr)
    assert len(d) > 1000 and d.max() <= 1e-4, (case, d.max())
    print(case, 'windows', len(d), 'max |dlogit|', float(d.max()))
