"""Session 3 gates (docs/rules.md, 'Ворота сессии 3') for the NT3_PRV_ch48 artifacts — before any NT3 calculation.

    (а) sha256 of every file of artifacts/pack/nt_manifest.json; the *_partK.npz of each run assemble without repeated keys
    (б) the latest pool summary: recipe, calibration transfer, flag (5ж) — cross-checked with the chain log of that summary
    (в) prod a / b / c: one data end, one overlay, one number of epochs = rule v4.9 (as research pool_epochs); every 'refit'
        (C and ALL) the same; data end against prod F3_g (floors of R6)
    (г) architecture in json / npz: 48 channels, kernel 3, dilations 1-2-4-8, head (96,), side 'nt_PRV' (17 columns, order of
        research SIDE_COLS printed from research when torch is installed), side_mu / side_sd of 17, 36 961 parameters
    (д) 'refit' on ALL: train_assets = prod set, same epochs; section (5з) ALL − C at BTC and panel (|Δ| > 2 SE — report)

    python scripts/nt3_gates.py [--out docs/checks/nt3/gates.json]

Paths of the training machine (Drive) are never written to the output: overlays are reduced to their folder name.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = 'NT3_PRV_ch48'
F3G_T_CUT = '2026-09-26 09:15:00'
N_PARAMS = 36961
SIDE_EXPECTED = (28, 29, 37, 38, 39, 40, 41, 42, 43, 44, 77, 78, 79, 86, 87, 88, 89)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def folder(p):
    return None if not p else os.path.basename(str(p).rstrip('/'))


def gate_a(man):
    bad = []
    for f in man['files']:
        p = os.path.join(ROOT, f['path'])
        if not os.path.isfile(p):
            bad.append({'path': f['path'], 'problem': 'missing'})
        elif os.path.getsize(p) != f['bytes'] or sha256(p) != f['sha256']:
            bad.append({'path': f['path'], 'problem': 'size or sha256'})
    parts = {}
    for p in sorted(glob.glob(os.path.join(ROOT, 'artifacts/calib', f'pool_*_{MODEL}_*part*.npz'))):
        run = re.sub(r'_part\d+\.npz$', '', os.path.basename(p))
        parts.setdefault(run, []).append(p)
    runs = {}
    for run, ps in parts.items():
        seen, dup = set(), []
        for p in ps:
            with np.load(p) as d:
                for k in d.files:
                    (dup.append(k) if k in seen else seen.add(k))
        runs[run] = {'parts': len(ps), 'keys': len(seen), 'repeated_keys': dup[:10], 'n_repeated': len(dup)}
    ok = not bad and all(r['n_repeated'] == 0 for r in runs.values()) and len(runs) == 6
    return {'files': len(man['files']), 'bad_files': bad, 'runs': runs, 'ok': bool(ok)}


def gate_b():
    summ = sorted(glob.glob(os.path.join(ROOT, 'artifacts/logs/nt/pool_summary_*.json')))[-1]
    with open(summ, encoding='utf-8') as fh:
        s = json.load(fh)
    stamp = os.path.basename(summ)[len('pool_summary_'):-len('.json')]
    log = None
    for p in sorted(glob.glob(os.path.join(ROOT, 'artifacts/logs/nt/smc_chain__*.log'))):
        with open(p, encoding='utf-8') as fh:
            txt = fh.read()
        if f'pool_summary_{stamp}.json' in txt:
            log = (os.path.basename(p), txt)
    out = {'summary': os.path.basename(summ), 'recipe_ok': s['recipe_ok'].get(MODEL), 'calib_ok': s['calib_ok'].get(MODEL),
           'nt_flag': s.get('nt_flag'), 'pine': s.get('prod_models', {}).get('pine'),
           'epochs_rule_summary': s.get('prod', {}).get('epochs', {}).get(MODEL)}
    if log:
        name, txt = log
        out['chain_log'] = name
        lines = {k: next((ln.strip() for ln in txt.splitlines() if re.search(rx, ln)), None) for k, rx in (
            ('recipe', rf"{MODEL} 'refit' \(\d+ эпох\) против 'stop'"),
            ('transfer', rf"[✅❌] {MODEL} \(сидов"),
            ('nt_vs_f3g', rf"· {MODEL} − F3_g: панель"))}
        out['log_lines'] = lines
        m = re.search(r"панель Δ VAL ([+-][\d.]+) ± ([\d.]+)", lines['recipe'] or '')
        out['recipe_delta'] = [float(m.group(1)), float(m.group(2))] if m else None
        m = re.search(r"перенос ×([\d.]+) / ×([\d.]+)", lines['transfer'] or '')
        out['transfer'] = [float(m.group(1)), float(m.group(2))] if m else None
        m = re.search(r"Δ VAL ([+-][\d.]+) ± ([\d.]+)", lines['nt_vs_f3g'] or '')
        out['nt_vs_f3g_panel_val'] = [float(m.group(1)), float(m.group(2))] if m else None
        out['log_consistent'] = bool(
            out['recipe_delta'] and out['recipe_delta'][0] >= -out['recipe_delta'][1]
            and out['transfer'] and all(abs(x - 1) <= 0.10 for x in out['transfer'])
            and out['nt_vs_f3g_panel_val'] and out['nt_vs_f3g_panel_val'][0] > -2 * out['nt_vs_f3g_panel_val'][1])
    out['ok'] = bool(out['recipe_ok'] and out['calib_ok'] and not out['nt_flag'] and out['pine'] == MODEL
                     and out.get('log_consistent'))
    return out


def load_json(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def gate_c():
    prod = {s: load_json(os.path.join(ROOT, f'artifacts/weights/prod_{MODEL}_L64_{s}.json')) for s in 'abc'}
    stop = {s: load_json(os.path.join(ROOT, f'artifacts/calib/pool_C_{MODEL}_L64_{s}.json')) for s in 'ab'}
    refit = {f'{st}_{s}': load_json(os.path.join(ROOT, f'artifacts/calib/pool_{st}_{MODEL}_L64_{s}_refit.json'))
             for st in ('C', 'ALL') for s in 'ab'}
    best = [int(stop[s]['best_ep']) + 1 for s in 'ab']
    rule = int(round(float(np.mean(best))))                 # research pool_epochs: Python round (half to even)
    f3g = {s: load_json(os.path.join(ROOT, f'artifacts/weights/prod_F3_g_L64_{s}.json'))['t_cut'] for s in 'abc'}
    row = lambda r: {'t_cut': r['t_cut'], 'overlay': folder(r.get('data_overlay')), 'epochs_cfg': r['cfg']['epochs'],
                     'epochs_run': r['epochs_run'], 'set': r['set'], 'mode': r['mode']}
    out = {'prod': {s: row(r) for s, r in prod.items()}, 'refit': {k: row(r) for k, r in refit.items()},
           'stop_best_ep_plus1': best, 'epochs_rule_v4_9': rule,
           'epochs_rule_half_up': int(np.floor(np.mean(best) + 0.5)), 'prod_F3_g_t_cut': f3g}
    pt = {r['t_cut'] for r in prod.values()}
    po = {folder(r.get('data_overlay')) for r in prod.values()}
    ep = {r['epochs_run'] for r in list(prod.values()) + list(refit.values())} | {
        r['cfg']['epochs'] for r in list(prod.values()) + list(refit.values())}
    out['one_data_end'] = len(pt) == 1
    out['one_overlay'] = po == {'data_fresh'}
    out['epochs_all_equal_rule'] = ep == {rule}
    out['same_end_as_F3_g'] = pt == set(f3g.values()) == {F3G_T_CUT}
    out['ok'] = bool(out['one_data_end'] and out['one_overlay'] and out['epochs_all_equal_rule'] and out['same_end_as_F3_g'])
    return out


def research_side_cols():
    try:
        import torch  # noqa: F401
    except ImportError:
        return None, 'torch not installed'
    os.environ.setdefault('SMC_MODE', 'none')
    os.environ.setdefault('SMC_PROJECT_PATH', tempfile.mkdtemp() + '/')
    sys.path.insert(0, os.path.join(ROOT, 'research'))
    import sumo_class_train as T
    cfg = T.make_cfg(MODEL, {'L': 64}, 'a')
    return {'side_cols': list(T.SIDE_COLS['nt_PRV']), 'cnn_ch': cfg['cnn_ch'], 'cnn_k': cfg['cnn_k'],
            'cnn_dil': list(cfg['cnn_dil']), 'head_hidden': list(cfg['head_hidden']), 'side': cfg['side'],
            'price': cfg['price'], 'side_names': [T.SIDE_NAMES[c] for c in T.SIDE_COLS['nt_PRV']]}, None


def gate_d():
    rows, ok = {}, True
    paths = [os.path.join(ROOT, f'artifacts/weights/prod_{MODEL}_L64_{s}') for s in 'abc']
    paths += [os.path.join(ROOT, f'artifacts/calib/pool_{st}_{MODEL}_L64_{s}_refit') for st in ('C', 'ALL') for s in 'ab']
    paths += [os.path.join(ROOT, f'artifacts/calib/pool_C_{MODEL}_L64_{s}') for s in 'ab']
    want = {'blocks.0.0.weight': (48, 4, 3), 'blocks.1.0.weight': (48, 48, 3), 'blocks.2.0.weight': (48, 48, 3),
            'blocks.3.0.weight': (48, 48, 3), 'head.0.weight': (96, 161), 'head.0.bias': (96,), 'head.3.weight': (1, 96),
            'head.3.bias': (1,), 'side_mu': (17,), 'side_sd': (17,), 'lvl_mu': (1,), 'lvl_sd': (1,), 's_sd': ()}
    for base in paths:
        r = load_json(base + '.json')
        cfg = r['cfg']
        npz = sorted(glob.glob(base + '_slim.npz')) or sorted(glob.glob(base + '_part*.npz'))
        sd = {}
        for p in npz:
            with np.load(p) as d:
                sd.update({k[3:]: d[k] for k in d.files if k.startswith('sd_')})
        shapes = {k: tuple(v.shape) for k, v in sd.items()}
        n_par = int(sum(sd[k].size for k in want if '.' in k))
        lvl_mu = float(np.asarray(sd['lvl_mu']).ravel()[0])
        good = (cfg['cnn_ch'] == 48 and cfg['cnn_k'] == 3 and list(cfg['cnn_dil']) == [1, 2, 4, 8]
                and list(cfg['head_hidden']) == [96] and cfg['side'] == 'nt_PRV' and cfg['price'] == 'multi_g'
                and cfg['cnn_valid'] and not cfg['cnn_bias'] and not cfg['cnn_bn'] and cfg['cnn_pool'] == 'mml'
                and set(shapes) == set(want) and all(shapes[k] == v for k, v in want.items())
                and n_par == N_PARAMS and lvl_mu == 0.0)
        rows[os.path.basename(base)] = {'n_params': n_par, 'lvl_mu': lvl_mu, 'shapes_ok': all(
            shapes.get(k) == v for k, v in want.items()), 'extra_keys': sorted(set(shapes) - set(want)), 'ok': bool(good)}
        ok &= good
    research, why = research_side_cols()
    out = {'models': rows, 'research': research, 'research_skipped': why}
    if research is not None:
        out['side_cols_match'] = tuple(research['side_cols']) == SIDE_EXPECTED
        ok &= out['side_cols_match'] and research['cnn_ch'] == 48 and research['head_hidden'] == [96]
    out['ok'] = bool(ok and research is not None)
    return out


def gate_e():
    prod = load_json(os.path.join(ROOT, f'artifacts/weights/prod_{MODEL}_L64_a.json'))
    out, ok = {}, True
    for s in 'ab':
        r = load_json(os.path.join(ROOT, f'artifacts/calib/pool_ALL_{MODEL}_L64_{s}_refit.json'))
        same = sorted(r['train_assets']) == sorted(prod['train_assets']) and len(r['train_assets']) == 31
        out[s] = {'train_assets': len(r['train_assets']), 'same_as_prod': same, 'epochs': r['cfg']['epochs'],
                  'epochs_run': r['epochs_run'], 't_cut': r['t_cut']}
        ok &= same and r['cfg']['epochs'] == prod['cfg']['epochs'] == r['epochs_run']
    log = next(p for p in sorted(glob.glob(os.path.join(ROOT, 'artifacts/logs/nt/smc_chain__*.log')))[::-1])
    with open(log, encoding='utf-8') as fh:
        txt = fh.read()
    sec = {}
    for nm in ('BTC', 'панель', 'B + C без панели', 'H (у ALL — в обучении)'):
        m = re.search(rf"ALL − C, Δ VAL — {re.escape(nm)}: ([+-][\d.]+) ± ([\d.]+)", txt)
        if m:
            sec[nm] = [float(m.group(1)), float(m.group(2))]
    out['section_5z'] = sec
    out['report_2se'] = [k for k in ('BTC', 'панель') if k in sec and abs(sec[k][0]) > 2 * sec[k][1]]
    out['ok'] = bool(ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs/checks/nt3/gates.json'))
    a = ap.parse_args()
    man = load_json(os.path.join(ROOT, 'artifacts/pack/nt_manifest.json'))
    res = {'model': MODEL, 'manifest_created': man['created']}
    for key, fn in (('a', lambda: gate_a(man)), ('b', gate_b), ('c', gate_c), ('d', gate_d), ('e', gate_e)):
        res[key] = fn()
        print(f"gate ({key}):", 'OK' if res[key]['ok'] else 'FAIL')
    res['all_ok'] = all(res[k]['ok'] for k in 'abcde')
    print(json.dumps(res, indent=1, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
