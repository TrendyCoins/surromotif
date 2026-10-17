"""Session 4 gates (docs/rules.md, 'Ворота сессии 4') for the PB_s64h24_d1 artifacts — before any PB calculation.

    (а) sha256 and size of every file of artifacts/pack/pb_manifest.json (checks_ok = true); the *_partK.npz of each run
        assemble without repeated keys
    (б) the latest artifacts/logs/pb/pool_summary_*.json: pb_final, recipe, pb_flag — cross-checked with the chain log of
        that summary (sections (3), (5и))
    (в) prod a / b / c: one data end, one overlay (data_fresh), epochs = rule v4.9 (Python round, as research
        pool_epochs); every 'refit' (C and ALL) the same; data end = prod NT3 and F3_g (floors of R6)
    (г) research RUNS['PB_s64h24_d1'] = RUNS['PB_s64h24'] + PB_D1_OVER; architecture in json / npz (64 channels, kernel 3,
        dilations 1-2-4-8, cnn_sep, head (24,), side 'nt_PRV' — research SIDE_COLS, 17 side_mu / side_sd, lvl_mu = 0),
        weight shapes, 18 697 parameters (= 'pine_w') and 32 304 multiplications ('pine'); the npz loads into research
        CNNBase(sep=True) with strict=True
    (д) 'refit' on ALL: train_assets = prod set (31), same epochs
    (е) nocal json: model, ver, src, seeds, winner / pine.ruler; rules = research NOCAL, edges = NC_EDGES, knots, zk =
        ndtri(knots); prod a / b / c = prod json; 37 strictly increasing quantiles in every ruler

    python scripts/pb_gates.py [--out docs/checks/pb/gates.json]

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
MODEL = 'PB_s64h24_d1'
BASE = 'PB_s64h24'
D1_OVER = {'cnn_cdrop': 0.1, 'dropout': 0.1}
REF_T_CUT = '2026-09-26 09:15:00'               # prod F3_g and NT3_PRV_ch48
N_PARAMS, N_MULT = 18697, 32304
SIDE_EXPECTED = (28, 29, 37, 38, 39, 40, 41, 42, 43, 44, 77, 78, 79, 86, 87, 88, 89)
NOCAL_JSON = 'artifacts/calib/nocal_PB_s64h24_d1_20261017_140953.json'
CH, HID, N_SIDE = 64, 24, 17
WANT = {'blocks.0.0.weight': (CH, 4, 3),
        **{f'blocks.{i}.0.weight': (CH, 1, 3) for i in (1, 2, 3)},
        **{f'blocks.{i}.1.weight': (CH, CH, 1) for i in (1, 2, 3)},
        'head.0.weight': (HID, 3 * CH + N_SIDE), 'head.0.bias': (HID,), 'head.3.weight': (1, HID), 'head.3.bias': (1,),
        'side_mu': (N_SIDE,), 'side_sd': (N_SIDE,), 'lvl_mu': (1,), 'lvl_sd': (1,), 's_sd': ()}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def folder(p):
    return None if not p else os.path.basename(str(p).rstrip('/'))


def load_json(path):
    with open(os.path.join(ROOT, path), encoding='utf-8') as fh:
        return json.load(fh)


def research_module():
    os.environ.setdefault('SMC_MODE', 'none')
    os.environ.setdefault('SMC_PROJECT_PATH', tempfile.mkdtemp() + '/')
    sys.path.insert(0, os.path.join(ROOT, 'research'))
    import sumo_class_train as T
    return T


def run_paths():
    """Bases (without extension) of the 7 runs with weights and of the 2 'stop' runs (json only)."""
    w = [f'artifacts/weights/prod_{MODEL}_L64_{s}' for s in 'abc']
    w += [f'artifacts/calib/pool_{st}_{MODEL}_L64_{s}_refit' for st in ('C', 'ALL') for s in 'ab']
    return w, [f'artifacts/calib/pool_C_{MODEL}_L64_{s}' for s in 'ab']


def npz_files(base):
    return sorted(glob.glob(os.path.join(ROOT, base + '_slim.npz'))) or sorted(glob.glob(os.path.join(ROOT, base + '_part*.npz')))


def gate_a(man):
    bad = []
    for f in man['files']:
        p = os.path.join(ROOT, f['path'])
        if not os.path.isfile(p):
            bad.append({'path': f['path'], 'problem': 'missing'})
        elif os.path.getsize(p) != f['bytes'] or sha256(p) != f['sha256']:
            bad.append({'path': f['path'], 'problem': 'size or sha256'})
    runs = {}
    for base in run_paths()[0][3:]:
        ps = npz_files(base)
        seen, dup = set(), []
        for p in ps:
            with np.load(p) as d:
                for k in d.files:
                    (dup.append(k) if k in seen else seen.add(k))
        runs[os.path.basename(base)] = {'parts': len(ps), 'keys': len(seen), 'n_repeated': len(dup), 'repeated': dup[:10]}
    ok = (man.get('checks_ok') is True and not bad and len(runs) == 4
          and all(r['parts'] > 1 and r['n_repeated'] == 0 for r in runs.values()))
    return {'files': len(man['files']), 'checks_ok': man.get('checks_ok'), 'bad_files': bad, 'runs': runs, 'ok': bool(ok)}


def summary_block(txt, stamp):
    """Lines of the chain log printed by the pool summary that saved pool_summary_<stamp>.json."""
    lines = txt.splitlines()
    end = next(i for i, ln in enumerate(lines) if f'pool_summary_{stamp}.json' in ln and '💾' in ln)
    start = max([i for i, ln in enumerate(lines[:end]) if '💾' in ln and 'pool_summary_' in ln] + [0])
    return lines[start:end + 1]


def gate_b():
    summ = sorted(glob.glob(os.path.join(ROOT, 'artifacts/logs/pb/pool_summary_*.json')))[-1]
    with open(summ, encoding='utf-8') as fh:
        s = json.load(fh)
    stamp = os.path.basename(summ)[len('pool_summary_'):-len('.json')]
    out = {'summary': os.path.basename(summ), 'pb_final': s.get('pb_final'), 'pb_base': s.get('pb_base'),
           'recipe_ok': s.get('recipe_ok', {}).get(MODEL), 'pb_flag': s.get('pb_flag'), 'pb_steps': s.get('pb_steps'),
           'epochs_summary': s.get('prod', {}).get('epochs', {}).get(MODEL),
           'prod_models_pine_in_summary': s.get('prod_models', {}).get('pine')}
    log = None
    for p in sorted(glob.glob(os.path.join(ROOT, 'artifacts/logs/pb/smc_chain__*.log'))):
        with open(p, encoding='utf-8') as fh:
            txt = fh.read()
        if f'pool_summary_{stamp}.json' in txt:
            log = (os.path.basename(p), summary_block(txt, stamp))
    if log:
        name, blk = log
        out['chain_log'] = name
        find = lambda rx: next((ln.strip() for ln in blk if re.search(rx, ln)), None)
        lines = {'recipe': find(rf"{MODEL} 'refit' \(\d+ эпох\) против 'stop'"),
                 'd1_step': find(rf"[✅❌] {MODEL} − {BASE}: панель"),
                 'final': find(r"🏆 итог этапа 2"),
                 'vs_nt3': find(rf"· {MODEL} − NT3_PRV_ch48 \(цена упрощения\)"),
                 'vs_f3g': find(rf"· {MODEL} − F3_g"),
                 'epochs': find(rf"эпох фиксированного расписания {MODEL}")}
        out['log_lines'] = lines
        num = lambda rx, ln: [float(x) for x in re.search(rx, ln or '').groups()] if re.search(rx, ln or '') else None
        out['recipe_delta_val'] = num(r"панель Δ VAL ([+-][\d.]+) ± ([\d.]+)", lines['recipe'])
        out['vs_nt3_panel_val'] = num(r"панель Δ TUNE [+-][\d.]+ ± [\d.]+ \(сидов \d\), Δ VAL ([+-][\d.]+) ± ([\d.]+)",
                                      lines['vs_nt3'])
        out['vs_nt3_btc_val'] = num(r"BTC Δ VAL ([+-][\d.]+) ± ([\d.]+)", lines['vs_nt3'])
        out['vs_nt3_h_val'] = num(r"H Δ VAL ([+-][\d.]+) ± ([\d.]+)", lines['vs_nt3'])
        out['vs_f3g_panel_val'] = num(r"панель Δ TUNE [+-][\d.]+ ± [\d.]+ \(сидов \d\), Δ VAL ([+-][\d.]+) ± ([\d.]+)",
                                      lines['vs_f3g'])
        out['epochs_log'] = num(r": (\d+) \(правило v4\.9: лучшие эпохи 'stop' на C (\d+) / (\d+)\)", lines['epochs'])
        rd, nv = out['recipe_delta_val'], out['vs_nt3_panel_val']
        out['log_consistent'] = bool(
            rd and rd[0] >= -rd[1] and nv and nv[0] > -2 * nv[1] and lines['final'] and MODEL in lines['final']
            and lines['d1_step'] and lines['d1_step'].startswith('✅') and out['epochs_log']
            and int(out['epochs_log'][0]) == out['epochs_summary'])
    out['ok'] = bool(out['pb_final'] == MODEL and out['recipe_ok'] and out['pb_flag'] is False and out.get('log_consistent'))
    return out


def gate_c():
    w, stops = run_paths()
    prod = {s: load_json(w[i] + '.json') for i, s in enumerate('abc')}
    refit = {os.path.basename(b): load_json(b + '.json') for b in w[3:]}
    stop = {s: load_json(stops[i] + '.json') for i, s in enumerate('ab')}
    best = [int(stop[s]['best_ep']) + 1 for s in 'ab']
    rule = int(round(float(np.mean(best))))                 # research pool_epochs: Python round (half to even)
    ref = {f'{m}_{s}': load_json(f'artifacts/weights/prod_{m}_L64_{s}.json')['t_cut']
           for m in ('F3_g', 'NT3_PRV_ch48') for s in 'abc'}
    row = lambda r: {'t_cut': r['t_cut'], 'overlay': folder(r.get('data_overlay')), 'epochs_cfg': r['cfg']['epochs'],
                     'epochs_run': r['epochs_run'], 'set': r['set'], 'mode': r['mode']}
    out = {'prod': {s: row(r) for s, r in prod.items()}, 'refit': {k: row(r) for k, r in refit.items()},
           'stop_best_ep_plus1': best, 'epochs_rule_v4_9': rule, 'epochs_rule_half_up': int(np.floor(np.mean(best) + 0.5)),
           'prod_ref_t_cut': ref}
    pt = {r['t_cut'] for r in prod.values()}
    ep = {r['epochs_run'] for r in list(prod.values()) + list(refit.values())} | {
        r['cfg']['epochs'] for r in list(prod.values()) + list(refit.values())}
    out['one_data_end'] = len(pt) == 1
    out['one_overlay'] = {folder(r.get('data_overlay')) for r in prod.values()} == {'data_fresh'}
    out['epochs_all_equal_rule'] = ep == {rule} == {22}
    out['same_end_as_F3_g_NT3'] = pt == set(ref.values()) == {REF_T_CUT}
    out['ok'] = bool(out['one_data_end'] and out['one_overlay'] and out['epochs_all_equal_rule'] and out['same_end_as_F3_g_NT3'])
    return out


def gate_d(T):
    import torch
    w, stops = run_paths()
    rows, ok = {}, True
    runs_ok = T.RUNS.get(MODEL) == {**T.RUNS[BASE], **D1_OVER} and T.PB_D1_OVER == D1_OVER
    cfg_r = T.make_cfg(MODEL, {'L': 64}, 'a')
    research = {'runs_equal_base_plus_d1': bool(runs_ok), 'side_cols': list(T.SIDE_COLS['nt_PRV']),
                'side_names': [T.SIDE_NAMES[c] for c in T.SIDE_COLS['nt_PRV']],
                **{k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg_r.items()
                   if k in ('cnn_ch', 'cnn_k', 'cnn_dil', 'cnn_sep', 'head_hidden', 'side', 'price', 'cnn_valid',
                            'cnn_bias', 'cnn_bn', 'cnn_pool', 'cnn_in', 'dropout', 'cnn_cdrop')},
                'pine_weights_cfg': T.pine_weights_cfg(cfg_r), 'pine_cost_cfg': T.pine_cost_cfg(cfg_r)}
    net = T.CNNBase(cfg_r['L'], cfg_r['cnn_ch'], cfg_r['cnn_k'], cfg_r['cnn_dil'], cfg_r['head_hidden'], cfg_r['dropout'],
                    1.0, cfg_r['cnn_bn'], cfg_r['cnn_res'], cfg_r['cnn_cdrop'], cfg_r['cnn_pool'], cfg_r['cnn_in'],
                    cfg_r['cnn_sdrop'], cfg_r['cnn_valid'], cfg_r['cnn_bias'], cfg_r['cnn_ma'], T.SIDE_COLS[cfg_r['side']], 4,
                    None, 'rw', None, T.N_LEVEL[cfg_r['price']], False, cfg_r['cnn_sep'])
    research['pine_weights_model'] = int(T.pine_weights(net))
    research['pine_cost_model'] = int(net.pine_cost())
    ok &= (runs_ok and tuple(research['side_cols']) == SIDE_EXPECTED and cfg_r['cnn_ch'] == CH and cfg_r['cnn_sep']
           and list(cfg_r['head_hidden']) == [HID] and research['pine_weights_cfg'] == research['pine_weights_model'] == N_PARAMS
           and research['pine_cost_cfg'] == research['pine_cost_model'] == N_MULT)

    def cfg_ok(cfg):
        return (cfg['cnn_ch'] == CH and cfg['cnn_k'] == 3 and list(cfg['cnn_dil']) == [1, 2, 4, 8] and cfg['cnn_sep'] is True
                and list(cfg['head_hidden']) == [HID] and cfg['side'] == 'nt_PRV' and cfg['price'] == 'multi_g'
                and cfg['cnn_valid'] and not cfg['cnn_bias'] and not cfg['cnn_bn'] and not cfg['cnn_res']
                and cfg['cnn_pool'] == 'mml' and cfg['cnn_in'] == 'steps' and cfg.get('name') == MODEL)
    for base in w:
        r = load_json(base + '.json')
        sd = {}
        for p in npz_files(base):
            with np.load(p) as d:
                sd.update({k[3:]: d[k] for k in d.files if k.startswith('sd_')})
        shapes = {k: tuple(v.shape) for k, v in sd.items()}
        n_par = int(sum(sd[k].size for k in WANT if '.' in k))
        lvl_mu = float(np.asarray(sd['lvl_mu']).ravel()[0])
        try:
            net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in sd.items()}, strict=True)
            strict = True
        except RuntimeError as e:
            strict = str(e)[:300]
        good = (cfg_ok(r['cfg']) and set(shapes) == set(WANT) and all(shapes[k] == v for k, v in WANT.items())
                and n_par == N_PARAMS == r.get('pine_w') and r.get('pine') == N_MULT and lvl_mu == 0.0 and strict is True)
        rows[os.path.basename(base)] = {'n_params': n_par, 'pine_w': r.get('pine_w'), 'pine': r.get('pine'), 'lvl_mu': lvl_mu,
                                        'cfg_ok': cfg_ok(r['cfg']), 'shapes_ok': all(shapes.get(k) == v for k, v in WANT.items()),
                                        'extra_keys': sorted(set(shapes) - set(WANT)), 'strict_load': strict, 'ok': bool(good)}
        ok &= good
    for base in stops:                                   # 'stop' on C: json only
        r = load_json(base + '.json')
        good = cfg_ok(r['cfg']) and r.get('pine_w') == N_PARAMS and r.get('pine') == N_MULT
        rows[os.path.basename(base)] = {'pine_w': r.get('pine_w'), 'pine': r.get('pine'), 'cfg_ok': cfg_ok(r['cfg']),
                                        'json_only': True, 'ok': bool(good)}
        ok &= good
    return {'research': research, 'models': rows, 'ok': bool(ok)}


def gate_e():
    prod = load_json(f'artifacts/weights/prod_{MODEL}_L64_a.json')
    out, ok = {}, True
    for s in 'ab':
        r = load_json(f'artifacts/calib/pool_ALL_{MODEL}_L64_{s}_refit.json')
        same = sorted(r['train_assets']) == sorted(prod['train_assets']) and len(r['train_assets']) == 31
        out[s] = {'train_assets': len(r['train_assets']), 'same_as_prod': same, 'epochs': r['cfg']['epochs'],
                  'epochs_run': r['epochs_run'], 't_cut': r['t_cut'], 'set': r['set'], 'mode': r['mode']}
        ok &= same and r['cfg']['epochs'] == prod['cfg']['epochs'] == r['epochs_run'] and r['set'] == 'ALL'
    out['ok'] = bool(ok)
    return out


def gate_f(T):
    """(е) the nocal json."""
    from scipy.special import ndtri
    nc = load_json(NOCAL_JSON)
    N_ = T.NOCAL
    rules_want = {k: N_[k] for k in nc['rules']}
    norm = lambda v: json.loads(json.dumps(v, default=lambda x: list(x) if isinstance(x, tuple) else float(x)))
    rules_eq = {k: norm(nc['rules'][k]) == norm(rules_want[k]) for k in nc['rules']}
    rule_keys = ('null', 'cond_cols', 'half_cut', 'span', 's_lo', 'warm', 'tol', 'min_n', 'ceil_pool', 'ceil_asset',
                 'pass_share', 'order', 'span_ref', 'short', 'unseen')
    knots = np.asarray(nc['knots'], dtype=np.float64)
    zk = np.asarray(nc['zk'], dtype=np.float64)
    out = {'model': nc['model'], 'ver': nc['ver'], 'src': nc['src'], 'seeds': nc['seeds'], 'winner': nc['winner'],
           'pine_ruler': nc['pine']['ruler'], 'when': nc['when'],
           'rules_equal': rules_eq, 'rules_keys_complete': sorted(nc['rules']) == sorted(rule_keys),
           'edges_equal': nc['edges'] == list(T.NC_EDGES[:-1]),
           'knots_equal': tuple(nc['knots']) == tuple(N_['knots']) and nc['pine']['knots'] == nc['knots'],
           'zk_max_abs_diff': float(np.max(np.abs(zk - ndtri(knots)))),
           'pine_zk_equal': nc['pine']['zk'] == nc['zk'],
           'pine_span_s_lo_warm': [nc['pine']['span'], nc['pine']['s_lo'], nc['pine']['warm']],
           'nocal_ver_research': T.NOCAL_VER, 'prod': {}}
    ok = (nc['model'] == MODEL and nc['ver'] == 2 == T.NOCAL_VER and nc['src'] == ['ALL', 'refit'] and nc['seeds'] == ['a', 'b']
          and nc['winner'] == 'glob_e' == nc['pine']['ruler'] and all(rules_eq.values()) and out['rules_keys_complete']
          and out['edges_equal'] and out['knots_equal'] and out['zk_max_abs_diff'] <= 1e-15 and out['pine_zk_equal']
          and out['pine_span_s_lo_warm'] == [N_['span'], N_['s_lo'], N_['warm']] and len(knots) == 37)
    for s in 'abc':
        c, r = nc['prod'][s], load_json(f'artifacts/weights/prod_{MODEL}_L64_{s}.json')
        q = np.asarray(c['glob_e']['q'], dtype=np.float64)
        row = {'tag_ok': c['tag'] == r['tag'], 't_cut_ok': c['t_cut'] == r['t_cut'],
               'overlay_ok': folder(c['data_overlay']) == folder(r.get('data_overlay')), 'pine_w': c['pine_w'],
               'pine': c['pine'], 'n_q': int(len(q)), 'q_strictly_increasing': bool(np.all(np.diff(q) > 0)),
               'q_first_last': [float(q[0]), float(q[-1])], 'n_null': c['n_null']}
        row['ok'] = bool(row['tag_ok'] and row['t_cut_ok'] and row['overlay_ok'] and c['pine_w'] == N_PARAMS
                         and c['pine'] == N_MULT and row['n_q'] == 37 and row['q_strictly_increasing'])
        out['prod'][s] = row
        ok &= row['ok']
    out['ok'] = bool(ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs/checks/pb/gates.json'))
    a = ap.parse_args()
    man = load_json('artifacts/pack/pb_manifest.json')
    T = research_module()
    res = {'model': MODEL, 'manifest_created': man['created']}
    for key, fn in (('а', lambda: gate_a(man)), ('б', gate_b), ('в', gate_c), ('г', lambda: gate_d(T)), ('д', gate_e),
                    ('е', lambda: gate_f(T))):
        res[key] = fn()
        print(f"gate ({key}):", 'OK' if res[key]['ok'] else 'FAIL')
    res['all_ok'] = all(res[k]['ok'] for k in 'абвгде')
    print(json.dumps(res, indent=1, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
