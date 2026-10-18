"""Session 5 gates (docs/rules.md, 'Ворота сессии 5') — before any calculation of the new line M.

    (а) files and code: the three research files, the Live and Ceil json, logs; the research files import without side
        effects; CEIL_VER = 1, LIVE_VER = '1.1'; the training code is v6.3 (NOCAL s_lo 0.75, 37 knots = nocal json);
        the Ceil stand locally (CEIL_MODE 'stand', paths in a temporary folder): 18 of 18
    (б) the Ceil json: model, ver, rules = CEIL field by field, the checks, (П), the choice, export, export.ruler, format
    (в) the Live json: model, verdict, rules = LIVE, numbers of the formula A = formula_a of the Ceil json bit for bit
    (г) export.live of the Ceil json = Live variants_a() '4032+144'; options_b -> B3 shows from 4096
    (д) after the Colab export (--export): sha256 by the manifest, meta of each tag, weeks = npz 'refit' ALL, BTC VAL
        windows where the floor acts — 834 for each tag

    python scripts/pb5_gates.py [--out docs/checks/pb5/gates.json] [--no-stand] [--export]

Paths of the training machine (Drive) are never written to the output.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mri import research5 as R5  # noqa: E402

LIVE_JSON = os.path.join(R5.CALIB, R5.LIVE_JSON)
CEIL_JSON = os.path.join(R5.CALIB, R5.CEIL_JSON)
NOCAL_JSON = sorted(glob.glob(os.path.join(R5.CALIB, f'nocal_{R5.MODEL}_*.json')))[-1]
PROD_JSON = os.path.join(ROOT, f'artifacts/weights/prod_{R5.MODEL}_L64_a.json')
REF_T_CUT = '2026-09-26 09:15:00'
FORMULA = '4032+144'
FA_FIELDS = ('bce', 'gp_col', 'ceil', 'share', 'cover3')
N_AFFECTED_BTC = 834
EXPORT_MANIFEST = os.path.join(R5.CALIB, 'ceil_f02_manifest.json')


def load(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class Gate:
    def __init__(self):
        self.rows = []

    def __call__(self, gate, name, ok, note=''):
        self.rows.append({'gate': gate, 'check': name, 'ok': bool(ok), 'note': note})
        print(f"   {'✅' if ok else '❌'} ({gate}) {name}{(' — ' + note) if note else ''}")
        return bool(ok)


def jsonable(x):
    """As research _jsonable (Bell): tuples -> lists, inf / nan -> None, numpy scalars -> Python."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(x) else float(x)
    return x


def import_check(rd):
    """Import each research file in a fresh interpreter (paths into rd): nothing printed, no file created, main() not run."""
    code = ("import importlib.util, sys\n"
            "for name in ('sumo_class_bell', 'sumo_class_live', 'sumo_class_ceil'):\n"
            "    spec = importlib.util.spec_from_file_location(name, sys.argv[1] + '/' + name + '.py')\n"
            "    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)\n"
            "import sumo_class_live as L, sumo_class_ceil as C, sumo_class_bell as B\n"
            "print(repr(L.LIVE_VER), repr(C.CEIL_VER), repr(B.BELL_VER), L.MODE, C.MODE, B.MODE)\n")
    before = sorted(os.listdir(rd))
    cwd = tempfile.mkdtemp(dir=rd)
    p = subprocess.run([sys.executable, '-c', code, R5.RESEARCH], cwd=cwd, env={**os.environ, **R5.env(rd, 'none')},
                       capture_output=True, text=True, timeout=600)
    created = sorted(set(os.listdir(rd)) - set(before) - {os.path.basename(cwd)}) + os.listdir(cwd)
    return p, created


def stand(rd):
    """Ceil stand in a subprocess: CEIL_MODE 'stand', every path in rd; returns (passed, total, seconds, tail of log)."""
    code = ("import importlib.util, sys\n"
            "spec = importlib.util.spec_from_file_location('sumo_class_ceil', sys.argv[1])\n"
            "m = importlib.util.module_from_spec(spec); sys.modules['sumo_class_ceil'] = m; spec.loader.exec_module(m)\n"
            "m.main()\n")
    os.makedirs(rd + 'project/', exist_ok=True)
    import time
    t0 = time.time()
    p = subprocess.run([sys.executable, '-c', code, os.path.join(R5.RESEARCH, 'sumo_class_ceil.py')], cwd=rd,
                       env={**os.environ, **R5.env(rd, 'stand')}, capture_output=True, text=True, timeout=7200)
    out = p.stdout + p.stderr
    m = re.findall(r'СТЕНД: (\d+) из (\d+) проверок', out)
    n_ok, n = (int(m[-1][0]), int(m[-1][1])) if m else (0, -1)
    return n_ok, n, time.time() - t0, out, p.returncode


def gate_a(G, rd, do_stand):
    files = ['research/sumo_class_bell.py', 'research/sumo_class_live.py', 'research/sumo_class_ceil.py',
             f'artifacts/calib/{R5.LIVE_JSON}', f'artifacts/calib/{R5.CEIL_JSON}']
    miss = [f for f in files if not os.path.isfile(os.path.join(ROOT, f))]
    logs = {k: sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, f'artifacts/logs/{k}/*.log')))
            for k in ('live', 'ceil')}
    G('а', 'на месте три research-файла и json Live / Ceil', not miss, ', '.join(miss) or f'{len(files)} файлов')
    G('а', 'логи Live и Ceil (стенд и run)', all(any('_stand_' in x for x in v) and any('_run_' in x for x in v)
                                                 for v in logs.values()), '; '.join(f'{k}: {", ".join(v)}'
                                                                                 for k, v in logs.items()))
    p, created = import_check(rd)
    vers = p.stdout.strip().split()
    G('а', 'research-файлы импортируются без побочных эффектов (ничего не напечатано, файлов не создано, main не запущен)',
      p.returncode == 0 and len(vers) == 6 and not created and not p.stderr.strip(),
      f'код {p.returncode}; печать импорта: {p.stdout.strip()!r}; новые файлы: {created}; stderr: {p.stderr.strip()[-300:]!r}')
    live_ver, ceil_ver = (vers[0], vers[1]) if len(vers) == 6 else ('?', '?')
    G('а', "CEIL_VER = 1", ceil_ver == '1', f'CEIL_VER = {ceil_ver}')
    G('а', "LIVE_VER = '1.1'", live_ver == "'1.1'", f'LIVE_VER = {live_ver} (json Live — ver {load(LIVE_JSON)["ver"]})')
    T, BB, LV, CE = R5.chain()
    nc = load(NOCAL_JSON)
    G('а', 'код обучения v6.3: NOCAL s_lo 0.75, 37 узлов = nocal json',
      T.NOCAL['s_lo'] == 0.75 and len(T.NOCAL['knots']) == 37
      and np.array_equal(np.asarray(T.NOCAL['knots'], dtype=np.float64), np.asarray(nc['knots'], dtype=np.float64))
      and 'v6.3' in open(os.path.join(R5.RESEARCH, 'sumo_class_train.py'), encoding='utf-8').read(),
      f"s_lo {T.NOCAL['s_lo']}, узлов {len(T.NOCAL['knots'])}")
    res = {'live_ver': live_ver, 'ceil_ver': ceil_ver, 'logs': logs}
    if do_stand:
        n_ok, n, sec, out, code = stand(tempfile.mkdtemp(prefix='ceil_stand_') + '/')
        bad = [ln.strip() for ln in out.splitlines() if ln.strip().startswith('❌')]
        G('а', 'стенд Ceil локально: 18 из 18', n_ok == 18 and n == 18 and code == 0,
          f'{n_ok} из {n}, {sec / 60:.1f} мин' + (f'; не прошли: {bad}' if bad else ''))
        res['stand'] = {'passed': n_ok, 'total': n, 'minutes': round(sec / 60, 2),
                        'checks': [ln.strip() for ln in out.splitlines() if ln.strip()[:1] in ('✅', '❌')]}
    return res, (T, BB, LV, CE)


def gate_b(G, CE, cj, nc, prod):
    G('б', "model 'PB_s64h24_d1', ver 1", cj['model'] == R5.MODEL and cj['ver'] == 1, f"{cj['model']}, ver {cj['ver']}")
    rules_mod = jsonable(CE.CEIL)
    diff = sorted(k for k in set(rules_mod) | set(cj['rules']) if rules_mod.get(k, '∅') != cj['rules'].get(k, '∅'))
    G('б', 'rules = CEIL research поле в поле', not diff, f"расходятся: {diff}" if diff else f"{len(rules_mod)} полей")
    r = cj['rules']
    want = {'floor': 0.2, 'formula': FORMULA, 'candidates': [10, 8, 5, 3], 'refs': [13, 21], 'fine_max': 5.0,
            'share_over': 0.1, 'share_nodata': 0.1, 'main': 'stop_C', 'confirm': 'refit_ALL', 'prod_seeds': ['a']}
    G('б', 'rules: floor 0.2, formula, candidates, refs, fine_max, share_*, main, confirm, prod_seeds',
      all(r.get(k) == v for k, v in want.items()), ', '.join(f'{k} {r.get(k)}' for k in want))
    G('б', 'формат: rules.edges кончается null (в CEIL — math.inf)',
      r['edges'][-1] is None and math.isinf(CE.CEIL['edges'][-1]) and r['edges'][:-1] == list(CE.CEIL['edges'][:-1]),
      f"edges {r['edges']}")
    by_c_keys = {k: sorted(v['by_c']) for k, v in cj['ceiling'].items()}
    G('б', "формат: ключи by_c — строки ('10')", all(all(isinstance(x, str) for x in v) and '10' in v
                                                     for v in by_c_keys.values()), str(by_c_keys))
    pf = cj['preflight']
    G('б', 'check_nocal.ok и check_live.ok', cj['check_nocal']['ok'] is True and cj['check_live']['ok'] is True,
      f"nocal {cj['check_nocal']['ok']}, live {cj['check_live']['ok']}; check_live — поля {sorted(cj['check_live'])}")
    G('б', 'preflight.ok (same_windows, same_unaffected = 0, dev_npz <= 0.02)',
      pf['ok'] is True and pf['same_windows'] is True and pf['same_unaffected'] == 0 and pf['dev_npz'] <= 0.02,
      f"same_windows {pf['same_windows']}, same_unaffected {pf['same_unaffected']}, dev_npz {pf['dev_npz']:.2e}, "
      f"n {pf['n']}, n_affected {pf['n_affected']}")
    gp = cj['gates_p']
    G('б', 'p_ok; gates_p — all у обоих «после»', cj['p_ok'] is True and gp['stop_C, после']['all'] is True
      and gp['refit_ALL, после']['all'] is True, '; '.join(f"{k}: {v}" for k, v in gp.items()))
    G('б', 'choice = 10, choice_old_floors = 10', cj['choice'] == 10 and cj['choice_old_floors'] == 10,
      f"{cj['choice']} / {cj['choice_old_floors']}")
    ex = cj['export']
    G('б', 'export: floor 0.2, ceiling 10, live {L 4032, n0 144, lag 64, show_from_window 4096}',
      ex['floor'] == 0.2 and ex['ceiling'] == 10 and ex['live'] == {'L': 4032, 'n0': 144, 'lag': 64, 'show_from_window': 4096},
      f"floor {ex['floor']}, ceiling {ex['ceiling']}, live {ex['live']}")
    ru = ex['ruler']
    G('б', "export.ruler: seed 'a', tag, t_cut = json боевой prod a, same_tag / same_t_cut, n_null = nocal",
      ru['seed'] == 'a' and ru['tag'] == f'prod_{R5.MODEL}_L64_a' == prod['tag'] and ru['t_cut'] == prod['t_cut'] == REF_T_CUT
      and ru['same_tag'] is True and ru['same_t_cut'] is True and ru['n_null'] == nc['prod']['a']['n_null'] == 3179790,
      f"seed {ru['seed']}, tag {ru['tag']}, t_cut {ru['t_cut']} (prod json {prod['t_cut']}), same {ru['same_tag']} / "
      f"{ru['same_t_cut']}, n_null {ru['n_null']} (nocal {nc['prod']['a']['n_null']})")
    q = np.asarray(ru['glob_e']['q'], dtype=np.float64)
    dq = float(np.max(np.abs(q - np.asarray(nc['prod']['a']['glob_e']['q'], dtype=np.float64))))
    G('б', 'export.ruler q — 37 строго возрастающих; max |q − nocal q| = dq_nocal <= 0.05',
      len(q) == 37 and bool(np.all(np.diff(q) > 0)) and dq == ru['dq_nocal'] and dq <= 0.05,
      f"{len(q)} чисел, max |Δq| {dq:.6f} (dq_nocal {ru['dq_nocal']:.6f}), floor {ru['floor']}")
    G('б', 'json ruler.a = export.ruler', cj['ruler']['a'] == ru, '')
    return {'dq_nocal': dq, 'n_affected_btc': pf['n_affected'], 'preflight_n': pf['n']}


def gate_c(G, LV, lj, cj):
    vd = lj['verdict']
    G('в', "model; verdict.a_pick '4032+144', b_pick '4032+144·B3'",
      lj['model'] == R5.MODEL and vd['a_pick'] == FORMULA and vd['b_pick'] == f'{FORMULA}·B3',
      f"{lj['model']}, {vd['a_pick']}, {vd['b_pick']}")
    ca, cb = vd['confirm']['a'], vd['confirm']['b']
    G('в', 'подтверждение пройдено (A3 и B3)', ca['ok'] is True and cb['ok'] is True,
      f"a: ok {ca['ok']}, лучший {ca.get('best')}; b: ok {cb['ok']}, лучший {cb.get('best')}")
    rules_mod = jsonable(LV.LIVE)
    diff = sorted(k for k in set(rules_mod) | set(lj['rules']) if rules_mod.get(k, '∅') != lj['rules'].get(k, '∅'))
    r = lj['rules']
    G('в', 'rules = LIVE research: lag 64, 4032 в L, 144 в f2_n0, null_cover_max 0.05',
      not diff and r['lag'] == 64 and 4032 in r['L'] and 144 in r['f2_n0'] and r['null_cover_max'] == 0.05,
      f"расходятся: {diff}" if diff else f"{len(rules_mod)} полей; lag {r['lag']}, L {r['L']}, f2_n0 {r['f2_n0']}")
    rows = []
    ok = True
    for src, key in (('main', 'stop_C'), ('confirm', 'refit_ALL')):
        a = lj[src]['a']['variants'][FORMULA]
        b = cj['formula_a'][f'{key}, до (npz)']
        same = {f: a[f] == b[f] for f in FA_FIELDS}
        ok &= all(same.values())
        rows.append(f"{key}: " + ', '.join(f"{f} {a[f]}" + ('' if same[f] else f" ≠ {b[f]}") for f in FA_FIELDS))
    G('в', 'формула А json Live (bce, gp_col, ceil, share, cover3) = formula_a json Ceil «до (npz)» бит в бит', ok,
      ' | '.join(rows))
    return {'formula_a_live': {k: {f: lj[s]['a']['variants'][FORMULA][f] for f in FA_FIELDS}
                               for s, k in (('main', 'stop_C'), ('confirm', 'refit_ALL'))}}


def gate_d(G, LV, cj):
    VA = {v['key']: v for v in LV.variants_a()}
    v = VA[FORMULA]
    ex = cj['export']['live']
    G('г', "export.live = variants_a() '4032+144' (L 4032, lag 64, prior 'null', n0 144)",
      v['L'] == ex['L'] == 4032 and v['lag'] == ex['lag'] == 64 and v['prior'] == 'null' and v['n0'] == ex['n0'] == 144
      and v['kind'] == 'win', f"L {v['L']}, lag {v['lag']}, prior {v['prior']}, n0 {v['n0']}, kind {v['kind']}")
    b3 = [o for o in LV.options_b(v) if o['opt'] == 'B3'][0]
    G('г', 'options_b -> показ B3 = 4096 = show_from_window', b3['show'] == 4096 == ex['show_from_window'],
      f"B3 show {b3['show']}")


def gate_e(G, CE, cj):
    """(д): the Colab export of the Ceil cache (artifacts/calib/ceil_f02_*)."""
    if not os.path.isfile(EXPORT_MANIFEST):
        G('д', 'выгрузка на месте', False, f'нет {os.path.relpath(EXPORT_MANIFEST, ROOT)}')
        return None
    man = load(EXPORT_MANIFEST)
    bad = []
    for part in man['parts']:
        p = os.path.join(R5.CALIB, part['file'])
        if not os.path.isfile(p) or os.path.getsize(p) != part['size'] or sha256(p) != part['sha256']:
            bad.append(part['file'])
    G('д', 'sha256 и размер частей — по manifest', not bad, ', '.join(bad) or f"{len(man['parts'])} частей")
    from mri.ceil_export import load_export
    ex = load_export()
    tags = (f'pool_ALL_{R5.MODEL}_L64_a_refit', f'pool_ALL_{R5.MODEL}_L64_b_refit', f'prod_{R5.MODEL}_L64_a')
    meta_ok = all(man['meta'][t]['tag'] == t and man['meta'][t]['floor'] == 0.2 and man['meta'][t]['ver'] == 1 for t in tags)
    G('д', 'meta каждого tag: tag, floor 0.2, ver 1', meta_ok,
      '; '.join(f"{t}: floor {man['meta'][t]['floor']}, ver {man['meta'][t]['ver']}" for t in tags))
    npz = {s: R5_npz(f'pool_ALL_{R5.MODEL}_L64_{s}_refit') for s in 'ab'}
    assets = list(cj['floors_trained'])
    wk_bad = []
    for t in tags:
        for a in assets:
            wk = ex[t][a]['wk']
            for s in 'ab':
                if not np.array_equal(wk.astype(np.int64), npz[s][f'{a}__val__wk'].astype(np.int64)):
                    wk_bad.append(f'{t} {a} vs npz {s}')
    G('д', "недели wk каждого ассета 'refit' ALL a / b и prod a = недели npz 'refit' ALL", not wk_bad,
      ', '.join(wk_bad[:10]) or f'{len(tags)} tag × {len(assets)} ассетов × 2 npz')
    fb = cj['floors_trained']['BTCUSDT']
    n_aff = {t: int(CE.affected(ex[t]['BTCUSDT']['med'], ex[t]['BTCUSDT']['mabs'], fb, CE.CEIL['floor']).sum()) for t in tags}
    G('д', f'у BTC VAL окон, где пол срабатывает, — {N_AFFECTED_BTC} у каждого tag',
      all(v == N_AFFECTED_BTC for v in n_aff.values()), f'{n_aff} (обученный пол BTC {fb})')
    return {'n_affected_btc': n_aff, 'parts': len(man['parts'])}


def R5_npz(tag):
    arr = {}
    for p in sorted(glob.glob(os.path.join(R5.CALIB, f'{tag}_part*.npz'))):
        with np.load(p) as d:
            arr.update({k: d[k] for k in d.files if not k.startswith('sd_')})
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='docs/checks/pb5/gates.json')
    ap.add_argument('--no-stand', action='store_true')
    ap.add_argument('--export', action='store_true', help='gate (д): the Colab export of the Ceil cache')
    a = ap.parse_args()
    G = Gate()
    cj, lj, nc, prod = load(CEIL_JSON), load(LIVE_JSON), load(NOCAL_JSON), load(PROD_JSON)
    out = {'ceil_json': R5.CEIL_JSON, 'live_json': R5.LIVE_JSON, 'nocal_json': os.path.basename(NOCAL_JSON)}
    print('(а) файлы и код')
    out['a'], (T, BB, LV, CE) = gate_a(G, R5.run_dir(), not a.no_stand)
    print('(б) json Ceil')
    out['b'] = gate_b(G, CE, cj, nc, prod)
    print('(в) json Live')
    out['c'] = gate_c(G, LV, lj, cj)
    print('(г) живая часть экспорта')
    gate_d(G, LV, cj)
    if a.export:
        print('(д) выгрузка с полом 0.2')
        out['e'] = gate_e(G, CE, cj)
    out['checks'] = G.rows
    out['ok'] = {g: all(r['ok'] for r in G.rows if r['gate'] == g) for g in dict.fromkeys(r['gate'] for r in G.rows)}
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    with open(os.path.join(ROOT, a.out), 'w', encoding='utf-8') as fh:
        json.dump(jsonable(out), fh, ensure_ascii=False, indent=1)
    print('итог:', out['ok'], '->', a.out)


if __name__ == '__main__':
    main()
