"""Catalog all organic ONNX prototypes by their match-vs-LIVE status.

For each known prototype (file path or built from .py), runs it against the
current submission.zip's live ONNX on the visible+arc-gen pair set, using the
VALIDATOR's encoding (convert_to_numpy) and decode ((result > 0).astype).
Classifies each as:

  MATCH_LIVE      — proto produces same output as live byte-exact on all pairs.
                    Safe to ship if cheaper than live.
  ALGORITHM_DIVERGE — proto's outputs differ from live on >=1 pair. Silent-fail
                    risk on hidden; do NOT ship.
  PROTO_BROKEN    — proto fails to load or run.
  NO_LIVE         — corresponding task isn't in submission.zip.

Also reports proto_cost vs live_cost and predicted LB delta if MATCH_LIVE.

Usage:
    python tools/catalog_organic_prototypes.py [--proto-dir DIR] [--out CSV]

Default --proto-dir scans both Research_Archive/organic_prototypes/ and
.notongit/.../submissions/_alpha_*_scratch/ for *.onnx files matching
task<NNN>*_organic*.onnx or task<NNN>*v[2-9]*.onnx (versioned prototypes).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import re
import sys
import tempfile
import types
import zipfile
from collections import defaultdict

# Stub IPython/matplotlib so neurogolf_utils imports
ipy = types.ModuleType('IPython')
ipy_disp = types.ModuleType('IPython.display')
ipy_disp.display = lambda *a, **kw: None
ipy_disp.FileLink = lambda x: x
ipy.display = ipy_disp
sys.modules.setdefault('IPython', ipy)
sys.modules.setdefault('IPython.display', ipy_disp)
sys.modules.setdefault('matplotlib', types.ModuleType('matplotlib'))
sys.modules.setdefault('matplotlib.pyplot', types.ModuleType('matplotlib.pyplot'))

os.environ.setdefault('ORT_LOGGING_LEVEL', '3')
import numpy as np
import onnxruntime as ort
ort.set_default_logger_severity(3)

WORKTREE = pathlib.Path(__file__).parent.parent
REPO = WORKTREE
NOGIT_ROOT = pathlib.Path(os.environ.get('NEUROGOLF_NOGIT_ROOT', WORKTREE / '.notongit' / 'NeuroGolf_nogit'))
DATA = pathlib.Path(os.environ.get('NEUROGOLF_DATA_DIR', NOGIT_ROOT / 'data' / 'competitions' / 'neurogolf-2026'))
sys.path.insert(0, str(WORKTREE / 'tools'))
sys.path.insert(0, str(DATA / 'neurogolf_utils'))
import neurogolf_utils as nu
import _score_v2 as scv2

SUB = REPO / 'submissions' / 'submission.zip'


def cost_of_bytes(b: bytes) -> int | None:
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        f.write(b)
        p = f.name
    try:
        r = scv2.score_network_v2(p)
        if r is None:
            return None
        return (r[1] or 0) + (r[2] or 0)
    finally:
        try: os.unlink(p)
        except OSError: pass


def find_prototype_files(scan_dirs: list[pathlib.Path]) -> dict[int, list[pathlib.Path]]:
    """Return {task_id: [paths_to_organic_onnx_or_py]}."""
    out: dict[int, list[pathlib.Path]] = defaultdict(list)
    for d in scan_dirs:
        if not d.exists():
            continue
        for p in d.rglob('*'):
            if p.is_dir():
                continue
            if not (p.suffix == '.onnx' or p.suffix == '.py'):
                continue
            m = re.search(r'task(\d{3})', p.name)
            if not m:
                continue
            if 'organic' not in p.name.lower() and 'prototype' not in p.name.lower():
                # also accept v2/v3/v4/v5 versions
                if not re.search(r'_v[2-9]', p.name):
                    continue
            tid = int(m.group(1))
            out[tid].append(p)
    return out


def analyze(tid: int, proto_path: pathlib.Path) -> dict:
    """Compare a single prototype to the live ONNX for task tid."""
    result = {
        'task_id': tid,
        'proto_path': str(proto_path),
        'proto_ext': proto_path.suffix,
        'proto_cost': None,
        'live_cost': None,
        'proto_vs_expected': None,
        'live_vs_expected': None,
        'proto_vs_live': None,
        'total_pairs': None,
        'verdict': None,
        'lb_delta_if_match': None,
    }
    # Load prototype
    if proto_path.suffix == '.onnx':
        with open(proto_path, 'rb') as f:
            proto_bytes = f.read()
    else:
        # .py — try to import and call _build_model
        spec_dir = proto_path.parent
        sys.path.insert(0, str(spec_dir))
        try:
            mod_name = proto_path.stem
            # Force fresh import
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            mod = __import__(mod_name)
            if not hasattr(mod, '_build_model'):
                result['verdict'] = 'PROTO_BROKEN_NO_BUILD_MODEL'
                return result
            proto_bytes = mod._build_model().SerializeToString()
        except Exception as e:
            result['verdict'] = f'PROTO_BROKEN_{type(e).__name__}'
            return result
        finally:
            sys.path = [p for p in sys.path if p != str(spec_dir)]

    proto_cost = cost_of_bytes(proto_bytes)
    if proto_cost is None:
        result['verdict'] = 'PROTO_BROKEN_SCORE_FAIL'
        return result
    result['proto_cost'] = proto_cost

    # Load live from submission.zip
    name = f'task{tid:03d}.onnx'
    with zipfile.ZipFile(SUB) as zin:
        try:
            live_bytes = zin.read(name)
        except KeyError:
            result['verdict'] = 'NO_LIVE'
            return result
    live_cost = cost_of_bytes(live_bytes)
    result['live_cost'] = live_cost

    # Load task data
    json_path = DATA / f'task{tid:03d}.json'
    if not json_path.exists():
        result['verdict'] = 'NO_DATA'
        return result
    data = json.loads(json_path.read_text())
    pairs = data['train'] + data['test'] + data.get('arc-gen', [])

    try:
        p_sess = ort.InferenceSession(proto_bytes, providers=['CPUExecutionProvider'])
        l_sess = ort.InferenceSession(live_bytes, providers=['CPUExecutionProvider'])
    except Exception as e:
        result['verdict'] = f'SESSION_FAIL_{type(e).__name__}'
        return result

    proto_vs_expected = 0
    live_vs_expected = 0
    proto_vs_live = 0
    for p in pairs:
        bench = nu.convert_to_numpy(p)
        if bench is None:
            continue
        try:
            py = (p_sess.run(['output'], {'input': bench['input']})[0] > 0.0).astype(np.float32)
            ly = (l_sess.run(['output'], {'input': bench['input']})[0] > 0.0).astype(np.float32)
            if np.array_equal(py, bench['output']):
                proto_vs_expected += 1
            if np.array_equal(ly, bench['output']):
                live_vs_expected += 1
            if np.array_equal(py, ly):
                proto_vs_live += 1
        except Exception:
            pass

    result['total_pairs'] = len(pairs)
    result['proto_vs_expected'] = proto_vs_expected
    result['live_vs_expected'] = live_vs_expected
    result['proto_vs_live'] = proto_vs_live

    if proto_vs_live == len(pairs):
        result['verdict'] = 'MATCH_LIVE'
        if live_cost and proto_cost:
            pts_proto = max(1, 25 - math.log(max(1, proto_cost)))
            pts_live = max(1, 25 - math.log(max(1, live_cost)))
            result['lb_delta_if_match'] = round(pts_proto - pts_live, 3)
    elif proto_vs_live > 0:
        result['verdict'] = 'PARTIAL_MATCH_LIVE'
    else:
        result['verdict'] = 'ALGORITHM_DIVERGE'

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proto-dirs', nargs='*', default=[
        str(WORKTREE / 'Research' / 'organic_prototypes'),
        str(NOGIT_ROOT / 'submissions' / '_alpha_a_scratch'),
        str(NOGIT_ROOT / 'submissions' / '_alpha_b_scratch'),
    ])
    ap.add_argument('--out', default=str(WORKTREE / 'Logs' / 'organic_prototype_catalog_2026-05-12.csv'))
    ap.add_argument('--limit', type=int, default=None, help='Limit number of prototypes scanned')
    args = ap.parse_args()

    proto_dirs = [pathlib.Path(p) for p in args.proto_dirs]
    candidates = find_prototype_files(proto_dirs)
    print(f'Found {sum(len(v) for v in candidates.values())} prototypes across {len(candidates)} task IDs')

    results = []
    n = 0
    for tid in sorted(candidates):
        for path in candidates[tid]:
            n += 1
            if args.limit and n > args.limit:
                break
            print(f'  [{n}] task{tid:03d}: {path.name} ...', end=' ', flush=True)
            try:
                r = analyze(tid, path)
            except Exception as e:
                r = {'task_id': tid, 'proto_path': str(path), 'verdict': f'ERROR_{type(e).__name__}'}
            print(f'verdict={r.get("verdict")}, proto_vs_live={r.get("proto_vs_live")}/{r.get("total_pairs")}')
            results.append(r)
        if args.limit and n > args.limit:
            break

    # Write CSV
    fields = ['task_id', 'proto_path', 'proto_ext', 'proto_cost', 'live_cost',
              'total_pairs', 'proto_vs_expected', 'live_vs_expected',
              'proto_vs_live', 'verdict', 'lb_delta_if_match']
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in fields})

    # Summary
    print(f'\n=== Summary ===')
    counts = defaultdict(int)
    for r in results:
        counts[r.get('verdict')] += 1
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f'  {k}: {v}')

    match_live = [r for r in results if r.get('verdict') == 'MATCH_LIVE']
    profitable = [r for r in match_live if (r.get('lb_delta_if_match') or 0) > 0]
    print(f'\nMATCH_LIVE prototypes that would GAIN LB (proto cheaper than live):')
    for r in sorted(profitable, key=lambda x: -(x.get('lb_delta_if_match') or 0))[:15]:
        print(f"  task{r['task_id']:03d}: lb_delta={r['lb_delta_if_match']:+.3f}, "
              f"cost {r['proto_cost']:,} vs live {r['live_cost']:,}, path={pathlib.Path(r['proto_path']).name}")

    print(f'\nCSV written: {args.out}')


if __name__ == '__main__':
    main()
