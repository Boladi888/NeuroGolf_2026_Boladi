"""Read the post-5833.74 spread_scan results, pick the best applied
pattern per task, run verify_against_live on each, write the verified
set so we can build a megastack."""
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PY = os.path.abspath('.venv/Scripts/python.exe')

with open('Logs/spread_scan_results/on_5833_74/spread_scan_20260520.json') as f:
    data = json.load(f)

# Pick best applied per task
best = {}
for t in data['tasks']:
    tid = t['task_id']
    orig = t.get('original') or {}
    oc = orig.get('cost', 0)
    best_cd = 0
    best_path = None
    best_pat = None
    for p in t.get('patterns', []):
        if p.get('verdict') != 'applied':
            continue
        cd = p.get('cost_delta') or 0
        cand = p.get('extras', {}).get('candidate_path')
        if cd < best_cd and cand:
            best_cd = cd
            best_path = cand
            best_pat = p['name']
    if best_path:
        best[tid] = {
            'pattern': best_pat,
            'cost_delta': best_cd,
            'path': best_path,
            'orig_cost': oc,
            'new_cost': oc + best_cd,
        }

print(f'Best-per-task applied: {len(best)} candidates', flush=True)
print(f'Total cost reduction: {sum(b["cost_delta"] for b in best.values()):,} bytes', flush=True)

# Verify each
verified, failed, errored = {}, {}, {}
t0 = time.time()
total = len(best)
for i, (tid, info) in enumerate(sorted(best.items()), 1):
    p = info['path']
    if not Path(p).exists():
        errored[tid] = 'missing'
        continue
    tnum = tid.replace('task', '')
    try:
        r = subprocess.run(
            [PY, 'tools/verify_against_live.py', '--proto', p, '--tid', tnum,
             '--live-zip', 'submissions/submission.zip'],
            capture_output=True, text=True, timeout=90,
        )
        out = (r.stdout or '') + (r.stderr or '')
        m = re.search(r'Prototype\s+vs LIVE\s*:\s*(\d+)\s*/\s*(\d+)', out)
        ok_line = 'OK Prototype matches LIVE' in out
        is_pass = bool(m) and m.group(1) == m.group(2) and ok_line
        if is_pass:
            verified[tid] = {**info, 'verify': f'{m.group(1)}/{m.group(2)}'}
        else:
            failed[tid] = {'pattern': info['pattern'], 'mline': m.group(0) if m else None}
    except subprocess.TimeoutExpired:
        errored[tid] = 'timeout'
    except Exception as e:
        errored[tid] = str(e)
    if i % 15 == 0 or i == total:
        print(f'  [{i}/{total}] elapsed={time.time()-t0:.0f}s pass={len(verified)} fail={len(failed)} err={len(errored)}', flush=True)

print(f'\nFinal: verified={len(verified)} failed={len(failed)} errored={len(errored)}', flush=True)
total_delta = sum(r['cost_delta'] for r in verified.values())
print(f'Verified cost reduction: {total_delta:,} bytes', flush=True)
total_pts = sum(math.log(r['orig_cost']) - math.log(r['new_cost'])
                for r in verified.values() if r['new_cost'] > 0)
print(f'Expected LB gain: +{total_pts:.4f}', flush=True)

with open('Logs/post_5833_74_verified.json', 'w') as f:
    json.dump({'verified': verified, 'failed': failed, 'errored': errored}, f, indent=2)
print('Wrote Logs/post_5833_74_verified.json', flush=True)

if failed:
    print(f'\nFailed ({len(failed)}, dropped):')
    for tid, info in list(failed.items())[:10]:
        print(f"  {tid}: {info.get('mline')} | {info['pattern']}")
