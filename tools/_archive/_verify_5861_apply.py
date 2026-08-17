"""Verify best applied candidates from 5861.43 scan and report."""
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

PY = os.path.abspath('.venv/Scripts/python.exe')

with open('Logs/spread_scan_results/on_5861_43/spread_scan_20260520.json') as f:
    data = json.load(f)

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

print(f'Best-per-task applied: {len(best)}', flush=True)
print(f'Total cost reduction: {sum(b["cost_delta"] for b in best.values()):,} bytes', flush=True)

verified, failed, errored = {}, {}, {}
t0 = time.time()
for i, (tid, info) in enumerate(sorted(best.items()), 1):
    if not Path(info['path']).exists():
        errored[tid] = 'missing'
        continue
    tnum = tid.replace('task', '')
    try:
        r = subprocess.run(
            [PY, 'tools/verify_against_live.py', '--proto', info['path'], '--tid', tnum,
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
    if i % 10 == 0 or i == len(best):
        print(f'  [{i}/{len(best)}] pass={len(verified)} fail={len(failed)} err={len(errored)}', flush=True)

print(f'\nFinal: verified={len(verified)} failed={len(failed)} errored={len(errored)}', flush=True)
total_delta = sum(r['cost_delta'] for r in verified.values())
print(f'Verified cost reduction: {total_delta:,} bytes', flush=True)
total_pts_loge = sum(math.log(r['orig_cost']) - math.log(r['new_cost'])
                     for r in verified.values() if r['new_cost'] > 0)
print(f'Log_e expected LB gain: +{total_pts_loge:.4f}', flush=True)
print(f'Calibrated LB estimate (x0.48): +{total_pts_loge * 0.48:.4f}', flush=True)

with open('Logs/post_5861_43_verified.json', 'w') as f:
    json.dump({'verified': verified, 'failed': failed, 'errored': errored}, f, indent=2)
print('Wrote Logs/post_5861_43_verified.json', flush=True)

print('\nTop wins:')
for tid, r in sorted(verified.items(), key=lambda x: x[1]['cost_delta'])[:15]:
    print(f"  {tid}: {r['orig_cost']} -> {r['new_cost']} ({r['cost_delta']:+}) [{r['pattern']}]")
