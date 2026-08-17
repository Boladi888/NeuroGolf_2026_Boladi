"""One-off: scan ALL candidate ONNXs in staged_submits, score against
current megastack45 submission.zip, find any task we missed."""
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PY = os.path.abspath('.venv/Scripts/python.exe')

with open('Logs/post_megastack162_verified.json') as f:
    prev = json.load(f)
already_shipping = set(prev['verified'].keys())
print(f'Already in megastack45: {len(already_shipping)} tasks', flush=True)

canon_path = 'submissions/submission.zip'
work = Path('Logs/scratch_canon_extract')
work.mkdir(exist_ok=True, parents=True)


def score(p):
    r = subprocess.run([PY, 'tools/spread_score_isolated.py', str(p)],
                       capture_output=True, text=True, timeout=60)
    try:
        return json.loads(r.stdout.strip())
    except Exception:
        return None


task_candidates = {}
for d in Path('.hermes/board/staged_submits').iterdir():
    if not d.is_dir():
        continue
    for f in d.rglob('*.onnx'):
        s = str(f).lower()
        if 'verify' in s or os.sep + 'base' + os.sep in s.lower():
            continue
        name = f.stem
        if name.startswith('task') and len(name) == 7:
            task_candidates.setdefault(name, []).append(f)

# Only tasks NOT in megastack45
candidate_tasks = {t: v for t, v in task_candidates.items() if t not in already_shipping}
print(f'Tasks with candidates NOT in megastack45: {len(candidate_tasks)}', flush=True)

canon_costs = {}
with zipfile.ZipFile(canon_path) as z:
    for tid in candidate_tasks:
        p = work / f'{tid}_megastack45.onnx'
        try:
            with z.open(f'{tid}.onnx') as f:
                p.write_bytes(f.read())
        except KeyError:
            continue
        s = score(p)
        if s and 'cost' in s:
            canon_costs[tid] = (s['cost'], p)

print(f'Scored canonicals for {len(canon_costs)} tasks', flush=True)

wins = {}
t0 = time.time()
n_total = len(canon_costs)
for i, (tid, (canon_cost, canon_p)) in enumerate(sorted(canon_costs.items()), 1):
    best = None
    for cand_path in task_candidates[tid]:
        s = score(cand_path)
        if s is None or 'cost' not in s:
            continue
        delta = s['cost'] - canon_cost
        if best is None or delta < best['delta']:
            best = {
                'delta': delta,
                'cand_cost': s['cost'],
                'path': str(cand_path),
                'canon_cost': canon_cost,
            }
    if best and best['delta'] < 0:
        wins[tid] = best
    if i % 20 == 0 or i == n_total:
        print(f'  [{i}/{n_total}] elapsed={time.time()-t0:.0f}s  new_wins={len(wins)}', flush=True)

print(f'\nNEW wins vs megastack45: {len(wins)}')
total_delta = sum(r['delta'] for r in wins.values())
print(f'Cost reduction: {total_delta:,}')
total_pts = sum(
    math.log(r['canon_cost']) - math.log(r['cand_cost'])
    for r in wins.values() if r['cand_cost'] > 0
)
print(f'Expected LB gain on top of megastack45: +{total_pts:.4f}')

with open('Logs/post_megastack45_extra_wins.json', 'w') as f:
    json.dump(wins, f, indent=2)

print('\nTop 15:')
for tid, r in sorted(wins.items(), key=lambda x: x[1]['delta'])[:15]:
    print(f"  {tid}: {r['canon_cost']} -> {r['cand_cost']} ({r['delta']:+}) | {Path(r['path']).parent.name}")
