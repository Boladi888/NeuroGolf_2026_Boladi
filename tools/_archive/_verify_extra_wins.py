"""Verify each extra-scan winner via verify_against_live before adding
to the next megastack. Wildcards (task379, task384) almost certainly
fail — this gate enforces that."""
import json
import os
import re
import subprocess
import time
from pathlib import Path

PY = os.path.abspath('.venv/Scripts/python.exe')

with open('Logs/post_megastack45_extra_wins.json') as f:
    wins = json.load(f)
print(f'Verifying {len(wins)} extra winners...', flush=True)

verified = {}
failed = {}
errored = {}
t0 = time.time()
for i, (tid, info) in enumerate(sorted(wins.items()), 1):
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
            failed[tid] = {'path': info['path'], 'mline': m.group(0) if m else None, 'excerpt': out[-200:]}
    except subprocess.TimeoutExpired:
        errored[tid] = 'timeout'
    except Exception as e:
        errored[tid] = str(e)
    if i % 15 == 0 or i == len(wins):
        print(f'  [{i}/{len(wins)}] elapsed={time.time()-t0:.0f}s  pass={len(verified)} fail={len(failed)} err={len(errored)}', flush=True)

print(f'\nFinal: verified={len(verified)}  failed={len(failed)}  errored={len(errored)}')
total = sum(r['delta'] for r in verified.values())
print(f'Verified cost reduction: {total:,}')
import math
total_pts = sum(math.log(r['canon_cost']) - math.log(r['cand_cost']) for r in verified.values() if r['cand_cost'] > 0)
print(f'Expected LB gain (verified only): +{total_pts:.4f}')

with open('Logs/post_megastack45_verified.json', 'w') as f:
    json.dump({'verified': verified, 'failed': failed, 'errored': errored}, f, indent=2)

if failed:
    print('\nFAILED (dropped from stack):')
    for tid, info in list(failed.items())[:15]:
        print(f"  {tid}: {info.get('mline')} | {Path(info['path']).parent.name}")
print('\nWrote Logs/post_megastack45_verified.json')
