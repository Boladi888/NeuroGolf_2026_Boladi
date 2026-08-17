"""Score + verify every staged_submits/_on_5835_30/ candidate that
isn't already in the current canonical. Split into HIGH confidence
(verify M/M PASS + large delta) and LOW confidence (verify imperfect
OR small delta OR unusual technique). Write inventory JSON."""
import hashlib
import json
import math
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path

PY = os.path.abspath('.venv/Scripts/python.exe')

# Build canonical reference from current submission.zip
canon_path = 'submissions/submission.zip'
work = Path('Logs/scratch_canon_5835_30')
work.mkdir(exist_ok=True, parents=True)
with zipfile.ZipFile(canon_path) as z:
    canonical_shas = {}
    for n in z.namelist():
        if n.endswith('.onnx'):
            tid = os.path.basename(n).replace('.onnx', '')
            data = z.read(n)
            canonical_shas[tid] = hashlib.sha256(data).hexdigest()[:8]

print(f'Canonical has {len(canonical_shas)} ONNX entries', flush=True)


def score(p):
    r = subprocess.run([PY, 'tools/spread_score_isolated.py', str(p)],
                       capture_output=True, text=True, timeout=60)
    try:
        return json.loads(r.stdout.strip())
    except Exception:
        return None


# Walk staged_submits looking for *_on_5835_30 dirs (built against current canon)
candidates = {}  # tid -> list of {path, staged_dir, manifest_score_delta}
for d in Path('.hermes/board/staged_submits').iterdir():
    if not d.is_dir():
        continue
    if 'on_5835_30' not in d.name:
        continue
    # Find the taskNNN.onnx inside
    tid = None
    onnx_path = None
    manifest = {}
    mf = d / 'manifest.json'
    if mf.exists():
        try:
            manifest = json.loads(mf.read_text(encoding='utf-8', errors='replace'))
            tid = manifest.get('task_id')
        except Exception:
            pass
    if not tid:
        # Infer tid from dir name
        m = re.match(r'(task\d{3})_', d.name)
        if not m:
            continue
        tid = m.group(1)
    for f in d.rglob('*.onnx'):
        if f.name == f'{tid}.onnx':
            onnx_path = f
            break
    if not onnx_path or not onnx_path.exists():
        continue
    # Skip if byte-identical to canonical
    cand_sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()[:8]
    if cand_sha == canonical_shas.get(tid):
        continue
    candidates.setdefault(tid, []).append({
        'path': str(onnx_path),
        'staged_dir': d.name,
        'manifest': manifest,
    })

print(f'{len(candidates)} unique tasks with candidate ONNXs on 5835.30', flush=True)

# Score each candidate, score canonical, pick best per task
results = {}
t0 = time.time()
for i, (tid, clist) in enumerate(sorted(candidates.items()), 1):
    canon_p = work / f'{tid}_canonical.onnx'
    if not canon_p.exists():
        with zipfile.ZipFile(canon_path) as z:
            try:
                with z.open(f'{tid}.onnx') as f:
                    canon_p.write_bytes(f.read())
            except KeyError:
                continue
    canon_score_obj = score(canon_p)
    if not canon_score_obj or 'cost' not in canon_score_obj:
        continue
    canon_cost = canon_score_obj['cost']
    best = None
    for c in clist:
        s = score(c['path'])
        if not s or 'cost' not in s:
            continue
        delta = s['cost'] - canon_cost
        if best is None or delta < best['delta']:
            best = {
                'delta': delta,
                'cand_cost': s['cost'],
                'canon_cost': canon_cost,
                'path': c['path'],
                'staged_dir': c['staged_dir'],
                'manifest': c['manifest'],
            }
    if best and best['delta'] < 0:
        results[tid] = best
    if i % 20 == 0 or i == len(candidates):
        wins = len(results)
        print(f'  [{i}/{len(candidates)}] elapsed={time.time()-t0:.0f}s wins={wins}', flush=True)

print(f'\nWins (cost-improving): {len(results)}', flush=True)
total_delta = sum(r['delta'] for r in results.values())
print(f'Total cost reduction: {total_delta:,} bytes', flush=True)

# Verify each via verify_against_live
print(f'\nRunning verify_against_live on {len(results)} winners...', flush=True)
verified = {}
verify_imperfect = {}
errored = {}
t1 = time.time()
for i, (tid, info) in enumerate(sorted(results.items()), 1):
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
        elif m:
            # Imperfect verify — record but separate
            pass_n, total_n = int(m.group(1)), int(m.group(2))
            verify_imperfect[tid] = {**info, 'verify': f'{pass_n}/{total_n}', 'pass_rate': pass_n / max(total_n, 1)}
        else:
            errored[tid] = {**info, 'excerpt': out[-200:]}
    except subprocess.TimeoutExpired:
        errored[tid] = {**info, 'note': 'timeout'}
    except Exception as e:
        errored[tid] = {**info, 'note': str(e)}
    if i % 20 == 0 or i == len(results):
        print(f'  [{i}/{len(results)}] elapsed={time.time()-t1:.0f}s pass={len(verified)} imperfect={len(verify_imperfect)} err={len(errored)}', flush=True)

print(f'\nFinal: verified PASS={len(verified)}  imperfect={len(verify_imperfect)}  errored={len(errored)}', flush=True)
total_pass_delta = sum(r['delta'] for r in verified.values())
print(f'Verified-PASS cost reduction: {total_pass_delta:,} bytes', flush=True)
total_pass_pts = sum(math.log(r['canon_cost']) - math.log(r['cand_cost'])
                     for r in verified.values() if r['cand_cost'] > 0)
print(f'Verified-PASS expected LB gain: +{total_pass_pts:.4f}', flush=True)

# Save inventory
out = {'verified': verified, 'verify_imperfect': verify_imperfect, 'errored': errored}
with open('Logs/inventory_on_5835_30.json', 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f'\nWrote Logs/inventory_on_5835_30.json', flush=True)

# Top 20 winners
print(f'\nTop 20 verified-PASS:')
for tid, r in sorted(verified.items(), key=lambda x: x[1]['delta'])[:20]:
    print(f"  {tid}: {r['canon_cost']} -> {r['cand_cost']} ({r['delta']:+}) | {r['staged_dir']}")

if verify_imperfect:
    print(f'\nImperfect verify (will batch carefully):')
    for tid, r in sorted(verify_imperfect.items(), key=lambda x: -x[1]['pass_rate'])[:15]:
        print(f"  {tid}: pass={r['verify']} delta={r['delta']:+} | {r['staged_dir']}")
