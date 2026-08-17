"""Verify submission.zip: every ONNX loads, scores, and tally cost/pts."""
import sys, os, zipfile, tempfile, math
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "tools"))
import onnxruntime as ort
ort.set_default_logger_severity(3)
import _score_v2 as scv2

zip_path = str(_REPO / "submissions" / "submission.zip")
with zipfile.ZipFile(zip_path) as zf:
    names = sorted([n for n in zf.namelist() if n.endswith('.onnx')])

print(f'ENTRIES: {len(names)}', flush=True)

total_cost = 0
total_pts = 0
load_fails = []
score_fails = []
for i, fn in enumerate(names):
    if i >= 388:
        print(f'  ... {i}/{len(names)} ({fn})', flush=True)
    elif i % 50 == 0:
        print(f'  ... {i}/{len(names)}', flush=True)
    fn_inner = fn
    with zipfile.ZipFile(zip_path) as zf:
        b = zf.read(fn_inner)
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        f.write(b)
        pp = f.name
    try:
        sess = ort.InferenceSession(pp, providers=['CPUExecutionProvider'])
        sc = scv2.score_network_v2(pp)
        if sc is None:
            score_fails.append(fn)
        else:
            cost = sc[1] + sc[2]
            pts = max(1, 25 - math.log(max(1, cost)))
            total_cost += cost
            total_pts += pts
    except Exception as e:
        load_fails.append((fn, str(e)[:80]))
    os.unlink(pp)

summary = [
    f'LOAD_FAILS: {len(load_fails)}',
    f'SCORE_FAILS: {len(score_fails)}',
    f'TOTAL_COST: {total_cost:,}',
    f'LOCAL_POINTS_IF_ALL_CORRECT: {total_pts:.2f}',
    'CONFIRMED_KAGGLE_PUBLIC_SCORE: UNKNOWN',
    'CANDIDATE_STATUS: UNVERIFIED_UNTIL_KAGGLE_PUBLIC_SCORE_CONFIRMS',
]
if load_fails:
    summary.append('LOAD_FAIL_SAMPLES:')
    for fn, err in load_fails[:10]:
        summary.append(f'  {fn}: {err}')
if score_fails:
    summary.append('SCORE_FAIL_SAMPLES:')
    for fn in score_fails[:10]:
        summary.append(f'  {fn}')
text = '\n'.join(summary)
print(text, flush=True)
with open(Path(tempfile.gettempdir()) / 'verify_result.txt', 'w') as f:
    f.write(text)
