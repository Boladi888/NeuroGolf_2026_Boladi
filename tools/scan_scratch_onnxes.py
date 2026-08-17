"""Score + verify every task*.onnx in the .notongit scratch dirs, write
results to The_400_Tasks/scratch_onnx_costs.json.

These are uncashed prototypes — pre-built ONNXes that didn't ship
(usually because too expensive vs the live payload, or rejected for
hidden-bad risk). Surfacing them is essential for not duplicating work.

For each scratch onnx:
  cost          = memory + params via _score_v2 (None if banned/error)
  verify_str    = "N/T" where N=passed, T=total visible+arc-gen pairs
  verify_frac   = N/T (0..1)
  size_bytes    = file size

Output JSON shape:
  {
    "<tid_int>": {
      "<filename>": {"cost": int, "verify": "N/T", "verify_frac": float, "size": int, "path": str}
    }
  }

Usage:
    python tools/scan_scratch_onnxes.py [--out PATH]
"""
import argparse, json, os, re, sys, tempfile
from pathlib import Path

os.environ['ORT_LOGGING_LEVEL'] = '4'
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / 'tools'))
NOGIT_ROOT = Path(os.environ.get('NEUROGOLF_NOGIT_ROOT', REPO / '.notongit' / 'NeuroGolf_nogit'))
import _score_v2 as scv2
import onnxruntime as ort
ort.set_default_logger_severity(4)
import numpy as np

SCRATCH_DIRS = [
    NOGIT_ROOT / 'submissions' / '_alpha_a_scratch',
    NOGIT_ROOT / 'submissions' / '_alpha_b_scratch',
]
DATA = Path(os.environ.get('NEUROGOLF_DATA_DIR', REPO / 'data' / 'competitions' / 'neurogolf-2026'))
DEFAULT_OUT = REPO / 'The_400_Tasks' / 'scratch_onnx_costs.json'

TASK_RE = re.compile(r'^task(\d{3})(_.*)?\.onnx$')


def encode(grid):
    a = np.array(grid, dtype=np.int64)
    h, w = a.shape
    out = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for c in range(10):
        out[0, c, :h, :w] = (a == c).astype(np.float32)
    out[0, 0, h:, :] = 1.0
    out[0, 0, :h, w:] = 1.0
    return out


def decode(arr, h, w):
    grid = []
    for r in range(h):
        row = []
        for c in range(w):
            vec = arr[0, :, r, c]
            idxs = np.where(np.isclose(vec, 1.0, atol=1e-4))[0]
            if len(idxs) == 1:
                row.append(int(idxs[0]))
            elif len(idxs) == 0:
                row.append(0)
            else:
                row.append(11)
        grid.append(row)
    return grid


def verify(onnx_path: str, tid: int) -> tuple[str, float]:
    task_json = DATA / f'task{tid:03d}.json'
    if not task_json.exists():
        return '0/0', 0.0
    data = json.loads(task_json.read_text())
    pairs = data.get('train', []) + data.get('test', []) + data.get('arc-gen', [])
    try:
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    except Exception:
        return 'session-err', 0.0
    correct = total = 0
    for p in pairs:
        inp_arr = np.array(p['input'])
        if inp_arr.shape[0] > 30 or inp_arr.shape[1] > 30:
            continue
        total += 1
        try:
            y = sess.run(['output'], {'input': encode(p['input'])})[0]
            h, w = len(p['output']), len(p['output'][0])
            pred = decode(y, h, w)
            if pred == p['output']:
                correct += 1
        except Exception:
            pass
    return f'{correct}/{total}', (correct / total if total else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    # Collect files
    scratch = {}  # tid -> list of (path, name)
    for d in SCRATCH_DIRS:
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            m = TASK_RE.match(f.name)
            if not m:
                continue
            tid = int(m.group(1))
            scratch.setdefault(tid, []).append((str(f), f.name))

    total_files = sum(len(v) for v in scratch.values())
    print(f'Found {total_files} scratch ONNXes across {len(scratch)} tasks', flush=True)

    results = {}
    done = 0
    for tid in sorted(scratch):
        results[str(tid)] = {}
        for path, fname in scratch[tid]:
            done += 1
            entry = {'size': os.path.getsize(path), 'path': path}
            try:
                r = scv2.score_network_v2(path)
                if r is None:
                    entry['cost'] = None
                else:
                    entry['cost'] = (r[1] or 0) + (r[2] or 0)
                    entry['macs'] = r[0] or 0
                    entry['memory'] = r[1] or 0
                    entry['params'] = r[2] or 0
            except Exception as e:
                entry['cost'] = None
                entry['error'] = f'{type(e).__name__}: {str(e)[:80]}'
            # Verify (skip if cost is None / banned)
            if entry.get('cost') is not None:
                v_str, v_frac = verify(path, tid)
                entry['verify'] = v_str
                entry['verify_frac'] = v_frac
            results[str(tid)][fname] = entry
            if done % 10 == 0:
                print(f'  {done}/{total_files} done', flush=True)

    args.out.write_text(json.dumps(results, indent=1), encoding='utf-8')
    print(f'Wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
