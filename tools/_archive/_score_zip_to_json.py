"""Score every ONNX in a submission zip, write results to JSON.

Saves incrementally every 50 tasks so a kill doesn't lose all work.

Usage:
    python tools/_score_zip_to_json.py [--zip ZIP] [--out JSON]
"""
import argparse, json, os, sys, tempfile
from pathlib import Path
from zipfile import ZipFile

os.environ['ORT_LOGGING_LEVEL'] = '4'
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / 'tools'))
import _score_v2 as scv2
import onnxruntime as ort
ort.set_default_logger_severity(4)

DEFAULT_ZIP = Path(r'<NOGIT_ROOT>\submissions\alpha_b_t287v4_5653_67\submission.zip')
DEFAULT_OUT = REPO / 'The_400_Tasks' / 'live_costs.json'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--zip', type=Path, default=DEFAULT_ZIP)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--start', type=int, default=0, help='Resume from this index')
    p.add_argument('--end', type=int, default=None, help='Stop at this index (exclusive)')
    args = p.parse_args()

    # Resume from existing JSON if any
    results: dict[str, dict] = {}
    if args.out.exists():
        try:
            results = json.loads(args.out.read_text(encoding='utf-8'))
            print(f'Resumed from {args.out} with {len(results)} existing entries', flush=True)
        except Exception:
            pass

    print(f'Scoring {args.zip}...', flush=True)
    with ZipFile(args.zip) as z:
        names = [n for n in sorted(z.namelist()) if n.startswith('task') and n.endswith('.onnx')]
        end = args.end or len(names)
        for i in range(args.start, end):
            name = names[i]
            tid = int(name[4:7])
            if str(tid) in results:
                continue
            tmp = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
            tmp.write(z.read(name))
            tmp.close()
            try:
                r = scv2.score_network_v2(tmp.name)
                if r:
                    results[str(tid)] = {
                        'macs': r[0] or 0,
                        'memory': r[1] or 0,
                        'params': r[2] or 0,
                        'cost': (r[1] or 0) + (r[2] or 0),
                    }
            except Exception:
                pass
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            # Incremental save every 25 tasks (cheap)
            if (i + 1) % 25 == 0:
                args.out.write_text(json.dumps(results, indent=1), encoding='utf-8')
                print(f'  {i+1}/{end} scored, {len(results)} saved', flush=True)
    # Final save
    args.out.write_text(json.dumps(results, indent=1), encoding='utf-8')
    print(f'Total scored: {len(results)}', flush=True)
    print(f'Wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
