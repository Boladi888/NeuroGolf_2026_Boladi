"""Quick scan for new algebraic pattern savings (bool_andfold, add_chain_to_sum, shared_cast_equal_fold)."""
import sys
import zipfile
import tempfile
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['ORT_LOGGING_LEVEL'] = '3'

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import onnx
from tools._score_v2 import score_network_v2
from tools.spread_patterns.pass_bool_andfold import _apply as ab
from tools.spread_patterns.pass_add_chain_to_sum import _apply as ac
from tools.spread_patterns.pass_shared_cast_equal_fold import _apply as sc

SUBMISSION_ZIP = Path(__file__).parents[1] / 'submissions' / 'submission.zip'


def main():
    start_task = sys.argv[1] if len(sys.argv) > 1 else 'task000'
    end_task = sys.argv[2] if len(sys.argv) > 2 else 'task999'

    results = []
    tmpdir = Path(tempfile.mkdtemp())
    try:
        with zipfile.ZipFile(SUBMISSION_ZIP) as z:
            names = [n.replace('.onnx', '') for n in sorted(z.namelist()) if n.endswith('.onnx')]
            subset = [n for n in names if start_task <= n <= end_task]
            print(f'Scanning {len(subset)} tasks ({start_task}..{end_task})', flush=True)

            for i, tid in enumerate(subset):
                try:
                    z.extract(f'{tid}.onnx', tmpdir)
                    src = tmpdir / f'{tid}.onnx'
                    m = onnx.load(str(src))
                    _, om, op = score_network_v2(str(src))
                    oc = om + op
                    sv = {}
                    for pn, pf in [('add_chain', ac), ('bool_fold', ab), ('cast_fold', sc)]:
                        try:
                            r = pf(m, tid)
                            if r.verdict == 'applied' and r.new_model:
                                cp = tmpdir / f'{tid}_{pn}.onnx'
                                onnx.save(r.new_model, str(cp))
                                _, nm, np2 = score_network_v2(str(cp))
                                d = (nm + np2) - oc
                                if d < 0:
                                    sv[pn] = d
                        except Exception:
                            pass
                    if sv:
                        results.append((tid, sv, sum(sv.values())))
                        print(f'HIT {tid}: {sv}', flush=True)
                    if (i + 1) % 50 == 0:
                        print(f'... {i+1}/{len(subset)} done', flush=True)
                except Exception:
                    pass
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    print('DONE', flush=True)
    for tid, sv, tot in sorted(results, key=lambda x: x[2]):
        print(f'RESULT {tid}: {tot:+d} {sv}')


if __name__ == '__main__':
    main()
