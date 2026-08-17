"""Apply SAFE math-equivalent ONNX optimization pipeline to fig13b 5652.54.

Source: K-NeuroGolf/submissions/alpha_b_t074organic_5652_54/submission.zip
Output: K-NeuroGolf/submissions/_alpha_a_scratch/fig13b_safe/submission.zip

Pipeline (per task ONNX):
  1. dedupe_inits (always safe — same data, same shape, same dtype = one init)
  2. optimize (dedupe + CSE + DCE + scalar algebraic identities incl. Concat-1-input)
  3. verify_equivalent on first train pair; if CSE broke output, FALLBACK to dedupe-only
"""
import os, sys, zipfile, pathlib

os.environ['ORT_LOGGING_LEVEL'] = '3'
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import onnx
from tools.onnx_opt_safe import (
    dedupe_inits, optimize, verify_equivalent,
)


def main():
    NG = pathlib.Path(r'<REPO_ROOT>')
    src_zip = NG / 'submissions/alpha_b_t074organic_5652_54/submission.zip'
    out_dir = NG / 'submissions/_alpha_a_scratch/fig13b_safe'
    out_dir.mkdir(parents=True, exist_ok=True)
    new_zip = out_dir / 'submission.zip'

    assert src_zip.exists(), f'missing source: {src_zip}'

    src_size = src_zip.stat().st_size
    print(f'Source: {src_zip}  ({src_size:,} bytes)', flush=True)

    total = {'dedupe': 0, 'cse': 0, 'dce_nodes': 0, 'dce_inits': 0, 'alg': 0}
    n_processed = 0
    n_failed = 0
    n_fallback = 0
    new_entries = {}
    with zipfile.ZipFile(src_zip) as zin:
        for name in sorted(zin.namelist()):
            data = zin.read(name)
            if not name.endswith('.onnx'):
                new_entries[name] = data
                continue
            try:
                m = onnx.load_from_string(data)
                stats = optimize(m)
                new_data = m.SerializeToString()
                tid = int(name[4:7])
                ok = verify_equivalent(data, new_data, tid)
                if ok is False:
                    m2 = onnx.load_from_string(data)
                    dedupe_inits(m2)
                    new_data = m2.SerializeToString()
                    n_fallback += 1
                else:
                    for k in total:
                        total[k] += stats[k]
                new_entries[name] = new_data
            except Exception as e:
                new_entries[name] = data
                n_failed += 1
                print(f'  WARN {name}: {e}', flush=True)
            n_processed += 1
            if n_processed % 50 == 0:
                print(f'  processed {n_processed}, fallback {n_fallback}, failed {n_failed}', flush=True)

    print(f'\nTotal: processed {n_processed}, fallback {n_fallback}, failed {n_failed}', flush=True)
    print(f'Stats (non-fallback only): {total}', flush=True)

    with zipfile.ZipFile(new_zip, 'w', zipfile.ZIP_DEFLATED) as zout:
        for name, data in sorted(new_entries.items()):
            zout.writestr(name, data)

    out_size = new_zip.stat().st_size
    delta = src_size - out_size
    pct = 100.0 * delta / src_size if src_size else 0
    print(f'\nWrote {new_zip}  ({out_size:,} bytes; {delta:+,} = {pct:+.2f}% vs src)', flush=True)


if __name__ == '__main__':
    main()
