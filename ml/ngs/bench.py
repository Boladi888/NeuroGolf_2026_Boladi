"""bench.py — measure search throughput (candidate graphs evaluated / sec).

The ngs search is ORT-inference-bound, so this is the number that decides how deep/wide it can
go in a given wall-clock budget. Run it in the project venv (CPU baseline) and again in an Arc-B70
ONNX env (OpenVINO/DML EP — find it with `python -m ml.ngs.backends`) to see the GPU speedup.

  .venv/Scripts/python.exe -m ml.ngs.bench [--n 200] [--pairs 6]
"""
import json
import pathlib
import sys
import time
import itertools
import numpy as np

from . import score, tasks
from .ops import base_vocab
from .builder import to_model, straight

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCH_DIR = pathlib.Path(__file__).resolve().parent / "_bench_onnx"


def _candidate_models(n):
    """Build up to n distinct valid candidate graphs (depth 1-2 programs over the vocab)."""
    vocab = base_vocab()
    progs = [[]]
    progs += [[op] for op in vocab]
    progs += [[a, b] for a, b in itertools.product(vocab, vocab)]
    models = []
    for p in progs:
        g = straight(p)
        m = to_model(g) if g is not None else None
        if m is not None:
            models.append(m)
        if len(models) >= n:
            break
    return models


def _bench(models, pairs, providers, label):
    t0 = time.perf_counter()
    solved = 0
    for m in models:
        ok, tot = score.evaluate(m, pairs, providers=providers)
        solved += (ok == tot)
    dt = time.perf_counter() - t0
    rate = len(models) / dt if dt else 0.0
    print(f"  {label:28} {len(models):4} graphs in {dt:6.2f}s  ->  {rate:7.1f} graphs/s   (solved {solved})")
    return rate


def _dump_candidates(n, npairs):
    """Build candidate .onnx + pairs.json into BENCH_DIR (project venv) for cross-env ov_eval."""
    import onnx
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    for f in BENCH_DIR.glob("*.onnx"):
        f.unlink()
    pairs = tasks.all_pairs(tasks.synthetic("colorswap", n=npairs, seed=1))
    models = _candidate_models(n)
    for i, m in enumerate(models):
        onnx.save(m, str(BENCH_DIR / f"cand_{i:04d}.onnx"))
    (BENCH_DIR / "pairs.json").write_text(
        json.dumps([[ig.tolist(), og.tolist()] for ig, og in pairs]), encoding="utf-8")
    return len(models), len(pairs)


def ov_compare(n, npairs):
    """B70 (OpenVINO GPU) vs OpenVINO CPU on the SAME candidate set — auto-targets the OpenVINO env."""
    from . import backends
    import subprocess
    nm, npr = _dump_candidates(n, npairs)
    print(f"== ngs OpenVINO bench: dumped {nm} graphs + {npr} pairs -> {BENCH_DIR} ==")
    rep = backends.load_report() or backends.scan()
    ov = backends.best_openvino_env(rep)
    if not ov:
        print("  no OpenVINO/B70 env found — run `python -m ml.ngs.backends` first"); return 1
    print(f"  OpenVINO env: {ov['env_path']}  (devices {ov.get('devices')})")
    for dev in ("GPU", "CPU"):
        print(f"  -- OpenVINO {dev} --", flush=True)
        subprocess.run([ov["env_path"], "-m", "ml.ngs.ov_eval", str(BENCH_DIR), dev], cwd=str(REPO_ROOT))
    print("  (compare these graphs/s to the project-venv CPU `python -m ml.ngs.bench` number;")
    print("   per-model GPU compile overhead means GPU wins only for HEAVY graphs / batched eval.)")
    return 0


def main():
    if "--ov" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 24
        npairs = int(sys.argv[sys.argv.index("--pairs") + 1]) if "--pairs" in sys.argv else 6
        return ov_compare(n, npairs)
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 200
    npairs = int(sys.argv[sys.argv.index("--pairs") + 1]) if "--pairs" in sys.argv else 6
    pairs = tasks.all_pairs(tasks.synthetic("transpose", n=npairs, seed=1))
    models = _candidate_models(n)
    print(f"== ngs search-throughput bench: {len(models)} graphs x {len(pairs)} pairs ==")
    print(f"  this interpreter providers: {score.ort.get_available_providers()}")

    cpu_rate = _bench(models, pairs, ["CPUExecutionProvider"], "CPU")
    gpu = [p for p in score.GPU_EPS if p in score.ort.get_available_providers()]
    if gpu:
        gpu_rate = _bench(models, pairs, [gpu[0], "CPUExecutionProvider"], gpu[0].replace("ExecutionProvider", ""))
        if cpu_rate:
            print(f"\n  GPU speedup vs CPU: {gpu_rate / cpu_rate:.2f}x")
    else:
        print("\n  (no ONNX GPU EP in this env — run `python -m ml.ngs.backends` to find a B70 ONNX env,")
        print("   then run this bench with that interpreter to measure the speedup.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
