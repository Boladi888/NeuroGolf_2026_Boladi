"""train.py — train the PolicyModel to guide the search (needs torch -> .venv-ml).

Primary loop = SEARCH-BOOTSTRAP IMITATION over the DSL vocab (the policy this trains is the
exact `.rank()` search.beam_search calls):
  1. run search on a pool of tasks (synthetic targets the DSL can solve + any real solves);
  2. every solving program yields supervision: each prefix -> next-op is a (state, action) pair;
  3. train PolicyModel (cross-entropy over the vocab) on those pairs;
  4. (RL step, documented) re-run guided search with the new policy, harvest fresh solves, repeat.

Honest status: with today's 8-op DSL the solvable pool is thin (mostly identity/transpose), so
this demonstrates a CORRECT, RUNNABLE loop more than a strong policy. Its value scales directly
with the DSL vocab (see mine.py's "ops to add next") and with real-task solves.

  .venv-ml/Scripts/python.exe -m ml.ngs.train          # train + save _corpus/policy.pt
"""
import pathlib
import sys

from .ops import base_vocab
from .builder import program_label
from . import tasks
from .search import beam_search
from .policy import PolicyModel, task_features

OUT = pathlib.Path(__file__).resolve().parent / "_corpus"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _auto_relaunch():
    """Re-exec training in the discovered B70 ngs env (onnx+ort+torch-xpu). Returns rc, or None
    if we're already in the right interpreter."""
    import subprocess
    from . import backends
    rep = backends.load_report() or backends.scan()
    target = rep.get("best_ngs_env") or rep.get("best_torch")
    if not target:
        print("  no B70 torch env discovered — run `python -m ml.ngs.backends` first"); return 1
    py = target["env_path"]
    if pathlib.Path(py) == pathlib.Path(sys.executable):
        return None
    print(f"  auto: re-launching training on '{target.get('device','?')}' via {py}")
    return subprocess.run([py, "-m", "ml.ngs.train", "--no-relaunch"], cwd=str(REPO_ROOT)).returncode


def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def collect_demonstrations(vocab, pool):
    """Run search on each task; return [(features_vec, op_index_sequence)] for every solve."""
    demos = []
    for kind_or_tid in pool:
        if isinstance(kind_or_tid, str):
            task = tasks.synthetic(kind_or_tid, n=10, seed=0)   # >=10 so CoordRemap fingerprints are unambiguous
        else:
            task = tasks.load_task(kind_or_tid)
            if task is None:
                continue
        pairs = tasks.all_pairs(task)
        best, _ = beam_search(pairs, vocab=vocab, max_depth=3, beam_width=8)
        if best and best["program"]:
            idx = {op.label: i for i, op in enumerate(vocab)}
            seq = [idx[op.label] for op in best["program"] if op.label in idx]
            if seq:
                demos.append((task_features(pairs), seq))
    return demos


def main(epochs=200, lr=1e-3):
    torch = _torch()
    if "--auto" in sys.argv and "--no-relaunch" not in sys.argv and torch is None:
        rc = _auto_relaunch()
        if rc is not None:
            return rc
        torch = _torch()
    if torch is None:
        print("torch not available in this interpreter. Options:")
        print("  python -m ml.ngs.train --auto      # auto-target the discovered B70 torch env")
        print("  <b70-env>/Scripts/python.exe -m ml.ngs.train    # see `python -m ml.ngs.backends`")
        return 2
    import torch.nn as nn

    vocab = base_vocab()
    # cover every data-aware op the DSL can demonstrate (recolor + the coordinate-remap family);
    # add real solved tasks with --scan N. Each synthetic target is guaranteed-solvable by design.
    pool = ["identity", "transpose", "colorswap", "rot180", "mirror2x2"]
    if "--scan" in sys.argv:
        pool += list(range(1, int(sys.argv[sys.argv.index("--scan") + 1]) + 1))
    demos = collect_demonstrations(vocab, pool)
    print(f"collected {len(demos)} solving demonstration(s) over a {len(vocab)}-op vocab")
    if not demos:
        print("no demonstrations — widen the DSL (mine.py) or the synthetic pool first"); return 1

    # expand each demo into (state-vector, next-op) training rows
    model = PolicyModel(len(vocab))
    X, Y = [], []
    for feat, seq in demos:
        hist = [0.0] * len(vocab)
        for a in seq:
            state = list(feat) + hist
            X.append(state); Y.append(a)
            hist[a] += 1.0
    X = torch.tensor(X); Y = torch.tensor(Y)
    opt = torch.optim.Adam(model.net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        opt.zero_grad()
        loss = lossf(model.net(X), Y)
        loss.backward(); opt.step()
    print(f"trained on {len(X)} (state,action) rows; final loss {loss.item():.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    model.save(OUT / "policy.pt")
    print(f"saved -> {OUT/'policy.pt'}  (load via PolicyModel(len(vocab)).load(path) and pass policy=model.rank to beam_search)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
