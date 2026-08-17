"""tasks.py — load a NeuroGolf task's example pairs (read-only, from the repo data dirs).

A task json is {"train":[...], "test":[...], "arc-gen":[...]}, each entry {"input":grid,"output":grid}.
We search/verify against train+test (the visible set); arc-gen is available for harder verification.
Also provides synthetic tasks (identity/transpose/...) so the search substrate can be tested with a
guaranteed-solvable target, independent of any real task's (unknown) rule.
"""
import json
import pathlib
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DATA_DIRS = [
    REPO_ROOT / "data" / "competitions" / "neurogolf-2026",
    REPO_ROOT / "GUI" / "client" / "dist" / "tasks",
]


def _find(tid):
    name = f"task{int(tid):03d}.json"
    for d in _DATA_DIRS:
        p = d / name
        if p.exists():
            return p
    return None


def load_task(tid, split=("train", "test")):
    """Return {'train':[(in,out)...],'test':[...]} as numpy int grids, or None if not found."""
    p = _find(tid)
    if p is None:
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for k in split:
        out[k] = [(np.array(e["input"], dtype=np.int64), np.array(e["output"], dtype=np.int64))
                  for e in d.get(k, [])]
    return out


def all_pairs(task):
    return [pair for split in task.values() for pair in split]


# ---- synthetic tasks (guaranteed-solvable targets to validate the substrate) ----
def _rand_grid(rng, h=None, w=None):
    h = h or int(rng.integers(2, 12)); w = w or int(rng.integers(2, 12))
    return rng.integers(0, 10, size=(h, w), dtype=np.int64)


def _recolor(g, m):
    o = g.copy()
    for a, b in m.items():
        o[g == a] = b           # condition reads original g -> safe for a permutation
    return o


def _mirror2x2(g):
    top = np.concatenate([g, g[:, ::-1]], axis=1)
    return np.concatenate([top, top[::-1, :]], axis=0)


# geometric kinds need a CONSTANT grid size (CoordRemap infers a *static* per-axis index map).
_FIXED_SIZE = {"rot180": (4, 4), "mirror2x2": (3, 4)}


def synthetic(kind="identity", n=6, seed=0):
    rng = np.random.default_rng(seed)
    fn = {
        "identity":  lambda g: g,
        "transpose": lambda g: g.T,
        "colorswap": lambda g: _recolor(g, {1: 2, 2: 1, 3: 5, 5: 3}),
        "rot180":    lambda g: g[::-1, ::-1],
        "mirror2x2": _mirror2x2,
    }[kind]
    hw = _FIXED_SIZE.get(kind)
    pairs = []
    for _ in range(n):
        g = _rand_grid(rng, *hw) if hw else _rand_grid(rng)
        pairs.append((g, np.asarray(fn(g))))
    return {"train": pairs[: max(1, n - 1)], "test": pairs[max(1, n - 1):]}
