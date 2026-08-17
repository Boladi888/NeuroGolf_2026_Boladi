"""task175 v2_fp16 full verify: visible+arc-gen vs LIVE + 9-mode adversarial battery."""
from __future__ import annotations

import os
import sys
import types
import json
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

ipy = types.ModuleType("IPython"); ipy_disp = types.ModuleType("IPython.display"); ipy_disp.display = lambda *a, **kw: None; ipy_disp.FileLink = lambda x: x; ipy.display = ipy_disp
sys.modules.setdefault("IPython", ipy); sys.modules.setdefault("IPython.display", ipy_disp); sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib")); sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))

import numpy as np
import onnxruntime as ort
ort.set_default_logger_severity(4)

WORKTREE = Path(r"<REPO_ROOT>/.claude/worktrees/loving-poincare-5e357f")
PROTO = WORKTREE / "The_400_Tasks" / "Task_by_Task" / "task175" / "prototypes" / "task175_v2_fp16.onnx"
LIVE = WORKTREE / "_tmp_task175_live.onnx"
DATA = Path(r"<REPO_ROOT>/data/competitions/neurogolf-2026/task175.json")

sys.path.insert(0, r"<NOGIT_ROOT>\data\competitions\neurogolf-2026\neurogolf_utils")
import neurogolf_utils as nu  # noqa: E402


def wedge_fill_grid(base, period, holes=None):
    g = np.zeros((21, 21), dtype=np.int32)
    for r in range(21):
        for c in range(21):
            m = min(r, c); d = abs(r - c)
            if d == 0: off = 0
            elif d <= m + 1: off = -1
            else: off = (d - (m + 2)) // (m + 2)
            g[r, c] = ((base - 1 + off) % period) + 1
    if holes:
        for r, c in holes:
            g[r, c] = 0
    return g


def encode(grid):
    a = grid.astype(np.int64); h, w = a.shape
    out = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for c in range(10):
        out[0, c, :h, :w] = (a == c).astype(np.float32)
    return out


def gen_valid_inputs(n, rng):
    out = []
    for _ in range(n):
        period = int(rng.integers(1, 10))
        base = int(rng.integers(1, period + 1))
        n_holes = int(rng.integers(0, 100))
        holes = rng.integers(0, 21, size=(n_holes, 2)).tolist()
        out.append(wedge_fill_grid(base, period, holes=[(int(r), int(c)) for r, c in holes]))
    return out


def color_perm(grid, rng):
    perm = list(range(1, 10))
    rng.shuffle(perm)
    arr = [0] + perm
    return np.array(arr, dtype=grid.dtype)[grid]


def transform_pad(g, mode):
    if mode == "hflip": return np.flip(g, axis=1).copy()
    if mode == "vflip": return np.flip(g, axis=0).copy()
    if mode == "rot90": return np.rot90(g, 1).copy()
    if mode == "rot180": return np.rot90(g, 2).copy()
    if mode == "rot270": return np.rot90(g, 3).copy()
    raise ValueError(mode)


def gen_period_range(rng):
    out = []
    for period in range(2, 10):
        for _ in range(7):
            base = int(rng.integers(1, period + 1))
            n_holes = int(rng.integers(5, 60))
            holes = rng.integers(0, 21, size=(n_holes, 2)).tolist()
            out.append(wedge_fill_grid(base, period, holes=[(int(r), int(c)) for r, c in holes]))
    return out


def gen_base_at_corner(rng):
    out = []
    for _ in range(50):
        period = int(rng.integers(1, 10))
        base = int(rng.integers(1, period + 1))
        diag_holes = []
        keep = rng.integers(2)
        for r in range(21):
            if (keep == 0 and r == 0) or (keep == 1 and r == 20):
                continue
            diag_holes.append((r, r))
        n_off = int(rng.integers(5, 30))
        off = rng.integers(0, 21, size=(n_off, 2)).tolist()
        all_holes = diag_holes + [(int(r), int(c)) for r, c in off if r != c]
        out.append(wedge_fill_grid(base, period, holes=all_holes))
    return out


def gen_diag_with_zeros(rng):
    out = []
    for _ in range(50):
        period = int(rng.integers(1, 10))
        base = int(rng.integers(1, period + 1))
        n_diag = int(rng.integers(1, 21))
        diag_idx = rng.choice(21, size=n_diag, replace=False)
        diag_holes = [(int(i), int(i)) for i in diag_idx]
        n_off = int(rng.integers(0, 50))
        off_raw = rng.integers(0, 21, size=(n_off, 2)).tolist()
        off = [(int(r), int(c)) for r, c in off_raw if r != c]
        out.append(wedge_fill_grid(base, period, holes=diag_holes + off))
    return out


def main():
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    proto = ort.InferenceSession(str(PROTO), so, providers=["CPUExecutionProvider"])
    live = ort.InferenceSession(str(LIVE), so, providers=["CPUExecutionProvider"])

    print("=== Visible+arc-gen byte-exact ===")
    data = json.loads(DATA.read_text())
    pairs = data.get("train", []) + data.get("test", []) + data.get("arc-gen", [])
    match = 0; total = 0
    for p in pairs:
        bench = nu.convert_to_numpy(p)
        if bench is None: continue
        total += 1
        py = (proto.run(["output"], {"input": bench["input"]})[0] > 0).astype(np.float32)
        ly = (live.run(["output"], {"input": bench["input"]})[0] > 0).astype(np.float32)
        if np.array_equal(py, ly): match += 1
    print(f"  proto vs LIVE: {match}/{total}")
    if match != total:
        print(f"  !! FAIL — divergence on {total-match} pair(s)")

    print("\n=== Adversarial battery ===")
    def compare_batch(grids, name):
        m = 0; t = 0
        for g in grids:
            if g.shape != (21, 21): continue
            t += 1
            inp_t = encode(g)
            py = (proto.run(["output"], {"input": inp_t})[0] > 0).astype(np.float32)
            ly = (live.run(["output"], {"input": inp_t})[0] > 0).astype(np.float32)
            if np.array_equal(py, ly): m += 1
        pct = 100.0 * m / max(1, t)
        status = "OK  " if pct >= 95 else "FAIL"
        print(f"  [{status}] {name:<24} {m:>3}/{t} = {pct:>5.1f}%")

    rng = np.random.default_rng(20260512)
    base_inputs = gen_valid_inputs(60, rng)
    for mode in ("hflip", "vflip", "rot90", "rot180", "rot270"):
        transformed = [transform_pad(g, mode) for g in base_inputs[:55]]
        compare_batch(transformed, mode)
    cperm = [color_perm(g, np.random.default_rng(int(rng.integers(0, 2**32)))) for g in base_inputs[:55]]
    compare_batch(cperm, "color_perm")
    compare_batch(gen_period_range(rng), "period_2_to_9")
    compare_batch(gen_base_at_corner(rng), "base_at_corner")
    compare_batch(gen_diag_with_zeros(rng), "diag_with_zeros")


if __name__ == "__main__":
    main()
