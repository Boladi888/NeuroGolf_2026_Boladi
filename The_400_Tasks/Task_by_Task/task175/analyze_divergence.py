"""Classify each adversarial failure into:
  - LIVE_BUG    : proto matches numpy_ref but LIVE diverges (LIVE has bug per spec)
  - PROTO_BUG   : LIVE matches numpy_ref but proto diverges (proto has bug per spec)
  - ALL_DIVERGE : all three differ (input outside valid spec)

Then aggregate per-mode counts. This tells us whether LIVE is buggy on edge cases
(in which case my proto is the CORRECT algorithm and may beat LIVE on hidden) or
whether my proto has a bug (in which case I should fix it).
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

ipy = types.ModuleType("IPython"); ipy_disp = types.ModuleType("IPython.display"); ipy_disp.display = lambda *a, **kw: None; ipy_disp.FileLink = lambda x: x; ipy.display = ipy_disp
sys.modules.setdefault("IPython", ipy); sys.modules.setdefault("IPython.display", ipy_disp); sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib")); sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))

import numpy as np
import onnxruntime as ort
ort.set_default_logger_severity(4)

WORKTREE = Path(r"<REPO_ROOT>/.claude/worktrees/loving-poincare-5e357f")
PROTO = WORKTREE / "The_400_Tasks" / "Task_by_Task" / "task175" / "prototypes" / "task175_v1_closed_form.onnx"
LIVE = WORKTREE / "_tmp_task175_live.onnx"

sys.path.insert(0, str(WORKTREE / "The_400_Tasks" / "Task_by_Task" / "task175" / "prototypes"))
from task175_numpy_reference import predict_task175


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


def decode_tensor(t):
    g = np.zeros((21, 21), dtype=int)
    for c in range(10):
        g[t[0, c, :21, :21] > 0.5] = c
    return g


def gen_period_range(rng):
    out = []
    for period in range(2, 10):
        for _ in range(7):
            base = int(rng.integers(1, period + 1))
            n_holes = int(rng.integers(5, 60))
            holes = rng.integers(0, 21, size=(n_holes, 2)).tolist()
            g = wedge_fill_grid(base, period, holes=[(int(r), int(c)) for r, c in holes])
            out.append(g)
    return out


def gen_base_at_corner(rng):
    out = []
    for _ in range(50):
        period = int(rng.integers(1, 10))
        base = int(rng.integers(1, period + 1))
        diag_holes = []
        keep_corner = rng.integers(2)
        for r in range(21):
            if (keep_corner == 0 and r == 0) or (keep_corner == 1 and r == 20):
                continue
            diag_holes.append((r, r))
        n_off = int(rng.integers(5, 30))
        off_holes = rng.integers(0, 21, size=(n_off, 2)).tolist()
        all_holes = diag_holes + [(int(r), int(c)) for r, c in off_holes if r != c]
        g = wedge_fill_grid(base, period, holes=all_holes)
        out.append(g)
    return out


def gen_diag_with_zeros(rng):
    out = []
    for _ in range(50):
        period = int(rng.integers(1, 10))
        base = int(rng.integers(1, period + 1))
        n_diag_holes = int(rng.integers(1, 21))
        diag_idx = rng.choice(21, size=n_diag_holes, replace=False)
        diag_holes = [(int(i), int(i)) for i in diag_idx]
        n_off = int(rng.integers(0, 50))
        off_holes_raw = rng.integers(0, 21, size=(n_off, 2)).tolist()
        off_holes = [(int(r), int(c)) for r, c in off_holes_raw if r != c]
        out.append(wedge_fill_grid(base, period, holes=diag_holes + off_holes))
    return out


def main():
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    proto = ort.InferenceSession(str(PROTO), so, providers=["CPUExecutionProvider"])
    live = ort.InferenceSession(str(LIVE), so, providers=["CPUExecutionProvider"])

    def analyze_batch(grids, mode_name):
        live_bug = 0
        proto_bug = 0
        all_diverge = 0
        agree = 0
        total = 0
        live_bug_cells = 0
        proto_bug_cells = 0
        all_diverge_cells = 0
        for g in grids:
            if g.shape != (21, 21):
                continue
            total += 1
            inp_t = encode(g)
            py = (proto.run(["output"], {"input": inp_t})[0] > 0).astype(np.float32)
            ly = (live.run(["output"], {"input": inp_t})[0] > 0).astype(np.float32)
            pg = decode_tensor(py)
            lg = decode_tensor(ly)
            ng = np.array(predict_task175(g.tolist()), dtype=int)
            diff_pl = int((pg != lg).sum())
            diff_pn = int((pg != ng).sum())
            diff_ln = int((lg != ng).sum())
            if diff_pl == 0:
                agree += 1
            elif diff_pn == 0 and diff_ln > 0:
                live_bug += 1
                live_bug_cells += diff_pl
            elif diff_ln == 0 and diff_pn > 0:
                proto_bug += 1
                proto_bug_cells += diff_pl
            else:
                all_diverge += 1
                all_diverge_cells += diff_pl
        print(f"\n[{mode_name}] total={total}")
        print(f"  proto==LIVE                : {agree:>3}/{total}")
        print(f"  LIVE_BUG (proto==numpy)    : {live_bug:>3}/{total}  ({live_bug_cells} cells diff)")
        print(f"  PROTO_BUG (LIVE==numpy)    : {proto_bug:>3}/{total}  ({proto_bug_cells} cells diff)")
        print(f"  ALL_DIVERGE                : {all_diverge:>3}/{total}  ({all_diverge_cells} cells diff)")

    rng = np.random.default_rng(20260512)
    # consume the gen_valid_inputs RNG state (60 calls of period/base/n_holes/holes)
    for _ in range(60):
        period = int(rng.integers(1, 10))
        base = int(rng.integers(1, period + 1))
        n_holes = int(rng.integers(0, 100))
        _ = rng.integers(0, 21, size=(n_holes, 2)).tolist()
    # consume color_perm RNG state (55 calls)
    for _ in range(55):
        _ = int(rng.integers(0, 2**32))
    # now analyze the three modes
    analyze_batch(gen_period_range(rng), "period_2_to_9")
    analyze_batch(gen_base_at_corner(rng), "base_at_corner")
    analyze_batch(gen_diag_with_zeros(rng), "diag_with_zeros")


if __name__ == "__main__":
    main()
