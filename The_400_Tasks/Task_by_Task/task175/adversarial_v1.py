"""task175 v1: adversarial battery, prototype vs LIVE byte-exact.

Modes (per directive 20260512T204031Z, threshold 95%, >=50 inputs/mode):
  hflip, vflip, rot90, rot180, rot270, color_perm,
  period_2_to_9, base_at_corner, diag_with_zeros

Compare tensor space (validator's (x > 0).astype) byte-exact.
"""
from __future__ import annotations

import os
import sys
import types
import json
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

ipy = types.ModuleType("IPython")
ipy_disp = types.ModuleType("IPython.display")
ipy_disp.display = lambda *a, **kw: None
ipy_disp.FileLink = lambda x: x
ipy.display = ipy_disp
sys.modules.setdefault("IPython", ipy)
sys.modules.setdefault("IPython.display", ipy_disp)
sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))

import numpy as np
import onnxruntime as ort

ort.set_default_logger_severity(4)

WORKTREE = Path(r"<REPO_ROOT>/.claude/worktrees/loving-poincare-5e357f")
PROTO = WORKTREE / "The_400_Tasks" / "Task_by_Task" / "task175" / "prototypes" / "task175_v1_closed_form.onnx"
LIVE = WORKTREE / "_tmp_task175_live.onnx"


def wedge_fill_grid(base, period, holes=None):
    """Compute full 21x21 wedge-fill grid, optionally punch holes at given (r,c) indices."""
    g = np.zeros((21, 21), dtype=np.int32)
    for r in range(21):
        for c in range(21):
            m = min(r, c)
            d = abs(r - c)
            if d == 0:
                off = 0
            elif d <= m + 1:
                off = -1
            else:
                off = (d - (m + 2)) // (m + 2)
            g[r, c] = ((base - 1 + off) % period) + 1
    if holes is not None:
        for r, c in holes:
            g[r, c] = 0
    return g


def encode(grid):
    """grid: (H,W) int32; returns (1,10,30,30) fp32 one-hot zero-padded."""
    a = grid.astype(np.int64)
    h, w = a.shape
    out = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for c in range(10):
        out[0, c, :h, :w] = (a == c).astype(np.float32)
    return out


def gen_valid_inputs(n, rng):
    """Generate `n` valid wedge-fill 21x21 inputs with random (base, period, hole pattern)."""
    inputs = []
    for _ in range(n):
        period = int(rng.integers(1, 10))  # 1..9
        base = int(rng.integers(1, period + 1))  # 1..period
        n_holes = int(rng.integers(0, 100))
        hole_positions = rng.integers(0, 21, size=(n_holes, 2)).tolist()
        g = wedge_fill_grid(base, period, holes=[(int(r), int(c)) for r, c in hole_positions])
        inputs.append(g)
    return inputs


def gen_period_range(rng):
    """Generate 50+ inputs spanning periods 2..9, ~6 each."""
    inputs = []
    for period in range(2, 10):  # 2..9
        for _ in range(7):
            base = int(rng.integers(1, period + 1))
            n_holes = int(rng.integers(5, 60))
            holes = rng.integers(0, 21, size=(n_holes, 2)).tolist()
            g = wedge_fill_grid(base, period, holes=[(int(r), int(c)) for r, c in holes])
            inputs.append(g)
    return inputs


def gen_base_at_corner(rng):
    """Ensure base color is the (0,0) or (20,20) cell."""
    inputs = []
    for _ in range(50):
        period = int(rng.integers(1, 10))
        base = int(rng.integers(1, period + 1))
        # Ensure base appears in (0,0) — wedge-fill naturally puts base on diagonal.
        # But we want to STRESS test: punch holes everywhere on diag EXCEPT (0,0) or (20,20)
        diag_holes = []
        keep_corner = rng.integers(2)  # 0: keep (0,0); 1: keep (20,20)
        for r in range(21):
            if (keep_corner == 0 and r == 0) or (keep_corner == 1 and r == 20):
                continue
            diag_holes.append((r, r))
        # Also punch some off-diag holes for variety
        n_off = int(rng.integers(5, 30))
        off_holes = rng.integers(0, 21, size=(n_off, 2)).tolist()
        all_holes = diag_holes + [(int(r), int(c)) for r, c in off_holes if r != c]
        g = wedge_fill_grid(base, period, holes=all_holes)
        inputs.append(g)
    return inputs


def gen_diag_with_zeros(rng):
    """Main diagonal has some zero cells (but at least one nonzero so fallback may or may not trigger)."""
    inputs = []
    for _ in range(50):
        period = int(rng.integers(1, 10))
        base = int(rng.integers(1, period + 1))
        n_diag_holes = int(rng.integers(1, 21))  # punch 1..20 diag cells
        diag_idx = rng.choice(21, size=n_diag_holes, replace=False)
        diag_holes = [(int(i), int(i)) for i in diag_idx]
        # Also some non-diag holes
        n_off = int(rng.integers(0, 50))
        off_holes_raw = rng.integers(0, 21, size=(n_off, 2)).tolist()
        off_holes = [(int(r), int(c)) for r, c in off_holes_raw if r != c]
        g = wedge_fill_grid(base, period, holes=diag_holes + off_holes)
        inputs.append(g)
    return inputs


def color_perm(grid, rng):
    """Permute nonzero colors using a random permutation of 1..9."""
    perm = list(range(1, 10))
    rng.shuffle(perm)
    perm_arr = [0] + perm  # color 0 stays 0
    perm_lookup = np.array(perm_arr, dtype=grid.dtype)
    return perm_lookup[grid]


def transform_pad(g, mode):
    """Apply transformation. g is 21x21."""
    if mode == "hflip":
        return np.flip(g, axis=1).copy()
    if mode == "vflip":
        return np.flip(g, axis=0).copy()
    if mode == "rot90":
        return np.rot90(g, 1).copy()
    if mode == "rot180":
        return np.rot90(g, 2).copy()
    if mode == "rot270":
        return np.rot90(g, 3).copy()
    raise ValueError(mode)


def main():
    rng = np.random.default_rng(20260512)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    proto = ort.InferenceSession(str(PROTO), so, providers=["CPUExecutionProvider"])
    live = ort.InferenceSession(str(LIVE), so, providers=["CPUExecutionProvider"])

    def compare_batch(grids, mode_name):
        match = 0
        total = 0
        first_diff = None
        for i, g in enumerate(grids):
            if g.shape != (21, 21):
                continue
            inp_t = encode(g)
            try:
                py = (proto.run(["output"], {"input": inp_t})[0] > 0).astype(np.float32)
                ly = (live.run(["output"], {"input": inp_t})[0] > 0).astype(np.float32)
            except Exception as e:
                print(f"  [{mode_name}] runtime error on input {i}: {e}")
                continue
            total += 1
            if np.array_equal(py, ly):
                match += 1
            elif first_diff is None:
                first_diff = (i, g, py, ly)
        pct = 100.0 * match / max(1, total)
        status = "OK " if pct >= 95 else "FAIL"
        print(f"  [{status}] {mode_name:<24} {match:>3}/{total} = {pct:>5.1f}%")
        if first_diff and pct < 100:
            i, g, py, ly = first_diff
            ng = np.zeros((21, 21), dtype=int)
            nl = np.zeros((21, 21), dtype=int)
            for c in range(10):
                ng[py[0, c, :21, :21] > 0.5] = c
                nl[ly[0, c, :21, :21] > 0.5] = c
            print(f"    First-diff input #{i}: proto-vs-live cells differing = {(ng!=nl).sum()}")

    # Base set of valid wedge-fill inputs
    base_inputs = gen_valid_inputs(60, rng)

    print(f"=== Adversarial battery: task175 v1 vs LIVE ===\n")

    # Mode 1-5: flip/rotate the base inputs
    for mode in ("hflip", "vflip", "rot90", "rot180", "rot270"):
        transformed = [transform_pad(g, mode) for g in base_inputs[:55]]
        compare_batch(transformed, mode)

    # Mode 6: color_perm
    cperm_inputs = [color_perm(g, np.random.default_rng(int(rng.integers(0, 2**32)))) for g in base_inputs[:55]]
    compare_batch(cperm_inputs, "color_perm")

    # Mode 7: period_2_to_9
    compare_batch(gen_period_range(rng), "period_2_to_9")

    # Mode 8: base_at_corner
    compare_batch(gen_base_at_corner(rng), "base_at_corner")

    # Mode 9: diag_with_zeros
    compare_batch(gen_diag_with_zeros(rng), "diag_with_zeros")


if __name__ == "__main__":
    main()
