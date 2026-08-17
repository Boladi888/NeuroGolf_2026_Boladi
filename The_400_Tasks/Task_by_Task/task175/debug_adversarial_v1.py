"""Debug first divergent adversarial case for period_2_to_9 + diag_with_zeros.

Compare proto vs LIVE vs numpy_ref on the same input. If proto matches numpy_ref
but not LIVE: algorithm divergence (LIVE has different behavior on this input).
If proto diverges from numpy_ref too: proto has a bug to fix.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

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
            m = min(r, c)
            d = abs(r - c)
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
    """Convert a (1,10,30,30) tensor to a 21x21 grid of color indices."""
    g = np.zeros((21, 21), dtype=int)
    for c in range(10):
        g[t[0, c, :21, :21] > 0.5] = c
    return g


def gen_period_range(rng):
    inputs = []
    for period in range(2, 10):
        for _ in range(7):
            base = int(rng.integers(1, period + 1))
            n_holes = int(rng.integers(5, 60))
            holes = rng.integers(0, 21, size=(n_holes, 2)).tolist()
            inputs.append((base, period, [(int(r), int(c)) for r, c in holes]))
    return inputs


def main():
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    proto = ort.InferenceSession(str(PROTO), so, providers=["CPUExecutionProvider"])
    live = ort.InferenceSession(str(LIVE), so, providers=["CPUExecutionProvider"])

    # Reproduce adversarial_v1.py's RNG sequence and find first divergent period_2_to_9 input.
    rng = np.random.default_rng(20260512)
    _ = [rng.integers(1, 10) for _ in range(60)]  # consume base_inputs RNG (approx)
    # Actually, the exact RNG state is hard to replay. Let me just regenerate inputs in same order
    # using a fresh RNG seeded the same way as adversarial_v1.py does AFTER gen_valid_inputs.

    # Replay adversarial_v1.py exactly:
    rng = np.random.default_rng(20260512)
    # base_inputs = gen_valid_inputs(60, rng) — consumes RNG
    def gen_valid_inputs(n, rng):
        out = []
        for _ in range(n):
            period = int(rng.integers(1, 10))
            base = int(rng.integers(1, period + 1))
            n_holes = int(rng.integers(0, 100))
            hole_positions = rng.integers(0, 21, size=(n_holes, 2)).tolist()
            g = wedge_fill_grid(base, period, holes=[(int(r), int(c)) for r, c in hole_positions])
            out.append(g)
        return out

    base_inputs = gen_valid_inputs(60, rng)
    # Then 5 transforms consume nothing extra. Then color_perm consumes one rng.integers(0, 2**32) per input × 55 inputs
    for _ in range(55):
        _ = int(rng.integers(0, 2**32))
    # Then gen_period_range
    period_inputs = gen_period_range(rng)

    for i, (base, period, holes) in enumerate(period_inputs):
        g = wedge_fill_grid(base, period, holes)
        if g.shape != (21, 21):
            continue
        inp_t = encode(g)
        py = (proto.run(["output"], {"input": inp_t})[0] > 0).astype(np.float32)
        ly = (live.run(["output"], {"input": inp_t})[0] > 0).astype(np.float32)
        pg = decode_tensor(py)
        lg = decode_tensor(ly)
        ng = np.array(predict_task175(g.tolist()), dtype=int)
        diff_pl = pg != lg
        diff_pn = pg != ng
        diff_ln = lg != ng
        if diff_pl.sum() > 0:
            print(f"\n=== period_2_to_9 input #{i} (base={base}, period={period}, n_holes={len(holes)}) ===")
            print(f"  cells differing: proto-vs-LIVE={diff_pl.sum()}, proto-vs-numpy={diff_pn.sum()}, LIVE-vs-numpy={diff_ln.sum()}")
            print(f"  Input grid:")
            for row in g.tolist():
                print("    " + " ".join(f"{x:>2}" for x in row))
            print(f"  numpy_ref output:")
            for row in ng.tolist():
                print("    " + " ".join(f"{x:>2}" for x in row))
            print(f"  Prototype output:")
            for row in pg.tolist():
                print("    " + " ".join(f"{x:>2}" for x in row))
            print(f"  LIVE output:")
            for row in lg.tolist():
                print("    " + " ".join(f"{x:>2}" for x in row))
            print(f"  Differing cells (proto vs LIVE):")
            for r in range(21):
                for c in range(21):
                    if pg[r, c] != lg[r, c]:
                        print(f"    ({r:>2},{c:>2}): proto={pg[r,c]}, LIVE={lg[r,c]}, numpy={ng[r,c]}, input={g[r,c]}")
            return


if __name__ == "__main__":
    main()
