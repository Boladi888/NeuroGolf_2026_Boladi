"""task175: compare numpy_reference output vs LIVE task175.onnx output, byte-exact.

Per directive 20260512T204031Z: FIRST ACTION is to determine whether the numpy_ref
algorithm matches LIVE on every visible+arc-gen pair. If yes -> can potentially ship
the closed-form (option B). If no -> must mirror LIVE structure (option A).

Uses validator's `neurogolf_utils.convert_to_numpy` (zero-padding) for encoding,
and `(x > 0)` for decode/compare (per lessons.md 2026-05-12 + directive notes).
"""
from __future__ import annotations

import json
import os
import sys
import types
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
DATA = Path(r"<REPO_ROOT>/data/competitions/neurogolf-2026/task175.json")
LIVE_ONNX = WORKTREE / "_tmp_task175_live.onnx"

sys.path.insert(0, r"<NOGIT_ROOT>\data\competitions\neurogolf-2026\neurogolf_utils")
import neurogolf_utils as nu  # noqa: E402

sys.path.insert(0, str(WORKTREE / "The_400_Tasks" / "Task_by_Task" / "task175" / "prototypes"))
from task175_numpy_reference import predict_task175  # noqa: E402


def grid_to_validator_tensor(grid):
    """Convert a grid (list of int rows) -> (1,10,30,30) one-hot tensor with zero-pad."""
    a = np.array(grid, dtype=np.int64)
    h, w = a.shape
    out = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for c in range(10):
        out[0, c, :h, :w] = (a == c).astype(np.float32)
    return out


def main():
    if not LIVE_ONNX.exists():
        print(f"LIVE onnx missing at {LIVE_ONNX}")
        return 2

    data = json.loads(DATA.read_text())
    pairs = data.get("train", []) + data.get("test", []) + data.get("arc-gen", [])
    print(f"Loaded {len(pairs)} pairs from task175.json")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(LIVE_ONNX), so, providers=["CPUExecutionProvider"])

    correct_numpy_vs_expected = 0
    correct_live_vs_expected = 0
    match_numpy_vs_live = 0
    first_diverge = None
    diverging_indices = []

    for i, p in enumerate(pairs):
        bench = nu.convert_to_numpy(p)
        if bench is None:
            continue
        expected_tensor = bench["output"]  # already one-hot
        inp_tensor = bench["input"]

        # numpy reference -> validator tensor
        pred_numpy = np.array(predict_task175(p["input"]), dtype=np.int64)
        numpy_tensor = np.zeros((1, 10, 30, 30), dtype=np.float32)
        h, w = pred_numpy.shape
        for c in range(10):
            numpy_tensor[0, c, :h, :w] = (pred_numpy == c).astype(np.float32)

        # LIVE
        y_live = sess.run(None, {"input": inp_tensor})[0]
        live_bin = (y_live > 0).astype(np.float32)

        if np.array_equal(numpy_tensor, expected_tensor):
            correct_numpy_vs_expected += 1
        if np.array_equal(live_bin, expected_tensor):
            correct_live_vs_expected += 1
        if np.array_equal(numpy_tensor, live_bin):
            match_numpy_vs_live += 1
        else:
            diverging_indices.append(i)
            if first_diverge is None:
                first_diverge = (i, p["input"], p["output"], pred_numpy, live_bin)

    total = sum(1 for p in pairs if nu.convert_to_numpy(p) is not None)
    print(f"\nResults (task175, {total} pairs):")
    print(f"  numpy_ref  vs expected: {correct_numpy_vs_expected:>4}/{total}")
    print(f"  LIVE       vs expected: {correct_live_vs_expected:>4}/{total}")
    print(f"  numpy_ref  vs LIVE   : {match_numpy_vs_live:>4}/{total}  <-- KEY")

    if match_numpy_vs_live == total:
        print("\n  OK numpy_ref tensor output matches LIVE on all visible+arc-gen pairs.")
        print("     Option (B) closed-form path is VIABLE if adversarial pass.")
    else:
        print(f"\n  !!  numpy_ref DIVERGES from LIVE on {len(diverging_indices)} pair(s)")
        print(f"      Diverging indices (first 20): {diverging_indices[:20]}")
        if first_diverge is not None:
            i, inp, expected, pred_n, live_bin = first_diverge
            inp_arr = np.array(inp)
            # decode live_bin to grid for human view (channel where ==1)
            h, w = len(expected), len(expected[0])
            pred_l = np.zeros((h, w), dtype=int)
            for c in range(10):
                mask = live_bin[0, c, :h, :w] > 0.5
                pred_l[mask] = c
            print(f"\n  First diverging pair: index {i} (shape {inp_arr.shape})")
            diff = (pred_n != pred_l)
            print(f"  Diverges on {diff.sum()}/{inp_arr.size} cells")
            print(f"  Input (first 8 rows):")
            for row in inp_arr.tolist()[:8]:
                print("    " + " ".join(f"{c:>2}" for c in row))
            print(f"  Expected (first 8 rows):")
            for row in np.array(expected).tolist()[:8]:
                print("    " + " ".join(f"{c:>2}" for c in row))
            print(f"  numpy_ref output (first 8 rows):")
            for row in pred_n.tolist()[:8]:
                print("    " + " ".join(f"{c:>2}" for c in row))
            print(f"  LIVE output (first 8 rows):")
            for row in pred_l.tolist()[:8]:
                print("    " + " ".join(f"{c:>2}" for c in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
