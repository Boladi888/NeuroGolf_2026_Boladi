"""task364 initial diagnostic.

Per REASSIGN directive 20260512T212530Z. Run:
  1. numpy_ref vs expected on all visible+arc-gen pairs (claim: 263/266)
  2. LIVE vs expected (test for broken-LIVE situation)
  3. numpy_ref vs LIVE (alignment check)

Identify the 3 numpy_ref failures and classify them.
"""
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
LIVE_ONNX = WORKTREE / "_tmp_task364_live.onnx"
DATA = Path(r"<REPO_ROOT>/data/competitions/neurogolf-2026/task364.json")

sys.path.insert(0, r"<NOGIT_ROOT>\data\competitions\neurogolf-2026\neurogolf_utils")
import neurogolf_utils as nu  # noqa: E402

sys.path.insert(0, str(WORKTREE / "The_400_Tasks" / "Task_by_Task" / "task364" / "prototypes"))
from task364_numpy_reference import predict_task364  # noqa: E402


def encode_tensor(grid):
    a = np.array(grid, dtype=np.int64); h, w = a.shape
    out = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for c in range(10):
        out[0, c, :h, :w] = (a == c).astype(np.float32)
    return out


def decode_tensor_to_grid(t, h, w):
    g = np.zeros((h, w), dtype=int)
    for c in range(10):
        g[t[0, c, :h, :w] > 0.5] = c
    return g


def main():
    data = json.loads(DATA.read_text())
    pairs_train = data.get("train", [])
    pairs_test = data.get("test", [])
    pairs_arcgen = data.get("arc-gen", [])
    pairs_all = pairs_train + pairs_test + pairs_arcgen
    print(f"Pairs: train={len(pairs_train)} test={len(pairs_test)} arc-gen={len(pairs_arcgen)} total={len(pairs_all)}")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(LIVE_ONNX), so, providers=["CPUExecutionProvider"])

    numpy_ok = 0
    live_ok = 0
    numpy_eq_live = 0
    total = 0
    failures = []

    for i, p in enumerate(pairs_all):
        inp_grid = p["input"]
        expected_grid = p["output"]
        inp_arr = np.array(inp_grid)
        exp_arr = np.array(expected_grid)
        h, w = exp_arr.shape

        # numpy_ref
        try:
            pred_numpy = predict_task364(inp_grid)
            pred_numpy_arr = np.array(pred_numpy)
            numpy_eq_expected = (pred_numpy == expected_grid)
        except Exception as e:
            pred_numpy_arr = None
            numpy_eq_expected = False

        # LIVE
        bench = nu.convert_to_numpy(p)
        if bench is None:
            continue
        try:
            y_live = sess.run(["output"], {"input": bench["input"]})[0]
            live_bin = (y_live > 0).astype(np.float32)
            # Decode to grid for comparison vs expected
            pred_live_arr = decode_tensor_to_grid(live_bin, h, w)
            live_bin_expected = bench["output"]
            live_eq_expected = np.array_equal(live_bin, live_bin_expected)
        except Exception as e:
            pred_live_arr = None
            live_eq_expected = False

        # numpy vs LIVE
        if pred_numpy_arr is not None and pred_live_arr is not None:
            n_eq_l = np.array_equal(pred_numpy_arr, pred_live_arr)
        else:
            n_eq_l = False

        total += 1
        if numpy_eq_expected: numpy_ok += 1
        if live_eq_expected: live_ok += 1
        if n_eq_l: numpy_eq_live += 1

        if not numpy_eq_expected:
            split = (
                "train" if i < len(pairs_train) else
                ("test" if i < len(pairs_train) + len(pairs_test) else "arc-gen")
            )
            failures.append({
                "idx": i, "split": split, "shape": inp_arr.shape,
                "numpy_correct": False,
                "live_correct": live_eq_expected,
                "numpy_eq_live": n_eq_l,
                "n_cells_diff_numpy_vs_expected": int((pred_numpy_arr != exp_arr).sum()) if pred_numpy_arr is not None else None,
            })

    print(f"\nResults (task364, {total} pairs):")
    print(f"  numpy_ref vs expected: {numpy_ok:>4}/{total}")
    print(f"  LIVE      vs expected: {live_ok:>4}/{total}")
    print(f"  numpy_ref vs LIVE   : {numpy_eq_live:>4}/{total}")

    print(f"\nnumpy_ref failures: {len(failures)}")
    for f in failures[:5]:
        print(f"  idx={f['idx']} split={f['split']} shape={f['shape']} numpy_correct=F live_correct={f['live_correct']} numpy_eq_live={f['numpy_eq_live']} cells_diff_numpy={f['n_cells_diff_numpy_vs_expected']}")

    # Dump first failure for inspection
    if failures:
        f = failures[0]
        p = pairs_all[f["idx"]]
        inp = np.array(p["input"])
        exp = np.array(p["output"])
        pred_n = np.array(predict_task364(p["input"]))
        bench = nu.convert_to_numpy(p)
        y_l = sess.run(["output"], {"input": bench["input"]})[0]
        live_bin = (y_l > 0).astype(np.float32)
        h, w = exp.shape
        pred_l = decode_tensor_to_grid(live_bin, h, w)
        print(f"\n=== First failure detail (idx={f['idx']}, split={f['split']}, shape={inp.shape}) ===")
        print("Input:")
        for row in inp.tolist()[:8]:
            print("  " + " ".join(f"{x:>2}" for x in row))
        print("Expected:")
        for row in exp.tolist()[:8]:
            print("  " + " ".join(f"{x:>2}" for x in row))
        print("numpy_ref output:")
        for row in pred_n.tolist()[:8]:
            print("  " + " ".join(f"{x:>2}" for x in row))
        print("LIVE output:")
        for row in pred_l.tolist()[:8]:
            print("  " + " ".join(f"{x:>2}" for x in row))


if __name__ == "__main__":
    main()
