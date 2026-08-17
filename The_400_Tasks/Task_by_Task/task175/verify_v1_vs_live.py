"""task175 v1: verify prototype output matches LIVE byte-exact (validator-tensor space).

Uses neurogolf_utils.convert_to_numpy for encoding and (x > 0) for decode-compare.
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
PROTO_ONNX = WORKTREE / "The_400_Tasks" / "Task_by_Task" / "task175" / "prototypes" / "task175_v1_closed_form.onnx"

sys.path.insert(0, r"<NOGIT_ROOT>\data\competitions\neurogolf-2026\neurogolf_utils")
import neurogolf_utils as nu  # noqa: E402


def main():
    data = json.loads(DATA.read_text())
    pairs = data.get("train", []) + data.get("test", []) + data.get("arc-gen", [])

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    live = ort.InferenceSession(str(LIVE_ONNX), so, providers=["CPUExecutionProvider"])
    proto = ort.InferenceSession(str(PROTO_ONNX), so, providers=["CPUExecutionProvider"])

    proto_vs_expected = 0
    live_vs_expected = 0
    proto_vs_live = 0
    diverging = []
    total = 0
    first_diverge_info = None

    for i, p in enumerate(pairs):
        bench = nu.convert_to_numpy(p)
        if bench is None:
            continue
        total += 1
        inp_t = bench["input"]
        out_t = bench["output"]
        py = (proto.run(["output"], {"input": inp_t})[0] > 0).astype(np.float32)
        ly = (live.run(["output"], {"input": inp_t})[0] > 0).astype(np.float32)
        if np.array_equal(py, out_t):
            proto_vs_expected += 1
        if np.array_equal(ly, out_t):
            live_vs_expected += 1
        if np.array_equal(py, ly):
            proto_vs_live += 1
        else:
            diverging.append(i)
            if first_diverge_info is None:
                first_diverge_info = (i, p["input"], py, ly)

    print(f"\nResults (task175, {total} pairs):")
    print(f"  Prototype  vs expected: {proto_vs_expected:>4}/{total}")
    print(f"  LIVE       vs expected: {live_vs_expected:>4}/{total}")
    print(f"  Prototype  vs LIVE   : {proto_vs_live:>4}/{total}  <-- KEY")

    if proto_vs_live == total:
        print("\n  OK Prototype matches LIVE byte-exact on all visible+arc-gen pairs.")
        return 0
    else:
        print(f"\n  !!  Prototype DIVERGES from LIVE on {len(diverging)} pair(s)")
        print(f"      Diverging indices (first 20): {diverging[:20]}")
        if first_diverge_info:
            i, inp, py, ly = first_diverge_info
            print(f"\n  First diverging pair: index {i}")
            inp_arr = np.array(inp)
            # decode py and ly to grid
            h, w = inp_arr.shape
            pg = np.zeros((h, w), dtype=int)
            lg = np.zeros((h, w), dtype=int)
            for c in range(10):
                pg[py[0, c, :h, :w] > 0.5] = c
                lg[ly[0, c, :h, :w] > 0.5] = c
            print(f"  Input (first 8 rows):")
            for row in inp_arr.tolist()[:8]:
                print("    " + " ".join(f"{x:>2}" for x in row))
            print(f"  Prototype output (first 8 rows):")
            for row in pg.tolist()[:8]:
                print("    " + " ".join(f"{x:>2}" for x in row))
            print(f"  LIVE output (first 8 rows):")
            for row in lg.tolist()[:8]:
                print("    " + " ".join(f"{x:>2}" for x in row))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
