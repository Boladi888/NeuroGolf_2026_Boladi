"""profile_real_cost.py — the WORKER_PROTOCOL gate-5 REAL-INPUT PROFILE (mandatory 2026-06-07).

`_score_v2.score_network_v2` profiles ONE dummy (empty-grid) inference; any
runtime-inflating surface (dynamic shapes, data-dependent slices) is invisible
to it (task225 credited 2.9x its local price; task360 v15 measured 358 -> 6,442).
This tool replays the OFFICIAL inputs (train + test, optionally arc-gen) through
ONE profiling session and calls the official `score_network` on the combined
trace — `calculate_memory` takes max(static, observed) per tensor across ALL
events, so the result IS the credited cost under real inputs.

Usage:
    .venv/Scripts/python tools/agent_helpers/profile_real_cost.py --task N --cand path.onnx [--arcgen K] [--max-n M]

Prints `static_cost` (dummy, _score_v2) vs `profile_cost` (real inputs) and
flags gaps >5% per the protocol (record profile_cost in the staging manifest).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys
import time

import numpy as np
import onnx
import onnxruntime

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_DATA_DIR = pathlib.Path(os.environ.get(
    "NEUROGOLF_DATA_DIR", str(_ROOT / "data" / "competitions" / "neurogolf-2026")))
sys.path.insert(0, str(_DATA_DIR / "neurogolf_utils"))

# _score_v2 stubs notebook deps and exposes score_network_v2 for the static baseline.
_spec = importlib.util.spec_from_file_location("score_v2", _ROOT / "tools" / "_score_v2.py")
_sv2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sv2)
import neurogolf_utils as nu  # noqa: E402  (importable after _score_v2 set sys.path)


def encode(grid: list[list[int]]) -> np.ndarray:
    x = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            x[0][color][r][c] = 1.0
    return x


def profile_cost(task: int, cand: str, arcgen: int = 0, max_n: int = 0) -> dict:
    data = json.loads((_DATA_DIR / f"task{task:03d}.json").read_text())
    grids = [p["input"] for p in data.get("train", [])] + [p["input"] for p in data.get("test", [])]
    if arcgen:
        grids += [p["input"] for p in data.get("arc-gen", [])[:arcgen]]
    if max_n and len(grids) > max_n:
        # keep size diversity: sort by (H, W), take an even stride sample
        grids.sort(key=lambda g: (len(g), len(g[0])))
        step = len(grids) / max_n
        grids = [grids[int(i * step)] for i in range(max_n)]

    model = onnx.load(cand)
    sanitized = nu.sanitize_model(model)
    if sanitized is None:
        return {"error": "sanitize_model returned None"}
    options = onnxruntime.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    prof_dir = _ROOT / "submissions"
    prof_dir.mkdir(exist_ok=True)
    options.profile_file_prefix = str(
        (prof_dir / f"_ort_profile_real_{os.getpid()}_{time.time_ns()}").resolve())
    session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
    for g in grids:
        session.run(None, {"input": encode(g)})
    trace_path = session.end_profiling()
    try:
        memory, params = nu.score_network(sanitized, trace_path)
    finally:
        try:
            os.remove(trace_path)
        except OSError:
            pass
    if memory is None:
        return {"error": "score_network returned None on real-input trace"}
    static = _sv2.score_network_v2(cand)
    static_cost = (static[1] + static[2]) if static else None
    return {
        "task": task, "n_inputs": len(grids),
        "static_cost": static_cost, "profile_memory": memory, "params": params,
        "profile_cost": memory + params,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--arcgen", type=int, default=0, help="also replay first K arc-gen inputs")
    ap.add_argument("--max-n", type=int, default=0, help="cap replayed inputs (size-diverse sample)")
    args = ap.parse_args()
    res = profile_cost(args.task, args.cand, args.arcgen, args.max_n)
    if "error" in res:
        print(f"FAIL: {res['error']}")
        return 1
    print(f"task{args.task:03d} {pathlib.Path(args.cand).name}: "
          f"static={res['static_cost']} profile={res['profile_cost']} "
          f"(memory {res['profile_memory']} + params {res['params']}; {res['n_inputs']} real inputs)")
    if res["static_cost"] and res["profile_cost"] > res["static_cost"] * 1.05:
        print(f"WARNING: profile_cost {res['profile_cost']} exceeds static {res['static_cost']} "
              f"by >5% — credit-probe class, NOT an exact ship (WORKER_PROTOCOL gate 5).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
