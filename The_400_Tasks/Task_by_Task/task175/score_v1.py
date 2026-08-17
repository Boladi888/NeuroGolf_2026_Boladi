"""task175 v1: static + runtime cost scoring."""
from __future__ import annotations

import os
import sys
import types
import tempfile
import math
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

import json
import numpy as np
import onnx
import onnxruntime as ort

ort.set_default_logger_severity(4)

NG_UTILS_DIR = Path(
    r"<NOGIT_ROOT>\data\competitions\neurogolf-2026\neurogolf_utils"
)
sys.path.insert(0, str(NG_UTILS_DIR))
sys.path.insert(0, r"<REPO_ROOT>\tools")

import neurogolf_utils as nu  # noqa: E402
try:
    import _score_v2 as scv2
    HAS_STATIC = True
except Exception:
    HAS_STATIC = False

PROTO = Path(
    r"<REPO_ROOT>\.claude\worktrees\loving-poincare-5e357f\The_400_Tasks\Task_by_Task\task175\prototypes\task175_v1_closed_form.onnx"
)
LIVE = Path(
    r"<REPO_ROOT>\.claude\worktrees\loving-poincare-5e357f\_tmp_task175_live.onnx"
)
DATA = Path(
    r"<REPO_ROOT>\data\competitions\neurogolf-2026\task175.json"
)


def score_one(path: Path, label: str):
    model = onnx.load(str(path))
    # Sanitize: validator does node.name = output[0]
    for node in model.graph.node:
        node.name = node.output[0] if node.output else node.name

    with tempfile.TemporaryDirectory(prefix="fps_") as tmp:
        tmpdir = Path(tmp)
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.profile_file_prefix = str(tmpdir / "p")
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        sess = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )

        # Run a few examples
        data = json.loads(DATA.read_text())
        pairs = data.get("train", []) + data.get("test", [])
        for p in pairs[:3]:
            bench = nu.convert_to_numpy(p)
            if not bench:
                continue
            sess.run(["output"], {"input": bench["input"]})
        trace_path = sess.end_profiling()

        memory = nu.calculate_memory(model, trace_path)
        params = nu.calculate_params(model)
        cost = memory + (params or 0)
        pts = max(1.0, 25.0 - math.log(max(1.0, cost)))
        print(f"[{label}] runtime memory={memory:,}, params={params:,}, cost={cost:,}, pts={pts:.3f}")

        if HAS_STATIC:
            sc = scv2.score_network_v2(str(path))
            if sc:
                # (cost, memory, params)
                s_cost, s_mem, s_par = sc
                s_pts = max(1.0, 25.0 - math.log(max(1.0, s_cost)))
                print(f"[{label}] static  memory={s_mem:,}, params={s_par:,}, cost={s_cost:,}, pts={s_pts:.3f}")
                print(f"[{label}] runtime - static = {memory - s_mem:+,}")


def main():
    print(f"=== Scoring task175 PROTOTYPE v1 ===")
    score_one(PROTO, "proto")
    print()
    print(f"=== Scoring task175 LIVE ===")
    score_one(LIVE, "live")


if __name__ == "__main__":
    main()
