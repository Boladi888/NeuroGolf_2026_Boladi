"""One-shot audit: verify spread_scan claims against real task data.

For each top-N task in the scan report:
  1. Extract original ONNX from submission.zip
  2. Polish via polish_candidate (which re-applies every pattern)
  3. Compare cost via score_network_v2 invoked directly (not subprocessed)
  4. Verify byte-exact output on REAL task JSON pairs (train+test, encoded)
  5. Check IR version + opset of polished ONNX
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

os.environ.setdefault("ORT_LOGGING_LEVEL", "3")

import numpy as np
import onnx
import onnxruntime as ort
ort.set_default_logger_severity(3)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools._score_v2 import score_network_v2
from tools.polish_candidate import polish

SUBMISSION_ZIP = REPO_ROOT / "submissions" / "submission.zip"
TASK_JSON_DIR = REPO_ROOT / "data" / "competitions" / "neurogolf-2026"


def _encode_grid(grid):
    """Encode an ARC grid (list of lists) to the [1,10,30,30] one-hot
    tensor that the validator's convert_to_numpy produces (ch0=0 for
    padding cells per the 2026-05-12 lesson)."""
    a = np.array(grid, dtype=np.int64)
    h, w = a.shape
    out = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for c in range(10):
        out[0, c, :h, :w] = (a == c).astype(np.float32)
    return out


def _load_task_inputs(task_id: str, max_inputs: int = 10) -> list[np.ndarray]:
    """Load real input grids from the task JSON. Use train+test pairs."""
    path = TASK_JSON_DIR / f"{task_id}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    inputs = []
    for section in ("train", "test"):
        for pair in data.get(section, []) or []:
            grid = pair.get("input")
            if isinstance(grid, list):
                try:
                    inputs.append(_encode_grid(grid))
                except Exception:
                    continue
                if len(inputs) >= max_inputs:
                    return inputs
    return inputs


def audit_task(task_id: str) -> dict:
    """Audit one task. Returns dict with all comparison data."""
    out = {"task_id": task_id}
    # 1. Extract original.
    with zipfile.ZipFile(SUBMISSION_ZIP) as z:
        orig_data = z.read(f"{task_id}.onnx")
    orig_model = onnx.load_from_string(orig_data)

    # 2. Polish.
    work_dir = Path(tempfile.mkdtemp(prefix="audit_"))
    polish_out = polish(orig_model, work_dir=work_dir, quiet=True)
    polished = polish_out["final_model"]
    out["polish_steps"] = polish_out["steps"]

    # 3. Cost comparison via direct score_network_v2 (not subprocessed).
    orig_tmp = work_dir / f"{task_id}_orig.onnx"
    polished_tmp = work_dir / f"{task_id}_polished.onnx"
    onnx.save(orig_model, str(orig_tmp))
    onnx.save(polished, str(polished_tmp))
    try:
        orig_score = score_network_v2(str(orig_tmp))
        new_score = score_network_v2(str(polished_tmp))
    except Exception as exc:
        orig_score = None
        new_score = None
        out["score_error"] = str(exc)
    out["orig_score"] = orig_score
    out["new_score"] = new_score
    if orig_score and new_score:
        orig_cost = int(orig_score[1]) + int(orig_score[2])
        new_cost = int(new_score[1]) + int(new_score[2])
        out["orig_cost"] = orig_cost
        out["new_cost"] = new_cost
        out["delta"] = new_cost - orig_cost

    # 4. IR + opset check.
    out["orig_ir_version"] = orig_model.ir_version
    out["polished_ir_version"] = polished.ir_version
    out["orig_opsets"] = [(o.domain, o.version) for o in orig_model.opset_import]
    out["polished_opsets"] = [(o.domain, o.version) for o in polished.opset_import]

    # 5. Real-input byte-exact check.
    real_inputs = _load_task_inputs(task_id, max_inputs=10)
    if not real_inputs:
        out["realinput_check"] = "no task JSON / no input pairs"
    else:
        try:
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            sess_a = ort.InferenceSession(orig_model.SerializeToString(), options, providers=["CPUExecutionProvider"])
            sess_b = ort.InferenceSession(polished.SerializeToString(), options, providers=["CPUExecutionProvider"])
            in_name = sess_a.get_inputs()[0].name
            divergent = 0
            max_diff_overall = 0.0
            for i, arr in enumerate(real_inputs):
                o_a = sess_a.run(None, {in_name: arr})[0]
                o_b = sess_b.run(None, {in_name: arr})[0]
                if not np.array_equal(o_a, o_b):
                    divergent += 1
                    max_diff_overall = max(max_diff_overall, float(np.max(np.abs(o_a.astype(np.float64) - o_b.astype(np.float64)))))
            out["realinput_check"] = {
                "n_inputs": len(real_inputs),
                "divergent": divergent,
                "max_diff": max_diff_overall,
                "byte_exact": (divergent == 0),
            }
        except Exception as exc:
            out["realinput_check"] = f"ORT error: {exc}"

    # Cleanup temp files
    for p in (orig_tmp, polished_tmp):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    return out


def main(task_ids: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    for tid in task_ids:
        print(f"\n=== {tid} ===")
        r = audit_task(tid)
        print(f"  IR version: orig={r.get('orig_ir_version')}  polished={r.get('polished_ir_version')}")
        print(f"  Opsets:     orig={r.get('orig_opsets')}  polished={r.get('polished_opsets')}")
        if "orig_cost" in r:
            print(f"  Cost: {r['orig_cost']} -> {r['new_cost']}  (delta={r['delta']:+d})")
        print(f"  Polish steps applied:")
        for s in r["polish_steps"]:
            if s["verdict"] == "applied":
                print(f"    + {s['pass']}: {s['delta']:+d}")
            elif s["verdict"] in ("verify-failed", "score-failed", "error"):
                print(f"    ! {s['pass']}: {s['verdict']} - {s.get('notes', '')[:60]}")
        ri = r.get("realinput_check")
        if isinstance(ri, dict):
            ok = "OK byte-exact" if ri["byte_exact"] else f"DIVERGENT {ri['divergent']}/{ri['n_inputs']} (max_diff={ri['max_diff']})"
            print(f"  Real-input check ({ri['n_inputs']} pairs): {ok}")
        else:
            print(f"  Real-input check: {ri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
