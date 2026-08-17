"""scan_freeout_candidates.py — find #67 FREE-OUTPUT conversion targets.

Scans every taskNNN.onnx in submissions/submission.zip for graphs whose output
node is `Pad` (i.e. tail `X -> Pad -> output`) and whose pre-pad tensor X is
"big" (default >= 1,800 B static). Those are candidates for the #67 Equal-tail /
#78 placer-Conv restructures: the expensive pre-pad tensor gets replaced by a
1-channel value canvas (or fused into the output node entirely) because
`calculate_memory` + the profiler skip the tensor literally named 'output'.

Rewrite of the lost 2026-06-06 worker-7 script (same method, +team_split join).
Output: CSV at Logs/spread_scan_results/freeout_scan_<YYYYMMDD>.csv + stdout table.

Usage:
    .venv/Scripts/python tools/scan_freeout_candidates.py [--min-bytes 1800] [--zip submissions/submission.zip]
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from pathlib import Path

import numpy as np  # noqa: F401  (kept for parity with sibling scanners)
import onnx
from onnx import shape_inference

ROOT = Path(__file__).resolve().parent.parent

DTYPE_SIZE = {
    onnx.TensorProto.FLOAT: 4, onnx.TensorProto.INT32: 4, onnx.TensorProto.UINT32: 4,
    onnx.TensorProto.FLOAT16: 2, onnx.TensorProto.INT16: 2, onnx.TensorProto.UINT16: 2,
    onnx.TensorProto.INT64: 8, onnx.TensorProto.UINT64: 8, onnx.TensorProto.DOUBLE: 8,
    onnx.TensorProto.INT8: 1, onnx.TensorProto.UINT8: 1, onnx.TensorProto.BOOL: 1,
}
DTYPE_NAME = {v: k for k, v in onnx.TensorProto.DataType.items()}


def tensor_bytes(vi: onnx.ValueInfoProto) -> tuple[int, str, str]:
    tt = vi.type.tensor_type
    dims = [d.dim_value for d in tt.shape.dim]
    if any(d <= 0 for d in dims):
        return -1, "?", "dynamic"
    n = 1
    for d in dims:
        n *= d
    size = DTYPE_SIZE.get(tt.elem_type, 4)
    return n * size, DTYPE_NAME.get(tt.elem_type, str(tt.elem_type)), "x".join(map(str, dims))


def load_team_split(csv_path: Path) -> dict[str, tuple[str, str]]:
    """task -> (team_split, current_cost)."""
    out: dict[str, tuple[str, str]] = {}
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            out[row["task"]] = (row.get("team_split", ""), row.get("current_cost", ""))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(ROOT / "submissions" / "submission.zip"))
    ap.add_argument("--min-bytes", type=int, default=1800)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    state = load_team_split(ROOT / "The_400_Tasks" / "task_state.csv")
    rows = []
    zf = zipfile.ZipFile(args.zip)
    for tid in range(1, 401):
        name = f"task{tid:03d}.onnx"
        try:
            raw = zf.read(name)
        except KeyError:
            continue
        try:
            model = onnx.load(io.BytesIO(raw))
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: LOAD FAIL {exc}", file=sys.stderr)
            continue
        g = model.graph
        out_node = next((n for n in g.node if "output" in n.output), None)
        if out_node is None or out_node.op_type != "Pad":
            continue
        prepad = out_node.input[0]
        try:
            inferred = shape_inference.infer_shapes(model, strict_mode=False)
        except Exception:  # noqa: BLE001
            inferred = model
        vis = {v.name: v for v in list(inferred.graph.value_info) + list(inferred.graph.output) + list(inferred.graph.input)}
        if prepad not in vis:
            size, dt, shape = -1, "?", "uninferred"
        else:
            size, dt, shape = tensor_bytes(vis[prepad])
        if size < args.min_bytes:
            continue
        # producer chain (2 levels) for triage
        producers = {o: n for n in g.node for o in n.output}
        p1 = producers.get(prepad)
        p1t = p1.op_type if p1 else "graph_input/init"
        p2t = ""
        if p1 is not None and p1.input:
            p2 = producers.get(p1.input[0])
            p2t = p2.op_type if p2 else ""
        team, cost = state.get(f"task{tid:03d}", ("?", "?"))
        rows.append({
            "task": tid, "prepad_bytes": size, "dtype": dt, "shape": shape,
            "producer": p1t, "producer2": p2t, "n_nodes": len(g.node),
            "team_split": team, "current_cost": cost,
        })
    rows.sort(key=lambda r: -r["prepad_bytes"])

    out_path = Path(args.out) if args.out else ROOT / "Logs" / "spread_scan_results" / "freeout_scan_20260607.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["task"])
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} candidates >= {args.min_bytes} B  ->  {out_path}")
    print(f"{'task':>4} {'prepad_B':>8} {'dtype':>8} {'shape':>14} {'producer':>12} {'prod2':>12} {'team':>10} {'cost':>8}")
    for r in rows:
        print(f"{r['task']:>4} {r['prepad_bytes']:>8} {r['dtype']:>8} {r['shape']:>14} "
              f"{r['producer']:>12} {r['producer2']:>12} {r['team_split']:>10} {r['current_cost']:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
