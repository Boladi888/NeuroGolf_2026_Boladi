"""Scan canonical task ONNX payloads for §28 hash-MLP signatures.

The scan is deliberately structural-first: it inspects graph topology and
initializer/value shapes directly from a submission ZIP. For probable hash-MLP
payloads it can optionally run the visible examples through the detected RBF
detector layer to measure active-detector counts.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import onnx
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "competitions" / "neurogolf-2026"
DEFAULT_SUBMISSION = ROOT / "submissions" / "submission.zip"
DEFAULT_LEDGER = ROOT / "The_400_Tasks" / "task_state.csv"
DEFAULT_OUT_DIR = ROOT / "Logs" / "spread_scan_results"

ZERO_WEIGHT_ORIGINAL = {18, 133, 157, 182, 255, 324, 366}
ZERO_WEIGHT_OVERTURNED = {66, 76, 277}


def _yn(value: bool) -> str:
    return "Y" if value else "N"


def _load_ledger(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = {}
        for row in csv.DictReader(f):
            try:
                tid = int(row.get("task_id") or row.get("task", "").replace("task", ""))
            except ValueError:
                continue
            rows[tid] = row
        return rows


def _visible_count(tid: int) -> int:
    path = DATA_DIR / f"task{tid:03d}.json"
    if not path.exists():
        return 0
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    return len(data.get("train", [])) + len(data.get("test", [])) + len(data.get("arc-gen", []))


def _shape_maps(model: onnx.ModelProto) -> tuple[dict[str, list[int | str]], dict[str, int]]:
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:  # noqa: BLE001
        inferred = model

    shapes: dict[str, list[int | str]] = {}
    types: dict[str, int] = {}

    def add_vi(vi: onnx.ValueInfoProto) -> None:
        tensor_type = vi.type.tensor_type
        if not tensor_type.HasField("elem_type"):
            return
        types[vi.name] = tensor_type.elem_type
        if tensor_type.HasField("shape"):
            dims: list[int | str] = []
            for dim in tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    dims.append(int(dim.dim_value))
                elif dim.HasField("dim_param"):
                    dims.append(str(dim.dim_param))
                else:
                    dims.append("?")
            shapes[vi.name] = dims

    for vi in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        add_vi(vi)
    for init in inferred.graph.initializer:
        shapes[init.name] = [int(d) for d in init.dims]
        # Initializer elem type is not always needed, but useful for output probes.
        types.setdefault(init.name, init.data_type)
    return shapes, types


def _consumers(model: onnx.ModelProto) -> dict[str, list[onnx.NodeProto]]:
    out: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for inp in node.input:
            out[inp].append(node)
    return out


def _producer(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {out: node for node in model.graph.node for out in node.output}


def _best_width(shape: list[int | str] | None, visible_pairs: int) -> int | None:
    if not shape:
        return None
    candidates = [d for d in shape if isinstance(d, int) and 1 < d <= 10000]
    if not candidates:
        return None
    if visible_pairs:
        return min(candidates, key=lambda d: (abs(d - visible_pairs), d))
    return max(candidates)


def _find_rbf(model: onnx.ModelProto, shapes: dict[str, list[int | str]], visible_pairs: int) -> dict[str, object]:
    consumers = _consumers(model)
    candidates: list[dict[str, object]] = []
    for sub in model.graph.node:
        if sub.op_type != "Sub" or not sub.output:
            continue
        sub_out = sub.output[0]
        for mul in consumers.get(sub_out, []):
            if mul.op_type != "Mul" or not mul.output:
                continue
            squared = list(mul.input).count(sub_out) >= 2
            after_mul = [(mul.output[0], "")]
            for reducer in consumers.get(mul.output[0], []):
                if reducer.op_type in {"ReduceSum", "ReduceMean"} and reducer.output:
                    after_mul.append((reducer.output[0], f"->{reducer.op_type}"))
            for value_name, mid_chain in after_mul:
                for less in consumers.get(value_name, []):
                    if less.op_type != "Less" or not less.output:
                        continue
                    detector_output = less.output[0]
                    # Prefer the Cast after Less when present; it is often the
                    # hidden activation fed into a later MatMul.
                    cast = next((n for n in consumers.get(detector_output, []) if n.op_type == "Cast" and n.output), None)
                    probe_output = cast.output[0] if cast is not None else detector_output
                    width = _best_width(shapes.get(probe_output) or shapes.get(detector_output), visible_pairs)
                    candidates.append(
                        {
                            "has_rbf_activation": True,
                            "hidden_width": width,
                            "detector_output": probe_output,
                            "rbf_chain": f"Sub->Mul{'(self)' if squared else ''}{mid_chain}->Less"
                            + ("->Cast" if cast else ""),
                        }
                    )
    if not candidates:
        return {
            "has_rbf_activation": False,
            "hidden_width": None,
            "detector_output": "",
            "rbf_chain": "",
        }

    def rank(candidate: dict[str, object]) -> tuple[int, int, int]:
        width = candidate.get("hidden_width")
        if not isinstance(width, int):
            return (1, 1_000_000, 1_000_000)
        if visible_pairs:
            return (0, abs(width - visible_pairs), width)
        return (0, 0, width)

    return min(candidates, key=rank)


def _find_add_relu(model: onnx.ModelProto, shapes: dict[str, list[int | str]], visible_pairs: int) -> dict[str, object]:
    consumers = _consumers(model)
    for add in model.graph.node:
        if add.op_type != "Add" or not add.output:
            continue
        for relu in consumers.get(add.output[0], []):
            if relu.op_type == "Relu" and relu.output:
                return {
                    "has_add_relu": True,
                    "add_relu_width": _best_width(shapes.get(relu.output[0]) or shapes.get(add.output[0]), visible_pairs),
                }
    return {"has_add_relu": False, "add_relu_width": None}


def _first_matmul_width(model: onnx.ModelProto, shapes: dict[str, list[int | str]], visible_pairs: int) -> int | None:
    init_shapes = {init.name: [int(d) for d in init.dims] for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        width = _best_width(shapes.get(node.output[0]) if node.output else None, visible_pairs)
        if width is not None:
            return width
        for inp in node.input:
            dims = init_shapes.get(inp)
            if dims:
                return _best_width(dims, visible_pairs)
    return None


def _runtime_probe(
    model: onnx.ModelProto,
    tid: int,
    detector_output: str,
    types: dict[str, int],
    max_examples: int,
) -> tuple[str, str, str, str]:
    if not detector_output:
        return "", "", "", "no_detector"
    path = DATA_DIR / f"task{tid:03d}.json"
    if not path.exists():
        return "", "", "", "no_json"
    try:
        import json
        import onnxruntime as ort

        sys.path.insert(0, str(DATA_DIR / "neurogolf_utils"))
        import neurogolf_utils as nu  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return "", "", "", f"import_error:{type(exc).__name__}"

    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    output_names = {out.name for out in probe.graph.output}
    if detector_output not in output_names:
        elem_type = types.get(detector_output, TensorProto.FLOAT)
        probe.graph.output.append(helper.make_tensor_value_info(detector_output, elem_type, None))

    try:
        sess = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
        examples_data = json.loads(path.read_text(encoding="utf-8"))
        examples = examples_data.get("train", []) + examples_data.get("test", []) + examples_data.get("arc-gen", [])
        counts: list[float] = []
        for example in examples[:max_examples]:
            bench = nu.convert_to_numpy(example)
            if bench is None:
                continue
            out = sess.run([detector_output], {"input": bench["input"]})[0]
            arr = np.asarray(out)
            if arr.size > 100000:
                return "", "", "", "detector_too_large"
            active = arr.astype(np.float32) > 0.5
            counts.append(float(active.reshape(1, -1).sum(axis=1)[0]))
        if not counts:
            return "", "", "", "no_converted_examples"
        return f"{min(counts):.3g}", f"{max(counts):.3g}", f"{float(np.mean(counts)):.3g}", f"n={len(counts)}"
    except Exception as exc:  # noqa: BLE001
        return "", "", "", f"runtime_error:{type(exc).__name__}"


def _score_cost(path: Path) -> tuple[str, str]:
    try:
        sys.path.insert(0, str(ROOT / "tools"))
        from _score_v2 import score_network_v2

        score = score_network_v2(str(path))
        if score is None:
            return "", "score_failed"
        _macs, memory, params = score
        return str(int(memory + params)), "score_network_v2"
    except Exception as exc:  # noqa: BLE001
        return "", f"score_error:{type(exc).__name__}"


def scan(args: argparse.Namespace) -> list[dict[str, object]]:
    ledger = _load_ledger(args.ledger)
    rows: list[dict[str, object]] = []
    with ZipFile(args.submission) as zf, tempfile.TemporaryDirectory(prefix="hash_mlp_scan_") as tmp_dir:
        tmp = Path(tmp_dir)
        entries = sorted(name for name in zf.namelist() if name.startswith("task") and name.endswith(".onnx"))
        for idx, name in enumerate(entries, 1):
            tid = int(Path(name).stem.replace("task", ""))
            blob = zf.read(name)
            model = onnx.load_model_from_string(blob)
            shapes, types = _shape_maps(model)
            visible_pairs = _visible_count(tid)
            ops = [node.op_type for node in model.graph.node]
            counts = Counter(ops)
            rbf = _find_rbf(model, shapes, visible_pairs)
            add_relu = _find_add_relu(model, shapes, visible_pairs)
            matmul_width = _first_matmul_width(model, shapes, visible_pairs)
            hidden_width = rbf["hidden_width"] or add_relu["add_relu_width"] or matmul_width
            hidden_width_int = int(hidden_width) if isinstance(hidden_width, int) else None
            width_close = bool(hidden_width_int and visible_pairs and abs(hidden_width_int - visible_pairs) <= args.width_tolerance)
            has_matmul = counts.get("MatMul", 0) > 0
            has_rbf = bool(rbf["has_rbf_activation"])
            has_add_relu = bool(add_relu["has_add_relu"])
            has_mlp_structure = has_rbf or has_add_relu or (has_matmul and width_close)
            probable = bool(width_close and (has_rbf or has_add_relu) and (has_matmul or has_rbf))

            ledger_row = ledger.get(tid, {})
            cost = ledger_row.get("current_cost", "")
            cost_source = "task_state.csv" if cost else ""
            if args.score_costs:
                tmp_path = tmp / name
                tmp_path.write_bytes(blob)
                scored_cost, scored_source = _score_cost(tmp_path)
                if scored_cost:
                    cost, cost_source = scored_cost, scored_source
                elif not cost:
                    cost, cost_source = scored_cost, scored_source

            detector_min = detector_max = detector_mean = detector_notes = ""
            if args.runtime_probe and probable and int(counts.get("MatMul", 0)) <= args.runtime_max_matmul:
                detector_min, detector_max, detector_mean, detector_notes = _runtime_probe(
                    model,
                    tid,
                    str(rbf.get("detector_output") or ""),
                    types,
                    args.runtime_max_examples,
                )

            if tid in ZERO_WEIGHT_ORIGINAL:
                zero_family = "zero_weight_family"
            elif tid in ZERO_WEIGHT_OVERTURNED:
                zero_family = "overturned_zero_weight"
            else:
                zero_family = ""

            row = {
                "task_id": f"task{tid:03d}",
                "has_mlp_structure": _yn(has_mlp_structure),
                "has_rbf_activation": _yn(has_rbf),
                "hidden_width": hidden_width_int if hidden_width_int is not None else "",
                "probable_hash_mlp": _yn(probable),
                "nodes": len(model.graph.node),
                "cost": cost,
                "cost_source": cost_source,
                "visible_pairs": visible_pairs,
                "hidden_width_close": _yn(width_close),
                "has_matmul": _yn(has_matmul),
                "has_add_relu": _yn(has_add_relu),
                "matmul_count": counts.get("MatMul", 0),
                "rbf_chain": rbf.get("rbf_chain", ""),
                "detector_active_min": detector_min,
                "detector_active_max": detector_max,
                "detector_active_mean": detector_mean,
                "detector_probe_notes": detector_notes,
                "zero_weight_crossref": zero_family,
                "live_arch": ledger_row.get("live_arch", ""),
                "cherry13b_tested": ledger_row.get("cherry13b_tested", ""),
                "structural_attempt": ledger_row.get("structural_attempt", ""),
                "first_ops": "->".join(ops[:10]),
                "op_signature": "|".join(f"{op}:{n}" for op, n in counts.most_common(8)),
            }
            rows.append(row)
            if args.verbose and (idx == 1 or idx % 25 == 0 or probable):
                print(
                    f"[{idx:03d}/{len(entries)}] task{tid:03d} probable={row['probable_hash_mlp']} "
                    f"width={row['hidden_width']} ops={row['first_ops']}",
                    flush=True,
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--score-costs", action="store_true")
    parser.add_argument("--runtime-probe", action="store_true")
    parser.add_argument("--runtime-max-examples", type=int, default=320)
    parser.add_argument("--runtime-max-matmul", type=int, default=8)
    parser.add_argument("--width-tolerance", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.out is None:
        stamp = dt.date.today().strftime("%Y%m%d")
        args.out = DEFAULT_OUT_DIR / f"hash_mlp_scan_{stamp}.csv"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = scan(args)
    fieldnames = [
        "task_id",
        "has_mlp_structure",
        "has_rbf_activation",
        "hidden_width",
        "probable_hash_mlp",
        "nodes",
        "cost",
        "cost_source",
        "visible_pairs",
        "hidden_width_close",
        "has_matmul",
        "has_add_relu",
        "matmul_count",
        "rbf_chain",
        "detector_active_min",
        "detector_active_max",
        "detector_active_mean",
        "detector_probe_notes",
        "zero_weight_crossref",
        "live_arch",
        "cherry13b_tested",
        "structural_attempt",
        "first_ops",
        "op_signature",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    probable = [row for row in rows if row["probable_hash_mlp"] == "Y"]
    print(f"wrote {args.out}")
    print(f"scanned {len(rows)} tasks; probable_hash_mlp={len(probable)}")
    if probable:
        print("probable:", ", ".join(str(row["task_id"]) for row in probable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
