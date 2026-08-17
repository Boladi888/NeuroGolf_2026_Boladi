"""Build and evaluate O4 ConstantOfShape scorer-confirmation probes.

This script writes only isolated research artifacts under Logs/research.
It intentionally does not stage or submit any task candidate.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import types
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "Logs" / "research" / "O4_constantofshape_scorer_confirmation_models"
RESULT_JSON = ROOT / "Logs" / "research" / "O4_constantofshape_scorer_confirmation_results.json"
DATA_DIR = ROOT / "data" / "competitions" / "neurogolf-2026"
GRID_SHAPE = [1, 10, 30, 30]
OPSET = 10
IR_VERSION = 10


def _install_notebook_stubs() -> None:
    for name in ("matplotlib", "matplotlib.pyplot", "IPython", "IPython.display"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["IPython.display"].display = lambda *a, **kw: None
    sys.modules["IPython.display"].FileLink = lambda *a, **kw: None
    sys.modules["IPython"].display = sys.modules["IPython.display"]
    sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]


def make_model(path: pathlib.Path, construction: str, dtype: int, value: Any | None) -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, GRID_SHAPE)
    y = helper.make_tensor_value_info("output", dtype, GRID_SHAPE)
    inits = []

    if construction == "dense_constant":
        np_dtype = helper.tensor_dtype_to_np_dtype(dtype)
        arr = np.full(GRID_SHAPE, value, dtype=np_dtype)
        node = helper.make_node("Constant", [], ["output"], value=numpy_helper.from_array(arr, "dense_value"))
    elif construction == "constantofshape_default_zero":
        shape = helper.make_tensor("shape_i64", TensorProto.INT64, [4], GRID_SHAPE)
        inits.append(shape)
        node = helper.make_node("ConstantOfShape", ["shape_i64"], ["output"])
    elif construction == "constantofshape_explicit":
        shape = helper.make_tensor("shape_i64", TensorProto.INT64, [4], GRID_SHAPE)
        inits.append(shape)
        np_dtype = helper.tensor_dtype_to_np_dtype(dtype)
        scalar = numpy_helper.from_array(np.asarray([value], dtype=np_dtype), "scalar_value")
        node = helper.make_node("ConstantOfShape", ["shape_i64"], ["output"], value=scalar)
    else:
        raise ValueError(construction)

    graph = helper.make_graph([node], "O4_constantofshape_probe", [x], [y], inits)
    model = helper.make_model(
        graph,
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name="O4_constantofshape_scorer_confirmation",
    )
    onnx.save(model, path)


def tensor_type_and_shape(model: onnx.ModelProto) -> tuple[str, list[int]]:
    out = model.graph.output[0]
    tensor_type = out.type.tensor_type
    dtype = TensorProto.DataType.Name(tensor_type.elem_type)
    shape = [dim.dim_value for dim in tensor_type.shape.dim]
    return dtype, shape


def run_ort(path: pathlib.Path) -> dict[str, Any]:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    dummy = np.zeros(GRID_SHAPE, dtype=np.float32)
    dummy[:, 0, :, :] = 1.0
    out = session.run(["output"], {"input": dummy})[0]
    unique = np.unique(out)
    return {
        "ok": True,
        "shape": list(out.shape),
        "dtype": str(out.dtype),
        "unique": [x.item() if hasattr(x, "item") else x for x in unique[:10]],
        "sum": float(np.asarray(out, dtype=np.float64).sum()),
        "threshold_sum": int((out > 0.0).sum()),
    }


def encode_output_grid(grid: list[list[int]]) -> np.ndarray | None:
    if not grid or not grid[0]:
        return None
    height = len(grid)
    width = len(grid[0])
    if height > 30 or width > 30:
        return None
    arr = np.zeros(GRID_SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if color < 0 or color >= 10:
                return None
            arr[0, color, r, c] = 1.0
    return arr


def task_validity_for_threshold(thresholded: np.ndarray) -> list[int]:
    matching = []
    for tid in range(1, 401):
        with open(DATA_DIR / f"task{tid:03d}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        examples = []
        for split in ("train", "test", "arc-gen"):
            examples.extend(data.get(split, []))
        if not examples:
            continue
        ok = True
        for example in examples:
            target = encode_output_grid(example["output"])
            if target is None or not np.array_equal(thresholded, target):
                ok = False
                break
        if ok:
            matching.append(tid)
    return matching


def scan_task_context() -> dict[str, Any]:
    all_zero_tensor = []
    all_one_tensor = []
    constant_onehot_tensor = []
    uniform_arc_outputs = []
    same_shape_same_color_uniform_arc_outputs = []
    oversize_tasks = set()

    for tid in range(1, 401):
        with open(DATA_DIR / f"task{tid:03d}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        examples = []
        for split in ("train", "test", "arc-gen"):
            examples.extend(data.get(split, []))
        encoded = []
        arc_sigs = []
        valid = True
        all_uniform_arc = True
        for example in examples:
            grid = example["output"]
            vals = sorted({cell for row in grid for cell in row})
            sig = [len(grid), len(grid[0]) if grid else 0, vals]
            arc_sigs.append(sig)
            if len(vals) != 1:
                all_uniform_arc = False
            arr = encode_output_grid(grid)
            if arr is None:
                valid = False
                oversize_tasks.add(tid)
            else:
                encoded.append(arr)
        if valid and encoded:
            if all(np.all(arr == 0) for arr in encoded):
                all_zero_tensor.append(tid)
            if all(np.all(arr == 1) for arr in encoded):
                all_one_tensor.append(tid)
            if all(np.array_equal(arr, encoded[0]) for arr in encoded):
                constant_onehot_tensor.append(
                    {
                        "task": tid,
                        "active_ones": int(encoded[0].sum()),
                        "first_signature": arc_sigs[0],
                        "examples": len(examples),
                    }
                )
        if all_uniform_arc:
            row = {"task": tid, "first_signatures": arc_sigs[:3], "examples": len(examples), "encodable": valid}
            uniform_arc_outputs.append(row)
            if arc_sigs and all(sig == arc_sigs[0] for sig in arc_sigs):
                same_shape_same_color_uniform_arc_outputs.append(row)

    return {
        "all_zero_tensor_tasks": all_zero_tensor,
        "all_one_tensor_tasks": all_one_tensor,
        "constant_onehot_tensor_tasks": constant_onehot_tensor,
        "uniform_arc_output_tasks": uniform_arc_outputs,
        "same_shape_same_color_uniform_arc_output_tasks": same_shape_same_color_uniform_arc_outputs,
        "oversize_or_unencodable_output_tasks": sorted(oversize_tasks),
    }


def evaluate_model(path: pathlib.Path, construction: str, note: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "file": str(path.relative_to(ROOT)),
        "construction": construction,
        "note": note,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

    try:
        model = onnx.load(path)
        onnx.checker.check_model(model, full_check=True)
        result["checker"] = "ok"
    except Exception as exc:  # noqa: BLE001
        result["checker"] = f"fail: {type(exc).__name__}: {exc}"
        return result

    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=True)
        dtype, shape = tensor_type_and_shape(inferred)
        result["shape_inference"] = "ok"
        result["output_dtype"] = dtype
        result["output_shape"] = shape
    except Exception as exc:  # noqa: BLE001
        result["shape_inference"] = f"fail: {type(exc).__name__}: {exc}"

    try:
        ort_result = run_ort(path)
        result["ort_run"] = ort_result
        thresholded = np.zeros(GRID_SHAPE, dtype=np.float32)
        if ort_result["threshold_sum"]:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
            dummy = np.zeros(GRID_SHAPE, dtype=np.float32)
            dummy[:, 0, :, :] = 1.0
            thresholded = (session.run(["output"], {"input": dummy})[0] > 0.0).astype(float)
        result["matching_tasks"] = task_validity_for_threshold(thresholded)
    except Exception as exc:  # noqa: BLE001
        result["ort_run"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        result["matching_tasks"] = []

    try:
        sys.path.insert(0, str(ROOT))
        from tools._score_v2 import score_network_v2

        score = score_network_v2(str(path))
        result["local_scorer"] = None if score is None else {"macs": score[0], "memory": score[1], "params": score[2]}
    except Exception as exc:  # noqa: BLE001
        result["local_scorer"] = {"error": f"{type(exc).__name__}: {exc}"}

    return result


def main() -> None:
    _install_notebook_stubs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    specs = [
        (
            "dense_constant_f32_zero_1x10x30x30.onnx",
            "dense_constant",
            TensorProto.FLOAT,
            0.0,
            "Dense Constant final output, F32 zero fill.",
        ),
        (
            "constantofshape_f32_default_zero_1x10x30x30.onnx",
            "constantofshape_default_zero",
            TensorProto.FLOAT,
            None,
            "ConstantOfShape final output, default F32 zero fill.",
        ),
        (
            "constantofshape_f32_explicit_zero_1x10x30x30.onnx",
            "constantofshape_explicit",
            TensorProto.FLOAT,
            0.0,
            "ConstantOfShape final output, explicit F32 scalar zero value attr.",
        ),
        (
            "dense_constant_u8_seven_1x10x30x30.onnx",
            "dense_constant",
            TensorProto.UINT8,
            7,
            "Dense Constant final output, U8 seven fill.",
        ),
        (
            "constantofshape_u8_explicit_seven_1x10x30x30.onnx",
            "constantofshape_explicit",
            TensorProto.UINT8,
            7,
            "ConstantOfShape final output, explicit U8 scalar seven value attr.",
        ),
    ]

    results = {"models": [], "task_context_scan": scan_task_context()}
    for filename, construction, dtype, value, note in specs:
        path = OUT_DIR / filename
        make_model(path, construction, dtype, value)
        results["models"].append(evaluate_model(path, construction, note))

    RESULT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
