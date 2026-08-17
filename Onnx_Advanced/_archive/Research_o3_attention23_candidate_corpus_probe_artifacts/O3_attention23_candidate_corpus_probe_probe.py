"""Build and run tiny Attention-23 smoke probes in the pinned NeuroGolf venv."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "Logs" / "research" / "O3_attention23_candidate_corpus_probe"
ATTN_MODEL = BASE.with_name(BASE.name + "_attention23.onnx")
EXPLICIT_MODEL = BASE.with_name(BASE.name + "_explicit_matmul_softmax.onnx")
OUT_JSON = BASE.with_name(BASE.name + "_probe_results.json")


def make_attention_model() -> onnx.ModelProto:
    graph = helper.make_graph(
        [
            helper.make_node(
                "Attention",
                ["input", "input", "input"],
                ["output"],
                name="attention23_self",
                scale=1.0,
            )
        ],
        "attention23_self_probe",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
    model.ir_version = 10
    return model


def make_explicit_model() -> onnx.ModelProto:
    graph = helper.make_graph(
        [
            helper.make_node("Transpose", ["input"], ["k_t"], name="transpose_k", perm=[0, 1, 3, 2]),
            helper.make_node("MatMul", ["input", "k_t"], ["scores"], name="qk_matmul"),
            helper.make_node("Softmax", ["scores"], ["weights"], name="softmax_scores", axis=-1),
            helper.make_node("MatMul", ["weights", "input"], ["output"], name="wv_matmul"),
        ],
        "explicit_attention_probe",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
    model.ir_version = 10
    return model


def check_status(model: onnx.ModelProto) -> str:
    try:
        onnx.checker.check_model(model, full_check=True)
        return "PASS"
    except Exception as exc:  # noqa: BLE001
        return f"FAIL: {type(exc).__name__}: {exc}"


def infer_status(model: onnx.ModelProto) -> tuple[str, list[dict[str, Any]]]:
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as exc:  # noqa: BLE001
        return f"FAIL: {type(exc).__name__}: {exc}", []
    rows: list[dict[str, Any]] = []
    for vi in inferred.graph.value_info:
        tt = vi.type.tensor_type
        shape = [int(d.dim_value) for d in tt.shape.dim]
        rows.append({"name": vi.name, "elem_type": int(tt.elem_type), "shape": shape})
    return "PASS", rows


def ort_status(path: Path, x: np.ndarray) -> tuple[str, np.ndarray | None]:
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        y = sess.run(["output"], {"input": x})[0]
        return f"PASS shape={list(y.shape)} dtype={y.dtype}", y
    except Exception as exc:  # noqa: BLE001
        return f"FAIL: {type(exc).__name__}: {exc}", None


def scorer_status(path: Path) -> str:
    try:
        import sys

        sys.path.insert(0, str(ROOT))
        from tools._score_v2 import score_network_v2  # noqa: PLC0415

        result = score_network_v2(str(path))
        return "None" if result is None else f"macs={result[0]} memory={result[1]} params={result[2]} cost={result[1] + result[2]}"
    except Exception as exc:  # noqa: BLE001
        return f"FAIL: {type(exc).__name__}: {exc}"


def main() -> int:
    ATTN_MODEL.parent.mkdir(parents=True, exist_ok=True)
    models = {
        "attention23_self_4d": (make_attention_model(), ATTN_MODEL),
        "explicit_matmul_softmax_matmul": (make_explicit_model(), EXPLICIT_MODEL),
    }
    for model, path in models.values():
        onnx.save(model, path)

    rng = np.random.default_rng(20260619)
    x = rng.normal(size=(1, 10, 30, 30)).astype(np.float32)

    results: dict[str, dict[str, Any]] = {}
    outputs: dict[str, np.ndarray | None] = {}
    for name, (model, path) in models.items():
        checker = check_status(model)
        infer, value_info = infer_status(model)
        ort_run, output = ort_status(path, x)
        scorer = scorer_status(path)
        outputs[name] = output
        results[name] = {
            "path": str(path.relative_to(ROOT)),
            "opset": 23,
            "inputs": "single NeuroGolf input [1,10,30,30] F32 named input",
            "checker": checker,
            "shape_inference": infer,
            "value_info": value_info,
            "ort_load_run": ort_run,
            "scorer": scorer,
        }

    attn_y = outputs["attention23_self_4d"]
    explicit_y = outputs["explicit_matmul_softmax_matmul"]
    if attn_y is not None and explicit_y is not None:
        results["numeric_compare"] = {
            "attention_vs_explicit_max_abs": float(np.max(np.abs(attn_y - explicit_y))),
            "attention_vs_explicit_allclose_rtol1e_4_atol1e_5": bool(
                np.allclose(attn_y, explicit_y, rtol=1e-4, atol=1e-5)
            ),
        }
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
