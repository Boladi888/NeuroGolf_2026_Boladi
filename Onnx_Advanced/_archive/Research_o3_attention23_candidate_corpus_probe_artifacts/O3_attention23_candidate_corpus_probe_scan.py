"""Scan NeuroGolf ONNX artifacts for Attention-23 rewrite candidates.

This script is intentionally read-only over the corpus. It writes a compact JSON
summary beside the research note.
"""

from __future__ import annotations

import json
import math
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx


ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "Logs" / "research" / "O3_attention23_candidate_corpus_probe_scan.json"

DTYPE_BYTES = {
    1: 4,   # float
    2: 1,   # uint8
    3: 1,   # int8
    4: 2,   # uint16
    5: 2,   # int16
    6: 4,   # int32
    7: 8,   # int64
    9: 1,   # bool
    10: 2,  # float16
    11: 8,  # double
    16: 2,  # bfloat16
}

DTYPE_NAMES = {
    1: "F32",
    2: "UINT8",
    3: "INT8",
    4: "UINT16",
    5: "INT16",
    6: "INT32",
    7: "INT64",
    9: "BOOL",
    10: "F16",
    11: "F64",
    16: "BF16",
}

TASK_RE = re.compile(r"task(\d{3})", re.IGNORECASE)


@dataclass(frozen=True)
class ModelSource:
    label: str
    path: str
    zip_member: str | None = None


def task_id_from_label(label: str) -> str | None:
    m = TASK_RE.search(label)
    return m.group(1) if m else None


def enumerate_models() -> list[ModelSource]:
    sources: list[ModelSource] = []

    zip_path = ROOT / "submissions" / "submission.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            for name in sorted(zf.namelist()):
                if name.lower().endswith(".onnx"):
                    sources.append(ModelSource(f"{zip_path.relative_to(ROOT)}::{name}", str(zip_path), name))

    roots = [
        ROOT,
        ROOT / "The_400_Tasks" / "Task_by_Task",
        ROOT / "staged_submits",
        ROOT / "work",
    ]
    seen: set[Path] = set()
    for base in roots:
        if not base.exists():
            continue
        if base == ROOT:
            paths = list(base.glob("*.onnx"))
        else:
            paths = list(base.rglob("*.onnx"))
        for path in sorted(paths):
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            rel = path.relative_to(ROOT).as_posix()
            sources.append(ModelSource(rel, str(path)))
    return sources


def load_model(src: ModelSource) -> onnx.ModelProto:
    if src.zip_member is None:
        return onnx.load(src.path)
    with zipfile.ZipFile(src.path) as zf:
        return onnx.load_model_from_string(zf.read(src.zip_member))


def tensor_shape_and_bytes(value_info: onnx.ValueInfoProto) -> dict[str, Any] | None:
    tt = value_info.type.tensor_type
    if not tt.HasField("elem_type") or not tt.HasField("shape"):
        return None
    dims: list[int] = []
    elems = 1
    for dim in tt.shape.dim:
        if dim.dim_value <= 0:
            return None
        dims.append(int(dim.dim_value))
        elems *= int(dim.dim_value)
    dtype = int(tt.elem_type)
    return {
        "dtype": DTYPE_NAMES.get(dtype, f"dtype{dtype}"),
        "shape": dims,
        "elements": elems,
        "bytes": elems * DTYPE_BYTES.get(dtype, 4),
    }


def byte_info(name: str, vi: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    info = vi.get(name)
    if not info:
        return None
    charged = name not in {"input", "output"}
    return {**info, "charged": charged, "charged_bytes": info["bytes"] if charged else 0}


def attrs_dict(node: onnx.NodeProto) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.INT:
            out[attr.name] = int(attr.i)
        elif attr.type == onnx.AttributeProto.FLOAT:
            out[attr.name] = float(attr.f)
        elif attr.type == onnx.AttributeProto.STRING:
            out[attr.name] = attr.s.decode("utf-8", errors="replace")
        else:
            out[attr.name] = str(onnx.helper.get_attribute_value(attr))[:120]
    return out


def compatible_attention_heuristic(
    softmax_node: onnx.NodeProto,
    final_node: onnx.NodeProto,
    score_info: dict[str, Any] | None,
    weight_info: dict[str, Any] | None,
    final_other_infos: list[dict[str, Any] | None],
) -> tuple[str, str]:
    """Return (yes/no/maybe, reason) for ONNX Attention semantics."""
    if score_info is None or weight_info is None:
        return "no", "missing static score/softmax shape"
    score_shape = score_info["shape"]
    weight_shape = weight_info["shape"]
    if len(score_shape) not in (3, 4):
        return "no", f"score rank {len(score_shape)} is not Attention rank 3/4"
    if score_shape != weight_shape:
        return "no", "softmax output shape differs from score shape"
    axis = attrs_dict(softmax_node).get("axis", -1)
    if axis not in (-1, len(score_shape) - 1):
        return "no", f"Softmax axis {axis} is not the key dimension"
    if final_node.op_type == "MatMul":
        candidates = [i for i in final_other_infos if i is not None]
        if not candidates:
            return "maybe", "MatMul value input shape unavailable"
        score_last = score_shape[-1]
        for info in candidates:
            shape = info["shape"]
            if len(shape) >= 2 and shape[-2] == score_last:
                return "maybe", "QK/softmax/value ranks line up; semantic Q/K/V still needs human proof"
        return "no", "MatMul value input does not align with score key dimension"
    return "maybe", "Einsum consumer needs equation-level human proof"


def is_square_pairwise(info: dict[str, Any] | None) -> bool:
    if not info:
        return False
    shape = info["shape"]
    if len(shape) < 2:
        return False
    if shape[-1] != shape[-2]:
        return False
    return shape[-1] >= 16


def analyze_model(src: ModelSource, model: onnx.ModelProto) -> dict[str, Any]:
    inferred = model
    infer_error = None
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as exc:  # noqa: BLE001
        infer_error = f"{type(exc).__name__}: {exc}"

    graph = inferred.graph
    nodes = list(graph.node)
    op_counts = Counter(node.op_type for node in nodes)
    producer = {out: idx for idx, node in enumerate(nodes) for out in node.output if out}
    consumers: dict[str, list[int]] = {}
    for idx, node in enumerate(nodes):
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(idx)

    vi: dict[str, dict[str, Any]] = {}
    for item in list(graph.input) + list(graph.value_info) + list(graph.output):
        info = tensor_shape_and_bytes(item)
        if info:
            vi[item.name] = info

    softmax_chains: list[dict[str, Any]] = []
    for idx, node in enumerate(nodes):
        if node.op_type != "Softmax" or not node.input or not node.output:
            continue
        input_name = node.input[0]
        output_name = node.output[0]
        prev_idx = producer.get(input_name)
        prev_node = nodes[prev_idx] if prev_idx is not None else None
        next_idxs = consumers.get(output_name, [])
        next_nodes = [nodes[j] for j in next_idxs]
        chain_consumers = [n for n in next_nodes if n.op_type in {"MatMul", "Einsum"}]
        prev_op = prev_node.op_type if prev_node is not None else "GRAPH_INPUT_OR_INIT"
        score_info = byte_info(input_name, vi)
        weight_info = byte_info(output_name, vi)
        for consumer in chain_consumers or next_nodes or [None]:
            final_other_infos: list[dict[str, Any] | None] = []
            if consumer is not None:
                for inp in consumer.input:
                    if inp != output_name:
                        final_other_infos.append(byte_info(inp, vi))
            compat, compat_reason = ("no", "no MatMul/Einsum value aggregation consumer")
            if consumer is not None and consumer.op_type in {"MatMul", "Einsum"}:
                compat, compat_reason = compatible_attention_heuristic(
                    node, consumer, score_info, weight_info, final_other_infos
                )
            current_bytes = 0
            for info in (score_info, weight_info):
                if info:
                    current_bytes += int(info["charged_bytes"])
            softmax_chains.append(
                {
                    "task": task_id_from_label(src.label),
                    "path": src.label,
                    "node_index": idx,
                    "chain": f"{prev_op}->{node.op_type}->{consumer.op_type if consumer is not None else 'NO_CONSUMER'}",
                    "prev_node": prev_node.name if prev_node is not None else None,
                    "softmax_node": node.name,
                    "consumer_node": consumer.name if consumer is not None else None,
                    "score_tensor": input_name,
                    "score": score_info,
                    "softmax_tensor": output_name,
                    "softmax": weight_info,
                    "consumer_other_inputs": final_other_infos,
                    "current_intermediate_bytes": current_bytes,
                    "attention_compatible": compat,
                    "attention_compat_reason": compat_reason,
                }
            )

    pairwise_near_misses: list[dict[str, Any]] = []
    for idx, node in enumerate(nodes):
        if node.op_type not in {"MatMul", "Einsum"}:
            continue
        for out in node.output:
            info = byte_info(out, vi)
            if not is_square_pairwise(info):
                continue
            next_nodes = [nodes[j] for j in consumers.get(out, [])]
            aggregators = [n for n in next_nodes if n.op_type in {"MatMul", "Einsum"}]
            if not aggregators:
                continue
            pairwise_near_misses.append(
                {
                    "task": task_id_from_label(src.label),
                    "path": src.label,
                    "node_index": idx,
                    "chain": f"{node.op_type}->{'/'.join(n.op_type for n in aggregators)}",
                    "tensor": out,
                    "tensor_info": info,
                    "current_intermediate_bytes": int(info["charged_bytes"]) if info else 0,
                    "reason_not_candidate": "pairwise MatMul/Einsum aggregation without Softmax; generic matrix product, not ONNX Attention",
                }
            )

    return {
        "path": src.label,
        "task": task_id_from_label(src.label),
        "nodes": len(nodes),
        "op_counts": dict(op_counts),
        "infer_error": infer_error,
        "softmax_chains": softmax_chains,
        "pairwise_near_misses": pairwise_near_misses,
    }


def main() -> int:
    sources = enumerate_models()
    all_op_counts: Counter[str] = Counter()
    load_failures: list[dict[str, str]] = []
    infer_failures: list[dict[str, str]] = []
    softmax_chains: list[dict[str, Any]] = []
    pairwise_near_misses: list[dict[str, Any]] = []
    models_with_softmax: list[str] = []
    models_with_attention: list[str] = []

    for src in sources:
        try:
            model = load_model(src)
        except Exception as exc:  # noqa: BLE001
            load_failures.append({"path": src.label, "error": f"{type(exc).__name__}: {exc}"})
            continue
        result = analyze_model(src, model)
        all_op_counts.update(result["op_counts"])
        if result["infer_error"]:
            infer_failures.append({"path": src.label, "error": result["infer_error"]})
        if result["op_counts"].get("Softmax", 0):
            models_with_softmax.append(src.label)
        if result["op_counts"].get("Attention", 0):
            models_with_attention.append(src.label)
        softmax_chains.extend(result["softmax_chains"])
        pairwise_near_misses.extend(result["pairwise_near_misses"])

    pairwise_near_misses.sort(key=lambda row: row["current_intermediate_bytes"], reverse=True)
    summary = {
        "scope": {
            "canonical_zip": "submissions/submission.zip::*.onnx",
            "filesystem_roots": [
                "*.onnx at repo root",
                "The_400_Tasks/Task_by_Task/**/*.onnx",
                "staged_submits/**/*.onnx",
                "work/**/*.onnx",
            ],
        },
        "models_enumerated": len(sources),
        "models_loaded": len(sources) - len(load_failures),
        "load_failures": load_failures[:100],
        "load_failure_count": len(load_failures),
        "shape_infer_failure_count": len(infer_failures),
        "shape_infer_failures_sample": infer_failures[:100],
        "op_counts": dict(sorted(all_op_counts.items())),
        "models_with_softmax": models_with_softmax,
        "models_with_attention": models_with_attention,
        "softmax_chain_count": len(softmax_chains),
        "softmax_chains": softmax_chains,
        "pairwise_near_miss_count": len(pairwise_near_misses),
        "pairwise_near_misses_top25": pairwise_near_misses[:25],
        "attention_candidate_count": sum(
            1 for row in softmax_chains if row["attention_compatible"] in {"yes", "maybe"}
        ),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "models_enumerated": summary["models_enumerated"],
        "models_loaded": summary["models_loaded"],
        "load_failure_count": summary["load_failure_count"],
        "shape_infer_failure_count": summary["shape_infer_failure_count"],
        "softmax_chain_count": summary["softmax_chain_count"],
        "attention_candidate_count": summary["attention_candidate_count"],
        "pairwise_near_miss_count": summary["pairwise_near_miss_count"],
        "models_with_softmax": models_with_softmax[:10],
        "models_with_attention": models_with_attention[:10],
        "out": str(OUT_JSON.relative_to(ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
