import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from onnx import TensorProto, helper, numpy_helper
from pydantic import BaseModel, Field

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise RuntimeError("onnx requires numpy to build tensor constants") from exc

import door_engine
import project_bridge

load_dotenv()

ROOT = Path(__file__).resolve().parent
CLIENT_DIST = ROOT / "client" / "dist"
CLIENT_PUBLIC = ROOT / "client" / "public"
# Official validator ban list (see Onnx_Advanced/core.md): banned op types plus
# any op whose name contains the substring "Sequence".
BANNED_OPS = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}
FILESIZE_LIMIT_BYTES = 1_509_949  # 1.44 MiB, the official validator limit
SUPPORTED_OPS = {
    "Input",
    "Output",
    "Constant",
    "Cast",
    "Identity",
    "Equal",
    "Greater",
    "Less",
    "GreaterOrEqual",
    "LessOrEqual",
    "Not",
    "And",
    "Or",
    "Xor",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Mod",
    "Min",
    "Max",
    "Sum",
    "Relu",
    "Abs",
    "Neg",
    "Floor",
    "Clip",
    "Sign",
    "Sqrt",
    "ReduceSum",
    "ReduceMax",
    "ReduceMin",
    "ArgMax",
    "Where",
    "Slice",
    "Pad",
    "Concat",
    "Transpose",
    "Tile",
    "Resize",
    "Conv",
    "RowIndex",
    "ColIndex",
}
CANVAS_SHAPE = [1, 1, 30, 30]
ONEHOT_SHAPE = [1, 10, 30, 30]  # the real competition encoding (one-hot on channels)
TASK_ID_RE = re.compile(r"^task\d{3}$")
INPUT_SLOT_ORDER = {
    "Cast": ["input"],
    "Identity": ["input"],
    "Not": ["input"],
    "ReduceSum": ["input"],
    "ArgMax": ["input"],
    "Slice": ["input"],
    "Pad": ["input"],
    "Transpose": ["input"],
    "Tile": ["input"],
    "Resize": ["input"],
    "Conv": ["input"],
    "Output": ["input"],
    "Equal": ["a", "b"],
    "Greater": ["a", "b"],
    "Less": ["a", "b"],
    "GreaterOrEqual": ["a", "b"],
    "LessOrEqual": ["a", "b"],
    "And": ["a", "b"],
    "Or": ["a", "b"],
    "Xor": ["a", "b"],
    "Add": ["a", "b"],
    "Sub": ["a", "b"],
    "Mul": ["a", "b"],
    "Div": ["a", "b"],
    "Mod": ["a", "b"],
    "Min": ["a", "b"],
    "Max": ["a", "b"],
    "Sum": ["a", "b"],
    "Relu": ["input"],
    "Abs": ["input"],
    "Neg": ["input"],
    "Floor": ["input"],
    "Clip": ["input"],
    "Sign": ["input"],
    "Sqrt": ["input"],
    "ReduceMax": ["input"],
    "ReduceMin": ["input"],
    "Where": ["condition", "true", "false"],
    "Concat": ["a", "b"],
}

app = FastAPI(title="NeuroGolf Lab")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExportPayload(BaseModel):
    projectName: str = "neurogolf-graph"
    taskId: str = "task000"
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]] = Field(default_factory=list)
    trainingPairs: list[dict[str, Any]] = Field(default_factory=list)
    # "scalar": the graph computes on color grids [1,1,30,30]; export wraps it
    #           with a one-hot decode/encode bridge so the artifact speaks the
    #           real competition encoding.
    # "onehot": the graph computes directly on the [1,10,30,30] competition
    #           encoding; export only renames I/O to "input"/"output".
    representation: str = "scalar"
    # Scalar-bridge padding mask: "input" masks the encoded output by the input
    # active region (correct for same-size tasks); "none" encodes all 900 cells.
    bridgeMask: str = "input"
    # Which official example sets must pass: subset of ["train","test","arc-gen"].
    exampleSets: list[str] = Field(default_factory=lambda: ["train", "test", "arc-gen"])


class RunPayload(ExportPayload):
    inputGrid: Any | None = None


class DoorPathPayload(BaseModel):
    taskId: str = "task001"
    path: list[dict[str, Any]] = Field(default_factory=list)


class DoorExportPayload(DoorPathPayload):
    ruleHypothesis: str = ""
    projectName: str = ""


class DoorWizardPayload(DoorPathPayload):
    ruleHypothesis: str = ""
    backend: str = "claude"


class ValidationError(Exception):
    pass


def _node_id(node: dict[str, Any]) -> str:
    value = str(node.get("id") or node.get("data", {}).get("id") or "").strip()
    if not value:
        raise ValueError("Every graph node must have a stable id")
    return value


def _op_type(node: dict[str, Any]) -> str:
    data = node.get("data") or {}
    return str(data.get("opType") or data.get("label") or node.get("type") or "").strip()


def _task_id(payload: ExportPayload) -> str:
    task_id = payload.taskId.strip().lower()
    if not TASK_ID_RE.match(task_id):
        raise ValueError("taskId must match taskXXX, for example task010")
    return task_id


def _parse_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "," in text:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _raw_attrs(data: dict[str, Any]) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if isinstance(data.get("attrs"), dict):
        attrs.update(data["attrs"])
    attrs_text = data.get("attrsText")
    if isinstance(attrs_text, str) and attrs_text.strip():
        try:
            parsed = json.loads(attrs_text)
            if not isinstance(parsed, dict):
                raise ValueError("attrsText must be a JSON object")
            attrs.update(parsed)
        except json.JSONDecodeError as exc:
            raise ValueError(f"attrsText is not valid JSON: {exc}") from exc
    return {str(key).strip(): _parse_literal(value) for key, value in attrs.items() if value not in ("", None)}


def _shape(data: dict[str, Any] | None, default: list[int] | None = None) -> list[int]:
    data = data or {}
    raw = data.get("shape") if data.get("shape") not in ("", None) else (default or CANVAS_SHAPE)
    raw = _parse_literal(raw)
    if isinstance(raw, str):
        raw = [int(part.strip()) for part in raw.replace("x", ",").split(",") if part.strip()]
    if not isinstance(raw, list) or not raw or not all(isinstance(item, int) and item > 0 for item in raw):
        raise ValueError("All tensors must have statically defined positive integer shapes")
    return raw


def _onnx_attrs(op: str, attrs: dict[str, Any]) -> dict[str, Any]:
    if op == "Cast":
        return {"to": int(attrs.get("to", TensorProto.FLOAT))}
    if op == "ArgMax":
        return {"axis": int(attrs.get("axis", 1)), "keepdims": int(attrs.get("keepdims", 1))}
    if op == "ReduceSum":
        return {"keepdims": int(attrs.get("keepdims", 1))}
    if op in {"ReduceMax", "ReduceMin"}:
        result = {"keepdims": int(attrs.get("keepdims", 1))}
        if "axes" in attrs:
            axes = attrs["axes"] if isinstance(attrs["axes"], list) else [attrs["axes"]]
            result["axes"] = [int(axis) for axis in axes]
        return result
    if op == "Concat":
        return {"axis": int(attrs.get("axis", 1))}
    if op == "Transpose":
        return {"perm": [int(item) for item in attrs.get("perm", [0, 1, 3, 2])]}
    if op == "Pad":
        return {"mode": str(attrs.get("mode", "constant"))}
    if op == "Resize":
        return {
            "mode": str(attrs.get("mode", "nearest")),
            "coordinate_transformation_mode": str(attrs.get("coordinate_transformation_mode", "asymmetric")),
            "nearest_mode": str(attrs.get("nearest_mode", "floor")),
        }
    if op == "Conv":
        result: dict[str, Any] = {}
        if "pads" in attrs:
            result["pads"] = [int(item) for item in attrs["pads"]]
        if "strides" in attrs:
            result["strides"] = [int(item) for item in attrs["strides"]]
        if "dilations" in attrs:
            result["dilations"] = [int(item) for item in attrs["dilations"]]
        return result
    if op == "Mod":
        return {"fmod": int(attrs.get("fmod", 0))}
    return {}


def _int_list(value: Any, default: list[int]) -> list[int]:
    parsed = _parse_literal(value) if value not in ("", None) else default
    if isinstance(parsed, int):
        parsed = [parsed]
    if isinstance(parsed, str):
        parsed = [int(part.strip()) for part in parsed.replace("x", ",").split(",") if part.strip()]
    if not isinstance(parsed, list) or not all(isinstance(item, int) for item in parsed):
        raise ValueError("expected an integer list")
    return parsed


def _axis(axis: int, rank: int) -> int:
    return (axis + rank) % rank


def _output_shape_for_reduction(input_shape: list[int], attrs: dict[str, Any]) -> list[int]:
    axes = attrs.get("axes")
    keepdims = int(attrs.get("keepdims", 1))
    if axes is None:
        axes = list(range(len(input_shape)))
    if isinstance(axes, int):
        axes = [axes]
    axes = [(axis + len(input_shape)) % len(input_shape) for axis in axes]
    if keepdims:
        return [1 if idx in axes else dim for idx, dim in enumerate(input_shape)]
    return [dim for idx, dim in enumerate(input_shape) if idx not in axes] or [1]


def _output_shape_for_argmax(input_shape: list[int], attrs: dict[str, Any]) -> list[int]:
    axis = int(attrs.get("axis", 1))
    keepdims = int(attrs.get("keepdims", 1))
    axis = (axis + len(input_shape)) % len(input_shape)
    if keepdims:
        return [1 if idx == axis else dim for idx, dim in enumerate(input_shape)]
    return [dim for idx, dim in enumerate(input_shape) if idx != axis] or [1]


def _slice_axis_length(dim: int, start: int, end: int, step: int) -> int:
    """Mirror ONNX Slice semantics for one axis, including negative steps."""
    if step > 0:
        start = min(max(start + dim if start < 0 else start, 0), dim)
        end = min(max(end + dim if end < 0 else end, 0), dim)
    else:
        # ONNX: for negative steps, start clamps to [0, dim-1] and end to
        # [-1, dim-1]; an end below -dim stays -1 (exclusive stop before 0).
        start = start + dim if -dim <= start < 0 else start
        start = min(max(start, 0), dim - 1)
        end = end + dim if -dim <= end < 0 else end
        end = min(max(end, -1), dim - 1)
    return max(0, len(range(start, end, step)))


def _output_shape_for_slice(input_shape: list[int], attrs: dict[str, Any]) -> list[int]:
    axes = _int_list(attrs.get("axes"), list(range(len(input_shape))))
    starts = _int_list(attrs.get("starts"), [0 for _ in axes])
    ends = _int_list(attrs.get("ends"), [input_shape[_axis(axis, len(input_shape))] for axis in axes])
    steps = _int_list(attrs.get("steps"), [1 for _ in axes])
    if not (len(starts) == len(ends) == len(axes) == len(steps)):
        raise ValueError("Slice starts, ends, axes, and steps must have the same length")
    shape = list(input_shape)
    for start, end, axis, step in zip(starts, ends, axes, steps):
        axis = _axis(axis, len(input_shape))
        if step == 0:
            raise ValueError("Slice steps must be non-zero")
        shape[axis] = _slice_axis_length(input_shape[axis], start, end, step)
        if shape[axis] <= 0:
            raise ValueError("Slice output dimensions must be positive")
    return shape


def _output_shape_for_pad(input_shape: list[int], attrs: dict[str, Any]) -> list[int]:
    pads = _int_list(attrs.get("pads"), [0, 0, 0, 0, 0, 0, 0, 0])
    if len(pads) != 2 * len(input_shape):
        raise ValueError(f"Pad pads must have {2 * len(input_shape)} values")
    return [dim + pads[idx] + pads[idx + len(input_shape)] for idx, dim in enumerate(input_shape)]


def _output_shape_for_concat(input_shapes: list[list[int]], attrs: dict[str, Any]) -> list[int]:
    axis = _axis(int(attrs.get("axis", 1)), len(input_shapes[0]))
    shape = list(input_shapes[0])
    shape[axis] = 0
    for idx, input_shape in enumerate(input_shapes, start=1):
        if len(input_shape) != len(shape):
            raise ValueError(f"Concat input {idx} rank does not match")
        for dim_index, dim in enumerate(input_shape):
            if dim_index != axis and dim != input_shapes[0][dim_index]:
                raise ValueError(f"Concat input {idx} shape {input_shape} is incompatible on axis {axis}")
        shape[axis] += input_shape[axis]
    return shape


def _output_shape_for_transpose(input_shape: list[int], attrs: dict[str, Any]) -> list[int]:
    perm = _int_list(attrs.get("perm"), [0, 1, 3, 2])
    if sorted(perm) != list(range(len(input_shape))):
        raise ValueError("Transpose perm must contain every input axis exactly once")
    return [input_shape[idx] for idx in perm]


def _output_shape_for_tile(input_shape: list[int], attrs: dict[str, Any]) -> list[int]:
    repeats = _int_list(attrs.get("repeats"), [1 for _ in input_shape])
    if len(repeats) != len(input_shape) or any(item <= 0 for item in repeats):
        raise ValueError("Tile repeats must be positive and match input rank")
    return [dim * repeat for dim, repeat in zip(input_shape, repeats)]


def _output_shape_for_resize(input_shape: list[int], attrs: dict[str, Any]) -> list[int]:
    if "sizes" in attrs:
        sizes = _int_list(attrs.get("sizes"), input_shape)
        if len(sizes) != len(input_shape) or any(item <= 0 for item in sizes):
            raise ValueError("Resize sizes must be positive and match input rank")
        return sizes
    scales = attrs.get("scales")
    if scales is None:
        return input_shape
    parsed = _parse_literal(scales)
    if not isinstance(parsed, list) or len(parsed) != len(input_shape):
        raise ValueError("Resize scales must match input rank")
    return [max(1, int(round(dim * float(scale)))) for dim, scale in zip(input_shape, parsed)]


def _output_shape_for_conv(input_shape: list[int], attrs: dict[str, Any], weight_shape: list[int]) -> list[int]:
    if len(input_shape) != 4 or len(weight_shape) != 4:
        raise ValueError("Conv currently requires NCHW input and OIHW weights")
    pads = _int_list(attrs.get("pads"), [0, 0, 0, 0])
    strides = _int_list(attrs.get("strides"), [1, 1])
    dilations = _int_list(attrs.get("dilations"), [1, 1])
    out_channels, in_channels, kernel_h, kernel_w = weight_shape
    if input_shape[1] != in_channels:
        raise ValueError(f"Conv weight input channels {in_channels} do not match tensor channels {input_shape[1]}")
    out_h = ((input_shape[2] + pads[0] + pads[2] - dilations[0] * (kernel_h - 1) - 1) // strides[0]) + 1
    out_w = ((input_shape[3] + pads[1] + pads[3] - dilations[1] * (kernel_w - 1) - 1) // strides[1]) + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Conv output dimensions must be positive")
    return [input_shape[0], out_channels, out_h, out_w]


def _tensor_type_for_cast(attrs: dict[str, Any]) -> int:
    return int(attrs.get("to", TensorProto.FLOAT))


def _validate_graph(payload: ExportPayload) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    if not payload.nodes:
        raise ValueError("Graph must contain at least one node")

    by_id: dict[str, dict[str, Any]] = {}
    for node in payload.nodes:
        node_id = _node_id(node)
        if node_id in by_id:
            raise ValueError(f"Duplicate node id: {node_id}")
        op = _op_type(node)
        if op in BANNED_OPS:
            raise ValueError(f"Banned ONNX operation(s): {op}")
        if op not in SUPPORTED_OPS:
            raise ValueError(f"Unsupported ONNX operation: {op}")
        by_id[node_id] = node

    incoming: dict[str, list[dict[str, Any]]] = {node_id: [] for node_id in by_id}
    for edge in payload.edges:
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        if source not in by_id:
            raise ValueError(f"Malformed graph: edge source {source!r} does not exist")
        if target not in by_id:
            raise ValueError(f"Malformed graph: edge target {target!r} does not exist")
        if source == target:
            raise ValueError(f"Malformed graph: self-edge on {source!r}")
        incoming[target].append(edge)

    if not any(_op_type(node) == "Input" for node in by_id.values()):
        raise ValueError("Graph must contain at least one Input node")
    if not any(_op_type(node) == "Output" for node in by_id.values()):
        raise ValueError("Graph must contain at least one Output node")

    sorted_nodes = _topological_sort(by_id, incoming)
    return by_id, incoming, sorted_nodes


def _topological_sort(by_id: dict[str, dict[str, Any]], incoming: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    deps = {node_id: {str(edge["source"]) for edge in edges} for node_id, edges in incoming.items()}
    ready = sorted(node_id for node_id, node_deps in deps.items() if not node_deps)
    ordered_ids: list[str] = []

    while ready:
        node_id = ready.pop(0)
        ordered_ids.append(node_id)
        for other_id in sorted(deps):
            if node_id in deps[other_id]:
                deps[other_id].remove(node_id)
                if not deps[other_id] and other_id not in ordered_ids and other_id not in ready:
                    ready.append(other_id)
        ready.sort()

    if len(ordered_ids) != len(by_id):
        raise ValueError("Malformed graph: cycle detected or unreachable dependency")
    return [by_id[node_id] for node_id in ordered_ids]


def _incoming_ids(node_id: str, op: str, incoming: dict[str, list[dict[str, Any]]]) -> list[str]:
    slot_order = INPUT_SLOT_ORDER.get(op, [])
    slot_index = {slot: index for index, slot in enumerate(slot_order)}
    seen_slots: set[str] = set()

    def sort_index(slot: str) -> int:
        if slot in slot_index:
            return slot_index[slot]
        if re.match(r"^in\d+$", slot):
            return int(slot[2:])
        return len(slot_order)

    for edge in incoming.get(node_id, []):
        slot = str(edge.get("targetHandle") or "").strip()
        if not slot:
            continue
        if slot not in slot_index and not re.match(r"^in\d+$", slot):
            raise ValueError(f"{op} node {node_id!r} has unknown input slot {slot!r}")
        if slot in seen_slots:
            raise ValueError(f"{op} node {node_id!r} has multiple edges for input slot {slot!r}")
        seen_slots.add(slot)
    edges = sorted(
        incoming.get(node_id, []),
        key=lambda edge: (
            sort_index(str(edge.get("targetHandle") or "").strip()),
            str(edge.get("source") or ""),
            str(edge.get("id") or ""),
        ),
    )
    return [str(edge["source"]) for edge in edges]


def _expect_inputs(op: str, node_id: str, ids: list[str]) -> None:
    required = {
        "Cast": 1,
        "Identity": 1,
        "Not": 1,
        "ReduceSum": 1,
        "ReduceMax": 1,
        "ReduceMin": 1,
        "ArgMax": 1,
        "Slice": 1,
        "Pad": 1,
        "Transpose": 1,
        "Tile": 1,
        "Resize": 1,
        "Conv": 1,
        "Equal": 2,
        "Greater": 2,
        "Less": 2,
        "GreaterOrEqual": 2,
        "LessOrEqual": 2,
        "And": 2,
        "Or": 2,
        "Xor": 2,
        "Add": 2,
        "Sub": 2,
        "Mul": 2,
        "Div": 2,
        "Mod": 2,
        "Where": 3,
        "Output": 1,
        "Relu": 1,
        "Abs": 1,
        "Neg": 1,
        "Floor": 1,
        "Clip": 1,
        "Sign": 1,
        "Sqrt": 1,
    }
    expected = required.get(op)
    if expected is not None and len(ids) != expected:
        raise ValueError(f"{op} node {node_id!r} requires {expected} input edge(s), got {len(ids)}")
    if op == "Concat" and len(ids) < 2:
        raise ValueError(f"Concat node {node_id!r} requires at least 2 input edge(s), got {len(ids)}")
    if op in {"Min", "Max", "Sum"} and len(ids) < 2:
        raise ValueError(f"{op} node {node_id!r} requires at least 2 input edge(s), got {len(ids)}")


def _constant_array(data: dict[str, Any], shape: list[int]) -> np.ndarray:
    values = data.get("values", data.get("value"))
    if values is None:
        return np.zeros(shape, dtype=np.float32)
    array = np.asarray(_parse_literal(values), dtype=np.float32)
    if array.size == 1 and int(np.prod(shape)) != 1:
        return np.full(shape, float(array.reshape(-1)[0]), dtype=np.float32)
    try:
        return array.reshape(shape).astype(np.float32)
    except ValueError as exc:
        raise ValueError(f"Constant values cannot be reshaped to {shape}") from exc


def _coordinate_array(kind: str, shape: list[int]) -> np.ndarray:
    arr = np.zeros(shape, dtype=np.float32)
    if kind == "row":
        arr[:, :, :, :] = np.arange(shape[-2], dtype=np.float32).reshape(1, 1, shape[-2], 1)
    else:
        arr[:, :, :, :] = np.arange(shape[-1], dtype=np.float32).reshape(1, 1, 1, shape[-1])
    return arr


def compile_graph(payload: ExportPayload) -> onnx.ModelProto:
    _task_id(payload)
    _by_id, incoming, sorted_nodes = _validate_graph(payload)

    initializers = []
    value_infos = []
    onnx_nodes = []
    graph_outputs = []
    graph_inputs = []
    tensor_name: dict[str, str] = {}
    tensor_shape: dict[str, list[int]] = {}
    tensor_type: dict[str, int] = {}

    for node in sorted_nodes:
        node_id = _node_id(node)
        data = node.get("data") or {}
        op = _op_type(node)
        attrs = _raw_attrs(data)
        output_name = f"{node_id}_out"

        if op == "Input":
            shape = _shape(data, CANVAS_SHAPE)
            if shape not in (CANVAS_SHAPE, ONEHOT_SHAPE):
                raise ValueError(
                    f"Input node {node_id!r} must use static shape {CANVAS_SHAPE} (scalar) "
                    f"or {ONEHOT_SHAPE} (one-hot)"
                )
            tensor_name[node_id] = output_name
            tensor_shape[node_id] = shape
            tensor_type[node_id] = TensorProto.FLOAT
            graph_inputs.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape))
            continue

        input_ids = _incoming_ids(node_id, op, incoming)

        if op == "Output":
            _expect_inputs(op, node_id, input_ids)
            source_id = input_ids[0]
            source_name = tensor_name[source_id]
            graph_outputs.append(helper.make_tensor_value_info(source_name, tensor_type[source_id], tensor_shape[source_id]))
            continue

        if op == "Constant":
            shape = _shape(data, CANVAS_SHAPE)
            array = _constant_array(data, shape)
            tensor = numpy_helper.from_array(array, name=f"{node_id}_value")
            tensor_name[node_id] = output_name
            tensor_shape[node_id] = shape
            tensor_type[node_id] = TensorProto.FLOAT
            onnx_nodes.append(helper.make_node("Constant", inputs=[], outputs=[output_name], name=node_id, value=tensor))
            value_infos.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape))
            continue

        if op in {"RowIndex", "ColIndex"}:
            shape = _shape(data, CANVAS_SHAPE)
            array = _coordinate_array("row" if op == "RowIndex" else "col", shape)
            tensor = numpy_helper.from_array(array, name=f"{node_id}_value")
            tensor_name[node_id] = output_name
            tensor_shape[node_id] = shape
            tensor_type[node_id] = TensorProto.FLOAT
            onnx_nodes.append(helper.make_node("Constant", inputs=[], outputs=[output_name], name=node_id, value=tensor))
            value_infos.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape))
            continue

        _expect_inputs(op, node_id, input_ids)
        inputs = [tensor_name[source_id] for source_id in input_ids]
        input_shapes = [tensor_shape[source_id] for source_id in input_ids]
        input_types = [tensor_type[source_id] for source_id in input_ids]
        shape = _shape(data, input_shapes[0])
        out_type = input_types[0]
        node_inputs = inputs

        same_shape_ops = {
            "Equal",
            "Greater",
            "Less",
            "GreaterOrEqual",
            "LessOrEqual",
            "And",
            "Or",
            "Xor",
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Mod",
            "Min",
            "Max",
            "Sum",
            "Where",
        }
        if op in same_shape_ops:
            for idx, input_shape in enumerate(input_shapes, start=1):
                if input_shape != input_shapes[0]:
                    raise ValueError(f"{op} node {node_id!r} input {idx} shape {input_shape} does not match {input_shapes[0]}")
            shape = input_shapes[0]
        if op in {"Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"}:
            out_type = TensorProto.BOOL
        elif op in {"Not", "And", "Or", "Xor"}:
            out_type = TensorProto.BOOL
        elif op == "Cast":
            out_type = _tensor_type_for_cast(attrs)
        elif op == "Where":
            if input_types[1] != input_types[2]:
                raise ValueError(f"Where node {node_id!r} true/false inputs must have matching tensor types")
            out_type = input_types[1]
        elif op in {"ReduceSum", "ReduceMax", "ReduceMin"}:
            axes = attrs.get("axes")
            shape = _output_shape_for_reduction(input_shapes[0], attrs)
            if op == "ReduceSum" and axes is not None:
                if isinstance(axes, int):
                    axes = [axes]
                axes_name = f"{node_id}_axes"
                initializers.append(helper.make_tensor(axes_name, TensorProto.INT64, [len(axes)], [int(axis) for axis in axes]))
                node_inputs = [inputs[0], axes_name]
        elif op == "Clip":
            if "min" in attrs or "max" in attrs:
                node_inputs = [inputs[0]]
                if "min" in attrs:
                    min_name = f"{node_id}_min"
                    initializers.append(helper.make_tensor(min_name, TensorProto.FLOAT, [], [float(attrs["min"])]))
                    node_inputs.append(min_name)
                elif "max" in attrs:
                    node_inputs.append("")
                if "max" in attrs:
                    max_name = f"{node_id}_max"
                    initializers.append(helper.make_tensor(max_name, TensorProto.FLOAT, [], [float(attrs["max"])]))
                    node_inputs.append(max_name)
        elif op == "ArgMax":
            shape = _output_shape_for_argmax(input_shapes[0], attrs)
            out_type = TensorProto.INT64
        elif op == "Slice":
            starts = _int_list(attrs.get("starts"), [0, 0, 0, 0])
            ends = _int_list(attrs.get("ends"), input_shapes[0])
            axes = _int_list(attrs.get("axes"), list(range(len(starts))))
            steps = _int_list(attrs.get("steps"), [1 for _ in starts])
            shape = _output_shape_for_slice(input_shapes[0], {"starts": starts, "ends": ends, "axes": axes, "steps": steps})
            for suffix, values in {"starts": starts, "ends": ends, "axes": axes, "steps": steps}.items():
                initializer_name = f"{node_id}_{suffix}"
                initializers.append(helper.make_tensor(initializer_name, TensorProto.INT64, [len(values)], [int(item) for item in values]))
                node_inputs.append(initializer_name)
        elif op == "Pad":
            pads = _int_list(attrs.get("pads"), [0 for _ in range(2 * len(input_shapes[0]))])
            shape = _output_shape_for_pad(input_shapes[0], {"pads": pads})
            pads_name = f"{node_id}_pads"
            value_name = f"{node_id}_constant_value"
            initializers.append(helper.make_tensor(pads_name, TensorProto.INT64, [len(pads)], [int(item) for item in pads]))
            initializers.append(helper.make_tensor(value_name, TensorProto.FLOAT, [], [float(attrs.get("value", 0))]))
            node_inputs = [inputs[0], pads_name, value_name]
        elif op == "Concat":
            shape = _output_shape_for_concat(input_shapes, attrs)
        elif op == "Transpose":
            shape = _output_shape_for_transpose(input_shapes[0], attrs)
        elif op == "Tile":
            repeats = _int_list(attrs.get("repeats"), [1 for _ in input_shapes[0]])
            shape = _output_shape_for_tile(input_shapes[0], {"repeats": repeats})
            repeats_name = f"{node_id}_repeats"
            initializers.append(helper.make_tensor(repeats_name, TensorProto.INT64, [len(repeats)], [int(item) for item in repeats]))
            node_inputs = [inputs[0], repeats_name]
        elif op == "Resize":
            shape = _output_shape_for_resize(input_shapes[0], attrs)
            roi_name = f"{node_id}_roi"
            scales_name = f"{node_id}_scales"
            sizes_name = f"{node_id}_sizes"
            initializers.append(helper.make_tensor(roi_name, TensorProto.FLOAT, [0], []))
            initializers.append(helper.make_tensor(scales_name, TensorProto.FLOAT, [0], []))
            initializers.append(helper.make_tensor(sizes_name, TensorProto.INT64, [len(shape)], [int(item) for item in shape]))
            node_inputs = [inputs[0], roi_name, scales_name, sizes_name]
        elif op == "Conv":
            weights = attrs.get("weights", attrs.get("kernel", [1]))
            weight_shape = _shape({"shape": attrs.get("weight_shape", attrs.get("kernel_shape", [1, input_shapes[0][1], 1, 1]))})
            weight_array = np.asarray(_parse_literal(weights), dtype=np.float32)
            if weight_array.size == 1 and int(np.prod(weight_shape)) != 1:
                weight_array = np.full(weight_shape, float(weight_array.reshape(-1)[0]), dtype=np.float32)
            else:
                weight_array = weight_array.reshape(weight_shape).astype(np.float32)
            weight_name = f"{node_id}_weights"
            initializers.append(numpy_helper.from_array(weight_array, name=weight_name))
            node_inputs = [inputs[0], weight_name]
            if "bias" in attrs:
                bias = np.asarray(_parse_literal(attrs["bias"]), dtype=np.float32).reshape([weight_shape[0]])
                bias_name = f"{node_id}_bias"
                initializers.append(numpy_helper.from_array(bias, name=bias_name))
                node_inputs.append(bias_name)
            shape = _output_shape_for_conv(input_shapes[0], attrs, weight_shape)

        tensor_name[node_id] = output_name
        tensor_shape[node_id] = shape
        tensor_type[node_id] = out_type
        onnx_nodes.append(helper.make_node(op, inputs=node_inputs, outputs=[output_name], name=node_id, **_onnx_attrs(op, attrs)))
        value_infos.append(helper.make_tensor_value_info(output_name, out_type, shape))

    if not graph_outputs:
        raise ValueError("Graph must contain at least one connected Output node")

    graph = helper.make_graph(
        onnx_nodes,
        "NeuroGolfLabGraph",
        graph_inputs,
        graph_outputs,
        initializer=initializers,
        value_info=value_infos,
    )
    model = helper.make_model(graph, producer_name="neurogolf-lab", opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


# ---------------------------------------------------------------------------
# Competition export: I/O named exactly "input"/"output", one-hot [1,10,30,30]
# encoding, validated with the official NeuroGolf semantics.
# ---------------------------------------------------------------------------

def _rename_tensor(graph: onnx.GraphProto, old: str, new: str) -> None:
    for node in graph.node:
        node.input[:] = [new if name == old else name for name in node.input]
        node.output[:] = [new if name == old else name for name in node.output]
    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        if value_info.name == old:
            value_info.name = new
    for initializer in graph.initializer:
        if initializer.name == old:
            initializer.name = new


def _single_io(model: onnx.ModelProto) -> tuple[onnx.ValueInfoProto, onnx.ValueInfoProto]:
    initializer_names = {item.name for item in model.graph.initializer}
    real_inputs = [item for item in model.graph.input if item.name not in initializer_names]
    if len(real_inputs) != 1:
        raise ValueError(f"Competition models need exactly 1 graph input, found {len(real_inputs)}")
    if len(model.graph.output) != 1:
        raise ValueError(f"Competition models need exactly 1 graph output, found {len(model.graph.output)}")
    return real_inputs[0], model.graph.output[0]


def compile_competition_model(payload: ExportPayload) -> onnx.ModelProto:
    """Compile the GUI graph into a Kaggle-valid model.

    - representation "onehot": graph already computes on [1,10,30,30]; only the
      I/O tensors are renamed to the hard-coded "input"/"output".
    - representation "scalar": graph computes on color grids [1,1,30,30]; it is
      wrapped with a decode (ArgMax over channels) and encode (Equal against a
      0..9 palette, optionally masked by the input active region) bridge.
    """
    representation = (payload.representation or "scalar").strip().lower()
    if representation not in {"scalar", "onehot"}:
        raise ValueError("representation must be 'scalar' or 'onehot'")
    inner = compile_graph(payload)
    graph_input, graph_output = _single_io(inner)
    input_shape = _shape_from_value_info(graph_input)

    if representation == "onehot":
        if input_shape != ONEHOT_SHAPE:
            raise ValueError(f"onehot representation requires the Input node shape {ONEHOT_SHAPE}, got {input_shape}")
        _rename_tensor(inner.graph, graph_input.name, "input")
        _rename_tensor(inner.graph, graph_output.name, "output")
        # I/O tensors must not linger in value_info (duplicate names are a scorer hard-fail).
        kept = [item for item in inner.graph.value_info if item.name not in {"input", "output"}]
        del inner.graph.value_info[:]
        inner.graph.value_info.extend(kept)
        onnx.checker.check_model(inner)
        return inner

    if input_shape != CANVAS_SHAPE:
        raise ValueError(f"scalar representation requires the Input node shape {CANVAS_SHAPE}, got {input_shape}")
    mask_mode = (payload.bridgeMask or "input").strip().lower()
    if mask_mode not in {"input", "none"}:
        raise ValueError("bridgeMask must be 'input' or 'none'")

    reserved = {
        "input",
        "output",
        "ng_bridge_argmax",
        "ng_bridge_scalar",
        "ng_bridge_scalar_f32",
        "ng_bridge_palette",
        "ng_bridge_palette_value",
        "ng_bridge_onehot",
        "ng_bridge_axes",
        "ng_bridge_zero",
        "ng_bridge_chsum",
        "ng_bridge_active",
    }
    existing = {item.name for item in inner.graph.initializer}
    existing.update(item.name for item in inner.graph.value_info)
    existing.update(name for node in inner.graph.node for name in node.output)
    collision = reserved & existing
    if collision:
        raise ValueError(f"Graph tensor names collide with bridge names: {sorted(collision)}")

    inner_input_name = graph_input.name
    inner_output_name = graph_output.name
    output_elem = graph_output.type.tensor_type.elem_type
    output_shape = _shape_from_value_info(graph_output)
    if output_shape != CANVAS_SHAPE:
        raise ValueError(
            f"scalar representation requires the graph output shape {CANVAS_SHAPE}, got {output_shape}. "
            "Use onehot representation for resized outputs."
        )

    nodes: list[onnx.NodeProto] = []
    value_infos: list[onnx.ValueInfoProto] = []
    initializers: list[onnx.TensorProto] = list(inner.graph.initializer)

    # Decode: one-hot [1,10,30,30] -> scalar colors [1,1,30,30] (padding decodes as 0).
    nodes.append(helper.make_node("ArgMax", ["input"], ["ng_bridge_argmax"], name="ng_bridge_argmax", axis=1, keepdims=1))
    value_infos.append(helper.make_tensor_value_info("ng_bridge_argmax", TensorProto.INT64, CANVAS_SHAPE))
    nodes.append(helper.make_node("Cast", ["ng_bridge_argmax"], [inner_input_name], name="ng_bridge_decode_cast", to=TensorProto.FLOAT))
    value_infos.append(helper.make_tensor_value_info(inner_input_name, TensorProto.FLOAT, CANVAS_SHAPE))

    # The user's graph, unchanged. Its output tensor becomes a plain
    # intermediate; skip names we re-declare to avoid duplicate value_info.
    nodes.extend(inner.graph.node)
    value_infos.extend(
        item for item in inner.graph.value_info if item.name not in {inner_input_name, inner_output_name}
    )
    value_infos.append(helper.make_tensor_value_info(inner_output_name, output_elem, CANVAS_SHAPE))

    # Encode: scalar colors -> one-hot. Values outside 0..9 produce all-zero
    # channels, so a scalar graph can express padding explicitly if it wants.
    scalar_name = inner_output_name
    if output_elem != TensorProto.FLOAT:
        nodes.append(helper.make_node("Cast", [inner_output_name], ["ng_bridge_scalar_f32"], name="ng_bridge_encode_cast", to=TensorProto.FLOAT))
        value_infos.append(helper.make_tensor_value_info("ng_bridge_scalar_f32", TensorProto.FLOAT, CANVAS_SHAPE))
        scalar_name = "ng_bridge_scalar_f32"
    palette = numpy_helper.from_array(
        np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), name="ng_bridge_palette_value"
    )
    nodes.append(helper.make_node("Constant", [], ["ng_bridge_palette"], name="ng_bridge_palette", value=palette))
    value_infos.append(helper.make_tensor_value_info("ng_bridge_palette", TensorProto.FLOAT, [1, 10, 1, 1]))

    if mask_mode == "input":
        nodes.append(helper.make_node("Equal", [scalar_name, "ng_bridge_palette"], ["ng_bridge_onehot"], name="ng_bridge_onehot"))
        value_infos.append(helper.make_tensor_value_info("ng_bridge_onehot", TensorProto.BOOL, ONEHOT_SHAPE))
        initializers.append(helper.make_tensor("ng_bridge_axes", TensorProto.INT64, [1], [1]))
        initializers.append(helper.make_tensor("ng_bridge_zero", TensorProto.FLOAT, [], [0.0]))
        nodes.append(helper.make_node("ReduceSum", ["input", "ng_bridge_axes"], ["ng_bridge_chsum"], name="ng_bridge_chsum", keepdims=1))
        value_infos.append(helper.make_tensor_value_info("ng_bridge_chsum", TensorProto.FLOAT, CANVAS_SHAPE))
        nodes.append(helper.make_node("Greater", ["ng_bridge_chsum", "ng_bridge_zero"], ["ng_bridge_active"], name="ng_bridge_active"))
        value_infos.append(helper.make_tensor_value_info("ng_bridge_active", TensorProto.BOOL, CANVAS_SHAPE))
        nodes.append(helper.make_node("And", ["ng_bridge_onehot", "ng_bridge_active"], ["output"], name="ng_bridge_output"))
    else:
        nodes.append(helper.make_node("Equal", [scalar_name, "ng_bridge_palette"], ["output"], name="ng_bridge_output"))

    graph = helper.make_graph(
        nodes,
        f"{_task_id(payload)}_gui",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ONEHOT_SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, ONEHOT_SHAPE)],
        initializer=initializers,
        value_info=value_infos,
    )
    model = helper.make_model(graph, producer_name="neurogolf-lab", opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def competition_prechecks(model: onnx.ModelProto) -> list[str]:
    """Mirror the official validator's rejection rules (Onnx_Advanced/core.md)."""
    failures: list[str] = []
    size = len(model.SerializeToString())
    if size > FILESIZE_LIMIT_BYTES:
        failures.append(f"file size {size} exceeds the {FILESIZE_LIMIT_BYTES} byte limit")
    if model.functions:
        failures.append("model has local functions, which the validator rejects")
    for opset in model.opset_import:
        if opset.domain not in ("", "ai.onnx"):
            failures.append(f"non-default opset domain {opset.domain!r}")
    banned_upper = {op.upper() for op in BANNED_OPS}
    for node in model.graph.node:
        if node.op_type.upper() in banned_upper:
            failures.append(f"banned op type {node.op_type}")
        if "sequence" in node.op_type.lower():
            failures.append(f"op type {node.op_type} contains 'Sequence'")
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                failures.append(f"node {node.name or node.op_type} has a subgraph attribute")
        if node.output and "kernel_time" in node.output[0]:
            failures.append(f"node output name {node.output[0]!r} contains 'kernel_time'")
    try:
        graph_input, graph_output = _single_io(model)
    except ValueError as exc:
        failures.append(str(exc))
        return failures
    if graph_input.name != "input":
        failures.append(f"graph input must be named 'input', got {graph_input.name!r}")
    if graph_output.name != "output":
        failures.append(f"graph output must be named 'output', got {graph_output.name!r}")
    if _shape_from_value_info(graph_input) != ONEHOT_SHAPE:
        failures.append(f"graph input shape must be {ONEHOT_SHAPE}, got {_shape_from_value_info(graph_input)}")
    seen: set[str] = set()
    for value_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if value_info.name in seen:
            failures.append(f"duplicate tensor name {value_info.name!r} across input/value_info/output")
        seen.add(value_info.name)
        for dim in value_info.type.tensor_type.shape.dim:
            if dim.HasField("dim_param") or (dim.HasField("dim_value") and dim.dim_value <= 0):
                failures.append(f"tensor {value_info.name!r} has a dynamic or non-positive dimension")
                break
    return failures


def _task_json_dirs() -> list[Path]:
    dirs = [CLIENT_DIST / "tasks", CLIENT_PUBLIC / "tasks"]
    env_dir = os.getenv("NEUROGOLF_DATA_DIR", "").strip()
    if env_dir:
        dirs.append(Path(env_dir).expanduser())
    if project_bridge.PROJECT_ROOT is not None:
        dirs.append(project_bridge.PROJECT_ROOT / "data" / "competitions" / "neurogolf-2026")
    return dirs


def _official_task_examples(task_id: str) -> dict[str, list[dict[str, Any]]]:
    for task_dir in _task_json_dirs():
        path = task_dir / f"{task_id}.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise ValidationError(f"No task JSON found for {task_id}")


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray | None:
    """Mirror neurogolf_utils.convert_to_numpy for a single grid."""
    height = len(grid)
    width = len(grid[0]) if height else 0
    if max(height, width) > 30:
        return None
    benchmark = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            benchmark[0][int(color)][r][c] = 1.0
    return benchmark


def validate_competition(model: onnx.ModelProto, task_id: str, example_sets: list[str]) -> dict[str, Any]:
    """Validate with the exact official semantics: one-hot input, run, threshold
    at > 0.0, and require equality with the one-hot expected output."""
    failures = competition_prechecks(model)
    if failures:
        raise ValidationError("Validator prechecks failed: " + "; ".join(failures))

    examples = _official_task_examples(task_id)
    sets = [name for name in (example_sets or ["train", "test", "arc-gen"]) if name in ("train", "test", "arc-gen")]
    if not sets:
        raise ValidationError("exampleSets must include at least one of train/test/arc-gen")

    try:
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ValidationError(f"ONNX Runtime could not load model: {exc}") from exc

    results: dict[str, dict[str, int]] = {}
    first_failure: dict[str, Any] | None = None
    for set_name in sets:
        right, wrong, skipped = 0, 0, 0
        for index, example in enumerate(examples.get(set_name, [])):
            input_onehot = _grid_to_onehot(example["input"])
            expected_onehot = _grid_to_onehot(example["output"])
            if input_onehot is None or expected_onehot is None:
                skipped += 1  # official convert_to_numpy returns None for >30 grids
                continue
            try:
                raw = session.run(["output"], {"input": input_onehot})[0]
                predicted = (raw > 0.0).astype(float)
            except Exception as exc:  # noqa: BLE001 - official verify counts runtime errors as wrong
                wrong += 1
                if first_failure is None:
                    first_failure = {"set": set_name, "index": index, "reason": f"runtime error: {exc}"}
                continue
            if predicted.shape == expected_onehot.shape and np.array_equal(predicted, expected_onehot):
                right += 1
            else:
                wrong += 1
                if first_failure is None:
                    bad = _first_bad_index(predicted, expected_onehot) if predicted.shape == expected_onehot.shape else []
                    first_failure = {
                        "set": set_name,
                        "index": index,
                        "reason": f"output mismatch at index {bad}" if bad else f"output shape {list(predicted.shape)}",
                    }
        results[set_name] = {"pass": right, "fail": wrong, "skipped": skipped}

    ready = all(item["fail"] == 0 for item in results.values())
    return {"ready": ready, "sets": results, "firstFailure": first_failure}


def save_model(model: onnx.ModelProto, payload: ExportPayload) -> Path:
    task_id = _task_id(payload)
    out_dir = Path(tempfile.mkdtemp(prefix="neurogolf_"))
    out_path = out_dir / f"{task_id}.onnx"
    onnx.save(model, out_path)
    return out_path


def _dim_value(dim: onnx.TensorShapeProto.Dimension) -> int | str | None:
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    if dim.HasField("dim_param"):
        return dim.dim_param
    return None


def _value_info_summary(value_info: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    return {
        "name": value_info.name,
        "elemType": int(tensor_type.elem_type),
        "shape": [_dim_value(dim) for dim in tensor_type.shape.dim],
    }


def _model_summary(model: onnx.ModelProto) -> dict[str, Any]:
    return {
        "inputs": [_value_info_summary(item) for item in model.graph.input],
        "outputs": [_value_info_summary(item) for item in model.graph.output],
    }


def _safe_id(value: str, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    return text or fallback


def _shape_from_value_info(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    tensor_type = value_info.type.tensor_type
    return [_dim_value(dim) for dim in tensor_type.shape.dim]


def _json_ready(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, onnx.TensorProto):
        array = numpy_helper.to_array(value)
        return {
            "kind": "tensor",
            "dataType": int(value.data_type),
            "shape": list(array.shape),
            "size": int(array.size),
            "valuesPreview": array.reshape(-1)[:12].tolist(),
        }
    if isinstance(value, onnx.GraphProto):
        return {"kind": "graph", "name": value.name, "nodeCount": len(value.node)}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _onnx_attrs_for_gui(node: onnx.NodeProto) -> dict[str, Any]:
    return {attr.name: _json_ready(helper.get_attribute_value(attr)) for attr in node.attribute}


def _best_onnx_bytes(task_id: str) -> tuple[bytes, str]:
    task_id = task_id.strip().lower()
    if not TASK_ID_RE.match(task_id):
        raise ValueError("taskId must match taskXXX, for example task010")
    candidates = [
        CLIENT_DIST / "best" / "onnx" / f"{task_id}.onnx",
        CLIENT_PUBLIC / "best" / "onnx" / f"{task_id}.onnx",
    ]
    for path in candidates:
        if path.exists():
            return path.read_bytes(), str(path.relative_to(ROOT))
    data = project_bridge.best_onnx_bytes(task_id)
    if data is not None:
        return data, f"submissions/submission.zip::{task_id}.onnx"
    raise FileNotFoundError(f"No best ONNX file found for {task_id}")


def onnx_to_gui_graph(task_id: str) -> dict[str, Any]:
    model_bytes, source = _best_onnx_bytes(task_id)
    model = onnx.load_from_string(model_bytes)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    producers: dict[str, str] = {}
    occupied_ids: set[str] = set()

    def unique_id(raw: str, fallback: str) -> str:
        base = _safe_id(raw, fallback)
        candidate = base
        suffix = 2
        while candidate in occupied_ids:
            candidate = f"{base}_{suffix}"
            suffix += 1
        occupied_ids.add(candidate)
        return candidate

    def add_node(raw_id: str, fallback: str, op_type: str, index: int, data: dict[str, Any]) -> str:
        node_id = unique_id(raw_id, fallback)
        nodes.append(
            {
                "id": node_id,
                "type": "op",
                "position": {"x": 70 + (index % 8) * 210, "y": 80 + (index // 8) * 130},
                "data": {"label": node_id, "opType": op_type, **data},
            }
        )
        return node_id

    index = 0
    initializer_names = {item.name for item in model.graph.initializer}
    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue
        node_id = add_node(
            value_info.name,
            f"input_{index}",
            "Input",
            index,
            {
                "shape": ",".join(str(item) for item in _shape_from_value_info(value_info)),
                "sourceName": value_info.name,
                "inputSlots": [],
            },
        )
        producers[value_info.name] = node_id
        index += 1

    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        node_id = add_node(
            initializer.name,
            f"init_{index}",
            "Constant",
            index,
            {
                "shape": ",".join(str(item) for item in array.shape),
                "sourceName": initializer.name,
                "inputSlots": [],
                "attrs": {
                    "source": "onnx-initializer",
                    "dataType": int(initializer.data_type),
                    "size": int(array.size),
                    "valuesPreview": array.reshape(-1)[:12].tolist(),
                },
            },
        )
        producers[initializer.name] = node_id
        index += 1

    for node_index, onnx_node in enumerate(model.graph.node, start=1):
        slots = [f"in{slot_index}" for slot_index, name in enumerate(onnx_node.input) if name]
        node_id = add_node(
            onnx_node.name or f"{onnx_node.op_type}_{node_index}",
            f"onnx_{node_index}",
            onnx_node.op_type,
            index,
            {
                "sourceName": onnx_node.name,
                "inputSlots": slots,
                "attrs": _onnx_attrs_for_gui(onnx_node),
            },
        )
        index += 1
        visible_slot = 0
        for input_index, input_name in enumerate(onnx_node.input):
            if not input_name:
                continue
            source_id = producers.get(input_name)
            if source_id:
                slot = f"in{visible_slot}"
                edges.append(
                    {
                        "id": f"e_{len(edges) + 1}",
                        "source": source_id,
                        "target": node_id,
                        "targetHandle": slot,
                        "data": {"tensor": input_name, "inputIndex": input_index},
                    }
                )
            visible_slot += 1
        for output_name in onnx_node.output:
            if output_name:
                producers[output_name] = node_id

    for output_index, output in enumerate(model.graph.output, start=1):
        node_id = add_node(
            f"output_{output.name}",
            f"output_{output_index}",
            "Output",
            index,
            {
                "shape": ",".join(str(item) for item in _shape_from_value_info(output)),
                "sourceName": output.name,
                "inputSlots": ["input"],
            },
        )
        index += 1
        source_id = producers.get(output.name)
        if source_id:
            edges.append(
                {
                    "id": f"e_{len(edges) + 1}",
                    "source": source_id,
                    "target": node_id,
                    "targetHandle": "input",
                    "data": {"tensor": output.name},
                }
            )

    return {
        "projectName": f"best-{task_id}-onnx",
        "taskId": task_id,
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "source": source,
            "rawOnnx": True,
            "nodeCount": len(model.graph.node),
            "initializerCount": len(model.graph.initializer),
            "opTypes": sorted({node.op_type for node in model.graph.node}),
        },
    }


def _arc_grid_to_canvas(value: Any) -> tuple[np.ndarray, tuple[int, int]]:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        height, width = arr.shape
        tensor = arr.reshape(1, 1, height, width)
    elif arr.ndim == 4:
        tensor = arr.astype(np.float32)
        height, width = tensor.shape[-2:]
    else:
        raise ValueError(f"expected a 2D ARC grid or NCHW tensor, got rank {arr.ndim}")
    if height > 30 or width > 30:
        raise ValueError(f"ARC grid shape {[height, width]} exceeds 30x30 canvas")
    canvas = np.zeros(CANVAS_SHAPE, dtype=np.float32)
    canvas[:, :, :height, :width] = tensor[:, :, :height, :width]
    return canvas, (height, width)


def _first_bad_index(actual: np.ndarray, expected: np.ndarray) -> list[int]:
    bad = np.argwhere(actual != expected)
    if bad.size == 0:
        return []
    idx = bad[0].tolist()
    return idx[-2:] if len(idx) >= 2 else idx


def _assert_color_bounds(label: str, tensor: np.ndarray) -> None:
    values = np.asarray(tensor)
    if values.dtype.kind == "b":
        raise ValidationError(f"Phase 3 Color Bounds failed: {label} produced boolean values, not ARC color integers")
    if not np.all(np.isfinite(values)):
        idx = np.argwhere(~np.isfinite(values))[0].tolist()
        raise ValidationError(f"Phase 3 Color Bounds failed: {label} produced non-finite value at index {idx[-2:]}")
    if not np.all(values == np.round(values)):
        idx = np.argwhere(values != np.round(values))[0].tolist()
        raise ValidationError(f"Phase 3 Color Bounds failed: {label} produced non-integer value at index {idx[-2:]}")
    if not np.all((values >= 0) & (values <= 9)):
        idx = np.argwhere((values < 0) | (values > 9))[0].tolist()
        bad_value = values[tuple(idx)]
        raise ValidationError(f"Phase 3 Color Bounds failed: {label} produced color {bad_value:g} at index {idx[-2:]}")


def _run_session(session: ort.InferenceSession, source_grid: Any) -> tuple[np.ndarray, tuple[int, int]]:
    inputs = session.get_inputs()
    if not inputs:
        raise ValidationError("Phase 1 Strict Equivalence failed: compiled model has no inputs")
    feed = {}
    region: tuple[int, int] | None = None
    for input_meta in inputs:
        if list(input_meta.shape) == ONEHOT_SHAPE:
            grid = np.asarray(source_grid)
            if grid.ndim != 2:
                raise ValidationError(f"one-hot model input requires a 2D ARC grid, got rank {grid.ndim}")
            onehot = _grid_to_onehot([[int(v) for v in row] for row in grid.tolist()])
            if onehot is None:
                raise ValidationError("ARC grid exceeds the 30x30 canvas")
            feed[input_meta.name] = onehot
            region = (grid.shape[0], grid.shape[1])
        elif list(input_meta.shape) == CANVAS_SHAPE:
            canvas, region = _arc_grid_to_canvas(source_grid)
            feed[input_meta.name] = canvas
        else:
            raise ValidationError(
                f"Model input {input_meta.name} has shape {input_meta.shape}, expected {CANVAS_SHAPE} or {ONEHOT_SHAPE}"
            )
    return session.run(None, feed)[0], region or (30, 30)


def _onehot_to_grid(tensor: np.ndarray) -> list[list[int]]:
    """Mirror neurogolf_utils.convert_from_numpy on a thresholded one-hot tensor:
    one active channel -> that color, none -> 10 (trimmed at edges), many -> 11."""
    mask = (np.asarray(tensor) > 0.0).astype(int)[0]
    counts = mask.sum(axis=0)
    colors = mask.argmax(axis=0)
    grid: list[list[int]] = []
    for r in range(counts.shape[0]):
        row: list[int] = []
        for c in range(counts.shape[1]):
            row.append(int(colors[r][c]) if counts[r][c] == 1 else (11 if counts[r][c] else 10))
        while row and row[-1] == 10:
            row.pop()
        grid.append(row)
    while grid and not grid[-1]:
        grid.pop()
    return grid


def _tensor_to_grid(tensor: np.ndarray, region: tuple[int, int]) -> list[list[int | float | bool]]:
    array = np.asarray(tensor)
    if array.ndim >= 4:
        array = array[0, 0]
    elif array.ndim == 3:
        array = array[0]
    elif array.ndim == 1:
        array = array.reshape(1, -1)
    height, width = region
    if array.ndim == 2:
        array = array[:height, :width]
    rounded = np.rint(array)
    if np.all(np.isfinite(array)) and np.all(array == rounded):
        return rounded.astype(int).tolist()
    return array.tolist()


def validate_model(model: onnx.ModelProto, payload: ExportPayload) -> dict[str, str]:
    if not payload.trainingPairs:
        raise ValidationError("Phase 1 Strict Equivalence failed: no ARC training pairs were supplied")

    try:
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ValidationError(f"Phase 1 Strict Equivalence failed: ONNX Runtime could not load model: {exc}") from exc

    strict_outputs: list[np.ndarray] = []
    canvas_outputs: list[np.ndarray] = []

    for index, pair in enumerate(payload.trainingPairs, start=1):
        expected_key = "output" if "output" in pair else "target"
        if "input" not in pair or expected_key not in pair:
            raise ValidationError(f"Phase 1 Strict Equivalence failed: Train {index} is missing input or output grid")
        try:
            actual, _input_region = _run_session(session, pair["input"])
            expected_canvas, (height, width) = _arc_grid_to_canvas(pair[expected_key])
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError(f"Phase 1 Strict Equivalence failed: Train {index} runtime error: {exc}") from exc
        if actual.ndim < 2 or actual.shape[-2] < height or actual.shape[-1] < width:
            raise ValidationError(
                f"Phase 1 Strict Equivalence failed: Train {index} output shape {list(actual.shape)} "
                f"cannot cover expected window {[height, width]}"
            )
        actual_window = actual[..., :height, :width]
        expected_window = expected_canvas[..., :height, :width]
        if not np.array_equal(actual_window, expected_window):
            bad_index = _first_bad_index(actual_window, expected_window)
            raise ValidationError(f"Phase 1 Strict Equivalence failed: Train {index} output mismatched at index {bad_index}")
        strict_outputs.append(actual)

    for index, pair in enumerate(payload.trainingPairs, start=1):
        try:
            actual, _region = _run_session(session, pair["input"])
            canvas_outputs.append(actual)
        except Exception as exc:
            raise ValidationError(f"Phase 2 Canvas Test failed: Train {index} 30x30 canvas runtime error: {exc}") from exc

    for index, tensor in enumerate(strict_outputs, start=1):
        _assert_color_bounds(f"Train {index}", tensor)
    for index, tensor in enumerate(canvas_outputs, start=1):
        _assert_color_bounds(f"Canvas Train {index}", tensor)

    return {"train": "passed", "shape": "passed", "colors": "passed"}


@app.post("/api/compile")
def compile_onnx(payload: ExportPayload):
    try:
        model = compile_graph(payload)
        return {
            "status": "compiled",
            "taskId": _task_id(payload),
            "modelBytes": len(model.SerializeToString()),
            "nodeCount": len(payload.nodes),
            "edgeCount": len(payload.edges),
            "io": _model_summary(model),
        }
    except Exception as exc:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": str(exc)})


@app.post("/api/run")
def run_onnx(payload: RunPayload):
    try:
        model = compile_graph(payload)
        source_grid = payload.inputGrid
        if source_grid is None:
            if not payload.trainingPairs:
                raise ValueError("Run requires inputGrid or at least one training pair")
            source_grid = payload.trainingPairs[0].get("input")
        if source_grid is None:
            raise ValueError("Run input grid is missing")
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        actual, region = _run_session(session, source_grid)
        if actual.ndim == 4 and actual.shape[1] == 10:
            grid = _onehot_to_grid(actual)
            decoded = "onehot"
        else:
            _assert_color_bounds("Run", actual)
            grid = _tensor_to_grid(actual, region)
            decoded = "scalar"
        return {
            "status": "ran",
            "taskId": _task_id(payload),
            "shape": list(actual.shape),
            "grid": grid,
            "decoded": decoded,
            "io": _model_summary(model),
        }
    except ValidationError as exc:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": str(exc)})
    except Exception as exc:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": str(exc)})


def _score_temp_model(model: onnx.ModelProto, task_id: str) -> dict[str, Any]:
    """Score through the official pipeline using a throwaway temp file."""
    temp_dir = Path(tempfile.mkdtemp(prefix="neurogolf_score_"))
    temp_path = temp_dir / f"{task_id}.onnx"
    try:
        onnx.save(model, temp_path)
        return project_bridge.score_file(temp_path)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
            temp_dir.rmdir()
        except OSError:
            pass


def _competition_check(payload: ExportPayload) -> dict[str, Any]:
    """Compile + officially validate + score. Shared by /api/validate and /api/export."""
    task_id = _task_id(payload)
    model = compile_competition_model(payload)
    validation = validate_competition(model, task_id, payload.exampleSets)
    score = _score_temp_model(model, task_id) if validation["ready"] else {"scored": False, "reason": "validation failed"}
    baseline = project_bridge.task_baseline(task_id)
    delta_points = None
    if score.get("scored") and baseline.get("available") and baseline.get("currentPoints") is not None:
        delta_points = round(score["points"] - baseline["currentPoints"], 6)
    return {
        "taskId": task_id,
        "model": model,
        "validation": validation,
        "score": score,
        "baseline": baseline,
        "deltaPoints": delta_points,
        "modelBytes": len(model.SerializeToString()),
        "opTypes": sorted({node.op_type for node in model.graph.node}),
    }


def _check_response(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "model"}


@app.post("/api/validate")
def validate_endpoint(payload: ExportPayload):
    """Competition-exact validation + scoring without writing any artifact."""
    try:
        result = _competition_check(payload)
        return {"status": "validated" if result["validation"]["ready"] else "not_ready", **_check_response(result)}
    except ValidationError as exc:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": str(exc)})
    except Exception as exc:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": str(exc)})


@app.post("/api/export")
def export_onnx(payload: ExportPayload):
    """Compile, validate with official semantics, score, and save the artifact
    into the project's Task_by_Task/<task>/prototypes/ folder (additive only)."""
    try:
        result = _competition_check(payload)
        if not result["validation"]["ready"]:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "failed",
                    "reason": "Official validation failed: "
                    + json.dumps(result["validation"]["firstFailure"] or result["validation"]["sets"]),
                    **_check_response(result),
                },
            )
        sidecar = {
            "taskId": result["taskId"],
            "projectName": payload.projectName,
            "tool": "neurogolf-lab-gui",
            "representation": payload.representation,
            "bridgeMask": payload.bridgeMask,
            "validation": result["validation"],
            "score": result["score"],
            "baseline": result["baseline"],
            "deltaPoints": result["deltaPoints"],
            "opTypes": result["opTypes"],
            "graph": {"nodes": payload.nodes, "edges": payload.edges},
        }
        artifact = project_bridge.save_artifact(result["model"].SerializeToString(), result["taskId"], sidecar)
        return {"status": "passed", "artifact": artifact["onnx"], "artifactPaths": artifact, **_check_response(result)}
    except ValidationError as exc:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": str(exc)})
    except Exception as exc:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": str(exc)})


@app.get("/api/best-graph/{task_id}")
def best_graph(task_id: str):
    try:
        return onnx_to_gui_graph(task_id)
    except Exception as exc:
        return JSONResponse(status_code=404, content={"status": "failed", "reason": str(exc)})


# ---------------------------------------------------------------------------
# Doors Maze game (DOORS_DESIGN.md) — play, lookahead, export
# ---------------------------------------------------------------------------

def _door_task_id(payload: DoorPathPayload) -> str:
    task_id = payload.taskId.strip().lower()
    if not TASK_ID_RE.match(task_id):
        raise ValueError("taskId must match taskXXX, for example task010")
    return task_id


@app.get("/api/door/library")
def door_library():
    return {"doors": [{"op": op, **meta} for op, meta in door_engine.DOORS.items()],
            "groups": ["Paint", "Scissors", "Mirror", "Magnet"]}


@app.post("/api/door/step")
def door_step(payload: DoorPathPayload):
    """Stateless: replay the whole path on every train example. Returns the
    boards, distance, win state, and the Tier-1 lookahead (glowing doors)."""
    try:
        task_id = _door_task_id(payload)
        examples = _official_task_examples(task_id)
        state = door_engine.path_state(examples["train"], payload.path)
        response: dict[str, Any] = {"status": "ok", "taskId": task_id, **state}
        if state["win"]:
            response["sets"] = door_engine.verify_sets(examples, payload.path)
            response["lookahead"] = []
        else:
            response["lookahead"] = door_engine.lookahead(examples["train"], payload.path)
        return response
    except door_engine.DoorError as exc:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": str(exc)})
    except Exception as exc:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": str(exc)})


@app.post("/api/door/export")
def door_export(payload: DoorExportPayload):
    """Winning path -> verified recipe. Tries the auto-compile route (lower to
    ExportPayload, ride the existing competition gate); falls back to a
    recipe-only JSON for the agent loop when lowering or validation fails."""
    try:
        task_id = _door_task_id(payload)
        examples = _official_task_examples(task_id)
        state = door_engine.path_state(examples["train"], payload.path)
        if not state["win"]:
            return JSONResponse(status_code=400, content={
                "status": "failed", "reason": "path does not solve all train examples yet"})
        sets = door_engine.verify_sets(examples, payload.path)

        recipe = {
            "schema": "neurogolf-doorpath/v1",
            "taskId": task_id,
            "ruleHypothesis": payload.ruleHypothesis,
            "win": {name: f"{counts['pass']}/{counts['pass'] + counts['fail']}" for name, counts in sets.items()},
            "path": payload.path,
            "provenance": {"tool": "neurogolf-lab-doors", "projectName": payload.projectName},
        }

        # Route (a): auto-compile through the existing competition gate.
        compile_error = None
        try:
            example_input = examples["train"][0]["input"]
            export_payload = ExportPayload(**door_engine.lower_to_payload(task_id, example_input, payload.path))
            result = _competition_check(export_payload)
            if result["validation"]["ready"]:
                sidecar = {**recipe, "exportRoute": "auto_compile",
                           "validation": result["validation"], "score": result["score"],
                           "baseline": result["baseline"], "deltaPoints": result["deltaPoints"]}
                artifact = project_bridge.save_artifact(result["model"].SerializeToString(), task_id, sidecar)
                return {"status": "passed", "route": "auto_compile", "artifact": artifact["onnx"],
                        "artifactPaths": artifact, "doorSets": sets, **_check_response(result)}
            compile_error = json.dumps(result["validation"]["firstFailure"] or result["validation"]["sets"])
        except (door_engine.DoorError, ValueError, ValidationError) as exc:
            compile_error = str(exc)

        # Route (b): recipe-only — a proven, replayable spec for the agents.
        recipe["exportRoute"] = "recipe_only"
        recipe["autoCompileError"] = compile_error
        saved = project_bridge.save_recipe(task_id, recipe)
        return {"status": "passed", "route": "recipe_only", "recipe": saved["recipe"],
                "doorSets": sets, "autoCompileError": compile_error,
                "baseline": project_bridge.task_baseline(task_id)}
    except door_engine.DoorError as exc:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": str(exc)})
    except Exception as exc:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": str(exc)})


@app.post("/api/door/rule-note")
def door_rule_note(payload: DoorExportPayload):
    """Save the human's rule hypothesis even WITHOUT a winning path — words
    are competition value too (they seed the agents and the wizard)."""
    try:
        task_id = _door_task_id(payload)
        if not payload.ruleHypothesis.strip():
            return JSONResponse(status_code=400, content={"status": "failed", "reason": "write the rule first"})
        examples = _official_task_examples(task_id)
        note: dict[str, Any] = {
            "schema": "neurogolf-rulenote/v1",
            "taskId": task_id,
            "ruleHypothesis": payload.ruleHypothesis.strip(),
            "exportRoute": "hypothesis_only",
        }
        try:
            state = door_engine.path_state(examples["train"], payload.path)
            note["path"] = payload.path
            note["pathDistance"] = state["distance"]
        except door_engine.DoorError:
            note["path"] = payload.path
        saved = project_bridge.save_recipe(task_id, note)
        return {"status": "saved", "recipe": saved["recipe"]}
    except Exception as exc:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": str(exc)})


def _wizard_prompt(task_id: str, examples: dict[str, Any], payload: DoorWizardPayload) -> str:
    train = [{"input": item["input"], "output": item["output"]} for item in examples["train"]]
    library = [{"op": op, "params": [p["name"] for p in meta["params"]], "blurb": meta["blurb"]}
               for op, meta in door_engine.DOORS.items()]
    return (
        "You are the Wizard in the NeuroGolf Doors Maze game (see GUI/DOORS_DESIGN.md in this repo).\n"
        f"Task: {task_id}. A human is solving an ARC grid puzzle by composing 'doors' (grid transforms).\n\n"
        f"TRAIN EXAMPLES (input -> output grids, colors 0-9, 0=background):\n{json.dumps(train)}\n\n"
        f"DOOR LIBRARY (op, param names, effect):\n{json.dumps(library)}\n\n"
        f"CURRENT PATH (applied so far): {json.dumps(payload.path)}\n"
        f"HUMAN'S RULE HYPOTHESIS: {payload.ruleHypothesis or '(none given)'}\n\n"
        "You may read The_400_Tasks/Task_by_Task/" + task_id + ".md and the Onnx_Advanced/ reference docs "
        "(task_techniques_index.md, Algorithm_Library/per_family_starting_points.md, by_symptom.md) for context.\n\n"
        "Figure out the transformation rule and how to express it. Respond with ONLY a JSON object, no prose:\n"
        '{"ruleHypothesis": "<one-sentence rule>", '
        '"doorPath": [{"op": "<door op>", "params": {...}}, ...], '
        '"notes": "<if the doors cannot express the rule, give a precise numpy-style algorithm here>"}\n'
        "doorPath must use ONLY ops from the library above, applied from the ORIGINAL input (not the current path). "
        "Use an empty doorPath if no door sequence can express the rule."
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:index + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


@app.post("/api/door/wizard")
def door_wizard(payload: DoorWizardPayload):
    """Ask the Wizard: spawn a ONE-SHOT headless CLI agent (claude -p by
    default, codex exec as alternative) with cwd = project root so it can read
    Onnx_Advanced/ itself. No polling loops; runs only when the player asks.
    The subprocess inherits the launcher's kill-on-close JobObject."""
    import shutil
    import subprocess

    try:
        task_id = _door_task_id(payload)
        examples = _official_task_examples(task_id)
        backend = (payload.backend or "claude").strip().lower()
        if backend not in ("claude", "codex"):
            return JSONResponse(status_code=400, content={"status": "failed", "reason": "backend must be claude or codex"})
        exe = shutil.which(backend)
        if not exe:
            return JSONResponse(status_code=400, content={
                "status": "failed", "reason": f"{backend} CLI not found on PATH"})
        prompt = _wizard_prompt(task_id, examples, payload)
        if backend == "claude":
            command = [exe, "-p", prompt, "--output-format", "json"]
        else:
            command = [exe, "exec", prompt]
        cwd = project_bridge.PROJECT_ROOT or ROOT
        # Spawn with a cleaned environment: when the GUI itself was launched
        # from inside a Claude Code session, inherited ANTHROPIC_API_KEY /
        # CLAUDE_* vars would override the user's normal CLI login.
        import os as _os

        env = {key: value for key, value in _os.environ.items()
               if key != "ANTHROPIC_API_KEY" and not key.startswith(("CLAUDE", "AI_AGENT"))}
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=300, cwd=str(cwd), env=env)
        raw = (completed.stdout or "").strip()
        if backend == "claude":
            try:
                envelope = json.loads(raw)
                raw = envelope.get("result", raw) if isinstance(envelope, dict) else raw
            except json.JSONDecodeError:
                pass
        answer = _extract_json_object(raw)
        if not answer:
            return JSONResponse(status_code=502, content={
                "status": "failed", "reason": "wizard gave no parseable JSON",
                "stdout": raw[-2000:], "stderr": (completed.stderr or "")[-500:]})
        response: dict[str, Any] = {"status": "ok", "backend": backend,
                                    "ruleHypothesis": answer.get("ruleHypothesis", ""),
                                    "doorPath": answer.get("doorPath") or [],
                                    "notes": answer.get("notes", "")}
        if response["doorPath"]:
            try:
                replay = door_engine.path_state(examples["train"], response["doorPath"])
                response["replay"] = {"distance": replay["distance"], "win": replay["win"]}
            except door_engine.DoorError as exc:
                response["replay"] = {"error": str(exc)}
        return response
    except subprocess.TimeoutExpired:
        return JSONResponse(status_code=504, content={"status": "failed", "reason": "wizard timed out (5 min)"})
    except Exception as exc:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": str(exc)})


def _campaign_seed() -> dict[str, Any]:
    seed_path = ROOT / "doors_campaign.json"
    if seed_path.is_file():
        try:
            return json.loads(seed_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


@app.get("/api/door/campaign")
def door_campaign():
    seed = _campaign_seed()
    return {"tutorials": seed.get("tutorials", []), "leads": seed.get("leads", []),
            "queue": project_bridge.campaign_queue()}


@app.get("/api/task-rule/{task_id}")
def task_rule(task_id: str):
    """The project's best-known answer for a task, for the 'reveal the rule'
    panels: the documented ## Rule section (taskNNN.md), any campaign ghost
    door-path, and recent saved recipes/notes. Read-only."""
    task_id = task_id.strip().lower()
    if not TASK_ID_RE.match(task_id):
        return JSONResponse(status_code=400, content={"status": "failed", "reason": "taskId must match taskXXX"})
    doc = project_bridge.task_rule_doc(task_id)
    seed = _campaign_seed()
    ghost = next((item.get("ghostPath") for item in
                  [*seed.get("tutorials", []), *seed.get("leads", []), *seed.get("solutions", [])]
                  if item.get("taskId") == task_id and item.get("ghostPath")), None)
    return {**doc, "taskId": task_id, "ghostPath": ghost,
            "recipes": project_bridge.list_recipes(task_id)}


@app.get("/api/project-status")
def project_status():
    return project_bridge.project_status()


@app.get("/api/task-baseline/{task_id}")
def task_baseline(task_id: str):
    task_id = task_id.strip().lower()
    if not TASK_ID_RE.match(task_id):
        return JSONResponse(status_code=400, content={"status": "failed", "reason": "taskId must match taskXXX"})
    return project_bridge.task_baseline(task_id)


@app.get("/best/manifest.json")
def best_manifest():
    manifest = project_bridge.best_manifest()
    if manifest is None:
        return JSONResponse(status_code=404, content={"status": "failed", "reason": "submissions/submission.zip not found"})
    return manifest


@app.get("/api/best-zip")
def best_zip():
    zip_path = project_bridge.best_zip_path()
    if zip_path is None:
        return JSONResponse(status_code=404, content={"status": "failed", "reason": "submissions/submission.zip not found"})
    return FileResponse(zip_path, filename=zip_path.name, media_type="application/zip")


@app.get("/api/best-onnx/{task_id}")
def best_onnx(task_id: str):
    try:
        data, _source = _best_onnx_bytes(task_id)
    except (ValueError, FileNotFoundError) as exc:
        return JSONResponse(status_code=404, content={"status": "failed", "reason": str(exc)})
    return Response(content=data, media_type="application/octet-stream", headers={
        "Content-Disposition": f'attachment; filename="{task_id.strip().lower()}.onnx"',
    })


@app.get("/tasks/{task_id}.json")
def task_json(task_id: str):
    task_id = task_id.strip().lower()
    if not re.fullmatch(r"task\d{3}", task_id):
        return JSONResponse(status_code=404, content={"status": "failed", "reason": "invalid task id"})
    try:
        return _official_task_examples(task_id)
    except ValidationError as exc:
        return JSONResponse(status_code=404, content={"status": "failed", "reason": str(exc)})


if CLIENT_DIST.exists():
    app.mount("/", StaticFiles(directory=str(CLIENT_DIST), html=True), name="static")
