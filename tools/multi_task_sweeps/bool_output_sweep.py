#!/usr/bin/env python3
"""Output-dtype redeclaration sweep (BOOL/F16/UINT8 Track B generalization).

Three patterns this sweep handles:
  A. `... -> Cast(F32) -> output_f32`: drop Cast, declare output to the
     pre-Cast dtype (BOOL, F16, UINT8).
  B. `... -> [dtype-preserving op] -> output_f32` where the chain produces
     a non-F32 dtype: change output declaration to match the produced dtype.
  C. Where the Pad input traces back to a bool-producing op: declare BOOL
     output (Track B generalization confirmed by task069 +2.54 LB Kaggle).

Safety: the official run_network applies `(out > 0.0).astype(float)`. BOOL
{0,1}, F16 {0,1}, UINT8 {0,1} all produce the same threshold result as F32
{0,1}. verify_against_live catches any mismatch.

Usage:
    .venv/Scripts/python tools/multi_task_sweeps/bool_output_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import onnx
from onnx import TensorProto, shape_inference

from tools.multi_task_sweeps._common import (
    iter_canonical_tasks, sweep_task, write_report,
)

BOOL_PRODUCING_OPS = {"Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual",
                      "And", "Or", "Not", "Xor", "IsNaN", "IsInf"}
DTYPE_PRESERVING_OPS = {"Pad", "Concat", "Reshape", "Transpose", "Slice", "Identity",
                        "Unsqueeze", "Squeeze", "Tile", "Expand", "Flatten",
                        "ReduceMax", "ReduceMin", "Max", "Min"}


def _producer_of(g, name):
    for n in g.node:
        if name in n.output:
            return n
    return None


def _infer_output_dtype(model: onnx.ModelProto) -> int | None:
    """Run shape inference and return the inferred dtype of the graph's output tensor."""
    try:
        m = shape_inference.infer_shapes(model)
    except Exception:
        return None
    out_name = m.graph.output[0].name
    # Check value_info
    for vi in list(m.graph.value_info) + list(m.graph.output):
        if vi.name == out_name and vi.type.tensor_type.elem_type:
            return vi.type.tensor_type.elem_type
    return None


def _trace_to_dtype_origin(g, start_tensor, depth=50):
    """Walk back through DTYPE_PRESERVING_OPS; return op_type of first non-preserving producer."""
    cur = start_tensor
    for _ in range(depth):
        prod = _producer_of(g, cur)
        if prod is None:
            return None
        if prod.op_type not in DTYPE_PRESERVING_OPS:
            return prod.op_type
        if not prod.input:
            return None
        cur = prod.input[0]
    return None


def transform_bool_output(model: onnx.ModelProto) -> onnx.ModelProto | None:
    """Main transformer: try Pattern A (drop final Cast), then Pattern B/C
    (redeclare output dtype)."""
    g = model.graph
    if len(g.output) != 1 or len(g.node) < 2:
        return None
    out_name = g.output[0].name
    declared_dtype = g.output[0].type.tensor_type.elem_type
    if declared_dtype != TensorProto.FLOAT:
        return None  # Only optimize F32-declared outputs

    # PATTERN A: final node is Cast(*, to=F32)
    cast_node = None
    cast_idx = -1
    for i, node in enumerate(g.node):
        if node.op_type == "Cast" and out_name in node.output:
            to_attr = next((a for a in node.attribute if a.name == "to"), None)
            if to_attr and to_attr.i == TensorProto.FLOAT:
                cast_node = node
                cast_idx = i
                break

    if cast_node is not None:
        cast_input = cast_node.input[0]
        # Determine the dtype produced upstream
        origin = _trace_to_dtype_origin(g, cast_input)
        if origin in BOOL_PRODUCING_OPS:
            target_dtype = TensorProto.BOOL
        else:
            # Try shape inference to determine the actual dtype of cast_input
            try:
                m_inf = shape_inference.infer_shapes(model)
            except Exception:
                return None
            target_dtype = None
            for vi in m_inf.graph.value_info:
                if vi.name == cast_input:
                    target_dtype = vi.type.tensor_type.elem_type
                    break
            if target_dtype is None or target_dtype == TensorProto.FLOAT:
                return None

        # Build new model: drop Cast, rename cast_input to out_name in producer,
        # declare new dtype.
        new_model = onnx.ModelProto()
        new_model.CopyFrom(model)
        new_g = new_model.graph
        new_nodes = [n for i, n in enumerate(new_g.node) if i != cast_idx]
        for n in new_nodes:
            for j, o in enumerate(n.output):
                if o == cast_input:
                    n.output[j] = out_name
                    break
        del new_g.node[:]
        new_g.node.extend(new_nodes)
        new_g.output[0].type.tensor_type.elem_type = target_dtype
        new_g.ClearField("value_info")
        return new_model

    # PATTERN B/C: graph doesn't end in Cast, but its actual output dtype
    # (inferred) is non-F32. Just redeclare the output.
    inferred = _infer_output_dtype(model)
    if inferred is None or inferred == TensorProto.FLOAT:
        return None
    # Don't pick weird types like int64
    if inferred not in (TensorProto.BOOL, TensorProto.FLOAT16, TensorProto.UINT8,
                        TensorProto.INT8):
        return None
    new_model = onnx.ModelProto()
    new_model.CopyFrom(model)
    new_model.graph.output[0].type.tensor_type.elem_type = inferred
    new_model.graph.ClearField("value_info")
    return new_model


def main():
    results = []
    skipped = 0
    pattern_a_count = 0
    pattern_bc_count = 0
    for tid, onnx_bytes in iter_canonical_tasks():
        r = sweep_task(tid, onnx_bytes, transform_bool_output, "bool_output")
        if r is not None:
            results.append(r)
            print(f"  WIN  {tid}: {r.base_cost:,} -> {r.candidate_cost:,} "
                  f"(+{r.raw_log_e_gain:.3f} LB)")
        else:
            skipped += 1
    write_report("bool_output", results)
    print(f"\nBOOL/F16-output sweep complete: {len(results)} winners / "
          f"{len(results) + skipped} tasks scanned")


if __name__ == "__main__":
    main()
