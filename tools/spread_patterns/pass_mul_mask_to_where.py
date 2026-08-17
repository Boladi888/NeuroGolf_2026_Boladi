"""Rewrite ``Mul(Cast(bool, F32), V)`` to ``Where(bool, V, 0)`` —
keeps the mask in bool dtype until the Where, eliminating the F32
mask intermediate.

Origin: 2026-05-20 worker insight on task021 — saved -3,119 bytes by
replacing F32 mask Mul chains with Where(bool, value, 0). The F32
intermediate (typically [1, C, H, W] = 4·C·H·W bytes) disappears; the
new Where takes a 1-byte/elem bool predicate.

Math equivalence: when mask ∈ {0.0, 1.0} (which it is when produced
by Cast(bool, F32)):
  Mul(mask_f32, V) == V * 1.0 (= V) where mask True
                   == V * 0.0 (= 0) where mask False
  Where(bool, V, 0) == V where True, 0 where False
Same result element-wise. Dtype of output: V's dtype (Where preserves
the data dtype; the condition is bool).

Strategy:
  1. Find Mul nodes with two inputs.
  2. Exactly one input is the output of a Cast node, and the Cast's
     source is a bool-producing op (Equal/Greater/Less/...).
  3. The Cast output must be consumed only by this Mul (else orphan).
  4. The Cast target dtype must be F32.
  5. Build a zero constant matching V's dtype (default F32).
  6. Replace the Mul with Where(bool_src, V, zero_init).
  7. Drop the Cast node.

Safety:
  * The bool mask path stays single-purpose. If the Cast feeds OTHER
    consumers (e.g., another Mul), we leave it alone (the F32 form
    is needed elsewhere).
  * V's dtype: we default the zero constant to F32 to match the
    eliminated Mul's output dtype. Most NeuroGolf graphs use F32
    throughout the mask-gating chain.

Out of scope:
  * The dual case where BOTH Mul inputs are Cast(bool, F32) — that's
    bool_andfold's territory.
  * Cases where the value being gated is itself a fused-multiply
    sub-expression. Could extend but rare.

Cross-ref:
  * `pass_bool_andfold` handles Mul(bool_cast, bool_cast).
  * `pass_or_cast_collapse` handles Add(bool_cast, bool_cast) under
    mutual exclusion.
  * This pattern handles Mul(bool_cast, value).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tools.spread_patterns._base import SpreadPattern, SpreadResult


BOOL_PRODUCING_OPS = frozenset({
    "Equal", "Greater", "Less", "Not", "And", "Or", "Xor",
    "GreaterOrEqual", "LessOrEqual",
})


def _build_consumer_map(graph):
    out = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                out.setdefault(inp, []).append(node)
    return out


def _cast_to_dtype(node):
    for a in node.attribute:
        if a.name == "to":
            return a.i
    return None


def _apply(model: onnx.ModelProto, task_id: str) -> SpreadResult:
    graph = model.graph
    producer: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer[out] = node
    consumers = _build_consumer_map(graph)

    matches: list[dict] = []
    for mul_node in graph.node:
        if mul_node.op_type != "Mul" or len(mul_node.input) != 2:
            continue
        a, b = mul_node.input
        cast_a = producer.get(a)
        cast_b = producer.get(b)

        def _is_bool_cast(node):
            if node is None or node.op_type != "Cast":
                return None
            if _cast_to_dtype(node) != TensorProto.FLOAT:
                return None
            if not node.input:
                return None
            src = producer.get(node.input[0])
            if src is None or src.op_type not in BOOL_PRODUCING_OPS:
                return None
            return node.input[0]

        bool_src_a = _is_bool_cast(cast_a)
        bool_src_b = _is_bool_cast(cast_b)

        # Skip the both-are-bool-casts case (bool_andfold's territory)
        if bool_src_a and bool_src_b:
            continue

        if bool_src_a:
            cast_node = cast_a
            value_input = b
            bool_src = bool_src_a
        elif bool_src_b:
            cast_node = cast_b
            value_input = a
            bool_src = bool_src_b
        else:
            continue

        # The Cast must be single-use (else we'd orphan it; leaving the
        # Cast means no savings on the F32 intermediate)
        if len(consumers.get(cast_node.output[0], [])) != 1:
            continue

        matches.append({
            "mul": mul_node,
            "cast": cast_node,
            "bool_src": bool_src,
            "value_input": value_input,
        })

    if not matches:
        return SpreadResult(
            pattern_name="mul_mask_to_where",
            verdict="no-op",
            notes="no Mul(Cast(bool,F32), value) patterns detected",
        )

    drop_nodes = set()
    appended: dict[int, list[onnx.NodeProto]] = {}
    new_initializers: list[onnx.TensorProto] = list(graph.initializer)
    zero_init_name = "_mul_mask_to_where_zero"

    # Add a shared F32 scalar 0 init if not already present
    if not any(init.name == zero_init_name for init in new_initializers):
        zero_arr = np.array([0.0], dtype=np.float32)
        new_initializers.append(numpy_helper.from_array(zero_arr, name=zero_init_name))

    counter = [0]

    def _uniq(prefix):
        counter[0] += 1
        return f"{prefix}_mmw_{counter[0]}"

    for m in matches:
        drop_nodes.add(id(m["mul"]))
        drop_nodes.add(id(m["cast"]))
        where_node = helper.make_node(
            "Where",
            inputs=[m["bool_src"], m["value_input"], zero_init_name],
            outputs=[m["mul"].output[0]],
            name=_uniq(f"{m['mul'].name or 'mul'}_to_where"),
        )
        appended[id(m["mul"])] = [where_node]

    new_nodes = []
    for node in graph.node:
        if id(node) in drop_nodes:
            if id(node) in appended:
                new_nodes.extend(appended[id(node)])
            continue
        new_nodes.append(node)

    # Drop value_info for the eliminated Cast outputs
    dropped_names = {m["cast"].output[0] for m in matches}
    new_value_info = [vi for vi in graph.value_info if vi.name not in dropped_names]

    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=graph.name,
        inputs=list(graph.input),
        outputs=list(graph.output),
        initializer=new_initializers,
        value_info=new_value_info,
    )
    new_model = helper.make_model(
        new_graph,
        opset_imports=list(model.opset_import),
        ir_version=model.ir_version,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
    )

    return SpreadResult(
        pattern_name="mul_mask_to_where",
        verdict="applied",
        new_model=new_model,
        notes=f"rewrote {len(matches)} Mul(bool_cast, value) into Where(bool, value, 0); eliminated F32 mask intermediate(s)",
        extras={"rewrites": len(matches)},
    )


PATTERN = SpreadPattern(
    name="mul_mask_to_where",
    category="math-equivalent",
    description="rewrite Mul(Cast(bool,F32), V) -> Where(bool, V, 0) — eliminates the F32 mask intermediate",
    apply=_apply,
    safe_by_construction=True,
)
