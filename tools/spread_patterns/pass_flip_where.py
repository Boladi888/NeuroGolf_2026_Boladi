"""Simplify ``Mul(b, Sub(1, Where(cond, b, 0)))`` to
``Where(cond, 0, b)`` when ``b`` is provably binary (0/1).

Origin: 2026-05-19 task196 worker insight — col is a binary 0/1
tensor, so ``col * (1 - Where(cond, col, 0))`` equals
``Where(cond, 0, col)``. The rewrite eliminates two intermediates
(the Sub-from-1 result and the original Where output).

Math truth-table (b in {0, 1}):
  b=0:        LHS = 0 * (1 - Where(cond, 0, 0))  = 0
              RHS = Where(cond, 0, 0)            = 0
  b=1, c=F:   LHS = 1 * (1 - 0)                  = 1
              RHS = Where(F, 0, 1)               = 1
  b=1, c=T:   LHS = 1 * (1 - 1)                  = 0
              RHS = Where(T, 0, 1)               = 0
All three cases match.

For b NOT binary (e.g. b=2), LHS != RHS — binary-ness is essential.

Strategy:
  * Find ``Mul(b, sub_out)`` where:
      - sub_out is produced by ``Sub(one_const, where_out)`` and
      - where_out is produced by ``Where(cond, w_a, w_b)`` with
        ``w_a == b`` and ``w_b`` a constant zero, and
      - one_const is a constant scalar 1.0 (or all-ones init), and
      - b is provably binary (traced back to ``Cast(bool, F32)``).
  * Rewrite: emit a new ``Where(cond, zero_const, b)`` with the same
    output name as the original Mul. Drop Mul, Sub, original Where.

Safety:
  * Each intermediate (Sub, Where, Cast-from-1) must be single-use
    (consumed only by the chain we're collapsing). Otherwise drop
    would orphan downstream readers.
  * b's binary-ness: the simplest proof is that b == Cast(bool, F32).
    We accept any direct ancestor that's a Cast from BOOL dtype.
  * The "1" and "0" constants are checked via initializer-or-scalar.
    For 1, we accept any tensor where every element is exactly 1.0.
    For 0, same: every element 0.0.

Out of scope (future):
  * The mirror form ``Mul(Sub(1, Where(c, b, 0)), b)`` — same idea,
    different arg order. Easy to add.
  * b proven binary via other means (e.g., direct 0/1 init, or by
    op-type ancestry like ReduceMax of a single bool).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tools.spread_patterns._base import SpreadPattern, SpreadResult


def _init_array(graph: onnx.GraphProto, name: str):
    for init in graph.initializer:
        if init.name == name:
            try:
                return numpy_helper.to_array(init)
            except Exception:  # noqa: BLE001
                return None
    return None


def _is_all_value(graph: onnx.GraphProto, name: str, target: float) -> bool:
    arr = _init_array(graph, name)
    if arr is None or arr.size == 0:
        return False
    return bool(np.all(np.abs(arr - target) < 1e-6))


def _is_cast_from_bool(graph_producer: dict, name: str) -> bool:
    node = graph_producer.get(name)
    if node is None or node.op_type != "Cast":
        return False
    # The Cast's input must be from a bool-producing op
    src_name = node.input[0] if node.input else ""
    src = graph_producer.get(src_name)
    if src is None:
        return False
    return src.op_type in ("Equal", "Greater", "Less", "Not", "And", "Or", "Xor",
                           "GreaterOrEqual", "LessOrEqual")


def _build_consumer_map(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    out: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                out.setdefault(inp, []).append(node)
    return out


def _apply(model: onnx.ModelProto, task_id: str) -> SpreadResult:
    graph = model.graph
    producer: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer[out] = node
    consumers = _build_consumer_map(graph)

    matches: list[dict] = []
    counter = [0]

    def _uniq(prefix: str) -> str:
        counter[0] += 1
        return f"{prefix}_flipwhere_{counter[0]}"

    for mul_node in graph.node:
        if mul_node.op_type != "Mul" or len(mul_node.input) != 2:
            continue
        # Both arg orders: try (b, sub_out) then (sub_out, b)
        for b_idx, sub_idx in [(0, 1), (1, 0)]:
            b_name = mul_node.input[b_idx]
            sub_name = mul_node.input[sub_idx]
            sub_node = producer.get(sub_name)
            if sub_node is None or sub_node.op_type != "Sub" or len(sub_node.input) != 2:
                continue
            one_name, where_name = sub_node.input
            if not _is_all_value(graph, one_name, 1.0):
                continue
            where_node = producer.get(where_name)
            if where_node is None or where_node.op_type != "Where" or len(where_node.input) != 3:
                continue
            cond_name, wa_name, wb_name = where_node.input
            if wa_name != b_name:
                continue
            if not _is_all_value(graph, wb_name, 0.0):
                continue
            # Single-use intermediates
            if len(consumers.get(sub_node.output[0], [])) != 1:
                continue
            if len(consumers.get(where_node.output[0], [])) != 1:
                continue
            # b must be binary
            if not _is_cast_from_bool(producer, b_name):
                continue
            matches.append({
                "mul": mul_node,
                "sub": sub_node,
                "where": where_node,
                "b_name": b_name,
                "cond_name": cond_name,
                "zero_init_name": wb_name,
            })
            break

    if not matches:
        return SpreadResult(
            pattern_name="flip_where",
            verdict="no-op",
            notes="no Mul(b, Sub(1, Where(c, b, 0))) patterns with provably-binary b detected",
        )

    drop_nodes = set()
    appended: dict[int, list[onnx.NodeProto]] = {}

    for m in matches:
        drop_nodes.add(id(m["mul"]))
        drop_nodes.add(id(m["sub"]))
        drop_nodes.add(id(m["where"]))
        new_where = helper.make_node(
            "Where",
            inputs=[m["cond_name"], m["zero_init_name"], m["b_name"]],
            outputs=[m["mul"].output[0]],
            name=_uniq(f"{m['mul'].name or 'mul'}_flipwhere"),
        )
        appended[id(m["mul"])] = [new_where]

    new_nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        if id(node) in drop_nodes:
            if id(node) in appended:
                new_nodes.extend(appended[id(node)])
            continue
        new_nodes.append(node)

    dropped_inter = set()
    for m in matches:
        dropped_inter.update(m["sub"].output)
        dropped_inter.update(m["where"].output)
    new_value_info = [vi for vi in graph.value_info if vi.name not in dropped_inter]

    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=graph.name,
        inputs=list(graph.input),
        outputs=list(graph.output),
        initializer=list(graph.initializer),
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
        pattern_name="flip_where",
        verdict="applied",
        new_model=new_model,
        notes=f"collapsed {len(matches)} Mul(b, Sub(1, Where(c, b, 0))) into Where(c, 0, b)",
        extras={"folds": len(matches)},
    )


PATTERN = SpreadPattern(
    name="flip_where",
    category="math-equivalent",
    description="rewrite Mul(b, Sub(1, Where(c, b, 0))) -> Where(c, 0, b) when b is provably binary 0/1",
    apply=_apply,
    safe_by_construction=True,
)
