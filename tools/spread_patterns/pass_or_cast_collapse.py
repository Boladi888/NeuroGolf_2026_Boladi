"""Collapse ``Cast(bool_a, F32) + Cast(bool_b, F32)`` into
``Cast(Or(bool_a, bool_b), F32)`` when ``bool_a`` and ``bool_b`` are
provably mutually exclusive predicates on the same value.

Origin: 2026-05-19 task196 worker insight — ``Cast(gt25, F32) +
Cast(lt15, F32)`` was equivalent to ``Cast(Or(gt25, lt15), F32)``
because ``gt25 = Greater(x, 25)`` and ``lt15 = Less(x, 15)`` are
mutually exclusive (no value is simultaneously > 25 and < 15).

Math equivalence under mutual exclusion: when at most one of two bool
predicates is True at a time, ``int(a) + int(b) == int(a OR b)``
exactly, because the sum is 0 (neither true), 1 (exactly one true),
or never 2 (both true is impossible).

Detected mutual-exclusion patterns (conservative — only obvious cases):

  * ``Greater(x, hi) + Less(x, lo)`` with ``hi >= lo``
  * ``GreaterOrEqual(x, hi) + Less(x, lo)`` with ``hi >= lo``
  * ``Greater(x, hi) + LessOrEqual(x, lo)`` with ``hi > lo``
  * ``Greater(x, c) + Less(x, c)``: special-case, both compared against
    same c, mutually exclusive (no x is both > c and < c).
  * symmetric: ``Less + Greater`` order also matched.

Savings:
  * One Cast node + one intermediate F32 tensor eliminated.
  * The eliminated F32 tensor is typically `[1, 10, 30, 30]` so ~36k
    bytes per fold — but the actual count depends on tensor shape.

Strategy:
  1. For every Add node with two inputs, both produced by Cast.
  2. Each Cast must convert a Bool-producing op output to F32.
  3. The two bool ops must be a Greater/Less pair on the same `x`.
  4. The thresholds must satisfy the mutual-exclusion condition above.
  5. Each Cast output must be consumed only by this Add.
  6. Rewrite: build a new Or(bool_a, bool_b), feed into a new Cast to
     F32 with the Add's output name. Drop both Casts and the Add.

Safety:
  * Conservative threshold check — only fires when arithmetic confirms
    mutual exclusion. Other patterns of mutual exclusion (e.g. derived
    via Equal on disjoint values) are out of scope.
  * Both Cast outputs must be single-use (else orphan an op).
  * If the Add output is a graph output, preserve its name on the new
    Cast.

Out of scope (future):
  * N-way mutually exclusive predicates (e.g. one-hot color groups);
    extending to Sum of N Casts under pairwise mutual exclusion.
  * Mutual exclusion via Equal on disjoint constants.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tools.spread_patterns._base import SpreadPattern, SpreadResult


def _init_scalar(graph: onnx.GraphProto, name: str):
    for init in graph.initializer:
        if init.name != name:
            continue
        try:
            arr = numpy_helper.to_array(init)
        except Exception:  # noqa: BLE001
            return None
        if arr.size != 1:
            return None
        try:
            return float(arr.flatten()[0])
        except (TypeError, ValueError):
            return None
    return None


def _build_consumer_map(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    out: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                out.setdefault(inp, []).append(node)
    return out


def _is_mutually_exclusive(
    op_a: str, val_a: float, op_b: str, val_b: float
) -> bool:
    """Return True iff ``op_a(x, val_a)`` and ``op_b(x, val_b)`` can
    never both be True for the same x. Normalizes so op_a is the
    "greater" family."""
    GREATER_FAMILY = {"Greater": ">", "GreaterOrEqual": ">="}
    LESS_FAMILY = {"Less": "<", "LessOrEqual": "<="}

    def normalize(op_x, v_x, op_y, v_y):
        if op_x in GREATER_FAMILY and op_y in LESS_FAMILY:
            return op_x, v_x, op_y, v_y
        if op_x in LESS_FAMILY and op_y in GREATER_FAMILY:
            return op_y, v_y, op_x, v_x
        return None

    norm = normalize(op_a, val_a, op_b, val_b)
    if norm is None:
        return False
    g_op, g_val, l_op, l_val = norm
    # Mutually exclusive iff there is no x with g_val < x < l_val (or with
    # adjusted bounds based on strictness).
    # x > g_val AND x < l_val has solutions iff g_val < l_val
    # x > g_val AND x <= l_val has solutions iff g_val < l_val
    # x >= g_val AND x < l_val has solutions iff g_val < l_val
    # x >= g_val AND x <= l_val has solutions iff g_val <= l_val
    if g_op == "GreaterOrEqual" and l_op == "LessOrEqual":
        return g_val > l_val
    return g_val >= l_val


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
        return f"{prefix}_orcast_{counter[0]}"

    for add_node in graph.node:
        if add_node.op_type != "Add" or len(add_node.input) != 2:
            continue
        a, b = add_node.input
        cast_a = producer.get(a)
        cast_b = producer.get(b)
        if cast_a is None or cast_a.op_type != "Cast":
            continue
        if cast_b is None or cast_b.op_type != "Cast":
            continue
        # Both casts must target F32 (the Add result type).
        def _cast_to_dtype(node):
            for at in node.attribute:
                if at.name == "to":
                    return at.i
            return None

        if _cast_to_dtype(cast_a) != TensorProto.FLOAT:
            continue
        if _cast_to_dtype(cast_b) != TensorProto.FLOAT:
            continue
        bool_a_name = cast_a.input[0] if cast_a.input else ""
        bool_b_name = cast_b.input[0] if cast_b.input else ""
        bool_a = producer.get(bool_a_name)
        bool_b = producer.get(bool_b_name)
        if bool_a is None or bool_b is None:
            continue
        if bool_a.op_type not in ("Greater", "Less", "GreaterOrEqual", "LessOrEqual"):
            continue
        if bool_b.op_type not in ("Greater", "Less", "GreaterOrEqual", "LessOrEqual"):
            continue
        # Both predicates must compare the SAME variable input.
        if len(bool_a.input) != 2 or len(bool_b.input) != 2:
            continue
        if bool_a.input[0] != bool_b.input[0]:
            continue
        val_a = _init_scalar(graph, bool_a.input[1])
        val_b = _init_scalar(graph, bool_b.input[1])
        if val_a is None or val_b is None:
            continue
        if not _is_mutually_exclusive(bool_a.op_type, val_a, bool_b.op_type, val_b):
            continue
        # Casts must be single-use.
        if len(consumers.get(cast_a.output[0], [])) != 1:
            continue
        if len(consumers.get(cast_b.output[0], [])) != 1:
            continue
        matches.append({
            "add": add_node,
            "cast_a": cast_a,
            "cast_b": cast_b,
            "bool_a_name": bool_a_name,
            "bool_b_name": bool_b_name,
        })

    if not matches:
        return SpreadResult(
            pattern_name="or_cast_collapse",
            verdict="no-op",
            notes="no Add(Cast(bool_a, F32), Cast(bool_b, F32)) with mutually-exclusive predicates detected",
        )

    drop_nodes = set()
    appended: dict[int, list[onnx.NodeProto]] = {}

    for m in matches:
        add_node = m["add"]
        drop_nodes.add(id(add_node))
        drop_nodes.add(id(m["cast_a"]))
        drop_nodes.add(id(m["cast_b"]))
        or_out = f"{add_node.output[0]}_orcast_bool"
        or_node = helper.make_node(
            "Or",
            inputs=[m["bool_a_name"], m["bool_b_name"]],
            outputs=[or_out],
            name=_uniq(f"{add_node.name or 'add'}_or"),
        )
        cast_node = helper.make_node(
            "Cast",
            inputs=[or_out],
            outputs=[add_node.output[0]],
            name=_uniq(f"{add_node.name or 'add'}_cast"),
            to=TensorProto.FLOAT,
        )
        appended[id(add_node)] = [or_node, cast_node]

    new_nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        if id(node) in drop_nodes:
            if id(node) in appended:
                new_nodes.extend(appended[id(node)])
            continue
        new_nodes.append(node)

    dropped_inter = set()
    for m in matches:
        dropped_inter.add(m["cast_a"].output[0])
        dropped_inter.add(m["cast_b"].output[0])
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
        pattern_name="or_cast_collapse",
        verdict="applied",
        new_model=new_model,
        notes=f"folded {len(matches)} Add(Cast,Cast) over mutually-exclusive predicates into Cast(Or)",
        extras={"pairs_folded": len(matches)},
    )


PATTERN = SpreadPattern(
    name="or_cast_collapse",
    category="math-equivalent",
    description="collapse Add(Cast(bool_a,F32),Cast(bool_b,F32)) -> Cast(Or(bool_a,bool_b),F32) when predicates are mutually exclusive",
    apply=_apply,
    safe_by_construction=True,
)
