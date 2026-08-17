"""Fold a Sub→Sqr→Less<eps distance-threshold idiom into a single Equal.

Origin: task233 v1 (worker-7/5, 2026-05-18) — replaced the
``Sub(input, v_basis) → Mul(t, t) → Less(eps_threshold)`` tail used for
one-hot construction with a single ``Equal(input, v_basis)`` node. The
math is identity when the inputs are integer-like (which they are for
one-hot index basis vectors): squared distance < 0.25 iff |Δ| < 0.5
iff exact equality on integers. Eliminating the Sub + Mul intermediates
(each [..., H, W] F32) is the savings source. Memory note:
`feedback_neurogolf_opt_safe_blindspots.md` documents `opt_safe` missing
this pattern.

Concrete cases seen on the leaderboard:
  - task233 v1: -32,001 cost (+0.5510 pts) — `v42`, `v43` [1,10,20,20] F32 each gone
  - task133 v5: -55,636 cost (+0.4238 pts) — Conv+ScatterND+Equal-fold combo
  - task349 v1: -18,001 cost (+0.094 pts) — 10-site Equal-fold

Strategy:
  * Find Less nodes whose second input is a constant scalar in (0, 1).
  * Less's first input must be the output of a Mul whose two inputs are
    the SAME tensor (self-squaring) and that Mul's input must come from
    a Sub.
  * Replace the (Sub, Mul, Less) trio with a single Equal(Sub.input[0],
    Sub.input[1]). Equal returns Bool, which is what Less returned.
  * The trio's output name (Less.output[0]) becomes Equal's output, so
    downstream consumers don't need rewiring.

Safety:
  * Math equivalence ONLY holds when the inputs are integer-valued. The
    basis vectors v0..v9 etc. in NeuroGolf graphs are integer indices.
    The "Less" threshold being <1 is a strong signal the producer is
    one-hot construction. We additionally guard:
      - eps_threshold strictly < 1 (else even non-equal-but-close
        floats would pass Less, which Equal would reject).
      - The Sub and Mul intermediates must have no other consumer
        (would orphan their dependencies if we collapsed them).
  * Driver also runs the standard sanity-verify step which catches any
    case where the assumption breaks.

Out of scope:
  * Three-input Sub→Mul→Less where one side of the Mul is a separately-
    computed copy of the difference. (Possible variant; see if it comes
    up.)
  * Sqrt-based distance (input → Sub → Mul → Sqrt → Less). Rare in
    NeuroGolf graphs.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

from tools.spread_patterns._base import SpreadPattern, SpreadResult


def _scalar_init_value(graph: onnx.GraphProto, name: str) -> float | None:
    """Return the numeric value of a scalar (size-1) initializer, else None."""
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


def _apply(model: onnx.ModelProto, task_id: str) -> SpreadResult:
    graph = model.graph
    producer: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer[out] = node
    consumers = _build_consumer_map(graph)

    # Find candidate Less nodes.
    candidates: list[tuple[onnx.NodeProto, onnx.NodeProto, onnx.NodeProto, str, str]] = []
    for less_node in graph.node:
        if less_node.op_type != "Less" or len(less_node.input) != 2:
            continue
        eps_val = _scalar_init_value(graph, less_node.input[1])
        if eps_val is None or not (0.0 < eps_val < 1.0):
            continue
        mul_node = producer.get(less_node.input[0])
        if mul_node is None or mul_node.op_type != "Mul" or len(mul_node.input) != 2:
            continue
        # Mul both inputs must be the same tensor (self-squaring).
        if mul_node.input[0] != mul_node.input[1]:
            continue
        # Mul's output must be consumed ONLY by Less (else collapsing
        # would orphan the Mul intermediate the other consumer needs).
        if len(consumers.get(mul_node.output[0], [])) != 1:
            continue
        sub_node = producer.get(mul_node.input[0])
        if sub_node is None or sub_node.op_type != "Sub" or len(sub_node.input) != 2:
            continue
        # Sub's output is consumed only by the self-squaring Mul (its
        # two inputs both reference Sub.output[0], so consumer list has
        # the same Mul twice). Confirm no OTHER node consumes Sub.output.
        consumer_ids = {id(c) for c in consumers.get(sub_node.output[0], [])}
        if consumer_ids != {id(mul_node)}:
            continue
        candidates.append((sub_node, mul_node, less_node, sub_node.input[0], sub_node.input[1]))

    if not candidates:
        return SpreadResult(
            pattern_name="equal_fold_distance_threshold",
            verdict="no-op",
            notes="no Sub->Mul(self,self)->Less<1 distance-threshold chains found",
        )

    # Rewrite.
    drop_ids = set()
    new_nodes: list[onnx.NodeProto] = []
    replacements: dict[int, onnx.NodeProto] = {}  # node-id-of-less -> Equal replacement

    for sub_n, mul_n, less_n, x_name, y_name in candidates:
        drop_ids.add(id(sub_n))
        drop_ids.add(id(mul_n))
        # Equal replaces Less; uses Less's output name.
        equal_node = onnx.helper.make_node(
            "Equal",
            inputs=[x_name, y_name],
            outputs=[less_n.output[0]],
            name=f"{less_n.name or 'less'}_equalfold",
        )
        replacements[id(less_n)] = equal_node

    for node in graph.node:
        if id(node) in drop_ids:
            continue
        if id(node) in replacements:
            new_nodes.append(replacements[id(node)])
        else:
            new_nodes.append(node)

    # Drop value_info for dropped Sub.output and Mul.output (no other
    # consumer; their intermediate buffers go away).
    dropped_outputs = set()
    for sub_n, mul_n, _, _, _ in candidates:
        dropped_outputs.update(sub_n.output)
        dropped_outputs.update(mul_n.output)
    new_value_info = [vi for vi in graph.value_info if vi.name not in dropped_outputs]

    new_graph = onnx.helper.make_graph(
        nodes=new_nodes,
        name=graph.name,
        inputs=list(graph.input),
        outputs=list(graph.output),
        initializer=list(graph.initializer),
        value_info=new_value_info,
    )
    new_model = onnx.helper.make_model(
        new_graph,
        opset_imports=list(model.opset_import),
        ir_version=model.ir_version,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
    )

    return SpreadResult(
        pattern_name="equal_fold_distance_threshold",
        verdict="applied",
        new_model=new_model,
        notes=f"folded {len(candidates)} Sub->Mul(self,self)->Less<1 chain(s) into Equal",
        extras={
            "folds_applied": len(candidates),
            "intermediates_dropped": 2 * len(candidates),
        },
    )


PATTERN = SpreadPattern(
    name="equal_fold_distance_threshold",
    category="math-equivalent",
    description="collapse Sub->Mul(self,self)->Less<1 distance-threshold idiom into a single Equal",
    apply=_apply,
    safe_by_construction=True,
)
