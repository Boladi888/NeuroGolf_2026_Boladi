"""Common-subexpression elimination for duplicate Slice nodes.

Origin: 2026-05-19 task306 worker insight — 4 "absent" Slice nodes were
duplicating tensors already produced as ``src_0_1`` / ``src_0_2`` /
``src_1_1`` / ``src_1_2`` elsewhere in the graph. Identical inputs,
identical outputs. The worker manually removed the duplicates and
rewired the downstream Where fallback inputs to reuse existing src
tensors.

``onnxsim`` ostensibly does graph-level CSE, but apparently misses
these on the production NeuroGolf graphs. This pattern targets the
specific Slice case (the most common shape we see).

Math equivalence: by construction. Two Slice nodes with the same
``(input, starts, ends, axes, steps)`` produce byte-identical output.

Strategy:
  * For every Slice node, build a hash key from
    ``(input_name, tuple(starts), tuple(ends), tuple(axes), tuple(steps))``.
  * Group by key. In each group with > 1 member, keep the FIRST node
    encountered as the canonical producer.
  * Rewire every other node in the group: change consumers of their
    output to point at the canonical Slice's output.
  * Drop the duplicate Slice nodes.
  * Drop any duplicate starts/ends/axes/steps initializers that are no
    longer consumed (the duplicates' params are dead).

Safety:
  * Skipped if a Slice has dynamic params (we only group when all
    params are extractable as integer lists).
  * Skipped if the duplicate Slice's output is a GRAPH output (we'd
    need to insert an Identity to preserve the output name; future
    work — not common in NeuroGolf).
  * Bool-by-construction: identical Slice = identical output, always.

Out of scope (future):
  * General CSE for other op types (Cast, Mul, etc). Could extend.
  * CSE across Slices with different param-init names but identical
    values (we already handle this — we hash on the VALUES, not the
    param-init names).
"""
from __future__ import annotations

from typing import Any

import onnx
from onnx import helper, numpy_helper

from tools.spread_patterns._base import SpreadPattern, SpreadResult


def _init_value(graph: onnx.GraphProto, name: str):
    for init in graph.initializer:
        if init.name == name:
            try:
                return numpy_helper.to_array(init)
            except Exception:  # noqa: BLE001
                return None
    return None


def _attr_ints(node: onnx.NodeProto, attr_name: str):
    for a in node.attribute:
        if a.name == attr_name and a.type == onnx.AttributeProto.INTS:
            return list(a.ints)
    return None


def _slice_key(graph: onnx.GraphProto, node: onnx.NodeProto):
    """Hash key (tuple) representing this Slice's identity, or None if
    we can't statically determine all params."""
    if len(node.input) >= 2:
        starts = _init_value(graph, node.input[1])
        ends = _init_value(graph, node.input[2]) if len(node.input) >= 3 else None
        axes = _init_value(graph, node.input[3]) if len(node.input) >= 4 else None
        steps = _init_value(graph, node.input[4]) if len(node.input) >= 5 else None
        if starts is None or ends is None:
            return None
        starts_t = tuple(int(x) for x in starts.tolist())
        ends_t = tuple(int(x) for x in ends.tolist())
        axes_t = tuple(int(x) for x in axes.tolist()) if axes is not None else tuple(range(len(starts_t)))
        steps_t = tuple(int(x) for x in steps.tolist()) if steps is not None else (1,) * len(starts_t)
    else:
        starts_l = _attr_ints(node, "starts")
        ends_l = _attr_ints(node, "ends")
        if starts_l is None or ends_l is None:
            return None
        starts_t = tuple(starts_l)
        ends_t = tuple(ends_l)
        axes_t = tuple(_attr_ints(node, "axes") or list(range(len(starts_t))))
        steps_t = (1,) * len(starts_t)
    return (node.input[0], starts_t, ends_t, axes_t, steps_t)


def _build_consumer_map(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    out: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                out.setdefault(inp, []).append(node)
    return out


def _is_graph_output(name: str, graph: onnx.GraphProto) -> bool:
    return any(o.name == name for o in graph.output)


def _apply(model: onnx.ModelProto, task_id: str) -> SpreadResult:
    graph = model.graph

    groups: dict[tuple, list[onnx.NodeProto]] = {}
    for node in graph.node:
        if node.op_type != "Slice":
            continue
        key = _slice_key(graph, node)
        if key is None:
            continue
        groups.setdefault(key, []).append(node)

    dup_groups = [g for g in groups.values() if len(g) > 1]
    if not dup_groups:
        return SpreadResult(
            pattern_name="slice_cse",
            verdict="no-op",
            notes="no duplicate Slice nodes detected",
        )

    rename: dict[str, str] = {}  # old_output -> canonical_output
    drop_nodes = set()  # node ids to drop
    dropped_outputs: set[str] = set()
    inits_maybe_drop: set[str] = set()

    for group in dup_groups:
        canonical = group[0]
        for dup in group[1:]:
            # If dup feeds a graph output, skip (would need Identity).
            if _is_graph_output(dup.output[0], graph):
                continue
            rename[dup.output[0]] = canonical.output[0]
            drop_nodes.add(id(dup))
            dropped_outputs.add(dup.output[0])
            for inp_name in dup.input[1:5]:
                if inp_name:
                    inits_maybe_drop.add(inp_name)

    if not drop_nodes:
        return SpreadResult(
            pattern_name="slice_cse",
            verdict="no-op",
            notes="duplicate Slices found but all feed graph outputs (Identity insertion out of scope)",
        )

    # Rewire downstream consumers.
    new_nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        if id(node) in drop_nodes:
            continue
        if any(inp in rename for inp in node.input):
            new_node = onnx.NodeProto()
            new_node.CopyFrom(node)
            for i, inp in enumerate(new_node.input):
                if inp in rename:
                    new_node.input[i] = rename[inp]
            new_nodes.append(new_node)
        else:
            new_nodes.append(node)

    # Find which inits are now orphaned (no remaining consumer).
    surviving_inputs: set[str] = set()
    for node in new_nodes:
        for inp in node.input:
            if inp:
                surviving_inputs.add(inp)
    inits_to_drop = {name for name in inits_maybe_drop if name not in surviving_inputs}

    final_inits = [init for init in graph.initializer if init.name not in inits_to_drop]

    new_value_info = [vi for vi in graph.value_info if vi.name not in dropped_outputs]

    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=graph.name,
        inputs=list(graph.input),
        outputs=list(graph.output),
        initializer=final_inits,
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
        pattern_name="slice_cse",
        verdict="applied",
        new_model=new_model,
        notes=f"eliminated {len(drop_nodes)} duplicate Slice node(s); dropped {len(inits_to_drop)} now-orphan param init(s)",
        extras={
            "duplicates_dropped": len(drop_nodes),
            "groups_merged": len(dup_groups),
            "inits_dropped": sorted(inits_to_drop),
        },
    )


PATTERN = SpreadPattern(
    name="slice_cse",
    category="math-equivalent",
    description="eliminate duplicate Slice nodes with identical (input, starts, ends, axes, steps) by rewiring consumers to first canonical Slice",
    apply=_apply,
    safe_by_construction=True,
)
