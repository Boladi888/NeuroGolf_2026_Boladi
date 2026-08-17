"""Merge a 2-step ``Slice(axis=A) -> Slice(axis=B)`` chain into one
multi-axis ``Slice(axes=[A, B])``.

Origin: 2026-05-19 task368 worker insight — a two-step Slice chain on
``inp_c`` was replaced by a single 2-axis Slice, saving 12,000 bytes
(the intermediate `[1, 10, 30, 30]` tensor between the two Slices was
eliminated).

Math equivalence: slicing on disjoint axes commutes. ``Slice(A) ->
Slice(B)`` with A != B produces the same result as a single
``Slice(axes=[A, B])`` with the combined starts/ends/steps. Output
shape, dtype, and values are identical.

Strategy:
  * Opset 10+ shape: starts/ends/axes/steps are tensor INPUTS sourced
    from initializers. We extract the numeric values from inits.
  * Opset 9 shape: starts/ends/axes/steps are NODE ATTRIBUTES.
  * Find pairs ``(slice1, slice2)`` where:
      - slice1.output[0] is consumed ONLY by slice2 (else we'd orphan
        the intermediate that other ops need).
      - slice1 and slice2 both operate on a SINGLE axis each, and the
        axes are DIFFERENT.
      - Both Slices have all-1 steps (the common case).
  * Emit a single new Slice with axes=[A, B], starts=[s1, s2],
    ends=[e1, e2], steps=[1, 1].
  * Drop slice1, slice2, and their now-orphaned per-axis init tensors.

Safety:
  * Skipped if either Slice has non-unit steps (commutativity still
    holds, but we'd need to add steps=[1, 1] -> [step1, step2]
    bookkeeping; out of scope for v1).
  * Skipped if axes overlap (axis_A == axis_B) — that's a different
    refinement, not a chain.
  * Skipped if any starts/ends/axes/steps is dynamic (not an init).

Out of scope (future):
  * Chains longer than 2 steps. Cascade by repeated application.
  * Slice + Pad collapse — see `crop_to_size_advisory`.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper

from tools.spread_patterns._base import SpreadPattern, SpreadResult


def _init_value(graph: onnx.GraphProto, name: str):
    """Return numpy array for an initializer, else None."""
    for init in graph.initializer:
        if init.name == name:
            try:
                return numpy_helper.to_array(init)
            except Exception:  # noqa: BLE001
                return None
    return None


def _attr_ints(node: onnx.NodeProto, attr_name: str):
    """Return list[int] for an INTS attribute, or None if absent."""
    for a in node.attribute:
        if a.name == attr_name and a.type == onnx.AttributeProto.INTS:
            return list(a.ints)
    return None


def _extract_slice_params(graph: onnx.GraphProto, node: onnx.NodeProto):
    """Return (starts, ends, axes, steps) as lists of ints, or None if
    the node uses dynamic params we can't read statically.

    Handles both opset 10+ (inputs 1..4 are initializers) and opset 9-
    (everything is attributes)."""
    if len(node.input) >= 2:
        # opset 10+
        starts = _init_value(graph, node.input[1])
        ends = _init_value(graph, node.input[2]) if len(node.input) >= 3 else None
        axes = _init_value(graph, node.input[3]) if len(node.input) >= 4 else None
        steps = _init_value(graph, node.input[4]) if len(node.input) >= 5 else None
        if starts is None or ends is None:
            return None
        starts_l = [int(x) for x in starts.tolist()]
        ends_l = [int(x) for x in ends.tolist()]
        axes_l = [int(x) for x in axes.tolist()] if axes is not None else list(range(len(starts_l)))
        steps_l = [int(x) for x in steps.tolist()] if steps is not None else [1] * len(starts_l)
        return starts_l, ends_l, axes_l, steps_l
    # opset 9- attributes
    starts_l = _attr_ints(node, "starts")
    ends_l = _attr_ints(node, "ends")
    if starts_l is None or ends_l is None:
        return None
    axes_l = _attr_ints(node, "axes") or list(range(len(starts_l)))
    steps_l = [1] * len(starts_l)
    return starts_l, ends_l, axes_l, steps_l


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

    # We require opset 10+ shape for safe rewriting (initializer inputs).
    # Mixing the two shapes is messy; skip opset 9- entirely for v1.
    opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 0)
    if opset < 10:
        return SpreadResult(
            pattern_name="slice_chain_to_multiaxis",
            verdict="no-op",
            notes=f"opset {opset} < 10; skip (attribute-form Slice rewriting out of scope)",
        )

    merge_pairs: list[tuple[onnx.NodeProto, onnx.NodeProto, tuple, tuple]] = []
    # Each: (slice1, slice2, params1, params2)

    for slice2 in graph.node:
        if slice2.op_type != "Slice":
            continue
        first_input = slice2.input[0] if slice2.input else ""
        slice1 = producer.get(first_input)
        if slice1 is None or slice1.op_type != "Slice":
            continue
        # slice1's output must be consumed ONLY by slice2
        if len(consumers.get(slice1.output[0], [])) != 1:
            continue
        params1 = _extract_slice_params(graph, slice1)
        params2 = _extract_slice_params(graph, slice2)
        if params1 is None or params2 is None:
            continue
        s1, e1, a1, st1 = params1
        s2, e2, a2, st2 = params2
        # Each Slice must touch exactly one axis.
        if len(a1) != 1 or len(a2) != 1:
            continue
        # Axes must differ.
        if a1[0] == a2[0]:
            continue
        # Both must have unit step.
        if st1[0] != 1 or st2[0] != 1:
            continue
        merge_pairs.append((slice1, slice2, (s1[0], e1[0], a1[0]), (s2[0], e2[0], a2[0])))

    if not merge_pairs:
        return SpreadResult(
            pattern_name="slice_chain_to_multiaxis",
            verdict="no-op",
            notes="no 2-step single-axis Slice chains detected",
        )

    drop_nodes = set()
    replacements: dict[int, onnx.NodeProto] = {}  # slice2 id -> merged node
    new_inits_to_add: list[onnx.TensorProto] = []
    inits_to_drop: set[str] = set()
    counter = [0]

    def _uniq(prefix: str) -> str:
        counter[0] += 1
        return f"{prefix}_slicemerge_{counter[0]}"

    for slice1, slice2, (s1, e1, a1), (s2, e2, a2) in merge_pairs:
        drop_nodes.add(id(slice1))
        drop_nodes.add(id(slice2))
        # Compose combined axes order to match starts/ends order.
        starts = np.array([s1, s2], dtype=np.int64)
        ends = np.array([e1, e2], dtype=np.int64)
        axes = np.array([a1, a2], dtype=np.int64)
        steps = np.array([1, 1], dtype=np.int64)
        s_name = _uniq("starts")
        e_name = _uniq("ends")
        ax_name = _uniq("axes")
        st_name = _uniq("steps")
        for name, arr in [(s_name, starts), (e_name, ends), (ax_name, axes), (st_name, steps)]:
            new_inits_to_add.append(numpy_helper.from_array(arr, name=name))
        merged = helper.make_node(
            "Slice",
            inputs=[slice1.input[0], s_name, e_name, ax_name, st_name],
            outputs=[slice2.output[0]],
            name=_uniq(slice2.name or "slice_merged"),
        )
        replacements[id(slice2)] = merged
        # Old slice1's params inits may be droppable if not used elsewhere.
        for old_inp_name in slice1.input[1:5]:
            if not old_inp_name:
                continue
            # Drop iff slice1 was the only consumer
            other_consumers = [c for c in consumers.get(old_inp_name, []) if id(c) != id(slice1)]
            if not other_consumers:
                inits_to_drop.add(old_inp_name)
        for old_inp_name in slice2.input[1:5]:
            if not old_inp_name:
                continue
            other_consumers = [c for c in consumers.get(old_inp_name, []) if id(c) != id(slice2)]
            if not other_consumers:
                inits_to_drop.add(old_inp_name)

    new_nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        if id(node) in drop_nodes:
            if id(node) in replacements:
                new_nodes.append(replacements[id(node)])
            continue
        new_nodes.append(node)

    # Compose initializer list: keep originals not in drop set, append new params.
    final_inits = [init for init in graph.initializer if init.name not in inits_to_drop]
    final_inits.extend(new_inits_to_add)

    # Drop value_info for intermediate Slice outputs (the slice1 outputs)
    intermediate_names = {p[0].output[0] for p in merge_pairs}
    new_value_info = [vi for vi in graph.value_info if vi.name not in intermediate_names]

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
        pattern_name="slice_chain_to_multiaxis",
        verdict="applied",
        new_model=new_model,
        notes=f"merged {len(merge_pairs)} 2-step single-axis Slice chain(s) into multi-axis Slice; dropped {len(inits_to_drop)} param init(s)",
        extras={"chains_merged": len(merge_pairs), "inits_dropped": sorted(inits_to_drop)},
    )


PATTERN = SpreadPattern(
    name="slice_chain_to_multiaxis",
    category="math-equivalent",
    description="merge Slice(axis=A) -> Slice(axis=B) into single Slice(axes=[A,B]); eliminates intermediate tensor",
    apply=_apply,
    safe_by_construction=True,
)
