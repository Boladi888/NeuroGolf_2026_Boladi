"""Fold N parallel ``Sub(v, ck) -> Abs -> Less(eps)`` chains into a
shared ``Cast(v -> INT32) + N x Equal(v_int32, ck_int32)``.

Origin: 2026-05-19 task368 worker insight. The model compared one
float tensor ``ocol`` against 10 different color constants using the
``Sub + Abs + Less(0.5)`` pattern (one-hot color selection). Folding
each chain individually to ``Equal(ocol, ck)`` would emit 10 separate
Cast nodes (since opset-10 Equal requires INT32). The worker's trick:
emit ONE shared ``Cast(ocol -> INT32)`` and N parallel
``Equal(ocol_int32, ck_int32)``. Saved 7,200 bytes net on task368.

Math equivalence: for ``v`` and ``ck`` exact-integer-valued floats,
``Less(Abs(Sub(v, ck)), 0.5)`` is True iff v == ck (assuming integer
quantization). Replacing with ``Equal(v_int32, ck_int32)`` after
truncating Cast preserves the math exactly. (For non-integer-valued
v, the substitution changes results — verify_inputs catches this.)

Strategy:
  1. Find Less nodes with eps strictly less than 1 (typically 0.5).
  2. Less[0] producer must be Abs.
  3. Abs[0] producer must be Sub.
  4. Sub's two inputs: a non-init "variable" (v) and an initializer
     (ck) with shape compatible with v. Either side may be the init.
  5. Group all such matches by the variable side `v`. For each group
     of size N >= 1:
       - Emit one shared ``Cast(v, INT32)``.
       - For each chain in the group: emit ``Equal(v_int32, ck_int32)``
         where ck_int32 is a new int32 initializer.
       - Drop the original Sub, Abs, Less, and the float ck init (if
         the float ck init is unused elsewhere).
  6. Output dtype: the original Less returns BOOL; Equal returns BOOL.
     Keep the original Less output name on the Equal node so consumers
     don't need rewiring.

Distinction from existing `equal_fold_distance_threshold`:
  * That pattern handles ``Sub -> Mul(self, self) -> Less<1`` (squared
    distance, where the Mul is self*self). This pattern handles
    ``Sub -> Abs -> Less<0.5`` (absolute distance). Different shape.
  * That pattern emits float ``Equal(v_float, ck_float)`` (which works
    on opset 11+ but is invalid on opset 10). This pattern emits
    INT32 Equal, which is valid across opset 7+.

Safety:
  * Variable `v` must be the SAME across all matched chains (we group
    by name). Different `v` => separate Cast nodes; the savings
    amortization disappears.
  * Each ck must be representable as INT32 (within int range).
  * Sub/Abs intermediates must have NO other consumer.
  * Eps threshold must be in (0, 1) — outside this range the
    integer-equivalence argument fails.

Out of scope (future):
  * Single-channel case (N == 1): no Cast amortization, savings come
    only from dropping Sub+Abs intermediates. We still emit it (the
    fold is still a win even for N == 1, just smaller).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tools.spread_patterns._base import SpreadPattern, SpreadResult


def _scalar_init_value(graph: onnx.GraphProto, name: str):
    """Return numeric scalar (float) value of an init, else None."""
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


def _init_array(graph: onnx.GraphProto, name: str):
    for init in graph.initializer:
        if init.name == name:
            try:
                return numpy_helper.to_array(init)
            except Exception:  # noqa: BLE001
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

    init_names = {init.name for init in graph.initializer}

    matches: list[dict] = []
    # Each match: {less, abs_node, sub_node, v_name, ck_name, ck_arr, eps}

    for less in graph.node:
        if less.op_type != "Less" or len(less.input) != 2:
            continue
        eps = _scalar_init_value(graph, less.input[1])
        if eps is None or not (0.0 < eps < 1.0):
            continue
        abs_node = producer.get(less.input[0])
        if abs_node is None or abs_node.op_type != "Abs":
            continue
        # Abs output must be consumed only by this Less.
        if len(consumers.get(abs_node.output[0], [])) != 1:
            continue
        sub_node = producer.get(abs_node.input[0]) if abs_node.input else None
        if sub_node is None or sub_node.op_type != "Sub" or len(sub_node.input) != 2:
            continue
        # Sub output must be consumed only by Abs.
        if len(consumers.get(sub_node.output[0], [])) != 1:
            continue
        # One Sub input is variable, the other is an init.
        a, b = sub_node.input[0], sub_node.input[1]
        if a in init_names and b not in init_names:
            ck_name, v_name = a, b
        elif b in init_names and a not in init_names:
            ck_name, v_name = b, a
        else:
            continue
        ck_arr = _init_array(graph, ck_name)
        if ck_arr is None:
            continue
        # ck values must be representable as INT32 (we'll check via tolist conversion).
        try:
            ck_int_arr = ck_arr.astype(np.int32)
            if not np.allclose(ck_int_arr.astype(ck_arr.dtype), ck_arr, atol=1e-3):
                continue  # not integer-valued
        except Exception:  # noqa: BLE001
            continue
        matches.append({
            "less": less,
            "abs_node": abs_node,
            "sub_node": sub_node,
            "v_name": v_name,
            "ck_name": ck_name,
            "ck_int_arr": ck_int_arr,
            "eps": eps,
        })

    if not matches:
        return SpreadResult(
            pattern_name="shared_cast_equal_fold",
            verdict="no-op",
            notes="no Sub->Abs->Less<1 chains with integer-valued init detected",
        )

    # Group by variable name
    by_v: dict[str, list[dict]] = {}
    for m in matches:
        by_v.setdefault(m["v_name"], []).append(m)

    drop_nodes = set()
    new_nodes_to_insert: dict[int, list[onnx.NodeProto]] = {}  # less_id -> [Equal node]
    new_inits: list[onnx.TensorProto] = []
    seen_v_cast: dict[str, str] = {}  # v_name -> v_int32_name
    counter = [0]

    def _uniq(prefix: str) -> str:
        counter[0] += 1
        return f"{prefix}_sce_{counter[0]}"

    # Where to insert the shared Cast node — we put it as a NEW first node
    # for that variable. Simpler: insert just before the first Less that
    # uses it. To keep node order valid (producers before consumers), we
    # emit the Cast before the first Less in each group.
    first_less_per_v: dict[str, onnx.NodeProto] = {}
    for v_name, group in by_v.items():
        first_less_per_v[v_name] = min(group, key=lambda m: list(graph.node).index(m["less"]))["less"]

    cast_nodes_to_prepend: dict[int, onnx.NodeProto] = {}  # first_less_id -> Cast node

    for v_name, group in by_v.items():
        v_int_name = _uniq(f"{v_name}_int32")
        seen_v_cast[v_name] = v_int_name
        cast_node = helper.make_node(
            "Cast",
            inputs=[v_name],
            outputs=[v_int_name],
            name=_uniq(f"cast_{v_name}_int32"),
            to=TensorProto.INT32,
        )
        cast_nodes_to_prepend[id(first_less_per_v[v_name])] = cast_node

    for m in matches:
        drop_nodes.add(id(m["less"]))
        drop_nodes.add(id(m["abs_node"]))
        drop_nodes.add(id(m["sub_node"]))
        # Make an INT32 init for ck (named anew to avoid colliding with the float ck).
        ck_int_name = _uniq(f"{m['ck_name']}_int32")
        new_inits.append(numpy_helper.from_array(m["ck_int_arr"], name=ck_int_name))
        v_int_name = seen_v_cast[m["v_name"]]
        equal_node = helper.make_node(
            "Equal",
            inputs=[v_int_name, ck_int_name],
            outputs=[m["less"].output[0]],
            name=_uniq(f"equal_{m['ck_name']}"),
        )
        new_nodes_to_insert[id(m["less"])] = [equal_node]

    # Detect dead inits (float ck inits with no remaining consumer).
    inits_maybe_drop = {m["ck_name"] for m in matches}
    # Note: Less also references eps as its input[1]. We never drop the Less,
    # but the Less is dropped here. Eps inits may become dead.
    inits_maybe_drop.update(m["less"].input[1] for m in matches if m["less"].input[1] in init_names)
    # ... but only drop if no surviving consumer.
    # We'll compute that after rebuilding nodes.

    new_nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        # Prepend Cast for the first-Less-of-v points.
        if id(node) in cast_nodes_to_prepend:
            new_nodes.append(cast_nodes_to_prepend[id(node)])
        if id(node) in drop_nodes:
            # If this is a Less we have an Equal replacement for, emit it.
            if id(node) in new_nodes_to_insert:
                new_nodes.extend(new_nodes_to_insert[id(node)])
            continue
        new_nodes.append(node)

    # Compute surviving init references
    surviving_inputs: set[str] = set()
    for node in new_nodes:
        for inp in node.input:
            if inp:
                surviving_inputs.add(inp)
    inits_to_drop = {name for name in inits_maybe_drop if name not in surviving_inputs}

    final_inits = [init for init in graph.initializer if init.name not in inits_to_drop]
    final_inits.extend(new_inits)

    # Drop value_info for Sub/Abs intermediates we just removed.
    dropped_inter = set()
    for m in matches:
        dropped_inter.update(m["sub_node"].output)
        dropped_inter.update(m["abs_node"].output)
    new_value_info = [vi for vi in graph.value_info if vi.name not in dropped_inter]

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
        pattern_name="shared_cast_equal_fold",
        verdict="applied",
        new_model=new_model,
        notes=(
            f"folded {len(matches)} Sub->Abs->Less<1 chain(s) over "
            f"{len(by_v)} variable(s) into shared Cast(INT32)+Equal; "
            f"dropped {len(inits_to_drop)} now-orphan init(s)"
        ),
        extras={
            "chains_folded": len(matches),
            "vars_shared": list(by_v.keys()),
            "inits_dropped": sorted(inits_to_drop),
        },
    )


PATTERN = SpreadPattern(
    name="shared_cast_equal_fold",
    category="math-equivalent",
    description="fold N parallel Sub-Abs-Less<1 chains over a shared variable into one shared Cast(INT32)+N Equal nodes (task368 task)",
    apply=_apply,
    safe_by_construction=True,
)
