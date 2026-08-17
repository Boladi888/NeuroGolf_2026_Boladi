"""Drop ``Cast(bool → F32) → Pad`` when the Pad output flows only to a
bool-tolerant consumer. ORT Pad-13 accepts BOOL input despite not
being in the official ONNX spec, so the cast is removable.

Origin: 2026-05-20 worker insight — saved -4,000 each on task250 and
task342 by removing the Cast and letting Pad operate directly on bool,
then keeping the BOOL graph output (downstream tooling thresholds
``(out > 0.0)`` which is identity on a 0/1-valued tensor regardless of
F32 vs BOOL).

Math equivalence: padding values for a 0/1 mask are 0 (BOOL false /
F32 0.0) or 1 (BOOL true / F32 1.0). Pad's constant_value attribute is
F32 in spec but ORT's UINT8/BOOL implementations accept and use the
literal 0/1 value. The padded output is the same logical mask, just
in BOOL dtype.

Strategy:
  1. Find Cast nodes that convert from a BOOL-producing op output to
     FLOAT (F32).
  2. The Cast output must be consumed by exactly ONE Pad node.
  3. The Pad output must EITHER be the graph output OR be consumed by
     a single Greater(>0) node (the safe threshold case). Any other
     consumer is rejected for safety.
  4. Rewrite:
       - Drop the Cast node.
       - Rewire Pad's input to the bool source directly.
       - If the Pad output was a graph output, change the graph
         output's dtype to BOOL.
       - If the Pad output was consumed by Greater(>0), drop that
         Greater (its output replaces the Pad output downstream, in
         BOOL dtype) — Greater(bool_tensor, 0) is identity for bool.

Safety:
  * ORT-specific. Bool Pad isn't portable to other ONNX runtimes.
    NeuroGolf submissions run on Kaggle's ORT so this is fine; safe
    note documented in `feedback_bool_first_compute`.
  * Conservative: any unexpected consumer chain bails out the pattern.
  * Pad constant_value (in the attribute form opset 10-) defaults to
    0.0 — semantically False in bool. If the model specifies a
    non-zero constant_value, we skip (rare in NeuroGolf graphs).

Out of scope:
  * Pad in input form (opset 11+) with non-zero constant_value input.
    Could extend with careful dtype handling.
  * Cast chains with > 1 hop between Cast and Pad. Could extend.
"""
from __future__ import annotations

from typing import Any

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


def _is_graph_output(name, graph):
    return any(o.name == name for o in graph.output)


def _pad_constant_value_safe(pad_node, graph):
    """Return True iff Pad's constant_value is 0 (the bool-False
    equivalent). Attribute form (opset 10-) reads `value` attribute;
    input form (opset 11+) reads input[2] if present."""
    # Attribute form
    val_attr = next((a for a in pad_node.attribute if a.name == "value"), None)
    if val_attr is not None:
        return abs(val_attr.f) < 1e-6
    # Input form: input[2] is constant_value initializer (optional)
    if len(pad_node.input) >= 3 and pad_node.input[2]:
        for init in graph.initializer:
            if init.name == pad_node.input[2]:
                try:
                    arr = numpy_helper.to_array(init)
                except Exception:  # noqa: BLE001
                    return False
                if arr.size == 0:
                    return True  # empty == default 0
                return abs(float(arr.flatten()[0])) < 1e-6
        return False
    return True  # no constant_value at all => Pad default 0


def _is_scalar_zero(graph, name):
    for init in graph.initializer:
        if init.name != name:
            continue
        try:
            arr = numpy_helper.to_array(init)
        except Exception:  # noqa: BLE001
            return False
        if arr.size == 0:
            return False
        return all(abs(float(v)) < 1e-6 for v in arr.flatten())
    return False


def _apply(model: onnx.ModelProto, task_id: str) -> SpreadResult:
    graph = model.graph
    producer: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer[out] = node
    consumers = _build_consumer_map(graph)

    matches: list[dict] = []

    for cast_node in graph.node:
        if cast_node.op_type != "Cast" or not cast_node.input:
            continue
        if _cast_to_dtype(cast_node) != TensorProto.FLOAT:
            continue
        bool_src = cast_node.input[0]
        src_node = producer.get(bool_src)
        if src_node is None or src_node.op_type not in BOOL_PRODUCING_OPS:
            continue
        # Cast output must be consumed only by Pad
        cast_out_consumers = consumers.get(cast_node.output[0], [])
        if len(cast_out_consumers) != 1 or cast_out_consumers[0].op_type != "Pad":
            continue
        pad_node = cast_out_consumers[0]
        # Pad constant_value must be 0 (or default)
        if not _pad_constant_value_safe(pad_node, graph):
            continue
        # Pad input[0] must equal Cast output (i.e., Cast result is the data being padded)
        if pad_node.input[0] != cast_node.output[0]:
            continue
        pad_out = pad_node.output[0]
        # Pad output: either graph output, or sole consumer is Greater(>0)
        pad_out_consumers = consumers.get(pad_out, [])
        gtg = None  # Greater node downstream that we can also drop
        is_graph_out = _is_graph_output(pad_out, graph)
        if not is_graph_out:
            if len(pad_out_consumers) != 1:
                continue
            cand = pad_out_consumers[0]
            if cand.op_type != "Greater" or len(cand.input) != 2:
                continue
            # Greater's second input must be 0
            if not _is_scalar_zero(graph, cand.input[1]):
                continue
            gtg = cand
        matches.append({
            "cast": cast_node,
            "pad": pad_node,
            "bool_src": bool_src,
            "is_graph_out": is_graph_out,
            "greater_to_drop": gtg,
        })

    if not matches:
        return SpreadResult(
            pattern_name="drop_cast_before_pad_bool",
            verdict="no-op",
            notes="no Cast(bool->F32)->Pad chain with bool-tolerant downstream detected",
        )

    drop_nodes = set()
    rename: dict[str, str] = {}  # old_output -> new_alias

    for m in matches:
        cast = m["cast"]
        pad = m["pad"]
        drop_nodes.add(id(cast))
        # Rewire Pad to consume bool_src directly. Need a new Pad node
        # because we have to update input[0]. We mutate in place via copy.
        new_pad = onnx.NodeProto()
        new_pad.CopyFrom(pad)
        new_pad.input[0] = m["bool_src"]
        # If we're also dropping a Greater downstream, rewire Pad's output
        # to take Greater's output name (so downstream consumers see the
        # name they expect, now BOOL-dtyped).
        if m["greater_to_drop"] is not None:
            drop_nodes.add(id(m["greater_to_drop"]))
            new_pad.output[0] = m["greater_to_drop"].output[0]
        # Mark Pad for replacement
        m["new_pad"] = new_pad

    new_nodes: list[onnx.NodeProto] = []
    pad_replacements = {id(m["pad"]): m["new_pad"] for m in matches}
    for node in graph.node:
        if id(node) in drop_nodes:
            continue
        if id(node) in pad_replacements:
            new_nodes.append(pad_replacements[id(node)])
        else:
            new_nodes.append(node)

    # Update graph outputs: any Pad output that's a graph output -> BOOL dtype
    new_outputs = []
    for o in graph.output:
        new_o = onnx.ValueInfoProto()
        new_o.CopyFrom(o)
        for m in matches:
            if m["is_graph_out"] and m["pad"].output[0] == o.name:
                new_o.type.tensor_type.elem_type = TensorProto.BOOL
        new_outputs.append(new_o)

    # Drop value_info for the now-removed Cast outputs and (optionally)
    # the dropped Greater outputs (since the Pad now writes to that name).
    dropped_names = {m["cast"].output[0] for m in matches}
    if any(m["greater_to_drop"] for m in matches):
        # Greater outputs now produced by Pad — but the dtype changed;
        # the safest is to drop those value_infos and let ORT reinfer.
        for m in matches:
            g = m["greater_to_drop"]
            if g is not None:
                dropped_names.update(g.output)
        # Also drop value_info for the old Pad output if it was named
        for m in matches:
            if m["greater_to_drop"] is not None:
                dropped_names.add(m["pad"].output[0])
    new_value_info = [vi for vi in graph.value_info if vi.name not in dropped_names]

    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=graph.name,
        inputs=list(graph.input),
        outputs=new_outputs,
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
        pattern_name="drop_cast_before_pad_bool",
        verdict="applied",
        new_model=new_model,
        notes=(
            f"dropped {len(matches)} Cast(bool,F32)->Pad chain(s); "
            f"{sum(1 for m in matches if m['is_graph_out'])} bool graph output(s); "
            f"{sum(1 for m in matches if m['greater_to_drop'] is not None)} downstream Greater(>0) also dropped"
        ),
        extras={"chains_dropped": len(matches)},
    )


PATTERN = SpreadPattern(
    name="drop_cast_before_pad_bool",
    category="math-equivalent",
    description="drop Cast(bool->F32) before Pad when downstream is bool-tolerant (ORT Pad-13 accepts bool input)",
    apply=_apply,
    safe_by_construction=True,
)
