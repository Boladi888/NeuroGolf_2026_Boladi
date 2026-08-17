"""Keep Concat in BOOL dtype when all inputs are bool-eligible (bool
source or 0/1-valued F32 init) and downstream is bool-tolerant.

Origin: 2026-05-20 worker insight — a [1,10,6,6] Concat output dropped
from F32 (1,440 bytes intermediate) to BOOL (360 bytes), part of a
chain that took a "near-floor" 2,678-cost task down to 1,058 — a 60%
reduction. The lesson: do NOT dismiss near-floor tasks; bool-keeping
through Concat unlocks 4x dtype-width savings.

Math equivalence: Concat just stacks tensors along an axis. If all
inputs are 0/1-valued and we represent them in BOOL instead of F32,
the logical mask values are identical, only the storage width changes.

Strategy:
  1. Find every Concat node.
  2. Classify each input:
       a. Cast(bool → F32) — bool-eligible (the underlying bool source
          can be wired directly).
       b. F32 initializer with all-0/1 values — bool-eligible
          (convert init to BOOL).
       c. Any other source — disqualifies the Concat for this pass.
  3. Concat output must be consumed by bool-tolerant ops only:
       - Where (as the condition input, or as a value where the OTHER
         value is also bool/BoolInit)
       - Greater(>0), Less(<1), Equal — bool-result ops
       - Pad — ORT-tolerant (see drop_cast_before_pad_bool)
       - Cast(BOOL → F32) — explicit conversion downstream (safe; we
         just move the Cast to AFTER Concat instead of before each input)
       - Or graph output that can be bool (we update its dtype).
  4. Rewrite:
       - Drop the Cast(bool → F32) nodes feeding Concat (if their
         output was consumed only by this Concat).
       - For each 0/1 F32 init feeding Concat, create a BOOL replica
         and rewire Concat input to use it.
       - Concat's output is now BOOL.
       - If a downstream Cast(BOOL → F32) existed, that Cast still
         applies (we may now be promoting a bool to F32 once, instead
         of N times). Leave it alone — its job is unchanged.
       - If the Concat output is a graph output, change graph output
         dtype to BOOL.

Safety:
  * Conservative: any non-classifiable Concat input bails out.
  * Cast inputs feeding Concat must be single-use (else we'd orphan
    the F32 view that another op still needs).
  * 0/1 init detection uses 1e-6 float tolerance.

Out of scope:
  * Mixed-dtype Concats where input dtypes already differ.
  * Concat inputs that come from a chain (e.g. Cast(bool, F32) →
    Reshape → Concat). Could extend later with depth-2 tracing.
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

# Ops that can consume a BOOL tensor directly (ORT-tolerant).
BOOL_TOLERANT_CONSUMERS = frozenset({
    "Where",     # condition input, or as a value where it gets cast
    "Greater", "Less", "Equal", "GreaterOrEqual", "LessOrEqual",
    "Not", "And", "Or", "Xor",
    "Pad",       # ORT Pad-13 accepts bool
    "Cast",      # explicit conversion; we just don't need the upstream casts
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


def _init_array(graph, name):
    for init in graph.initializer:
        if init.name == name:
            try:
                return init, numpy_helper.to_array(init)
            except Exception:  # noqa: BLE001
                return init, None
    return None, None


def _is_zero_one_float(arr):
    if arr is None or arr.size == 0:
        return False
    try:
        flat = arr.astype(np.float64, copy=False)
    except (TypeError, ValueError):
        return False
    zero_close = np.abs(flat - 0.0) < 1e-6
    one_close = np.abs(flat - 1.0) < 1e-6
    return bool(np.all(zero_close | one_close))


def _is_graph_output(name, graph):
    return any(o.name == name for o in graph.output)


def _apply(model: onnx.ModelProto, task_id: str) -> SpreadResult:
    graph = model.graph
    producer = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer[out] = node
    consumers = _build_consumer_map(graph)

    matches: list[dict] = []
    new_initializers = list(graph.initializer)
    new_init_names = set()

    for concat_node in graph.node:
        if concat_node.op_type != "Concat":
            continue
        if not concat_node.input:
            continue
        # Classify each input
        input_rewrites = []  # list of (kind, bool_src_name | bool_init_name)
        ok = True
        casts_to_drop = []  # Cast nodes safe to drop
        inits_to_create = []  # (orig_name, new_bool_name)
        for inp in concat_node.input:
            p = producer.get(inp)
            if p is not None and p.op_type == "Cast" and _cast_to_dtype(p) == TensorProto.FLOAT:
                src = p.input[0] if p.input else ""
                src_op = producer.get(src)
                if src_op is None or src_op.op_type not in BOOL_PRODUCING_OPS:
                    ok = False
                    break
                # Cast must be single-use
                if len(consumers.get(p.output[0], [])) != 1:
                    ok = False
                    break
                input_rewrites.append(("cast", src))
                casts_to_drop.append(p)
                continue
            # Initializer 0/1 case
            init_proto, init_arr = _init_array(graph, inp)
            if init_proto is not None and _is_zero_one_float(init_arr):
                new_bool_name = f"{inp}_as_bool"
                inits_to_create.append((inp, new_bool_name, init_arr))
                input_rewrites.append(("init", new_bool_name))
                continue
            # Disqualify
            ok = False
            break
        if not ok or not input_rewrites:
            continue

        # Concat output must be consumed only by bool-tolerant ops OR be graph output
        concat_out = concat_node.output[0]
        concat_consumers = consumers.get(concat_out, [])
        is_graph_out = _is_graph_output(concat_out, graph)
        if not is_graph_out and not all(c.op_type in BOOL_TOLERANT_CONSUMERS for c in concat_consumers):
            continue
        if not is_graph_out and not concat_consumers:
            continue  # unused; let DCE handle it

        # Create bool-init replicas (deduplicated)
        for orig_name, new_name, arr in inits_to_create:
            if new_name in new_init_names:
                continue
            bool_arr = (arr > 0.5).astype(bool)
            new_initializers.append(numpy_helper.from_array(bool_arr, name=new_name))
            new_init_names.add(new_name)

        matches.append({
            "concat": concat_node,
            "input_rewrites": input_rewrites,
            "casts_to_drop": casts_to_drop,
            "is_graph_out": is_graph_out,
        })

    if not matches:
        return SpreadResult(
            pattern_name="concat_bool_inputs",
            verdict="no-op",
            notes="no Concat with all-bool-eligible inputs detected",
        )

    drop_nodes = set()
    replacements: dict[int, list[onnx.NodeProto]] = {}

    for m in matches:
        concat = m["concat"]
        new_concat = onnx.NodeProto()
        new_concat.CopyFrom(concat)
        # Rewire inputs according to input_rewrites
        for i, (kind, new_inp_name) in enumerate(m["input_rewrites"]):
            new_concat.input[i] = new_inp_name
        replacements[id(concat)] = [new_concat]
        for cast in m["casts_to_drop"]:
            drop_nodes.add(id(cast))

    new_nodes = []
    for node in graph.node:
        if id(node) in drop_nodes:
            continue
        if id(node) in replacements:
            new_nodes.extend(replacements[id(node)])
        else:
            new_nodes.append(node)

    # Update graph output dtype for any Concat output that's a graph output
    new_outputs = []
    concat_outputs_now_bool = {m["concat"].output[0] for m in matches}
    for o in graph.output:
        new_o = onnx.ValueInfoProto()
        new_o.CopyFrom(o)
        if o.name in concat_outputs_now_bool:
            new_o.type.tensor_type.elem_type = TensorProto.BOOL
        new_outputs.append(new_o)

    # Drop value_info for the dropped Cast outputs and for Concat outputs
    # (dtype changed; let ORT reinfer)
    dropped_names = set()
    for m in matches:
        for cast in m["casts_to_drop"]:
            dropped_names.update(cast.output)
        dropped_names.add(m["concat"].output[0])
    new_value_info = [vi for vi in graph.value_info if vi.name not in dropped_names]

    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=graph.name,
        inputs=list(graph.input),
        outputs=new_outputs,
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

    n_casts_dropped = sum(len(m["casts_to_drop"]) for m in matches)
    n_inits_added = len(new_init_names)
    return SpreadResult(
        pattern_name="concat_bool_inputs",
        verdict="applied",
        new_model=new_model,
        notes=(
            f"converted {len(matches)} Concat to bool dtype; "
            f"dropped {n_casts_dropped} pre-Concat Cast(bool,F32); "
            f"added {n_inits_added} bool init replica(s); "
            f"{sum(1 for m in matches if m['is_graph_out'])} graph output(s) -> BOOL"
        ),
        extras={
            "concats_converted": len(matches),
            "casts_dropped": n_casts_dropped,
            "bool_inits_added": sorted(new_init_names),
        },
    )


PATTERN = SpreadPattern(
    name="concat_bool_inputs",
    category="math-equivalent",
    description="keep Concat in BOOL dtype when all inputs are bool-eligible (Cast(bool,F32) source OR 0/1-valued F32 init); 4x reduction on the Concat output intermediate",
    apply=_apply,
    safe_by_construction=True,
)
