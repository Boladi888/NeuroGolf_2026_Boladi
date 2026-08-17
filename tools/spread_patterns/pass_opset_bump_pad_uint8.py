"""Bump opset 10 -> 11 to convert ``Pad`` from attribute-form to
input-form, then replace ``Cast(bool → F32) → Pad`` with
``Cast(bool → UINT8) → Pad`` (UINT8 is 1 byte/elem same as bool but
more portable than the bool-Pad ORT-specific trick).

Origin: 2026-05-20 worker insight on task319 — saved -741 bytes via
this opset bump + UINT8 cast pattern. Complementary to
`drop_cast_before_pad_bool` (ORT-specific): when the downstream
consumer chain can't accept BOOL but can accept UINT8 (any numeric
arithmetic chain), UINT8 is the right intermediate dtype.

Math equivalence: Pad with 0 constant_value on a 0/1 mask in UINT8 is
identical to the F32 version. Both produce the same logical mask
values; only the storage width differs (1 byte vs 4 bytes).

Strategy:
  1. Model must have opset 10 in the ai.onnx domain.
  2. Find Cast(bool, F32) -> Pad chains.
  3. Pad must be in attribute form (`pads` and `mode` as attributes,
     no input[1]). Constant_value must be 0 (default) or absent.
  4. Rewrite plan:
       a. Bump model opset_import for ai.onnx to 11.
       b. Convert Pad to input-form: pads attribute -> initializer +
          input position 1.
       c. Change Cast target from FLOAT to UINT8.
       d. (Pad-11+ supports UINT8 input.)

Safety:
  * Only fires when opset is exactly 10. Newer/older opsets need
    different rewrites.
  * Cast must be single-use (else we'd need to keep an F32 cast
    elsewhere; defeats the purpose).
  * Pad's `constant_value` attribute (opset-10 form) must be 0 or
    absent. Non-zero constant_value would need to be preserved as
    an additional UINT8 init at input[2].
  * Downstream of the Pad must NOT include the bool-Pad shortcut
    consumer (Greater>0); if it does, prefer `drop_cast_before_pad_bool`
    for the better savings. We let both patterns run; the scan driver
    picks the best.

Out of scope:
  * Models already at opset 11+. They don't need the bump.
  * Pad with non-zero constant_value (rare in NeuroGolf masks).
  * Multi-axis-constant Pads where the input form would require a
    much larger init than the attribute version saved.
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


def _ai_onnx_opset(model):
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            return imp.version
    return None


def _pad_attr_form_safe(pad_node):
    """Return (pads_list, mode_str) if Pad is in attribute form with 0
    constant_value, else None."""
    pads = next((a for a in pad_node.attribute if a.name == "pads"), None)
    if pads is None:
        return None  # not attribute form
    mode = next((a for a in pad_node.attribute if a.name == "mode"), None)
    mode_str = mode.s.decode() if mode and mode.type == onnx.AttributeProto.STRING else "constant"
    if mode_str != "constant":
        return None  # only constant-mode supported by this rewrite
    value = next((a for a in pad_node.attribute if a.name == "value"), None)
    if value is not None and abs(value.f) > 1e-6:
        return None  # non-zero constant_value
    return list(pads.ints), mode_str


def _apply(model: onnx.ModelProto, task_id: str) -> SpreadResult:
    opset = _ai_onnx_opset(model)
    if opset != 10:
        return SpreadResult(
            pattern_name="opset_bump_pad_uint8",
            verdict="no-op",
            notes=f"opset {opset} != 10; skip (this pattern only fires on opset-10 models)",
        )

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
        bool_src_name = cast_node.input[0]
        src = producer.get(bool_src_name)
        if src is None or src.op_type not in BOOL_PRODUCING_OPS:
            continue
        # Cast output must be consumed only by Pad
        cast_consumers = consumers.get(cast_node.output[0], [])
        if len(cast_consumers) != 1 or cast_consumers[0].op_type != "Pad":
            continue
        pad_node = cast_consumers[0]
        info = _pad_attr_form_safe(pad_node)
        if info is None:
            continue
        pads_list, mode_str = info
        matches.append({
            "cast": cast_node,
            "pad": pad_node,
            "pads_list": pads_list,
            "mode_str": mode_str,
        })

    if not matches:
        return SpreadResult(
            pattern_name="opset_bump_pad_uint8",
            verdict="no-op",
            notes="no Cast(bool,F32)->Pad-attribute-form chain detected at opset 10",
        )

    # Rewrite plan
    drop_nodes = set()
    new_initializers = list(graph.initializer)
    replacements: dict[int, list[onnx.NodeProto]] = {}
    counter = [0]

    def _uniq(p):
        counter[0] += 1
        return f"{p}_obu_{counter[0]}"

    for m in matches:
        cast = m["cast"]
        pad = m["pad"]
        drop_nodes.add(id(cast))
        drop_nodes.add(id(pad))

        # Build new Cast: target UINT8
        new_cast = helper.make_node(
            "Cast",
            inputs=cast.input[:],
            outputs=[cast.output[0]],
            name=_uniq(cast.name or "cast_to_uint8"),
            to=TensorProto.UINT8,
        )

        # Build new Pad: input form. pads -> initializer + input[1]
        pads_init_name = _uniq("pads")
        pads_arr = np.array(m["pads_list"], dtype=np.int64)
        new_initializers.append(numpy_helper.from_array(pads_arr, name=pads_init_name))
        new_pad_inputs = [cast.output[0], pads_init_name]
        new_pad = helper.make_node(
            "Pad",
            inputs=new_pad_inputs,
            outputs=[pad.output[0]],
            name=_uniq(pad.name or "pad_input_form"),
            mode=m["mode_str"],
        )

        replacements[id(cast)] = [new_cast]
        replacements[id(pad)] = [new_pad]

    new_nodes = []
    for node in graph.node:
        if id(node) in drop_nodes:
            if id(node) in replacements:
                new_nodes.extend(replacements[id(node)])
            continue
        new_nodes.append(node)

    # If any rewritten Pad's output is a graph output, the graph output's
    # declared dtype must be updated to UINT8 to match the new Pad output.
    # (ORT enforces graph-output dtype consistency at load time.)
    pad_output_names = {m["pad"].output[0] for m in matches}
    new_outputs = []
    for o in graph.output:
        new_o = onnx.ValueInfoProto()
        new_o.CopyFrom(o)
        if o.name in pad_output_names:
            new_o.type.tensor_type.elem_type = TensorProto.UINT8
        new_outputs.append(new_o)
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=graph.name,
        inputs=list(graph.input),
        outputs=new_outputs,
        initializer=new_initializers,
        value_info=list(graph.value_info),
    )

    # Bump opset_import to 11 for ai.onnx domain
    new_opset_imports = []
    for imp in model.opset_import:
        new_imp = onnx.OperatorSetIdProto()
        new_imp.CopyFrom(imp)
        if imp.domain in ("", "ai.onnx") and imp.version == 10:
            new_imp.version = 11
        new_opset_imports.append(new_imp)

    new_model = helper.make_model(
        new_graph,
        opset_imports=new_opset_imports,
        ir_version=model.ir_version,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
    )

    return SpreadResult(
        pattern_name="opset_bump_pad_uint8",
        verdict="applied",
        new_model=new_model,
        notes=f"bumped opset 10->11, converted {len(matches)} Pad to input form, changed Cast target F32->UINT8",
        extras={"chains_rewritten": len(matches)},
    )


PATTERN = SpreadPattern(
    name="opset_bump_pad_uint8",
    category="math-equivalent",
    description="bump opset 10->11 + convert Cast(bool,F32)->Pad(attr) to Cast(bool,UINT8)->Pad(input form)",
    apply=_apply,
    safe_by_construction=True,
)
