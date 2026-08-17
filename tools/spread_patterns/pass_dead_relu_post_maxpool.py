"""Drop ``Relu(Sub(MaxPool(X, pad-symmetric, stride=1), X))`` — the
inner Sub is provably ``>= 0`` pointwise, so the Relu is identity.

Origin: 2026-05-19 task093 worker insight — 4 dead Relu nodes
(indices 11, 13, 15, 17) followed `Sub(MaxPool(X), X)`. Worker manually
removed them. The MaxPool's window at output[pos] is guaranteed to
include input[pos] under stride=1 + same-shape-preserving padding, so
``MaxPool(X)[pos] >= X[pos]`` for every pos, hence ``Sub >= 0``, hence
``Relu`` is a no-op.

Math (sufficient condition for safe drop):
  1. MaxPool has strides=[1, 1, ...] in all spatial dims (or strides
     absent — ONNX default).
  2. MaxPool has kernel_shape with each K >= 1.
  3. MaxPool has pads such that pad_start[d] + pad_end[d] == K[d] - 1
     for each spatial dim d. (This is the "same padding" / "preserves
     input shape" condition.) Implies pad_start[d] <= K[d] - 1.
  4. Sub's second input IS the same tensor that the MaxPool consumed
     as its input.

Under (1-3), the receptive field at output[pos] covers input[pos -
pad_start..pos - pad_start + K - 1], which always contains input[pos]
since 0 <= pad_start <= K - 1. Hence MaxPool(X)[pos] >= X[pos]
pointwise.

Strategy:
  * Find Relu nodes whose input is Sub(A, B).
  * A must be the output of a MaxPool node satisfying the safe condition.
  * B must equal MaxPool's input tensor name.
  * Rewire Relu's output -> Sub's output for all consumers; drop the
    Relu node.

Safety:
  * Asymmetric padding (pad_start[d] + pad_end[d] != K[d] - 1) is
    rejected — would change output shape and could miss input[pos].
  * Non-unit stride rejected — output positions may not include
    input[pos].
  * Wider receptive field (kernel > 1) is fine; the rule only requires
    the window to contain input[pos].

Out of scope:
  * Relu(Sub(MaxPool(X, pad-arbitrary), Y)) where Y != X. Generic non-
    negativity isn't provable.
  * Other dead-op-after-pool variants (Min, AveragePool, etc.). Could
    extend with per-op semantic checks.
"""
from __future__ import annotations

import onnx
from onnx import helper

from tools.spread_patterns._base import SpreadPattern, SpreadResult


def _attr_ints(node: onnx.NodeProto, name: str):
    for a in node.attribute:
        if a.name == name and a.type == onnx.AttributeProto.INTS:
            return list(a.ints)
    return None


def _maxpool_safe(node: onnx.NodeProto) -> bool:
    """Return True iff MaxPool's receptive field at every output
    position is guaranteed to include the corresponding input position."""
    if node.op_type != "MaxPool":
        return False
    kernel = _attr_ints(node, "kernel_shape")
    if kernel is None or any(k < 1 for k in kernel):
        return False
    strides = _attr_ints(node, "strides") or [1] * len(kernel)
    if any(s != 1 for s in strides):
        return False
    pads = _attr_ints(node, "pads")
    if pads is None or len(pads) != 2 * len(kernel):
        return False
    n = len(kernel)
    for d in range(n):
        pad_start = pads[d]
        pad_end = pads[d + n]
        if pad_start < 0 or pad_end < 0:
            return False
        if pad_start + pad_end != kernel[d] - 1:
            return False
        if pad_start > kernel[d] - 1:
            return False
    return True


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

    drops: list[tuple[onnx.NodeProto, str]] = []  # (relu_node, sub_out_to_alias)

    for relu in graph.node:
        if relu.op_type != "Relu" or not relu.input:
            continue
        sub_node = producer.get(relu.input[0])
        if sub_node is None or sub_node.op_type != "Sub" or len(sub_node.input) != 2:
            continue
        a_in, b_in = sub_node.input[0], sub_node.input[1]
        mp = producer.get(a_in)
        if mp is None or mp.op_type != "MaxPool":
            continue
        if not _maxpool_safe(mp):
            continue
        # Sub's second input must equal MaxPool's input
        if not mp.input or mp.input[0] != b_in:
            continue
        drops.append((relu, sub_node.output[0]))

    if not drops:
        return SpreadResult(
            pattern_name="dead_relu_post_maxpool",
            verdict="no-op",
            notes="no Relu(Sub(MaxPool, X)) dead-op patterns detected",
        )

    drop_ids = {id(r) for r, _ in drops}
    rename = {r.output[0]: alias for r, alias in drops}

    new_nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        if id(node) in drop_ids:
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

    # If a Relu output was a graph output, we need an Identity to preserve
    # the output name. For simplicity, skip those cases (none in NeuroGolf).
    graph_outs = {o.name for o in graph.output}
    if any(r.output[0] in graph_outs for r, _ in drops):
        return SpreadResult(
            pattern_name="dead_relu_post_maxpool",
            verdict="no-op",
            notes="dead Relu detected but feeds graph output (Identity insertion out of scope)",
        )

    dropped_names = {r.output[0] for r, _ in drops}
    new_value_info = [vi for vi in graph.value_info if vi.name not in dropped_names]

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
        pattern_name="dead_relu_post_maxpool",
        verdict="applied",
        new_model=new_model,
        notes=f"dropped {len(drops)} provably-dead Relu node(s) following MaxPool-Sub-self",
        extras={"relus_dropped": len(drops)},
    )


PATTERN = SpreadPattern(
    name="dead_relu_post_maxpool",
    category="math-equivalent",
    description="drop Relu(Sub(MaxPool(X, sym-pad, stride=1), X)) — MaxPool >= X pointwise so Sub >= 0 and Relu is identity",
    apply=_apply,
    safe_by_construction=True,
)
