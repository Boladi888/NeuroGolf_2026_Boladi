"""builder.py — finalize a GState (DAG) into a valid NeuroGolf ONNX model.

The graph output is the LAST-built tensor. We dead-node-eliminate (keep only nodes/initializers
reachable from it) so dangling search branches cost nothing, then rename that tensor to "output".
Enforces the validator contract: one input "input", one output "output", default opset, no banned ops.
"""
import onnx
from onnx import helper

from . import spec
from .ops import GState

OPSET = 18


def to_model(g):
    """g: GState -> onnx.ModelProto (or None on a banned op)."""
    out = g.tensors[-1]
    needed = {out.name}
    kept = []
    for n in reversed(g.nodes):                 # reverse topo: keep nodes reachable from the output
        if n.output[0] in needed:
            kept.append(n)
            needed.update(n.input)
    kept.reverse()

    if out.name == spec.IN_NAME or not kept:    # empty graph == identity: input -> output
        kept = [helper.make_node("Identity", [spec.IN_NAME], [spec.OUT_NAME])]
        out_dtype, out_shape = spec.FLOAT, spec.IN_SHAPE
    else:
        for n in kept:
            if n.output[0] == out.name:
                _rename(n, out.name, spec.OUT_NAME)
        out_dtype, out_shape = out.dtype, out.shape

    for n in kept:
        if n.op_type.upper() in spec.BANNED or "SEQUENCE" in n.op_type.upper():
            return None

    ref = {i for n in kept for i in n.input}
    inits = [t for t in g.inits if t.name in ref]
    graph = helper.make_graph(
        kept, "ngs",
        inputs=[helper.make_tensor_value_info(spec.IN_NAME, spec.FLOAT, list(spec.IN_SHAPE))],
        outputs=[helper.make_tensor_value_info(spec.OUT_NAME, out_dtype, list(out_shape))],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 9
    return model


def straight(ops, pairs=None):
    """Apply ops linearly (each op's first candidate) -> GState, or None. For simple/bench builds."""
    g = GState.start()
    for op in ops:
        cands = op.candidates(g)
        if not cands:
            return None
        g = op.build(g, cands[0], pairs)
        if g is None:
            return None
    return g


def _rename(node, old, new):
    out = [new if o == old else o for o in node.output]
    del node.output[:]; node.output.extend(out)


def program_label(ops):
    return " -> ".join(op.label for op in ops) if ops else "(identity)"
