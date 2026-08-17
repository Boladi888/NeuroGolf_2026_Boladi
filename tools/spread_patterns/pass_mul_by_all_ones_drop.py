"""Drop Mul nodes where one input is a tensor of all ones.

Origin: feedback_neurogolf_opt_safe_blindspots.md flags this as one of
the 4 algebraic-identity patterns that `opt_safe` misses on production
NeuroGolf graphs. Workers have seen tasks where a normalize/scale step
multiplies by a baked-in ones initializer that gets dead-code-eliminated
upstream but the Mul itself remains.

Math: x * 1.0 == x for any numeric x. Replacing Mul(x, ones) with x
preserves output exactly. The ones initializer also becomes garbage
collectable if no other op uses it (handled implicitly by score_v2 —
unreferenced initializers are dropped from memory accounting).

Strategy:
  * Find Mul nodes where ONE input is an initializer with all-ones
    values (within float tolerance: |v - 1.0| < 1e-6 for every element).
  * Rewrite consumers: every place that used Mul.output[0] now uses
    the other (non-ones) Mul input.
  * Drop the Mul node. The ones initializer survives only if another
    op still references it.

Safety:
  * Math is exact for any float dtype. The Mul input dtypes must match
    (standard ONNX requirement; we don't relax that).
  * If the Mul output is the GRAPH output, rename it to preserve the
    output binding — emit an Identity node to the original output name.
    (Just retargeting the output without an Identity would change the
    graph's output tensor name, breaking downstream consumers of the
    package.)

Out of scope (future):
  * Mul by a tensor where all elements are some other constant scalar
    (could be folded into a scalar Mul + algebraic simplifications, but
    requires more care around dtype and broadcasting).
  * Mul by Identity input (input itself is Identity'd from a producer);
    onnxsim usually catches that already.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import numpy_helper

from tools.spread_patterns._base import SpreadPattern, SpreadResult


def _is_all_ones_init(graph: onnx.GraphProto, name: str) -> bool:
    for init in graph.initializer:
        if init.name != name:
            continue
        try:
            arr = numpy_helper.to_array(init)
        except Exception:  # noqa: BLE001
            return False
        try:
            return arr.size > 0 and bool(np.all(np.abs(arr - 1.0) < 1e-6))
        except (TypeError, ValueError):
            return False
    return False


def _is_graph_output(name: str, graph: onnx.GraphProto) -> bool:
    return any(o.name == name for o in graph.output)


def _apply(model: onnx.ModelProto, task_id: str) -> SpreadResult:
    graph = model.graph
    drops: list[tuple[onnx.NodeProto, str]] = []  # (mul_node, pass-through-tensor-name)

    for node in graph.node:
        if node.op_type != "Mul" or len(node.input) != 2:
            continue
        a, b = node.input[0], node.input[1]
        if _is_all_ones_init(graph, a):
            drops.append((node, b))
        elif _is_all_ones_init(graph, b):
            drops.append((node, a))

    if not drops:
        return SpreadResult(
            pattern_name="mul_by_all_ones_drop",
            verdict="no-op",
            notes="no Mul(x, all-ones init) patterns detected",
        )

    drop_ids = {id(n) for n, _ in drops}
    rename: dict[str, str] = {}  # mul.output -> pass-through input name
    identity_replacements: list[onnx.NodeProto] = []  # for graph-output muls

    for mul_node, passthrough in drops:
        out = mul_node.output[0]
        if _is_graph_output(out, graph):
            # Preserve the graph output tensor name with an Identity.
            identity_replacements.append(onnx.helper.make_node(
                "Identity",
                inputs=[passthrough],
                outputs=[out],
                name=f"{mul_node.name or 'mul_ones'}_identity",
            ))
        else:
            rename[out] = passthrough

    # Build new node list. Replace dropped Muls with Identity if they fed
    # a graph output; otherwise drop entirely and rewire downstream.
    new_nodes: list[onnx.NodeProto] = []
    ident_iter = iter(identity_replacements)
    for node in graph.node:
        if id(node) in drop_ids:
            # If this Mul fed a graph output, emit the matching Identity here.
            out = node.output[0]
            if _is_graph_output(out, graph):
                new_nodes.append(next(ident_iter))
            continue
        # Rewire any inputs that pointed at a dropped Mul output.
        if any(inp in rename for inp in node.input):
            new_node = onnx.NodeProto()
            new_node.CopyFrom(node)
            for i, inp in enumerate(new_node.input):
                if inp in rename:
                    new_node.input[i] = rename[inp]
            new_nodes.append(new_node)
        else:
            new_nodes.append(node)

    new_graph = onnx.helper.make_graph(
        nodes=new_nodes,
        name=graph.name,
        inputs=list(graph.input),
        outputs=list(graph.output),
        initializer=list(graph.initializer),
        value_info=[vi for vi in graph.value_info if vi.name not in rename],
    )
    new_model = onnx.helper.make_model(
        new_graph,
        opset_imports=list(model.opset_import),
        ir_version=model.ir_version,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
    )
    return SpreadResult(
        pattern_name="mul_by_all_ones_drop",
        verdict="applied",
        new_model=new_model,
        notes=f"dropped {len(drops)} Mul(x, all-ones) nodes (rewired {len(rename)}, identity-preserved {len(identity_replacements)})",
        extras={"muls_dropped": len(drops)},
    )


PATTERN = SpreadPattern(
    name="mul_by_all_ones_drop",
    category="math-equivalent",
    description="drop Mul(x, all-ones-initializer) -> x (algebraic identity)",
    apply=_apply,
    safe_by_construction=True,
)
