"""Collapse a chain of N consecutive Add ops into a single Sum(N+1 inputs).

Origin: task327 v1 (worker-5, 2026-05-18) — replaced a 5-Add chain with a
single Sum taking 6 inputs, saving -5760 cost / +0.350 pts. The math is
identity (Add is associative+commutative), and ONNX Sum accepts ≥1 inputs
with the same broadcast semantics as Add. Eliminating intermediate Add
nodes drops N-1 intermediate tensors from the score_network_v2 memory
total.

Strategy:
  * Walk the graph to find maximal chains of Add nodes where each Add's
    first input is the previous Add's output (i.e., a strict left-fold:
    Add(Add(Add(a,b),c),d) → [a,b,c,d] → Sum(a,b,c,d)).
  * Each second input must be a graph value (init or another node's
    output) used ONLY by this Add (no other consumer). Otherwise we'd
    create an extra consumer of an intermediate and break some other
    op's expected shape/order.
  * The chain head's first input can be a leaf (input/init/independent
    producer). Subsequent first inputs are the prior Add's output.
  * Only collapse chains of length ≥2 (≥3 inputs in the Sum). A single
    Add → Sum(2) is a no-op refactor that doesn't save bytes.
  * Preserve dtype, shape, broadcast semantics. Sum picks the
    broadcasted output dtype the same way Add does.

Cost mechanism:
  * Each eliminated Add's output value_info is removed from the graph,
    dropping its placeholder allocation from score_network_v2's memory
    sum. With shape [1, C, H, W] F32 outputs, this is 4·C·H·W bytes per
    eliminated Add.

Out of scope (future passes if needed):
  * Tree-shaped Add reductions (chains that branch). Sum could still
    apply, but the rewrite would change first-parent identity and may
    interact with score_network_v2's memory accounting differently.
  * Mixed Add+Sub chains. Sub is not commutative; collapsing would
    require careful sign tracking.
"""
from __future__ import annotations

from typing import Any

import onnx

from tools.spread_patterns._base import SpreadPattern, SpreadResult


def _build_consumer_map(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    """Map tensor name -> list of nodes that consume it. Excludes graph
    outputs (those aren't node consumers but we treat them as a 'use' too
    when checking exclusivity)."""
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
    # Index nodes by their producing output name (assumption: SSA — each
    # output name produced by exactly one node, which is ONNX's contract).
    producer: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer[out] = node

    consumers = _build_consumer_map(graph)

    # Walk each Add and try to extend it into a maximal chain. We grow the
    # chain by replacing the head with the producer of the chain-head's
    # first input (when that producer is also an Add). We collect inputs
    # left-to-right so the resulting Sum has the same evaluation order.
    visited_nodes: set[int] = set()  # use id() since NodeProto isn't hashable
    chains: list[tuple[list[onnx.NodeProto], list[str]]] = []
    for node in graph.node:
        if node.op_type != "Add" or id(node) in visited_nodes:
            continue
        # Walk the chain by extending TAIL forward: start at `node` (which
        # may itself be mid-chain), so first walk UP to find the head.
        head = node
        while True:
            first_inp = head.input[0] if head.input else ""
            up = producer.get(first_inp)
            if up is None or up.op_type != "Add" or id(up) in visited_nodes:
                break
            # The Add up must have exactly one consumer (head). Otherwise
            # collapsing would drop a tensor that another node depends on.
            if len(consumers.get(up.output[0], [])) != 1 or _is_graph_output(up.output[0], graph):
                break
            head = up

        # Now walk DOWN from head, collecting the chain.
        chain: list[onnx.NodeProto] = [head]
        sum_inputs: list[str] = [head.input[0], head.input[1]] if len(head.input) >= 2 else []
        cursor = head
        while True:
            # Find a child Add that consumes cursor.output[0] as its FIRST input.
            child_candidates = consumers.get(cursor.output[0], [])
            child = None
            for c in child_candidates:
                if (
                    c.op_type == "Add"
                    and len(c.input) >= 2
                    and c.input[0] == cursor.output[0]
                    and id(c) not in visited_nodes
                ):
                    child = c
                    break
            if child is None:
                break
            # Cursor.output[0] must be consumed ONLY by `child` (and not
            # used as a graph output) for the collapse to be safe.
            if len(child_candidates) != 1 or _is_graph_output(cursor.output[0], graph):
                break
            chain.append(child)
            sum_inputs.append(child.input[1])
            cursor = child

        # Mark whole chain as visited so we don't restart from mid-chain.
        for n in chain:
            visited_nodes.add(id(n))

        if len(chain) >= 2:
            chains.append((chain, sum_inputs))

    if not chains:
        return SpreadResult(
            pattern_name="add_chain_to_sum",
            verdict="no-op",
            notes="no Add chains of length >= 2 detected",
        )

    # Rewrite: replace each chain with one Sum. The Sum takes the
    # concatenated inputs in original left-to-right order. Sum's output
    # name reuses the chain-tail's output name so downstream consumers
    # don't need rewiring.
    new_nodes: list[onnx.NodeProto] = []
    chain_node_ids = {id(n): (chain_idx, pos) for chain_idx, (chain, _) in enumerate(chains) for pos, n in enumerate(chain)}
    chain_tail_outputs: dict[int, str] = {chain_idx: chain[-1].output[0] for chain_idx, (chain, _) in enumerate(chains)}
    chain_sum_inputs: dict[int, list[str]] = {chain_idx: sum_inputs for chain_idx, (_, sum_inputs) in enumerate(chains)}

    # Counter so multiple Sum nodes get unique names even when the
    # underlying Add chains have empty `name` fields (NeuroGolf graphs
    # frequently leave node names blank). Without this, every Sum would
    # default to "add_chain_to_sum" and ORT would reject the model with
    # "two nodes with same node name".
    _counter = [0]

    def _uniq(prefix: str) -> str:
        _counter[0] += 1
        return f"{prefix}_{_counter[0]}"

    emitted_for_chain: set[int] = set()
    for node in graph.node:
        ci = chain_node_ids.get(id(node))
        if ci is None:
            new_nodes.append(node)
            continue
        chain_idx, pos = ci
        if chain_idx in emitted_for_chain:
            continue  # other chain members are absorbed
        # Emit one Sum at the head's position
        if pos != 0:
            # head wasn't the first in graph order; defer until we hit head
            continue
        head_name = chains[chain_idx][0][0].name or "add_chain"
        sum_node = onnx.helper.make_node(
            "Sum",
            inputs=chain_sum_inputs[chain_idx],
            outputs=[chain_tail_outputs[chain_idx]],
            name=_uniq(f"{head_name}_to_sum"),
        )
        new_nodes.append(sum_node)
        emitted_for_chain.add(chain_idx)

    # Some chains may have heads not in graph order before tails; emit any
    # remaining Sums at the end (rare, only if graph is out-of-order).
    for chain_idx in range(len(chains)):
        if chain_idx in emitted_for_chain:
            continue
        head_name = chains[chain_idx][0][0].name or "add_chain"
        sum_node = onnx.helper.make_node(
            "Sum",
            inputs=chain_sum_inputs[chain_idx],
            outputs=[chain_tail_outputs[chain_idx]],
            name=_uniq(f"{head_name}_to_sum_tail"),
        )
        new_nodes.append(sum_node)

    # Drop value_infos for the eliminated intermediate Add outputs.
    keep_outputs = set()
    for chain, _ in chains:
        keep_outputs.add(chain[-1].output[0])  # tail output preserved
    drop_intermediate = set()
    for chain, _ in chains:
        for n in chain[:-1]:  # all but tail
            drop_intermediate.add(n.output[0])

    new_value_info = [vi for vi in graph.value_info if vi.name not in drop_intermediate]

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

    eliminated = sum(len(c) - 1 for c, _ in chains)
    return SpreadResult(
        pattern_name="add_chain_to_sum",
        verdict="applied",
        new_model=new_model,
        notes=f"collapsed {len(chains)} Add-chain(s); eliminated {eliminated} intermediate Add(s) -> Sum",
        extras={
            "chain_count": len(chains),
            "intermediates_eliminated": eliminated,
            "chain_lengths": [len(c) for c, _ in chains],
        },
    )


PATTERN = SpreadPattern(
    name="add_chain_to_sum",
    category="math-equivalent",
    description="collapse N consecutive Add nodes (left-fold chain) into a single Sum",
    apply=_apply,
    safe_by_construction=True,
)
