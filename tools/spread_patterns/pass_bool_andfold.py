"""Replace Cast-then-Mul on bool masks with And + single trailing Cast.

Origin: 2026-05-19 worker insight on task293 — saved 6,750 bytes
(675 B/channel x 10 channels) by keeping the mask in bool dtype through
the And and casting only once at the end.

Two variants are handled:

  **Variant A (Cast x Cast)** — both Mul inputs are Cast-from-bool to F32:
    Mul(Cast(bool_a, F32), Cast(bool_b, F32))
      -> Cast(And(bool_a, bool_b), F32)
    This is the task293 case (10 such pairs, one per output channel).
    Wins: 2 intermediate F32 tensors collapse to 1 bool tensor (4x
    smaller); the second Cast is eliminated entirely (node count -1).

  **Variant B (Cast x 0/1-float-init)** — one input is Cast-from-bool,
    other is a 0/1-valued float initializer:
    Mul(Cast(bool_op, F32), F_init_0_or_1)
      -> Cast(And(bool_op, F_init != 0 as bool), F32)
    Wins: F32 init (4 bytes/elem) becomes bool init (1 byte/elem);
    intermediate Cast(bool, F32) replaced by And(bool, bool).

Math equivalence: ``a * b`` where each is 0/1 == ``a AND b`` cast back
to the same numeric dtype. Same for Cast x init case when init is 0/1.

Savings sources:
  * Intermediate ``Cast(bool, F32)`` output (4 bytes/elem) is no longer
    produced — downstream consumer is ``Cast(And, F32)`` which sees the
    smaller bool tensor as input.
  * The 0/1-valued F32 initializer (4 bytes/elem) is replaced by a bool
    initializer (1 byte/elem). If no other consumer needs the F32 form,
    the original init becomes garbage-collectable.
  * Node count is conserved (Mul -> And, intermediate Cast stays — but
    on a smaller tensor).

Strategy:
  1. Find ``Mul`` nodes where:
       a. One input is the output of a ``Cast`` node whose input is a
          bool-producing op (Equal/Greater/Less/Not/And/Or/Xor) AND
          whose target dtype is F32.
       b. The OTHER input is an initializer whose every element is
          exactly 0.0 or 1.0 (within float tolerance).
  2. Build a new bool initializer where ``bool[i] = (float[i] == 1.0)``.
  3. Rewrite:
       - Drop the intermediate Cast (its output becomes orphaned).
       - Replace the Mul with ``And(bool_op_output, new_bool_init)``.
       - Append ``Cast(And_output, F32)`` to preserve Mul.output[0]'s name.
       Note: the And+Cast combo has the same node count as the original
       Cast+Mul, but the F32 init -> bool init swap is a pure win.
  4. Drop the original F32 init only if no other consumer references it.

Safety:
  * The original Mul must consume exactly the Cast output and the F32
    init. If the F32 mask init has OTHER consumers, the init is kept
    (only the Mul is rewritten).
  * The Cast output must have ONLY this Mul as consumer; otherwise the
    Cast stays (still some byte win from bool init, but smaller).
  * If the Mul output is the graph output, the substituted Cast(And, F32)
    must preserve the original output name (handled by reusing
    Mul.output[0] on the appended Cast).
  * Bool-producing ops list is conservative: Equal/Greater/Less/Not/And/
    Or/Xor/GreaterOrEqual/LessOrEqual. Any other op is treated as
    "dtype unknown" and skipped.

Out of scope:
  * Mul where the float-mask side is a Constant NODE (not an initializer).
    Could extend, but onnxsim usually folds Constant -> initializer first.
  * Float masks with non-0/1 values that happen to equal a single
    multiplicative scalar (those go through scalar_broadcast_ones).
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


def _init_array(graph: onnx.GraphProto, name: str):
    for init in graph.initializer:
        if init.name == name:
            try:
                return init, numpy_helper.to_array(init)
            except Exception:  # noqa: BLE001
                return init, None
    return None, None


def _is_zero_one_float(arr) -> bool:
    if arr is None or arr.size == 0:
        return False
    try:
        flat = arr.astype(np.float64, copy=False)
    except (TypeError, ValueError):
        return False
    # Every element within 1e-6 of either 0.0 or 1.0
    zero_close = np.abs(flat - 0.0) < 1e-6
    one_close = np.abs(flat - 1.0) < 1e-6
    return bool(np.all(zero_close | one_close))


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
    producer: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer[out] = node
    consumers = _build_consumer_map(graph)

    # Two rewrite shapes:
    #   variant 'CC': (mul, cast_a, bool_src_a, cast_b, bool_src_b)  — Cast x Cast
    #   variant 'CI': (mul, cast, bool_src, init_name, bool_arr, new_bool_name) — Cast x init
    rewrites_cc: list[tuple] = []
    rewrites_ci: list[tuple] = []

    def _is_cast_from_bool_to_f32(node):
        """Return (cast_src_name) if node is Cast(bool_op_output, F32), else None."""
        if node is None or node.op_type != "Cast":
            return None
        if not node.input:
            return None
        cast_src = node.input[0]
        src_node = producer.get(cast_src)
        if src_node is None or src_node.op_type not in BOOL_PRODUCING_OPS:
            return None
        to_attr = next((a for a in node.attribute if a.name == "to"), None)
        if to_attr is None or to_attr.i != TensorProto.FLOAT:
            return None
        return cast_src

    for mul_node in graph.node:
        if mul_node.op_type != "Mul" or len(mul_node.input) != 2:
            continue
        a, b = mul_node.input[0], mul_node.input[1]
        cast_a = producer.get(a)
        cast_b = producer.get(b)
        bool_src_a = _is_cast_from_bool_to_f32(cast_a) if cast_a else None
        bool_src_b = _is_cast_from_bool_to_f32(cast_b) if cast_b else None

        # Variant CC: both inputs are Cast-from-bool-to-F32
        if bool_src_a and bool_src_b:
            # We allow shared Casts (e.g. task293's `valid_mask` feeds 10
            # parallel Muls). Bookkeeping decides later whether each Cast
            # node can be dropped (drop iff ALL its consumers are folded
            # Muls in this rewrite set).
            rewrites_cc.append((mul_node, cast_a, bool_src_a, cast_b, bool_src_b))
            continue

        # Variant CI: one is Cast-from-bool, other is 0/1-float init
        for cast_node, bool_src, init_inp in [
            (cast_a, bool_src_a, b),
            (cast_b, bool_src_b, a),
        ]:
            if cast_node is None or bool_src is None:
                continue
            if len(consumers.get(cast_node.output[0], [])) != 1:
                continue
            init_proto, init_arr = _init_array(graph, init_inp)
            if init_proto is None or not _is_zero_one_float(init_arr):
                continue
            bool_arr = (init_arr > 0.5).astype(bool)
            new_bool_name = f"{init_inp}_bool_andfold"
            rewrites_ci.append((mul_node, cast_node, bool_src, init_inp, bool_arr, new_bool_name))
            break

    if not rewrites_cc and not rewrites_ci:
        return SpreadResult(
            pattern_name="bool_andfold",
            verdict="no-op",
            notes="no Cast(bool,F32)+Mul patterns detected",
        )

    drop_nodes = set()  # ids
    new_nodes: list[onnx.NodeProto] = []
    new_initializers: list[onnx.TensorProto] = list(graph.initializer)
    seen_new_inits: set[str] = set()
    appended_pairs: dict[int, list[onnx.NodeProto]] = {}

    # Variant CC: drop the Mul, emit And(bool_a, bool_b) + Cast.
    # A given Cast node may feed multiple Muls — only drop the Cast if every
    # consumer is a Mul we're rewriting in this pass.
    rewritten_mul_ids = {id(mul) for mul, *_ in rewrites_cc}
    cast_safe_to_drop: dict[int, bool] = {}
    for mul_node, cast_a, _, cast_b, _ in rewrites_cc:
        for cast in (cast_a, cast_b):
            if id(cast) in cast_safe_to_drop:
                continue
            all_folded = all(
                id(c) in rewritten_mul_ids for c in consumers.get(cast.output[0], [])
            )
            cast_safe_to_drop[id(cast)] = all_folded

    # Counter for unique node names — Mul nodes in NeuroGolf graphs are
    # often unnamed, so falling back to the mul name + suffix creates
    # duplicate node names. Disambiguate by appending a serial index.
    _counter = [0]

    def _next_name(base: str) -> str:
        _counter[0] += 1
        return f"{base}_{_counter[0]}"

    for mul_node, cast_a, bool_src_a, cast_b, bool_src_b in rewrites_cc:
        drop_nodes.add(id(mul_node))
        if cast_safe_to_drop.get(id(cast_a)):
            drop_nodes.add(id(cast_a))
        if cast_safe_to_drop.get(id(cast_b)):
            drop_nodes.add(id(cast_b))
        and_out = f"{mul_node.output[0]}_andfold_bool"
        and_node = helper.make_node(
            "And",
            inputs=[bool_src_a, bool_src_b],
            outputs=[and_out],
            name=_next_name(f"{mul_node.name or 'mul'}_andfold_and"),
        )
        cast_back = helper.make_node(
            "Cast",
            inputs=[and_out],
            outputs=[mul_node.output[0]],
            name=_next_name(f"{mul_node.name or 'mul'}_andfold_cast"),
            to=TensorProto.FLOAT,
        )
        appended_pairs[id(mul_node)] = [and_node, cast_back]

    # Variant CI: same as before
    for mul_node, cast_node, bool_src, init_name, bool_arr, new_bool_name in rewrites_ci:
        drop_nodes.add(id(mul_node))
        drop_nodes.add(id(cast_node))
        if new_bool_name not in seen_new_inits:
            new_initializers.append(numpy_helper.from_array(bool_arr, name=new_bool_name))
            seen_new_inits.add(new_bool_name)
        and_out = f"{mul_node.output[0]}_andfold_bool"
        and_node = helper.make_node(
            "And",
            inputs=[bool_src, new_bool_name],
            outputs=[and_out],
            name=_next_name(f"{mul_node.name or 'mul'}_andfold_and"),
        )
        cast_back = helper.make_node(
            "Cast",
            inputs=[and_out],
            outputs=[mul_node.output[0]],
            name=_next_name(f"{mul_node.name or 'mul'}_andfold_cast"),
            to=TensorProto.FLOAT,
        )
        appended_pairs[id(mul_node)] = [and_node, cast_back]

    # Detect F32 init names that still have non-Mul consumers; we must keep them.
    f32_init_dropped = set()
    for _, _, _, init_name, _, _ in rewrites_ci:
        still_used = False
        for n in graph.node:
            if id(n) in drop_nodes:
                continue
            if init_name in n.input:
                still_used = True
                break
        if not still_used:
            f32_init_dropped.add(init_name)

    for node in graph.node:
        if id(node) in drop_nodes:
            if id(node) in appended_pairs:
                new_nodes.extend(appended_pairs[id(node)])
            continue
        new_nodes.append(node)

    final_inits = [init for init in new_initializers if init.name not in f32_init_dropped]

    # Drop value_info entries for tensors that no longer exist (cast output names)
    dropped_intermediate_names = set()
    for _, cast_a, _, cast_b, _ in rewrites_cc:
        dropped_intermediate_names.add(cast_a.output[0])
        dropped_intermediate_names.add(cast_b.output[0])
    for _, cast_node, _, _, _, _ in rewrites_ci:
        dropped_intermediate_names.add(cast_node.output[0])
    new_value_info = [vi for vi in graph.value_info if vi.name not in dropped_intermediate_names]

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
    total = len(rewrites_cc) + len(rewrites_ci)
    return SpreadResult(
        pattern_name="bool_andfold",
        verdict="applied",
        new_model=new_model,
        notes=(
            f"folded {len(rewrites_cc)} Cast x Cast Mul(s) + {len(rewrites_ci)} "
            f"Cast x 0/1-init Mul(s); dropped {len(f32_init_dropped)} F32 init(s)"
        ),
        extras={
            "pairs_folded_cc": len(rewrites_cc),
            "pairs_folded_ci": len(rewrites_ci),
            "f32_inits_dropped": sorted(f32_init_dropped),
            "bool_inits_added": sorted(seen_new_inits),
        },
    )


PATTERN = SpreadPattern(
    name="bool_andfold",
    category="math-equivalent",
    description="fold Cast(bool,F32)+Mul(0/1-float-init) into Cast(And(bool, bool-init), F32) — keeps mask bool until final cast",
    apply=_apply,
    safe_by_construction=True,
)
