"""Advisory: flag graphs that end with `Cast(F16 -> F32) -> output`.

Origin: task117 worker-7 (+1.1249 LB) and task092 worker-7 (+0.1258 LB)
both use this trick: when the model's final node is a Cast from F16 to
F32, the F16 tensor's values are typically {0.0, 1.0} (binary mask),
and the official `run_network` post-processing does
`(out > 0).astype(float)` — which produces identical results from F16
{0,1} and F32 {0,1}. Dropping the final Cast saves ~18k+ bytes per task
(the Cast node plus the bigger F32 output buffer that score_v2 counts).
Memory note: `feedback_neurogolf_f16_output_trick.md`.

This is intentionally ADVISORY rather than auto-applied. The transform
itself is structurally trivial (rename the F16 tensor to the model
output, drop the Cast), but the byte-exact verifier in the scanner
compares raw output bytes between original (F32) and new (F16). F16
{0.0, 1.0} bytes != F32 {0.0, 1.0} bytes, so the scanner would always
report verify-failed even though the official scoring would pass. So
we flag the opportunity here and let a worker apply it with manual
verification against the official scoring contract.

Detection criteria:
  * One of the graph outputs has a producer that is a Cast node.
  * The Cast's `to` attribute is FLOAT (F32).
  * The Cast's input tensor has dtype FLOAT16 (per value_info or producer
    output type).
  * Strong-signal upstream: walk back from the Cast input; if any node in
    the chain to the Cast is a boolean op (Greater, Less, Equal, And,
    Or, Not, Xor, GreaterOrEqual, LessOrEqual), the F16 values are
    almost certainly {0, 1} and the drop is safe.

Estimated savings (advisory): the Cast node itself is ~32 bytes of
graph overhead; the bigger win is the F32 output buffer being F16. For
a typical output [1, 10, 30, 30] the F16-vs-F32 delta is 36000 - 18000
= 18000 bytes. We report the per-task estimate.
"""
from __future__ import annotations

from typing import Any

import onnx
from onnx import TensorProto

from tools.spread_patterns._base import SpreadPattern, SpreadResult


# ONNX boolean ops whose output is {0, 1} (or {False, True} cast to numeric)
BOOLEAN_PRODUCERS = frozenset({
    "Greater", "Less", "Equal", "And", "Or", "Not", "Xor",
    "GreaterOrEqual", "LessOrEqual",
})


def _producer_map(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    out: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for o in node.output:
            if o:
                out[o] = node
    return out


def _cast_target_dtype(cast_node: onnx.NodeProto) -> int | None:
    for attr in cast_node.attribute:
        if attr.name == "to":
            return int(attr.i)
    return None


def _input_dtype_from_value_info(graph: onnx.GraphProto, name: str) -> int | None:
    for vi in list(graph.value_info) + list(graph.input):
        if vi.name == name:
            try:
                return int(vi.type.tensor_type.elem_type)
            except Exception:  # noqa: BLE001
                return None
    return None


def _walk_producer_chain_for_boolean(
    producer: dict[str, onnx.NodeProto],
    start: str,
    max_depth: int = 6,
) -> bool:
    seen: set[int] = set()
    frontier = [(start, 0)]
    while frontier:
        name, depth = frontier.pop()
        if depth > max_depth:
            continue
        p = producer.get(name)
        if p is None or id(p) in seen:
            continue
        seen.add(id(p))
        if p.op_type in BOOLEAN_PRODUCERS:
            return True
        for inp in p.input:
            if inp:
                frontier.append((inp, depth + 1))
    return False


def _output_shape_from_vi(graph: onnx.GraphProto, name: str) -> list[int] | None:
    for vi in list(graph.output) + list(graph.value_info):
        if vi.name == name:
            try:
                return [
                    int(d.dim_value) if d.HasField("dim_value") else 0
                    for d in vi.type.tensor_type.shape.dim
                ]
            except Exception:  # noqa: BLE001
                return None
    return None


def _apply(model: onnx.ModelProto, task_id: str) -> SpreadResult:
    graph = model.graph
    producer = _producer_map(graph)

    flagged: list[dict[str, Any]] = []
    est_total_savings = 0
    for out_vi in graph.output:
        out_name = out_vi.name
        cast = producer.get(out_name)
        if cast is None or cast.op_type != "Cast":
            continue
        target_dtype = _cast_target_dtype(cast)
        if target_dtype != TensorProto.FLOAT:
            continue
        # Cast input dtype: prefer value_info, fall back to producer chain
        cast_in = cast.input[0] if cast.input else ""
        in_dtype = _input_dtype_from_value_info(graph, cast_in)
        if in_dtype != TensorProto.FLOAT16:
            continue
        # Strong-signal upstream: boolean producer?
        bool_upstream = _walk_producer_chain_for_boolean(producer, cast_in)
        shape = _output_shape_from_vi(graph, out_name) or []
        element_count = 1
        for d in shape:
            element_count *= max(d, 1)
        est_byte_savings = element_count * 2  # F32 (4 bytes) - F16 (2 bytes) = 2 bytes per element
        est_total_savings += est_byte_savings
        flagged.append({
            "output_name": out_name,
            "cast_input": cast_in,
            "output_shape": shape,
            "boolean_upstream": bool_upstream,
            "estimated_savings_bytes": est_byte_savings,
        })

    if not flagged:
        return SpreadResult(
            pattern_name="final_cast_f16_to_f32_advisory",
            verdict="no-op",
            notes="no Cast(F16->F32)->output tails detected",
        )

    notes = (
        f"{len(flagged)} F16->F32 final-cast tail(s); est. ~{est_total_savings} B savings if applied. "
        f"Manual apply only: byte-exact verifier rejects F16 outputs even when official run_network accepts."
    )
    return SpreadResult(
        pattern_name="final_cast_f16_to_f32_advisory",
        verdict="advisory",
        cost_delta=-est_total_savings,
        notes=notes,
        extras={
            "tails_flagged": len(flagged),
            "estimated_savings_bytes": est_total_savings,
            "tails": flagged,
        },
    )


PATTERN = SpreadPattern(
    name="final_cast_f16_to_f32_advisory",
    category="advisory",
    description="flag graphs ending Cast(F16->F32)->output; manual drop saves output-buffer bytes (task117/092 trick)",
    apply=_apply,
    safe_by_construction=False,
)
