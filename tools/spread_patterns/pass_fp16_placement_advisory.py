"""Advisory: flag tasks where placement-aware F16 casting could save
bytes that blanket ``fp16_selective`` misses.

Origin: 2026-05-19 F16 placement strategy insight (task055 et al).
The existing ``pass_fp16_selective`` runs ``onnxconverter_common`` over
the whole graph and frequently regresses because it pre-casts the full
``[1, 10, 30, 30]`` input — the Cast itself is large, and the F16
benefit on a wide tensor doesn't recoup the Cast overhead. Workers
discovered that the right placement is AFTER a narrowing op (Slice,
Gather) so the Cast operates on a smaller tensor.

This pattern doesn't transform the graph. It identifies candidate
"placement points" and reports them as advisories so workers (or a
future smarter transform) can act.

Three flagged shapes:

  **POST_SLICE_F16**: a Slice consumed by an arithmetic chain. If the
    Slice output is smaller than its input, inserting F16 here is
    cheaper than upstream of the Slice. Reported tensor: Slice output
    name; estimated savings: (input_elems - output_elems) * 2 bytes
    (the 2-byte F16-vs-F32 differential on the avoided wide path).

  **POST_GATHER_PER_CHANNEL**: a Gather node that selects individual
    channels. Per-channel placement (Cast each gathered channel
    separately) beats pre-casting the full tensor when fewer than ~6
    channels are selected. Reported per Gather group: channels_count,
    estimated_bytes_saved.

  **HASH_DECODE_PROHIBITED**: a Gather feeding MatMul with a large
    weight initializer (a hash-decode pattern). Flag as "DO NOT cast
    this branch to F16" — workers must keep this side F32. Useful as
    a guardrail.

The detector is conservative: only reports candidates where shape
inference resolved sizes and the savings estimate is positive. No
transform is emitted (verdict is "advisory").
"""
from __future__ import annotations

from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

from tools.spread_patterns._base import SpreadPattern, SpreadResult


# Ops that are safe-ish to run in F16 (will accept F16 inputs and
# produce same logical result modulo rounding). NeuroGolf workloads
# are integer-quantized at the input, so most ops here are lossless.
F16_SAFE_OPS = frozenset({
    "Add", "Sub", "Mul", "Div", "Neg", "Abs", "Relu", "Sigmoid",
    "Tanh", "Sum", "Mean", "ReduceSum", "ReduceMax", "ReduceMin",
    "ReduceMean", "Min", "Max", "Concat", "Slice", "Gather",
    "Identity", "Transpose", "Reshape", "Squeeze", "Unsqueeze",
})

# Ops we should NEVER cast through — precision-critical or unsupported.
F16_AVOID_OPS = frozenset({
    "MatMul",  # hash-decode killer
    "Gemm",
    "Conv",
    "ConvTranspose",
    "Equal",   # opset-10 INT32 only
    "Greater", "Less", "And", "Or", "Not", "Xor",  # bool ops
})


def _tensor_size(value_info: onnx.ValueInfoProto) -> int | None:
    """Return product of static dims if all known, else None."""
    if not value_info.type.tensor_type.shape.dim:
        return None
    n = 1
    for d in value_info.type.tensor_type.shape.dim:
        if d.dim_value <= 0:
            return None
        n *= d.dim_value
    return n


def _build_consumer_map(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    out: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                out.setdefault(inp, []).append(node)
    return out


def _value_info_index(graph: onnx.GraphProto) -> dict[str, onnx.ValueInfoProto]:
    idx = {vi.name: vi for vi in graph.value_info}
    for vi in graph.input:
        idx.setdefault(vi.name, vi)
    for vi in graph.output:
        idx.setdefault(vi.name, vi)
    return idx


def _apply(model: onnx.ModelProto, task_id: str) -> SpreadResult:
    graph = model.graph
    consumers = _build_consumer_map(graph)
    vi = _value_info_index(graph)
    init_names = {i.name: i for i in graph.initializer}

    post_slice_candidates: list[dict] = []
    gather_groups: dict[str, list[onnx.NodeProto]] = {}  # input -> Gathers
    hash_decode_warnings: list[dict] = []

    for node in graph.node:
        if node.op_type == "Slice":
            inp = node.input[0] if node.input else ""
            out = node.output[0] if node.output else ""
            in_sz = _tensor_size(vi.get(inp)) if inp in vi else None
            out_sz = _tensor_size(vi.get(out)) if out in vi else None
            if in_sz is None or out_sz is None or out_sz >= in_sz:
                continue
            # Downstream chain must be F16-safe.
            downstream = consumers.get(out, [])
            if not downstream:
                continue
            if any(d.op_type in F16_AVOID_OPS for d in downstream):
                continue
            if not all(d.op_type in F16_SAFE_OPS for d in downstream):
                continue
            savings = (in_sz - out_sz) * 2  # F32 vs F16 byte differential
            post_slice_candidates.append({
                "slice_node": node.name or f"<slice@{out}>",
                "input": inp,
                "output": out,
                "in_elems": in_sz,
                "out_elems": out_sz,
                "estimated_bytes_saved": savings,
            })
        elif node.op_type == "Gather":
            inp = node.input[0] if node.input else ""
            gather_groups.setdefault(inp, []).append(node)
            # If Gather feeds a MatMul with a large weight init, that's a
            # hash-decode signal — warn against casting this branch.
            for c in consumers.get(node.output[0], []):
                if c.op_type in ("MatMul", "Gemm"):
                    # Check that c has a large weight initializer input
                    for other_inp in c.input:
                        if other_inp == node.output[0]:
                            continue
                        if other_inp in init_names:
                            try:
                                arr = numpy_helper.to_array(init_names[other_inp])
                                if arr.size >= 1000:
                                    hash_decode_warnings.append({
                                        "matmul_node": c.name or f"<matmul@{c.output[0]}>",
                                        "weight_init": other_inp,
                                        "weight_elems": int(arr.size),
                                        "advice": "DO NOT cast this branch to F16 (hash-decode protected)",
                                    })
                            except Exception:  # noqa: BLE001
                                pass

    # Per-channel Gather candidate: group of N>=2 Gathers sharing the same input
    # AND each selecting a different scalar/small slice. If N is moderate
    # (2..6) and the full input is much larger than each gathered channel,
    # per-channel cast wins.
    per_channel_candidates: list[dict] = []
    for inp, gathers in gather_groups.items():
        if len(gathers) < 2 or len(gathers) > 9:
            continue
        in_sz = _tensor_size(vi.get(inp)) if inp in vi else None
        if in_sz is None:
            continue
        gathered_total = 0
        skip = False
        for g in gathers:
            gout = g.output[0] if g.output else ""
            gsz = _tensor_size(vi.get(gout)) if gout in vi else None
            if gsz is None:
                skip = True
                break
            gathered_total += gsz
        if skip:
            continue
        if gathered_total >= in_sz:
            continue
        # Per-channel: cast each gathered output (gathered_total * 2 bytes).
        # Pre-cast: cast full input (in_sz * 2 bytes).
        # Diff = (in_sz - gathered_total) * 2.
        savings = (in_sz - gathered_total) * 2
        if savings <= 0:
            continue
        per_channel_candidates.append({
            "input": inp,
            "n_gathers": len(gathers),
            "input_elems": in_sz,
            "gathered_elems": gathered_total,
            "estimated_bytes_saved": savings,
        })

    if not post_slice_candidates and not per_channel_candidates and not hash_decode_warnings:
        return SpreadResult(
            pattern_name="fp16_placement_advisory",
            verdict="no-op",
            notes="no fp16 placement opportunities detected",
        )

    summary_parts = []
    if post_slice_candidates:
        summary_parts.append(f"{len(post_slice_candidates)} post-Slice candidate(s)")
    if per_channel_candidates:
        summary_parts.append(f"{len(per_channel_candidates)} per-channel Gather candidate(s)")
    if hash_decode_warnings:
        summary_parts.append(f"{len(hash_decode_warnings)} hash-decode no-cast warning(s)")
    total_savings = (
        sum(c["estimated_bytes_saved"] for c in post_slice_candidates)
        + sum(c["estimated_bytes_saved"] for c in per_channel_candidates)
    )

    return SpreadResult(
        pattern_name="fp16_placement_advisory",
        verdict="advisory",
        notes=f"flagged: {'; '.join(summary_parts)}; est savings if all applied: ~{total_savings} bytes",
        extras={
            "post_slice_candidates": post_slice_candidates,
            "per_channel_candidates": per_channel_candidates,
            "hash_decode_warnings": hash_decode_warnings,
            "estimated_bytes_saved_total": total_savings,
        },
    )


PATTERN = SpreadPattern(
    name="fp16_placement_advisory",
    category="advisory",
    description="flag placement-aware F16 cast opportunities (post-Slice, per-channel Gather) and hash-decode no-cast warnings; no transform emitted",
    apply=_apply,
    safe_by_construction=True,
)
