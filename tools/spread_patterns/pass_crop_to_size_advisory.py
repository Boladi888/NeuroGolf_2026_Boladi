"""Advisory: report when crop-to-actual-size could win big.

Origin: 2026-05-11 lesson — task320 v1→v4 went 55k → 13.3k (4.2× reduction)
purely from cropping the [1,10,30,30] input to the task's max actual
H×W before running the algorithm, then padding back at the end.

This pattern is TASK-SPECIFIC (max H/W is hardcoded per task), so it
can't be auto-rewritten safely. Instead, this module looks at the task's
JSON to compute the max active grid size and the ONNX's current
intermediate dimensions, and reports the gap as advisory.

Refinements from worker-7 Group R session (2026-05-19):
  - Skip when the graph is ALREADY INTERNALLY CROPPED (an early Slice on
    spatial axes reduces dims below 30; downstream [...,30,30] intermediates
    are pre-Slice scaffolding that won't actually shrink). False positives
    seen on task002 (cropped to 20×20), task084 (21×21), task098 (20×22).
  - Skip when total cost is trivially small (<1000); even 100% reduction
    is below noise (~+0.001 LB). False positive seen on task016 (cost 10,
    single Gather).
  - Flag Gather/ArgMax nodes that index into the 30-wide axis as a risk
    note — cropping breaks the index math. Seen on task029, task355.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import onnx
from onnx import numpy_helper

from tools.spread_patterns._base import SpreadPattern, SpreadResult

# Skip the advisory when total cost would yield < ~0.005 LB even at 100%
# reduction. log(1000/1) ≈ 6.9 ≈ +0.07 LB — generous upper bound. Below
# 1000 cost, the advisory is more noise than signal.
COST_SIGNIFICANCE_FLOOR = 1000

# Ops whose indices/attributes break when the 30-wide axis is cropped.
INDEX_ATTACHED_OPS = {"Gather", "GatherElements", "GatherND", "Scatter", "ScatterElements", "ScatterND", "ArgMax", "ArgMin"}

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_JSON_DIR = REPO_ROOT / "data" / "competitions" / "neurogolf-2026"


def _max_active_grid(task_id: str) -> tuple[int, int] | None:
    """Return (max_H, max_W) across train+test+arc-gen visible pairs."""
    path = TASK_JSON_DIR / f"{task_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    max_h, max_w = 0, 0
    for section in ("train", "test", "arc-gen"):
        pairs = data.get(section) or []
        for pair in pairs:
            for key in ("input", "output"):
                grid = pair.get(key)
                if not isinstance(grid, list):
                    continue
                h = len(grid)
                w = max((len(row) for row in grid if isinstance(row, list)), default=0)
                max_h = max(max_h, h)
                max_w = max(max_w, w)
    if max_h == 0 or max_w == 0:
        return None
    return max_h, max_w


def _intermediate_spatial_dims(model: onnx.ModelProto) -> list[tuple[str, tuple[int, ...]]]:
    """Inspect every value_info and node output's declared shape and
    return a list of (name, shape) for tensors whose last two dims are
    > 1 (i.e. spatial).

    CRITICAL: skip tensors literally named 'input' or 'output' — the
    competition's calculate_memory explicitly excludes those from the
    cost (see neurogolf_utils.py line 240: `if tensor_name in ['input',
    'output']: continue`). They're free. Including them in big_count
    inflates the advisory's expected savings on tasks where the crop
    would shrink ONLY those free tensors and add a non-free input_cropped
    intermediate, NET-INCREASING cost (worker-7 Group V lesson 2026-05-19).
    """
    out: list[tuple[str, tuple[int, ...]]] = []
    for v in list(model.graph.value_info) + list(model.graph.output):
        if v.name in ("input", "output"):
            continue
        shape = v.type.tensor_type.shape
        if not shape.dim:
            continue
        dims = tuple(d.dim_value if d.dim_value > 0 else -1 for d in shape.dim)
        if len(dims) >= 2 and dims[-2] > 1 and dims[-1] > 1:
            out.append((v.name, dims))
    return out


def _early_input_slice_crop(model: onnx.ModelProto) -> tuple[int, int] | None:
    """Detect if the graph crops the input on spatial axes within the first
    few nodes. Returns (sliced_h, sliced_w) if found, else None.

    Pattern: Slice(input, starts, ends, axes, steps) where axes includes
    the spatial axes (2 and/or 3 for [N,C,H,W]) and ends < 30.
    """
    if not model.graph.input:
        return None
    input_name = model.graph.input[0].name
    inits = {i.name: i for i in model.graph.initializer}

    def _scalar_list(name):
        init = inits.get(name)
        if init is None:
            return None
        try:
            return numpy_helper.to_array(init).flatten().tolist()
        except Exception:  # noqa: BLE001
            return None

    # Walk first ~10 nodes; if any Slice operates on the input (directly
    # or via an Identity chain) and reduces spatial dims, return the size.
    producer = {o: n for n in model.graph.node for o in n.output}
    seen = set()
    frontier = [input_name]
    for _ in range(15):
        next_frontier = []
        for name in frontier:
            for node in model.graph.node:
                if name not in node.input:
                    continue
                if id(node) in seen:
                    continue
                seen.add(id(node))
                if node.op_type == "Slice" and len(node.input) >= 4:
                    starts = _scalar_list(node.input[1])
                    ends = _scalar_list(node.input[2])
                    axes = _scalar_list(node.input[3])
                    if starts is None or ends is None or axes is None:
                        continue
                    sliced_h = sliced_w = None
                    for axis, end, start in zip(axes, ends, starts):
                        if axis in (2, -2) and 0 <= (end - start) < 30:
                            sliced_h = end - start
                        elif axis in (3, -1) and 0 <= (end - start) < 30:
                            sliced_w = end - start
                    if sliced_h is not None and sliced_w is not None:
                        return sliced_h, sliced_w
                # Otherwise follow Identity/Reshape/Cast to chase the chain
                if node.op_type in ("Identity", "Cast", "Reshape"):
                    for o in node.output:
                        next_frontier.append(o)
        frontier = next_frontier
        if not frontier:
            break
    return None


def _flatten_matmul_blockers(model: onnx.ModelProto, max_h: int, max_w: int) -> list[str]:
    """Detect Flatten -> {MatMul, Gemm, Gather} patterns where cropping
    breaks because the consumer references positions beyond max_h * max_w
    (the cropped flat dimension).

    Origin: worker-7 Group M (2026-05-19) — task158/task363 had KNN-style
    Flatten-then-MatMul lookups; cropping would drop non-zero weight rows
    at indices >= max_h*max_w, breaking the lookup.

    Extended (worker-7 next session, 2026-05-19) — task215/task313 had
    the same KNN-style problem but with Flatten->Gather instead of
    Flatten->MatMul. The Gather indices reference positions in the flat
    900-dim that exceed crop_flat after the crop.

    Returns a list of warning strings (one per blocker found).
    """
    import numpy as _np

    warnings: list[str] = []
    inits = {i.name: i for i in model.graph.initializer}
    producer: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for out in node.output:
            if out:
                producer[out] = node

    crop_flat = max_h * max_w
    full_flat = 30 * 30  # NeuroGolf always-30x30 input dim

    def _is_downstream_of_flatten(node_input: str) -> bool:
        upstream = producer.get(node_input)
        for _ in range(5):
            if upstream is None:
                return False
            if upstream.op_type == "Flatten":
                return True
            if upstream.op_type in ("Reshape", "Identity", "Cast"):
                if upstream.input:
                    upstream = producer.get(upstream.input[0])
                    continue
            return False
        return False

    for node in model.graph.node:
        # --- MatMul / Gemm consumer of Flatten ---
        if node.op_type in ("MatMul", "Gemm") and len(node.input) >= 2:
            if not _is_downstream_of_flatten(node.input[0]):
                continue
            w_input = node.input[1]
            if w_input not in inits:
                continue
            try:
                w = numpy_helper.to_array(inits[w_input])
            except Exception:  # noqa: BLE001
                continue
            if w.ndim < 1 or w.shape[0] < full_flat:
                continue
            if crop_flat >= w.shape[0]:
                continue
            beyond_crop = w[crop_flat:]
            if _np.any(beyond_crop != 0):
                nonzero_rows = int(
                    _np.sum(_np.any(beyond_crop != 0, axis=tuple(range(1, beyond_crop.ndim))))
                    if beyond_crop.ndim > 1 else _np.count_nonzero(beyond_crop)
                )
                warnings.append(
                    f"Flatten->{node.op_type}({node.name or '?'}) has {nonzero_rows}"
                    f" non-zero weight rows beyond crop_flat={crop_flat} (full {full_flat});"
                    f" cropping breaks the KNN lookup"
                )
            continue

        # --- Gather consumer of Flatten with hardcoded indices ---
        if node.op_type == "Gather" and len(node.input) >= 2:
            data_input, idx_input = node.input[0], node.input[1]
            if not _is_downstream_of_flatten(data_input):
                continue
            if idx_input not in inits:
                continue
            try:
                idx_arr = numpy_helper.to_array(inits[idx_input]).flatten()
            except Exception:  # noqa: BLE001
                continue
            # Indices that reference positions >= crop_flat will dangle
            # after cropping (their data positions no longer exist).
            bad = [int(v) for v in idx_arr if isinstance(v.item() if hasattr(v, "item") else v, (int, _np.integer)) and int(v) >= crop_flat]
            if bad:
                warnings.append(
                    f"Flatten->Gather({node.name or '?'}) has {len(bad)} index value(s)"
                    f" >= crop_flat={crop_flat} (max idx {max(bad)}); cropping breaks the flat-index lookup"
                )
    return warnings


def _index_attached_warnings(model: onnx.ModelProto, max_dim: int) -> list[str]:
    """Find Gather/Scatter/ArgMax ops whose indices touch values >= max_dim
    along the 30-wide axis. These break under crop."""
    warnings: list[str] = []
    inits = {i.name: i for i in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type not in INDEX_ATTACHED_OPS:
            continue
        # Try to read the indices/data input (slot varies by op).
        idx_input = None
        if node.op_type in ("Gather", "GatherElements", "GatherND") and len(node.input) >= 2:
            idx_input = node.input[1]
        elif node.op_type in ("Scatter", "ScatterElements", "ScatterND") and len(node.input) >= 2:
            idx_input = node.input[1]
        if idx_input and idx_input in inits:
            try:
                arr = numpy_helper.to_array(inits[idx_input]).flatten().tolist()
            except Exception:  # noqa: BLE001
                arr = []
            if any(isinstance(v, (int, float)) and v >= max_dim for v in arr):
                warnings.append(f"{node.op_type}({node.name or '?'}) indices touch >= {max_dim}")
        elif node.op_type in ("ArgMax", "ArgMin"):
            warnings.append(f"{node.op_type}({node.name or '?'}) — output indexes the 30-wide axis")
    return warnings


def _apply(model: onnx.ModelProto, task_id: str) -> SpreadResult:
    grid = _max_active_grid(task_id)
    if grid is None:
        return SpreadResult(
            pattern_name="crop_to_size_advisory",
            verdict="skipped",
            notes=f"could not read task JSON or compute grid bounds for {task_id}",
        )
    max_h, max_w = grid
    if max_h >= 30 and max_w >= 30:
        return SpreadResult(
            pattern_name="crop_to_size_advisory",
            verdict="no-op",
            notes="task already runs at 30×30 (no crop opportunity)",
            extras={"max_h": max_h, "max_w": max_w},
        )
    intermediates = _intermediate_spatial_dims(model)
    # Count how many [..., 30, 30] intermediates the graph currently has.
    big_count = sum(1 for _, d in intermediates if d[-2:] == (30, 30))
    if big_count == 0:
        return SpreadResult(
            pattern_name="crop_to_size_advisory",
            verdict="no-op",
            notes=(
                f"max active grid is {max_h}×{max_w} but no [...,30,30]"
                " intermediates detected — graph may already be cropped"
            ),
            extras={"max_h": max_h, "max_w": max_w, "intermediate_count": len(intermediates)},
        )

    # Refinement 1 (worker-7 Group R lesson): detect if the graph already
    # has an early Slice cropping the input. This is INFORMATIONAL only —
    # task400 succeeded at +0.076 despite having an existing crop, because
    # adding the prefix Slice/Pad still cropped the upstream [...,30,30]
    # scaffolding. So we record the fact in extras and downgrade the
    # advisory's confidence (lower expected savings) but DON'T skip.
    early_slice = _early_input_slice_crop(model)
    if early_slice is not None:
        sh, sw = early_slice
        # If the existing slice is EQUAL or SMALLER than max_grid, savings
        # are diminished but not zero — keep advisory at reduced confidence.
        already_partially_cropped = True
    else:
        already_partially_cropped = False
        sh = sw = 0

    # Refinement 2 (worker-7 Group R lesson): trivial-cost graphs aren't
    # worth advising on. Even 100% reduction on cost=100 is ~+0.005 LB.
    total_params = sum(
        int(numpy_helper.to_array(init).size) for init in model.graph.initializer
        if init.HasField("name")
    ) if False else 0  # cheap heuristic: count initializer element-count proxy
    # Better: just count nodes + initializers; below 5 each is "trivial graph".
    if len(model.graph.node) < 5 and len(model.graph.initializer) < 5:
        return SpreadResult(
            pattern_name="crop_to_size_advisory",
            verdict="no-op",
            notes=(
                f"graph is trivially small ({len(model.graph.node)} nodes, "
                f"{len(model.graph.initializer)} initializers); advisory is "
                f"noise below the significance floor"
            ),
            extras={"max_h": max_h, "max_w": max_w, "trivial_graph": True},
        )

    # Refinement 3 (worker-7 Group R lesson): flag index-attached ops
    # (Gather, Scatter, ArgMax, etc.) that touch the 30-wide axis. These
    # break under crop without index-rewiring; manual workers should
    # widen the crop or use a different technique.
    index_warnings = _index_attached_warnings(model, max_dim=max(max_h, max_w))

    # Refinement 5 (worker-7 Group M lesson, 2026-05-19): flag Flatten
    # -> MatMul/Gemm patterns where weight tensor has nonzero entries
    # beyond max_h*max_w (the cropped flat-spatial dimension). Cropping
    # breaks the KNN-style lookup because those weight rows aren't all
    # zero. Seen on task158 and task363 — both had ~hundreds of KB in
    # nominal savings but were infeasible due to this pattern.
    flatten_matmul_blockers = _flatten_matmul_blockers(model, max_h, max_w)

    # Rough savings estimate: each [1,10,30,30] float32 intermediate costs
    # 36k bytes. Cropping to max_h×max_w drops it to ~max_h*max_w*40 bytes.
    # If the graph is already partially-cropped, scale savings down (the
    # downstream-of-existing-slice paths won't shrink further).
    cropped_bytes_each = max_h * max_w * 40
    saved_each = max(0, 36000 - cropped_bytes_each)
    estimated_savings = saved_each * big_count
    if already_partially_cropped:
        estimated_savings = estimated_savings // 4  # rough discount

    # Refinement 6 (worker-7 Group U lesson, 2026-05-19): when big_count is
    # small (1-2), the few [...,30,30] tensors are typically output-bound
    # scaffolding produced AFTER an internal small-dim algorithmic core.
    # Cropping the INPUT doesn't propagate to those output-side tensors
    # without rewriting Gather/Concat/Pad ops downstream — usually
    # infeasible without a full algorithm rewrite. Workers tried this on
    # task219 (big_count=1, max 15x10) and task242 (big_count=1, max 16x16)
    # and both abandoned. Apply aggressive discount so the net-savings
    # check filters them out cleanly.
    small_count_discount = 1.0
    if big_count == 1:
        small_count_discount = 0.25
    elif big_count == 2:
        small_count_discount = 0.50
    if small_count_discount < 1.0:
        estimated_savings = int(estimated_savings * small_count_discount)

    # Refinement 4 (worker-7 Group V lesson, 2026-05-19): the crop
    # transformation ADDS overhead: a new `input_cropped` intermediate
    # at [1,10,max_h,max_w] F32 (= max_h*max_w*40 B) AND in some shapes
    # a `output_padded` of the same size. Both DO count in score_v2
    # (only literal 'input'/'output' are skipped). On tasks where the
    # original [...,30,30] tensors that big_count counts are downstream-
    # short or already-cropped, this overhead can exceed savings,
    # net-INCREASING cost. Proven on task073 (1411 -> 2423). Subtract
    # overhead from savings and downgrade to no-op when net is negative
    # or trivially small.
    crop_overhead = 2 * cropped_bytes_each  # input_cropped + output_padded
    net_savings = estimated_savings - crop_overhead
    if net_savings <= 0:
        return SpreadResult(
            pattern_name="crop_to_size_advisory",
            verdict="no-op",
            notes=(
                f"max active grid is {max_h}×{max_w} and graph has {big_count}"
                f" [...,30,30] intermediate(s), but the crop's input_cropped"
                f"/output_padded overhead (~{crop_overhead}B) exceeds the"
                f" estimated savings ({estimated_savings}B). Net cost would"
                f" INCREASE — skip."
            ),
            extras={
                "max_h": max_h, "max_w": max_w,
                "intermediates_at_30x30_counted": big_count,
                "raw_savings_estimate": estimated_savings,
                "crop_overhead_estimate": crop_overhead,
                "net_savings_estimate": net_savings,
                "already_partially_cropped_to": [sh, sw] if already_partially_cropped else None,
            },
        )
    # Update the headline number to reflect net savings, not gross.
    estimated_savings = net_savings
    notes = (
        f"max active grid is {max_h}×{max_w} but graph has {big_count}"
        f" [...,30,30] intermediate(s). Crop-process-pad could save"
        f" ~{estimated_savings} bytes (rough est)."
    )
    if already_partially_cropped:
        notes += (
            f" NOTE: graph already has an early Slice cropping to {sh}×{sw};"
            f" savings are partial (estimate discounted)."
        )
    if index_warnings:
        notes += f" RISK: {len(index_warnings)} index-attached op(s) touch the 30-axis"
        notes += " — crop will require rewriting indices."
    if flatten_matmul_blockers:
        notes += f" BLOCKER: {len(flatten_matmul_blockers)} Flatten->MatMul/Gemm pattern(s) with"
        notes += " non-zero weight rows beyond cropped flat-dim — crop infeasible without weight rewrite."
    return SpreadResult(
        pattern_name="crop_to_size_advisory",
        verdict="advisory",
        cost_delta=-estimated_savings,
        notes=notes,
        extras={
            "max_h": max_h,
            "max_w": max_w,
            "intermediates_at_30x30": big_count,
            "estimated_savings_bytes": estimated_savings,
            "already_partially_cropped_to": [sh, sw] if already_partially_cropped else None,
            "index_op_warnings": index_warnings,
            "flatten_matmul_blockers": flatten_matmul_blockers,
        },
    )


PATTERN = SpreadPattern(
    name="crop_to_size_advisory",
    category="advisory",
    description="Flag tasks whose max active grid << 30×30 but graph runs at 30×30 (crop-process-pad opportunity)",
    apply=_apply,
    safe_by_construction=True,  # advisory only, no graph rewrite
)
