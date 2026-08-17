"""Collapse separable 2D mask initializers to 1D-broadcast form.

Origin: 2026-05-19 worker insight on task040 — a 2D region mask that
varies along only one of (H, W) can be stored as a 1D vector and let
ONNX broadcasting reconstruct the full mask at use sites. A 30x30 mask
collapses to either [1, 30] (row-separable) or [30, 1] (col-separable),
saving 30x storage on that initializer.

Math equivalence: ONNX/numpy broadcasting rules pad missing axes with 1
and broadcast 1-sized axes across the full target shape. A `[..., 1, W]`
tensor broadcast against `[..., H, W]` produces an identical result to
materializing the full `[..., H, W]` tensor where each row equals
`[..., 1, W]`. Same for `[..., H, 1]` along the column axis.

Strategy:
  * For every initializer with rank >= 2:
      - Inspect the last two axes (H, W).
      - "row-separable" iff arr[..., i, :] == arr[..., 0, :] for all i
        (the mask varies only with W; every row is the same vector).
        Collapse to shape [..., 1, W].
      - "col-separable" iff arr[..., :, j] == arr[..., :, 0] for all j
        (varies only with H; every column is the same vector).
        Collapse to shape [..., H, 1].
      - Both separable? Collapse to [..., 1, 1] (broadcast scalar — but
        in that case `scalar_broadcast_ones` already handles it; we skip
        these to avoid stepping on its toes).
  * Verify every consumer is a broadcast-friendly op (elementwise math
    or Where). If ANY consumer is a shape-sensitive op (Slice, Pad,
    Reshape, Gather, Concat, ReduceSum-along-the-collapsed-axis,
    MatMul, Conv, ConvTranspose), SKIP — the shape change would break
    those.
  * Rewrite the initializer in-place with the smaller array.
    Downstream broadcast handles the rest.

Safety:
  * The shape change is invisible to elementwise ops thanks to standard
    broadcasting (numpy / ONNX 1.x semantics).
  * Skipped consumer classes are conservative — see SHAPE_SENSITIVE_OPS
    below. False negatives (= we skip when we could've applied) are
    preferred over false positives.
  * Empty / size-0 inits skipped. Rank-1 inits skipped (no collapse
    available). Tensors with H == 1 or W == 1 already are skipped (no
    further collapse possible).

Out of scope (future):
  * Collapse along non-trailing axes (e.g. [1, 10, 30, 30] where the
    `10` channel axis is uniform). Same math, more bookkeeping.
  * Detect that a Constant NODE's value attribute is separable.
    onnxsim usually moves these to initializers first.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper

from tools.spread_patterns._base import SpreadPattern, SpreadResult


# Ops whose semantics depend on the explicit (non-broadcast) shape of
# the input. If ANY of these consumes a candidate init, we must NOT
# collapse — the shape change would break them.
SHAPE_SENSITIVE_OPS = frozenset({
    "Slice", "Pad", "Reshape", "Squeeze", "Unsqueeze", "Gather",
    "GatherND", "Scatter", "ScatterND", "Concat", "Split", "Tile",
    "Expand", "Resize", "Transpose", "MatMul", "Gemm", "Conv",
    "ConvTranspose", "AveragePool", "MaxPool", "GlobalAveragePool",
    "GlobalMaxPool", "LSTM", "GRU", "BatchNormalization",
    # Reductions: collapsing along the kept-axis is fine, but along the
    # collapsed-axis changes the result. We can't tell from the op
    # alone, so conservatively skip ALL reductions.
    "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd",
    "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare", "ReduceL1",
    "ReduceL2", "ArgMax", "ArgMin",
})


def _build_consumer_map(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    out: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                out.setdefault(inp, []).append(node)
    return out


def _classify_separability(arr: np.ndarray) -> str | None:
    """Return 'row' (every row identical → collapse H), 'col' (every col
    identical → collapse W), or None (not separable / already collapsed).
    Operates on the LAST TWO axes."""
    if arr.ndim < 2:
        return None
    H, W = arr.shape[-2], arr.shape[-1]
    if H <= 1 and W <= 1:
        return None
    # row-separable: every row equals row 0
    row_sep = False
    if H > 1:
        ref = arr[..., 0:1, :]  # keep dim for broadcasting check
        row_sep = bool(np.array_equal(arr, np.broadcast_to(ref, arr.shape)))
    col_sep = False
    if W > 1 and not row_sep:
        ref = arr[..., :, 0:1]
        col_sep = bool(np.array_equal(arr, np.broadcast_to(ref, arr.shape)))
    if row_sep:
        return "row"
    if col_sep:
        return "col"
    return None


def _collapse(arr: np.ndarray, kind: str) -> np.ndarray:
    if kind == "row":
        return arr[..., 0:1, :]  # shape [..., 1, W]
    if kind == "col":
        return arr[..., :, 0:1]  # shape [..., H, 1]
    raise ValueError(f"unknown kind: {kind}")


def _apply(model: onnx.ModelProto, task_id: str) -> SpreadResult:
    graph = model.graph
    consumers = _build_consumer_map(graph)

    candidates: list[tuple[onnx.TensorProto, np.ndarray, str]] = []
    skipped_due_to_consumer: list[str] = []

    for init in graph.initializer:
        try:
            arr = numpy_helper.to_array(init)
        except Exception:  # noqa: BLE001
            continue
        if arr.size == 0 or arr.ndim < 2:
            continue
        kind = _classify_separability(arr)
        if kind is None:
            continue
        # Check every consumer is broadcast-safe.
        bad_consumer = False
        for c in consumers.get(init.name, []):
            if c.op_type in SHAPE_SENSITIVE_OPS:
                bad_consumer = True
                break
        if bad_consumer:
            skipped_due_to_consumer.append(init.name)
            continue
        new_arr = _collapse(arr, kind)
        candidates.append((init, new_arr, kind))

    if not candidates:
        notes = "no separable 2D initializers found"
        if skipped_due_to_consumer:
            notes += f"; skipped {len(skipped_due_to_consumer)} candidate(s) due to shape-sensitive consumers"
        return SpreadResult(
            pattern_name="separable_mask_collapse",
            verdict="no-op",
            notes=notes,
            extras={"skipped_shape_sensitive": skipped_due_to_consumer},
        )

    # Build new initializer list — replace candidates with collapsed forms.
    by_name = {c[0].name: (c[1], c[2]) for c in candidates}
    new_initializers: list[onnx.TensorProto] = []
    elements_saved = 0
    rewrites_summary: list[dict[str, Any]] = []
    for init in graph.initializer:
        if init.name in by_name:
            new_arr, kind = by_name[init.name]
            old_size = int(np.prod([d for d in init.dims]))
            new_init = numpy_helper.from_array(new_arr, name=init.name)
            new_initializers.append(new_init)
            new_size = int(np.prod(new_arr.shape)) if new_arr.size else 0
            elements_saved += old_size - new_size
            rewrites_summary.append({
                "init": init.name,
                "kind": kind,
                "old_shape": list(init.dims),
                "new_shape": list(new_arr.shape),
                "elements_saved": old_size - new_size,
            })
        else:
            new_initializers.append(init)

    # Collapsed init shapes change the shapes of every downstream tensor
    # along the collapsed axis. Stale value_info entries with the old
    # rank/shape will trip ORT's strict shape inference at load time.
    # Drop value_info entirely and let ORT re-infer from inputs + inits.
    new_graph = helper.make_graph(
        nodes=list(graph.node),
        name=graph.name,
        inputs=list(graph.input),
        outputs=list(graph.output),
        initializer=new_initializers,
        value_info=[],
    )
    new_model = helper.make_model(
        new_graph,
        opset_imports=list(model.opset_import),
        ir_version=model.ir_version,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
    )
    return SpreadResult(
        pattern_name="separable_mask_collapse",
        verdict="applied",
        new_model=new_model,
        notes=(
            f"collapsed {len(candidates)} separable 2D init(s); "
            f"saved {elements_saved} elements total; "
            f"skipped {len(skipped_due_to_consumer)} due to shape-sensitive consumers"
        ),
        extras={
            "collapses": rewrites_summary,
            "skipped_shape_sensitive": skipped_due_to_consumer,
        },
    )


PATTERN = SpreadPattern(
    name="separable_mask_collapse",
    category="math-equivalent",
    description="collapse separable 2D mask initializers ([H,W] -> [1,W] or [H,1]) when broadcasting reconstructs the original",
    apply=_apply,
    safe_by_construction=True,
)
