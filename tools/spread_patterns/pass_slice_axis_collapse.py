"""Collapse a 4-d Slice that has axis-0 as no-op to a 3-d Slice.

Origin: task000 v2_scalar1 rev2 (2026-05-18). The live shift Slice was
``Slice(input, starts=[0,0,1,1], ends=[1,10,11,11], axes=[0,1,2,3], steps=[1,1,1,1])``
— the axis-0 entry was a no-op since input dim 0 is always 1. Collapsing
to ``axes=[1,2,3]`` removed a duplicate entry from each shared
``starts``/``ends``/``axes``/``steps`` init and let it deduplicate with
other 3-d Slices via dedupe_inits.

Strategy:
  * Find Slice nodes where axes contains a dim that's the batch/leading
    axis (axis 0) AND the corresponding starts == 0 and ends >= dim_size.
  * Rewrite the Slice to drop that entry.

Limitations: opset-10 Slice takes (input, starts, ends, axes, steps) all
as inputs. We only rewrite when starts/ends/axes/steps are initializers
(so we can edit them) and the axis-0 slice is genuinely a no-op given
the input's shape.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

from tools.spread_patterns._base import SpreadPattern, SpreadResult


def _read_int_init(model: onnx.ModelProto, name: str) -> list[int] | None:
    for init in model.graph.initializer:
        if init.name == name:
            try:
                return numpy_helper.to_array(init).flatten().tolist()
            except Exception:  # noqa: BLE001
                return None
    return None


def _set_int_init(model: onnx.ModelProto, name: str, values: list[int]) -> bool:
    for idx, init in enumerate(model.graph.initializer):
        if init.name == name:
            new = numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)
            # Keep the same data_type as the original (int64 by default).
            new.data_type = init.data_type
            model.graph.initializer[idx].CopyFrom(new)
            return True
    return False


def _input_axis0_size(model: onnx.ModelProto, input_name: str) -> int | None:
    """Best-effort: read the leading dim from graph.input or value_info."""
    candidates = list(model.graph.input) + list(model.graph.value_info)
    for v in candidates:
        if v.name != input_name:
            continue
        shape = v.type.tensor_type.shape
        if not shape.dim:
            return None
        d = shape.dim[0]
        if d.dim_value > 0:
            return d.dim_value
    return None


def _apply(model: onnx.ModelProto, task_id: str) -> SpreadResult:
    working = onnx.ModelProto()
    working.CopyFrom(model)

    rewrites = 0
    for node in working.graph.node:
        if node.op_type != "Slice":
            continue
        if len(node.input) < 4:
            # opset-9 Slice with attribute axes — out of scope here.
            continue
        starts_name = node.input[1]
        ends_name = node.input[2]
        axes_name = node.input[3]
        steps_name = node.input[4] if len(node.input) >= 5 else None
        starts = _read_int_init(working, starts_name)
        ends = _read_int_init(working, ends_name)
        axes = _read_int_init(working, axes_name)
        steps = _read_int_init(working, steps_name) if steps_name else None
        if starts is None or ends is None or axes is None:
            continue
        if len(starts) != len(ends) or len(starts) != len(axes):
            continue
        # Find axis-0 entries that are no-ops.
        drop_indices: list[int] = []
        input_axis0_size = _input_axis0_size(working, node.input[0])
        for i, axis in enumerate(axes):
            if axis != 0:
                continue
            if starts[i] != 0:
                continue
            if input_axis0_size is None:
                continue
            # ends >= input_axis0_size means "take all" along axis 0.
            if ends[i] < input_axis0_size:
                continue
            if steps and steps[i] != 1:
                continue
            drop_indices.append(i)
        if not drop_indices:
            continue
        # Rewrite the four/five inits (per-node copies of starts/ends/axes/steps).
        # Note: if these init tensors are shared with other Slices we'd corrupt
        # them. To be safe we only rewrite when each init is referenced exactly
        # once across the graph.
        names_to_rewrite = [starts_name, ends_name, axes_name]
        if steps_name:
            names_to_rewrite.append(steps_name)
        share_ok = True
        for n_name in names_to_rewrite:
            refs = sum(
                1
                for nn in working.graph.node
                for inp in nn.input
                if inp == n_name
            )
            if refs > 1:
                share_ok = False
                break
        if not share_ok:
            continue
        keep_mask = [i for i in range(len(axes)) if i not in drop_indices]
        _set_int_init(working, starts_name, [starts[i] for i in keep_mask])
        _set_int_init(working, ends_name, [ends[i] for i in keep_mask])
        _set_int_init(working, axes_name, [axes[i] for i in keep_mask])
        if steps_name and steps is not None:
            _set_int_init(working, steps_name, [steps[i] for i in keep_mask])
        rewrites += 1

    if rewrites == 0:
        return SpreadResult(
            pattern_name="slice_axis_collapse",
            verdict="no-op",
            notes="no Slice nodes with redundant axis-0 entries to drop",
        )
    return SpreadResult(
        pattern_name="slice_axis_collapse",
        verdict="applied",
        new_model=working,
        notes=f"collapsed axis-0 no-op from {rewrites} Slice node(s)",
        extras={"rewrites": rewrites},
    )


PATTERN = SpreadPattern(
    name="slice_axis_collapse",
    category="math-equivalent",
    description="Drop axis-0 no-op entries from per-Slice starts/ends/axes/steps inits (lets dedupe reuse 3-d inits)",
    apply=_apply,
    safe_by_construction=False,  # shape-correctness verified per-input
)
