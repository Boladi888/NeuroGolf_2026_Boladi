"""Replace large all-ones (or all-zeros) initializers with scalar + broadcast.

Origin: task000 v2_scalar1 rev1 (2026-05-18). The live ``ones10`` init
(shape ``[1,1,10,10]``, value ``1.0``) was used in a Sub op. Replacing
it with a scalar 1.0 (shape ``[]``) keeps the math identical via
broadcasting but drops 99 of the 100 ``params``. Δ–99 bytes verified
byte-exact.

This pattern generalizes: ANY initializer whose data is all the same
scalar value (1.0, 0.0, etc.) and is consumed by an op that broadcasts
the second operand can be replaced with a scalar of that value.

Safety:
  * We only rewrite when the consuming op-types are known to broadcast
    elementwise (Add, Sub, Mul, Div, Pow, Max, Min, Where).
  * We refuse to rewrite when the initializer is also consumed by an op
    that does NOT broadcast (e.g. Conv weights, Slice indices,
    Reshape shape arguments, Gather indices).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import onnx
from onnx import numpy_helper

from tools.spread_patterns._base import SpreadPattern, SpreadResult

BROADCAST_OPS = frozenset({
    "Add", "Sub", "Mul", "Div", "Pow", "Max", "Min", "Where",
    "Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual",
})


def _consumers(model: onnx.ModelProto, name: str) -> Iterable[tuple[onnx.NodeProto, int]]:
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp == name:
                yield node, i


def _is_uniform_init(init: onnx.TensorProto) -> tuple[bool, float, int]:
    """Return (is_uniform, fill_value, element_count). Only float32 inits."""
    if init.data_type != onnx.TensorProto.FLOAT:
        return False, 0.0, 0
    arr = numpy_helper.to_array(init)
    if arr.size <= 1:
        return False, 0.0, arr.size
    flat = arr.flatten()
    first = float(flat[0])
    if not np.all(flat == first):
        return False, 0.0, arr.size
    return True, first, arr.size


def _apply(model: onnx.ModelProto, task_id: str) -> SpreadResult:
    working = onnx.ModelProto()
    working.CopyFrom(model)

    replaced: list[tuple[str, float, int]] = []
    new_inits = list(working.graph.initializer)
    init_lookup = {init.name: idx for idx, init in enumerate(new_inits)}

    for idx, init in enumerate(list(new_inits)):
        is_uniform, fill, element_count = _is_uniform_init(init)
        if not is_uniform or element_count < 4:
            continue
        # Check all consumers broadcast.
        consumers = list(_consumers(working, init.name))
        if not consumers:
            continue
        if any(n.op_type not in BROADCAST_OPS for n, _ in consumers):
            continue
        # Build a scalar replacement with the same data_type.
        scalar = onnx.helper.make_tensor(
            init.name,
            init.data_type,
            [],  # rank-0 scalar
            [fill],
        )
        new_inits[idx] = scalar
        replaced.append((init.name, fill, element_count))

    if not replaced:
        return SpreadResult(
            pattern_name="scalar_broadcast_ones",
            verdict="no-op",
            notes="no uniform-valued float initializers consumed only by broadcasting ops",
        )

    # Write back. graph.initializer is a repeated field on the proto.
    del working.graph.initializer[:]
    working.graph.initializer.extend(new_inits)
    total_saved_elements = sum(elts - 1 for _, _, elts in replaced)
    notes = (
        f"replaced {len(replaced)} uniform init(s) with scalars; "
        f"~{total_saved_elements} fewer param elements"
    )
    return SpreadResult(
        pattern_name="scalar_broadcast_ones",
        verdict="applied",
        new_model=working,
        notes=notes,
        extras={"replaced": replaced, "saved_elements": total_saved_elements},
    )


PATTERN = SpreadPattern(
    name="scalar_broadcast_ones",
    category="math-equivalent",
    description="Uniform-value initializer → scalar (broadcasts to original shape under elementwise ops)",
    apply=_apply,
    safe_by_construction=False,  # broadcast semantics need verification per consumer
)
