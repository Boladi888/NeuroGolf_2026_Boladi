"""Constant nodes → graph.initializer (math-equivalent free memory).

History: 2026-05-12 lesson — Constant node outputs are counted as
intermediate memory by ``calculate_memory``, while initializers are
not. Moving each Constant tensor to ``graph.initializer`` (keeping the
same name binding) drops memory cost with byte-exact output. Combined
across 7 tasks: +7.07 LB.

Wraps the existing tool ``tools/const_to_init.py``.
"""
from __future__ import annotations

import onnx

from tools.const_to_init import convert_constants_to_initializers
from tools.spread_patterns._base import SpreadPattern, SpreadResult


def _apply(model: onnx.ModelProto, task_id: str) -> SpreadResult:
    # Work on a copy so the original isn't mutated.
    working = onnx.ModelProto()
    working.CopyFrom(model)
    try:
        new_model, converted = convert_constants_to_initializers(working)
    except Exception as exc:  # noqa: BLE001
        return SpreadResult(
            pattern_name="const_to_init",
            verdict="error",
            notes=f"const_to_init raised {type(exc).__name__}: {exc}",
        )
    if converted == 0:
        return SpreadResult(
            pattern_name="const_to_init",
            verdict="no-op",
            notes="no Constant nodes with a 'value' attribute to convert",
        )
    return SpreadResult(
        pattern_name="const_to_init",
        verdict="applied",
        new_model=new_model,
        notes=f"moved {converted} Constant node(s) into graph.initializer",
        extras={"converted_count": converted},
    )


PATTERN = SpreadPattern(
    name="const_to_init",
    category="math-equivalent",
    description="Move Constant node values into graph.initializer (calculate_memory ignores initializers)",
    apply=_apply,
    safe_by_construction=True,
)
