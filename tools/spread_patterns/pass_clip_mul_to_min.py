"""Clip[0,1] + Mul(0/1_mask) → Min(x, mask).

History: 2026-05-12 lesson — for ``Mul(Clip[0,1](x), m)`` where
``m ∈ {0, 1}``: ``Clip[0,1](x) * m == min(x, m)`` (since Clip[0,1] is
``min(x, 1)`` for x ≥ 0 and Conv-of-bool always is). Replacing
``Clip + Mul`` with a single ``Min`` saves one intermediate per
replacement. +0.57 LB across 3 tasks.

Wraps the existing tool ``tools/clip_mul_to_min_v2.py``.
"""
from __future__ import annotations

import onnx

from tools.clip_mul_to_min_v2 import optimize as clip_mul_optimize
from tools.spread_patterns._base import SpreadPattern, SpreadResult


def _apply(model: onnx.ModelProto, task_id: str) -> SpreadResult:
    working = onnx.ModelProto()
    working.CopyFrom(model)
    try:
        n_replaced = clip_mul_optimize(working)
    except Exception as exc:  # noqa: BLE001
        return SpreadResult(
            pattern_name="clip_mul_to_min",
            verdict="error",
            notes=f"clip_mul_to_min raised {type(exc).__name__}: {exc}",
        )
    if n_replaced == 0:
        return SpreadResult(
            pattern_name="clip_mul_to_min",
            verdict="no-op",
            notes="no Clip[0,1] + Mul(mask) pairs detected",
        )
    return SpreadResult(
        pattern_name="clip_mul_to_min",
        verdict="applied",
        new_model=working,
        notes=f"replaced {n_replaced} Clip+Mul pair(s) with Min",
        extras={"replaced_count": n_replaced},
    )


PATTERN = SpreadPattern(
    name="clip_mul_to_min",
    category="math-equivalent",
    description="Clip[0,1] + Mul(0/1 mask) → Min(x, mask) (one fewer intermediate)",
    apply=_apply,
    safe_by_construction=False,  # mask-content assumption — verify byte-exact
)
