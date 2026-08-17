"""onnxsim.simplify() pass — deeper constant folding + algebraic simplification.

History: 2026-05-12 lesson — onnxsim found additional simplifications across
107 tasks saving 140,806 bytes total → +3.84 LB. Idempotent on this
submission set after the first pass.
"""
from __future__ import annotations

import onnx

from tools.spread_patterns._base import SpreadPattern, SpreadResult


def _apply(model: onnx.ModelProto, task_id: str) -> SpreadResult:
    try:
        from onnxsim import simplify
    except ImportError:
        return SpreadResult(
            pattern_name="onnxsim",
            verdict="skipped",
            notes="onnxsim not installed (pip install onnxsim)",
        )
    try:
        new_model, ok = simplify(model)
    except Exception as exc:  # noqa: BLE001
        return SpreadResult(
            pattern_name="onnxsim",
            verdict="error",
            notes=f"onnxsim raised {type(exc).__name__}: {exc}",
        )
    if not ok:
        return SpreadResult(
            pattern_name="onnxsim",
            verdict="verify-failed",
            notes="onnxsim's internal check returned False",
        )
    # Detect no-op: serialized graph identical to input.
    if new_model.SerializeToString() == model.SerializeToString():
        return SpreadResult(
            pattern_name="onnxsim",
            verdict="no-op",
            notes="onnxsim made no changes",
        )
    return SpreadResult(
        pattern_name="onnxsim",
        verdict="applied",
        new_model=new_model,
        notes="onnxsim found additional simplifications",
    )


PATTERN = SpreadPattern(
    name="onnxsim",
    category="math-equivalent",
    description="onnxsim.simplify — deeper constant folding and algebraic simplification than onnx_opt_safe",
    apply=_apply,
    safe_by_construction=True,
)
