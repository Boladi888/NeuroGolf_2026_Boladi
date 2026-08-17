"""onnx_opt_safe (dedupe + CSE + DCE + algebraic identities) + extended.

History: the math-equivalent pass pipeline shipped pre-2026-05-12. Combined
with const_to_init and onnxsim, the whole pipeline delivered +12.49 LB.
This module re-runs both safe-opt and extended passes (idempotent at
worst), since new candidates may not have had them applied.
"""
from __future__ import annotations

import onnx

from tools.onnx_opt_extended import optimize as extended_optimize
from tools.onnx_opt_safe import optimize as safe_optimize
from tools.spread_patterns._base import SpreadPattern, SpreadResult


def _apply(model: onnx.ModelProto, task_id: str) -> SpreadResult:
    working = onnx.ModelProto()
    working.CopyFrom(model)
    try:
        safe_stats = safe_optimize(working)
        ext_stats = extended_optimize(working)
    except Exception as exc:  # noqa: BLE001
        return SpreadResult(
            pattern_name="safe_opt",
            verdict="error",
            notes=f"safe_opt raised {type(exc).__name__}: {exc}",
        )
    safe_changes = sum(safe_stats.values()) if isinstance(safe_stats, dict) else 0
    ext_changes = sum(ext_stats.values()) if isinstance(ext_stats, dict) else 0
    total = safe_changes + ext_changes
    if total == 0:
        return SpreadResult(
            pattern_name="safe_opt",
            verdict="no-op",
            notes="no changes from safe-opt + extended passes",
        )
    summary_parts = []
    if isinstance(safe_stats, dict):
        nonzero = {k: v for k, v in safe_stats.items() if v}
        if nonzero:
            summary_parts.append(f"safe={nonzero}")
    if isinstance(ext_stats, dict):
        nonzero = {k: v for k, v in ext_stats.items() if v}
        if nonzero:
            summary_parts.append(f"extended={nonzero}")
    return SpreadResult(
        pattern_name="safe_opt",
        verdict="applied",
        new_model=working,
        notes="; ".join(summary_parts) or "applied passes",
        extras={"safe_stats": dict(safe_stats), "extended_stats": dict(ext_stats)},
    )


PATTERN = SpreadPattern(
    name="safe_opt",
    category="math-equivalent",
    description="onnx_opt_safe (dedupe + CSE + DCE + algebraic) + onnx_opt_extended (elim_* passes)",
    apply=_apply,
    safe_by_construction=True,
)
