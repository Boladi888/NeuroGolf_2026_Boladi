"""FP16 conversion (gated by cost-delta sanity check).

History: 2026-05-12 lessons (two flavors):
  * ``fp16_convert.py convert_clean`` on submission.zip → +1.34 LB across
    7 simple tasks (binary-ish intermediates).
  * Naive ``onnxconverter_common.float16.convert_float_to_float16`` on
    high-cost graphs INCREASED cost by 100k–900k per task because the
    added FP32↔FP16 Casts at every I/O boundary outweigh the savings.

Resolution: only keep the FP16 version when it ACTUALLY reduces
intermediate memory. The driver verifies via score_network_v2.
"""
from __future__ import annotations

import onnx

from tools.fp16_convert import convert_clean
from tools.spread_patterns._base import SpreadPattern, SpreadResult


def _apply(model: onnx.ModelProto, task_id: str) -> SpreadResult:
    try:
        new_model = convert_clean(model)
    except Exception as exc:  # noqa: BLE001
        return SpreadResult(
            pattern_name="fp16_selective",
            verdict="error",
            notes=f"fp16_convert raised {type(exc).__name__}: {exc}",
        )
    if new_model.SerializeToString() == model.SerializeToString():
        return SpreadResult(
            pattern_name="fp16_selective",
            verdict="no-op",
            notes="fp16 conversion produced an identical model",
        )
    return SpreadResult(
        pattern_name="fp16_selective",
        verdict="applied",
        new_model=new_model,
        notes=(
            "fp16-converted; driver must verify cost actually drops and"
            " output is within tolerance (onnxconverter_common often adds"
            " Casts that increase cost — keep ONLY if cost_delta < 0)"
        ),
    )


PATTERN = SpreadPattern(
    name="fp16_selective",
    category="selective",
    description="FP16 conversion via onnxconverter_common (kept only when cost actually drops)",
    apply=_apply,
    safe_by_construction=False,  # numerical tolerance required, not byte-exact
)
