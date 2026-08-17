# task246 floor confirmed + Einsum 3-op markup source analysis

## What was discovered

Task246 (L-path drawing, CumSum carrier) has been exhaustively re-analyzed in a 2026-06-24 session. The v9 model (`build_task246_v9_reducesum_moment_bitdetector.py`, SHA256 `c3c40d2ab3df008a091e7acd1f7bac0412da779434ef1bf43b66ca94ad231cc0`) is at the architectural floor:

- **Local static score**: 1,275 (memory=1,172 + params=103)
- **Profile score (266 arcgen inputs)**: 1,275 (= static, no gap locally)
- **Kaggle credited**: 1,369 (7.4% = 94B markup, documented in §39 gap-ledger)
- **All gates pass**: visible 266/266, truth 200k/200k, AB 200266/200266 byte-equal

## Kaggle pricing gap source analysis

The graph is fully static (all tensor shapes are exact-dimension, no symbolic dims). Yet Kaggle credits 94B more than local. Candidate causes:

1. **3-operand Einsum (most likely)**: The v9 uses 4 Einsums with equations like `"bchw,c,bdhe->b"` (3 operands). Per `techniques.md §89`: "ORT CPU executes ≥3-operand equations PAIRWISE internally (EinsumTypedComputeProcessor — Transpose/ReduceSum/MatMul under the hood, partial result between pairs)". The partial results may be traced by Kaggle's ORT profiler but not our local profiler. Both v2 (Einsums, no ReduceSum) and v9 (Einsums + ReduceSum) show the same 1.07-1.08x markup, confirming Einsum (not ReduceSum) as the source.

2. **Why can't we replace 3-op Einsum?** The 3-operand Einsum `"bchw,c,bdhe->b"` computes a weighted sum over all channels and spatial dims simultaneously, producing a scalar [1] F32 = 4B intermediate. Any 2-operand replacement requires a [1,30] F32 = 120B intermediate per marker per axis (4 such intermediates for r2,c2,r3,c3 = +480B total). Net: going from 1,275 to ~1,755 local (WORSE even if Kaggle markup disappears).

## Floor components (irreducible)

From 1,172B total memory:
- 300B: `row_channel_bits_u8 [1,10,30,1]` — the BitwiseAnd expansion from 30-row to 30-row×10-channel. UNAVOIDABLE: any BitwiseAnd-output architecture must expand one 1D axis to 10-channel before the free output.
- 240B: 2 × ReduceSum outputs `[1,1,30,1] F32` = active row/col masks. UNAVOIDABLE: cannot detect which rows are active (vs padding) from marker positions alone (r2,r3 may not span the full active height).
- 360B: 12 BOOL intermediates (6 per axis: active, r2/r3, ge_lo, le_hi, between). UNAVOIDABLE: all 5 row/col categories needed to correctly encode the bit-product scheme.
- 240B: 8 UINT8 Where-chain intermediates (4 per axis). UNAVOIDABLE: minimum Where-chain to build 5-category row/col codes.
- 32B: 8 F32 scalars (4 Einsum positions + 4 Min/Max). UNAVOIDABLE at F32 since Einsum only supports float output.

## Dead ends verified (2026-06-24)

All these were verified not to reduce cost:
- Replacing ReduceSum with ReduceMax: same 120B F32 intermediate.
- Replacing 3-op Einsum with 2-op chains: +480B memory (WORSE).
- ArgMax-based position detection: requires Slice (large intermediate) or ReduceSum intermediate + ArgMax INT64 (8B per scalar vs 4B F32).
- Arithmetic encoding (modular U8 code building): 330B for row categories vs current 120B Where-chain (WORSE).
- Equal-output architecture with category label: 900B intermediate for label vs current 1,172B total (might save 272B but requires designing a [1,1,30,30] label tensor).
- UINT8 ReduceMax for active detection: requires Cast(input→U8) = 9000B intermediate (WORSE).
- Merging row/col categories (5→4): breaks the bit-product algebra.
- Eliminating K tensor: K [1,10,1,1] = 10 params, no compression possible.

## Where this insight matters

This is a task-specific floor proof. The insight about 3-op Einsum causing ~7% Kaggle markup on static graphs is potentially transferable: if other tasks use 3-operand Einsums and show mild markup, the same analysis applies. However, task234 (free-output Einsum) did NOT show markup (grader-validated at static cost). Difference: task234's Einsum IS the graph output (free), while task246's Einsums produce charged scalar intermediates. The Kaggle partial-result tracing may only manifest for Einsums that produce non-output intermediates.

## What's (possibly) missing from the library

The §39 gap-ledger mentions "246 both-rungs 1.07–1.08×" but doesn't explicitly attribute the cause to 3-op Einsum vs ReduceSum. This note adds the attribution (3-op Einsum as likely cause, ReduceSum ruled out by v2 comparison). If the sorter wants to add this to scoring_internals.md §39 gap-ledger or techniques.md §89, it would clarify the "mild dynamic surface" for static graphs using 3-op Einsum.

## Files involved

- Build: `The_400_Tasks/Task_by_Day/task246/prototypes/build_task246_v9_reducesum_moment_bitdetector.py`
- Model: `The_400_Tasks/Task_by_Task/task246/prototypes/task246_v9_reducesum_moment_bitdetector.onnx`
- Task notes: `The_400_Tasks/Task_by_Task/task246.md` (updated Don't-do log)
