# Multicolor/range window test: `sum != max*count` beats the complement-MaxPool pair (task158 v130, −1,950 B)

**Claim.** To test "≥2 distinct non-zero values in a KxK window" of a small-value U8 plane `Z` (values ≤ 9),
the standard complement trick (`min = K_c − MaxPool(K_c − Z)` gated to non-zero, i.e. planes
`gtZ, s11, Z2, Z2M=MaxPool, aZ=Add, gW=Greater` = 6 full planes) is NOT tensor-minimal.
Byte-identical replacement with only 3 planes:

```
SUMc  = QLinearConv(Z, ones_KxK)      # window sum of values      (<= 81 for 3x3, v<=9: no overflow)
MulZN = Mul(WindowMax, WindowCount)   # max*count, Mul(U8,U8) — LEGAL at opset >= 14 (ort_reality §41)
eqs   = Equal(SUMc, MulZN)            # true  <=>  all non-zero values equal the max  <=>  NOT multicolor
```
Identity: `sum(v_i) == max*count ⟺ every v_i == max ⟺ <2 distinct values`; empty window gives `0==0`
(not multicolor) — matching the complement-pair semantics exactly. `WindowMax = MaxPool(Z)` and
`WindowCount = QLinearConv(Min(Z,1), ones)` are usually already present (locator score planes).
Check the overflow budget first: need `max_value * window_area <= 255`.

**Where it landed.** task158 v130: replaced Sho's `gtZ/s11/Z2/Z2M/aZ/gW` with `SUMc/MulZN/eqs`
(−1,950 B, byte-equal AB 10,266 pairs div=0), which funded the forced +2,601 `Min(CP,76)` OOB-Gather
crash clamp and still landed −14 below the live bar (23,187 vs 23,201). Staged
`The_400_Tasks/Task_by_Task/task158/staged_candidates/task158_v130_sumtest_clamp_cost23187_sha84BF2DB2/`.

**Trap fixed in passing.** A stale "Mul(U8,U8) ORT-banned" note (opset-13-era) in a 2026-07-09 floor doc
blocked this rewrite on paper; the graph was opset 17 where U8 `Mul` is legal. When a floor proof cites
an op ban, re-check the ban against the LIVE graph's actual opset before accepting the floor.

**Also transferable.** Micro-folds that broke the tie vs the bar: `Concat(scalars, axis=1)` directly
produces `[1,2,1,1]` (delete the Reshape + its 4-param shape init); scalar (rank-0) index initializers
for corner Gathers delete downstream Reshapes via broadcasting — but re-check any `Reduce*` axes
attributes on tensors whose rank drops (task158 needed ReduceL1 axes [3]→[0]).
