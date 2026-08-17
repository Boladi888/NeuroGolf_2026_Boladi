# DEAD ROUTE — free-input / sparse-marginal read of the ch3 plane (task042)

- **Bar when explored:** live 2,522 → beaten to **2,322** this batch (v3 select-tail); this record is about the *read* half only.
- **Live sha (2,522 base):** c7d8b30fc976e4256083661e0d034d636031faa92dddf53e617534e51603ae25
- **Prompt direction tested:** "Use free-input Einsum/GatherND marginals for sparse reads … do not use a materialized full-channel slice for a handful of cells." Goal: kill the `x3_f32` BOOL/F32 400 B read.

## Why the read is FORCED (not a formulation artifact)

The detection is a genuine **2-D diagonal self-correlation** (green-plane ANDed with itself at fixed diagonal offsets), rank>1 (#110). It needs the WHOLE 10×10 green plane, not "a handful of cells":
1. A QLinearConv is the plane-minimal realization (W4 Plan 1: boolean shift-AND is ~10× the planes; separable/free-output Einsum #109/#119 does NOT fold a rank>1 self-correlation — no-gate all-distance OR fails 196/266).
2. QLinearConv needs a materialized **u8 [1,1,10,10]** feed. The only path from the F32 input is `Slice → x3_f32 F32[1,1,10,10]=400 → Cast → x3_u8=100` (Slice preserves dtype; `Cast(input)` first is [1,10,30,30]=9,000).
3. The *count* CAN be read cheaply without x3_f32 (`ReduceSum(input, axes=[2,3]) → [1,10,1,1]=40 → Gather ch3`), but that does NOT drop x3_f32 — it's still needed for the Cast→u8. Moving the count off x3_f32 only ADDS a 40 B plane. Strictly worse.

**Marginals (row/col projections) lose the joint (r,c) pairing** that decides which cell to paint; 1-D reads cannot reconstruct a rank>1 2-D coincidence.

## Verdict
`x3_f32` 400 + `x3_u8` 100 = **500 B read is forced** for any QConv-detector formulation. The "sparse read" premise does not apply (the read is a full plane, not a few cells).

## One remaining hypothesis (unexplored, low confidence)
A fundamentally different detector that never materializes the u8 plane — e.g. a float `Conv` directly on `x3_f32` — but its F32 outputs are 4× the u8 conv outputs ([1,K,10,10] F32 = 4×100), net worse on the detection block. Only worth revisiting if a detector is found whose output is a single small F32 scalar/vector, which the 2-D coincidence rules out.
