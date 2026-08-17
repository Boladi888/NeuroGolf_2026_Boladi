# DEAD ROUTE — int8-mask + ConvInteger free-output render (task072)

- **live bar**: 243 (manifest, sha `5b9ec34339f5766910957f8a2dc7c3cc5235e2fc4cae94477d2e3027adf29765`), owner Vamshi, 2026-07-02.
- **date attempted**: 2026-07-03 (batch 48, bo_ng_cld_worker1).

## Route
Only lever that could beat the 242 floor: shrink the render **input** tensor from F16 (60 B) to
int8 (30 B) by feeding an int8 signed mask to `ConvInteger` (int32 output, which is the free graph
`output`). Projected: memory 180 (z 120 + k 30 + s_i8 30) + params 32 = **212**.

## Why it's dead (two independent blockers)
1. **ORT has no int8 `Where` kernel.** `Where(k, neg_one_i8, one_i8)` →
   `NotImplemented: Could not find an implementation for Where(9) node`. So the {−1,+1} int8 mask
   cannot be produced by `Where`. Producing it in F16 then `Cast`→int8 adds a 30 B tensor on top of
   the 60 B F16 (net worse than 210). Int8 arithmetic (`2·m−1`) needs opset-14 `Mul`/`Sub` plus two
   extra int8 tensors → also net worse.
2. **ConvInteger cannot render two colors from a scalar mask.** It has no bias, and its x/w zero-points
   give `out = (x−xzp)(w−wzp)`. A {0,1} mask needs channel0=1 at bg (x=0) AND channel3=1 at xor (x=1)
   simultaneously; the two required values differ by 1, forcing a non-integer weight (e.g. `2·w=1`).
   No integer (xzp, w) solves both. A ±1 mask fixes this but requires blocker #1.

`score_v2` also returns `None` on the ConvInteger+int8-Where graph (session init fails on the Where).

## One remaining hypothesis (untested, low promise)
If a future ORT build ships an int8 `Where` kernel, revisit: `Cast(z,BOOL)→Where(k, -1_i8, +1_i8)→
ConvInteger(pads-expand)` would price at ~212 and (by the LIVE decode-tolerance evidence) should gate
byte-clean. Until then, the F16 render input is forced → memory floor 210, cost floor **242**.
