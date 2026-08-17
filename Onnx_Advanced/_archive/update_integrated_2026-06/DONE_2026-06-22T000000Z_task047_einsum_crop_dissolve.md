# task047 — dissolve F32 marker crops via free-input Einsum; col_bits floor confirmed; ORT Einsum is F16-only

## What I discovered

**1. F32 marker-locating sub-crops are fully dissolvable to free-input Einsum marginals (#42).**
task047's team-best (Aditya, cost 862) Slices two small F32 windows out of `input`
(`ch7 [1,1,7,6]=168B`, `ch8 [1,1,6,6]=144B` = **312B**) purely to `Cast->ReduceMax`
the marker's row/col. None of that is needed: read the 4 marker coordinates as
0-memory scalars straight off the one-hot input:
- `Einsum('bchw,c,h->b', input, sel_k, ramp30) -> r_k  [1]`  (row of marker k)
- `Einsum('bchw,c,w->b', input, sel_k, ramp30) -> c_k  [1]`  (col of marker k)
where `sel_k` is a one-hot [10] channel selector. Then rebuild the boundary-correct
code vectors with Equal-ramp masks (#67) over the 9 ACTIVE positions only
(`Equal(arange9, r_k) -> [1,1,9,1]`), Where in the codes, `Pad(const 0)` to 30.
Result: memory **826->448**, cost **862->537** (+0.473 pts), byte-equal 200265/200265,
truth 200000/200000, profile==static EXACT. Staged
`staged_submits/task047_einsum_dissolve_cost537/`. Builder
`The_400_Tasks/Task_by_Task/task047/prototypes/build_task047_einsum_dissolve.py`.
The crops are NOT load-bearing — markers are border-free so the 9-position Equal-ramp
reproduces the exact same boolean masks the crop+ReduceMax produced.

**2. The other half of Aditya's 862 win (already in #141 territory but worth pinning):**
he collapses the 10-channel column operand to ONE `BitwiseAnd(col_code_30 [1,1,1,30],
channel_mask [1,10,1,1]) -> col_bits_30 [1,10,1,30]`. That deletes the 240B of
per-channel column `Where` lanes that our old `bitset_and` (1,003) carried. The bit
scheme: pack each (row-state, col-state, channel) so the cross + crossings fall out of
`row_bits & col_bits`. channel_mask=[1,0,96,0,0,0,0,6,24,0]; row codes
other=21/orange=38/cyan=88; col codes other=11/orange=70/cyan=56.

**3. ORT Einsum has NO UINT8 kernel — F16 is the floor dtype (probe-confirmed, ORT 1.x, opset 18).**
`Einsum('ar,ac->rc')` on UINT8 -> `NOT_IMPLEMENTED`; F16 and F32 work. This KILLS the
otherwise-attractive free-output sum-product route for bitwise-intersection outputs:
I found a valid 7-bit monochromatic-rectangle scheme where
`Σ_b RB[b,r]·CB[b,c]·CM[b,ch]` gives shared-bit-count exactly 0/1 per cell (so the
Einsum output [1,10,30,30] IS the exact one-hot, FREE) — but the runtime bit-plane
operands `RB[1,7,30,1]`/`CB[1,7,1,30]` must be F16 = 420B each = **840B > the 300B
BitwiseAnd col_bits**. So for axis-bitset intersection outputs, the packed-U8
`BitwiseAnd` (#141) stays the floor; the unpacked free-output Einsum loses on the
F16 plane tax. (If ORT ever ships a U8/bool Einsum kernel, re-price this — 7-bit
planes at U8 = 210B each = 420B, still > 300B; would need <5 bits to win.)

## Where
task047. Build script + staged dir above. Decode/pricing all in
`The_400_Tasks/Task_by_Task/task047.md` (Cost analysis + Don't-do, updated 2026-06-22;
the stale "1,003 floor / NO-CANDIDATE" proof is corrected there).

## Why it matters
- General lever: any "Slice a small F32 sub-window only to ReduceMax/ArgMax a marker
  position" is a free-input Einsum marginal in disguise — dissolve the crop, keep only
  the [1] scalar. #42 already says marginals cost only their [1,N] output; this adds the
  concrete "crop-to-locate-marker" trigger and the row/col equations.
- The F16-only Einsum dtype fact belongs in `ort_reality.md` dtype×opset matrix — it
  bounds when free-output sum-product Einsum can beat a packed BitwiseAnd.

## What's missing from the library
- #42 / #141 don't explicitly call out "marker-locating F32 crop -> free-input Einsum
  marginal" as a dissolve trigger, nor the F16-only Einsum constraint that gates the
  free-output-Einsum-vs-BitwiseAnd crossover. Both are confirmed measurements here.
