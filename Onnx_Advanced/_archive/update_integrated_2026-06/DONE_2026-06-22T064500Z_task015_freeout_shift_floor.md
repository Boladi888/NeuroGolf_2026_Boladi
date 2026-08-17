# DONE: free-output Einsum does NOT beat dense Conv[10,10,3,3]=900 for LOCAL-SHIFT stamp rules (task015)

**Topic:** cost-floor reasoning — when a `+-1` spatial-neighbour (rook/bishop) stamp rule keeps the dense `Conv[10,10,3,3]=900` as the genuine floor, and exactly why the #109/#119 free-output separable Einsum (the named lever in the prompt) LOSES here. Direct companion to `DONE_2026-06-22T062544Z_dense_conv_floor_vs_crop_onehot.md` (task095): same dense-Conv-floor family, but the failing alternative is the shift-basis Einsum rather than the crop+one-hot.

**Context:** task015 = generator `task_0ca9ddb6` (per-color twinkle stamp). Fixed 9×9. color-1 pixel → color-7 rook cross (4 axis neighbours); color-2 pixel → color-4 bishop X (4 diagonal neighbours); colors 6/8 copied; bg=0. Generator guarantees: twinklers scooched to rows/cols in [1,7] (stamps never go off-grid), and a pre-drop removes any sampled pixel whose stamp would clobber a kept pixel (so stamps only ever overwrite background ch0). Team-best/LIVE = single dense `Conv[10,10,3,3]` = 900 params, memory 0 (re-measured `(0,0,900)`).

## The dissolve IS correct (functionally) — it just costs more

`output = copy(input) + rookShift(ch1)->ch7 + bishopShift(ch2)->ch4 - (those cells from ch0)`.
With `N[30,30]` the `|i-j|==1` axis-neighbour band matrix:
- `rookShift(P)   = N@P + P@N`   (two separable terms: row-shift + col-shift)
- `bishopShift(P) = N@P@N`       (one separable term: diagonal shift, N reused both axes)

Built the WHOLE `[1,10,30,30]` output as a SINGLE free-output Einsum (named `output` ⇒ never charged), input as a free F32 operand:
```
equation = 'bkrc,ts,sro,tu,ucq,tkj->bjoq'
operands = input, Rpick[4,2], B[2,30,30], Cpick[4,2], B[2,30,30], Ksel[4,10,10]
# B = stack([I30, N]); term t in {copy,(N,I),(I,N),(N,N)}; Ksel routes ch1->ch7/-ch0, ch2->ch4/-ch0
```
Gates (clean): **visible 265/265 (100%)**, **truth full 200000/200000 (100.000%)**, **profile==static==2216** (free-output Einsum reprices EXACTLY — #109 free-contraction-temp law holds, no dynamic tax). The math is exact (3000/3000 fresh generator pairs + ORT 3000/3000). Builder: `The_400_Tasks/Task_by_Task/task015/prototypes/build_task015_freeout_einsum.py`.

## Why it costs 2216 > 900 (the floor proof — reusable for any LOCAL-SHIFT rule)

1. **A `+-1` shift on a 30-wide canvas is irreducibly a 900-element object.** The band/shift matrix `N[30,30]` has rank ~9 over the active 9×9 region (measured `matrix_rank=8` on the restricted band; full band rank 30). It is NOT low-rank and NOT digit-factorable, so NO `[3,30]`-style placement basis (the #118 trick that worked for task195's separable kron) can represent it. `params` counts ELEMENT count, not nonzero — a banded `[30,30]` is always 900. This is the standing kill-table verdict ("local neighbor access is not separable row/col masking") — CONFIRMED by build, not just asserted.

2. **Keeping `output` as a single free Einsum forces the identity-COPY term inside it**, which needs an `I[30,30]` operand ALONGSIDE `N[30,30]` ⇒ the spatial basis stack is `B[2,30,30] = 1800` before any channel routing. Pull the copy OUT of the einsum (`Add(input, delta)`) and `delta` becomes a charged `[1,10,30,30]` = 36000 B intermediate. Either way you lose: 1800 (in-einsum copy) or 36000 (out-of-einsum copy) vs the Conv's 900.

3. Total measured = B(1800) + Rpick(8) + Cpick(8) + Ksel(400) = **2216**.

**General rule (add to the dense-Conv-floor doctrine / kill-table for shift-stamp tasks):** When a rule needs `+-1` (or any small fixed) SPATIAL neighbour access AND a 10-channel one-hot output AND cross-channel recolor, the dense `Conv[10,10,3,3]=900` is the floor. The free-output separable Einsum (#109/#119) is the right tool ONLY when the output channels are LOW-RANK outer products of row/col indicators (solid rects, separable masks, kron self-products). A SHIFT is a near-full-rank band matrix, so the Einsum pays ≥900 PER shift basis plus a second ≥900 for the copy term — it structurally cannot beat the Conv. Discriminator: rank-test the per-axis placement/shift matrix over the active region FIRST; rank>1 (here ~9) ⇒ Conv floor, do not build the Einsum expecting a win.

## What's already in the library / what's new
- #118/#119 already say "basis matrices ARE the einsum factors" and #119's "conditional shift guardrail (task382)" already says `S[k,W,W]` dense shift basis is `O(W^2)` and a `[K,W]` Gather bank usually wins. The NEW, task015-specific datapoint: for a FIXED (non-data-dependent) `+-1` stamp where `output` must stay the single free Einsum, even the Gather-bank guardrail doesn't apply (shift is fixed, not per-row variable) and the copy-term forces a SECOND full basis ⇒ 2216, a clean +1316 LOSS vs 900. This nails the kill-table's "killed by lower bound" row with a measured number and the rank argument.
- task015.md kill-table row "Free-output Einsum with spatial-shift bases ... killed by lower bound" can be upgraded from a-priori to MEASURED: built, gated 100%/200k clean, priced 2216.
