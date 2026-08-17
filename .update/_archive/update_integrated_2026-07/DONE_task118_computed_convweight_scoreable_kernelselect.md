# Transferable insight — a COMPUTED (Where-selected) QLinearConv weight is scoreable; folds a data-selected Gather-shift into one conv

**Source**: task118 batch 2026-07-08 (`bo_ng_cld_worker3`, Claude). Shipped a byte-equal win 13,260 → **12,621** (−639) on the structural cross-cover graph. Complements [[DONE_task053_dynamic_weight_convtranspose_shift]] (input-as-weight Toeplitz) — this one is about a **Where-selected kernel** + a scorer gotcha.

## The scorer gotcha (avoid a FALSE dead-end)
`_score_v2.score_network_v2` returned `None` for a toy graph `QLinearConv(x[1,1,5,5], Where(cond,kA,kB))` — which looks like "computed conv weight is not scoreable". **That is a false negative.** The scorer's static profile feeds a standard `[1,10,30,30]` F32 dummy; a toy graph whose input is `[1,1,5,5] uint8` fails the dummy inference → `None`, regardless of the conv weight. **Both constant AND computed weights return `None` in isolation.** Test the fold inside the REAL full-I/O graph before declaring it unscoreable — there, a computed `QLinearConv` weight scores fine (task118 clamp graph: static=profile=12,621).

## The fold (the technique)
When a graph needs a **data-selected directional/radius shift or kernel** — e.g. `neighbor = choose ? right_shift(x) : down_shift(x)` — LIVE-style code materialises TWO `Gather` shifts + a `Where` blend = 3 full-grid maps + 2 index inits. Replace with ONE conv whose weight is `Where(scalar_cond, kernelA, kernelB)` (the kernels are tiny 3×3 inits; the Where output is ~9 B, free):
```
sel_kernel = Where(cond, kr_A, kr_B)              # tiny, not a charged map
neighbor   = QLinearConv(x, ..., sel_kernel, ...) # ONE 675-B map, out_scale=1, single-1 kernel => exact shift
```
Gather axis is a static attribute so you CANNOT data-select the axis of one Gather; the conv's kernel CAN encode either direction, so the kernel-select is the only single-map route. Saves ~2 maps.

## Byte-equality caveat (the trap that cost half the win)
A `Gather` shift with clamp indices `[1..W-1, W-1]` **edge-clamps** the boundary; a `QLinearConv` **zero-pads** it. They diverge only at the last row/col — invisible on interior-only official examples (task118 was 267/267 visible) but the `ab --full` corpus caught **78 lateral divergences** (spurious boundary selections). truth was a TIE (same accuracy), so it was NOT more-correct-than-LIVE (§DUAL-TRUTH) — just non-equivalent. Fix: `Pad(x, edge, bottom+right +1)` then `QLinearConv(..., pads=[1,1,0,0])` replicates the clamp exactly → byte_equal 200267/200267. The edge-Pad costs one padded map (here +728 B), so net the fold is only worth it when the two removed shift-maps + index inits exceed the pad. Always run `ab --full`, not just `visible`, on any boundary-touching op swap.

## Applies to
Any graph with a **scalar-gated directional/radius Gather-shift or dual-kernel-then-blend** pattern (tie-break neighbors, directional erosions, radius-selected stamps). Check: shipped graphs with `Gather(...,axis=2)` + `Gather(...,axis=3)` + `Where` over the same source, or two convs differing only by kernel then `Where`.
