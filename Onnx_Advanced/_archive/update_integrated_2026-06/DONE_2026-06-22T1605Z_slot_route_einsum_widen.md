# Slot-width routing + constant channel-widening operand in a free-output Einsum

**Task:** task010 (rank 4 vertical bars by height). 877 → 600 (-277), Path-A byte-equal vs LIVE.

## Transferable insight (#118 / #119 extension)

When a free-output separable Einsum routes a DATA-DEPENDENT one-hot to the full
`[1,10,30,30]` output's 10-channel axis, do NOT materialize the routing tensor at
full channel width. The full-width data-dependent operand is a charged F32
intermediate that scales with `terms × output_channels`.

**Pattern:** keep the data-dependent routing at the minimal SLOT width (here 5 =
bg + 4 active bars) as `route[1,T,Q]`, then widen `Q → output_channels` with a
**CONSTANT** selector basis `S[Q, K]` folded as an extra Einsum operand:

```
output = Einsum('bjrc,tj,tc,btq,qk->bkrc', input, P[T,K], bcol[T,W], route[1,T,Q], S[Q,K])
```

`S` costs only `Q*K` PARAMS (1/elem, dtype-blind) and ZERO memory — vs the
data-dependent full-width operand which costs `T*K*4` bytes of F32 memory. On
task010 this dissolved the 360 B `krout[1,5,10]`+`k_bar[1,4,10]` Concat zero-pad
into a 100 B `route[1,5,5]` + a 50-param `S[5,10]` (free memory). The widening
matrix is exactly the #118 basis-operand idiom but applied to the
data-dependent SLOT→CHANNEL remap instead of a spatial up/place remap.

## Two floor traps confirmed on this task

1. **Don't split a constant term out to shrink the route width.** Splitting the bg
   term into its own additive Einsum forces an `Add` of two charged
   `[1,10,30,30]` planes (~36 KB). Keep all terms (incl. constant bg) as slots of
   the single route operand. The route's slot axis = number of distinct
   placement/color groups, NOT reducible by peeling constants.
2. **`ReduceSum` has no U8/BOOL kernel** — the pairwise-Greater rank-count path
   (#124) must `Cast` BOOL→F32 before `ReduceSum`, so `taller_f32[1,N,N]` is an
   F32-locked floor item. (Consistent with traps §8f / ort_reality reduce rows.)

## Build the route one-hot WITHOUT a Concat zero-pad

`route = Cast(Equal(slot_idx[1,T,1], arange_Q[1,1,Q]))` builds the `[1,T,Q]`
one-hot directly from a slot-index vector — no per-channel zero-pad Concat
intermediate (the 877 graph paid `k_zero_lead`/`k_zero_tail` Concats for this).
Prepend the constant bg slot via a single `Concat(slot0[1,1,1], ranks[1,N,1])`.

Net: any free-output separable Einsum whose data-dependent routing is materialized
at full output-channel width is leaving the #118 constant-widening trade on the
table.
