# Packed dual-marginal residue does NOT close — additive band contamination

**Where:** task253 (L-tromino orientation → fixed-corner template), w-253 session 2026-06-22.
Team-best 525 (v8); local staged primary is v9 = **507** (`staged_submits/task253_scalar_targets_cost507/`),
which already beats team-best by 18 B (param-only, scalar residue targets).

## What I discovered (a reusable dead-end for the #86 "positional-code pack" idea)

The standard #42 weighted-marginal orientation detector needs TWO independent scalar
marginals per channel: `wr = Einsum(input, iota_row)` and `wc = Einsum(input, iota_col)`,
each `[1,C,1]` F32 (40 B), then `mr=Mod(wr,3)`, `mc=Mod(wc,3)`. That is 160 B of forced
F32 detection (the v9 floor).

The tempting #86-style lever: pack BOTH positional codes into ONE weighted sum
(`packed = wr + K·wc`) so a single `[1,C,1]` tensor + a divmod tail decodes both
residue bits, deleting `wc`+`mc` (~80 B). **It prices out for three independent reasons:**

1. **No separable single Einsum.** The only einsum that mixes the `h` and `w` axes into
   one output is `bchw,hwy->bcy` with weight `W[h,w] = h + K·w`. That is **additive, not
   rank-1 separable**, so it cannot reuse a `[30,1]` iota — it needs the full dense
   `[30,30,1]` weight = **900 params**. That alone loses the 80 B by 820.

2. **No single-Mod demux even granting the weight.** Brute search `K∈{3,9,30,90} ×
   M∈{3,9,27,90}` finds NO `Mod(packed, M)` that maps all 4 orientations to 4 distinct
   single values. Reason: the tromino POSITION is free (top-left of the bbox ranges over
   the grid), so `wr` ranges widely; the additive high band of `K·wc` is contaminated by
   `wr`'s carry — `packed // K = wc + wr // K`, not `wc % 3`. Decoding needs the two
   residues kept in *disjoint* bands, which additive packing of a position-variable sum
   cannot guarantee.

3. **Add-fold saves nothing.** Computing `packed = wr + K·wc` from the two existing
   einsums still materializes BOTH einsums plus an Add tensor → net ≥ 0.

## Why it matters

This is the general counter to "pack two marginals into one to halve detection cost":
**positional-code packing (#86) only works when each code lives in a fixed, bounded,
disjoint numeric band.** When the underlying quantity has a variable offset (free object
position), the low code's carry overflows into the high code's band and a single Mod
can't demux. Two genuinely independent 1-D reductions stay two tensors. This is the
explicit "wall" condition the task253 prompt asked to test — confirmed priced out.

## Library gap

`techniques.md #86` (positional-code packing / max-bound existence) should carry the
**disjoint-band precondition**: packing N codes into one weighted sum + single-Mod decode
requires `code_i ∈ [0, B_i)` with the pack base ≥ Π B_j for j<i; if any code has a
variable additive offset (free position), the carry breaks the demux and the pack costs a
dense `[L,L]` mixing weight instead of a reused `[L,1]` iota. Cross-link to #42
(free-input weighted marginals stay independent) and the task253 floor proof.
