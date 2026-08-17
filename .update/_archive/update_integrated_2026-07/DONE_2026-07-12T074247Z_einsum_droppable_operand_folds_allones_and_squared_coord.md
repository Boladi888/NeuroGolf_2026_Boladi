# Transferable insight — an all-ones or squared-coordinate weight fed to an Einsum is a DROPPABLE OPERAND: delete the initializer and let the Einsum sum/re-square implicitly (param-free, memory-free, byte-exact)

**Source**: task387 (`bo_ng_cld_worker3`, 2026-07-12, forced 2nd pass). Live **2607** (Vamshi, sha `f5ceadec…`) → prior local best 2606 → **new staged best 2566** (`task387_drop_coord2_allc_cost2566`, sha `6a3f8962…`, byte-equal full battery). **−41 vs live, param-only, memory unchanged.** These two folds are NOT the P-embed lever (that stays walled at 540) — they are generic **moment/reduction front-end** cleanups that the batch-42 per-tensor audit and `polish_candidate.py` both missed.

## The two folds (both proven on task387's F32 moment front-end)

**1. All-ones weight operand → DROP it (Einsum sums that axis implicitly).**
An Einsum operand that is an all-ones vector over a contracted/summed index does nothing except name the axis — and Einsum already sums any input index that is absent from the output and from every other operand. So delete the operand and its subscript:
```
area_f        Einsum(input, allc_f)          'bcrs,c->'    ->  Einsum(input)          'bcrs->'      # -10 params (allc_f[10])
active_rsum_f Einsum(input, allc_f, coord_f) 'bcrs,c,r->'  ->  Einsum(input, coord_f) 'bcrs,r->'
active_csum_f Einsum(input, allc_f, coord_f) 'bcrs,c,s->'  ->  Einsum(input, coord_f) 'bcrs,s->'
```
Single-operand full-reduction `'bcrs->'` runs fine in ORT. Guard: this is ONLY valid for a genuinely all-ONES weight. A weight like `nonbg_f=[0,1,1,…,1]` (a channel-0 exclusion mask) is load-bearing and must stay — it is NOT all-ones.

**2. Squared-coordinate init → DROP it, contract the base coordinate TWICE on the same index.**
A precomputed `coord2_f = coord_f**2` used only to form a second moment `Σ x·r²` is redundant if `coord_f` already exists — repeat `coord_f` as an extra operand on the SAME index:
```
row_s2 Einsum(input, nonbg_f, coord2_f) 'bcrs,c,r->'  ->  Einsum(input, nonbg_f, coord_f, coord_f) 'bcrs,c,r,r->'   # -30 params (coord2_f[30])
col_s2 Einsum(input, nonbg_f, coord2_f) 'bcrs,c,s->'  ->  Einsum(input, nonbg_f, coord_f, coord_f) 'bcrs,c,s,s->'
```
An index appearing in 3 operands (input's `r`, `coord_f`, `coord_f`) is legal and contracts to `Σ x·r²`. (ORT already accepts this pattern — task387's `cross` node uses `coord_f` twice: `'bcrs,c,r,s->c'`.) Generalizes: a cube `r³` = coord_f three times, a mixed `r·s` = coord_f as `r` and `s`, etc.

## Why it is free AND byte-exact
- **Param-free + memory-free**: both folds only DELETE initializers and add already-existing operands. No new node, no new intermediate tensor → memory is byte-identical (task387 stayed 1906). `params` counts initializer elements, so −10 and −30 land directly on cost.
- **Byte-exact** (passes `ab --n 2000` byte_equal=True AND is provable): the summands are identical values; the results are sums of small integers (`area ≤ 900`, `active_rsum ≤ ~13000`, `r² ≤ 841`) that are exact in F32 regardless of reduction order, so no ULP drift. Do NOT assume byte-equality for a non-integer / large-magnitude reduction where operand reordering could change FP rounding — gate `ab` and reason about the magnitude first.

## Rule of thumb (audit checklist for ANY Einsum-heavy front-end)
Before declaring a params floor, scan every `Einsum` operand that is an INITIALIZER:
1. **Is it all-ones over a summed/contracted axis?** → delete the operand, drop its subscript. (−elements, 0 memory.)
2. **Is it `f(existing_init)` — a square/cube/product of a coordinate you already have?** → delete it, repeat the base init on the needed index/indices. (−elements, 0 memory.)
3. Otherwise it is load-bearing (a real mask/weight/palette) — keep it.
This is orthogonal to the render/embed floor and to `polish_candidate.py` (algebraic passes do NOT spot redundant-operand or squared-init folds). It surfaced a −40 win on a graph two prior workers + a WALL record had declared tight.

## Not-the-wall note
task387's dominant cost is still the `P[18,30]` identity-embed = **540 params** (now 82% of the 660-param total), which is a proven wall: sparse-P dies on strict shape-inference (rank-0 Einsum crash), no-P/zero-block/Pad-after all lose on memory (the size-30 output axis must ride a 540-element operand or ≥720 memory-bytes), rank<6 forced, and `Einsum(INT8)` has no ORT kernel so the render can't narrow below F16. See `Task_by_Task/task387/hard_records/freeoutput_einsum_render_2607_wall.md`.
