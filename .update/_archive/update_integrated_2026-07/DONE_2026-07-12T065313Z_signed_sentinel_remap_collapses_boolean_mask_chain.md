# Transferable insight — a "signed-sentinel remap" in an upstream linear/value map collapses a multi-plane boolean mask chain into ONE sign comparison (−1250 B byte-equal, task286)

**Source**: task286 (`bo_ng_cld_worker3`, 2026-07-12). Audited LIVE **18672** (Sho,
sha `20036e365eb9…`, 2026-07-10) — an Einsum-transfer-matrix maze flood, NOT a bit-pack.
This is a **new byte-equal win**: staged **17422** (−1250), LOCAL_GOLD, ab 2265/2265 byte-equal,
truth 2000/2000, profile==static. Cousin of the general "fold a class into a cheaper upstream
encoding" family, but the specific lever is dtype-sign, not op-fusion.

## The pattern it kills
A graph derives a mask as **"A AND NOT B"** (or a chain of them) by materialising each predicate as
its own charged BOOL plane, then XOR/AND-combining. task286 computed the flood's `open` mask as
`active XOR wall` and `seed` as `open XOR bg`, using **5** charged BOOL `[25,25]` planes:
`active=Greater(v,0)`, `wall=Equal(v,8)`, `bg=Equal(v,10)`, `open=active^wall`, `seed=open^bg`.
Every `[25,25]` BOOL plane is 625 charged bytes (`calculate_memory` charges each non-input/output
tensor; BOOL = 1 B/elem).

## The lever
The value grid `v` comes from an upstream **linear map** (here Einsum `bphw,p,...` with a color→value
weight `v_w1[10]`). The "excluded" sentinel class (wall) had value **8 > 0**, so `active=Greater(v,0)`
had to *include* wall and a separate `Equal(v,8)` + XOR was needed to remove it. **Remap the sentinel
to a NEGATIVE value** (`v_w1[8]: 8 → −1`). Now the excluded class sits below 0 *alongside* the other
naturally-excluded class (out-of-grid cells = 0, since a linear map of an all-zero one-hot column = 0).
A single `Greater(v, 0)` becomes the `open` mask directly — it excludes BOTH out-of-grid(0) AND
wall(−1) at once. Masks collapse **5 → 3** planes (`open`, `bg`, `seed`), **−2 × 625 = −1250 B**.
Op deltas: `Equal 3→2`, `Xor 2→1`. Byte-equal, params delta +0.

## The three gotchas (all required for byte-equality)
1. **Compare on the SIGNED tensor, not the U8 cast.** The graph also casts `v` to U8 for a cheap
   integer tail (`v_t1`). On U8, wall −1 wraps to **255 > 0**, so `Greater(u8,0)` would re-include it.
   Move the mask comparisons to the **F32** value grid (`v_t0`) where −1 is seen signed. Cost-neutral
   (BOOL out either way); just rewire the Greater/Equal inputs.
2. **Keep the sentinel decodable.** The unreached-cell fill preserves the original value, and the final
   `Equal(v_grid, decode_map)` one-hots it back. Update the decode entry for the remapped class to the
   sentinel's U8 image: `v_w8[wall_channel]: 8 → 255` (U8 of −1). Walls then still decode to their
   channel. Out-of-grid stays 0 (matches no decode entry → all-false, which was already the expected
   output for beyond-grid cells).
3. **The other excluded class must already be structurally 0.** This works because out-of-grid cells
   are the linear map of an all-zero one-hot → exactly 0, so `Greater(v,0)` already excluded them; the
   remap only had to push the *second* excluded class to the same side. If your two excluded classes
   aren't naturally co-signed, one remap still merges them iff you can put both < threshold.

## When to reach for it (checklist)
- The mask feeds an op needing a specific dtype anyway (here Einsum → F16 cast), so the BOOL plane is
  a genuine extra intermediate, not free.
- The class values originate in an editable upstream **linear/value/LUT map** (Conv/Einsum/MatMul/Gather
  weight) you can re-number without changing other consumers (or you fix the other consumers, as with
  the decode map here).
- You're computing `active AND NOT sentinel` where `active = sign/threshold test`. Push the sentinel
  across the threshold → the AND-NOT disappears.
- **Verify byte-equal** (`ng_gate ab --n 2000` + in-proc compare on all real inputs) — this is a
  relabeling, so a correct application is exactly output-preserving; any divergence means a consumer of
  the remapped value (fill, decode, a second comparison) wasn't updated.

## What is NOT reducible here (so the next batch doesn't re-grind — see task286/hard_records/)
The flood core is at floor: 3 chained separable-dilation Einsums (75 steps) forced — measured corpus
max 8-conn convergence depth **59** (2 einsums=50 steps fail *locally*), einsum alphabet caps 25
steps/axis, radius-1 dilation forced (radius-2 jumps walls), Min clamp load-bearing (removing it →
inf-into-3rd-einsum → nan, 262/265 fail). Entry `v_t0` F32 `[25,25]` 2500 B forced by the F32 one-hot
input. Remaining slack, if any, is only in the ~5.6 KB BOOL/U8 mask+paint planes.
