# Free-output "growth-op charge law": only `'output'` is free, never its predecessor

**Source:** task057 SpriteHtileFloorBreakAttempt (w-057, 2026-06-22). Transferable; not task-specific.

**Lesson (a sharper statement of the free-output rule, core.md §53):**
The scorer skips EXACTLY the two tensors named literally `'input'` and `'output'`. Therefore any op that
**grows** a smaller answer piece up to the full output shape — `Pad(X)`, `Concat(A,B,…)`, `Tile(X)`,
`Expand(X)` — makes its INPUT a charged `value_info` intermediate. The "free output" trick only saves the
FINAL tensor; it never makes the thing fed INTO the growth-op free.

**Corollary for placement maps (e.g. tile/scatter to a wide canvas):**
A single `Einsum` whose placement maps reach the full output dim writes `[…,FULL]` directly with ZERO
intermediate — the maps' params ARE the price of skipping the growth-op. Trading those map-params for a
"smaller map + Pad/Concat/Tile to grow" is a strict loss whenever:
`(charged growth-predecessor bytes)  >  (param bytes saved by the smaller map)`.
At BOOL the predecessor is 1 B/elem (often still > the param save); at F32 it is 4 B/elem (catastrophic).

**task057 instance (numbers):** htile output `[1,10,30,30]`. Aditya's 4-operand Einsum uses `rowmap[3,30]` +
`colmap[3,30]` = 180 params, no intermediate. The "shrink colmap" lever (`maps [3,3]+[3,6]`=27 params + `Pad`
of a `[1,10,3,6]` `onehot6`): saves 153 params but charges onehot6 = 180 B (BOOL) ⇒ net **+27**; Concat-slab
variants charge `[1,10,30,3]` F32 = 3,600 B each. **Map-share** (`colmap = rowmap + shift3`) is impossible: the
arrays are unequal (identity vs htile), and an in-graph `Add` to derive one from the other charges a
`[3,30]` intermediate (≥180 B) > the 90-param save. **180 map-params are firm.**

**Decision rule to apply before any "free-output + grow" idea:** price the growth-op's INPUT tensor as a
charged intermediate (`numel × dtype_itemsize`, BOOL=1) and compare to the params the smaller map saves. If
input bytes ≥ param save, the single-Einsum-with-full-maps wins. This closes the task057 588 floor.
