# Transferable insight — free-output GATHER-index floor; when "free-output Equal/Pad" is dominated

**Source**: task197 batch56 2026-07-09 (`bo_ng_cld_worker1`, Claude). LIVE **277** = column-copy
`Gather(input, idx30 INT32 [30], axis=3)` emitting `[1,10,30,30]` directly (free output), no spatial
plane materialized. Verdict: **WALL at 277** — an independent from-scratch rebuild reproduced 277
exactly (visible 267/267, profile 277); the prompt's `FreeOutputPadTailSweep` direction measured DEAD.
`floor_est ~110` was below the single forced 120 B index ⇒ unreachable. Applies to any
data-dependent column/row permutation-with-duplication (copy/remap) family.

**The discriminator: a free final node only helps if its INPUT is cheaper than the rival's charged
intermediate.** For a column-copy `output[:,:,r,c] = input[:,:,r,f(c)]`, three "free-output" tails
compete, and you pay for whatever feeds the free node:

| Free final op | What feeds it (CHARGED) | Bytes | Verdict |
|---|---|---:|---|
| `Gather(input, idx, axis=3)` → `[1,10,30,30]` | `idx [30]` INT32 (indices must be I32/I64) | **120** | WINNER |
| `Equal(plane, chan_iota)` → one-hot | `plane [1,1,30,30]` (one label/cell, U8) | ≥900 | dominated |
| `Pad(gathered, tail)` → `[1,10,30,30]` | `gathered [1,10,30,W]` F32 (W-wide Gather) | 12000 | dominated |

**Tool 1 — The full-width Gather index is a hard 120 B floor; don't try to shrink it.**
To make the `[1,10,30,30]` output FREE, the Gather index must be `[30]` (one source col per output
col) and INT32/INT64 (Gather rejects U8/INT16). `[30]` INT32 = 120 B, irreducible. Shrinking to a
`[10]` index does NOT help: `Gather(input,[10])` → `[1,10,30,10]` F32 is a **charged 12000 B**
intermediate *before* the "free" Pad (measured 12175). The final Pad is only free when its input is
already `[1,10,30,30]` — which needs the `[30]` index. So `floor_est < 120` for this family is a
mirage (it assumes the width-30 expansion is free after a narrow Gather; it is not).

**Tool 2 — "Free-output Equal on a label plane" is a TRAP when a 1-D index already suffices.**
Emitting the one-hot as `Equal(labelplane [1,1,30,30], chan_iota [1,10,1,1])` looks clever (output is
free, "just one channel of labels"), but the labelplane it consumes is a **full-resolution spatial
plane** (`H·W` values). Building it means Conv-reading the input to a `[1,1,30,30]` F32 (3600 B) and
column-copying THAT (another 3600 B) — i.e. it *adds* a spatial plane on top of still needing the
same selection logic. Measured **7488**. Cropping to `[1,1,14,10]` doesn't save you: you still need a
`[1,1,30,30]` (≥900 B U8) pre-Equal OR a `[1,10,14,10]` (≥1400 B) pre-Pad, both already > a 120 B
index. **Free-output Equal only wins when the rival would materialize an even bigger charged one-hot
(e.g. a rank-forced Einsum canvas) — never when a 1-D Gather index encodes the whole map.**

**Tool 3 — The cheap-cast staging rule for an INT-index built from narrow logic.**
Build all data-dependent selectors in U8 (10 B/vec), Pad the `[W]`→`[30]` in U8 (30 B), then a SINGLE
Cast to `[30]` INT32 (120 B). Cast-before-pad (`[10]` I32 = 40 + `[30]` I32 = 120) is +10 B worse.
The col-0 reference colour is free via a **second Conv with crop pads** (no `col0_idx` init) — this is
the 278→277 param trim (Sho). `ArgMin(BOOL)` is illegal (needs the U8 cast); `idx_pads` can't move to
Pad-10 attributes because opset-10 `Equal(float/uint8)` is illegal (whole graph can't drop opsets).

**Recipe to close a copy/remap family as WALL:**
1. `largest_intermediate` the LIVE onnx — find the single dominant charged tensor (here `idx30`=120 B, 45.6%).
2. Ask: is it the index of a free-output Gather? If yes, its size = `output_axis_len × 4` is a floor.
3. Enumerate the free-output rivals (Equal-plane, Pad-after-narrow-Gather) and price *their* feeding
   intermediate — build one and score it; they charge the spatial plane you were trying to avoid.
4. Independently rebuild the winner from scratch; a TIE at the same cost = reproducible floor proof.
5. `hard_records/<route>.md` with the forced-component arithmetic + the rival cost table.

**Applies to**: any single-Gather / column-copy / position-remap task (cf. task262 col-position remap)
where the LIVE already emits `[1,C,H,W]` free via Gather and someone proposes an Equal/Pad free-output
"sweep" — the sweep is dominated; prove it in one build and move on.
