# task117 hard record — paint-over-input floor ~3795 (prompt `floor_est ~1300` is UNREACHABLE)

- **Live bar when recorded:** 3890 (authoritative manifest, `ng_current_bar.py --task 117`, owner Boladi, sha `c42208e175de23dc7ebe9aa9d2fb055e18f7f905f2214e66160f0ad8be16b88c`).
- **New floor rung:** 3795, sha `ca5d1578592f658a6678a88ea38f4086d923ab0944d372375c7c5b19336bdb3e` (`staged_candidates/task117_lean_color_cost3795/`), byte-equal to the 3890 best.
- **Recorded:** 2026-07-08 by `bo_ng_cld_worker2` (batch54).

## Verdict: WALL. The ~1300 `floor_est` is a `3890/e` heuristic, not a structural floor.

The rule: paint the OTHER of the two foreground colors at the 4-fold reflection of its cells around the X-color's 3x3-X center, over the input. Class STRUCTURAL, but the from-scratch rewrite lands at ~3795, not ~1300.

### Forced-component budget (measured on all 265 official+arc-gen)
- `condition` BOOL[1,1,30,30] = **900B FORCED** — the final `Where(condition, object_color[1,10,1,1], input)` must broadcast the condition to `[1,10,30,30]`, so it is `[1,1,30,30]` (BOOL = 1 byte, minimal). Paint-over the FREE full input is the cheap route because the input carries the empty-vs-background distinction; a scalar-canvas+`Equal` route must rebuild that with an active-grid mask (the 5865 route paid 900+450 for it). Cropping the output to 15x15 forces a `[1,10,15,15]` F32 input-crop base = 4500B — worse.
- Foreground `F32[1,1,14,14]` 784 + `BOOL[1,1,14,14]` 196 = **980B FORCED** — F32 input, first 10->1 channel reduction `Conv(input, W[1,10,1,1])` = sum(fg)-bg is F32->F32 (Conv can't emit 1-byte; QLinearConv needs a 9000B U8 input). Foreground reaches row/col 13 (all 265) so 14x14 is the tight crop.
- Reflection = six `BOOL[1,1,13,13]` = **1014B near-forced** — measured max other-color offset from center = 6, so the centered local frame is exactly radius-6 = 13x13. 4-fold OR = fg_local + leg_local(XOR static body mask) + 2 flips + 2 ORs.

`900 + 980 + 1014 = 2894B` forced; remaining ~900B is near-minimal detection(215)+color(160)+scalars(~250)+params(~250).

### Routes priced/tried this batch (all >= 3795) — do NOT re-explore
1. **Lean object_color (SHIPPED rung 3795)** — Sub/Mul over `[1,10,1,1]` vectors replaces the 7-tensor color chain. Only -95 available.
2. **Gather / absolute-frame reflection** — strictly worse than the centered local frame: foreground rows 0..13 reflect around center 5..9 up to row 15, so the frame must be 16x16 (256B/tensor > 169B) and loses the static body-removal mask; full-30x30 reflection is 900B/tensor. ~4000+.
3. **Scalar-canvas + final `Equal`** — the historical 5865 family; needs the active-grid mask to distinguish empty from background. Worse.
4. **Free-output Einsum** — center-dependent reflect basis >= `[5,15,30]` = 2250 params (library #109 2D-region boundary, #119). Dominated before center/color logic.

### Free-output `Equal` lever (reviewer batch54 rework) — DROP, visible-zero + cost adds
Hypothesis: the 900B `condition` is charged only because it feeds `Where`; rebuild the final stage as free-output
(`argmax_local = ArgMax(Slice(input,center±6))` -> `value_local = Where(condition_local, other_id, argmax_local)` U8
-> `Pad(value_local, bg_id)` -> `Equal(ch_ids)` -> `Cast` -> output), targeting ~3113. **Falsified 3 independent ways:**
1. **Visible-zero (MEASURED, numpy sim on the 265 official pairs = the visible gate): 0/265.** `Pad(value_local, bg_id=0)`
   fills the whole out-of-grid region with **background**, but LIVE leaves it **empty** (all-channels-0). `ArgMax` gives 0
   for BOTH empty and background, so it cannot distinguish them. Even a sentinel Pad fill (`10`->empty) is **25/265 (9.43%)**:
   out-of-grid cells *inside* the [center±6] window still get `ArgMax`=0=bg and can't be marked empty without an active-grid
   mask. Prereq on 5000 fresh arc-gen: crop-bound (a),(b) HOLD, but empty-vs-bg diverges on **5000/5000 (100%)**.
2. **The 900B is NOT deleted — it's relabeled.** The free-output `Equal` produces `[1,10,30,30]` from `value_canvas` =
   `Pad(...)` = `[1,1,30,30]` U8 = **900B** (same as `condition`). The 30x30 spatial output forces a 30x30 charged input to
   ANY final broadcast op (Where OR Equal); `Where` vs `Equal` is irrelevant. The output was already free in the Where route.
3. **Cost ADDS.** A per-cell color id (for `Equal`) needs `ArgMax` over channels = a `Slice(input,center±6)`=`[1,10,13,13]`
   F32 = **6760B** (or full-input argmax `[1,1,30,30]` INT64 = 7200B) that the BOOL-mask `Where` route never materialises.
   Real cost ≈ 3795 − 900(cond) − 120(color) + 340 + 900(value_canvas) + 6760(slice) ≈ **~10775**, not 3113.
   **`Where(cond, color_vec[1,10,1,1], input)` is strictly cheaper than `Equal(color_canvas, ch_ids)`**: Where factors the
   paint color into a free `[1,10,1,1]` vector; Equal must bake per-cell colors into the charged 30x30 canvas.

### One remaining hypothesis (low confidence)
The only way below ~3795 is a formulation that avoids BOTH a 30x30 final tensor AND an F32 channel-reduction simultaneously. None found. If a future op set gains a cheap 1-byte channel-reduction on F32 input, foreground drops 784->196 (~-590). Absent that, treat 3795 as the floor and spend LB budget elsewhere.
