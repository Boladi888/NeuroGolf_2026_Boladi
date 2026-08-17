# Separable-Einsum free-output crop: the carrier-dtype + dual-30-axis floor (task091 v15 KILL)

## What I discovered

The free-output separable Einsum (`out = Einsum('bchw,hH,wW->bcHW', input, M_row, M_col)`,
techniques §89/§91/§109) is a **correct but losing** formulation for a variable bbox crop-and-place
of a 10-channel one-hot grid into a `[1,10,30,30]` output. It is NOT a universal "dissolve the canvas"
win — it has a hard floor that a worker should price BEFORE building.

Two structural cost facts, both measured on task091 (officials 266/266 either way):

1. **Both selectors are irreducibly `[30,30]`.** The source axis spans the full 30 (the crop top
   `r0` can land anywhere in the grid, so source rows 0..29 are all reachable), and the output axis
   spans 30 (the graph output is `[1,10,30,30]`). You cannot shrink either dimension. Trying to shrink
   the OUTPUT to `[1,10,14,14]` + a free top-left `Pad` does not help: the Einsum's `[1,10,14,14]`
   result is now a CHARGED intermediate (it is no longer the graph output), and at F32 that is
   `10*14*14*4 = 7,840 B` — worse than keeping the selectors `[30,30]` and the Einsum output free.

2. **A 10-channel one-hot input forces an F32 carrier.** ORT CPU Einsum requires all operands to share
   dtype (float/int32/int64 only; U8 NOT_IMPLEMENTED). To get 2-byte F16 selectors you must
   `Cast(input, F16)` first — but that materializes a CHARGED `[1,10,30,30]` F16 plane = **18,000 B**.
   So in practice the carrier is locked to F32, and each selector is `30*30*4 = 3,600 B`.

**Hard floor of the route = two F32 `[30,30]` selectors = 7,200 B**, before any compare/cast
intermediates. Measured costs:
- F32-selector form: **11,797** (7,200 selectors + the `[30,30]` BOOL `Equal`/`And` build chain).
- F16-selector form: **26,199** (the 18,000-B input Cast dominates).

The team-best gather/slice tail for the same task is **2,576**. The separable Einsum is ~2.8x above
the gather tail at its theoretical floor, ~4.5x as actually built. KILLED.

## The transferable LAW (the missing-from-library bit)

The separable-Einsum free-output lever wins ONLY when the two selectors `M_row + M_col` are *cheaper
than the build-chain/canvas they replace*. §91 already states the crossover for a SINGLE crop input
(`crop_bytes + map_params < onehot_bytes + build_chain`), but it is framed around an already-cropped
small `[1,10,h_in,w_in]` F32 crop operand. The under-stated trap for the **whole-output, no-canvas**
variant (§109) is the **carrier-dtype tax**: when the Einsum input is the full multi-channel one-hot,
the selectors inherit the input's F32 (4-byte) dtype, and a `[30,30]` selector pair (7,200 B) is
already larger than a tight gather/slice color-index tail (~2,000-2,600 B). So:

> When the cropped CONTENT is data-dependent (arbitrary colors, not a few solid rectangles), the
> separable free-output Einsum carries the full `[1,10,30,30]` one-hot at F32 and its `[30,30]`x2
> selector floor (7,200 B) loses to a single-channel color-index gather/slice + Equal-tail. The
> Einsum lever is for LOW-RANK output content (§109 rank-1 rectangles), or for replacing a build-chain
> bigger than the selectors — NOT for moving arbitrary grid content that a 1-channel U8 gather already
> shifts for ~200 B per axis.

Concretely: for task091 the gather/slice tail's dominant tensors are a single-channel U8 color plane
(225 B) and a U8 canvas (900 B), both << a `[30,30]` F32 selector (3,600 B). Per-axis crop on a
1-channel U8 plane is `Gather` (~200 B); per-axis crop via Einsum selector is 3,600 B. The selector
loses by ~18x per axis on this geometry.

## Where

- Builder (documented dead-end): `The_400_Tasks/Task_by_Task/task091/prototypes/build_task091_v15_einsum_crop.py`
- task091.md Attempts row + Don't-do entry updated 2026-06-18.

## Why it matters

Saves the next worker the same ~1h build: the prompt for task091 explicitly routed the separable
Einsum as the path to "~1900", but the carrier-dtype tax and dual-30-axis floor make it impossible
to reach. The gather/slice tail (v13 static 3,274 / v14 dynamic-risk 2,576) remains the floor.
Suggest §91 / §109 get a one-line "carrier-dtype tax" caveat: a free-output Einsum over a full
multi-channel one-hot is F32-locked; price `2 * H_out * W_src * 4` against the canvas/gather tail
before building.
