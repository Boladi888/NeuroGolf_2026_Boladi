# task217 — a "forced F16 detector copy" floor was carrying 324B of slack (falsified)

## What I discovered

A "DISSOLVE FLOOR" floor-proof on task217 claimed two F16 [1,1,9,9] detector
copies (`ch0_f16` Cast + `nonbg` complement, 162B each = 324B) were FORCED because
the topleft locator (`ReduceMax → ArgMax`) and the BOOL sprite source "needed F16".
That is wrong. The detector runs in U8/BOOL at opset 13 with no F16 anywhere:

- `ch0_u8 = Cast(ch0_crop F32, U8)` — 81B, replaces `ch0_f16` (162B).
- `nonbg_b = Less(ch0_u8, 1) → BOOL` — 81B, the foreground complement. **ORT GOTCHA
  confirmed: `Sub(uint8, uint8)` has NO kernel below opset 14 (InvalidGraph), so you
  CANNOT do `Sub(1, ch0)` in U8 — use `Less(ch0, 1)` for the 1-ch0 complement.**
- `nonbg = Cast(nonbg_b, U8)` feeds `ReduceMax(U8) → ArgMax(U8)`. ReduceMax(U8) is
  opset 12+ and **ArgMax(U8) is valid** (only ArgMax(BOOL) is rejected, per the
  ort_op_dtype_matrix ArgMax row). The BOOL `nonbg_b` feeds the sprite Gathers directly.

This dropped cost 2178 → **1979** (opset 13, −199), all gates byte-equal/100%.

## The opset-20 stretch also landed

Reduce `nonbg_b` (BOOL) DIRECTLY with `ReduceMax(BOOL)` (opset 20, axes-input schema
from opset 18, ir_version 9 — already documented in ort_reality.md lines 22/488,
task191 v6). This drops the separate `nonbg` U8 copy; Cast each row_any/col_any BOOL
result to U8 before ArgMax (ArgMax(BOOL) still invalid). 1979 → **1920** (−59). ORT
executed ReduceMax(BOOL) cleanly on the local build; all gates byte-equal/truth-100%/
adversarial 4788/4788, profile==static.

## Why it matters / transferable lesson

**A locator/detector that ends in ArgMax does NOT force an F16 copy.** The common
reflex "I need ReduceMax/ArgMax, therefore F16" is false: ReduceMax has a U8 kernel
(opset 12+) and ArgMax accepts U8. For a 1-ch0 foreground mask the whole detector
chain stays 1-byte (U8) or BOOL. Any floor-proof that lists "F16 detector copies" as
forced/irreducible should be re-checked against the U8/BOOL detector path — here it
hid 324B (15%) of slack behind a "FLOOR" label.

## Where

- Tasks/build: `The_400_Tasks/Task_by_Task/task217/prototypes/build_task217_dissolve_u8_v6.py`
  (opset-13, 1979) and `..._v7_op20.py` (opset-20, 1920).
- Staged: `staged_submits/task217_dissolve_op20_boolreduce_cost1920/` (primary),
  `staged_submits/task217_dissolve_u8nobg_op13_cost1979/` (opset-13 fallback).

## What's missing from the library

The ort_op_dtype_matrix rows for ReduceMax(U8 opset12+) and ArgMax(BOOL-rejected,
U8-OK) are already present and correct — the gap was a task-level floor-proof
trusting "F16 needed for ArgMax" without consulting them. Possibly worth a one-line
note in traps.md §FAMILY-LABEL / floor-proof discipline: "a detector ending in
ArgMax does not force F16 — ReduceMax(U8)+ArgMax(U8) keeps it 1-byte."
