# PLANE-TAX LAW: bank-Gather vs runtime-generated mask — when killing a param bank LOSES

**Where:** task104 (2x2 expand free-output). Builders:
`The_400_Tasks/Task_by_Task/task104/prototypes/build_task104_{freeout_bank,bank_padlast,gen_padlast,flipbank_padlast}.py`.
Staged: `staged_submits/task104_expand_bank_cost1209/` (1,384 -> 1,209, byte-equal 200k, profile==static).

## The transferable insight (extends #79 parameter-LUT crossover law, and #81/#67)

When a small fixed N-pattern bank ([K,h,w] init) feeds a `Gather` to produce a canvas,
**replacing the bank with a runtime-COMPUTED mask only pays if the computation adds <= ~1
extra [h,w] plane.** Every intermediate `[h,w]` tensor (compare/Where/And/Mul output) is
charged at `h*w*itemsize` by `calculate_memory` (it sums ALL named non-output tensors).

Measured on task104 (h*w = 81, one-hot floor 810 B):
- `freeout_bank` / `bank_padlast`: [4,9,9]=324-param bank, Gather -> canvas, **0 extra planes**. cost 1,298 (padlast) .
- `gen_padlast`: bank killed -> 64 params, but mask built from row-ramp/col-ramp/Where(S-vector)+2 interval compares+And = **4 extra [9,9] planes (~519 B)**. cost **1,435 = REGRESSION** (the 519 B of planes outweighs the 260 params saved).
- `flipbank_padlast` (WINNER): store ONLY the canonical pattern P0 [9,9] (81 params); the other 3 are flips (P1=fliplr, P2=flipud, P3=rot180); conditional row/col reverse via `Where(bit, rev, fwd)` -> `Gather` perm. Adds **+1 Gather plane (81 B) + 2 tiny perm vectors**. cost **1,209** (-175 vs team-best).

**Rule of thumb to add to the param-LUT crossover (#79):** price the route as
`params_saved` vs `extra_planes * h*w * itemsize`. Below ~150 params the plane-tax usually
wins for the bank; a SYMMETRY decomposition (store 1 canonical + flip/rotate via Gather-perm)
is the middle ground — it cuts params hard while adding only ~1 plane.

**Two cheap sub-lessons also confirmed here:**
- **int32 (not int64) Gather permutation/index vectors**: halves their memory (8->4 B/elem). On task104 the two [9] perm vectors went 72->36 B each, -72 cost for free.
- **u8 elementwise Mul is opset-14+** (#74 trap re-confirmed at opset 13 — `Mul(u8)` ShapeInferenceError "unsupported type tensor(uint8)"). Use `Where(mask, val_u8, 0_u8)` to stamp a scalar value into a u8 canvas instead of `Mul(mask_u8, val_u8)` — Where(bool,u8,u8) is fine at opset 13.

## Floor note for task104 (forced-component)
ch3 mask = OR of TWO 4x4 rectangles with crossed (row-band,col-band) couplings (ts!=bs all 4
patterns) => rank-2 => no rank-1 broadcast free-output (~600 B) can express it; 10-ch one-hot
must materialize. PAD-LAST one-hot 810 B < Equal-tail 900 B canvas (#81 confirmed at 9x9). The
prompt's ~650 target presumes a single-rectangle (rank-1) output and is unreachable for this rule.

## Library-gap check
Searched #79/#81/#67/#82 — the param-LUT crossover (#79) prices output-LUT vs param-LUT vs
computed, but does NOT state the PLANE-TAX (each runtime mask op is a charged [h,w] plane) nor
the symmetry-decomposition middle ground. Suggest extending #79 (or #81) with the measured
plane-tax inequality and the "store-1-canonical + flip-Gather" pattern.
