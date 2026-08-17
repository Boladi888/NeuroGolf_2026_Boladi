# Tucker core init-folding: dead end for negative-valued initializers

## What was discovered

In the task362 v8 model (Tucker free-output #119), the Einsum has 4 copies of U[2,2,2] and 1 copy of J[2].
An apparent simplification: fold J into a new tensor V = J * U where V[r,a,b] = J[r] * U[r,a,b].
This would save 2 params (J no longer needed, V replaces one U copy).

**This fails for two reasons:**

1. **Net params = +6, not -2**: U appears in 4 positions in the Einsum (`rad`, `rbe`, `rpf`, `rqg`).
   Even if V absorbs one U copy, U is still needed for the other 3. Adding V[2,2,2]=8 params
   while keeping U[2,2,2]=8 params = net +8-2=+6. The fold only saves if V REPLACES ALL 4 copies of U simultaneously.

2. **Simultaneous fold requires J^(1/4) scaling**: to fold J into all 4 U copies at once,
   we need V = J^(1/4) * U so that V*V*V*V = J * U^4. But J[1] = -1.0, so J^(1/4)
   is complex (not real). The simultaneous fold is mathematically invalid for negative J values.

**General rule**: Init-folding into a Tucker core is only valid when:
- The init appears exactly ONCE in the Einsum, AND
- Its values are all non-negative (for root-based multi-fold approaches)

## Where encountered
- task362, w19 session 2026-06-24
- Attempted model: `The_400_Tasks/Task_by_Task/task362/prototypes/task362_v8_J_folded.onnx` (scored 113)
- Reference model: `The_400_Tasks/Task_by_Task/task362/prototypes/task362_current_submission.onnx` (cost 107)

## Why it matters
Workers might be tempted to fold a scalar initializer into a Tucker core to save params.
This analysis shows the conditions under which it fails: repeated-use cores cannot be
folded piecemeal, and negative values prevent root-based simultaneous folding.

## What's missing from the library
No existing section in `techniques.md` / `cost_floor_reference.md` discusses Tucker core
init-folding and its limits. This belongs as a note in the Tucker/free-output discussion (#119)
and possibly cost_floor_reference.md under "init count floor" reasoning.

## Sorter note: techniques.md §119 — extend with "init-folding into Tucker core: conditions and dead-ends" based on this analysis
## Sorter note: cost_floor_reference.md — add a line to the floor-verdict ledger for task362 at cost=107
