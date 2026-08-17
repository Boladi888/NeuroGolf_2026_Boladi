# A both-input-contraction Einsum is NOT intrinsically a gate timeout — fix the operand ORDER first

## What I discovered

When a free-output Einsum folds TWO spatial factors inline from two copies of the
same full `input` operand, ORT's default contraction path can materialize the
dense cross-product of the two `[1,10,30,30]` operands and blow up runtime by
~1000x. The fix is purely operand ORDER (no shape/dtype change, byte-identical
output): list each `input` copy ADJACENT to its own small factor operands so ORT
contracts each side down to a tiny vector FIRST, then combines.

Concrete (task293, the line/band crossing low-rank reconstruction):

- SLOW (`foldbasis_direct`): `pc,bdhy,bexw,qd,re,y,x,pqr->bchw`
  → first-run 9569 ms, steady **6308 ms/pair** → 200k truth ≈ 21000 min → 120s
    visible gate timed out. This made a worker (2026-06-19) log the whole
    both-fold as a dead-end ("not the reviewer candidate").
- FAST (`foldbasis_fast`): `bdhy,qd,y,bexw,re,x,pc,pqr->bchw`
  → ORT contracts `bdhy,qd,y->bqh` (row factor) and `bexw,re,x->brw` (col factor)
    as cheap pairwise steps first, then `bqh,brw,pc,pqr->bchw`.
  → first-run 5.8 ms, steady **3.8 ms/pair** → 200k truth ≈ 13 min. ALL gates
    green, byte-identical to the slow version and to the prior fold_q ship.

The operands are mathematically identical (Einsum is order-invariant in result);
only ORT's greedy pairwise contraction heuristic differs. Putting an operand's
contraction partners next to it steers the heuristic to the cheap path.

## Where I encountered it

- task293. Builder: `The_400_Tasks/Task_by_Task/task293/prototypes/build_task293_foldbasis_fast.py`
- Staged: `staged_submits/task293_foldbasis_fast_cost358/` (cost 358, was 598 fold_q, was 838 team-best file).
- The slow predecessor builder: `.../prototypes/build_task293_foldbasis_direct.py`.

## Why it matters

Saved 240 B (598→358 static & profile; +0.51 pts over the prior local floor,
+0.85 pts vs the 838 team-best file) AND retired a false dead-end. A worker had
abandoned the both-fold purely on the 120s timeout without trying operand
reordering. The lesson generalizes: ANY Einsum that contracts two large operands
into a small output is a reorder candidate before you conclude it's intractable.
Verify with a 1-input timing probe (build, `InferenceSession`, time one
`sess.run`) BEFORE arming a 200k gate.

## What's missing from the library

`techniques.md §89` (Einsum operand order / in-kernel temps) covers that operand
order is free and controls intermediates, but I did not find an explicit, dated
WARNING that ORT's path heuristic can produce a ~1000x cross-product blowup for
TWO-full-operand contractions, nor the diagnostic recipe (per-pair ms timing
probe before gating). This is the actionable bit for the next worker who sees a
both-fold "timeout".

## Sorter note: techniques.md §89 — extend with "two-full-operand Einsum: reorder so each large operand is adjacent to its own small factors; ORT's greedy path otherwise materializes the dense cross-product (~1000x, task293 6308ms→3.8ms/pair). Probe one sess.run timing before gating."
## Sorter note: by_symptom.md — add route "Einsum gate timeout / very slow inference but small static cost → suspect contraction-path cross-product; reorder operands (task293), don't abandon the fold."
