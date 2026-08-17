# Single-axis-round single-Einsum flood beats combined-step two-Einsum (task243, 6985 → 4718)

## What I discovered
For a linear-algebra (walk-counting) Einsum flood-fill, the per-step **operator choice sets the
einsum letter budget, which sets how many `state` intermediates you pay for.**

- **Combined 4-conn step** `Op = T⊗I + I⊗T` (spread both axes per step, via a shared stack index:
  `A=[T,I]`, `C=[I,T]`, factor `Iab,Ist`) costs **3 letters/step** → ≤~15 steps per Einsum
  (52-letter cap) → to reach ~28-step LIVE reach you need **two** Einsums → **two** `state`
  [1,1,18,18] F32 tensors (2×1296 B). This is Vamshi's 6985 graph.
- **Single-axis round** with a band matrix `M` (|i−j|≤1, includes self) gated by `allowed` each
  round (factor `nxt·cur`, one new coord) costs **1 letter/round** → **45 rounds fit in ONE
  Einsum** → only **one** `state` tensor. Saves 1296 B of memory. task243: 6985 → **4718** (−2267,
  −32.5%). Byte-reproducible; profile==static==4718.

The single 1296-B second `state` tensor in the two-Einsum form is *purely the price of the combined
operator's 3-letters/step*. Switching to 1-letter/round rounds buys it back.

## The correctness catch — and why single-axis still wins here (DUAL-TRUTH)
Single-axis alternating rounds do NOT produce the same fill *shape* as combined steps mid-flood, so
the cheap graph is **not byte-equal to LIVE**. Naively that reads as a fail. But task243 is
**ambiguous-class**: LIVE truncates at ~28 combined steps and *under-fills* long corridors, so LIVE
is itself <100% on ARC-GEN ground truth. Same-corpus per-grid attribution (run candidate AND LIVE on
identical grids, diff vs ground truth):

- n=4000: LIVE misses 4, max45 misses 1 ⊆ LIVE — 0 regressions, 3 fixes.
- n=8000: LIVE misses 5, max45 misses 2 ⊆ LIVE — 0 regressions, 3 fixes.

**max45's miss set is a strict subset of LIVE's on every corpus** → strictly more-correct-than-LIVE,
0 regressions, AND cheaper. Shippable per traps §DUAL-TRUTH.

Zero-regression theory: task243 is strictly **4-connected**, so no reachable path is purely diagonal
(diagonal cells aren't 4-adjacent) — every path is Manhattan. 45 alternating single-axis rounds reach
graph-distance ≈45 ≥ LIVE's 28 combined steps, so the single-axis fill ⊇ LIVE's fill on all 4-conn
grids. Both under-fill only extreme snakes needing >45 rounds, where single-axis is still ≥ LIVE.

## The general lesson / trap to record
1. **When you see a walk-counting Einsum flood split across N Einsums "because of the letter budget",
   check whether the per-step operator can be swapped from combined (3 letters) to single-axis
   (1 letter). Round count is FREE inside one Einsum (cost is set by the number of `state` tensors,
   not the number of einsum factors), so fitting more rounds in fewer Einsums directly deletes
   1296-B (F32 18×18) state intermediates.**
2. Single-axis ≠ combined mid-flood → expect ab-vs-LIVE divergence. Do NOT abandon on divergence for
   an ambiguous-class (LIVE<100%) task — run same-corpus per-grid miss attribution and check subset
   dominance (traps §DUAL-TRUTH). The prior task243 "sub-10k probes fail LIVE equality 0/265" don't-do
   note was about *wrong-rule* dilated/closure probes; an *exact-rule* single-axis flood that fills
   ≥ LIVE is a different, valid animal.
3. Stale floor-proofs bite: task243's local `context_pack`/taskNNN.md still carried an 11,315
   "formulation floor, NO-CANDIDATE" verdict (U8 bit-lane) that was already superseded by Vamshi's
   6985 Einsum and now 4718. Always re-check the live OnnxlyFans manifest/xlsx bar first; local
   task_state and archived floor proofs lag.

## Where
- Task 243 (region fill). Builder: `The_400_Tasks/Task_by_Task/task243/prototypes/build_task243_rank1_single.py`
  (`build(schedule=MAXALT)` → 45 rounds). Staged:
  `The_400_Tasks/Task_by_Task/task243/staged_candidates/task243_regionfillrewrite_cost4718/`
  (onnx sha256 `f97ee83e18141a5cb1fefdec94f36da64c005f5c0f3e6e2b1792a89517d4f8aa`).
- Prior team-best Vamshi 6985 (combined-step, two-Einsum) at OnnxlyFans manifest sha `8c3e6ba5…`.

## What's missing from the library
`techniques.md §37` (diameter floor) documents the round-count floor for the U8-bitpack alternating
flood but frames alternating-vs-combined as a *within-U8-bitpack* micro-optimization. It does not
record the **Einsum letter-budget → state-tensor-count** mechanism, nor the single-axis(1 letter) vs
combined(3 letters) trade that lets you collapse a two-Einsum walk-flood into one. Searched §37 +
`by_symptom.md` "einsum flood" — not present.

## Sorter note: techniques.md §37 — extend with "Einsum walk-flood: per-step operator letter budget sets state-tensor count; single-axis(1 letter/round) collapses combined-step(3 letters/step) two-Einsum into one, deleting a 1296-B F32 state (task243 6985→4718)".
## Sorter note: traps.md §DUAL-TRUTH — add task243 exemplar: cheap single-axis flood diverges from LIVE but miss-set ⊆ LIVE miss-set on same corpus (n=8000, 0 regressions) → shippable; 4-conn no-diagonal-path argument backs zero-regression.
