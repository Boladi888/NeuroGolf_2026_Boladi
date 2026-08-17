# Floor audits must check redundant-valued initializers + removable remask nodes, not just function-changing configs

## What I discovered
A "byte-equal safe compression" (same function, fewer bytes) can beat a graph
that a prior floor proof declared minimal — because the floor proof only
searched *function*-changing variants (different flood depth / repair groups)
and asserted "the exact graph is already minimal" without a mechanical audit
of the graph itself.

Two byte-equal shave classes that a floor audit must always run before
declaring a WALL on a hand-built structural-LIVE graph:

1. **Duplicate-valued initializers.** Group every initializer by exact array
   value; any group with >1 member is free params. Reuse one survivor, delete
   the rest, rewire consumers. In task002 this found 7 dupes worth −52 params:
   a `shift1(=1)` scalar duplicating `one_u32`, `top_idx1/bot_idx1` int64[20]
   index arrays byte-identical to `prev_idx/next_idx` (max(0,r-1)/min(19,r+1)
   == the flood prev/next gathers), `bit_masks_mid` == `bit_masks_lo`, and 3
   single-element `repair_row_idx*` scalars ([1],[2],[0]) duplicating existing
   `pack_steps/pack_axes/pack_starts`. Params are counted as ELEMENT COUNTS,
   so a 20-elem index dup is −20 to cost, cheap to miss and cheap to reclaim.

2. **Removable remask / idempotent-clip nodes.** A `X & mask` node whose result
   is re-masked by the same `& mask` later in the pipeline is often removable:
   feed the pre-mask tensor forward and let the downstream clip do the work.
   In task002 the seed remask `reach_0 = ray_unmasked & open_bits` fed a flood
   loop whose every iteration ends in `& open_bits`; dropping reach_0 and
   feeding `ray_unmasked` straight into iter-0 is byte-equal (−80 bytes: one
   uint32 [1,1,20,1] tensor). CAVEAT: this is only byte-equal if the downstream
   clip fully re-establishes the invariant on the corpus — VERIFY with
   `ab --full` (officials+corpus), never officials-only. task002 was
   Kaggle-confirmed byte-equal on 200268 tensors.

Net: task002 LIVE 9545 -> 9413 (-132), byte-equal, Kaggle-confirmed +0.014 LB.

## Where
- task002 (enclosed-fill flood-fill / structural-LIVE compression).
- Builder: `The_400_Tasks/Task_by_Task/task002/prototypes/build_dedupe_9413.py`
  (deterministic in-place fold of the canonical LIVE onnx; reproduces the
  confirmed SHA `8c8340bc…` exactly).
- Staged: `Task_by_Task/task002/staged_candidates/task002_enclosedfill_cost9413/`.

## Why it matters
A prior full re-derivation of this same task (2026-06-30, same worker id)
declared "WALL @ 9545, exact-semantics floor, byte-equal-cheaper impossible."
That verdict was wrong by −132 and would have permanently left the win on the
table if the confirmed artifact hadn't survived in `work/`. The trap: proving a
FUNCTION floor (no cheaper correct algorithm) is NOT proving a BYTE floor (no
cheaper encoding of the same algorithm). Both must be shown before a WALL.

## What's missing from the library
The floor-proof discipline in traps/WORKER_PROTOCOL emphasizes forced-component
arguments (dtype kernels, binary-op fan-in) but does not prescribe the two
mechanical byte-equal audits above as a required pre-WALL checklist. A tiny
reusable "dedupe initializers + splice idempotent remask, then `ab --full`"
pass would have caught this directly.

## Sorter note: traps.md — extend the floor-proof / WALL guidance with a required "byte-equal audit" checklist (1) dedupe equal-valued initializers, (2) splice remask nodes whose invariant is re-established downstream — verified via ab --full, before any WALL verdict. Cross-ref #157 unpack + task002.
## Sorter note: techniques_catalog.md — add a 1-line verdict: "task002 safe-compression -132 byte-equal (dup-initializer reuse + idempotent-remask splice), Kaggle-confirmed."
