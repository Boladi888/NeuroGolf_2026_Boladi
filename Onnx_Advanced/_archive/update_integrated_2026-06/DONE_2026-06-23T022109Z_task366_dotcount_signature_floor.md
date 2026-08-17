# DONE: task366 dot-count-signature angle = the lean-coord solver (does NOT beat 37,577)

**Date:** 2026-06-23 · **Worker:** worker-1 (beta_p21_claude_cli) · **Task:** 366 (two-half scene/library box-copy)

## Transferable insight (for traps/cost-floor reference)

When a prompt proposes "reduce a per-CELL flat (`[1,255]`/`[H*W]`) bookkeeping tensor to a
per-COMPONENT scalar descriptor to delete it," **first check whether the flat is a
used/consumed-state substrate, not just a presence readout.** On task366 the `[1,255]`
flats feed `ScatterElements(used_mask)` + `GatherElements(present)`: a placement consumes
SPECIFIC scene marker cells, so "which cells are used" is inherently per-cell and cannot
collapse to a per-component scalar. The descriptor match (count,color) only replaces the
SELECTION step, not the consumption step — so the flat survives in a different guise.

## Cost-anatomy method that settled it (reusable diagnostic)

Group counted intermediates by SHAPE and by PRODUCER OP, not just top-N bytes:
- by-shape exposed that `[1,1,15,17]`×44 = 11,220B (half-plane working tensors) is the
  BULK of the 37,577, while the `[1,255]`×22 = 7,650B "deletion target" is secondary.
- by-op (on the from-scratch lean coord solver `lean_final.onnx`, 53,097) exposed that
  CC propagation (MaxPool) is cheap (1,530B); the cost is Cast 8,486 + Where 8,118 +
  And 6,720 + Reshape 5,728 = the per-candidate MATCHING combinatorics expressed as
  separate static ops. ONNX has no Loop/in-place, and the cardinality-ordered 3→2→1
  schedule forces per-level placement planes — each a counted tensor.

## Floor verdict

Optimistic perfectly-fused floor estimate ~10.6k (label Conv 3600 + out canvas 900 +
~6k of single-live-copy `[15,17]` role planes) shows the win is NOT
information-theoretically impossible. But 198 structural versions (best safe 49,866) AND
two dedicated from-scratch coord/dot-count sessions (best 53,097) all LOST to the
byte-equal Aditya polish 37,577. The gap is the optimistic-floor-vs-realizable-DAG gap:
ORT static-shape, no-Loop, no-in-place + the required cardinality schedule prevent the
tensor reuse the floor assumes. 37,577 (static==profile==37,577, truth 99.33%@20k,
visible 255/255) stands as team-best.

## Suggested routing
- cost_floor_reference.md: add "per-cell consumed-state flats don't collapse to
  per-component scalars" to the plane-floor taxonomy / floor-verdict ledger.
- (No library file edited directly per protocol — sorter assigns §.)
