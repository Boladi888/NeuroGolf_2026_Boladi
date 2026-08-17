# opset-20 ReduceMax(BOOL) removes a Cast-to-UINT8 inserted only to feed a reducer

## What I discovered

A `Cast(BOOL -> UINT8)` that exists **only** so a BOOL mask can be reduced by `ReduceMax`
(BOOL is rejected by ReduceMax at opset <= 17) is removable by bumping the model to **opset 20**,
where `ReduceMax(BOOL)` is supported (ort 1.24.4; ort_reality.md already records this as
Kaggle-confirmed via task191 v6, 2026-06-02).

The non-obvious refinement: the reduced result often feeds a **second** op that still rejects BOOL
(here `ArgMax`, which has no BOOL kernel). The win is to reduce in BOOL and then `Cast` only the
**small reduced tensor** to UINT8, instead of casting the full pre-reduction tensor. Cost delta is
`full_pre_reduce_bytes - reduced_bytes`.

On task319 the v35 graph cast the full BOOL bbox mask `mask_3_cropped_b [1,3,19,19] = 1083 B` into a
full UINT8 twin `mask_3_cropped [1,3,19,19] = 1083 B` purely so two `ReduceMax(axis=2/3)` could run.
v36 does `ReduceMax(BOOL)` on the BOOL mask directly (axes as INPUT, required at opset >= 18), giving
BOOL `[1,3,19]` row/col-presence, then casts those two tiny tensors to UINT8 (2 x 57 B) for ArgMax.

Result: -1083 + ~114 = ~-969 B raw; **13,681 -> 12,714 after polish (-967 B, -7.1%)** vs the prior
team-best. Byte-equal to the shipped parent over the full 200,267-pair A/B (0 divergences); static ==
profile cost.

**Gotcha when bumping past opset 18:** ALL `ReduceMax`/`ReduceSum`/etc. nodes in the graph must move
`axes` from an attribute to an op INPUT, or ORT rejects the model with INVALID_GRAPH. Convert every
reducer in the graph, not just the one you changed, in the same bump.

## Where

- task319, builder `The_400_Tasks/Task_by_Task/task319/prototypes/build_task319_structural_v36.py`
- staged `staged_submits/task319_dilated_cropconv_boolreduce_cost12714_sha6DA41F43_gatesgreen/`
- sha256 6DA41F43A2D86EA37CB3B1AEF831A511B5A66E7144D9069F5890EC60BDF99BB5

## Why it matters

Generalizes the existing ort_reality.md ReduceMax(BOOL) note into a concrete cost lever + a builder
checklist item (convert all reducers' axes on opset bump). Any structural graph that casts a BOOL mask
to UINT8 solely to bbox-reduce it is carrying a removable full-size tensor.

## What's missing from the library

ort_reality.md records that ReduceMax(BOOL) works at opset 20, but does NOT spell out (a) the
"cast the reduced result, not the source" cost pattern, nor (b) the "convert ALL reducers' axes to
input when bumping past opset 18" build trap. Both bit me and are worth a one-liner.

## Sorter note: ort_reality.md (ReduceMax(BOOL) section ~line 271-276) — extend with the cost-lever pattern: when a BOOL->UINT8 Cast exists only to feed ReduceMax, opset-20 ReduceMax(BOOL) removes it; if the reduced result feeds an op that still rejects BOOL (ArgMax), Cast the small reduced tensor instead of the full source (save full_bytes - reduced_bytes). Add build-trap: opset>=18 requires ALL reducer axes as INPUT not attribute — convert every reducer on the bump.
## Sorter note: task_techniques_index.md (task319 row) — update from 16,552 to 12,714 (v36, opset-20 ReduceMax(BOOL)); current builder build_task319_structural_v36.py.
