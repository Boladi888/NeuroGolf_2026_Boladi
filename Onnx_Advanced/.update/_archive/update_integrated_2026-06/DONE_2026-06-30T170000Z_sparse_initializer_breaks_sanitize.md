# sparse_initializer is a DEAD param-reduction lever (breaks the official scorer)

## What I discovered

`onnx.SparseTensorProto` initializers look like a free params win: the official
`neurogolf_utils.calculate_params` counts a sparse initializer by its **nnz**
(`math.prod(sparse_init.values.dims)`), not its dense element count, and
`calculate_memory` adds sparse names to `init_names` so they're excluded from
memory too. So converting a mostly-zero conv weight to sparse *appears* to cut
params to the nonzero count for free.

**It does not work.** The official `sanitize_model(model)` (in
`data/competitions/neurogolf-2026/neurogolf_utils/neurogolf_utils.py`) iterates
`graph.input` and `graph.initializer` and renames them via `get_safe_name`, then
rewrites every node input/output. It **never iterates `graph.sparse_initializer`**.
Result: a node input that referenced the sparse tensor gets renamed to
`safe_name_N`, but the `SparseTensorProto.values.name` keeps its original name →
dangling reference → `onnxruntime.InferenceSession(...)` fails to load →
`tools/_score_v2.py: score_network_v2()` catches the exception and returns `None`
(you also see ORT log `Could not write a profile because no model was loaded`).

Empirically verified 2026-06-30 on task204: dense baseline scores
`(0, 6652, 1030)`; the identical model with `paint_w`/`topleft_w` moved to
`graph.sparse_initializer` scores `None`.

## Where

- task204 (`The_400_Tasks/Task_by_Task/task204/prototypes/build_task204_v2.py`),
  floor re-audit session. Probe was a throwaway; the dead-end is recorded in
  `The_400_Tasks/Task_by_Task/task204.md` don't-do log.

## Why it matters

This is a tempting, **task-independent** lever — any task with a large sparse
initializer (paint/stamp banks, one-hot tables, masked conv weights) will look
like it can shave hundreds of params for free. It can't, under the current
sanitize. Workers should not burn time on it unless/until `sanitize_model`
starts iterating `sparse_initializer`. (If the live Kaggle grader's sanitize
differs, that's a separate question, but the LOCAL gate/captain pipeline rejects
it, so it's unstageable here regardless.)

## What's missing from the library

I did not find any existing note on sparse initializers in the cost/scoring
docs. Searched `lib_query.py search "sparse initializer"` and the floor docs —
no hit. Worth a one-liner in the scoring-internals / grader-gap material so the
next worker doesn't re-derive it.

## Related (same session, also task-independent framing)

**Zero-sum theorem for all-U8 pipelines.** Scorer cost =
`sum(intermediate bytes, dtype-AWARE)` + `sum(initializer elements, dtype-BLIND)`,
with `input`/`output` tensors free. For a pipeline whose intermediates are all
U8, 1 intermediate byte ≡ 1 param element ≡ 1 cost unit. Therefore moving work
between conv weights (params) and conv outputs (memory) is exactly zero-sum, and
adding any intermediate plane is strictly additive. Practical consequence:
"split this conv into smaller-kernel buckets" or "value-code the detector" style
rewrites cannot beat a tight U8 floor — the param savings are paid back byte-for-
byte in extra planes. The only real levers are (a) deleting a load-bearing
tensor, or (b) shrinking a forced **F32** intermediate (F32 is the only 4×-dtype
in these graphs, so F32 slices/casts are where the asymmetric savings live).

## Sorter note: scoring_internals.md (or grader-gap section) — add as new rule "sparse_initializer_dead_under_sanitize": sparse inits count by nnz in calculate_params but sanitize_model never renames sparse_initializer, so node refs dangle → ORT load fail → score None. Empirically tested task204 2026-06-30.

## Sorter note: scoring_internals.md — add as new rule "u8_pipeline_zero_sum": for all-U8 graphs, 1 intermediate byte ≡ 1 param element; weight↔output reshuffles are zero-sum, extra planes strictly additive; real levers are deleting load-bearing tensors or shrinking forced F32 intermediates.
