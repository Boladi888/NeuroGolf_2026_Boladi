# Root `.update` Backlog Drain — REPO-ACCESS card

**Deliverable type:** repo-access librarian drain packet.

**Goal:** integrate or dismiss the lesson drops currently sitting in the repo-root `.update/` folder. The official Librarian digest only reads `Agenters/Librarian/Updates_*` and `Onnx_Advanced/.update/DONE_*.md`, so these root drops are invisible unless explicitly drained.

## Contract

Work read-first and librarian-style:

- Do not submit to Kaggle.
- Do not edit team submission files.
- Dedup against current `Onnx_Advanced/` and `library_db` before proposing edits.
- Prefer compact reviewed `library-fact` blocks when the lesson is granular.
- Promote to `techniques.md`, `traps.md`, `op_cost_patterns.md`, `ort_reality.md`, `cost_floor_reference.md`, or `diagnostics_recipes.md` only when the lesson is durable enough and not already covered.
- After integration, archive consumed root drops or explicitly list them as dismissed/duplicate.

## Backlog Source

Root folder:

```text
<REPO_ROOT>\.update
```

As of 2026-07-11, it contains these 25 drops:

```text
DONE_2026-07-08T065936Z_conv_inner_crop_transport_dilated_lowparam.md
DONE_2026-07-09T115500Z_relive_baseline_and_offline_pregate.md
DONE_2026-07-09T184440Z_gather_index_beats_free_output_for_permute_tile.md
DONE_2026-07-09T191447Z_freeoutput_padtail_selfdefeating_when_best_already_freeoutput.md
DONE_2026-07-09T201927Z_ort_samesize_resize_shortcircuit_and_resize_beats_freeoutput.md
DONE_2026-07-10T202012Z_argmax_indexplane_is_int64_costtrap_coloridx_collapse_falsefloor.md
DONE_2026-07-10T220523Z_paramdeletion_not_totalcost_variablescale_stamp_render_floor.md
DONE_2026-07-11T041239Z_task208_coloridx_falsefloor_second_confirmation.md
DONE_task014_compare_vector_crop_fold.md
DONE_task042_select_tail_gated_union_combine.md
DONE_task050_marker_side_D_reuse_2026-07-09.md
DONE_task052_einsum_cp_rank_reduction.md
DONE_task053_dynamic_weight_convtranspose_shift.md
DONE_task056_pad_float_only_coupling.md
DONE_task090_int8_detect_domain_reuse_render.md
DONE_task118_computed_convweight_scoreable_kernelselect.md
DONE_task129_mode_fill_scalar_collapse_floor.md
DONE_task155_reversesequence_illegal_axis_locked.md
DONE_task163_freeoutput_cp_forced_floor_proofs_2026-07-09.md
DONE_task197_freeoutput_gather_index_floor_2026-07-09.md
DONE_task211_separable_tiling_einsum_signchange_floor.md
DONE_task223_upscale_into_canvas_maxroipool_floor.md
DONE_task262_col_position_remap_single_einsum_floor.md
DONE_task326_sparse_init_banned_2x2crop_floor.md
DONE_task344_freeoutput_recolor_threshold_base_floor.md
```

## Initial Grouping

Use this as a starting sort, not as a final decision:

| Group | Drops | Likely destination |
|---|---|---|
| Free-output / false-floor cluster | task344, task254/2026-07-09T191447, task249/2026-07-09T184440, task289/2026-07-09T201927, task064/2026-07-10T202012, task133/2026-07-10T220523, task208/2026-07-11T041239, task197, task163, task050, task052, task211, task262, task223, task129 | `techniques.md`/#119 family, `traps.md`, `op_cost_patterns.md`, `cost_floor_reference.md`, `library_db` facts |
| Live-first / scorer-truth workflow | 2026-07-09T115500 relive baseline/offline pregate | `traps.md`, `diagnostics_recipes.md`, `techniques.md`, `library_db` facts |
| Structural byte shaves | 2026-07-08T065936, task118, task090, task042, task014, task053 | `techniques.md`, `op_cost_patterns.md`, `diagnostics_recipes.md`, `library_db` facts |
| ORT/scorer dead levers | task326, task155, task056 | Dedup/extend `core.md`, `ort_reality.md`, `scoring_internals.md`, `traps.md`, `library_db` facts |

Also inspect HandOff no-packet discoveries:

```text
Handoff/logs/raw/runner/task008/worker_bo_ng_worker1/20260707T063202Z.final.md
Handoff/logs/raw/runner_cdx2/task176/worker_bo_ng_cdx2_worker3/20260709T160802Z.final.md
```

Expected reusable lessons:

- `task008`: INT32-to-INT8 coordinate/pad-vector narrowing, positive F32-to-INT8 truncation deleting `Floor`, INT8 Slice indices dead.
- `task176`: shared `basis[2,30]` row/col factor replacement for free-output/Einsum; tied color/spatial basis visible-fails.

## Required Output

Write `Logs/research/ROOT_UPDATE_BACKLOG_DRAIN_20260710.md` with:

1. **Drain Summary** — integrated, duplicate/dismissed, deferred.
2. **Dedup Table** — each source file, existing coverage, final action.
3. **Reviewed Fact Candidates** — copy-ready `~~~library-fact` blocks.
4. **Doc Patch Plan** — exact root doc sections to extend or replace, with size-budget risks.
5. **Archive Plan** — exact consumed root `.update` files to move and destination.

## Verification Checklist

- `python Agenters/Librarian/librarian_ctl.py digest`
- `Get-ChildItem .update -Filter DONE_*.md`
- `python Onnx_Advanced/library_db/lib_query.py status`
- If any library docs are edited: rebuild `techniques_catalog.md` when `techniques.md` changes, rebuild `library_db`, then run `python Onnx_Advanced/.update/tools/lib_lint.py`.
