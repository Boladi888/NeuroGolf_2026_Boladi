# task387 worker packet - bo_ng_cdx_worker2 - cost2557

## Candidate

- Staged ONNX: `The_400_Tasks/Task_by_Task/task387/staged_candidates/task387_lte_active_extent_cost2557/task387.onnx`
- Manifest: `The_400_Tasks/Task_by_Task/task387/staged_candidates/task387_lte_active_extent_cost2557/manifest.json`
- Builder: `The_400_Tasks/Task_by_Task/task387/prototypes/build_lte_active_extent.py`
- Prototype: `The_400_Tasks/Task_by_Task/task387/prototypes/task387_lte_active_extent.onnx`
- Cost/profile cost: `score_v2=(0,1898,659) -> 2557`; `profile=2557`
- SHA-256: `46920fcb9f53881e756b0a0acc4986c55a5185b88012e20ece8a99f69a84374f`

## Bar Used

Authoritative bar from `python tools/ng_current_bar.py --task 387 --json`:

```text
live_cost=2566
live_sha256=6a3f8962a08024e78a830376e3c693a6de53fb5c81f8691d6e7886792e17c8e4
owner=Boladi
updated_utc=2026-07-12T23:59:53Z
authoritative_source=manifest
```

Delta vs live: `2557 - 2566 = -9`.

## What Changed

The parent computed active masks as `coord16 < Cast(extent_m1 + 1, F16)`.
Since `coord16` is integral and `extent_m1` is exactly `H-1` or `W-1`, this is equivalent to `coord16 <= Cast(extent_m1, F16)`.

The rewrite deletes:

```text
one_f initializer: -1 param
height_f scalar Add output: -4 profile-memory bytes
width_f scalar Add output: -4 profile-memory bytes
```

Net official profile delta: `memory 1906 -> 1898`, `params 660 -> 659`, `cost 2566 -> 2557`.

`ng_graph_cost_diff` was run. It identified `one_f`, `height_f`, and `width_f` as the changed structures, but its static scalar accounting printed a tie (`2414 -> 2414`); `_score_v2` and `profile_real_cost` are authoritative for this profiled graph.

## Gates

```text
visible: [visible] 266/266 (100.00%)  PASS
score: task387_lte_active_extent.onnx (0, 1898, 659) cost 2557
profile: task387 task387.onnx: static=2557 profile=2557 (memory 1898 + params 659; 266 real inputs)
truth --n 8000: [truth] vs ARC-GEN ground truth: 8000/8000 (100.000%)
ab --n 8000: [ab] vs submission.zip: match 8266/8266 | divergences=0 | byte_equal=True (first_div=None)
safety: PASS, 0 failures, 0 warnings
polish: original_cost=2557 final_cost=2557; all polish passes no-op
```

## Classification

```text
candidate_class: PATH_A_BYTE_EQUAL
parent_source: The_400_Tasks/Task_by_Task/task387/staged_candidates/task387_drop_coord2_allc_cost2566/task387.onnx
parent_sha256: 6a3f8962a08024e78a830376e3c693a6de53fb5c81f8691d6e7886792e17c8e4
candidate_sha256: 46920fcb9f53881e756b0a0acc4986c55a5185b88012e20ece8a99f69a84374f
x86_required: false
kaggle_probe_required: false
status: LOCAL_GOLD
```

Raw safety JSON is embedded in the staged manifest.

## Tried / Remaining

- Tried the scalar active-extent inequality fold; it passed all gates and is staged.
- Polish pipeline was a no-op.
- Did not re-run known-dead P carrier routes already recorded in `hard_records/freeoutput_einsum_render_2607_wall.md`.

Remaining floor hypothesis: the large remaining lever is still `P[18,30]=540` params. A legal way to provide the final Einsum's size-30 embed without dense `P` and without a charged materialized tensor is the only obvious large route left.

Lesson packet: none.
