# task387 worker packet - bo_ng_cdx_worker2

## Staged candidate

- Candidate: `The_400_Tasks/Task_by_Task/task387/staged_candidates/task387_absorb_batchW_cost2606/task387.onnx`
- Manifest: `The_400_Tasks/Task_by_Task/task387/staged_candidates/task387_absorb_batchW_cost2606/manifest.json`
- Prototype: `The_400_Tasks/Task_by_Task/task387/prototypes/task387_absorb_batchW.onnx`
- Builder: `The_400_Tasks/Task_by_Task/task387/prototypes/build_absorb_batch_into_W.py`
- Cost/profile cost: `score_v2=(0,1906,700) -> 2606`; `profile=2606`
- SHA-256: `474ac7a55706f67dd7928343ffb99cb50a0833a1daee060fb4c8d2e1d7e7ce53`

## Bar used

Authoritative bar: `2607`, from:

```text
python tools/ng_current_bar.py --task 387 --json
authoritative_source=manifest
owner=Vamshi
sha256=f5ceadec7d5555c4de729447f7ea848f8cda1c27e4f47b272b0c1f61e0b1b7f2
updated_utc=2026-07-02T02:35:15Z
path=<TEAM_REPO>\Team_Submission\Backend\manifest.csv
```

Stale sources ignored: main-repo `Team_Best_Onnx` cost `2455` for the same SHA, task_state cost `3516`, `live_costs.json` cost `3842`, stale library DB cost `100793`.

Delta vs live bar: `2606 - 2607 = -1` byte saved.

## Named saving

Command:

```text
python tools/ng_graph_cost_diff.py --task 387 --live <TEAM_REPO>\Team_Submission\Backend\Team_Best_Onnx\task387.onnx --candidate The_400_Tasks/Task_by_Task/task387/prototypes/task387_absorb_batchW.onnx
```

Important output:

```text
removed initializer batch1: float16 [1] gone (-1 params)
shape initializer W: [6, 4] -> [1, 6, 4] (float16)
NET COST DELTA -1 -> candidate is CHEAPER
```

Note: this helper's absolute live/candidate totals printed `2455 -> 2454`, which is stale for this graph. The named structure delta matches measured `score_v2/profile` exactly: params `701 -> 700`, memory unchanged.

## Gates

```text
score: (0, 1906, 700) 2606
visible: [visible] 266/266 (100.00%)  PASS
profile: task387 task387_absorb_batchW.onnx: static=2606 profile=2606 (memory 1906 + params 700; 266 real inputs)
truth --n 2000: [truth] vs ARC-GEN ground truth: 2000/2000 (100.000%)
ab --n 2000: [ab] vs submission.zip: match 2266/2266 | divergences=0 | byte_equal=True (first_div=None)
truth --full: [truth] vs ARC-GEN ground truth: 200000/200000 (100.000%)
ab --full: [ab] vs submission.zip: match 200266/200266 | divergences=0 | byte_equal=True (first_div=None)
battery: === battery task387: PASS ===
[visible] 266/266 (100.00%)  PASS
[truth] vs ARC-GEN ground truth: 2000/2000 (100.000%)
[ab] vs submission.zip: match 2266/2266 | divergences=0 | byte_equal=True (first_div=None)
[score] cost=2606 (mem 1906+par 700)  points=17.134
```

## What changed

The live final decoder used `batch1[1]` only to introduce the singleton output batch axis:

```text
kt,tc,ku,ur,kv,vs,b->bcrs
```

I reshaped initializer `W` from `[6,4]` to `[1,6,4]`, rewrote the final equation to:

```text
bkt,tc,ku,ur,kv,vs->bcrs
```

and removed `batch1`. This preserves byte-equal outputs and deletes one charged initializer element.

## Tried and failed

- Did not re-run sparse-P, no-P, or rank<6 builds; `hard_records/freeoutput_einsum_render_2607_wall.md` already falsifies them with exact gates/costs.
- The requested rank-5/color-share collapse remains dead under that record: rows 2 and 3 share color but have distinct row/column outer products, so merging loses a spatial layer.
- No new dead-route hard record created; this pass found a small valid graph shave, not a new failed family.

## Remaining floor hypothesis

This is a staged live shave, not the ~1600 rank/factor floor breakthrough. The remaining floor lever is still the dense `P[18,30]` identity embed charged as 540 params. A legal way to encode P's shape/value without dense params and without a charged materialized tensor would reopen the large win; sparse-P and no-P are already dead under current checker/shape-inference/scorer behavior.

Lesson packet: none.
