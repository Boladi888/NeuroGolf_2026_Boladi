# DEAD ROUTE - task196 free-output QLinearConv tail

Recorded by `bo_ng_worker1` for the reviewer rework on task196.

## Live bar

- Bar command: `python tools/ng_current_bar.py --task 196 --json`
- live_cost: **3167**
- live/team-best path: `Team_Best_Onnx/task196.onnx`
- live/team-best sha256: `e6cbb4d49482e5a4712ade0076a9059ea273598306b62d7a8857ef2ea956745c`
- source: authoritative manifest row, owner Boladi, updated 2026-07-14T00:05:59Z

## Attempted route

Reviewer-locked hypothesis: keep the K=2 detector unchanged and replace only
`Min(paint,one) -> Cast -> Pad(cond) -> Where(cond,onehot3,input)` with a
free-output QLinearConv placer-style tail per techniques #85/#67.

Built exactly one variant:

- builder: `The_400_Tasks/Task_by_Task/task196/prototypes/build_task196_freeout_tail_probe.py`
- candidate: `The_400_Tasks/Task_by_Task/task196/prototypes/task196_freeout_tail_active15.onnx`
- sha256: `fd0f0b419fee92a5e5a78df6c0310643df315a5c5f2fc797fc5d9ac48917439c`

Tail form:

```text
Concat(active15, one, paint) -> QLinearConv(output)
```

The final QLinearConv writes U8 `[1,10,30,30]` directly as graph output. The 15x15 active mask is a
225-param U8 initializer, not a 30x30 carrier.

## Fast probe results

Static/profile cost:

```text
score_network_v2: (0, 2363, 378) => 2741
python tools/agent_helpers/profile_real_cost.py --task 196 --cand ... --arcgen 266
task196 task196_freeout_tail_active15.onnx: static=2741 profile=2741 (memory 2363 + params 378; 266 real inputs)
```

Largest intermediate:

```text
Total intermediate bytes (static value_info): 2,363
rank         bytes    pct   dtype  shape                      producer         name
   1           900   38.1     F32  [1, 1, 15, 15]             Slice            one_f
   2           675   28.6   UINT8  [1, 3, 15, 15]             Concat           tail_in
   3           338   14.3   UINT8  [1, 2, 13, 13]             QLinearConv      det
   4           225    9.5   UINT8  [1, 1, 15, 15]             Cast             one
   5           225    9.5   UINT8  [1, 1, 15, 15]             QLinearConv      paint
```

This proves the 900 B `[1,1,30,30]` `cond` plane is gone, and profile cost is strictly below 3167.

Named cost delta:

```text
python tools/ng_graph_cost_diff.py --task 196 --live Team_Best_Onnx/task196.onnx --candidate ... --top 30
live memory/params/cost      3038 / 129 / 3167
candidate memory/params/cost 2363 / 378 / 2741
delta                        -675 / +249 / -426
```

Main charged changes:

- removed `cond` BOOL `[1,1,30,30]`: `-900 B`
- added `tail_in` U8 `[1,3,15,15]`: `+675 B`
- removed `ring` and `loop` 15x15 planes: `-450 B`
- added `active15` initializer: `+225 params`
- removed `onehot3`, `padv`, `padv_axes`; added `wtail`, `btail`

## Failed gate

```text
python tools/ng_gate.py visible --task 196 --cand The_400_Tasks/Task_by_Task/task196/prototypes/task196_freeout_tail_active15.onnx
[visible] 89/266 (33.46%)  FAIL
```

No truth, AB, battery, or safety gates were run after the visible fail.

## Why it is dead

The free-output QLinearConv tail can cheaply write channels 1 and 3 from `one` and `paint`, but it
also has to write channel 0 only inside the true active grid. Task196 visible/corpus active sizes
are variable: 9x9, 12x12, and 15x15. A fixed 15x15 active carrier over-paints channel 0 in padding
for the smaller grids. The exact repair requires copying/deriving the active/background mask from
the F32 input, e.g. a channel-0 crop/cast carrier, which adds at least another 900 B F32 slice before
the U8 carrier and pushes the route above the bar / violates the reviewer stop condition on
reintroduced 900 B carriers.

## Remaining hypothesis

Only revisit this if a zero/low-cost active-grid mask becomes available from an already-paid tensor.
With the current K=2 detector tensors (`one`, `det`, `paint`) there is no information to distinguish
background cells inside a 9x9/12x12 active grid from padding cells inside the fixed 15x15 crop.
