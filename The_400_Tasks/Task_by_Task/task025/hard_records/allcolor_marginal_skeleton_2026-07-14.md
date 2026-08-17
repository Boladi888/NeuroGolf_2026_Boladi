# task025 — all-color marginal skeleton route — REJECT_LOCAL — 2026-07-14

## Live bar used
- Authoritative bar from `python tools/ng_current_bar.py --task 25 --json`:
  **8834** bytes, owner **Sho**, manifest source, updated
  `2026-07-09T01:09:17Z`.
- Live SHA-256:
  `e0d68f3dae9effa13145415702900b856a159bd153ff1692c4705d800e72802b`.
- Re-profile command:
  `python tools/agent_helpers/profile_real_cost.py --task 25 --cand Team_Best_Onnx/task025.onnx --arcgen 266`
  → `static=8834 profile=8834 (memory 8430 + params 404; 266 real inputs)`.
- Live visible check:
  `python tools/ng_gate.py visible --task 25 --cand Team_Best_Onnx/task025.onnx`
  → `[visible] 266/266 (100.00%)  PASS`.

## Attempted route
Route name: **all-color marginal side-membership core**.

Instead of Sho's TopK-4 slot representation, compute line masks for all 10
colors and both orientations:
- `row_count_f32`, `col_count_f32`: all-color `[1,10,30]` row/column
  marginals.
- `row_line_b`, `col_line_b`: full-line masks by comparing marginals to
  active width/height.
- `v_side_counts_f32`, `h_side_counts_f32`: same-color source-exists-on-side
  descriptors from the input, the all-color line masks, and a side relation
  matrix.

Prototype builder:
`The_400_Tasks/Task_by_Task/task025/prototypes/build_task025_allcolor_marginal_skeleton.py`

Skeleton ONNX:
`The_400_Tasks/Task_by_Task/task025/prototypes/task025_allcolor_marginal_skeleton.onnx`

Skeleton SHA-256:
`cd270b4aaff07215e0e4ea3c33397444262ea5d52b9b228f1e7ba80c34b357f1`

This graph is intentionally **not a candidate**. It has a scalar dummy output so
the pre-render descriptor tensors remain charged; no output tail is included.

## Fast numpy/marginal check
Fresh ARC-GEN sample generated in-memory with seed offset 9025:

`fresh_arcgen_seed9025 numpy_ref=200/200`

Line/color stats:
- `line_count_hist={1: 9, 2: 80, 3: 77, 4: 34}`
- `distinct_line_color_hist={1: 9, 2: 80, 3: 77, 4: 34}`
- `max_same_color_sources_on_one_side=1`

Conclusion: the all-color marginal representation is semantically aligned on
fresh generator samples; this route dies on cost, not rule mismatch.

## Exact failed gate / price check
Profile command:

`python tools/agent_helpers/profile_real_cost.py --task 25 --cand The_400_Tasks/Task_by_Task/task025/prototypes/task025_allcolor_marginal_skeleton.onnx --arcgen 266`

Result:

`task025 task025_allcolor_marginal_skeleton.onnx: static=13692 profile=13692 (memory 11880 + params 1812; 266 real inputs)`

Largest charged tensors:
- `v_side_counts_f32 [1,10,2,30]` = 2400 B
- `h_side_counts_f32 [1,10,2,30]` = 2400 B
- `row_count_f32 [1,10,30,1]` = 1200 B
- `col_count_f32 [1,10,1,30]` = 1200 B
- `row_line_f32 [1,10,30,1]` = 1200 B
- `col_line_f32 [1,10,1,30]` = 1200 B
- side relation initializer `[2,30,30]` = 1800 params

The pre-render core alone is **13,692**, already **+4,858** over the live
8,834-byte bar. Adding any legal render tail would only increase cost. Per the
gate ladder, no visible/truth/ab candidate gates were run on this skeleton.

## Why the route is dead
Sho's graph pays for only 4 color slots: two F32 `[1,4,30]` position encoders
and a compact rank-26 render. The all-color route replaces that with full
`[10,30]` F32 marginals and side descriptors. Even before rendering it costs
more than the entire incumbent.

Do not re-explore this route unless the new formulation avoids **both**:
- materializing all-color `[1,10,30]` F32 row/column marginals, and
- materializing all-color `[1,10,2,30]` side-membership descriptors.

## One remaining hypothesis
A cheaper-than-8834 route would need a different slot detector that preserves
Sho's 4-line 99.84% behavior but computes line position plus side membership in
less than the incumbent's two F32 `[1,4,30]` encoders and decode soup. This pass
found no concrete encoding for that; all-color marginals are strictly too large.
