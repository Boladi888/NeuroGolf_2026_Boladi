# task314 worker packet — bo_ng_cdx_worker2 reviewer rework

## Current bar used
- Command: `python tools/ng_current_bar.py --task 314 --json`
- Authoritative source: manifest, `K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/manifest.csv`
- Team-best/live cost: 100
- Team-best/live SHA-256: `91294db3a76db1cf55efd0eeb6e6c475503684a7d282ec1b5611be5a48729242`
- Owner/update: Vamshi, `2026-07-12T15:15:54Z`

## Reviewer demand answered
Reviewed and cited:
- `Onnx_Advanced/INDEX.md`
- `techniques.md #67`: output node is free; make the expensive op write `output`
- `techniques.md #89`: Einsum-final in-kernel temporaries are scorer-invisible; split only for large latency wins
- `by_symptom.md` free-output family: splitting a free-output Conv can materialize a charged intermediate
- `techniques.md #87`: free-output separability trap; selector-bank rank/cost kill test first
- `cost_floor_reference.md`: params are element-count and dtype-agnostic; generated tensors charge unless absorbed in output/free op

Conclusion: the locked single-Einsum route is infeasible below 100. The first nonseparable state is background suppression: endpoint products must both emit their same non-background channel and turn off output channel 1.

## Fresh ARC-GEN separability probe
- Probe path: `The_400_Tasks/Task_by_Task/task314/prototypes/task314_einsum_separability_probe.py`
- Probe JSON: `The_400_Tasks/Task_by_Task/task314/prototypes/task314_einsum_separability_probe.json`
- Probe command: `python The_400_Tasks/Task_by_Task/task314/prototypes/task314_einsum_separability_probe.py`
- Fresh rows: 64 ARC-GEN pairs, `seed_offset=731400`
- Results:
  - `fill_cells`: 113
  - `bg_suppression_cells`: 113
  - `hv_conflict_cells`: 0
  - center+endpoint products without background suppression: `10/64`
  - with background suppression: `64/64`
  - with background suppression and vertical priority: `64/64`

## Selector skeleton
- Skeleton path: `The_400_Tasks/Task_by_Task/task314/prototypes/task314_einsum_channel_route_skeleton.onnx`
- Skeleton builder: `The_400_Tasks/Task_by_Task/task314/prototypes/build_task314_einsum_channel_route_skeleton.py`
- Manifest path: none; not staged
- Skeleton SHA-256: `bb133e249723321a85e041eff4a03f45360c7991db524055b607d98aaa741215`
- Skeleton class: `SELECTOR_SKELETON_FLOOR_PROOF`
- Skeleton graph: one final `Einsum("bqhw,kq->bkhw")`, one `channel_route [10,10]`
- Skeleton static/profile cost: 100 / 100
- Skeleton-live delta: `100 - 100 = 0`; not strictly below bar

## Proof commands run
- `python tools/ng_current_bar.py --task 314 --json`
- `python Onnx_Advanced/library_db/lib_query.py status`
- `python Onnx_Advanced/library_db/lib_query.py context task314 --limit 8`
- `python Onnx_Advanced/library_db/lib_query.py search "task314 SeparableMomentCore" --limit 8`
- `python -B -m neurogolf_provider.cli context task314`
- `python -B -m neurogolf_provider.cli consistency task314`
- `python -B -m neurogolf_provider.cli search "task314 SeparableMomentCore"`
- `python The_400_Tasks/Task_by_Task/task314/prototypes/task314_einsum_separability_probe.py`
- `python The_400_Tasks/Task_by_Task/task314/prototypes/build_task314_einsum_channel_route_skeleton.py`
- `python tools/agent_helpers/inspect_graph.py The_400_Tasks/Task_by_Task/task314/prototypes/task314_einsum_channel_route_skeleton.onnx`
- `python tools/agent_helpers/profile_real_cost.py --task 314 --cand The_400_Tasks/Task_by_Task/task314/prototypes/task314_einsum_channel_route_skeleton.onnx --arcgen 266`

## Proof results
- Library/provider consistency: manifest cost 100; legacy `live_costs` 2889 and library 8594 are stale/context only.
- Probe: background suppression is required on fresh generator rows.
- Channel route matrix for one endpoint product has dense shape `[10,10]`, dense params 100, rank 8, rank-factor lower-bound params 160.
- Skeleton inspect: `n_nodes: 1`, `n_inits: 1`, `channel_route F32 [10,10] params=100`, no static intermediates.
- Skeleton profile: `task314 task314_einsum_channel_route_skeleton.onnx: static=100 profile=100 (memory 0 + params 100; 266 real inputs)`
- No finalized ONNX candidate was built after skeleton pricing hit the live bar; no `visible`, `truth`, `ab`, `battery`, `ng_graph_cost_diff`, or `ng_safety` was run for a reviewer-ready candidate.

## Candidate fields
- candidate_class: `SELECTOR_SKELETON_FLOOR_PROOF`
- parent_source: `Team_Best_Onnx/task314.onnx`
- parent_sha256: `91294db3a76db1cf55efd0eeb6e6c475503684a7d282ec1b5611be5a48729242`
- candidate_sha256: `bb133e249723321a85e041eff4a03f45360c7991db524055b607d98aaa741215`
- safety: null
- x86_required: false
- kaggle_probe_required: false
- status: `REJECT_LOCAL`

## Floor statement
The locked #89 contraction needs the channel route
`q>1 endpoint_product -> output q (+1) and output channel 1 (-1)`. A dense operand for that route costs 100 params, and low-rank factorization is worse at 160 params minimum. Since #67/#89 only make the final output free, not initializer params, this reaches the 100-byte team-best before adding any row/column/cell-phase endpoint selectors or center-copy machinery. The route cannot produce a profile cost strictly below 100.

Dead-route breadcrumb: `The_400_Tasks/Task_by_Task/task314/hard_records/einsum_channel_route_floor.md`
