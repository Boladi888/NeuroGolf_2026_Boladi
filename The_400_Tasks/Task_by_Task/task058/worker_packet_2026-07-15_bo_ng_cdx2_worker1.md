# task058 worker packet - bo_ng_cdx2_worker1 - 2026-07-15

## Bar used

Authoritative bar from `python tools/ng_current_bar.py --task 58 --json`: live/team-best cost 128, owner Vamshi, source `manifest`, updated 2026-07-02T02:35:15Z, sha256 `795ce7543fc7358b026acfaf9e6db16a21813c32d9f43c5dd9b7b1afd232eff9`.

Stale sources observed but not used as the bar: local `task_state.csv` cost 980, legacy `live_costs.json`/library cost 19,826.

Current team-best profile check:
`task058 task058.onnx: static=128 profile=128 (memory 16 + params 112; 22 real inputs)`

## Candidate/probe paths

No staged candidate and no manifest were produced.

Rejected probe: `The_400_Tasks/Task_by_Task/task058/prototypes/task058_basis20_probe.onnx`
- builder: `The_400_Tasks/Task_by_Task/task058/prototypes/build_task058_basis20_probe_20260715.py`
- candidate_sha256: `55e385b2302fa4e7041d11081c9970766a25f7e61d5db24c20959e53b77c3b44`
- static score: `None` from `score_network_v2`; graph inspection shows the intended tensors would be memory 16 + params 82 = 98 if ORT accepted the `Einsum` shape
- profile cost: not valid; `profile_real_cost.py` hit the same ORT `Einsum operands could not be broadcast together` runtime error
- cost delta vs live: no valid cost; intended static graph-inspection delta would have been `98 - 128 = -30`, but it is not scoreable
- reviewer-ready: no

Prior 127-byte probes read, not rerun:
- `task058_prefix_einsum127.onnx`, sha256 `a4bca924c72a9eeb7d9a1c144050e05e93a164b10205ea10565d481e05671d91`, static `(0, 0, 127)`, prior visible `[visible] 0/22 (0.00%)  FAIL`
- `task058_prefix_diag127.onnx`, sha256 `d4a925ee6946072d545f7f7ce5c7c00294213eb5e8f4f111f1fa762d6dae14bb`, static `(0, 0, 127)`, skipped by prior worker as same dead family

## Gates and exact result lines

Bar:
`python tools/ng_current_bar.py --task 58 --json` -> live_cost `128`, live_sha256 `795ce7543fc7358b026acfaf9e6db16a21813c32d9f43c5dd9b7b1afd232eff9`, authoritative_source `manifest`.

Static/profile baseline:
`Team_Best_Onnx/task058.onnx (0, 16, 112)`
`task058 task058.onnx: static=128 profile=128 (memory 16 + params 112; 22 real inputs)`

Candidate static:
`The_400_Tasks/Task_by_Task/task058/prototypes/task058_basis20_probe.onnx None`

Candidate visible:
`[visible] ERROR: InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Einsum node. Name:'output' Status Message: Einsum operands could not be broadcast together. Please check input shapes/equation provided.Input shape of operand 1 is incompatible in the dimension 0. The shape is: {20,3}Another operand has a dim value of 30 in the same dimension`

No truth/ab/battery/heavy gates were run on this candidate.

## Classification packet

- candidate_class: `PATH_B_SHAPE_TRUNCATION_REJECT_LOCAL`
- parent_source: `Team_Best_Onnx/task058.onnx`
- parent_sha256: `795ce7543fc7358b026acfaf9e6db16a21813c32d9f43c5dd9b7b1afd232eff9`
- candidate_sha256: `55e385b2302fa4e7041d11081c9970766a25f7e61d5db24c20959e53b77c3b44`
- proof_commands_run: `ng_current_bar`, `inspect_graph.py`, `profile_real_cost.py` on parent, `score_network_v2`, `ng_gate visible` on candidate
- proof_results: visible runtime error, static scorer `None`, no valid profile cost
- safety: `null` (`ng_safety.py` not run because the candidate did not pass visible and did not beat the bar with a valid score)
- x86_required: `false`
- kaggle_probe_required: `false`
- status: `REJECT_LOCAL`

## What was tried and failed

1. Active-max coordinate truncation: current analytic graph stores a `[30,3]` coordinate basis even though ARC-GEN size is 5..20. Truncating it to `[20,3]` would have intended cost 98, but ORT rejects the final `Einsum` because the shared spatial label also appears on the `[1,10,30,30]` input. Hard record: `The_400_Tasks/Task_by_Task/task058/hard_records/basis20_einsum_broadcast.md`.
2. Generated zero-param coordinate basis: `Range`/`Pow`/`ConstantOfShape` would delete 90 stored params but materializes hundreds of charged runtime bytes before the final `Einsum`. Hard record: `The_400_Tasks/Task_by_Task/task058/hard_records/generated_basis_price_floor.md`.
3. Diagonal-prefix one-Einsum idea: using `nyuu` diagonal input features could price at 127, but a linear separability check over all generator sizes 5..20 was infeasible for the current `q=-0.75` basis and nearby one-parameter grids; no ONNX was emitted.

## Remaining hypothesis

The live 128-byte graph is close to the analytic floor: 90 coordinate-basis params + 9 coeff params + 10 channel-sign params + 3 phase-base params + 16 scalar/vector memory. A sub-128 win likely needs an exact single-output formulation that removes the 16-byte `ReduceL2`/`Pow` phase memory without adding a dense selector, or a legal way to encode the sparse 10-channel sign vector/3x3 coefficient matrix without sparse initializers.
