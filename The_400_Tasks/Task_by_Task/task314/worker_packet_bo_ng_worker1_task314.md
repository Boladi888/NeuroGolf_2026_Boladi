# task314 worker packet — bo_ng_worker1

## Current bar used
- Authoritative command: `python tools/ng_current_bar.py --task 314 --json`
- Source: manifest, `K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/manifest.csv`
- Live/team-best cost: 100
- Live/team-best SHA-256: `91294db3a76db1cf55efd0eeb6e6c475503684a7d282ec1b5611be5a48729242`
- Owner/update: Vamshi, `2026-07-12T15:15:54Z`

## Parent inspection
- Parent source: `Team_Best_Onnx/task314.onnx`
- Parent graph: one `Conv` node writing `output`, grouped depthwise, `weights [10,1,3,3]` + `bias [10]`
- Parent profile command: `python tools/agent_helpers/profile_real_cost.py --task 314 --cand Team_Best_Onnx/task314.onnx --arcgen 266`
- Parent profile result: `task314 task314.onnx: static=100 profile=100 (memory 0 + params 100; 266 real inputs)`
- Parent visible command: `python tools/ng_gate.py visible --task 314 --cand Team_Best_Onnx/task314.onnx`
- Parent visible result: `[visible] 266/266 (100.00%)  PASS`

## Probe candidate
- Candidate path: `The_400_Tasks/Task_by_Task/task314/prototypes/task314_conv_no_bias.onnx`
- Builder path: `The_400_Tasks/Task_by_Task/task314/prototypes/build_task314_conv_no_bias.py`
- Manifest path: none; not staged
- Candidate static cost/profile cost: estimated static 90 from graph params; profile not run because visible failed
- Candidate-live delta: `90 - 100 = -10` estimated static bytes, invalidated by visible failure
- Candidate SHA-256: `e20eb50ac2885a6a0e0376964f572fabe9772ea532e4538486585cc5abc22d8d`
- Candidate class: `LOCAL_PROBE_CONV_NO_BIAS`
- Parent source: `Team_Best_Onnx/task314.onnx`
- Parent SHA-256: `91294db3a76db1cf55efd0eeb6e6c475503684a7d282ec1b5611be5a48729242`
- Candidate SHA-256: `e20eb50ac2885a6a0e0376964f572fabe9772ea532e4538486585cc5abc22d8d`
- x86_required: false
- kaggle_probe_required: false
- status: `REJECT_LOCAL`

## Proof commands run
- `python The_400_Tasks/Task_by_Task/task314/prototypes/build_task314_conv_no_bias.py`
- `python tools/agent_helpers/inspect_graph.py The_400_Tasks/Task_by_Task/task314/prototypes/task314_conv_no_bias.onnx`
- `python tools/ng_gate.py visible --task 314 --cand The_400_Tasks/Task_by_Task/task314/prototypes/task314_conv_no_bias.onnx`

## Proof results
- Candidate graph: one `Conv` node writing `output`; one initializer `weights F32 [10,1,3,3] params=90`; no value_info intermediates.
- Candidate visible gate: `[visible] 0/266 (0.00%)  FAIL`
- No `truth`, `ab`, `battery`, `profile_real_cost`, `ng_graph_cost_diff`, or `ng_safety` was run on the candidate after visible failure.

## Safety
`ng_safety.py` JSON object: null

Reason: safety is only meaningful for reviewer-ready cost wins; this probe failed the first gate and is not stageable.

## Tried and failed
Removed the Conv bias from the current one-node grouped Conv. This would have removed 10 params, but the visible gate went to 0/266. The negative per-channel threshold is load-bearing for the fill predicate.

Dead-route breadcrumb: `The_400_Tasks/Task_by_Task/task314/hard_records/no_bias_conv.md`

## Remaining floor/rewrite hypothesis
The live 100 graph is already a one-node free-output depthwise Conv with zero memory. A safe sub-100 route needs a way to keep the threshold with fewer than 10 charged elements, or a different final op with an implicit threshold. The known subdeclared Conv-bias shortcut is not a candidate because routed traps classify it as unsafe/visible-unstable. Free-output Einsum-style rewrites appear unlikely below 100 because spatial shifts would need dense selector matrices, while Conv encodes the four dilation-3 neighbor reads in 9 kernel params per channel.
