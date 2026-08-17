# task095 worker packet - bo_ng_cdx2_worker3

## Reviewer Request Answered
- addressed locked hypothesis only: dense d=5 re-solve of current single-node free-output F32 Einsum parent (#89/#109/#119).
- did not retry sparse V.
- did not try output-tail, crop, Pad, Equal, or Where rewrites.
- used exactly three d=5 solver initializations/formulations.
- ONNX build condition was not met; no candidate was staged.

## Bar
- authoritative command: `python tools/ng_current_bar.py --task 95 --json`
- bar source: manifest
- live/team-best: `Team_Best_Onnx/task095.onnx`
- live cost/profile: 208/208
- live sha256: `05651c20d66acf86fd6952b5887f6e9aed7f0de06a2a12ba19bedb4792f3fd73`
- owner: Vamshi
- updated: 2026-07-12T15:15:54Z

## Parent
- parent_source: `Team_Best_Onnx/task095.onnx`
- parent_sha256: `05651c20d66acf86fd6952b5887f6e9aed7f0de06a2a12ba19bedb4792f3fd73`
- graph: one `Einsum`, equation `nkhw,ha,ra,wb,cb,ks,nprc,pq,oe,qes->norc`
- parent params: `V[30,6]=180`, `F[10,2]=20`, `C[2,2,2]=8`, total 208

## Locked d=5 Probe
- solver path: `The_400_Tasks/Task_by_Task/task095/prototypes/task095_rank5_resolve.py`
- summary path: `The_400_Tasks/Task_by_Task/task095/prototypes/task095_rank5_resolve_summary.json`
- candidate_class: `PATH_B_DENSE_RANK5_SINGLE_EINSUM_SOLVE`
- intended cost if prefilter had passed: 178 (`V[30,5]=150`, `F[10,2]=20`, `C[2,2,2]=8`)
- profile cost: not run; no ONNX built
- candidate path: none
- candidate_sha256: null
- manifest path: none
- cost delta vs live: not claimable; intended class would be `178 - 208 = -30`, but no candidate passed prefilter

## Proof Commands Run
- `python tools/ng_current_bar.py --task 95 --json`
- `python tools/agent_helpers/inspect_graph.py Team_Best_Onnx/task095.onnx`
- `python The_400_Tasks/Task_by_Task/task095/prototypes/task095_rank5_resolve.py`

## Proof Results
- bar: manifest live cost 208, sha256 `05651c20d66acf86fd6952b5887f6e9aed7f0de06a2a12ba19bedb4792f3fd73`
- fresh decoded-rule check inside solver: `3000/3000`
- attempt 1 `eigh5_parent_fc`: best official `265/265`, fresh `2999/3000`, step 160, loss `9.337270422007804e-05`
- attempt 2 `eigh5_parent_fc_noisy`: best official `265/265`, fresh `2998/3000`, step 1080, loss `6.093712941567679e-05`
- attempt 3 `random_dense_vfc`: best official `0/265`, fresh `0/3000`, step 1, loss `709.884903931502`
- ONNX build result: `no_onnx_built=prefilter_did_not_reach_265_and_3000`
- ng_gate candidate gates: none, because reviewer required prefilter pass before ONNX build

## Safety and Status
- safety: null
- x86_required: false
- kaggle_probe_required: false
- status: `REJECT_LOCAL`

## Breadcrumb
- infeasibility note: `The_400_Tasks/Task_by_Task/task095/hard_records/rank5_resolve_reviewer_return.md`
