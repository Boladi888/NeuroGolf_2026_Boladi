# task095 worker packet - bo_ng_cdx2_worker2

## Bar
- authoritative command: `python tools/ng_current_bar.py --task 95 --json`
- bar source: manifest, owner Vamshi, updated 2026-07-12T15:15:54Z
- live/team-best: `Team_Best_Onnx/task095.onnx`
- live cost/profile: 208/208
- live sha256: `05651c20d66acf86fd6952b5887f6e9aed7f0de06a2a12ba19bedb4792f3fd73`
- live delta baseline: 0

## Parent Inspection
- graph: single `Einsum`, equation `nkhw,ha,ra,wb,cb,ks,nprc,pq,oe,qes->norc`
- params: `V[30,6]=180`, `F[10,2]=20`, `C[2,2,2]=8`
- profile command: `python tools/agent_helpers/profile_real_cost.py --task 95 --cand Team_Best_Onnx/task095.onnx --arcgen 266`
- profile result: `task095 task095.onnx: static=208 profile=208 (memory 0 + params 208; 265 real inputs)`
- visible command: `python tools/ng_gate.py visible --task 95 --cand Team_Best_Onnx/task095.onnx`
- visible result: `[visible] 265/265 (100.00%)  PASS`

## Probes
- rule probe: `python The_400_Tasks/Task_by_Task/task095/prototypes/task095_postmerge_probe.py`
- result: `numpy_rule_arcgen=3000/3000`
- parent local formula prefilter: official `265/265`, fresh `997/1000`

### Sparse V
- candidate_class: `SPARSE_INIT_PROBE`
- path: `The_400_Tasks/Task_by_Task/task095/prototypes/task095_sparseV_probe.onnx`
- candidate_sha256: `a31a8a70bf504f02033eeaf3286fd9d2dfa93e6cb782a04450ccd8491427c91b`
- apparent cost if accepted: 82 params; official cost: invalid / `None`
- proof command: `score_network_v2(...)`
- proof result: `None`
- diagnostic: strict shape inference fails with `(op_type:Einsum): [ShapeInferenceError] Rank of input 1 (0) does not match the equation indices (2).`; raw ORT load passed.
- safety: not run; invalid scorer path
- x86_required: false
- kaggle_probe_required: false
- status: `REJECT_LOCAL`

### Rank-5 Column Prune
- candidate_class: `PATH_B_FACTOR_PROJECTION`
- path: `The_400_Tasks/Task_by_Task/task095/prototypes/task095_rank5_drop5_probe.onnx`
- candidate_sha256: `7e02f27cfaa28451c49e23bf4c19afb2594ba57d7bc9d20fae55342fe6595aa3`
- cost/profile: static 178; profile not run after visible fail
- proof command: `python tools/ng_gate.py visible --task 95 --cand The_400_Tasks/Task_by_Task/task095/prototypes/task095_rank5_drop5_probe.onnx --no-watch-queue`
- proof result: `[visible] 152/265 (57.36%)  FAIL`
- safety: not run; visible fail
- x86_required: false
- kaggle_probe_required: false
- status: `REJECT_LOCAL`

## Staging
- no staged candidate; nothing beat the 208 team-best bar.
- manifest path: none.

## Breadcrumbs
- `The_400_Tasks/Task_by_Task/task095/hard_records/sparse_v_einsum.md`
- `The_400_Tasks/Task_by_Task/task095/hard_records/rank5_column_prune.md`

## Remaining Hypothesis
A true sub-208 result likely needs a newly solved lower-rank factorization rather than column-pruning, or a non-sparse way to avoid paying the 21 forced zero rows of `V[30,6]`. Any route adding a real intermediate is priced out against the 208 single-node parent.
