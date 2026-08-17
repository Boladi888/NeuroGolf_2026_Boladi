# axiscode_capacity6_rework dead route

Live bar used: cost/profile cost 190, sha256 `d8a937d60fabb9a7e847cf1ec19fa2425e7157607022d94bb3cf51fb0e8e50a2` from `python tools/ng_current_bar.py --task 287 --json` / manifest.

Reviewer-locked route: shrink the live final-Einsum mirror-class axis basis. Live charged tensors are `low[30,4]` = 120 params, `high[30,2]` = 60 params, and `mask[10]` = 10 params; total 190, zero memory.

Attempted route: replace `low[30,4] + high[30,2]` with a smaller 3x2 shared product code: `low3[30,3]` = 90 params plus `high2[30,2]` = 60 params, keeping `mask[10]`. Candidate cost would be 160 if correct.

Prototype: `The_400_Tasks/Task_by_Task/task287/prototypes/task287_axiscode_3x2_capacity6_probe.onnx`; builder/proof script: `The_400_Tasks/Task_by_Task/task287/prototypes/build_task287_axiscode_capacity6_probe.py`; proof JSON: `The_400_Tasks/Task_by_Task/task287/prototypes/task287_axiscode_capacity6_proof.json`.

Exact gates:
- builder/checker/proof: `valid_found=False checked=4111`; `saved ... task287_axiscode_3x2_capacity6_probe.onnx bytes=944`
- visible: `[visible] 0/267 (0.00%)  FAIL`

Why dead: the output has 8 distinct mirror classes per axis. The proof enumerated every canonical <=6-bin shared class merge over the 267 visible output class signatures and found no valid mapping. The representative 3x2 product-code graph visible-failed immediately, so smaller shared one-hot/product axis-code bases are dead under the visible-fail axiom. Profile/truth/ab were not run.

One remaining hypothesis: only a non-product continuous/polynomial terminal basis that represents the 8-class equality relation with fewer than the live 6 total axis columns could revive this route. The standard one-hot product-code family needs capacity >=8, whose minimum total factor width is 6 (`4+2`, `3+3`, or `2+2+2`), so it cannot beat 190 once `mask[10]` is included.
