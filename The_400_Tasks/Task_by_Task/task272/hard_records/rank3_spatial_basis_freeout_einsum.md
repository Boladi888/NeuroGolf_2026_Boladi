# task272 hard record: rank-3 spatial-basis free-output Einsum

Live/team-best bar: cost 148, sha256 `7dec9a3acc9fe49a7aa493e027a894192cb626ba18b927095d953e98299de7c3`, source `python tools/ng_current_bar.py --task 272 --json` / team manifest, owner Vamshi, updated 2026-07-12T15:15:54Z.

Attempted route: replace the current single-node rank-4 free-output Einsum's exact 5-row/5-col path-adjacency basis `R[2,2,30]` (120 params) with a strict-below-bar rank-3 symmetric eigentruncation `R3[3,30]` plus `W3[3]`, reusing the current channel selector `S[2,10]` and logic tensor `Q[2,2,2]`. Probe path: `The_400_Tasks/Task_by_Task/task272/prototypes/task272_bo_ng_worker1_rank3_einsum.onnx`, sha256 `5f04d1cf6eaa115b90aed460099ac6e6455887b4d6cf4a6c459cfbc8bf3e0b5a`. Static charged dims are 121 params and zero intermediates by graph inspection.

Exact failed gate: `python tools/ng_gate.py visible --task 272 --cand The_400_Tasks/Task_by_Task/task272/prototypes/task272_bo_ng_worker1_rank3_einsum.onnx --max-seconds 180` returned `[visible] 102/267 (38.20%)  FAIL`. A first shell attempt with a 120s shell timeout preempted `ng_gate`; the recorded gate above is the controlled foreground `ng_gate` run.

Why dead: the documented rule passed a fresh ARC-GEN numpy probe (`3000/3000 PASS`, all 3x3 through 5x5 shape combinations present), and the 5-node path adjacency matrix needed for orthogonal-neighbor detection has rank 4. Dropping to rank 3 deletes one required eigenmode, so the vertical/horizontal neighbor predicate is no longer exact on visible officials. Per the visible-fail axiom, this candidate is a certain hidden zero; no truth/AB/profile/safety gates were run.

One remaining hypothesis: a sub-148 route would need to avoid paying the zero-padded 30-wide rank-4 spatial basis without `sparse_initializer` and without materializing a charged generated basis. No such legal output-node formulation was found in this pass.
