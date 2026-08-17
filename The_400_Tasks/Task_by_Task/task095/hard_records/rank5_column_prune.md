# task095 rank-5 column-pruned parent route dead

- live bar used: 208, manifest/team-best sha256 `05651c20d66acf86fd6952b5887f6e9aed7f0de06a2a12ba19bedb4792f3fd73`, owner Vamshi, updated 2026-07-12T15:15:54Z.
- attempted route: keep the current single free-output Einsum equation and `F/C` channel factorization, then drop one column from `V[30,6]` to price at 178 params.
- candidate: `The_400_Tasks/Task_by_Task/task095/prototypes/task095_rank5_drop5_probe.onnx`, sha256 `7e02f27cfaa28451c49e23bf4c19afb2594ba57d7bc9d20fae55342fe6595aa3`.
- cost: static `score_network_v2(...) -> (0, 0, 178)`.
- local prefilter: best dropped column was column 5; generator-shaped numpy prefilter `official=152/265`, `fresh=610/1000`. Other dropped columns were worse (`0/265`, `0/265`, `1/265`, `1/265`, `9/265`).
- exact failed gate: `python tools/ng_gate.py visible --task 95 --cand The_400_Tasks/Task_by_Task/task095/prototypes/task095_rank5_drop5_probe.onnx --no-watch-queue` -> `[visible] 152/265 (57.36%)  FAIL`.
- why dead: a column-pruned rank-5 version of the current factorization destroys the spatial neighborhood margin. One visible miss is a certain hidden 0, so do not run truth/ab/profile-heavy gates or more dropped-column variants.
- one remaining hypothesis: a genuinely re-solved rank-5 factorization would need a new constrained fit/lower-bound argument, not a projection of the Vamshi parent.
