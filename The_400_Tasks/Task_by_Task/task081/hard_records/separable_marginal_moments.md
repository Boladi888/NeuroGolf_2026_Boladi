# task081 hard record - separable marginal/moment route

- live bar used: cost 395, sha256 `455d15b1b8ab854784d8462e31b376b7bd028655ab69505c6aeba85beeb663b2` from `python tools/ng_current_bar.py --task 81 --json` (authoritative source: manifest, owner Boladi, updated 2026-06-22T16:50:27Z).
- attempted route: collapse the fixed 7x7 cyan evidence into row/column marginals or low-degree row/column moments before building any dense per-cell output tail.
- probe artifact: `The_400_Tasks/Task_by_Task/task081/prototypes/probe_marginal_moment_infeasible.py`; it writes `task081_marginal_skeleton.onnx`.
- exact failed probe: `python The_400_Tasks/Task_by_Task/task081/prototypes/probe_marginal_moment_infeasible.py` produced a fresh ARC-GEN row/column-count collision at sample 31: identical degree-0 signature `(0,0,2,1,2,2,2,1,2,0,2,1,1,2)` required different blue-hole maps.
- skeleton price: emitted degree-0 marginal skeleton scored `memory=252 params=8 cost=260` before any output tail. Analytic pre-tail lower bounds were degree 0 = 260, degree 1 = 330, degree 2 = 400.
- why dead: raw row/column counts are not functionally sufficient; degree-2 moments already exceed the 395 live bar before reconstructing the 7x7 one-hot output, and degree-1 leaves only 65 bytes for the entire 2D reconstruction tail, which is below the known cost of any legal boundary-safe tail for this task.
- remaining hypothesis: a non-marginal row-bitset/bitwise route might still be worth a separate proof if it can compute row hole bitmasks with very few vector intermediates and use a free-output bitwise final, but it is a different formulation from the separable moment route and must beat the current `x8/cyan_u8/two-feature` 395 wall.
