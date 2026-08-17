# task260 hard record — scaled F16 polynomial tail

- Live/team bar used: manifest `task260` cost 1063, sha256 `10e1101fff3e32dcfe0ca5ae56b9ab25bb3cd2d2c84956f4ad607c37c36bd973`; task-local staged rung `task260_sharedpow_cost866` cost/profile 866 is the effective reviewer-ready bar.
- Attempted route: evaluate the same degree-6 terminal polynomial in scaled coordinates `x/4`, move the no-wedge sentinel to `k=21` (`x=12`), and cast the terminal polynomial tail to F16 to halve `coefPQ`, `coef7`, and `sign` memory.
- Exact failed gate: `prototypes/task260_worker1_scaled_f16poly.onnx` scored `_score_v2 (0, 332, 430) total=762`, but `ng_gate visible` returned `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: although the profile target would be below 866 on paper, the F16/scaled-margin representation is not visible-correct. Per worker rule, a visible miss kills this representation family; do not tune epsilon/scale/sentinel variants without a materially different proof.
- One remaining hypothesis: keep the polynomial tail F32 and search only for exact algebraic selector/factor folds, or abandon polynomial-free-output for a genuinely different output family.
