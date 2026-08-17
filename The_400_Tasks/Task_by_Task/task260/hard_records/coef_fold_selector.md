# task260 hard record — coefficient selector fold

- Live/team bar used: manifest `task260` cost 1063, sha256 `10e1101fff3e32dcfe0ca5ae56b9ab25bb3cd2d2c84956f4ad607c37c36bd973`; task-local staged rung `task260_sharedpow_cost866` cost/profile 866 is the effective reviewer-ready bar.
- Attempted route: replace `coefPQ = Gather(coef7, coefKIdx)` with a final-Einsum coefficient selector so the `[7,7]` coefficient tensor is not materialized.
- Exact failed gate:
  - dense selector `prototypes/task260_worker1_coef_densefold.onnx`: `[visible] 266/266 (100.00%)  PASS`; `_score_v2 (0, 244, 671) total=915`; `profile_real_cost static=915 profile=915 (memory 244 + params 671; 266 real inputs)`.
  - sparse selector: `onnx.shape_inference.infer_shapes(... strict_mode=True)` fails at final Einsum with `Rank of input 3 (0) does not match the equation indices (3)`.
- Why dead: dense `[7,7,7]` selector removes 196 B of `coefPQ` memory and 98 params of `coefKIdx/binomM`, but adds 343 params; net profile is +49 over the 866 staged rung. Sparse would have been the only sub-866 selector path, but the official shape-inference/scorer path sees sparse initializer inputs as rank 0.
- One remaining hypothesis: a non-selector algebraic factorization of the anti-diagonal coefficient map would need fewer than 98 added params and no charged `[7,7]` output; no such factorization was found here.
