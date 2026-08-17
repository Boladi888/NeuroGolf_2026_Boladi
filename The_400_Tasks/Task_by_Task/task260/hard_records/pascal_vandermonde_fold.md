# task260 hard record — Pascal/Vandermonde coefficient fold

- Live/team bar used: manifest `task260` cost 1063, sha256 `10e1101fff3e32dcfe0ca5ae56b9ab25bb3cd2d2c84956f4ad607c37c36bd973`; task-local staged rung `task260_sharedpow_cost866` cost/profile 866 remains the effective reviewer-ready bar.
- Locked reviewer route: starting from sharedpow, eliminate `coefPQ` without a dense `[7,7,7]` selector by retuning the terminal polynomial into a Pascal/Vandermonde convolution fold.
- Symbolic certificate: `prototypes/prove_pascal_vandermonde_fold.py` / `task260_pascal_vandermonde_proof.json`.
  - The Pascal tensor becomes length-7 truncated convolution after factorial/sign retuning.
  - CP/Vandermonde rank lower bound is `2n-1 = 13` terms.
  - Any exact independent coefficient/row/col factor fold needs at least `13*(7+7+7)=273` factor elements.
  - Versus sharedpow this removes `coefKIdx` 49 + `binomM` 49 params and `coefPQ` 196 B memory, so the best CP fold estimate is `273 - 98 - 196 = -21` profile bytes before numerical/gate risk.
- ONNX build attempted: `prototypes/task260_worker3_pascal_vandermonde.onnx`, built by `prototypes/build_task260_worker3_pascal_vandermonde.py`; params printed by builder: 601.
- Exact failed gate: `python tools/ng_gate.py visible --task 260 --cand The_400_Tasks/Task_by_Task/task260/prototypes/task260_worker3_pascal_vandermonde.onnx` returned `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: the exact algebraic CP fold exists and is small enough on paper, but its F32 Vandermonde realization is not visible-correct; reviewer requested immediate stop on visible failure, so no second build/profile/full gates were run.
- Remaining hypothesis: only a numerically stable non-Vandermonde realization of the same rank-13 convolution fold could revive this; it would still have the 273-element lower-bound cost and must clear visible before any profile/full gates.
