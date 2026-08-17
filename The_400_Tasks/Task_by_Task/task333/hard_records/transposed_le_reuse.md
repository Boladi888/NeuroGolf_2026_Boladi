# transposed_le_reuse hard record

- Live/team-best bar: cost/profile 635, sha256 `bb14c2aac494c18167f7eab7cc1ac124e94634011c4d2e81da68a8acb1321c61`, source `NG_OnnxlyFans_TaskStatus.xlsx` row `task333` and `Team_Submission/Backend/manifest.csv` row `task333` refreshed 2026-07-03. No `Staged_Boladi/task333.json` pending.
- Reviewer-locked route: start from Vamshi's one-Einsum graph; replace `relation_feature[2,10,10]` with one dense `le[1,10,10]`; rewrite relation uses to consume `le(a,b)` and `le(b,a)`.
- Built candidate: `The_400_Tasks/Task_by_Task/task333/prototypes/task333_transposed_le_eqprod_probe.onnx` via `prototypes/build_task333_transposed_le_probe.py`.
- Cost gate: `score_network_v2` `(0, 0, 526) 526`; profile `task333_transposed_le_eqprod_probe.onnx: static=526 profile=526 (memory 0 + params 526; 265 real inputs)`.
- Failed gate: visible `[visible] 0/265 (0.00%)  FAIL`.
- Why this route is dead below 635: with one dense `le` plane and no second coordinate selector plane, a single Einsum relation test can only use one fixed monomial per coordinate pair: `1`, `le(a,b)`, `le(b,a)`, or `le(a,b)*le(b,a)` (equality). Strict directions require two monomials:
  - `a < b = le(a,b) - le(a,b)*le(b,a)`
  - `a > b = le(b,a) - le(a,b)*le(b,a)`
  A fixed one-monomial relation leaks equality or misses strict positions. The built equality-product monomial proves the cost-valid transposed reuse collapses the strict-direction signal and fails every visible pair.
- Why the obvious repairs cannot remain below 635: selecting `le`, transposed `le`, and product terms inside one Einsum needs either a second 10x10 coordinate plane such as `[one, le]` or `[le, le.T]` (at least 100 params added back, giving a tie or worse before any decoder change), or a materialized relation/output partial with at least hundreds to thousands of charged memory bytes. Both erase the 109-byte saving from the 526 probe.
- One remaining hypothesis: only a different dense factorization that reduces a small core or the `active_map` without losing strict direction can beat 635; do not retry one-plane transposed-`le` relation reuse.
