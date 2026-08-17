# task333

Family: Gather / CumSum / anchor-line drawing

## Rule
Input is a fixed 10x10 grid. A green 2x2 block is the anchor. Non-green
markers that share either anchor row or either anchor column draw a same-color
straight run through zero cells toward the anchor, stopping before the green
block. Off-axis markers and the green block are preserved.

ARC-GEN source: `external/ARC-GEN/tasks/task_d43fd935.py`.

## Graph Summary
- 57 nodes: 8x Gather, 8x CumSum, 8x Equal, 8x Cast, 4x Less, 4x Where, 4x Add, 3x Mul, 3x Sub, 1x Slice, 1x Conv, 1x Concat, 1x Pad
- Input: (1,10,30,30), Output: (1,10,30,30)
- Slices to 10x10 sub-grid (all task grids are exactly 10x10), processes, Pads back to 30x30
- Algorithm: CumSum-based region detection for anchor-attracted line drawing

## Cost Baseline
- cost=25693, memory=25640, params=53, file_size=4462 bytes
- current team-best bar: cost=2783 from
  `K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/manifest.csv`, creator
  Aditya, SHA
  `9b42379d91db50442b683faebee29489daa45e364dc6b9398ec53c9e0d1f226b`.
- 2026-07-03 refresh: authoritative workbook + team manifest bar is Vamshi
  cost/profile 635, SHA
  `bb14c2aac494c18167f7eab7cc1ac124e94634011c4d2e81da68a8acb1321c61`.
  Treat the older 2783 local/staged note as stale context only.

## Attempts

### 2026-07-03: reviewer rework, transposed one-plane LE reuse — DEAD BELOW BAR
- Builder: `prototypes/build_task333_transposed_le_probe.py`.
- Candidate: `prototypes/task333_transposed_le_eqprod_probe.onnx`.
- Locked hypothesis tested: replace `relation_feature[2,10,10]` with one
  dense `le[1,10,10]` and rewrite the one-Einsum relation pairs so the
  cost-valid monomial consumes `le(a,b)*le(b,a)` as equality.
- Cost/profile: `score_network_v2` `(0, 0, 526) 526`; profile
  `static=526 profile=526 (memory 0 + params 526; 265 real inputs)`.
- Gate: visible `[visible] 0/265 (0.00%)  FAIL`; no truth/AB/full gates run.
- Verdict: no staged candidate. A single fixed monomial cannot preserve both
  strict directions: `lt = le(a,b)-eq` and `gt = le(b,a)-eq` need a second
  monomial selected inside the relation test. The required selector is another
  10x10 coordinate plane or a charged relation/output intermediate, which
  erases the 109-byte saving. See `hard_records/transposed_le_reuse.md`.

### 2026-07-03: sub-635 one-Einsum probes — NO STAGED CANDIDATE
- Scratch plan: `prototypes/task333_scratch_plan_20260703.md`.
- Authoritative team-best graph is a single zero-memory `Einsum` with 635
  params: `active_map` 300, `relation_feature` 200, small cores 135.
- Sparse-initializer probe:
  `prototypes/task333_sparse_inits_probe.onnx`; failed static score/checker
  (`ShapeInferenceError` rank-0 sparse Einsum input; `score_network_v2=None`).
  See `hard_records/sparse_einsum.md`.
- One-plane relation probe:
  `prototypes/task333_relation_le_only_probe.onnx`; cost/profile 526 but
  visible `[visible] 0/265 (0.00%)  FAIL`. The `not_equal` relation plane is
  load-bearing for strict/equality direction decoding. See
  `hard_records/relation_le_only.md`.
- Verdict: no reviewer-ready candidate. Do not retry sparse Einsum or one-plane
  relation-feature variants; remaining hypothesis is a different dense
  factorization/top-left-green selector that saves at least one small-core param.

### 2026-06-30: scalar Einsum strict-clean scratch probe — CORRECT BUT TOO COSTLY
- Builder: `prototypes/build_task333_scalar_einsum.py`
- Candidate: `prototypes/task333_scalar_einsum.onnx`
- Rule implementation: `Einsum` color-index crop, coordinate-derived fill masks,
  scalar Pad with sentinel, free `Equal` tail.
- Gates/cost: visible `265/265 PASS`; profile `static=6203 profile=6203
  (memory 5530 + params 673; 265 real inputs)`.
- Verdict: learning/probe only; above the documented 2783 team-best bar.

### 2026-06-30: team-best Reshape->Unsqueeze shave — STAGED WIN
- Builder: `prototypes/build_task333_teambest_unsqueeze_shave.py`
- Staged:
  `staged_candidates/task333_teambest_unsqueeze_cost2775/task333.onnx`
- Change: replace two opset-12 `Reshape` nodes that only add singleton
  dimensions with attribute-based `Unsqueeze`; remove `shape_row_idx` and
  `shape_col_idx` initializers.
- Cost/profile: `cost=2775`, `profile=2775` (`memory 2412 + params 363`),
  delta `-8` vs team-best 2783.
- Gates: visible `265/265 PASS`; truth-full `200000/200000 (100.000%)`;
  battery `PASS`; full A/B vs `submission.zip` `200265/200265`, byte-equal.

### 2026-05-19: Neg(v59) instead of Sub(v22,v22)+Sub(v60,v59) — NO IMPROVEMENT
- Replaced the two Sub nodes with Neg node; same memory/params cost (25693)
- The cost scorer's memory count is dominated by CumSum intermediate tensors, not removable Sub nodes

### 2026-05-19: Identity Gather approach (replace 8 Equal/Cast) — WORSE (cost=38601)
- Replaced 8 Equal/Cast nodes with Gather from identity matrix
- The (100,10) intermediate Gather output is larger than 8 separate (1,1,10,10) tensors
- Revert: Equal/Cast approach is more cost-efficient for small grids

### Conclusion
- task333 is inherently memory-expensive (25640 from CumSum intermediates)
- The 10x10 grid is already minimal (all inputs are 10x10)
- No significant cost reduction found with polish-style tricks
- No staged candidate
