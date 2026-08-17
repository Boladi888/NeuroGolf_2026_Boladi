# task156 one-channel QLinear dead route

Date: 2026-07-03

## Live/team bar

- Team-best bar: 980 cost / profile_cost 980.
- Source: `NG_OnnxlyFans_TaskStatus.xlsx` Tasks row and `Team_Submission/Backend/manifest.csv` `task156`.
- Owner/sha: Vamshi, `2de15be6a1ab426e7f61ed2700cfd1956d1b403908ec48454d5fbcc2cd5ac7b8`.
- Refreshed profile line for team-best: `task156 task156.onnx: static=980 profile=980 (memory 725 + params 255; 265 real inputs)`.

## Attempted route

One-channel ternary-code QLinearConv rewrite:

- Candidate: `The_400_Tasks/Task_by_Task/task156/prototypes/task156_onechan_qlinear.onnx`
- Builder: `The_400_Tasks/Task_by_Task/task156/prototypes/build_task156_onechan_qlinear.py`
- SHA-256: `20aa1ad34c3a2dadeb9c7f44650b4cfa78cb2ae2b6df51acd0d0564899026aaf`
- Static score: `(0, 615, 143) 758`
- Existing profile record: `task156 task156_onechan_qlinear.onnx: static=758 profile=758 (memory 615 + params 143; 265 real inputs)`

## Failed gate

- Fresh visible gate: `[visible] 0/265 (0.00%)  FAIL`

## Why this route is dead

The one-channel ternary code collapses active background and QLinearConv padded
outside-canvas cells into the same effective zero code. That leaks the background
channel into the padded 20-pixel region and fails every visible pair. The prior
task-local manifest also records a linear-separability search over one-channel
effective codes in `[-5,5]` with no single 3x3 threshold bank separating active
background, blue interior, red interior, yellow border, and no-output padding.

Per the visible-fail axiom, do not run truth/AB/battery on this visible-zero
family and do not stage it despite the lower static/profile cost.

## Side probes during this pass

- QLinearConv default zero-point omission would save 1-2 params, but ONNX strict
  shape inference rejects each omitted zero-point with type-inference errors.
- Sparse QLinearConv bias would save 7 params, but strict shape inference rejects
  `sparse_tensor(int32)` for the `B` input.
- Biasless two-channel QLinearConv was checked as a zero-threshold separability
  problem over the official/generated task-shaped windows; blue/red interior
  channels are infeasible without a bias under the current two-channel features.

## Remaining hypothesis

A valid sub-980 candidate likely needs a different non-linear precode that keeps
padding distinct from active background without materializing another full 10x10
plane, or a genuinely new feature encoding that preserves the accepted two-channel
semantics while deleting at least one charged scalar/vector under strict ONNX
shape inference.
