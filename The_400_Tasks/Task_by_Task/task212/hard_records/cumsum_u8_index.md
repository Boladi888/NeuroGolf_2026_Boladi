# task212 hard record - U8 CumSum index route

Worker: `bo_ng_cdx2_worker2`
Date: 2026-07-04

## Live bar used

- Source: `python tools/ng_current_bar.py --task 212 --json`.
- Authoritative source: `Team_Submission/Backend/manifest.csv`.
- Team-best owner: Vamshi.
- Cost/profile cost: 616/616.
- SHA-256: `ff7a2c2b9d49a698aedd992e7c72a3116382b90dd9c6628d2a6533c1379f28cd`.
- Baseline gates:
  - `score_v2`: `(0, 0, 616) 616`
  - `profile_real_cost`: `task212 task212.onnx: static=616 profile=616 (memory 0 + params 616; 265 real inputs)`
  - `visible`: `[visible] 265/265 (100.00%)  PASS`

## Attempted route

Prompt route: collapse the stale 10-channel F16 flood plane into a single-channel
U8 index, use CumSum for the flood, then emit with an Equal-tail.

Task-local probe:

- Builder: `The_400_Tasks/Task_by_Task/task212/prototypes/build_task212_u8_cumsum_probe.py`
- ONNX: `The_400_Tasks/Task_by_Task/task212/prototypes/task212_u8_cumsum_probe.onnx`
- SHA-256: `15e03019892a6e63034d90a9ac90bea9b2adcfd1eb82cee1f22095f1dea7fb92`
- Raw ONNX bytes: 1,016

The probe slices a 10x10 blue plane, casts it to U8, runs `CumSum`, then
pads/concats to the required output shape. It is an acceptability probe, not a
correctness candidate.

## Exact failed gate

- `onnx.checker.check_model(full_check=True)`:
  `InferenceError: [ShapeInferenceError] (op_type:CumSum, node name: u8_cumsum_h): x typestr: T, has unsupported type: tensor(uint8)`
- `score_v2`: `None None`
- `visible`:
  `[visible] ERROR: InvalidGraph: [ONNXRuntimeError] : 10 : INVALID_GRAPH : This is an invalid model. Type Error: Type 'tensor(uint8)' of input parameter (safe_name_8) of operator (CumSum) in node (safe_name_9) is invalid.`
- `profile_real_cost`:
  `InvalidGraph: Type 'tensor(uint8)' of input parameter (safe_name_8) of operator (CumSum) in node (safe_name_9) is invalid.`

## Why this route is dead

`CumSum` does not accept U8 on this scorer/ORT path. The failure happens before
any task-specific correctness question, so the U8 index formulation cannot be
priced or gated. No truth, AB, battery, or full gates were run.

The refreshed live graph is already a zero-memory one-node `Einsum` at 616 dense
initializer elements. Operational F16 CumSum may be legal, but it would
materialize charged 10x10 crop/prefix/index planes and is not the prompt's U8
floor.

## One remaining hypothesis

A sub-616 win likely requires a new dense sign-coded direct-output factorization
of the separator-ray rule. Prior task-local notes already rule out sparse
`Einsum` initializers and simple subset drops from Vamshi's 616 factorization.
