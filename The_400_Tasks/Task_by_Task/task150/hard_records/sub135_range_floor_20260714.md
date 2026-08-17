# task150 dead route - sub135 scalar/index shave

Date: 2026-07-14
Worker: bo_ng_cdx2_worker3

## Live bar

Authoritative bar from `python tools/ng_current_bar.py --task 150 --json`:
cost 135, owner Vamshi, updated 2026-07-12T15:15:54Z, sha256
`e399c6d31e3ba25c4562a6c87e71e032986bf13b6dbd5088e0fba03f04c11add`.
The team manifest/workbook agree at 135; local task_state 158 and legacy live_costs
162 are stale context.

Current parent graph:
`ReduceL2(input)->Cast(INT8)->Shrink(N-1)->Cast(INT32)->Shrink(N-31)->Range(step=-1)->Gather(axis=3)`.
Measured by `profile_real_cost.py --task 150 --cand Team_Best_Onnx/task150.onnx --arcgen 266`:
`static=135 profile=135 (memory 134 + params 1; 266 real inputs)`.

## Attempted route

Looked for a strict sub-135 formulation using the prompt's separable/moment angle:
recover `N` from scalar or row/column moments, then avoid or shrink the dynamic
width-30 permutation tail.

Fresh ARC-GEN numpy probe over 1000 rows confirmed:
`ReduceL2(onehot)=N`, row and column active marginals both sum to `N`, and
the negative-index Gather sequence from `Range(N-1, N-31, -1)` exactly produces
the in-active-region horizontal flip for square N in 3..9.

## Exact failed gate

The only known sub-135 representation family is full-axis reverse by negative-step
Slice, already present at `prototypes/probe_slice_reverse.onnx`.

Command:
`python tools/ng_gate.py visible --task 150 --cand The_400_Tasks/Task_by_Task/task150/prototypes/probe_slice_reverse.onnx`

Result:
`[visible] 0/266 (0.00%)  FAIL`

Per traps visible-fail axiom, this family is dead: it relocates the top-left active
N-block to columns 30-N..29 instead of keeping the flipped block anchored at col 0.

## Why sub135 is dead in this formulation

The current parent has only one non-scalar intermediate:
`cols [30] INT32 = 120 B`, produced by `Range`. This is forced by the output width
and `Gather`'s INT32/INT64 index type floor.

The remaining 15 bytes are already minimized:
- `side_f [] F32 = 4 B` is the required input-derived size scalar.
- `side_i8 [] INT8 = 1 B` and `start_i8 [] INT8 = 1 B` are the cheapest scratch
  path for `N` and `N-1`; doing the subtract in F32 or INT32 is larger.
- `range_start [] INT32 = 4 B` and `range_limit [] INT32 = 4 B` are required
  because `Range` must output INT32 indices for `Gather`.
- `neg_one [] INT32` initializer is 1 param; deriving it dynamically costs at
  least one extra scalar node and loses.

Using a static arange plus `Sub` restores the old 30-param arange floor. Using Slice
or Pad/Concat to avoid the `[30]` index either fails visible or materializes a large
F32 crop before the free output. ReverseSequence is unavailable for this axis and is
also validator-rejected by the `Sequence` op-type rule.

## Remaining hypothesis

Only a new ORT primitive or scorer quirk that writes a variable-length in-region
reverse directly to `output` could beat 135. Under the current valid op set, the
strict floor is `120 index bytes + 14 scalar bytes + 1 scalar param = 135`.
