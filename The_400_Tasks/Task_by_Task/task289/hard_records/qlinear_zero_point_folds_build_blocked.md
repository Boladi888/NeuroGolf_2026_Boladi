# task289 hard_record — QLinearConv zero-point folds build-blocked

**Worker:** bo_ng_cdx_worker2, task289, 2026-07-14

**Authoritative live bar:** 324, SHA-256
`ffbb8ecbbd332816e69ab20d1539bda1c778bc7372b01a37ba061d95766de62e`
(`python tools/ng_current_bar.py --task 289 --json`, manifest owner Sho).

**Effective task-local staged bar:** 322, SHA-256
`ebefa45e3ef20fcae3a8e004fc67445be413917101cfa230b6a66f6eaef8a65b`
(`staged_candidates/task289_roi_broadcast_fold_cost322/manifest.json`).

## Attempted route

Prototype builder:
`The_400_Tasks/Task_by_Task/task289/prototypes/build_task289_zero_point_folds/build_task289_zero_point_folds.py`

Tried three exact-looking variants over the 322 parent:

- convert the tail QLinearConv weights to signed INT8 actual weights and omit all zero-point inputs that are semantically zero;
- same INT8 tail but retain explicit `y_zero_point` inputs;
- keep the U8 tail weight zero point (`20`) but omit scalar `q_zero_u8` where the QLinearConv default should be zero.

## Exact failed gate

No `ng_gate` correctness gate was run because all variants failed ONNX strict shape inference during build:

`[ShapeInferenceError] input and zero_point pair is expected to have be same type in node QLinearConv (logits4_u8).`

Downstream type inference then left `MaxPool`, the second `QLinearConv`, and `Resize` with null input types.

## Why dead

The pinned ONNX shape-inference path does not accept empty optional zero-point inputs for this QLinearConv form. A generated zero-point tensor would cost at least the scalar initializer being removed, so there is no sub-322 saving through zero-point omission. Converting the tail weights to INT8 still needs an explicit INT8 zero point and does not reduce the official param count.

## One remaining hypothesis

Only a different exact k-detector that deletes the `present_u8 -> present_sum_u8 -> Cast` chain can plausibly beat 322; U8/INT32 `ReduceSum` remains rejected, and no cheaper exact OR-then-count kernel is known.
