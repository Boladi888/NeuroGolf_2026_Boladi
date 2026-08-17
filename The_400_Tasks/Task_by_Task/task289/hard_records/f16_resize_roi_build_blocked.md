# task289 hard_record — F16 Resize ROI build-blocked

**Worker:** bo_ng_cdx_worker2, task289, 2026-07-14

**Authoritative live bar:** 324, SHA-256
`ffbb8ecbbd332816e69ab20d1539bda1c778bc7372b01a37ba061d95766de62e`
(`python tools/ng_current_bar.py --task 289 --json`, manifest owner Sho).

**Effective task-local staged bar:** 322, SHA-256
`ebefa45e3ef20fcae3a8e004fc67445be413917101cfa230b6a66f6eaef8a65b`
(`staged_candidates/task289_roi_broadcast_fold_cost322/manifest.json`).

## Attempted route

Prototype builder:
`The_400_Tasks/Task_by_Task/task289/prototypes/build_task289_f16_roi/build_task289_f16_roi.py`

The 322 parent computes a tiny F32 chain:

`present_sum_u8 -> Cast(F32) -> Div(roi_num4_f, present_sum_f) -> roi -> Resize`

The attempted fold changed `present_sum_f`, `roi_num4`, `roi`, and `scales30_hw` to F16. If accepted, this would have saved 10 memory bytes (`present_sum`: 4 -> 2, `roi`: 16 -> 8) with unchanged params.

## Exact failed gate

No `ng_gate` correctness gate was run because the model failed ONNX strict shape inference during build:

`[ShapeInferenceError] ParseData type mismatch for tensor: scales30_hw_f16. Expected:float Actual:float16`

## Why dead

In this pinned scorer path, `Resize` requires F32 scales. Keeping F32 scales while making only the ROI arithmetic F16 cannot produce the intended memory saving and would require extra casts. This route is blocked before visible verification.

## One remaining hypothesis

A future ORT/schema path that accepts F16 `Resize` scales/ROI could reopen a small scalar-memory shave, but it is not valid under the current NeuroGolf scorer.
