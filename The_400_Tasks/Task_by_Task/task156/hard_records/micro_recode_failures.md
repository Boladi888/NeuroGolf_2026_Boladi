# task156 micro-recode dead routes

Date: 2026-07-07

## Live/team bar

- Authoritative bar: cost/profile_cost 980.
- Source: `python tools/ng_current_bar.py --task 156 --json`; authoritative source `manifest`, workbook agrees.
- Owner/sha: Vamshi, `2de15be6a1ab426e7f61ed2700cfd1956d1b403908ec48454d5fbcc2cd5ac7b8`.

## Scalar background route

- Attempted route: replace `background_codes=[1,0]` with broadcast scalar `zero_u8`.
- Candidate: `The_400_Tasks/Task_by_Task/task156/prototypes/task156_scalar_background.onnx`
- Failed gate: `[visible] 0/265 (0.00%)  FAIL`
- Why dead: `background_codes` is not zero; `[1,0]` distinguishes active background from QLinear padded no-output cells. Broadcasting zero collapses active background with padding.
- Remaining hypothesis: only a recode that keeps padding distinct from active background can delete this 2-element tensor.

## INT8 feature route

- Attempted route: convert feature `Where` outputs and final QLinearConv output to INT8 so `zero_i8` can replace `zero_u8`.
- Candidate: `The_400_Tasks/Task_by_Task/task156/prototypes/task156_i8_features.onnx`
- Failed gate: `[visible] ERROR: NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for Where(9) node with name 'safe_name_14'`
- Why dead: ORT does not implement `Where` for INT8 on this path; adding casts would cost at least a full feature plane and lose the scalar saving.
- Remaining hypothesis: dtype unification would need a UINT8-compatible path or a non-Where feature constructor.

## UINT8 weights with zero-point 1 route

- Attempted route: change QLinear weights to UINT8 and use `one_u8` as `w_zero_point`, deleting `zero_i8`.
- Candidate: `The_400_Tasks/Task_by_Task/task156/prototypes/task156_u8_weights_zp1.onnx`
- Failed gate: `[visible] 168/265 (63.40%)  FAIL`
- Why dead: bounded integer filters with effective weights `raw - 1` were separability-derived from bundled windows but did not preserve the live rule under ONNX execution.
- Remaining hypothesis: a source-derived low-negative-weight construction might exist, but the visible miss kills this fitted-weight formulation.
