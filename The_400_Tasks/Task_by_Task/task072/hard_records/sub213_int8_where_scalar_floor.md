# task072 hard record - sub-213 INT8/one-scalar route

- **live bar used**: 213 from `python tools/ng_current_bar.py --task 72 --json`.
- **live source**: team manifest, owner Sho, updated 2026-07-09T01:09:17Z.
- **live sha256**: `7e32730aa7d0ddfdbc37659c53d7d7fdecf09563882563507b57fe10d9cdd230`.
- **live profile**: `static=213 profile=213 (memory 180 + params 33; 268 real inputs)`.

## Attempted route

Rechecked the old projected 212 graph:

`Conv/ConvTranspose red diff -> Cast(BOOL) -> Where(INT8 -1/+1) -> ConvInteger`

Projected cost if ORT accepted it: memory 180 (`F32 d` 120 + `BOOL c` 30 + `INT8 s` 30) + params 32
(`read W` 20 + `render W` 10 + two branch scalars) = **212**, one byte below Sho's current 213.

Prototype:

`The_400_Tasks/Task_by_Task/task072/prototypes/task072_xor_convinteger_recheck.onnx`

sha256: `c409272edeb712aa6d0be46d74f0e43c70188f2013164ba70bca026354a71bbc`

## Exact failed gate

Build/score command:

`python The_400_Tasks/Task_by_Task/task072/prototypes/build_task072_xor_convinteger.py --variant convinteger --out The_400_Tasks/Task_by_Task/task072/prototypes/task072_xor_convinteger_recheck.onnx`

Result:

`score: REJECTED (None)`

`onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for Where(9) node with name 'signed_mask_i8'`

Visible command:

`python tools/ng_gate.py visible --task 72 --cand The_400_Tasks/Task_by_Task/task072/prototypes/task072_xor_convinteger_recheck.onnx`

Result:

`[visible] ERROR: NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for Where(9) node with name 'safe_name_6'`

Profile command:

`python tools/agent_helpers/profile_real_cost.py --task 72 --cand The_400_Tasks/Task_by_Task/task072/prototypes/task072_xor_convinteger_recheck.onnx --arcgen 0`

Result: same ORT `Where(9)` NotImplemented during session initialization.

## Why the current 213 scalar triplet appears forced

Sho's live graph avoids unsupported INT8 `Where` by producing a UINT8 mask:

`Cast(d, BOOL) -> Where(c, two=2, zero=0) -> ConvInteger(m, W8, x_zero_point=one=1)`

The three scalar values `{0, 1, 2}` are not cosmetic. With a one-tap ConvInteger renderer,
the two mask values must be on opposite sides of `x_zero_point`; if either branch equals the
zero point, that class produces exactly zero and fails the `>0` decoder on its required output
channel. Therefore the two branch values and zero point must be three distinct integer scalars.

Replacing `Where(0,2)` with `Cast(c, UINT8)` saves the two branch scalars but gives mask values
`{0,1}`. No integer zero point lies strictly between them, so one of bg or XOR necessarily decodes
as zero in its required channel. Generating `2*Cast(c)` with arithmetic adds at least one extra
30-byte UINT8 tensor, losing far more than the two scalar params it saves.

## Remaining hypothesis

Only revisit this floor if one of these becomes true:

1. ORT adds a working `Where(BOOL, INT8, INT8)` kernel for the scorer environment.
2. A supported op can map `BOOL -> UINT8 {0,2}` without branch scalars and without materializing an
   additional 30-byte tensor.
3. A legal ConvInteger/QLinearConv renderer provides a bias term cheaper than the two UINT8 branch
   scalars.
