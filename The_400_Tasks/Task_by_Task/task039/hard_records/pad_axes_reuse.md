# task039 hard record - pad_axes_reuse

Live/team bar used: cost 441, sha256
`dc9191feb18062f9d159affc29e09b0bd8b77a3259cefb129ada0876dd915e0b`,
from workbook and team manifest refreshed 2026-07-03.

Attempted route: reuse the existing `grid_axes=[2,3]` initializer as Pad-18's
axes input, deleting the separate `pad_axes=[2,3]` initializer. Expected static
cost if legal: 439.

Exact failed gates:
- `build_task039_reuse_pad_axes_v1.py` with INT32 `grid_axes` failed
  `onnx.checker.check_model(full_check=True)`:
  `[ShapeInferenceError] ParseData type mismatch for tensor: grid_axes. Expected:int64 Actual:int32`.
- A quick source patch using INT64 `grid_axes` for both Slice and Pad failed:
  `[ShapeInferenceError] (op_type:Slice, node name: source_crop): axes has inconsistent type tensor(int64)`.

Why dead: in this ONNX/ORT scorer path, Pad-18's axes input is effectively
INT64-only, while Slice requires its `starts`, `ends`, and `axes` inputs to share
the same index type. Promoting the whole Slice index chain to INT64 would add
16 charged bytes to `starts`/`ends`, so the net cost would be 455, not a win.

One remaining hypothesis: only a scorer/checker-supported Pad axes relaxation or
a detector that already uses INT64 starts/ends for another reason could make axes
sharing beneficial here.
