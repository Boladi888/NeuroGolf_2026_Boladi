# int32_scatternd_indices dead route

Live/team-best bar: 488 cost, sha256
`10a94404614387089c8f0c6f2a4ea399c3ee0b8f857cc9d178be70d0c3313e85`
from team manifest and workbook `Tasks` row, updated 2026-07-02T02:35:15Z.

Attempted route: clone the Vamshi 488 sparse-write graph and change the
ScatterND index path from INT64 to INT32 (`prefix_idx` INT32 and
`Cast(bottom_f)` to INT32).

Exact failed gate: build/checker before scoring.

Failure line:
`onnx.onnx_cpp2py_export.shape_inference.InferenceError: [ShapeInferenceError] (op_type:ScatterND, node name: scatter_add_output): indices typestr: tensor(int64), has unsupported type: tensor(int32)`

Why dead: ONNX/ORT scorer stack has `ScatterND`/`GatherND` indices effectively
INT64-only; `ScatterElements` can use INT32, but this graph needs multi-axis
trailing-slice writes to dynamic row and fixed channels 0/4. Replacing it with
axis-local ScatterElements requires much larger update/index planes.

One remaining hypothesis: avoid ScatterND entirely with a different final
construction that writes both ch0-clear and ch4-set without materializing a full
delta canvas.
