# task014 — sparse initializer probe REJECT_LOCAL

## Live bar
- Authoritative bar: **1306** from `python tools/ng_current_bar.py --task 14 --json`, source `manifest`.
- Parent: `Team_Best_Onnx/task014.onnx`
- Parent sha256: `2b7ec8991a744169495b1e10dfe6f5de29a5309917d082728695146a3cccdedb`

## Attempted route
`prototypes/build_task014_sparse_init_probe.py` tried to convert sparse one-hot / mostly-zero constants to ONNX `SparseTensorProto` initializers to reduce params without adding intermediates.

## Exact failed gate
- Sparse `E0..E4`: `onnx.checker.check_model(full_check=True)` failed with `[ShapeInferenceError] (op_type:Gather): data typestr: T, has unsupported type: sparse_tensor(float)`.
- Sparse `Amid`, `oh0`, `w10` only: `onnx.checker.check_model(full_check=True)` failed with `[ShapeInferenceError] (op_type:Einsum): Inputs typestr: T, has unsupported type: sparse_tensor(float)`.

## Why dead
The current binary-radix graph consumes the parameter-heavy constants through `Gather` and `Einsum`, and ONNX strict shape inference rejects sparse float tensors for both consumers on the pinned stack. The route never reaches score or visible.

## Remaining hypothesis
Only a graph that removes `Gather`/`Einsum` consumption of these constants could revisit sparse params, but that would no longer be a local sparse-initializer polish of the current parent.
