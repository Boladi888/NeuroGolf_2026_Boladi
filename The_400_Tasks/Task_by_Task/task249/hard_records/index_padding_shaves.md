# task249 index padding shave dead routes

Live/team bar: cost 246, sha256 1e0ed5ec19fb8d0caefc1f3c3425cf88d8f8a242a11667806c4caae1782cb611, source `Team_Submission/Backend/manifest.csv` and workbook row.

Attempted route 1: convert the INT32 `Pad(gather_idx10)` from opset-11 input form to opset-10 attribute form, removing `pad_pads` and `pad_value` params. Candidate `prototypes/task249_worker3_attrpad243.onnx`, intended cost 243.

Exact failed gate: `score_network_v2` returned `None`; `onnx.checker.check_model(full_check=True)` -> `[ShapeInferenceError] (op_type:Pad): data typestr: T, has unsupported type: tensor(int32)`.

Attempted route 2: sparse-encode `idx_table10` to save 6 params. Candidate `prototypes/task249_worker3_sparse_idx_probe.onnx`, intended cost 240.

Exact failed gate: `score_network_v2` returned `None`; checker/shape inference -> `(op_type:Gather): [ShapeInferenceError] data tensor must have rank >= 1` and downstream Pad input null.

Attempted route 3: sparse-encode `pad_pads` to save 1 param. Candidate `prototypes/task249_worker3_sparse_pad_pads_probe.onnx`, intended cost 245.

Exact failed gate: `score_network_v2` returned `None`; checker -> `[ShapeInferenceError] (op_type:Pad): pads typestr: tensor(int64), has unsupported type: sparse_tensor(int64)`.

Why dead: the only cheap sub-246 shaves of the current graph rely on representations rejected by strict checker/shape inference before scoring. Sparse op inputs and Pad-10 integer data are not legal for this scorer path.

One remaining hypothesis: replace the `LpPool -> ReduceSum -> Sub -> Cast -> Gather(table) -> Pad(index)` chain with a legal full-canvas generator that produces the 30-column index vector without a charged INT32 `[30]` intermediate; no such ONNX primitive was identified in this pass.
