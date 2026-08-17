# task253 Sum(uint8) dead route

- Live/team-best bar used: cost 367, SHA256 `d9d16502e41c01a6afabc784c000e383f0a7717061dcc597bf9c697fecc66226`, source `python tools/ng_current_bar.py --task 253 --json` / manifest owner Sho.
- Attempted route: replace the parent `Mul(row_residue, 3) -> Add(col_residue)` with `Sum(row_residue, row_residue, row_residue, col_residue)` to delete one 10-byte UINT8 intermediate.
- Prototype builder: `The_400_Tasks/Task_by_Task/task253/prototypes/build_task253_sho367_sum_u8.py`.
- Failed gate: build/checker, before candidate save. Exact failure: `onnx.onnx_cpp2py_export.shape_inference.InferenceError: [ShapeInferenceError] (op_type:Sum, node name: v_n6_sum3): data_0 typestr: T, has unsupported type: tensor(uint8)`.
- Why dead: ONNX/ORT scorer path cannot accept `Sum(uint8)`, so this cannot be staged or gated.
- One remaining hypothesis: if a supported uint8 n-ary arithmetic op appears in a future opset/runtime, the same fusion would save 10 memory bytes; on the current pinned stack, keep the existing `Mul` + `Add`.
