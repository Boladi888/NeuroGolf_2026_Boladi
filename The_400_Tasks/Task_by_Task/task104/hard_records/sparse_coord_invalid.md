# sparse_coord_invalid

- Live bar used: cost 118, SHA-256 `34571a7d5ea37c04e7562fe7ab21d293de3197cf9746aa02dec81bacda25d84c`, source `python tools/ng_current_bar.py --task 104 --json` / team manifest.
- Attempted route: encode mostly-zero `coord[30,3]` as an ONNX sparse initializer; prototypes `../prototypes/task104_sparse_coord_linear*.onnx` and `../prototypes/task104_sparse_coord_coo*.onnx`.
- Exact failed gate: no visible gate run; ONNX checker/scorer rejected the models before gating with rank/type inference errors on the sparse `Einsum` input.
- Why dead: ONNX Runtime/scoring path does not treat sparse initializer `coord` as a dense rank-2 tensor for `Einsum`; adding dense `value_info` changes the error to sparse-vs-tensor type mismatch.
- One remaining hypothesis: a non-sparse generated-coordinate graph could be tested, but Constant/Range materialization must beat the 90-param dense coord and is unlikely under profile cost.
