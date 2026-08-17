live bar: 3298 bytes, sha d76b6fa5e7bc3e2286979210740cfc099bc50ea82e8e7f5857d32a9024cbbd37 (`ng_current_bar.py --task 44 --json`, manifest authoritative).

attempted route: replace the dense one-hot selector initializer `Ssel` `[30,10]` with an ONNX sparse initializer so the repeated `Einsum` operands would charge only the 10 stored nonzeros instead of 300 params.

exact failed gate: `onnx.checker.check_model(..., full_check=True)` on `prototypes/worker2_pairwise_shaves/task044_sparse_ssel_probe.onnx` failed with `InferenceError('[ShapeInferenceError] (op_type:Einsum): Inputs typestr: T, has unsupported type: sparse_tensor(float)')`.

why dead: ORT/ONNX shape inference rejects sparse tensors as `Einsum` inputs before scoring or runtime, so this cannot be a valid NeuroGolf candidate even though the param accounting would be attractive.

one remaining hypothesis: reduce the selector tax only by changing the formulation to avoid `Ssel` entirely; sparse initializer accounting is not available for this graph family.
