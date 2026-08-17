# task292 hard record: sparse P for single-Einsum best

- live bar: cost 80, sha256 `f5dbb65451bb16d37f5e44041120e559dd95ac030ffb9d13dc5ac7c8850119d5`, source `python tools/ng_current_bar.py --task 292 --json` / team manifest, owner Sho, updated 2026-07-09T01:09:17Z.
- attempted route: preserve the current one-node `Einsum("ncij,cr,kr,rj->nkij", input, P, P, W)` exactly, replace dense `P[10,2]` with a sparse initializer containing the three non-zero entries, keeping dense `W[2,30]`.
- exact failed gate: builder `prototypes/build_task292_sparse_p_probe.py` failed at `onnx.checker.check_model(..., full_check=True)` before save/visible/score with `ShapeInferenceError: (op_type:Einsum): Rank of input 1 (0) does not match the equation indices (2).`
- why dead: ONNX strict shape inference does not expose sparse initializer rank to `Einsum`; this matches the project sparse-fed-op warning, so sparse `P` cannot be a reviewer-ready sub-80 path.
- one remaining hypothesis: a non-sparse sub-80 route would need a different one-node output family or a periodic-column generator that does not materialize a charged mask; no such legal dense factorization was found in this pass.
