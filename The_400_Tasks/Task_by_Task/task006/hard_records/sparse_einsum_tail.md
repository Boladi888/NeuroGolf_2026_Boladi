# task006 hard record: sparse_einsum_tail

- Date: 2026-07-15
- Live/team-best bar: cost 66, sha256 `4f10ce1e8486a3353d26d1540552f141a61e23e5bdf0396a596622040ccaca1c`, source `python tools/ng_current_bar.py --task 6 --json` / team manifest, owner Sho, updated 2026-07-09T01:09:17Z.
- Attempted route: keep Sho's first Conv detector (`v_t0 [1,1,3,3]`, 36 B memory, 20 params) and replace the 10-param output Conv with final-output `Einsum("bnij,cn,ir,js->bcrs")`. Dense row/col/color factors are legal but cost 246. Sparse row/col/color factors would price at 64 if legal.
- Exact failed gate: `python The_400_Tasks/Task_by_Task/task006/prototypes/build_task006_sparse_einsum_tail.py` reports `sparse tail rejected before save: InferenceError: [ShapeInferenceError] Inference error(s): (op_type:Einsum, node name: v_tail): [ShapeInferenceError] Rank of input 1 (0) does not match the equation indices (2).`
- Why dead: ONNX full-check / strict shape inference treats sparse initializer inputs to `Einsum` as rank-0, so the candidate cannot reach the official scorer. Adding dense value_info would bill the dense `[3,30]` row and col factors and regress to the dense 246-byte control.
- One remaining hypothesis: reopen only if a legal final-output op can use sparse/factorized spatial placement without dense value_info or if the scorer's sparse-initializer handling changes; otherwise the 66-byte dense Conv bottleneck remains the practical floor for this formulation.
