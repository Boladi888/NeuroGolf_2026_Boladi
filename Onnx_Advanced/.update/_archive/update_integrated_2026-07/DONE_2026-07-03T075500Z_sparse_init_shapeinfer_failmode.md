# Sparse initializer — the ACTUAL grader failure signature (extends traps §8e / scoring_internals §194-196)

## What I discovered
`scoring_internals.md:195` and traps §8e already say `graph.sparse_initializer` counts only `values.dims` (nonzeros) for params but is "not usable for submissions" — attributed there to *op inputs can't reach it / scorer doesn't remap sparse names*. On task387 I actually built one (dense `P[18,30]` identity-embed → sparse init, 18 nonzeros, would drop params 540→18 ⇒ cost 2085) and hit a **different, more fundamental blocker worth recording**:

- **ORT itself loads the model fine** (`InferenceSession` builds, materializes sparse→dense internally). The block is upstream in the scorer.
- Grader `neurogolf_utils.calculate_memory` (line 195-196) runs `onnx.checker.check_model(model, full_check=True)` then `onnx.shape_inference.infer_shapes(model, strict_mode=True)` **before** touching the profiler. Both raise:
  `[ShapeInferenceError] (op_type:Einsum, node name: output): Rank of input 3 (0) does not match the equation indices (2).`
  because **onnx shape-inference does not propagate a sparse initializer's shape to consumer ops** — the Einsum's P input is seen as rank-0/unknown. `score_v2` → `None` (unscoreable). This will hit ANY shape-sensitive consumer (Einsum, Reshape, MatMul, Gemm), not just Einsum.
- **The obvious fix is self-defeating**: adding P to `graph.value_info` with shape `[18,30]` satisfies shape-inference, but `calculate_memory` iterates `input+value_info+output` and charges every value_info tensor its dense bytes — so P's 1080 memory-bytes (540×F16) get re-added. Net +558 vs just keeping the dense initializer. So there is no legal path to bank the sparse param discount when the tensor feeds a shape-sensitive op.

**Bottom line to add to the existing note:** the sparse-init discount is dead not only via name-remapping but because the grader's mandatory `check_model(full_check)` + `shape_inference(strict_mode)` reject a graph whose op-input shape is known only through a sparse initializer. Keep constants as **dense** initializers.

## Corollary confirmed by measurement (params=elements vs memory=bytes)
Dropping P entirely and widening the render basis to 30 (`task387_noP.onnx`, numerically equivalent) measured `score_v2=(0,2962,173)=3135` vs baseline 2607: params −528 but memory **+1056**. An F16 constant of N elements costs N params as a dense initializer but **2N memory-bytes** the moment it is materialized as any node output. ⇒ **A dense initializer is always the cheapest home for a constant** (strictly, whenever dtype itemsize > 1 byte). Corollary of `scoring_internals.md:29-30`; useful as an explicit "don't move constants into the compute graph" rule.

## Where
- Build/probe scripts: `The_400_Tasks/Task_by_Task/task387/prototypes/build_sparseP.py`, `build_noP.py`, `_membreak.py` (per-tensor memory replicator).
- Full task387 wall analysis: `The_400_Tasks/Task_by_Task/task387/hard_records/freeoutput_einsum_render_2607_wall.md`.

## Sorter note: traps.md §8e — extend with the strict-shape-inference failure signature (`check_model(full_check)`/`infer_shapes(strict_mode)` reject unknown-rank Einsum/Reshape/MatMul input fed by a sparse initializer; ORT itself loads fine; value_info fix re-charges memory). Keep the existing name-remap point.
## Sorter note: scoring_internals.md §Scoring-mechanics-quick-list (line ~194-196) — append the corollary "dense initializer is the cheapest home for a constant: N params vs 2N memory-bytes if materialized (F16); measured task387 no-P +528".
