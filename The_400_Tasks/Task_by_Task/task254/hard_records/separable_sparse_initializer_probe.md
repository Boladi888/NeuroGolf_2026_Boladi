# DEAD ROUTE - task254 separable sparse-initializer probe (2026-07-14)

**Live/team-best bar used:** cost 254, manifest-authoritative via
`python tools/ng_current_bar.py --task 254 --json`.
Owner Vamshi, SHA256
`d71b4cb3ec548150846dbf5b14a8c50d2356dd300882cbb118e02ec52f77f5e1`,
updated 2026-07-12T15:15:54Z. Local task_state/library costs 329/9274 are stale.

**Parent graph:** `Team_Best_Onnx/task254.onnx`, static/profile cost 254
(`memory 88 + params 166`). Structure is 4 nodes:
`Einsum -> ReduceMax -> Sub -> Einsum`; charged intermediates are
`scores[2,5]` 40 B, `mx[2,1]` 8 B, `colmask[2,5]` 40 B.
Visible gate on parent: `[visible] 265/265 (100.00%)  PASS`.

## Attempted route

The current 254 graph has sparse dense constants, especially `x[2,30]`.
If `Einsum` accepted ONNX sparse initializers, replacing dense zero-bearing
constants with sparse initializers would drop params below 166 and produce a
real sub-254 candidate without changing graph math.

Builder:
`The_400_Tasks/Task_by_Task/task254/prototypes/build_task254_sparse_init_probe.py`

Generated probes:
- `prototypes/task254_sparse_x_probe.onnx`
  SHA256 `a87162869d8c3af141eb106017586d1e5f756f0c4d036b9aca5e5b9973e7396d`
- `prototypes/task254_sparse_zero_bearing_probe.onnx`
  SHA256 `f4f5310e4ad63e1a4d63458693808fc955d61e217783965f51d54f0178fd0d32`

## Exact failed gate

Static scorer:
`score_network_v2(prototypes/task254_sparse_x_probe.onnx) -> None`
`score_network_v2(prototypes/task254_sparse_zero_bearing_probe.onnx) -> None`

Checker:
`onnx.checker.check_model(..., full_check=True)` fails both with:
`InferenceError [ShapeInferenceError] (op_type:Einsum): Inputs typestr: T, has unsupported type: sparse_tensor(float)`

No `ng_gate visible/truth/ab` was run on these probes because they are
build-time invalid and cannot score.

## Why this route is dead

The only promising one-shot sub-254 shave from the current graph was to reduce
initializer element count without changing the 4-node separable moment core.
ONNX shape inference rejects sparse tensors as `Einsum` inputs, so sparse
initializer accounting cannot be used here. Dense initializers remain the legal
floor for this representation family.

## One remaining hypothesis

A non-sparse improvement would need a different final-output family that keeps
the current 2x5 decision core but avoids either `colmask[2,5]` (40 B) or part
of the 60-param width basis `x[2,30]` without materializing a generated width
operand. Obvious generation routes (`Range`/`Pad`/`Expand`/`Concat`) are likely
more expensive than the 60 params, so price before building.
