# DEAD ROUTE - task054 runtime-only micro-shaves around v28

**Recorded:** 2026-07-13 by `bo_ng_cdx_worker2`.

## Bars
- Authoritative live bar from `python tools/ng_current_bar.py --task 54 --json`: **13,454**, sha `5bbf53b7c5993090ff3fa08bbca4d330e139328d6589a1bca16e7f4620184e75`, owner Boladi, source team manifest.
- Lower task-local team-best bar used for reviewer-ready decisions: **13,301**, sha `8116ec57563e07a2282af47e7a3642f173773b77258eb3d2d14ce0d13e67630e`, source `staged_candidates/task054_v28_component_safety_cost13301/manifest.json`.

## Attempted routes and exact failed gates

| route | prototype/builder | static delta | failed gate/result | why dead |
|---|---|---:|---|---|
| `CumSum(UINT8)` strip components | `prototypes/build_task054_probe_u8_cumsum.py` | predicted large save | build/checker: `ShapeInferenceError (op_type:CumSum, node name: comp_h): x typestr: T, has unsupported type: tensor(uint8)` | The strip component labels fit in U8, but ONNX/ORT does not support U8 CumSum here. F16 CumSum core remains forced. |
| omit unused `TopK` values output | `prototypes/build_task054_probe_empty_topk_values.py` | -16 B static if legal | checker: `ValidationError: Node (axis_topvals)'s output 0 is marked single but has an empty string` | `TopK` values output is required, even though v28 does not consume it. |
| scalar `ReduceSum` axes reuse | `prototypes/task054_probe_scalar_reducesum_axes.onnx` | -2 params static | `score_network_v2=None`; visible error: `ReduceSum ... axes_tensor->Shape().NumDimensions() == 1 was false. An axes tensor must be a vector tensor.` | Shape inference accepts scalar axes, but ORT runtime rejects them. The `[1]` axis initializers are forced. |
| `TopK(UINT8)` marker slots | `prototypes/task054_probe_u8_topk_slots.onnx` | -40 B static | `score_network_v2=None`; visible error: `NOT_IMPLEMENTED : Could not find an implementation for TopK(11)` | The marker slot scores are 0/1 U8, but ORT has no CPU kernel for U8 TopK. F16 widen/narrow is forced. |
| broadcast `ScatterElements` index `[4]` | `prototypes/task054_probe_scatterelements_broadcast_idx.onnx` | -480 B static | `score_network_v2=None`; visible error: `ScatterElements ... Indices and updates must have the same rank` | Static shape inference under-prices it, but ORT requires index/update rank match. |
| rank-2 `ScatterElements` index `[1,4]` | `prototypes/task054_probe_scatterelements_rank2_idx.onnx` | -464 B static | `score_network_v2=None`; visible error: `ScatterElements ... Indices vs updates dimensions differs at position=0 1 vs 30` | ORT requires the full `[30,4]` index tensor, not just rank compatibility. |

## Verdict
These are **static-only shaves**, not valid candidates. Do not stage or full-gate them. They either fail checker or fail the cheapest ORT execution/visible gate before any correctness comparison.

## One remaining hypothesis
Only a different line-write formulation that avoids `ScatterElements` entirely could remove `idxV`; the direct broadcast/rank-shrink variants are dead. Prior `striprunscatter_7000_route_dead.md` already prices the point-enumeration replacement as worse, so this is not worth reopening without a genuinely new write op/formulation.
