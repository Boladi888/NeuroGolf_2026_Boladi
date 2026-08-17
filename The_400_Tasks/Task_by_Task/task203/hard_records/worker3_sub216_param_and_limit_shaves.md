# task203 worker3 sub-216 param/default/limit shaves

Live bar and sha:
- Authoritative live bar: 216 from `python tools/ng_current_bar.py --task 203 --json`, source `manifest`, owner `Vamshi`, sha256 `45f547e4ec9b55f592b273d418dfa6304c0efea40015d1964ded0b64e60957a8`, updated `2026-07-12T15:15:54Z`.
- Current 216 graph profile: `python tools/agent_helpers/profile_real_cost.py --task 203 --cand Team_Best_Onnx/task203.onnx --arcgen 266` => `profile=216 (memory 194 + params 22; 267 real inputs)`. Static dummy scoring is `None` because the manifest graph intentionally OOBs on the dummy all-ones scorer input at `ScatterElements`.

Attempted route:
- Preserve the current Path-B `ReduceL1 -> QuantizeLinear -> ScatterElements -> Gather(input, remap_i)` formulation and shave below 216 by removing params or scalar/vector plumbing.
- Built probes with `The_400_Tasks/Task_by_Task/task203/prototypes/build_task203_worker3_probes.py`.

Exact failed gates / probes:
- `sparse_zero10_nnz0` and `sparse_zero10_nnz1`: build/checker failed before gate with `InferenceError: [ShapeInferenceError] (op_type:ScatterElements): data typestr: T, has unsupported type: sparse_tensor(uint8)`.
- `sparse_colors`: build/checker failed before gate with `InferenceError: [ShapeInferenceError] (op_type:ScatterElements): updates typestr: T, has unsupported type: sparse_tensor(uint8)`.
- `task203_worker3_colors_base.onnx`: replaced `ScatterElements(zero10, s_i, colors)` with `ScatterElements(colors, s_i, colors)` to delete the 10-param zero base. Exact failed gate: `python tools/ng_gate.py visible --task 203 --cand The_400_Tasks/Task_by_Task/task203/prototypes/task203_worker3_colors_base.onnx` => `[visible] 140/267 (52.43%)  FAIL`.
- `task203_worker3_constant_outer6.onnx`: replaced dynamic `outer_s = ReduceMax(s_i)` with an INT32[1] initializer to save the 4-byte scalar memory at +1 param. Official input precheck showed visible `outer_s` varies (`train`: 2,3,4,5; `test`: 6; `arc-gen`: 2..8). Exact failed gate: `python tools/ng_gate.py visible --task 203 --cand The_400_Tasks/Task_by_Task/task203/prototypes/task203_worker3_constant_outer6.onnx` => `[visible] 41/267 (15.36%)  FAIL`.

Why dead:
- Sparse initializers cannot feed either `data` or `updates` for `ScatterElements` under strict ONNX shape inference, so sparse params cannot shave `zero10` or `colors`.
- The zero default in `by_s` is load-bearing for absent output channels; using `colors` as the scatter base maps those absent channels to present input colors on many visible examples.
- The dynamic reversal limit is load-bearing because official ring depth varies; a constant limit cannot be correct.

One remaining hypothesis:
- Beating 216 likely requires a different final-output representation that avoids the inverse-permutation scatter/gather vector altogether. Within the current QLinear rank + inverse-permutation Path-B graph, the remaining charged pieces (`area_f`, `s_i`, `comp_s`, `remap_i`, `by_s/remap_u8`, `s_i8`, and 22 params) all have a tested or dtype-forced reason to stay.
