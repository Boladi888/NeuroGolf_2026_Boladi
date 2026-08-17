# task162 bo_ng_cdx_worker2 polish/no-gain probe

- Live bar: 3496, SHA `b8951b5f95d030b98762cdab822b350a38b440247ccb68987af5551aa4317c92`
  from `python tools/ng_current_bar.py --task 162 --json` (manifest, owner Sho,
  updated 2026-07-09T01:09:17Z).
- Attempted route: run the standard math-equivalent polish stack against the current best and re-price
  the result under the current 3496 bar. Output:
  `prototypes/task162_bo_ng_cdx_worker2_polish_probe.onnx`.
- Exact failed gate: strict-below-current-best gate. `python tools/ng_graph_cost_diff.py --task 162
  --live Team_Best_Onnx/task162.onnx --candidate
  The_400_Tasks/Task_by_Task/task162/prototypes/task162_bo_ng_cdx_worker2_polish_probe.onnx`
  reported `NET COST DELTA 0  -> tie (no charged-cost change).` The probe also profiled as
  `static=3496 profile=3496 (memory 3320 + params 176; 266 real inputs)`.
- Why dead: the polished ONNX is byte-identical to the current best (same SHA-256), and all polish passes
  were no-ops: onnxsim, safe_opt, const_to_init, clip_mul_to_min, scalar_broadcast_ones, and
  slice_axis_collapse. It cannot be staged because it does not beat 3496.
- One remaining hypothesis: a legal op that consumes the F32 input and emits a <=1-byte sliding seed plane
  directly would delete the forced 1296-byte F32 head; no known ORT-valid op does this
  (`QLinearConv`/`ConvInteger` need int input or emit wider tensors).
