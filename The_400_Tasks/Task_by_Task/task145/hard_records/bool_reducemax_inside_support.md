# task145 BOOL ReduceMax inside-support route

- Live bar used: cost **12855**, SHA-256 `2dbfb6f9d5ea294d2e0be0fbfd5fbb2460a1e249de794bd3bb31f95a48b4f883`, source `python tools/ng_current_bar.py --task 145 --json` authoritative_source=`manifest`.
- Attempted route: replace Sho/Vamshi's F32 active row/column support scans (`MaxPool(z_f32)` -> `Cast`) with opset-20 `ReduceMax(z_bool, axes=[3/2])` BOOL vectors. Prototype: `The_400_Tasks/Task_by_Task/task145/prototypes/t145_reducemax_bool_inside.onnx`; builder: `prototypes/build_task145_reducemax_bool_inside.py`.
- Cost/probe result before correctness: `_score_v2` returned `(0, 12584, 113) cost 12697`, a theoretical **-158** vs the 12855 bar.
- Exact failed gate: `python tools/ng_gate.py visible --task 145 --cand The_400_Tasks/Task_by_Task/task145/prototypes/t145_reducemax_bool_inside.onnx` -> `[visible] 0/267 (0.00%)  FAIL`.
- Why dead: the scorer/shape-inference accepts the opset-20 BOOL `ReduceMax`, but the resulting graph is not semantically equivalent on any visible official. Per visible-fail axiom, this cannot be a reviewer-ready candidate; stop all variants that rely on BOOL `ReduceMax` for row/column active support.
- One remaining hypothesis: find a non-BOOL vector active-support primitive that consumes the existing F32 `z_f32` or existing U8 marker planes without materializing the two F32 vectors plus casts; plain F32 `ReduceMax` is not useful because it keeps the same vector bytes and adds axes params.
