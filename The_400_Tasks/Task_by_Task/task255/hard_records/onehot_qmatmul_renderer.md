# task255 onehot qmatmul renderer hard record

- Live bar used: 5,976, SHA-256 `d5e491ad0f517d08ac23ed2c6d984319fa1db09ab237b5d1ed5ad1a2675d224c`, from `python tools/ng_current_bar.py --task 255 --json` authoritative source `manifest` owner Vamshi updated `2026-07-12T15:15:54Z`.
- Attempted route: replace the staged 5,827 renderer's hand-built three category masks plus six `Unsqueeze`s with `Cast(ci)->OneHot(depth=3)->Transpose`, built at `../prototypes/task255_live_qmatmul_onehot_render.onnx`.
- Exact failed gate: `python tools/ng_gate.py visible --task 255 --cand The_400_Tasks/Task_by_Task/task255/prototypes/task255_live_qmatmul_onehot_render.onnx` -> `[visible] ERROR: NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for OneHot(11) node with name 'safe_name_58'`.
- Why dead: ORT/scorer rejects this OneHot lowering before correctness is reached, so the apparent 1-D glue saving is unshippable.
- One remaining hypothesis: keep the explicit U8 category masks, or find a non-OneHot way to produce both `[1,1,30,3]` and `[1,1,3,30]` category carriers cheaper than the current Less/Greater/Where plus Unsqueeze glue.
