# task242 pad_axes18 cost regression

- live bar: 422, SHA-256 `18963c9bb81868c1787bee15757387cd2e370259f6bff9f5a9f844b7eb4c48d6` (`Team_Best_Onnx/task242.onnx`, manifest owner Vamshi, updated 2026-07-12T15:15:54Z).
- attempted route: bump the 422 moment-crop graph to opset 18 and use Pad's optional `axes` input so final padding is expressed on axes `[1,2,3]` with pads `[1,0,0,0,27,27]`.
- prototype: `The_400_Tasks/Task_by_Task/task242/prototypes/task242_vamshi422_pad_axes18.onnx`, SHA-256 `37c2c03ebf5e680ef3b103b820b84e1337d08aceebafb1545051b629e9d89578`.
- gates: `ng_gate visible --task 242 --cand ...pad_axes18.onnx` passed `266/266`; `_score_v2` returned `(0, 364, 59)` and `profile_real_cost --arcgen 266` returned `static=423 profile=423`.
- why dead: `pads[8]` at 8 params becomes `pads_axes18[6] + pad_axes18[3]` at 9 params. `ng_graph_cost_diff` names this as net `+1` cost with no memory change.
- one remaining hypothesis: a sub-422 Pad-side shave would need attribute-form Pad, but the current graph needs `Einsum`, which is unavailable before opset 12 where Pad already uses input-form pads.
