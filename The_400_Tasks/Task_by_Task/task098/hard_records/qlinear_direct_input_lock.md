route: direct QLinearConv / U8 tiny carrier on current 270 live graph

Live bar: 270, sha256 `42d0ac9b0b331728a533bdf493b13f5b9cdd13811b4f3e3c8a9604c548530fa1`, source `python tools/ng_current_bar.py --task 98 --json` / team manifest, owner Vamshi.

Attempted route: apply the prompt's QLinear/U8 carrier idea to the refreshed live graph by replacing or narrowing the current final F32 Conv carrier.

Exact failed gate:
- Pre-build dtype/price gate after inspecting `Team_Best_Onnx/task098.onnx`: the current graph is already a single final-output `Conv(input F32, W[10,1,3,9]) -> output`, with `profile_real_cost.py` reporting `static=270 profile=270 (memory 0 + params 270; 266 real inputs)`.
- `ort_reality.md` / `traps.md` blocker: `QLinearConv` requires INT8/UINT8 input; the NeuroGolf gate harness feeds the graph input as F32. Casting the whole one-hot input to U8 costs 9,000 bytes before params, and the named-U8-input bridge is explicitly rejected by scorer sanitation.

Why dead: the stale 14,635 graph had F32 intermediate carriers to narrow. The refreshed live graph has no charged carrier plane at all; its only cost is the 270 dense Conv weight elements. QLinear dtype narrowing cannot reduce params because params are dtype-blind, and it would need an expensive input cast to run.

Remaining hypothesis: a QLinear route would need a pre-existing cheap U8/INT8 feature producer, which this current graph does not have. For task098, the plausible sub-270 route must avoid materializing any U8 carrier and instead compress the final predicate itself.
