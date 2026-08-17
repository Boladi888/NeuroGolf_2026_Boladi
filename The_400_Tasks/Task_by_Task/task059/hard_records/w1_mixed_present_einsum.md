# task059 w1_mixed_present_einsum dead route

- live bar: 470, SHA256 `105da16b59be13e9d797c5d0121e3cc1fb749eb4799b254155921b686bf0900d` (`python tools/ng_current_bar.py --task 59 --json`, manifest source, owner Sho, updated 2026-07-09T01:09:17Z).
- attempted route: delete Sho v28's `Cast(present_f -> present_f16)` 20B carrier and feed F32 `present_f` directly to the final F16 `Einsum`.
- exact failed gate: `python tools/ng_gate.py visible --task 59 --cand The_400_Tasks/Task_by_Task/task059/prototypes/task059_w1_mixed_present_einsum.onnx --json --no-watch-queue` failed at ORT load: `Type parameter (T) of Optype (Einsum) bound to different types (tensor(float16) and tensor(float) in node (output).`
- why dead: ORT requires one dtype for all `Einsum` operands. Promoting the renderer to F32 is already documented by worker3 as cost 524, so the 20B F16 carrier is forced for the current final-Einsum representation.
- one remaining hypothesis: only revisit if ORT adds mixed-dtype `Einsum` or a different final op can consume F32 color presence without promoting `state4`/basis tensors.

