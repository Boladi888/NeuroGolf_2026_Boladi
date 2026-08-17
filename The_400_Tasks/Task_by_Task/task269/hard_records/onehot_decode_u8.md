# onehot_decode_u8

Live bar: 468 bytes, manifest sha256 8bc8c0fc9401965b9a1889dab5aaba275f3c5e9996e1805f7b2c559863731a0f.

Attempted route: replace `Equal(labels, channel_values) -> Cast(U8)` with `Squeeze -> Cast(INT32) -> OneHot(U8)`, preserving the existing U8 Pad and final Resize tail.

Failed gate: `python tools/ng_gate.py visible --task 269 --cand The_400_Tasks/Task_by_Task/task269/prototypes/task269_onehot_decode_resize.onnx` -> `[visible] ERROR: NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for OneHot(11) node with name 'safe_name_12'`.

Why dead: the cheap U8 OneHot kernel is unavailable; supported numeric OneHot variants would materialize larger INT/F32 planes and lose to the 468-byte live graph.

Remaining hypothesis: a non-OneHot decode that emits U8 directly from labels would be needed; current `Equal -> Cast` remains the cheapest working decode seen.
