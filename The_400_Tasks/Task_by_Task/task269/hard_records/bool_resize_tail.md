# bool_resize_tail

Live bar: 468 bytes, manifest sha256 8bc8c0fc9401965b9a1889dab5aaba275f3c5e9996e1805f7b2c559863731a0f.

Attempted route: delete charged `Cast(source_bool -> source_u8)` and feed BOOL through the existing sentinel Pad and final `Resize`, with graph output dtype BOOL.

Failed gate: `python tools/ng_gate.py visible --task 269 --cand The_400_Tasks/Task_by_Task/task269/prototypes/task269_bool_resize_tail.onnx` -> `[visible] ERROR: NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for Resize(18) node with name 'output'`.

Why dead: ORT 1.24.4 has no BOOL Resize kernel here, so the 90 B Cast is load-bearing.

Remaining hypothesis: reduce the decode path before U8 Resize; do not retry BOOL Resize unless the runtime/kernel set changes.
