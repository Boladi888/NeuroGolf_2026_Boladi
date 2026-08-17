live bar: manifest/team-best task325 cost 1658, sha256 962a051fea10a5c87401b078ce282ee1752ee088e8597640c6b04062e4c02049

attempted route: replace the F16 correction-scalar `Sum` in Aditya's anchor counter with a UINT8 `Sum`, keeping `n0_u8` and casting correction booleans to U8.

failed gate: `python tools/ng_gate.py visible --task 325 --cand The_400_Tasks/Task_by_Task/task325/prototypes/task325_anchor_u8sum.onnx` -> `InvalidGraph: Type 'tensor(uint8)' of input parameter ... of operator (Sum) ... is invalid`.

why dead: ORT rejects UINT8 inputs to `Sum` even at opset 14; the valid sibling is a U8 `Add` chain, staged as `task325_anchor_u8add_cost1657`.

remaining hypothesis: only use U8 scalar arithmetic through opset-14 `Add`/`Sub`/`Mul` kernels, not variadic `Sum`; larger savings need avoiding the full occupancy Slice, not scalar dtype tweaks.
