live bar: 879 @ d4f6ee3d715fd6d664f903ae71cfcd9297d4de6481a48c91e0e0fad910d27c36
route: replace `Equal(ids_4d, channel_values10_nozero)` + `Cast(BOOL->UINT8)` with direct UINT8 `OneHot`.
failed gate: visible/load on `prototypes/task213_worker3_onehot_tail.onnx` -> `NOT_IMPLEMENTED : Could not find an implementation for OneHot(11)`.
why dead: ORT supports OneHot values for F16/F32/INT64 only; BOOL/UINT8 OneHot output has no kernel, and F16/F32 output cannot feed ConvInteger without losing the byte win.
remaining hypothesis: a final-output OneHot/F16 assembler might be useful only in a non-ConvInteger rewrite; it is dead for this tail.
