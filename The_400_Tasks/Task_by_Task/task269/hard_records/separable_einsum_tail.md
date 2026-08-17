# separable_einsum_tail

Live bar: 468 bytes, manifest sha256 8bc8c0fc9401965b9a1889dab5aaba275f3c5e9996e1805f7b2c559863731a0f.

Reviewer-locked route: keep the live 468 head through `source_u8` and `foreground_count`, replace `Pad/Div/Resize` with dynamic row/col selectors `selector_i(H) = i*N <= H < (i+1)*N`, and make the graph output `Einsum('bcij,iH,jW->bcHW', source, row_sel, col_sel)`.

Citations used: `Onnx_Advanced/techniques.md` #89 (free-output Einsum in-kernel temporaries), #109 (multi-term separable free-output, plus carrier/runtime-selector caveats), #119 (deep separable output), and `traps.md` §FREE-OUTPUT.

U8 variant: `The_400_Tasks/Task_by_Task/task269/prototypes/task269_separable_einsum_u8.onnx`, sha256 D77D979B58E3AA9DAD93C66DB8AD2C306DD6D8C95548672BB92D306542E1CCBB.

Failed gate: `python tools/ng_gate.py visible --task 269 --cand The_400_Tasks/Task_by_Task/task269/prototypes/task269_separable_einsum_u8.onnx` -> `[visible] ERROR: NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for Einsum(12) node with name 'output'`.

F32 fallback: `The_400_Tasks/Task_by_Task/task269/prototypes/task269_separable_einsum_f32.onnx`, sha256 5C4E1B279AAE8A5640A2DD9A0E52EDA96B76D55884255E33F0F9E429DEF617AB.

Passed visible: `[visible] 266/266 (100.00%) PASS`.

Cost proof: `score_network_v2` -> `(0, 1234, 96) 1330`; `profile_real_cost.py --arcgen 266` -> `static=1330 profile=1330 (memory 1234 + params 96; 266 real inputs)`. Delta vs live = +862 bytes, so no truth/AB/full gates.

Why dead: the U8 carrier is blocked by ORT dtype support, and the legal F32 carrier pays runtime selector memory. Largest charged tensors are `source_f32` 360 B, `selector_f32` 360 B, and three selector BOOL planes at 90 B each; this is exactly the #109 runtime-decoded selector caveat and loses to the live 468 Resize tail.

Remaining hypothesis: only a way to feed the final free-output Einsum with a supported sub-F32 carrier, or to make the N-dependent selectors fixed params, could revive this tail. Under ORT 1.24.4 with runtime N, this route is cost-dead.
