# DEAD ROUTE — task379 contact BitShift dtype shave

**Live/team-best bar used:** 4771 from `python tools/ng_current_bar.py --task 379 --json`, authoritative manifest owner Boladi, sha `8b83c00f09d92dd325ace2ea186cf37b2b0fc1c0b1d22536aeca4843ebc30143`, updated `2026-07-13T02:24:44Z`.

**Attempted route:** Replace the v6 contact halo `Div(contact_base_i32, 2)` and `Mul(contact_base_i32, 2)` with `BitShift(..., 1)` to drop `i32_2`, then a broader uint32 mask-chain variant to make `BitShift` legal.

**Exact failed gates/checks:**
- `python The_400_Tasks/Task_by_Task/task379/prototypes/build_task379_v7_contact_bitshift_probe.py` failed ONNX checker/shape inference: `BitShift` input type `tensor(int32)` is unsupported.
- `python tools/ng_gate.py visible --task 379 --cand The_400_Tasks/Task_by_Task/task379/prototypes/task379_v7_u32_mask_chain.onnx` failed before scoring: ORT `NOT_IMPLEMENTED`, no `Where(16)` implementation for the resulting uint32 `Where` node.

**Why dead:** The cheap shift op requires unsigned integer inputs, but the current bitmask chain needs `Where` nodes and ORT does not support `Where(uint32,uint32,uint32)` here. Casting around the shift or replacing the uint32 `Where`s adds at least one `[1,20]` 4-byte tensor and loses the 1-param/4-byte theoretical shave.

**One remaining hypothesis:** Only a separate scalar/control-flow fold, not dtype-changing the row masks, is likely below the staged v7 cost 4768.
