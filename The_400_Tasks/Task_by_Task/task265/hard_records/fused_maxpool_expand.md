# fused_maxpool_expand

- live bar: 3637, sha `0c5b087e1708ee1e348809c46c3bc1e5dcb94234fc4c11444875f9696cc4dcf3`. Current staged floor: v9 = 3349.
- attempted route (2nd pass, genuinely different assembly): fuse the dilation and the 18×18→30×30 expansion into ONE `MaxPool(17×17 → 30×30, kernel 2×2, pads=[1,1,13,13])`, deleting v9's 18×18 `red_u8`+`red_bool` (324+324) and the separate `Pad` — a different dominant-intermediate structure (30×30 U8 instead of 18×18).
- exact failed gate: `ng_gate visible --task 265 --cand build_task265_v10_maxpool_to30.onnx` → **load FAIL**: `PoolAttributes ... pads[dim] < kernel_shape[dim] ... was false. Pad should be smaller than kernel.` (ORT 1.24.x). `score_v2` returns `None` (won't load).
- why dead: ONNX/ORT MaxPool **requires pad < kernel size per dim**. A 2×2 kernel allows pad ≤ 1, so a single MaxPool can expand 17×17 by at most +2 → 18×18, never to 30×30. The 18×18→30×30 expansion therefore MUST be a separate `Pad` (charged 900 B BOOL). Even if it had loaded, pricing is worse: working at 30×30 costs two 900-B planes (`red30_u8` + `red30_bool`) vs v9's 324+324+900 = 1548 → +~185 B. Builder kept at `prototypes/build_task265_v10_maxpool_to30.py` as the dead-route record.
- one remaining hypothesis: none for this lever — the pad<kernel constraint is a hard ORT invariant. The 18×18 dilation + Pad-to-30 structure (v9) is forced.
