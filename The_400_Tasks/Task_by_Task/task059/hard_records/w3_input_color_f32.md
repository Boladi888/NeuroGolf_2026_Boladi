# task059 w3_input_color_f32 dead route

- live bar: 470, SHA256 `105da16b59be13e9d797c5d0121e3cc1fb749eb4799b254155921b686bf0900d` (`python tools/ng_current_bar.py --task 59 --json`, manifest source, 2026-07-09T01:09:17Z).
- attempted route: delete `ReduceMax(input)->present_f` and `Cast(present_f)->present_f16`; use the free graph input directly inside the final separable `Einsum` as the color-presence operand.
- exact failed gate: static score `(macs=0, memory=277, params=247)`, cost 524; visible still passed `266/266 (100%)`.
- why dead: deleting the present-vector path saves 60B, but the final `Einsum` must run in F32 when it consumes the F32 input, promoting `winner_f16`, `winner4_f16`, and `state4` to F32 for +114B; net `+54` memory versus live.
- one remaining hypothesis: only revisit if mixed-dtype final `Einsum` becomes accepted and profiled as static-cheap; current ORT/scorer requires the F32 promotion, so this #119-style input-as-color fold is dominated.
