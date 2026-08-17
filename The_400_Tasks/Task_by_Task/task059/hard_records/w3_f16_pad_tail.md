# task059 w3_f16_pad_tail dead route

- live bar: 470, SHA256 `105da16b59be13e9d797c5d0121e3cc1fb749eb4799b254155921b686bf0900d` (`python tools/ng_current_bar.py --task 59 --json`, manifest source, 2026-07-09T01:09:17Z).
- attempted route: replace Sho v28's 1-param `ConvTranspose(winner_f16, one_f16)` 3x3-to-4x4 tail with explicit `Pad(winner_f16, winner_pad_to_4)`.
- exact failed gate: static score `(macs=0, memory=223, params=254)`, cost 477; visible still passed `266/266 (100%)`.
- why dead: memory is unchanged, but `Pad` needs an INT64 `[8]` pads initializer; `ng_graph_cost_diff` attributes `+8 params winner_pad_to_4` and `-1 param one_f16`, net `+7` cost versus live.
- one remaining hypothesis: only revisit if an ORT-accepted zero-param pad-to-4 representation appears; ordinary `Pad` is dominated by the existing negative-pad `ConvTranspose`.
