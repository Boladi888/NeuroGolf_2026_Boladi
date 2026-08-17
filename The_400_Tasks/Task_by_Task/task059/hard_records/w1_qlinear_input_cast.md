# task059 w1_qlinear_input_cast dead route

- live bar: 470, SHA256 `105da16b59be13e9d797c5d0121e3cc1fb749eb4799b254155921b686bf0900d` (`python tools/ng_current_bar.py --task 59 --json`, manifest source, owner Sho, updated 2026-07-09T01:09:17Z).
- attempted route: replace the F32 background-count `Conv(input, bg_count_w)` with `Cast(input -> UINT8)` plus `QLinearConv` so the count carrier and `ReduceMin` run as U8.
- exact failed gate: visible passed `266/266 (100%)`, but static/profile cost regressed to `9446` (`memory=9193`, `params=253`) via `python tools/agent_helpers/profile_real_cost.py --task 59 --cand ... --arcgen 266`.
- why dead: the full-frame `input_u8 [1,10,30,30]` cast adds 9000 charged bytes; the QLinearConv path only narrows `region_bg_sums` by 27B and `min_bg` by 3B. `ng_graph_cost_diff` names net `+8970` memory from this structure before scalar QLC params.
- one remaining hypothesis: revisit QLinearConv only if a U8/INT8 feed already exists before the compartment counter, or if the raw F32 input read can be fused into a different producer that avoids materializing `input_u8`.

