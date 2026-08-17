# task167 where_pad_terminal_writer

- live bar: 75 profile/live cost, parent `Team_Best_Onnx/task167.onnx`, sha256 `8eed41c53ef36cb09a52800dcefd1993c6d73394314b4262bd06b20740b1faa2` (`ng_current_bar.py --task 167 --json`, manifest source).
- attempted route: replace the parent `ConvInteger` output writer with `Greater(line,0) -> Where(fg_vec,bg_vec) -> Pad`, keeping the same distinct-count and packed-mask prefix.
- exact failed gate: `python tools/agent_helpers/profile_real_cost.py --task 167 --cand The_400_Tasks\Task_by_Task\task167\prototypes\task167_where_pad_probe.onnx --arcgen 266` -> `static=193 profile=193 (memory 154 + params 39; 268 real inputs)`. Visible did pass: `[visible] 268/268 (100.00%) PASS`.
- why dead: the route materializes `small_onehot` `[1,10,3,3]` UINT8 = 90 B plus `fg_mask` `[1,1,3,3]` BOOL = 9 B and Pad parameter inputs; it is 118 B above the current 75-byte live bar. The parent's `ConvInteger(pads=...)` keeps the 10-channel writer as the free graph output.
- one remaining hypothesis: only scalar-level rewrites inside the packed-mask path can plausibly pay; the `BitShift(cellmask,count)` zero-point remap is the successful version staged as `task167_shift_table74_cost74`.
