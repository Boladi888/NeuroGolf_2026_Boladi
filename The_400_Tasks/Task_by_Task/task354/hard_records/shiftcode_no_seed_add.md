# task354 shiftcode_no_seed_add dead record

- live bar: 1,721 profile/live cost, SHA256 `4237c45753091b850714952f70b23bccd246ec3c534821972b41e09ee676f294`, from `python tools/ng_current_bar.py --task 354 --json` (`authoritative_source=manifest`) and `profile_real_cost.py --arcgen 266`.
- attempted route: start from current `task354_shiftcode_qtail_v16`, feed `top_code = top_id + 1` into the four-round masked flood, remove the `seed_code = seed4 + 1` UINT8 `[1,1,8,10]` plane, and retune the QLinear tail so channel 0 fires for code `0` or `1`.
- exact failed gate: `python tools/ng_gate.py visible --task 354 --cand The_400_Tasks/Task_by_Task/task354/prototypes/task354_v17_shiftcode_no_seed_add.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- why dead: body background code `0` is indistinguishable from QLinearConv's padded outside-canvas code `0`; making channel 0 fire for body background therefore also emits channel 0 in the required all-zero 30x30 padding area. The current `seed_code` plane is load-bearing because it keeps in-canvas background at code `1` while QLinear padding remains code `0`.
- one remaining hypothesis: a lower route must delete flood work without making in-canvas background share the QLinear padding sentinel, or replace the QLinear tail with a decoder that has a separate active-mask/padding sentinel below the 80 B plane cost.
